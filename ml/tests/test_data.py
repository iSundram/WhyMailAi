"""
ml/tests/test_data.py
---------------------
Unit tests for the data pipeline (download, preprocess, split, validate).
These tests do NOT hit the internet — they exercise logic with synthetic data.
"""

import pytest

from ml.data.preprocess import clean_text, preprocess_records, preprocess_summarization_records
from ml.data.split import stratified_split, save_splits, load_split
from ml.data.validate import (
    ValidationReport,
    validate_classification_dataset,
    validate_summarization_dataset,
)


# ---------------------------------------------------------------------------
# clean_text
# ---------------------------------------------------------------------------

class TestCleanText:
    def test_basic_lowercase(self):
        assert clean_text("Hello World") == "hello world"

    def test_url_replacement(self):
        out = clean_text("Visit http://example.com for info")
        assert "<URL>" in out
        assert "http://example.com" not in out

    def test_email_replacement(self):
        out = clean_text("Contact user@domain.com")
        assert "<EMAIL>" in out
        assert "user@domain.com" not in out

    def test_html_stripping(self):
        out = clean_text("<b>Bold</b> text with <a href='#'>link</a>")
        assert "<b>" not in out
        assert "Bold" in out
        assert "link" in out

    def test_truncation(self):
        long_text = "a" * 5000
        out = clean_text(long_text, max_chars=100)
        assert len(out) == 100

    def test_control_char_removal(self):
        out = clean_text("Hello\x00World\x01Test")
        assert "\x00" not in out
        assert "\x01" not in out
        assert "Hello" in out

    def test_preserves_newlines_and_tabs(self):
        out = clean_text("line1\nline2\ttabbed", lowercase=False)
        assert "\n" in out
        assert "\t" in out

    def test_html_entities(self):
        out = clean_text("AT&amp;T &lt;example&gt;")
        assert "&amp;" not in out
        assert "at&t" in out.lower()

    def test_empty_string(self):
        assert clean_text("") == ""

    def test_whitespace_normalisation(self):
        out = clean_text("hello    world")
        assert out == "hello world"


# ---------------------------------------------------------------------------
# preprocess_records
# ---------------------------------------------------------------------------

class TestPreprocessRecords:
    def _make_records(self, texts, labels=None):
        labels = labels or [0] * len(texts)
        return [{"text": t, "label": l, "source": "test"} for t, l in zip(texts, labels)]

    def test_basic(self):
        records = self._make_records(["Hello world", "Buy now free offer"])
        out = preprocess_records(records)
        assert len(out) == 2

    def test_deduplication(self):
        records = self._make_records(["hello world"] * 5)
        out = preprocess_records(records, deduplicate=True)
        assert len(out) == 1

    def test_min_chars_filter(self):
        records = self._make_records(["hi", "this is a proper email message"])
        out = preprocess_records(records, min_chars=10)
        assert len(out) == 1

    def test_no_dedup(self):
        records = self._make_records(["hello world"] * 3)
        out = preprocess_records(records, deduplicate=False)
        assert len(out) == 3


# ---------------------------------------------------------------------------
# stratified_split
# ---------------------------------------------------------------------------

class TestStratifiedSplit:
    def _make_records(self, n_pos, n_neg):
        records = (
            [{"text": f"spam{i}", "label": 1} for i in range(n_pos)]
            + [{"text": f"ham{i}", "label": 0} for i in range(n_neg)]
        )
        return records

    def test_sizes_sum_to_total(self):
        records = self._make_records(100, 200)
        train, val, test = stratified_split(records)
        assert len(train) + len(val) + len(test) == 300

    def test_approximate_ratios(self):
        records = self._make_records(100, 100)
        train, val, test = stratified_split(records, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
        assert abs(len(train) - 160) <= 4   # some rounding tolerance
        assert abs(len(val) - 20) <= 4
        assert abs(len(test) - 20) <= 4

    def test_reproducible(self):
        records = self._make_records(50, 50)
        t1, v1, te1 = stratified_split(records, seed=42)
        t2, v2, te2 = stratified_split(records, seed=42)
        assert [r["text"] for r in t1] == [r["text"] for r in t2]

    def test_invalid_ratio_raises(self):
        records = self._make_records(10, 10)
        with pytest.raises(AssertionError):
            stratified_split(records, train_ratio=0.7, val_ratio=0.2, test_ratio=0.2)


# ---------------------------------------------------------------------------
# validate_classification_dataset
# ---------------------------------------------------------------------------

class TestValidateClassificationDataset:
    def _records(self, n=20):
        return [
            {"text": f"This is email number {i} with enough content to be valid", "label": i % 2}
            for i in range(n)
        ]

    def test_clean_dataset_passes(self):
        report = validate_classification_dataset(self._records(40))
        assert report.is_ok
        assert report.valid == 40

    def test_empty_dataset_fails(self):
        report = validate_classification_dataset([])
        assert not report.is_ok

    def test_missing_text_field(self):
        records = [{"label": 0, "source": "test"}]
        report = validate_classification_dataset(records)
        assert not report.is_ok

    def test_missing_label_field(self):
        records = [{"text": "Some text here that is long enough to be valid"}]
        report = validate_classification_dataset(records)
        assert not report.is_ok

    def test_too_short_text_is_warned(self):
        records = [{"text": "hi", "label": 0}]
        report = validate_classification_dataset(records, min_chars=5)
        assert len(report.warnings) > 0

    def test_unexpected_label_is_error(self):
        records = [{"text": "valid text here that is long enough", "label": 99}]
        report = validate_classification_dataset(records, expected_labels={0, 1})
        assert not report.is_ok

    def test_duplicate_rate_computed(self):
        records = [{"text": "duplicate email content", "label": 0}] * 10
        report = validate_classification_dataset(records)
        assert report.duplicate_rate > 0


# ---------------------------------------------------------------------------
# validate_summarization_dataset
# ---------------------------------------------------------------------------

class TestValidateSummarizationDataset:
    def _records(self, n=10):
        return [
            {
                "text": "This is a long email thread about the project deadline and next steps.",
                "summary": "Project deadline discussion.",
            }
            for _ in range(n)
        ]

    def test_valid_dataset(self):
        report = validate_summarization_dataset(self._records(10))
        assert report.is_ok

    def test_missing_summary(self):
        records = [{"text": "Valid long dialogue text here that meets the length requirement"}]
        report = validate_summarization_dataset(records)
        assert not report.is_ok
