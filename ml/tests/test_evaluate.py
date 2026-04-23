"""
ml/tests/test_evaluate.py
--------------------------
Unit tests for evaluation metrics and the evaluation harness logic.
"""

import numpy as np
import pytest

from ml.evaluate.metrics import (
    ClassificationMetrics,
    compute_classification_metrics,
    compute_latency_metrics,
    compute_rouge_metrics,
    LatencyMetrics,
    SummarizationMetrics,
)


# ---------------------------------------------------------------------------
# compute_classification_metrics
# ---------------------------------------------------------------------------

class TestClassificationMetrics:
    def test_perfect_classifier(self):
        y_true = [0, 0, 1, 1, 1]
        y_pred = [0, 0, 1, 1, 1]
        m = compute_classification_metrics(y_true, y_pred)
        assert m.precision == pytest.approx(1.0)
        assert m.recall == pytest.approx(1.0)
        assert m.f1 == pytest.approx(1.0)
        assert m.false_positive_rate == pytest.approx(0.0)
        assert m.false_negative_rate == pytest.approx(0.0)

    def test_all_false_positives(self):
        y_true = [0, 0, 0]
        y_pred = [1, 1, 1]
        m = compute_classification_metrics(y_true, y_pred)
        assert m.false_positive_rate == pytest.approx(1.0)

    def test_all_false_negatives(self):
        y_true = [1, 1, 1]
        y_pred = [0, 0, 0]
        m = compute_classification_metrics(y_true, y_pred)
        assert m.false_negative_rate == pytest.approx(1.0)

    def test_with_scores(self):
        y_true = [0, 0, 1, 1]
        y_pred = [0, 0, 1, 1]
        y_score = [0.1, 0.2, 0.8, 0.9]
        m = compute_classification_metrics(y_true, y_pred, y_score)
        assert m.roc_auc == pytest.approx(1.0)
        assert m.average_precision > 0

    def test_support_count(self):
        y_true = [0, 1, 1, 0, 1]
        y_pred = [0, 1, 0, 0, 1]
        m = compute_classification_metrics(y_true, y_pred)
        assert m.support == 5

    def test_to_dict_keys(self):
        m = compute_classification_metrics([0, 1], [0, 1])
        d = m.to_dict()
        assert "precision" in d
        assert "recall" in d
        assert "f1" in d
        assert "false_positive_rate" in d
        assert "false_negative_rate" in d

    def test_passed_gate_passing(self):
        m = ClassificationMetrics(
            precision=0.95, recall=0.92, f1=0.93,
            false_positive_rate=0.05, false_negative_rate=0.08,
            roc_auc=0.98, average_precision=0.97, support=100,
        )
        assert m.passed_gate(min_precision=0.90, min_recall=0.85, min_f1=0.88)

    def test_passed_gate_failing_precision(self):
        m = ClassificationMetrics(
            precision=0.80, recall=0.92, f1=0.93,
            false_positive_rate=0.05, false_negative_rate=0.08,
            roc_auc=0.98, average_precision=0.97, support=100,
        )
        assert not m.passed_gate(min_precision=0.90, min_recall=0.85, min_f1=0.88)

    def test_passed_gate_failing_recall(self):
        m = ClassificationMetrics(
            precision=0.95, recall=0.70, f1=0.80,
            false_positive_rate=0.05, false_negative_rate=0.30,
            roc_auc=0.93, average_precision=0.91, support=100,
        )
        assert not m.passed_gate(min_precision=0.90, min_recall=0.85, min_f1=0.88)


# ---------------------------------------------------------------------------
# compute_latency_metrics
# ---------------------------------------------------------------------------

class TestLatencyMetrics:
    def test_basic(self):
        latencies = [10.0, 20.0, 30.0, 40.0, 50.0]
        m = compute_latency_metrics(latencies)
        assert m.mean_ms == pytest.approx(30.0)
        assert m.p50_ms == pytest.approx(30.0)
        assert m.p95_ms > 40.0
        assert m.throughput_rps > 0

    def test_empty(self):
        m = compute_latency_metrics([])
        assert m.mean_ms == 0.0
        assert m.throughput_rps == 0.0

    def test_single_value(self):
        m = compute_latency_metrics([100.0])
        assert m.mean_ms == pytest.approx(100.0)
        assert m.p50_ms == pytest.approx(100.0)
        assert m.p99_ms == pytest.approx(100.0)

    def test_to_dict(self):
        m = compute_latency_metrics([10.0, 20.0])
        d = m.to_dict()
        assert "mean_ms" in d
        assert "p95_ms" in d
        assert "throughput_rps" in d


# ---------------------------------------------------------------------------
# compute_rouge_metrics
# ---------------------------------------------------------------------------

class TestRougeMetrics:
    def test_perfect_match(self):
        preds = ["the cat sat on the mat"]
        refs = ["the cat sat on the mat"]
        m = compute_rouge_metrics(preds, refs)
        # When evaluate library is available
        if m.sample_count > 0:
            assert m.rouge1 > 90.0

    def test_returns_zeros_on_failure(self):
        # Provide empty lists — should not crash
        m = compute_rouge_metrics([], [])
        assert m.rouge1 == 0.0

    def test_to_dict(self):
        m = SummarizationMetrics(rouge1=42.0, rouge2=20.0, rougeL=38.0, sample_count=10)
        d = m.to_dict()
        assert d["rouge1"] == 42.0
        assert d["sample_count"] == 10
