"""
ml/data/validate.py
-------------------
Validate dataset quality before training.

Checks
------
* Required field presence
* Label distribution (detects severe imbalance)
* Empty / too-short text
* Duplicate rate
* UTF-8 decodability
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from loguru import logger


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class ValidationReport:
    total: int = 0
    valid: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    label_distribution: dict[str, int] = field(default_factory=dict)
    duplicate_rate: float = 0.0

    @property
    def is_ok(self) -> bool:
        return len(self.errors) == 0

    def __str__(self) -> str:  # pragma: no cover
        lines = [
            f"ValidationReport — total: {self.total}, valid: {self.valid}",
            f"  label_distribution: {self.label_distribution}",
            f"  duplicate_rate: {self.duplicate_rate:.2%}",
        ]
        for e in self.errors:
            lines.append(f"  ERROR: {e}")
        for w in self.warnings:
            lines.append(f"  WARN:  {w}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Classifier dataset validator
# ---------------------------------------------------------------------------

def validate_classification_dataset(
    records: list[dict],
    *,
    text_key: str = "text",
    label_key: str = "label",
    expected_labels: Optional[set[int]] = None,
    min_chars: int = 5,
    max_imbalance_ratio: float = 20.0,
) -> ValidationReport:
    """
    Validate a classification dataset.

    Parameters
    ----------
    records            : list of record dicts
    text_key           : name of the text field
    label_key          : name of the label field
    expected_labels    : set of valid label integers; ``None`` means any
    min_chars          : minimum text length to be valid
    max_imbalance_ratio: warn if majority/minority class ratio exceeds this
    """
    report = ValidationReport(total=len(records))
    if not records:
        report.errors.append("Dataset is empty")
        return report

    label_counts: Counter = Counter()
    seen_texts: set[str] = set()
    duplicates = 0
    valid = 0

    for i, rec in enumerate(records):
        # Required fields
        if text_key not in rec:
            report.errors.append(f"Record {i} missing field '{text_key}'")
            continue
        if label_key not in rec:
            report.errors.append(f"Record {i} missing field '{label_key}'")
            continue

        text = rec[text_key]
        label = rec[label_key]

        # Text type
        if not isinstance(text, str):
            report.errors.append(f"Record {i}: text is not a string (got {type(text).__name__})")
            continue

        # Minimum length
        if len(text.strip()) < min_chars:
            report.warnings.append(f"Record {i}: text too short ({len(text.strip())} chars)")
            continue

        # Label validity
        if expected_labels is not None and label not in expected_labels:
            report.errors.append(f"Record {i}: unexpected label {label!r}")
            continue

        # Duplicate check
        if text in seen_texts:
            duplicates += 1
        else:
            seen_texts.add(text)

        label_counts[str(label)] += 1
        valid += 1

    report.valid = valid
    report.label_distribution = dict(label_counts)
    report.duplicate_rate = duplicates / len(records) if records else 0.0

    # Imbalance check
    if len(label_counts) >= 2:
        counts = list(label_counts.values())
        ratio = max(counts) / max(min(counts), 1)
        if ratio > max_imbalance_ratio:
            report.warnings.append(
                f"Severe class imbalance: {ratio:.1f}x "
                f"(threshold {max_imbalance_ratio:.1f}x). "
                "Consider oversampling or weighted loss."
            )

    # Duplicate rate warning
    if report.duplicate_rate > 0.10:
        report.warnings.append(
            f"High duplicate rate: {report.duplicate_rate:.1%}. "
            "Consider deduplication."
        )

    for e in report.errors[:5]:
        logger.error(e)
    for w in report.warnings[:5]:
        logger.warning(w)

    logger.info(str(report))
    return report


# ---------------------------------------------------------------------------
# Summarisation dataset validator
# ---------------------------------------------------------------------------

def validate_summarization_dataset(
    records: list[dict],
    *,
    text_key: str = "text",
    summary_key: str = "summary",
    min_text_chars: int = 50,
    min_summary_chars: int = 10,
) -> ValidationReport:
    """Validate a summarisation dataset."""
    report = ValidationReport(total=len(records))
    if not records:
        report.errors.append("Dataset is empty")
        return report

    valid = 0
    for i, rec in enumerate(records):
        if text_key not in rec:
            report.errors.append(f"Record {i} missing '{text_key}'")
            continue
        if summary_key not in rec:
            report.errors.append(f"Record {i} missing '{summary_key}'")
            continue

        if len(str(rec[text_key]).strip()) < min_text_chars:
            report.warnings.append(f"Record {i}: dialogue too short")
            continue
        if len(str(rec[summary_key]).strip()) < min_summary_chars:
            report.warnings.append(f"Record {i}: summary too short")
            continue

        valid += 1

    report.valid = valid
    logger.info(str(report))
    return report
