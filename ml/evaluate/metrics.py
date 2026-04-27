"""
ml/evaluate/metrics.py
-----------------------
Evaluation metrics for all WhyMail AI tasks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

@dataclass
class ClassificationMetrics:
    precision: float
    recall: float
    f1: float
    false_positive_rate: float
    false_negative_rate: float
    roc_auc: float
    average_precision: float
    support: int
    report: dict = field(default_factory=dict)

    def passed_gate(
        self,
        *,
        min_precision: float = 0.90,
        min_recall: float = 0.85,
        min_f1: float = 0.88,
    ) -> bool:
        return (
            self.precision >= min_precision
            and self.recall >= min_recall
            and self.f1 >= min_f1
        )

    def to_dict(self) -> dict:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "false_positive_rate": self.false_positive_rate,
            "false_negative_rate": self.false_negative_rate,
            "roc_auc": self.roc_auc,
            "average_precision": self.average_precision,
            "support": self.support,
        }


def compute_classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_score: Sequence[float] | None = None,
    *,
    pos_label: int = 1,
) -> ClassificationMetrics:
    """
    Compute all classification metrics for binary (spam/phishing) tasks.

    Parameters
    ----------
    y_true   : ground-truth integer labels
    y_pred   : predicted integer labels
    y_score  : predicted positive-class probabilities (used for AUC)
    pos_label: positive class (default 1)
    """
    yt = np.array(y_true)
    yp = np.array(y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        yt, yp, average="binary", pos_label=pos_label, zero_division=0
    )

    cm = confusion_matrix(yt, yp, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    fpr = fp / max(fp + tn, 1)
    fnr = fn / max(fn + tp, 1)

    roc_auc = 0.0
    ap = 0.0
    if y_score is not None:
        ys = np.array(y_score)
        try:
            roc_auc = float(roc_auc_score(yt, ys))
            ap = float(average_precision_score(yt, ys))
        except ValueError:
            pass

    report = classification_report(yt, yp, output_dict=True)

    return ClassificationMetrics(
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        false_positive_rate=fpr,
        false_negative_rate=fnr,
        roc_auc=roc_auc,
        average_precision=ap,
        support=int(len(yt)),
        report=report,
    )


# ---------------------------------------------------------------------------
# Summarisation metrics
# ---------------------------------------------------------------------------

@dataclass
class SummarizationMetrics:
    rouge1: float
    rouge2: float
    rougeL: float
    sample_count: int

    def to_dict(self) -> dict:
        return {
            "rouge1": self.rouge1,
            "rouge2": self.rouge2,
            "rougeL": self.rougeL,
            "sample_count": self.sample_count,
        }


def compute_rouge_metrics(
    predictions: list[str],
    references: list[str],
) -> SummarizationMetrics:
    """Compute ROUGE-1/2/L scores using the ``evaluate`` library."""
    try:
        import evaluate as hf_evaluate
        rouge = hf_evaluate.load("rouge")
        result = rouge.compute(
            predictions=predictions,
            references=references,
            use_stemmer=True,
        )
        return SummarizationMetrics(
            rouge1=round(result["rouge1"] * 100, 2),
            rouge2=round(result["rouge2"] * 100, 2),
            rougeL=round(result["rougeL"] * 100, 2),
            sample_count=len(predictions),
        )
    except Exception:
        return SummarizationMetrics(rouge1=0.0, rouge2=0.0, rougeL=0.0, sample_count=0)


# ---------------------------------------------------------------------------
# Latency / throughput metrics
# ---------------------------------------------------------------------------

@dataclass
class LatencyMetrics:
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    throughput_rps: float   # requests per second

    def to_dict(self) -> dict:
        return {
            "mean_ms": self.mean_ms,
            "p50_ms": self.p50_ms,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms,
            "throughput_rps": self.throughput_rps,
        }


def compute_latency_metrics(latencies_ms: list[float]) -> LatencyMetrics:
    if not latencies_ms:
        return LatencyMetrics(0, 0, 0, 0, 0)
    arr = np.array(latencies_ms)
    total_ms = float(arr.sum())
    return LatencyMetrics(
        mean_ms=float(arr.mean()),
        p50_ms=float(np.percentile(arr, 50)),
        p95_ms=float(np.percentile(arr, 95)),
        p99_ms=float(np.percentile(arr, 99)),
        throughput_rps=len(arr) / max(total_ms / 1000, 1e-9),
    )
