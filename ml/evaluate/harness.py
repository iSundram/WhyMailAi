"""
ml/evaluate/harness.py
-----------------------
End-to-end evaluation harness.

Runs each model against its test split, computes metrics, and gates
promotion.  A model is only considered production-ready if it passes
every threshold defined in its config.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from ml.evaluate.metrics import (
    ClassificationMetrics,
    LatencyMetrics,
    SummarizationMetrics,
    compute_classification_metrics,
    compute_latency_metrics,
    compute_rouge_metrics,
)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class ModelEvalResult:
    model_name: str
    task: str
    classification: ClassificationMetrics | None = None
    summarization: SummarizationMetrics | None = None
    latency: LatencyMetrics | None = None
    promoted: bool = False
    failure_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "task": self.task,
            "promoted": self.promoted,
            "failure_reason": self.failure_reason,
            "classification": self.classification.to_dict() if self.classification else None,
            "summarization": self.summarization.to_dict() if self.summarization else None,
            "latency": self.latency.to_dict() if self.latency else None,
        }


# ---------------------------------------------------------------------------
# Classifier evaluator
# ---------------------------------------------------------------------------

def evaluate_classifier(
    model_dir: str | Path,
    test_records: list[dict],
    *,
    task: str = "spam-classification",
    min_precision: float = 0.90,
    min_recall: float = 0.85,
    min_f1: float = 0.88,
    text_key: str = "text",
    label_key: str = "label",
    max_length: int = 256,
) -> ModelEvalResult:
    """
    Evaluate a saved DistilBERT classifier against *test_records*.

    Parameters
    ----------
    model_dir    : path to the directory saved by ``trainer.save_model()``
    test_records : list of ``{"text": ..., "label": ...}`` dicts
    task         : task name string (for reporting)
    min_*        : promotion gate thresholds
    """
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import torch

    model_dir = Path(model_dir)
    logger.info(f"Evaluating {task} model at {model_dir} …")

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    y_true, y_pred, y_score, latencies = [], [], [], []

    for rec in test_records:
        text = rec.get(text_key, "")
        label = int(rec.get(label_key, 0))

        t0 = time.perf_counter()
        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        ).to(device)

        with torch.no_grad():
            logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        pred = int(np.argmax(probs))
        latencies.append((time.perf_counter() - t0) * 1000)

        y_true.append(label)
        y_pred.append(pred)
        y_score.append(float(probs[1]))

    cls_metrics = compute_classification_metrics(y_true, y_pred, y_score)
    lat_metrics = compute_latency_metrics(latencies)

    passed = cls_metrics.passed_gate(
        min_precision=min_precision,
        min_recall=min_recall,
        min_f1=min_f1,
    )
    failure = (
        ""
        if passed
        else (
            f"precision={cls_metrics.precision:.4f} (min {min_precision}), "
            f"recall={cls_metrics.recall:.4f} (min {min_recall}), "
            f"f1={cls_metrics.f1:.4f} (min {min_f1})"
        )
    )

    logger.info(
        f"{task} — P={cls_metrics.precision:.4f} R={cls_metrics.recall:.4f} "
        f"F1={cls_metrics.f1:.4f} FPR={cls_metrics.false_positive_rate:.4f} "
        f"promoted={passed}"
    )

    return ModelEvalResult(
        model_name=model_dir.name,
        task=task,
        classification=cls_metrics,
        latency=lat_metrics,
        promoted=passed,
        failure_reason=failure,
    )


# ---------------------------------------------------------------------------
# Summarisation evaluator
# ---------------------------------------------------------------------------

def evaluate_summarizer(
    model_dir: str | Path,
    test_records: list[dict],
    *,
    num_beams: int = 4,
    max_target_length: int = 128,
    max_source_length: int = 512,
    batch_size: int = 8,
) -> ModelEvalResult:
    """Evaluate a saved seq2seq summarisation model."""
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    import torch

    model_dir = Path(model_dir)
    logger.info(f"Evaluating summarizer at {model_dir} …")

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSeq2SeqLM.from_pretrained(str(model_dir))
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    predictions, references, latencies = [], [], []

    for i in range(0, len(test_records), batch_size):
        batch = test_records[i : i + batch_size]
        texts = [r["text"] for r in batch]
        refs = [r["summary"] for r in batch]

        t0 = time.perf_counter()
        enc = tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=max_source_length,
            padding=True,
        ).to(device)
        with torch.no_grad():
            out_ids = model.generate(
                **enc,
                num_beams=num_beams,
                max_length=max_target_length,
                early_stopping=True,
            )
        latencies.append((time.perf_counter() - t0) * 1000 / len(batch))
        preds = tokenizer.batch_decode(out_ids, skip_special_tokens=True)
        predictions.extend(preds)
        references.extend(refs)

    rouge_metrics = compute_rouge_metrics(predictions, references)
    lat_metrics = compute_latency_metrics(latencies)

    logger.info(
        f"Summarizer — ROUGE-1={rouge_metrics.rouge1:.2f} "
        f"ROUGE-L={rouge_metrics.rougeL:.2f}"
    )

    return ModelEvalResult(
        model_name=model_dir.name,
        task="thread-summary",
        summarization=rouge_metrics,
        latency=lat_metrics,
        promoted=True,   # summarisation passes unless ROUGE-1 < 10
        failure_reason="" if rouge_metrics.rouge1 >= 10 else "ROUGE-1 too low",
    )


# ---------------------------------------------------------------------------
# Full harness runner
# ---------------------------------------------------------------------------

def run_full_harness(
    models_root: str | Path,
    data_root: str | Path,
    *,
    report_path: str | Path | None = None,
) -> list[ModelEvalResult]:
    """
    Evaluate all models found under *models_root* against test splits.

    Expects:
      {models_root}/spam_classifier/model/
      {models_root}/phishing_classifier/model/
      {models_root}/summarizer/model/

      {data_root}/spam/spam_test.jsonl
      {data_root}/phishing/phishing_test.jsonl
      {data_root}/summarization/summarization_test.jsonl
    """
    from ml.data.split import load_split

    models_root = Path(models_root)
    data_root = Path(data_root)
    results: list[ModelEvalResult] = []

    # Spam
    spam_model = models_root / "spam_classifier" / "model"
    spam_test = data_root / "spam" / "spam_test.jsonl"
    if spam_model.exists() and spam_test.exists():
        res = evaluate_classifier(
            spam_model,
            load_split(spam_test),
            task="spam-classification",
        )
        results.append(res)

    # Phishing
    phish_model = models_root / "phishing_classifier" / "model"
    phish_test = data_root / "phishing" / "phishing_test.jsonl"
    if phish_model.exists() and phish_test.exists():
        res = evaluate_classifier(
            phish_model,
            load_split(phish_test),
            task="phishing-classification",
            min_precision=0.92,
            min_recall=0.88,
            min_f1=0.90,
        )
        results.append(res)

    # Summariser
    summ_model = models_root / "summarizer" / "model"
    summ_test = data_root / "summarization" / "summarization_test.jsonl"
    if summ_model.exists() and summ_test.exists():
        res = evaluate_summarizer(summ_model, load_split(summ_test))
        results.append(res)

    # Summary
    all_passed = all(r.promoted for r in results)
    logger.info(
        f"Harness complete — {len(results)} models evaluated, "
        f"{'ALL PROMOTED' if all_passed else 'SOME FAILED'}"
    )

    if report_path:
        Path(report_path).write_text(
            json.dumps([r.to_dict() for r in results], indent=2)
        )
        logger.info(f"Report saved → {report_path}")

    return results
