"""
ml/train/spam_classifier.py
---------------------------
Fine-tune DistilBERT for binary spam / ham classification.

Usage
-----
    python -m ml.train.spam_classifier \\
        --data-dir data/spam \\
        --output-dir models/spam_classifier

The script:
1. Loads train/val/test JSONL splits from *data_dir*.
2. Tokenises with DistilBERT tokeniser.
3. Fine-tunes for *num_epochs* with AdamW + linear LR schedule.
4. Evaluates on the test set.
5. Gates promotion via configurable precision/recall/F1 thresholds.
6. Saves the model + tokeniser to *output_dir*.
7. Logs metrics to MLflow.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import mlflow
import numpy as np
import torch
from datasets import Dataset as HFDataset
from loguru import logger
from sklearn.metrics import classification_report, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from ml.data.split import load_split
from ml.train.config import ClassifierConfig, SPAM_CONFIG


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _load_splits(
    data_dir: Path,
    prefix: str = "spam_",
) -> tuple[list[dict], list[dict], list[dict]]:
    train = load_split(data_dir / f"{prefix}train.jsonl")
    val = load_split(data_dir / f"{prefix}val.jsonl")
    test = load_split(data_dir / f"{prefix}test.jsonl")
    logger.info(f"Loaded train={len(train):,}, val={len(val):,}, test={len(test):,}")
    return train, val, test


def _to_hf_dataset(records: list[dict]) -> HFDataset:
    return HFDataset.from_list([{"text": r["text"], "labels": int(r["label"])} for r in records])


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", pos_label=1, zero_division=0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(
    cfg: ClassifierConfig = SPAM_CONFIG,
    *,
    experiment_name: str = "whymail-spam-classifier",
) -> Path:
    """
    Run the full training pipeline.

    Returns the path to the saved model directory.
    """
    set_seed(cfg.seed)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(cfg.data_dir)

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        mlflow.log_params(cfg.to_dict())

        # ------------------------------------------------------------------ #
        # 1. Load data
        # ------------------------------------------------------------------ #
        train_records, val_records, test_records = _load_splits(data_dir)

        # ------------------------------------------------------------------ #
        # 2. Tokenise
        # ------------------------------------------------------------------ #
        logger.info(f"Loading tokeniser: {cfg.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

        def tokenize(batch):
            return tokenizer(
                batch["text"],
                truncation=True,
                max_length=cfg.max_length,
                padding=False,  # dynamic padding via DataCollator
            )

        train_ds = _to_hf_dataset(train_records).map(tokenize, batched=True, remove_columns=["text"])
        val_ds = _to_hf_dataset(val_records).map(tokenize, batched=True, remove_columns=["text"])
        test_ds = _to_hf_dataset(test_records).map(tokenize, batched=True, remove_columns=["text"])

        collator = DataCollatorWithPadding(tokenizer)

        # ------------------------------------------------------------------ #
        # 3. Model
        # ------------------------------------------------------------------ #
        logger.info(f"Loading model: {cfg.model_name}")
        model = AutoModelForSequenceClassification.from_pretrained(
            cfg.model_name,
            num_labels=cfg.num_labels,
            id2label={0: "ham", 1: "spam"},
            label2id={"ham": 0, "spam": 1},
        )

        # ------------------------------------------------------------------ #
        # 4. Training arguments
        # ------------------------------------------------------------------ #
        training_args = TrainingArguments(
            output_dir=str(output_dir / "checkpoints"),
            num_train_epochs=cfg.num_epochs,
            per_device_train_batch_size=cfg.per_device_train_batch_size,
            per_device_eval_batch_size=cfg.per_device_eval_batch_size,
            learning_rate=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            warmup_ratio=cfg.warmup_ratio,
            max_grad_norm=cfg.max_grad_norm,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            fp16=cfg.fp16 and torch.cuda.is_available(),
            report_to="none",   # we handle logging ourselves via mlflow
            seed=cfg.seed,
            logging_steps=50,
        )

        # ------------------------------------------------------------------ #
        # 5. Compute class weights for imbalanced data
        # ------------------------------------------------------------------ #
        labels = [r["label"] for r in train_records]
        n_pos = sum(labels)
        n_neg = len(labels) - n_pos
        pos_weight = n_neg / max(n_pos, 1)

        class CustomTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                lbl = inputs.get("labels")
                outputs = model(**inputs)
                logits = outputs.get("logits")
                weight = torch.tensor(
                    [1.0, pos_weight], dtype=torch.float, device=logits.device
                )
                loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
                loss = loss_fn(logits, lbl)
                return (loss, outputs) if return_outputs else loss

        # ------------------------------------------------------------------ #
        # 6. Train
        # ------------------------------------------------------------------ #
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=tokenizer,
            data_collator=collator,
            compute_metrics=_compute_metrics,
        )

        logger.info("Starting training …")
        trainer.train()

        # ------------------------------------------------------------------ #
        # 7. Evaluate on test set
        # ------------------------------------------------------------------ #
        logger.info("Evaluating on test set …")
        test_results = trainer.evaluate(test_ds)
        precision = test_results.get("eval_precision", 0.0)
        recall = test_results.get("eval_recall", 0.0)
        f1 = test_results.get("eval_f1", 0.0)

        logger.info(
            f"Test results — precision: {precision:.4f}, recall: {recall:.4f}, F1: {f1:.4f}"
        )
        mlflow.log_metrics(
            {"test_precision": precision, "test_recall": recall, "test_f1": f1}
        )

        # ------------------------------------------------------------------ #
        # 8. Promotion gate
        # ------------------------------------------------------------------ #
        if precision < cfg.min_precision or recall < cfg.min_recall or f1 < cfg.min_f1:
            msg = (
                f"Model did NOT pass promotion gate. "
                f"Required precision≥{cfg.min_precision}, recall≥{cfg.min_recall}, "
                f"F1≥{cfg.min_f1}. Got {precision:.4f}/{recall:.4f}/{f1:.4f}."
            )
            logger.error(msg)
            mlflow.log_param("promoted", False)
            raise ValueError(msg)

        mlflow.log_param("promoted", True)
        logger.info("Model passed promotion gate ✓")

        # ------------------------------------------------------------------ #
        # 9. Save model
        # ------------------------------------------------------------------ #
        model_save_path = output_dir / "model"
        trainer.save_model(str(model_save_path))
        tokenizer.save_pretrained(str(model_save_path))

        # Save config for reproducibility
        cfg.save(output_dir / "train_config.json")
        (output_dir / "test_metrics.json").write_text(
            json.dumps({"precision": precision, "recall": recall, "f1": f1}, indent=2)
        )

        logger.info(f"Model saved → {model_save_path}")
        return model_save_path


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _parse_args() -> ClassifierConfig:
    parser = argparse.ArgumentParser(description="Train WhyMail spam classifier")
    parser.add_argument("--data-dir", default=SPAM_CONFIG.data_dir)
    parser.add_argument("--output-dir", default=SPAM_CONFIG.output_dir)
    parser.add_argument("--model-name", default=SPAM_CONFIG.model_name)
    parser.add_argument("--epochs", type=int, default=SPAM_CONFIG.num_epochs)
    parser.add_argument("--batch-size", type=int, default=SPAM_CONFIG.per_device_train_batch_size)
    parser.add_argument("--lr", type=float, default=SPAM_CONFIG.learning_rate)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=SPAM_CONFIG.seed)
    args = parser.parse_args()

    cfg = ClassifierConfig(
        model_name=args.model_name,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        fp16=args.fp16,
        seed=args.seed,
    )
    return cfg


if __name__ == "__main__":
    cfg = _parse_args()
    train(cfg)
