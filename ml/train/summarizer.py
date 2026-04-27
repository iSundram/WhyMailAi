"""
ml/train/summarizer.py
-----------------------
Fine-tune DistilBART (sshleifer/distilbart-cnn-12-6) on DialogSum for
email thread summarisation.

Usage
-----
    python -m ml.train.summarizer \\
        --data-dir data/summarization \\
        --output-dir models/summarizer

The trained model can summarise email threads of up to 512 tokens into
concise summaries of up to 128 tokens.  It is also usable zero-shot
(via the pre-trained BART checkpoint) before fine-tuning completes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow
import numpy as np
import torch
from datasets import Dataset as HFDataset
from loguru import logger
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

try:
    import evaluate as hf_evaluate
    _rouge = hf_evaluate.load("rouge")
    _HAS_ROUGE = True
except Exception:
    _HAS_ROUGE = False

from ml.data.split import load_split
from ml.train.config import SummarizationConfig, SUMMARIZATION_CONFIG


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _load_splits(data_dir: Path) -> tuple[list[dict], list[dict], list[dict]]:
    train = load_split(data_dir / "summarization_train.jsonl")
    val = load_split(data_dir / "summarization_val.jsonl")
    test = load_split(data_dir / "summarization_test.jsonl")
    logger.info(f"Loaded train={len(train):,}, val={len(val):,}, test={len(test):,}")
    return train, val, test


def _make_tokenize_fn(tokenizer, cfg: SummarizationConfig):
    def tokenize(batch):
        model_inputs = tokenizer(
            batch["text"],
            max_length=cfg.max_source_length,
            truncation=True,
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                batch["summary"],
                max_length=cfg.max_target_length,
                truncation=True,
            )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return tokenize


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train(
    cfg: SummarizationConfig = SUMMARIZATION_CONFIG,
    *,
    experiment_name: str = "whymail-summarizer",
) -> Path:
    """Fine-tune summarisation model. Returns saved model path."""
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

        tokenize_fn = _make_tokenize_fn(tokenizer, cfg)
        columns_to_remove = ["text", "summary", "topic", "source"]

        def prep(records):
            ds = HFDataset.from_list(
                [{"text": r["text"], "summary": r["summary"]} for r in records]
            )
            existing_cols = [c for c in columns_to_remove if c in ds.column_names]
            return ds.map(tokenize_fn, batched=True, remove_columns=existing_cols)

        train_ds = prep(train_records)
        val_ds = prep(val_records)
        test_ds = prep(test_records)

        collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8)

        # ------------------------------------------------------------------ #
        # 3. Model
        # ------------------------------------------------------------------ #
        logger.info(f"Loading model: {cfg.model_name}")
        model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_name)

        # ------------------------------------------------------------------ #
        # 4. Metrics
        # ------------------------------------------------------------------ #
        def compute_metrics(eval_pred):
            preds, labels = eval_pred
            if isinstance(preds, tuple):
                preds = preds[0]
            decoded_preds = tokenizer.batch_decode(
                np.where(preds != -100, preds, tokenizer.pad_token_id),
                skip_special_tokens=True,
            )
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            decoded_preds = [p.strip() for p in decoded_preds]
            decoded_labels = [l.strip() for l in decoded_labels]
            if _HAS_ROUGE:
                result = _rouge.compute(
                    predictions=decoded_preds,
                    references=decoded_labels,
                    use_stemmer=True,
                )
                return {k: round(v * 100, 2) for k, v in result.items()}
            return {}

        # ------------------------------------------------------------------ #
        # 5. Training arguments
        # ------------------------------------------------------------------ #
        training_args = Seq2SeqTrainingArguments(
            output_dir=str(output_dir / "checkpoints"),
            num_train_epochs=cfg.num_epochs,
            per_device_train_batch_size=cfg.per_device_train_batch_size,
            per_device_eval_batch_size=cfg.per_device_eval_batch_size,
            learning_rate=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            warmup_ratio=cfg.warmup_ratio,
            fp16=cfg.fp16 and torch.cuda.is_available(),
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            predict_with_generate=True,
            generation_max_length=cfg.max_target_length,
            generation_num_beams=cfg.num_beams,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="rouge1",
            report_to="none",
            seed=cfg.seed,
            logging_steps=50,
        )

        # ------------------------------------------------------------------ #
        # 6. Train
        # ------------------------------------------------------------------ #
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=tokenizer,
            data_collator=collator,
            compute_metrics=compute_metrics,
        )

        logger.info("Starting summariser training …")
        trainer.train()

        # ------------------------------------------------------------------ #
        # 7. Evaluate
        # ------------------------------------------------------------------ #
        logger.info("Evaluating on test set …")
        test_results = trainer.evaluate(test_ds)
        rouge1 = test_results.get("eval_rouge1", 0.0)
        rougeL = test_results.get("eval_rougeL", 0.0)

        logger.info(f"Test results — ROUGE-1: {rouge1:.2f}, ROUGE-L: {rougeL:.2f}")
        mlflow.log_metrics({"test_rouge1": rouge1, "test_rougeL": rougeL})

        # ------------------------------------------------------------------ #
        # 8. Save
        # ------------------------------------------------------------------ #
        model_save_path = output_dir / "model"
        trainer.save_model(str(model_save_path))
        tokenizer.save_pretrained(str(model_save_path))
        cfg.save(output_dir / "train_config.json")
        (output_dir / "test_metrics.json").write_text(
            json.dumps({"rouge1": rouge1, "rougeL": rougeL}, indent=2)
        )

        logger.info(f"Summariser saved → {model_save_path}")
        return model_save_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> SummarizationConfig:
    parser = argparse.ArgumentParser(description="Train WhyMail summarizer")
    parser.add_argument("--data-dir", default=SUMMARIZATION_CONFIG.data_dir)
    parser.add_argument("--output-dir", default=SUMMARIZATION_CONFIG.output_dir)
    parser.add_argument("--model-name", default=SUMMARIZATION_CONFIG.model_name)
    parser.add_argument("--epochs", type=int, default=SUMMARIZATION_CONFIG.num_epochs)
    parser.add_argument("--batch-size", type=int, default=SUMMARIZATION_CONFIG.per_device_train_batch_size)
    parser.add_argument("--lr", type=float, default=SUMMARIZATION_CONFIG.learning_rate)
    parser.add_argument("--fp16", action="store_true", default=False)
    args = parser.parse_args()

    return SummarizationConfig(
        model_name=args.model_name,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        fp16=args.fp16,
    )


if __name__ == "__main__":
    cfg = _parse_args()
    train(cfg)
