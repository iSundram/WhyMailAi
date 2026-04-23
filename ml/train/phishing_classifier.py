"""
ml/train/phishing_classifier.py
--------------------------------
Fine-tune DistilBERT for binary phishing / legitimate email classification.

Shares the same training loop as spam_classifier but uses a stricter
promotion gate (higher required precision/recall) and the phishing dataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger

from ml.train.config import PhishingClassifierConfig, PHISHING_CONFIG
from ml.train.spam_classifier import train as _base_train


def train(
    cfg: PhishingClassifierConfig = PHISHING_CONFIG,
    *,
    experiment_name: str = "whymail-phishing-classifier",
) -> Path:
    """
    Run the phishing classifier training pipeline.

    Internally re-uses the spam classifier training loop — the only
    differences are the dataset path, label names, and the stricter
    promotion gate embedded in PhishingClassifierConfig.

    Returns the path to the saved model directory.
    """
    logger.info("Starting phishing classifier training …")
    return _base_train(cfg, experiment_name=experiment_name)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> PhishingClassifierConfig:
    parser = argparse.ArgumentParser(description="Train WhyMail phishing classifier")
    parser.add_argument("--data-dir", default=PHISHING_CONFIG.data_dir)
    parser.add_argument("--output-dir", default=PHISHING_CONFIG.output_dir)
    parser.add_argument("--model-name", default=PHISHING_CONFIG.model_name)
    parser.add_argument("--epochs", type=int, default=PHISHING_CONFIG.num_epochs)
    parser.add_argument("--batch-size", type=int, default=PHISHING_CONFIG.per_device_train_batch_size)
    parser.add_argument("--lr", type=float, default=PHISHING_CONFIG.learning_rate)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=PHISHING_CONFIG.seed)
    args = parser.parse_args()

    cfg = PhishingClassifierConfig(
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
