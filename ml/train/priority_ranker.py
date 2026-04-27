"""
ml/train/priority_ranker.py
---------------------------
Train a lightweight logistic-regression model that scores an email's
inbox priority based on hand-crafted signal features.

Features
--------
* has_urgency_words   — "urgent", "deadline", "today", "asap", …
* has_meeting_ref     — "meeting", "call", "schedule", "calendar", …
* has_question_mark   — ends in a question (implies reply needed)
* has_action_request  — "please", "could you", "can you", "follow up", …
* text_length_bucket  — short(0) / medium(1) / long(2)
* has_attachment_ref  — "attached", "attachment", "see file", …

Training data
--------------
The Enron spam/ham dataset labels are repurposed: ham email whose subject
contains urgency signals are marked high-priority; the rest are medium/low.
A richer label set can be injected via the ``--train-jsonl`` argument.

Output
------
Saves a scikit-learn Pipeline (TF-IDF + LogReg) to *output_dir*/model.pkl.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.data.split import load_split
from ml.train.config import PriorityRankerConfig, PRIORITY_CONFIG


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

URGENCY_WORDS = {"urgent", "asap", "immediately", "deadline", "today", "emergency"}
MEETING_WORDS = {"meeting", "call", "schedule", "calendar", "invite", "standup", "sync"}
ACTION_WORDS = {"please", "could you", "can you", "follow up", "kindly", "need you"}
ATTACHMENT_WORDS = {"attached", "attachment", "see file", "enclosed", "document", "spreadsheet"}


def extract_features(text: str) -> list[float]:
    lower = text.lower()
    words = set(lower.split())

    has_urgency = float(bool(words & URGENCY_WORDS))
    has_meeting = float(bool(words & MEETING_WORDS))
    has_question = float("?" in text)
    has_action = float(any(p in lower for p in ACTION_WORDS))
    has_attachment = float(bool(words & ATTACHMENT_WORDS))

    n = len(text)
    if n < 100:
        length_bucket = 0.0
    elif n < 500:
        length_bucket = 1.0
    else:
        length_bucket = 2.0

    return [
        has_urgency,
        has_meeting,
        has_question,
        has_action,
        has_attachment,
        length_bucket,
    ]


def _label_priority(record: dict) -> int:
    """
    Heuristically assign a priority label from an Enron-style record.
    0 = low, 1 = medium, 2 = high
    """
    text = record.get("text", "")
    feats = extract_features(text)
    urgency, meeting, question, action, attachment, _ = feats
    score = urgency * 2 + meeting + question + action * 0.5 + attachment * 0.5
    if score >= 2.0:
        return 2  # high
    if score >= 0.5:
        return 1  # medium
    return 0       # low


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    cfg: PriorityRankerConfig = PRIORITY_CONFIG,
    *,
    records: list[dict] | None = None,
    data_dir: str | Path | None = None,
    experiment_name: str = "whymail-priority-ranker",
) -> Path:
    """
    Train the priority ranker.

    Provide either pre-loaded *records* or *data_dir* containing
    ``spam_train.jsonl`` (any text corpus works — labels are auto-generated
    by heuristic signals so we do not need separate priority data).
    """
    import mlflow

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        mlflow.log_params(cfg.to_dict())

        # ------------------------------------------------------------------ #
        # 1. Load / assemble data
        # ------------------------------------------------------------------ #
        if records is None:
            data_dir = Path(data_dir or cfg.output_dir).parent / "spam"
            train_records = load_split(data_dir / "spam_train.jsonl")
            val_records = load_split(data_dir / "spam_val.jsonl")
        else:
            split = int(len(records) * 0.9)
            train_records, val_records = records[:split], records[split:]

        # ------------------------------------------------------------------ #
        # 2. Feature extraction + label assignment
        # ------------------------------------------------------------------ #
        X_train = [extract_features(r["text"]) for r in train_records]
        y_train = [_label_priority(r) for r in train_records]
        X_val = [extract_features(r["text"]) for r in val_records]
        y_val = [_label_priority(r) for r in val_records]

        logger.info(
            f"Training priority ranker on {len(X_train):,} examples "
            f"(val: {len(X_val):,})"
        )

        # ------------------------------------------------------------------ #
        # 3. Train
        # ------------------------------------------------------------------ #
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=cfg.max_iter,
                C=cfg.C,
                class_weight=cfg.class_weight,
                random_state=cfg.seed,
            )),
        ])
        model.fit(X_train, y_train)

        # ------------------------------------------------------------------ #
        # 4. Evaluate
        # ------------------------------------------------------------------ #
        y_pred = model.predict(X_val)
        report = classification_report(y_val, y_pred, output_dict=True)
        logger.info(f"Validation report:\n{classification_report(y_val, y_pred)}")
        mlflow.log_metrics(
            {
                "val_accuracy": report["accuracy"],
                "val_macro_f1": report["macro avg"]["f1-score"],
            }
        )

        # ------------------------------------------------------------------ #
        # 5. Save
        # ------------------------------------------------------------------ #
        model_path = output_dir / "model.pkl"
        with open(model_path, "wb") as fh:
            pickle.dump(model, fh)

        cfg.save(output_dir / "train_config.json")
        (output_dir / "val_report.json").write_text(json.dumps(report, indent=2))
        logger.info(f"Priority ranker saved → {model_path}")
        return model_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train WhyMail priority ranker")
    parser.add_argument("--output-dir", default=PRIORITY_CONFIG.output_dir)
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args()

    cfg = PriorityRankerConfig(output_dir=args.output_dir)
    train(cfg, data_dir=args.data_dir)
