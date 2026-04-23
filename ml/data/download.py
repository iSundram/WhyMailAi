"""
ml/data/download.py
-------------------
Download and cache email datasets from HuggingFace Hub.

Supported datasets
------------------
* spam/ham classification
  - sms_spam          (UCI SMS Spam Collection via HuggingFace)
  - SetFit/enron_spam (Enron email spam, ~33 k examples)

* phishing classification
  - ealvaradob/phishing-email-dataset

* summarisation (thread → summary)
  - knkarthick/dialogsum  (dialogue / conversation summaries)

All datasets are normalised to a common schema:
  {"text": str, "label": int, "source": str}
and saved to <data_dir>/<split>.jsonl
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Iterator

from datasets import load_dataset, DatasetDict, Dataset
from loguru import logger

# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

LABEL_HAM = 0
LABEL_SPAM = 1
LABEL_LEGITIMATE = 0
LABEL_PHISHING = 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(records):,} records → {path}")


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# sms_spam  (HuggingFace: ucirvine/sms_spam or sms_spam)
# ---------------------------------------------------------------------------

def download_sms_spam(data_dir: Path) -> DatasetDict:
    """
    Download the UCI SMS Spam Collection.
    HuggingFace name: "sms_spam"
    Schema: {sms: str, label: 0/1}  where 1 = spam
    """
    logger.info("Downloading sms_spam …")
    ds = load_dataset("sms_spam", split="train", trust_remote_code=True)
    records = [
        {
            "text": row["sms"],
            "label": int(row["label"]),
            "source": "sms_spam",
        }
        for row in ds
    ]
    out = data_dir / "raw" / "sms_spam.jsonl"
    _save_jsonl(records, out)
    return records


# ---------------------------------------------------------------------------
# SetFit/enron_spam
# ---------------------------------------------------------------------------

def download_enron_spam(data_dir: Path) -> list[dict]:
    """
    Download the Enron spam dataset via SetFit.
    HuggingFace name: "SetFit/enron_spam"
    Schema: {subject: str, message: str, label: 0/1}  where 1 = spam
    """
    logger.info("Downloading SetFit/enron_spam …")
    splits = load_dataset("SetFit/enron_spam", trust_remote_code=True)
    records: list[dict] = []
    for split_name, split_ds in splits.items():
        for row in split_ds:
            subject = row.get("subject", "") or ""
            message = row.get("message", "") or ""
            text = f"{subject}\n\n{message}".strip()
            records.append(
                {
                    "text": text,
                    "label": int(row["label"]),
                    "source": f"enron_spam/{split_name}",
                }
            )
    out = data_dir / "raw" / "enron_spam.jsonl"
    _save_jsonl(records, out)
    return records


# ---------------------------------------------------------------------------
# Phishing email dataset
# ---------------------------------------------------------------------------

def download_phishing_emails(data_dir: Path) -> list[dict]:
    """
    Download a phishing email dataset.
    HuggingFace name: "ealvaradob/phishing-email-dataset"
    Schema: {text_combined: str, label: 0/1}  where 1 = phishing
    Falls back to building a small synthetic phishing set if the dataset
    is unavailable (so the pipeline never hard-fails in offline environments).
    """
    logger.info("Downloading phishing email dataset …")
    try:
        ds = load_dataset(
            "ealvaradob/phishing-email-dataset",
            split="train",
            trust_remote_code=True,
        )
        text_col = "text_combined" if "text_combined" in ds.column_names else ds.column_names[0]
        label_col = "label" if "label" in ds.column_names else ds.column_names[-1]
        records = [
            {
                "text": str(row[text_col]),
                "label": int(row[label_col]),
                "source": "phishing-email-dataset",
            }
            for row in ds
        ]
    except Exception as exc:
        logger.warning(f"Could not download phishing dataset ({exc}); using synthetic fallback.")
        records = _synthetic_phishing_fallback()

    out = data_dir / "raw" / "phishing_emails.jsonl"
    _save_jsonl(records, out)
    return records


def _synthetic_phishing_fallback() -> list[dict]:
    """Minimal synthetic phishing examples for offline development."""
    phishing_texts = [
        "Verify your account immediately. Click here: http://bit.ly/phish",
        "Your password has expired. Login now to reset: http://login.evil.com",
        "Dear customer, your account has been compromised. Wire transfer $500 immediately.",
        "URGENT: Security alert — confirm your credentials at http://secure-bank-update.com",
        "You won a prize! Claim by entering your credit card: http://prize.scam.net",
    ]
    legit_texts = [
        "Hi Sarah, just following up on yesterday's meeting. Let me know your thoughts.",
        "Please find attached the quarterly report. Best, John",
        "Team sync tomorrow at 10am. Dial-in details in the calendar invite.",
        "Thanks for the great work on the project. Really appreciate the effort!",
        "Could you send me the updated spreadsheet when you get a chance?",
    ]
    records = (
        [{"text": t, "label": LABEL_PHISHING, "source": "synthetic"} for t in phishing_texts]
        + [{"text": t, "label": LABEL_LEGITIMATE, "source": "synthetic"} for t in legit_texts]
    )
    return records


# ---------------------------------------------------------------------------
# knkarthick/dialogsum  (summarisation)
# ---------------------------------------------------------------------------

def download_dialogsum(data_dir: Path) -> list[dict]:
    """
    Download DialogSum for email thread summarisation training.
    HuggingFace name: "knkarthick/dialogsum"
    Schema: {dialogue: str, summary: str, topic: str}
    """
    logger.info("Downloading knkarthick/dialogsum …")
    splits = load_dataset("knkarthick/dialogsum", trust_remote_code=True)
    records: list[dict] = []
    for split_name, split_ds in splits.items():
        for row in split_ds:
            records.append(
                {
                    "text": row["dialogue"],
                    "summary": row["summary"],
                    "topic": row.get("topic", ""),
                    "source": f"dialogsum/{split_name}",
                }
            )
    out = data_dir / "raw" / "dialogsum.jsonl"
    _save_jsonl(records, out)
    return records


# ---------------------------------------------------------------------------
# Main entry-point
# ---------------------------------------------------------------------------

def download_all(data_dir: str | Path = "data") -> dict[str, list[dict]]:
    """
    Download all datasets and return them as a dict of lists.

    Parameters
    ----------
    data_dir : path to the root data directory (default ``data/``)

    Returns
    -------
    dict with keys: "spam", "phishing", "summarization"
    """
    data_dir = Path(data_dir)
    (data_dir / "raw").mkdir(parents=True, exist_ok=True)

    sms = download_sms_spam(data_dir)
    enron = download_enron_spam(data_dir)
    spam_records = sms + enron

    phishing_records = download_phishing_emails(data_dir)
    summarization_records = download_dialogsum(data_dir)

    # Save combined spam corpus
    combined_spam_path = data_dir / "raw" / "spam_combined.jsonl"
    _save_jsonl(spam_records, combined_spam_path)

    logger.info(
        f"Download complete — spam: {len(spam_records):,}, "
        f"phishing: {len(phishing_records):,}, "
        f"summarization: {len(summarization_records):,}"
    )

    return {
        "spam": spam_records,
        "phishing": phishing_records,
        "summarization": summarization_records,
    }


if __name__ == "__main__":
    import sys

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    download_all(data_dir)
