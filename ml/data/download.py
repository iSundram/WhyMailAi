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

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from datasets import load_dataset, DatasetDict, Dataset
from loguru import logger

# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

LABEL_HAM = 0
LABEL_SPAM = 1
LABEL_LEGITIMATE = 0
LABEL_PHISHING = 1
KAGGLE_SLUG_PATTERN = r"[A-Za-z0-9_]+/[A-Za-z0-9_][A-Za-z0-9_-]*"


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


def _normalise_binary_label(value) -> int | None:
    s = str(value).strip().lower()
    if s in {"1", "true", "yes", "spam", "junk", "phishing", "malicious"}:
        return 1
    if s in {"0", "false", "no", "ham", "legitimate", "safe", "not spam"}:
        return 0
    try:
        return 1 if float(s) >= 0.5 else 0
    except Exception:
        return None


def _guess_column(columns: list[str], candidates: list[str]) -> str | None:
    lowered = {c.lower(): c for c in columns}
    for c in candidates:
        if c.lower() in lowered:
            return lowered[c.lower()]
    for col in columns:
        for c in candidates:
            if c.lower() in col.lower():
                return col
    return None


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
# Extended / optional spam sources
# ---------------------------------------------------------------------------

def download_extra_hf_spam(data_dir: Path, dataset_names: list[str]) -> list[dict]:
    """
    Best-effort downloader for additional HF spam datasets.
    Tries to infer text/label columns and ignores unsupported datasets.
    """
    out_records: list[dict] = []
    for name in dataset_names:
        dataset_name = name.strip()
        if not dataset_name:
            continue
        logger.info(f"Downloading extra HuggingFace dataset: {dataset_name}")
        try:
            ds = load_dataset(dataset_name, split="train", trust_remote_code=True)
            columns = list(ds.column_names)
            text_col = _guess_column(columns, ["text", "message", "email", "content", "body"])
            label_col = _guess_column(columns, ["label", "target", "class", "spam"])
            if text_col is None or label_col is None:
                logger.warning(
                    f"Skipping {dataset_name}: unable to infer text/label columns from {columns}"
                )
                continue
            kept = 0
            for row in ds:
                label = _normalise_binary_label(row.get(label_col))
                text = str(row.get(text_col, "")).strip()
                if label is None or not text:
                    continue
                out_records.append(
                    {
                        "text": text,
                        "label": label,
                        "source": f"hf-extra/{dataset_name}",
                    }
                )
                kept += 1
            logger.info(f"Loaded {kept:,} records from {dataset_name}")
        except Exception as exc:
            logger.warning(f"Skipping {dataset_name}: {exc}")

    if out_records:
        _save_jsonl(out_records, data_dir / "raw" / "spam_extra_hf.jsonl")
    return out_records


def download_kaggle_spam(data_dir: Path, dataset_slug: str) -> list[dict]:
    """
    Best-effort Kaggle downloader via kaggle CLI.
    Expects CSV files with inferable text/label columns.
    """
    if not dataset_slug:
        return []
    if not re.fullmatch(KAGGLE_SLUG_PATTERN, dataset_slug):
        logger.warning(
            "Invalid Kaggle dataset slug format. Expected 'owner/dataset'; skipping."
        )
        return []
    kaggle_bin = shutil.which("kaggle")
    if kaggle_bin is None:
        logger.warning("kaggle CLI not found; skipping Kaggle dataset.")
        return []

    records: list[dict] = []
    timeout_s = 300
    try:
        timeout_s = max(60, int(os.environ.get("WHYMAIL_KAGGLE_DOWNLOAD_TIMEOUT_SECONDS", "300")))
    except Exception:
        logger.warning(
            "Invalid WHYMAIL_KAGGLE_DOWNLOAD_TIMEOUT_SECONDS value; using default 300."
        )
        timeout_s = 300
    with tempfile.TemporaryDirectory(prefix="whymail-kaggle-") as tmp:
        cmd = [kaggle_bin, "datasets", "download", "-d", dataset_slug, "-p", tmp, "--unzip"]
        logger.info(f"Downloading Kaggle dataset: {dataset_slug}")
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=timeout_s)
        except Exception as exc:
            logger.warning(f"Failed to download Kaggle dataset {dataset_slug}: {exc}")
            return []

        csv_paths = list(Path(tmp).rglob("*.csv"))
        if not csv_paths:
            logger.warning(f"No CSV files found in Kaggle dataset {dataset_slug}")
            return []

        for csv_path in csv_paths:
            try:
                with open(csv_path, "r", encoding="utf-8", errors="ignore") as fh:
                    reader = csv.DictReader(fh)
                    if not reader.fieldnames:
                        continue
                    columns = list(reader.fieldnames)
                    text_col = _guess_column(columns, ["text", "message", "email", "content", "body"])
                    label_col = _guess_column(columns, ["label", "target", "class", "spam"])
                    if text_col is None or label_col is None:
                        continue
                    for row in reader:
                        label = _normalise_binary_label(row.get(label_col))
                        text = str(row.get(text_col, "")).strip()
                        if label is None or not text:
                            continue
                        records.append(
                            {
                                "text": text,
                                "label": label,
                                "source": f"kaggle/{dataset_slug}",
                            }
                        )
            except Exception as exc:
                logger.warning(f"Skipping CSV {csv_path}: {exc}")

    if records:
        _save_jsonl(records, data_dir / "raw" / "spam_kaggle.jsonl")
    logger.info(f"Kaggle dataset added {len(records):,} records")
    return records


# ---------------------------------------------------------------------------
# Main entry-point
# ---------------------------------------------------------------------------

def download_all(
    data_dir: str | Path = "data",
    *,
    include_extended_hf: bool = False,
    kaggle_dataset: str = "",
) -> dict[str, list[dict]]:
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
    if include_extended_hf:
        extra_names = os.environ.get(
            "WHYMAIL_EXTRA_HF_SPAM_DATASETS",
            "mrm8488/sms_spam,ShinoharaHare/Spam-Detection",
        )
        spam_records += download_extra_hf_spam(data_dir, extra_names.split(","))
    if kaggle_dataset:
        spam_records += download_kaggle_spam(data_dir, kaggle_dataset)

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
    parser = argparse.ArgumentParser(description="Download WhyMail training datasets")
    parser.add_argument("data_dir", nargs="?", default="data")
    parser.add_argument(
        "--include-extended-hf",
        action="store_true",
        help="Download additional HuggingFace spam datasets (best effort)",
    )
    parser.add_argument(
        "--kaggle-dataset",
        default="",
        help="Optional Kaggle dataset slug for extra spam data, e.g. user/dataset",
    )
    args = parser.parse_args()
    download_all(
        args.data_dir,
        include_extended_hf=args.include_extended_hf,
        kaggle_dataset=args.kaggle_dataset,
    )
