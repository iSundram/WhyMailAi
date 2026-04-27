"""
ml/data/split.py
----------------
Produce reproducible, stratified train / validation / test splits.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import TypeVar

from loguru import logger

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Core split function
# ---------------------------------------------------------------------------

def stratified_split(
    records: list[dict],
    *,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    label_key: str = "label",
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Return (train, val, test) with stratified class distribution.

    Parameters
    ----------
    records     : list of dicts; each must contain *label_key*
    train_ratio : fraction allocated to training (default 0.80)
    val_ratio   : fraction allocated to validation (default 0.10)
    test_ratio  : fraction allocated to test (default 0.10)
    label_key   : field used for stratification (default ``"label"``)
    seed        : random seed for reproducibility (default 42)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, (
        "Ratios must sum to 1.0"
    )

    rng = random.Random(seed)

    # Group by label
    by_label: dict[int, list[dict]] = {}
    for rec in records:
        lbl = rec.get(label_key, 0)
        by_label.setdefault(lbl, []).append(rec)

    train, val, test = [], [], []

    for lbl, group in by_label.items():
        shuffled = list(group)
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train.extend(shuffled[:n_train])
        val.extend(shuffled[n_train : n_train + n_val])
        test.extend(shuffled[n_train + n_val :])

    # Final shuffle so labels are interleaved
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    logger.info(
        f"Split → train: {len(train):,}, val: {len(val):,}, test: {len(test):,} "
        f"| class dist train: {dict(Counter(r[label_key] for r in train))}"
    )
    return train, val, test


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def save_splits(
    train: list[dict],
    val: list[dict],
    test: list[dict],
    out_dir: Path,
    prefix: str = "",
) -> None:
    """Write train/val/test JSONL files to *out_dir*."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, split in [("train", train), ("val", val), ("test", test)]:
        fname = f"{prefix}{name}.jsonl" if prefix else f"{name}.jsonl"
        path = out_dir / fname
        with open(path, "w", encoding="utf-8") as fh:
            for rec in split:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        logger.info(f"Saved {len(split):,} records → {path}")


def load_split(path: Path) -> list[dict]:
    """Load a JSONL split file."""
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
