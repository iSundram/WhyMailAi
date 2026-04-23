"""
ml/train/embedder.py
---------------------
Load and persist a sentence-transformer embedding model for semantic search.

We use ``sentence-transformers/all-MiniLM-L6-v2`` which is excellent
out-of-the-box (trained on 1B+ sentence pairs).  This module wraps the
loading, health-checking, and persistence logic so the inference server
can load it identically.

To fine-tune on email-specific triplets (advanced), extend this module
with a ``SentenceTransformer.fit()`` call using email contrastive pairs
from the Enron corpus.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer, util

from ml.train.config import EmbedderConfig, EMBEDDER_CONFIG


# ---------------------------------------------------------------------------
# Load / save helpers
# ---------------------------------------------------------------------------

def load_embedder(cfg: EmbedderConfig = EMBEDDER_CONFIG) -> SentenceTransformer:
    """Load the sentence-transformer model (downloads once, then cached)."""
    logger.info(f"Loading embedder: {cfg.model_name}")
    return SentenceTransformer(cfg.model_name)


def save_embedder(model: SentenceTransformer, output_dir: str | Path) -> Path:
    """Save the model to *output_dir* for later serving."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save(str(out))
    logger.info(f"Embedder saved → {out}")
    return out


# ---------------------------------------------------------------------------
# Health / smoke tests
# ---------------------------------------------------------------------------

def smoke_test(model: SentenceTransformer) -> dict:
    """
    Run a quick embedding smoke test.

    Returns a dict with cosine-similarity scores for known-similar and
    known-dissimilar sentence pairs.
    """
    pairs = [
        ("Can we reschedule the meeting?", "Please move the call to next week."),
        ("Your account has been suspended.", "Verify your credentials immediately."),
        ("Hi Alice, here's the report you asked for.", "The meeting is at 10am tomorrow."),
    ]
    results = {}
    for a, b in pairs:
        emb_a = model.encode(a, convert_to_tensor=True)
        emb_b = model.encode(b, convert_to_tensor=True)
        score = float(util.cos_sim(emb_a, emb_b).item())
        results[f"{a[:30]}…"] = round(score, 4)
    logger.info(f"Embedder smoke test results: {results}")
    return results


# ---------------------------------------------------------------------------
# Main entry-point
# ---------------------------------------------------------------------------

def setup(
    cfg: EmbedderConfig = EMBEDDER_CONFIG,
) -> Path:
    """
    Download/load the sentence-transformer, run smoke tests, and save.

    Returns the path to the saved model directory.
    """
    model = load_embedder(cfg)
    results = smoke_test(model)
    out = save_embedder(model, cfg.output_dir)

    metrics_path = Path(cfg.output_dir) / "smoke_test.json"
    metrics_path.write_text(json.dumps(results, indent=2))

    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Set up WhyMail embedder")
    parser.add_argument("--model-name", default=EMBEDDER_CONFIG.model_name)
    parser.add_argument("--output-dir", default=EMBEDDER_CONFIG.output_dir)
    args = parser.parse_args()

    cfg = EmbedderConfig(model_name=args.model_name, output_dir=args.output_dir)
    setup(cfg)
