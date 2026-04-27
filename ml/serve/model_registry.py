"""
ml/serve/model_registry.py
--------------------------
Load, cache, and version-track all trained models used by the inference
server.  Models are loaded lazily on first use and kept in memory for the
lifetime of the server process.

Model versions are determined by reading a ``version.json`` file that is
written to each model directory by the training pipeline.  If the file is
absent the version defaults to ``"1.0.0"``.

All inference is synchronous (no async) so the FastAPI server can call
models from thread-pool workers.
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
from loguru import logger


# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------

DEFAULT_SPAM_MODEL_DIR = "models/spam_classifier/model"
DEFAULT_PHISHING_MODEL_DIR = "models/phishing_classifier/model"
DEFAULT_SUMMARIZER_MODEL_DIR = "models/summarizer/model"
DEFAULT_EMBEDDER_MODEL_DIR = "models/embedder"
DEFAULT_PRIORITY_MODEL_DIR = "models/priority_ranker/model.pkl"

_MAX_TEXT_LEN = 4096


# ---------------------------------------------------------------------------
# Version helper
# ---------------------------------------------------------------------------

def _read_version(model_dir: str | Path, default: str = "1.0.0") -> str:
    p = Path(model_dir) / "version.json"
    if p.exists():
        try:
            return json.loads(p.read_text()).get("version", default)
        except Exception:
            pass
    return default


# ---------------------------------------------------------------------------
# Individual model wrappers
# ---------------------------------------------------------------------------

class _DistilBertClassifier:
    """Wrapper around a fine-tuned DistilBERT sequence classifier."""

    def __init__(self, model_dir: str, name: str) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch

        self.name = name
        self.version = _read_version(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self._device)
        logger.info(f"Loaded {name} v{self.version} ({self._device})")

    def predict(
        self, text: str, *, max_length: int = 256
    ) -> tuple[int, float, dict[str, float]]:
        """
        Returns (predicted_label, confidence, label_probs).
        ``label_probs`` maps label names → probability.
        """
        import torch

        enc = self.tokenizer(
            text[:_MAX_TEXT_LEN],
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        ).to(self._device)

        with torch.no_grad():
            logits = self.model(**enc).logits

        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        pred = int(np.argmax(probs))
        label_probs = {
            self.model.config.id2label[i]: float(p) for i, p in enumerate(probs)
        }
        confidence = float(probs[pred])
        return pred, confidence, label_probs


class _Summarizer:
    """Wrapper around a fine-tuned seq2seq summarisation model."""

    def __init__(self, model_dir: str, name: str = "summarizer") -> None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        import torch

        self.name = name
        self.version = _read_version(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
        self.model.eval()
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self._device)
        logger.info(f"Loaded summarizer v{self.version} ({self._device})")

    def summarize(
        self,
        text: str,
        *,
        max_source: int = 512,
        max_target: int = 128,
        num_beams: int = 4,
    ) -> str:
        import torch

        enc = self.tokenizer(
            text[:_MAX_TEXT_LEN],
            return_tensors="pt",
            truncation=True,
            max_length=max_source,
            padding=True,
        ).to(self._device)

        with torch.no_grad():
            out_ids = self.model.generate(
                **enc,
                num_beams=num_beams,
                max_length=max_target,
                early_stopping=True,
            )

        return self.tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()


class _SentenceEmbedder:
    """Wrapper around a sentence-transformer embedding model."""

    def __init__(self, model_dir: str, name: str = "embedder") -> None:
        from sentence_transformers import SentenceTransformer

        self.name = name
        self.version = _read_version(model_dir)
        self.model = SentenceTransformer(model_dir)
        logger.info(f"Loaded embedder v{self.version}")

    def embed(self, text: str) -> list[float]:
        return self.model.encode(text[:_MAX_TEXT_LEN], normalize_embeddings=True).tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        trimmed = [t[:_MAX_TEXT_LEN] for t in texts]
        return self.model.encode(trimmed, normalize_embeddings=True).tolist()


class _PriorityRanker:
    """Wrapper around a scikit-learn priority ranker pipeline."""

    def __init__(self, model_path: str, name: str = "priority_ranker") -> None:
        self.name = name
        self.version = "1.0.0"
        with open(model_path, "rb") as fh:
            self.model = pickle.load(fh)
        logger.info(f"Loaded priority ranker")

    def score(self, features: list[float]) -> tuple[int, float]:
        """Returns (priority_class, probability_of_that_class)."""
        proba = self.model.predict_proba([features])[0]
        cls = int(np.argmax(proba))
        return cls, float(proba[cls])


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class ModelRegistry:
    """
    Singleton model registry.  Models are loaded once on first access
    (lazy loading) and cached for the process lifetime.

    Environment-variable overrides for all model paths:
      WHYMAIL_SPAM_MODEL_DIR
      WHYMAIL_PHISHING_MODEL_DIR
      WHYMAIL_SUMMARIZER_MODEL_DIR
      WHYMAIL_EMBEDDER_MODEL_DIR
      WHYMAIL_PRIORITY_MODEL_PATH
    """

    def __init__(
        self,
        spam_dir: str = DEFAULT_SPAM_MODEL_DIR,
        phishing_dir: str = DEFAULT_PHISHING_MODEL_DIR,
        summarizer_dir: str = DEFAULT_SUMMARIZER_MODEL_DIR,
        embedder_dir: str = DEFAULT_EMBEDDER_MODEL_DIR,
        priority_path: str = DEFAULT_PRIORITY_MODEL_DIR,
    ) -> None:
        self._spam_dir = spam_dir
        self._phishing_dir = phishing_dir
        self._summarizer_dir = summarizer_dir
        self._embedder_dir = embedder_dir
        self._priority_path = priority_path

        self._spam: _DistilBertClassifier | None = None
        self._phishing: _DistilBertClassifier | None = None
        self._summarizer: _Summarizer | None = None
        self._embedder: _SentenceEmbedder | None = None
        self._priority: _PriorityRanker | None = None
        self._lock = Lock()

    # ------------------------------------------------------------------ #
    # Lazy loaders
    # ------------------------------------------------------------------ #

    def _load_spam(self) -> _DistilBertClassifier:
        with self._lock:
            if self._spam is None:
                self._spam = _DistilBertClassifier(self._spam_dir, "spam_classifier")
        return self._spam

    def _load_phishing(self) -> _DistilBertClassifier:
        with self._lock:
            if self._phishing is None:
                self._phishing = _DistilBertClassifier(self._phishing_dir, "phishing_classifier")
        return self._phishing

    def _load_summarizer(self) -> _Summarizer:
        with self._lock:
            if self._summarizer is None:
                self._summarizer = _Summarizer(self._summarizer_dir)
        return self._summarizer

    def _load_embedder(self) -> _SentenceEmbedder:
        with self._lock:
            if self._embedder is None:
                self._embedder = _SentenceEmbedder(self._embedder_dir)
        return self._embedder

    def _load_priority(self) -> _PriorityRanker:
        with self._lock:
            if self._priority is None:
                self._priority = _PriorityRanker(self._priority_path)
        return self._priority

    # ------------------------------------------------------------------ #
    # Availability checks (used by health endpoint)
    # ------------------------------------------------------------------ #

    def models_available(self) -> dict[str, bool]:
        return {
            "spam": Path(self._spam_dir).exists(),
            "phishing": Path(self._phishing_dir).exists(),
            "summarizer": Path(self._summarizer_dir).exists(),
            "embedder": Path(self._embedder_dir).exists(),
            "priority": Path(self._priority_path).exists(),
        }

    # ------------------------------------------------------------------ #
    # High-level inference calls
    # ------------------------------------------------------------------ #

    def score_spam(self, text: str) -> dict[str, Any]:
        t0 = time.perf_counter()
        clf = self._load_spam()
        pred, conf, probs = clf.predict(text)
        ms = (time.perf_counter() - t0) * 1000
        return {
            "spam_score": probs.get("spam", conf if pred == 1 else 1 - conf),
            "label": "spam" if pred == 1 else "ham",
            "confidence": conf,
            "label_probs": probs,
            "model_name": clf.name,
            "model_version": clf.version,
            "latency_ms": round(ms, 2),
        }

    def score_phishing(self, text: str) -> dict[str, Any]:
        t0 = time.perf_counter()
        clf = self._load_phishing()
        pred, conf, probs = clf.predict(text)
        ms = (time.perf_counter() - t0) * 1000
        return {
            "phishing_score": probs.get("phishing", conf if pred == 1 else 1 - conf),
            "label": "phishing" if pred == 1 else "legitimate",
            "confidence": conf,
            "label_probs": probs,
            "model_name": clf.name,
            "model_version": clf.version,
            "latency_ms": round(ms, 2),
        }

    def summarize(self, text: str) -> dict[str, Any]:
        t0 = time.perf_counter()
        s = self._load_summarizer()
        summary = s.summarize(text)
        ms = (time.perf_counter() - t0) * 1000
        return {
            "summary": summary,
            "model_name": s.name,
            "model_version": s.version,
            "latency_ms": round(ms, 2),
        }

    def embed(self, text: str) -> dict[str, Any]:
        t0 = time.perf_counter()
        e = self._load_embedder()
        vector = e.embed(text)
        ms = (time.perf_counter() - t0) * 1000
        return {
            "embedding": vector,
            "dim": len(vector),
            "model_name": e.name,
            "model_version": e.version,
            "latency_ms": round(ms, 2),
        }

    def rank_priority(self, text: str) -> dict[str, Any]:
        from ml.train.priority_ranker import extract_features

        t0 = time.perf_counter()
        ranker = self._load_priority()
        feats = extract_features(text)
        cls, conf = ranker.score(feats)
        ms = (time.perf_counter() - t0) * 1000
        priority_label = ["low", "medium", "high"][min(cls, 2)]
        # Normalise to 0-1 score: low=0.2, medium=0.5, high=0.9
        score = [0.2, 0.5, 0.9][min(cls, 2)]
        return {
            "priority_score": score,
            "priority_label": priority_label,
            "confidence": conf,
            "model_name": ranker.name,
            "model_version": ranker.version,
            "latency_ms": round(ms, 2),
        }


# ---------------------------------------------------------------------------
# Global singleton (imported by app.py)
# ---------------------------------------------------------------------------

import os

_registry: ModelRegistry | None = None
_registry_lock = Lock()


def get_registry() -> ModelRegistry:
    global _registry
    with _registry_lock:
        if _registry is None:
            _registry = ModelRegistry(
                spam_dir=os.environ.get("WHYMAIL_SPAM_MODEL_DIR", DEFAULT_SPAM_MODEL_DIR),
                phishing_dir=os.environ.get("WHYMAIL_PHISHING_MODEL_DIR", DEFAULT_PHISHING_MODEL_DIR),
                summarizer_dir=os.environ.get("WHYMAIL_SUMMARIZER_MODEL_DIR", DEFAULT_SUMMARIZER_MODEL_DIR),
                embedder_dir=os.environ.get("WHYMAIL_EMBEDDER_MODEL_DIR", DEFAULT_EMBEDDER_MODEL_DIR),
                priority_path=os.environ.get("WHYMAIL_PRIORITY_MODEL_PATH", DEFAULT_PRIORITY_MODEL_DIR),
            )
    return _registry
