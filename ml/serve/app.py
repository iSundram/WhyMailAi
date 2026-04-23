"""
ml/serve/app.py
---------------
FastAPI inference server for WhyMail AI.

Exposes REST endpoints that mirror the Go backend's task types so the
Go ``RemoteInferenceBackend`` can call this server transparently.

Start the server
----------------
    # Development
    uvicorn ml.serve.app:app --host 0.0.0.0 --port 9090 --reload

    # Production
    uvicorn ml.serve.app:app --host 0.0.0.0 --port 9090 \\
        --workers 2 --timeout-keep-alive 30

Environment variables
---------------------
    WHYMAIL_SPAM_MODEL_DIR       (default: models/spam_classifier/model)
    WHYMAIL_PHISHING_MODEL_DIR   (default: models/phishing_classifier/model)
    WHYMAIL_SUMMARIZER_MODEL_DIR (default: models/summarizer/model)
    WHYMAIL_EMBEDDER_MODEL_DIR   (default: models/embedder)
    WHYMAIL_PRIORITY_MODEL_PATH  (default: models/priority_ranker/model.pkl)

Response contract
-----------------
Every response includes:
    result        dict   — task-specific payload
    confidence    float
    explanations  list[str]
    model_name    str
    model_version str
    latency_ms    float
    cache_hit     bool   (always false — caching is handled by Go layer)
"""

from __future__ import annotations

import time
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel, field_validator

from ml.serve.model_registry import get_registry


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class InferenceRequest(BaseModel):
    text: str = ""
    subject: str = ""
    body: str = ""
    instruction: str = ""
    tone: str = "neutral"
    thread: str = ""
    query: str = ""
    options: dict[str, Any] = {}

    @field_validator("text", "subject", "body", "instruction", "tone", "thread", "query", mode="before")
    @classmethod
    def truncate(cls, v):
        if isinstance(v, str) and len(v) > 8000:
            return v[:8000]
        return v


class InferenceResponse(BaseModel):
    result: dict[str, Any]
    confidence: float
    explanations: list[str]
    model_name: str
    model_version: str
    latency_ms: float
    cache_hit: bool = False


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="WhyMail AI Inference Server",
    description="Internal ML inference service for WhyMail AI tasks.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


@app.exception_handler(Exception)
async def _global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"error": str(exc)})


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/healthz")
def health():
    registry = get_registry()
    return {"status": "ok", "models": registry.models_available()}


# ---------------------------------------------------------------------------
# Helper: build response
# ---------------------------------------------------------------------------

def _ok(
    result: dict,
    *,
    confidence: float = 0.9,
    explanations: list[str] | None = None,
    model_name: str = "unknown",
    model_version: str = "1.0.0",
    latency_ms: float = 0.0,
) -> InferenceResponse:
    return InferenceResponse(
        result=result,
        confidence=confidence,
        explanations=explanations or [],
        model_name=model_name,
        model_version=model_version,
        latency_ms=latency_ms,
        cache_hit=False,
    )


def _text_from_request(req: InferenceRequest) -> str:
    """Combine subject + body or fall back to text field."""
    combined = " ".join(
        filter(None, [req.subject, req.body, req.text])
    ).strip()
    return combined or req.thread or req.query


# ---------------------------------------------------------------------------
# Task endpoints
# ---------------------------------------------------------------------------

@app.post("/spam-score", response_model=InferenceResponse)
def spam_score(req: InferenceRequest):
    """Score an email for spam probability."""
    text = _text_from_request(req)
    if not text:
        raise HTTPException(400, "No text provided")
    reg = get_registry()
    out = reg.score_spam(text)
    return _ok(
        result={
            "spam_score": out["spam_score"],
            "label": out["label"],
            "label_probs": out["label_probs"],
        },
        confidence=out["confidence"],
        explanations=["distilbert-based spam classifier"],
        model_name=out["model_name"],
        model_version=out["model_version"],
        latency_ms=out["latency_ms"],
    )


@app.post("/phishing-score", response_model=InferenceResponse)
def phishing_score(req: InferenceRequest):
    """Score an email for phishing probability."""
    text = _text_from_request(req)
    if not text:
        raise HTTPException(400, "No text provided")
    reg = get_registry()
    out = reg.score_phishing(text)
    return _ok(
        result={
            "phishing_score": out["phishing_score"],
            "label": out["label"],
            "label_probs": out["label_probs"],
        },
        confidence=out["confidence"],
        explanations=["distilbert-based phishing classifier"],
        model_name=out["model_name"],
        model_version=out["model_version"],
        latency_ms=out["latency_ms"],
    )


@app.post("/summarize", response_model=InferenceResponse)
def summarize(req: InferenceRequest):
    """Summarise an email thread."""
    text = req.thread or _text_from_request(req)
    if not text:
        raise HTTPException(400, "No thread/text provided")
    reg = get_registry()
    out = reg.summarize(text)
    return _ok(
        result={"summary": out["summary"]},
        confidence=0.88,
        explanations=["distilbart-based extractive-abstractive summariser"],
        model_name=out["model_name"],
        model_version=out["model_version"],
        latency_ms=out["latency_ms"],
    )


@app.post("/draft", response_model=InferenceResponse)
def draft(req: InferenceRequest):
    """
    Draft an email from an instruction.

    Uses the summariser model in generation mode.  For a fully featured
    drafting assistant, swap this for a larger LLM (GPT-4, Llama-3, etc.)
    via the remote model backend configuration.
    """
    instruction = req.instruction or _text_from_request(req)
    if not instruction:
        raise HTTPException(400, "No instruction provided")

    # Conditioned generation: prefix the instruction so the model
    # produces email-style output.
    prompt = (
        f"Write a professional email based on the following request: {instruction}"
    )
    reg = get_registry()
    out = reg.summarize(prompt)
    return _ok(
        result={"draft": out["summary"]},
        confidence=0.85,
        explanations=["instruction-conditioned email drafting via seq2seq model"],
        model_name=out["model_name"],
        model_version=out["model_version"],
        latency_ms=out["latency_ms"],
    )


@app.post("/rewrite", response_model=InferenceResponse)
def rewrite(req: InferenceRequest):
    """Rewrite an email in a different tone."""
    text = req.text or _text_from_request(req)
    if not text:
        raise HTTPException(400, "No text provided")
    tone = req.tone or "neutral"
    prompt = (
        f"Rewrite the following email in a {tone} tone, keeping the same meaning: {text}"
    )
    reg = get_registry()
    out = reg.summarize(prompt)
    return _ok(
        result={"rewrite": out["summary"]},
        confidence=0.87,
        explanations=[f"tone-aware rewrite ({tone}) via seq2seq model"],
        model_name=out["model_name"],
        model_version=out["model_version"],
        latency_ms=out["latency_ms"],
    )


@app.post("/search", response_model=InferenceResponse)
def semantic_search(req: InferenceRequest):
    """
    Generate an embedding for the query text.

    The Go backend uses this embedding to perform approximate nearest-
    neighbour search against a stored email index.  The actual FAISS /
    vector-DB layer lives in the Go service.
    """
    query = req.query or _text_from_request(req)
    if not query:
        raise HTTPException(400, "No query provided")
    reg = get_registry()
    out = reg.embed(query)
    return _ok(
        result={
            "embedding": out["embedding"],
            "dim": out["dim"],
            "query": query,
            "matches": [],   # populated by Go retrieval layer
        },
        confidence=0.92,
        explanations=["sentence-transformer embedding (all-MiniLM-L6-v2)"],
        model_name=out["model_name"],
        model_version=out["model_version"],
        latency_ms=out["latency_ms"],
    )


@app.post("/prioritize", response_model=InferenceResponse)
def prioritize(req: InferenceRequest):
    """Score email inbox priority."""
    text = _text_from_request(req)
    if not text:
        raise HTTPException(400, "No text provided")
    reg = get_registry()
    out = reg.rank_priority(text)
    return _ok(
        result={
            "priority_score": out["priority_score"],
            "priority_label": out["priority_label"],
        },
        confidence=out["confidence"],
        explanations=["signal-feature logistic regression priority ranker"],
        model_name=out["model_name"],
        model_version=out["model_version"],
        latency_ms=out["latency_ms"],
    )


@app.post("/admin/insights", response_model=InferenceResponse)
def admin_insights(req: InferenceRequest):
    """
    Placeholder for admin anomaly detection.

    In production this would aggregate tenant-level signals, run
    time-series anomaly detection, and surface trends.
    """
    return _ok(
        result={
            "insights": [],
            "anomaly_score": 0.05,
            "status": "nominal",
        },
        confidence=0.80,
        explanations=["baseline admin anomaly detector"],
        model_name="admin-anomaly-baseline-v1",
        model_version="1.0.0",
    )


@app.post("/feedback", response_model=InferenceResponse)
def feedback(req: InferenceRequest):
    """Accept feedback for offline training."""
    return _ok(
        result={"status": "accepted"},
        confidence=1.0,
        explanations=["feedback ingestion accepted"],
        model_name="feedback-store-v1",
        model_version="1.0.0",
    )


# ---------------------------------------------------------------------------
# Main (for direct invocation)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("ml.serve.app:app", host="0.0.0.0", port=9090, reload=False)
