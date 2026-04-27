"""
ml/tests/test_serve.py
-----------------------
Unit tests for the FastAPI inference server.

These tests mock the model registry so no ML models need to be loaded.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_registry():
    """A mock ModelRegistry that returns deterministic outputs."""
    reg = MagicMock()
    reg.models_available.return_value = {
        "spam": True,
        "phishing": True,
        "summarizer": True,
        "embedder": True,
        "priority": True,
    }
    reg.score_spam.return_value = {
        "spam_score": 0.85,
        "label": "spam",
        "confidence": 0.90,
        "label_probs": {"ham": 0.10, "spam": 0.90},
        "model_name": "spam_classifier",
        "model_version": "1.0.0",
        "latency_ms": 12.3,
    }
    reg.score_phishing.return_value = {
        "phishing_score": 0.92,
        "label": "phishing",
        "confidence": 0.92,
        "label_probs": {"legitimate": 0.08, "phishing": 0.92},
        "model_name": "phishing_classifier",
        "model_version": "1.0.0",
        "latency_ms": 14.1,
    }
    reg.summarize.return_value = {
        "summary": "Meeting scheduled for Monday at 10am.",
        "model_name": "summarizer",
        "model_version": "1.0.0",
        "latency_ms": 45.0,
    }
    reg.embed.return_value = {
        "embedding": [0.1] * 384,
        "dim": 384,
        "model_name": "embedder",
        "model_version": "1.0.0",
        "latency_ms": 8.5,
    }
    reg.rank_priority.return_value = {
        "priority_score": 0.9,
        "priority_label": "high",
        "confidence": 0.85,
        "model_name": "priority_ranker",
        "model_version": "1.0.0",
        "latency_ms": 2.1,
    }
    return reg


@pytest.fixture()
def client(mock_registry):
    with patch("ml.serve.app.get_registry", return_value=mock_registry):
        from ml.serve.app import app
        with TestClient(app) as c:
            yield c


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------

class TestHealth:
    def test_ok(self, client):
        resp = client.get("/healthz")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "models" in data


# ---------------------------------------------------------------------------
# Spam score
# ---------------------------------------------------------------------------

class TestSpamScore:
    def test_basic(self, client):
        resp = client.post("/spam-score", json={"subject": "Free offer", "body": "Buy now!"})
        assert resp.status_code == 200
        data = resp.json()
        assert "spam_score" in data["result"]
        assert data["confidence"] > 0

    def test_missing_text_returns_400(self, client):
        resp = client.post("/spam-score", json={})
        assert resp.status_code == 400

    def test_response_fields(self, client):
        resp = client.post("/spam-score", json={"text": "Buy now discount offer"})
        data = resp.json()
        assert "result" in data
        assert "confidence" in data
        assert "explanations" in data
        assert "model_name" in data
        assert "model_version" in data
        assert "latency_ms" in data


# ---------------------------------------------------------------------------
# Phishing score
# ---------------------------------------------------------------------------

class TestPhishingScore:
    def test_basic(self, client):
        resp = client.post(
            "/phishing-score",
            json={"subject": "Verify account", "body": "Click http://evil.com/login"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "phishing_score" in data["result"]

    def test_missing_text(self, client):
        resp = client.post("/phishing-score", json={})
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Summarize
# ---------------------------------------------------------------------------

class TestSummarize:
    def test_basic(self, client):
        resp = client.post(
            "/summarize",
            json={"thread": "Alice: Can we meet Monday? Bob: Sure, 10am works. Alice: Perfect."},
        )
        assert resp.status_code == 200
        assert "summary" in resp.json()["result"]

    def test_missing_text(self, client):
        resp = client.post("/summarize", json={})
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Draft
# ---------------------------------------------------------------------------

class TestDraft:
    def test_basic(self, client):
        resp = client.post("/draft", json={"instruction": "Follow up on the Q3 report"})
        assert resp.status_code == 200
        assert "draft" in resp.json()["result"]

    def test_missing_instruction(self, client):
        resp = client.post("/draft", json={})
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Rewrite
# ---------------------------------------------------------------------------

class TestRewrite:
    def test_basic(self, client):
        resp = client.post(
            "/rewrite",
            json={"text": "Hey could u send that file?", "tone": "formal"},
        )
        assert resp.status_code == 200
        assert "rewrite" in resp.json()["result"]

    def test_missing_text(self, client):
        resp = client.post("/rewrite", json={})
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Search (embedding)
# ---------------------------------------------------------------------------

class TestSearch:
    def test_basic(self, client):
        resp = client.post("/search", json={"query": "emails about project deadline"})
        assert resp.status_code == 200
        data = resp.json()
        assert "embedding" in data["result"]
        assert len(data["result"]["embedding"]) == 384

    def test_missing_query(self, client):
        resp = client.post("/search", json={})
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Prioritize
# ---------------------------------------------------------------------------

class TestPrioritize:
    def test_basic(self, client):
        resp = client.post("/prioritize", json={"subject": "URGENT", "body": "Please respond ASAP"})
        assert resp.status_code == 200
        data = resp.json()
        assert "priority_score" in data["result"]
        assert "priority_label" in data["result"]

    def test_missing_text(self, client):
        resp = client.post("/prioritize", json={})
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Admin insights
# ---------------------------------------------------------------------------

class TestAdminInsights:
    def test_basic(self, client):
        resp = client.post("/admin/insights", json={"text": "tenant stats"})
        assert resp.status_code == 200
        data = resp.json()
        assert "insights" in data["result"]
        assert "anomaly_score" in data["result"]


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------

class TestFeedback:
    def test_basic(self, client):
        resp = client.post("/feedback", json={"text": "This was not spam"})
        assert resp.status_code == 200
        assert resp.json()["result"]["status"] == "accepted"


# ---------------------------------------------------------------------------
# Text truncation
# ---------------------------------------------------------------------------

class TestInputTruncation:
    def test_long_text_is_accepted(self, client):
        long_text = "spam word " * 1000
        resp = client.post("/spam-score", json={"text": long_text})
        assert resp.status_code == 200
