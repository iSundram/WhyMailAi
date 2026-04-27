"""
ml/tests/test_train.py
-----------------------
Smoke tests for training pipelines.

These tests do NOT download real models or datasets from the internet.
They use tiny synthetic data and a minimal model config to verify that
the training code runs end-to-end without errors.
"""

import json
import pickle
import tempfile
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_jsonl(records, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


def _spam_records(n=30, label=None):
    texts_spam = [
        "buy now free offer discount limited time urgent winner",
        "click here unsubscribe free gift prize today offer",
    ]
    texts_ham = [
        "please find the report attached as discussed in the meeting",
        "hi team could you review the document before friday thanks",
    ]
    records = []
    for i in range(n):
        lbl = label if label is not None else (i % 2)
        text = texts_spam[i % 2] if lbl == 1 else texts_ham[i % 2]
        records.append({"text": f"{text} example {i}", "label": lbl, "source": "test"})
    return records


# ---------------------------------------------------------------------------
# Config smoke tests
# ---------------------------------------------------------------------------

class TestTrainingConfig:
    def test_classifier_config_to_dict(self):
        from ml.train.config import ClassifierConfig
        cfg = ClassifierConfig()
        d = cfg.to_dict()
        assert "model_name" in d
        assert "num_epochs" in d

    def test_config_save_load(self, tmp_path):
        from ml.train.config import ClassifierConfig
        cfg = ClassifierConfig(num_epochs=3, seed=99)
        path = tmp_path / "cfg.json"
        cfg.save(path)
        loaded = ClassifierConfig.load(path)
        assert loaded.num_epochs == 3
        assert loaded.seed == 99

    def test_summarization_config(self):
        from ml.train.config import SummarizationConfig
        cfg = SummarizationConfig()
        assert cfg.max_source_length > 0
        assert cfg.max_target_length > 0

    def test_priority_ranker_config(self):
        from ml.train.config import PriorityRankerConfig
        cfg = PriorityRankerConfig()
        assert cfg.max_iter > 0


# ---------------------------------------------------------------------------
# Priority ranker (no external model, fast)
# ---------------------------------------------------------------------------

class TestPriorityRanker:
    def _make_data(self, n=100):
        texts = [
            "urgent meeting deadline today asap",
            "please find the report attached",
            "hi could you help me with this when you get a chance",
            "important: action required immediately",
            "this is just a general update fyi",
        ]
        return [{"text": texts[i % len(texts)], "label": 0, "source": "test"} for i in range(n)]

    def test_extract_features_shape(self):
        from ml.train.priority_ranker import extract_features
        feats = extract_features("urgent meeting today please")
        assert len(feats) == 6
        assert all(isinstance(f, float) for f in feats)

    def test_urgency_detection(self):
        from ml.train.priority_ranker import extract_features
        feats = extract_features("urgent deadline today asap")
        has_urgency = feats[0]
        assert has_urgency == 1.0

    def test_no_signals(self):
        from ml.train.priority_ranker import extract_features
        feats = extract_features("hello world")
        # no urgency, no meeting, no question, no action, no attachment
        assert feats[0] == 0.0
        assert feats[1] == 0.0

    def test_train_and_predict(self, tmp_path):
        from ml.train.priority_ranker import train, extract_features, PRIORITY_CONFIG
        from ml.train.config import PriorityRankerConfig

        cfg = PriorityRankerConfig(
            output_dir=str(tmp_path / "priority"),
            max_iter=100,
        )
        records = self._make_data(80)
        model_path = train(cfg, records=records)
        assert model_path.exists()

        # Load and predict
        with open(model_path, "rb") as fh:
            model = pickle.load(fh)
        feats = extract_features("urgent meeting deadline today")
        proba = model.predict_proba([feats])[0]
        assert len(proba) > 0
        assert abs(sum(proba) - 1.0) < 1e-6

    def test_label_priority_heuristic(self):
        from ml.train.priority_ranker import _label_priority
        high = _label_priority({"text": "urgent deadline today asap meeting"})
        low = _label_priority({"text": "here is a casual update"})
        assert high > low


# ---------------------------------------------------------------------------
# Feature extraction edge cases
# ---------------------------------------------------------------------------

class TestFeatureEdgeCases:
    def test_empty_text(self):
        from ml.train.priority_ranker import extract_features
        feats = extract_features("")
        assert len(feats) == 6

    def test_question_detection(self):
        from ml.train.priority_ranker import extract_features
        feats = extract_features("Can you please review this?")
        has_question = feats[2]
        assert has_question == 1.0

    def test_attachment_detection(self):
        from ml.train.priority_ranker import extract_features
        feats = extract_features("Please see the attached document")
        has_attachment = feats[4]
        assert has_attachment == 1.0

    def test_length_bucket_short(self):
        from ml.train.priority_ranker import extract_features
        feats = extract_features("short")
        assert feats[5] == 0.0

    def test_length_bucket_long(self):
        from ml.train.priority_ranker import extract_features
        feats = extract_features("word " * 200)
        assert feats[5] == 2.0
