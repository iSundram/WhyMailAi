"""
ml/train/config.py
------------------
Centralised, reproducible training hyper-parameters.

All hyper-parameters are expressed as plain Python dataclasses so they
can be serialised to JSON / YAML for experiment tracking via MLflow.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

@dataclass
class BaseTrainConfig:
    seed: int = 42
    output_dir: str = "models"
    logging_dir: str = "logs"

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def from_dict(cls, d: dict):
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def load(cls, path: str | Path):
        return cls.from_dict(json.loads(Path(path).read_text()))


# ---------------------------------------------------------------------------
# Classification (spam / phishing)
# ---------------------------------------------------------------------------

@dataclass
class ClassifierConfig(BaseTrainConfig):
    # Model
    model_name: str = "distilbert-base-uncased"
    num_labels: int = 2

    # Training
    num_epochs: int = 4
    per_device_train_batch_size: int = 32
    per_device_eval_batch_size: int = 64
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    max_grad_norm: float = 1.0
    fp16: bool = False          # set True when a CUDA GPU is available

    # Tokenisation
    max_length: int = 256

    # Evaluation gates (model is NOT promoted if below these)
    min_precision: float = 0.90
    min_recall: float = 0.85
    min_f1: float = 0.88

    # Data
    data_dir: str = "data/spam"
    output_dir: str = "models/spam_classifier"


@dataclass
class PhishingClassifierConfig(ClassifierConfig):
    min_precision: float = 0.92
    min_recall: float = 0.88
    min_f1: float = 0.90
    data_dir: str = "data/phishing"
    output_dir: str = "models/phishing_classifier"


# ---------------------------------------------------------------------------
# Summarisation
# ---------------------------------------------------------------------------

@dataclass
class SummarizationConfig(BaseTrainConfig):
    model_name: str = "sshleifer/distilbart-cnn-12-6"
    # sshleifer/distilbart-cnn-12-6 is a pre-trained DistilBART that works
    # well out-of-the-box; fine-tuning on DialogSum improves email style.

    # Training
    num_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 8
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    fp16: bool = False
    gradient_accumulation_steps: int = 4

    # Tokenisation
    max_source_length: int = 512
    max_target_length: int = 128
    min_target_length: int = 10

    # Generation
    num_beams: int = 4
    length_penalty: float = 1.0

    # Data
    data_dir: str = "data/summarization"
    output_dir: str = "models/summarizer"


# ---------------------------------------------------------------------------
# Sentence embedder (semantic search)
# ---------------------------------------------------------------------------

@dataclass
class EmbedderConfig(BaseTrainConfig):
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    # This model is excellent out-of-the-box; we load it directly.
    # Fine-tuning on email triplets can be added later via SBERT training.
    output_dir: str = "models/embedder"


# ---------------------------------------------------------------------------
# Priority ranker  (lightweight logistic regression over email signals)
# ---------------------------------------------------------------------------

@dataclass
class PriorityRankerConfig(BaseTrainConfig):
    max_iter: int = 1000
    C: float = 1.0
    class_weight: str = "balanced"
    output_dir: str = "models/priority_ranker"


# ---------------------------------------------------------------------------
# Global defaults
# ---------------------------------------------------------------------------

SPAM_CONFIG = ClassifierConfig()
PHISHING_CONFIG = PhishingClassifierConfig()
SUMMARIZATION_CONFIG = SummarizationConfig()
EMBEDDER_CONFIG = EmbedderConfig()
PRIORITY_CONFIG = PriorityRankerConfig()
