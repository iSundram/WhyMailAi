# WhyMail AI — Training Workflow

This document describes how to download data, train all ML models, evaluate
them, and wire the trained models into the live inference server.

---

## Overview

The WhyMail AI ML pipeline (`ml/`) produces four models:

| Model | Task | Base Architecture |
|---|---|---|
| `spam_classifier` | Binary spam / ham classification | DistilBERT |
| `phishing_classifier` | Binary phishing / legitimate classification | DistilBERT |
| `summarizer` | Email thread → concise summary | DistilBART |
| `embedder` | Semantic search embeddings | all-MiniLM-L6-v2 |
| `priority_ranker` | Inbox priority scoring | Logistic Regression |

These models are served by a FastAPI inference server (`ml/serve/app.py`)
which is called by the Go API layer via the `RemoteInferenceBackend`.

---

## Prerequisites

```bash
# Python 3.10+ required
pip install -r ml/requirements.txt
```

---

## Step 1 — Download and Prepare Datasets

```bash
bash scripts/download_datasets.sh data/
```

This script:
1. Downloads `sms_spam` and `SetFit/enron_spam` from HuggingFace (~39 k combined spam/ham emails).
2. Downloads `ealvaradob/phishing-email-dataset` (phishing emails).
3. Downloads `knkarthick/dialogsum` (conversation summarisation).
4. Cleans, normalises, and anonymises all records.
5. Produces stratified 80/10/10 train/val/test splits.
6. Validates label distribution and quality.

### Manual download (alternative)

```bash
python -m ml.data.download data/
```

---

## Step 2 — Train All Models

```bash
go run ./cmd/trainer --task all --data-dir data/ --models-dir models/
```

Or train individual models:

```bash
# Spam classifier
go run ./cmd/trainer --task spam

# Phishing classifier
go run ./cmd/trainer --task phishing

# Summariser
go run ./cmd/trainer --task summarizer

# Sentence embedder (download + smoke test)
go run ./cmd/trainer --task embedder

# Priority ranker
go run ./cmd/trainer --task priority
```

### Training directly via Python

```bash
python -m ml.train.spam_classifier \
    --data-dir data/spam \
    --output-dir models/spam_classifier \
    --epochs 4 \
    --fp16   # set if GPU available

python -m ml.train.phishing_classifier \
    --data-dir data/phishing \
    --output-dir models/phishing_classifier

python -m ml.train.summarizer \
    --data-dir data/summarization \
    --output-dir models/summarizer

python -m ml.train.embedder \
    --output-dir models/embedder

python -m ml.train.priority_ranker \
    --output-dir models/priority_ranker \
    --data-dir data/spam
```

### Hyper-parameters

All training hyper-parameters are defined in `ml/train/config.py`.
Edit `ClassifierConfig`, `SummarizationConfig`, etc. before running training.

Metrics are logged to MLflow (default tracking URI: `./mlruns`).
View them with:

```bash
mlflow ui
```

---

## Step 3 — Evaluate Models

```bash
go run ./cmd/evaluator \
    --models-dir models/ \
    --data-dir data/ \
    --report-out evaluation_report.json
```

Or directly:

```bash
python -c "
from ml.evaluate.harness import run_full_harness
results = run_full_harness('models/', 'data/', report_path='eval.json')
for r in results:
    print(r.model_name, '→', 'PROMOTED' if r.promoted else 'FAILED')
"
```

### Promotion gates

A model is promoted to production only if it passes **all** thresholds:

| Model | min Precision | min Recall | min F1 |
|---|---|---|---|
| spam_classifier | 0.90 | 0.85 | 0.88 |
| phishing_classifier | 0.92 | 0.88 | 0.90 |

If a model fails the gate, the training script raises a `ValueError` and
the model is **not** saved.  The previous version remains in `models/`.

---

## Step 4 — Start the Inference Server

```bash
uvicorn ml.serve.app:app --host 0.0.0.0 --port 9090

# With GPU and multiple workers:
uvicorn ml.serve.app:app --host 0.0.0.0 --port 9090 \
    --workers 2 --timeout-keep-alive 30
```

### Model path overrides (environment variables)

```bash
export WHYMAIL_SPAM_MODEL_DIR=models/spam_classifier/model
export WHYMAIL_PHISHING_MODEL_DIR=models/phishing_classifier/model
export WHYMAIL_SUMMARIZER_MODEL_DIR=models/summarizer/model
export WHYMAIL_EMBEDDER_MODEL_DIR=models/embedder
export WHYMAIL_PRIORITY_MODEL_PATH=models/priority_ranker/model.pkl
```

---

## Step 5 — Connect the Go API

Set `WHYMAIL_REMOTE_INFERENCE_URL` before starting the Go server:

```bash
export WHYMAIL_REMOTE_INFERENCE_URL=http://localhost:9090
go run ./cmd/ai-api
```

The `RemoteInferenceBackend` will be automatically selected when available.
If the Python server is unreachable, the Go server falls back silently to
the built-in heuristic backends — mail delivery is never interrupted.

---

## Testing

### Python tests

```bash
cd /path/to/whymail-ai
pytest ml/tests/ -v
```

Tests mock the ML models so no GPU or internet connection is required.

### Go tests

```bash
go test ./...
```

---

## Architecture Summary

```
User Request
    │
    ▼
Go API Layer (cmd/ai-api)
    │
    ▼
Orchestration Router
    ├── RemoteInferenceBackend  ──►  Python FastAPI (ml/serve/app.py)
    │       (preferred)                 │
    │                                   ├── spam_classifier (DistilBERT)
    │                                   ├── phishing_classifier (DistilBERT)
    │                                   ├── summarizer (DistilBART)
    │                                   ├── embedder (all-MiniLM-L6-v2)
    │                                   └── priority_ranker (LogReg)
    │
    └── LocalPrimaryBackend     (fallback when Python server unavailable)
        LocalFallbackBackend
```

---

## Rollback

To roll back to a previous model version:

1. Keep previous model directories versioned (e.g. `models/spam_classifier_v1/`).
2. Point `WHYMAIL_SPAM_MODEL_DIR` at the previous version's `model/` subdirectory.
3. Restart the inference server.

---

## Experiment Tracking

All training runs are logged to MLflow.  To view:

```bash
mlflow ui --port 5000
```

Navigate to `http://localhost:5000` to compare experiments, view metrics, and
download artefacts.
