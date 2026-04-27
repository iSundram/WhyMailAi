#!/usr/bin/env bash
# scripts/download_datasets.sh
# ----------------------------
# Download and preprocess all training datasets for WhyMail AI.
#
# Requirements:
#   pip install -r ml/requirements.txt
#
# Usage:
#   bash scripts/download_datasets.sh [DATA_DIR]
#
# DATA_DIR defaults to ./data

set -euo pipefail

DATA_DIR="${1:-data}"
PYTHON="${PYTHON:-python3}"
INCLUDE_EXTENDED_HF="${WHYMAIL_INCLUDE_EXTENDED_HF:-0}"
KAGGLE_DATASET="${WHYMAIL_KAGGLE_DATASET:-}"

echo "=== WhyMail AI Dataset Downloader ==="
echo "Data directory: ${DATA_DIR}"
echo "Python: ${PYTHON}"
echo ""

# Create directory structure
mkdir -p "${DATA_DIR}/raw"
mkdir -p "${DATA_DIR}/spam"
mkdir -p "${DATA_DIR}/phishing"
mkdir -p "${DATA_DIR}/summarization"

# -------------------------------------------------------------------
# 1. Download all datasets
# -------------------------------------------------------------------
echo "[1/4] Downloading datasets from HuggingFace …"
${PYTHON} -c "
from ml.data.download import download_all
download_all(
    '${DATA_DIR}',
    include_extended_hf=bool(int('${INCLUDE_EXTENDED_HF}')),
    kaggle_dataset='${KAGGLE_DATASET}',
)
"

# -------------------------------------------------------------------
# 2. Preprocess
# -------------------------------------------------------------------
echo ""
echo "[2/4] Preprocessing …"
${PYTHON} -c "
import json
from pathlib import Path
from ml.data.preprocess import preprocess_records, preprocess_summarization_records
from ml.data.validate import validate_classification_dataset, validate_summarization_dataset

data_dir = Path('${DATA_DIR}')

# Spam
spam_raw = [json.loads(l) for l in open(data_dir / 'raw' / 'spam_combined.jsonl')]
spam_clean = preprocess_records(spam_raw)
validate_classification_dataset(spam_clean, expected_labels={0, 1})
spam_out = data_dir / 'spam' / 'spam_all.jsonl'
spam_out.parent.mkdir(parents=True, exist_ok=True)
with open(spam_out, 'w') as f:
    for r in spam_clean:
        f.write(json.dumps(r) + '\n')
print(f'Spam: {len(spam_clean):,} clean records → {spam_out}')

# Phishing
phish_raw = [json.loads(l) for l in open(data_dir / 'raw' / 'phishing_emails.jsonl')]
phish_clean = preprocess_records(phish_raw)
validate_classification_dataset(phish_clean, expected_labels={0, 1})
phish_out = data_dir / 'phishing' / 'phishing_all.jsonl'
phish_out.parent.mkdir(parents=True, exist_ok=True)
with open(phish_out, 'w') as f:
    for r in phish_clean:
        f.write(json.dumps(r) + '\n')
print(f'Phishing: {len(phish_clean):,} clean records → {phish_out}')

# Summarization
summ_raw = [json.loads(l) for l in open(data_dir / 'raw' / 'dialogsum.jsonl')]
summ_clean = preprocess_summarization_records(summ_raw)
validate_summarization_dataset(summ_clean)
summ_out = data_dir / 'summarization' / 'dialogsum_clean.jsonl'
summ_out.parent.mkdir(parents=True, exist_ok=True)
with open(summ_out, 'w') as f:
    for r in summ_clean:
        f.write(json.dumps(r) + '\n')
print(f'Summarization: {len(summ_clean):,} clean records → {summ_out}')
"

# -------------------------------------------------------------------
# 3. Split into train / val / test
# -------------------------------------------------------------------
echo ""
echo "[3/4] Splitting datasets …"
${PYTHON} -c "
import json
from pathlib import Path
from ml.data.split import stratified_split, save_splits

data_dir = Path('${DATA_DIR}')

def load_jsonl(p):
    return [json.loads(l) for l in open(p)]

def split_and_save(path, out_dir, prefix):
    records = load_jsonl(path)
    train, val, test = stratified_split(records)
    save_splits(train, val, test, out_dir, prefix=prefix)
    return len(train), len(val), len(test)

tr, v, te = split_and_save(
    data_dir / 'spam' / 'spam_all.jsonl',
    data_dir / 'spam', 'spam_'
)
print(f'Spam splits — train:{tr:,} val:{v:,} test:{te:,}')

tr, v, te = split_and_save(
    data_dir / 'phishing' / 'phishing_all.jsonl',
    data_dir / 'phishing', 'phishing_'
)
print(f'Phishing splits — train:{tr:,} val:{v:,} test:{te:,}')

tr, v, te = split_and_save(
    data_dir / 'summarization' / 'dialogsum_clean.jsonl',
    data_dir / 'summarization', 'summarization_'
)
print(f'Summarization splits — train:{tr:,} val:{v:,} test:{te:,}')
"

# -------------------------------------------------------------------
# 4. Validate final splits
# -------------------------------------------------------------------
echo ""
echo "[4/4] Validating final splits …"
${PYTHON} -c "
import json
from pathlib import Path
from ml.data.validate import validate_classification_dataset

data_dir = Path('${DATA_DIR}')

def load_jsonl(p):
    return [json.loads(l) for l in open(p)]

for name, prefix, subdir in [('spam', 'spam_', 'spam'), ('phishing', 'phishing_', 'phishing')]:
    for split in ['train', 'val', 'test']:
        p = data_dir / subdir / f'{prefix}{split}.jsonl'
        records = load_jsonl(p)
        report = validate_classification_dataset(records, expected_labels={0, 1})
        print(f'{name}/{split}: {report.valid:,} valid, dist={report.label_distribution}')
"

echo ""
echo "=== Dataset preparation complete ==="
echo "Run training with:"
echo "  go run ./cmd/trainer --task all --data-dir ${DATA_DIR}"
