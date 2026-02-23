#!/bin/bash
set -e


if [ -z "$1" ]; then
  echo "usage (within repo): ./setup BUCKET_LOCATION"
  echo "ie. BUCKET_NAME/.../REPO_DIR"
  exit 1
fi
S3_BUCKET="$1"

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_NAME="$(basename "$EXPERIMENT_DIR")"

REPO_DIR="$(pwd)"
if [ ! -d "$REPO_DIR/.git" ]; then
  echo "could not find repo root (no .git found)"
  exit 1
fi

S3_BASE="s3://${S3_BUCKET}/toy-transformers/experiments/${EXPERIMENT_NAME}"

echo "[SETUP] experiment: $EXPERIMENT_DIR"
echo "[SETUP] s3 base: $S3_BASE"
echo ""

# setup
DATA_CACHE_DIR="${EXPERIMENT_DIR}/data"
mkdir -p "$DATA_CACHE_DIR"

CACHE_EXISTS=$(aws s3 ls "${S3_BASE}/data/train.pt" 2>/dev/null | wc -l)

if [ "$CACHE_EXISTS" -gt 0 ]; then
  echo "[PREPROCESS] pulling vocab + tokenized cache..."
  aws s3 sync "${S3_BASE}/data" "$DATA_CACHE_DIR/"
else
  echo "[PREPROCESS] no cache found"

  RAW_DIR="${REPO_DIR}/data/raw/simplebooks/simplebooks-92-raw"

  if [ ! -f "${RAW_DIR}/train.txt" ]; then
    echo "[PREPROCESS] downloading data..."
    mkdir -p "$(dirname "$RAW_DIR")"
    wget -q --show-progress -O /tmp/simplebooks.zip \
      "https://dldata-public.s3.us-east-2.amazonaws.com/simplebooks.zip"
    unzip -q /tmp/simplebooks.zip -d "${REPO_DIR}/data/raw/simplebooks/"
    rm /tmp/simplebooks.zip
  else
    echo "[PREPROCESS] raw dataset found locally"
  fi

  echo "[PREPROCESS] running preprocess script..."
  python "${EXPERIMENT_DIR}/preprocess.py"
  echo "[PREPROCESS] pushing vocab + cache to S3..."
  aws s3 sync "${DATA_CACHE_DIR}/" "${S3_BASE}/data/"
  echo "[PREPROCESS] preprocess complete."
fi

echo "[CHECKPOINTS] pull checkpoints..."
CKPT_DIR="${EXPERIMENT_DIR}/checkpoints"
mkdir -p "${CKPT_DIR}"
aws s3 sync "${S3_BASE}/checkpoints/" "$CKPT_DIR/" || true

echo "[TRAINING] starting training..."
python "${EXPERIMENT_DIR}/train.py" "${S3_BUCKET}"