#!/bin/bash
set -e

if [ -z "$1" ]; then
  echo "usage (within repo): ./setup.bash BUCKET_NAME"
  echo "ie. the raw S3 bucket name (not a full path)"
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

DATA_DIR="${EXPERIMENT_DIR}/data"
SHARD_DIR="${DATA_DIR}/shuffled"
mkdir -p "$SHARD_DIR"

# check that the sharded data exists on S3
METADATA_EXISTS=$(aws s3 ls "${S3_BASE}/data/shuffled/metadata.json" 2>/dev/null | wc -l)
if [ "$METADATA_EXISTS" -eq 0 ]; then
  echo "[DATA] ERROR: no sharded data found at ${S3_BASE}/data/shuffled/"
  echo "[DATA] The FineWeb-Edu dataset must be preprocessed and uploaded to S3 separately."
  exit 1
fi

# pull vocab
echo "[DATA] pulling vocab..."
aws s3 cp "${S3_BASE}/data/vocab_32k.json" "${DATA_DIR}/vocab_32k.json"

# pull shard metadata so train.py can reference it locally
echo "[DATA] pulling shard metadata..."
aws s3 cp "${S3_BASE}/data/shuffled/metadata.json" "${SHARD_DIR}/metadata.json"

echo "[DATA] shards will be streamed on demand during training."
echo ""

# pull existing checkpoints for resume
CKPT_DIR="${EXPERIMENT_DIR}/checkpoints"
mkdir -p "${CKPT_DIR}"
echo "[CHECKPOINTS] pulling checkpoints..."
aws s3 sync "${S3_BASE}/checkpoints/" "${CKPT_DIR}/" || true
echo ""

echo "[TRAINING] starting training..."
python "${EXPERIMENT_DIR}/train.py" "${S3_BUCKET}"
