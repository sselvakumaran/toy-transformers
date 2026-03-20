#!/bin/bash
set -euo pipefail

CONFIG_FILE="${1:?usage: ./setup.sh <config_file> [branch]}"
BRANCH="${2:-main}"
REPO_URL="https://github.com/sselvakumaran/toy-transformers.git"
REPO_DIR="toy-transformers"

# ── load secrets ──
if [[ ! -f .env ]]; then echo "ERROR: .env not found"; exit 1; fi
source .env
: "${AWS_ACCESS_KEY_ID:?missing}" "${AWS_SECRET_ACCESS_KEY:?missing}" "${BUCKET:?missing}"

# ── system deps (skip what's already present) ──
echo "[SETUP] checking system dependencies..."
MISSING=()
command -v gcc    >/dev/null || MISSING+=(build-essential)
command -v git    >/dev/null || MISSING+=(git)
command -v aws    >/dev/null || MISSING+=(awscli)
python3 --version >/dev/null 2>&1 || MISSING+=(python3)

# these have no binaries to check — use dpkg
dpkg -s python3-venv >/dev/null 2>&1 || MISSING+=(python3-venv)
dpkg -s python3-dev  >/dev/null 2>&1 || MISSING+=(python3-dev)

if [[ ${#MISSING[@]} -gt 0 ]]; then
  echo "[SETUP] installing: ${MISSING[*]}"
  apt-get update -qq && apt-get install -y -qq "${MISSING[@]}"
else
  echo "[SETUP] all system deps present"
fi

# ── verify CUDA toolkit for torch.compile ──
if command -v nvcc >/dev/null; then
  NVCC_VER=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')
  echo "[SETUP] CUDA toolkit: $NVCC_VER"
else
  echo "[WARN] nvcc not found — torch.compile will likely fail"
  echo "[WARN] consider using a devel base image"
fi

# ── AWS config ──
aws configure set aws_access_key_id "$AWS_ACCESS_KEY_ID"
aws configure set aws_secret_access_key "$AWS_SECRET_ACCESS_KEY"
aws configure set region "${AWS_REGION:-us-east-1}"

# ── clone / update repo ──
if [[ -d "$REPO_DIR" ]]; then
  echo "[SETUP] repo exists, pulling latest..."
  cd "$REPO_DIR" && git fetch && git checkout "$BRANCH" && git pull
else
  git clone --single-branch --branch "$BRANCH" "$REPO_URL" "$REPO_DIR"
  cd "$REPO_DIR"
fi

RUN_USER="${RUN_USER:-$(whoami)}"
chown -R "$RUN_USER":"$RUN_USER" .

# ── python env + deps (conditional) ──
if python3 -c "import torch" 2>/dev/null; then
  echo "[SETUP] torch already available, skipping venv creation"
  pip install --quiet numpy tqdm pyarrow 2>/dev/null \
    || pip install --quiet --break-system-packages numpy tqdm pyarrow
else
  echo "[SETUP] torch not found, setting up venv..."
  python3 -m venv venv
  source venv/bin/activate
  pip install --upgrade pip

  TORCH_INDEX="https://download.pytorch.org/whl/cu126"
  echo "[SETUP] installing torch from $TORCH_INDEX"
  pip install torch==2.8.0 --index-url "$TORCH_INDEX"
  pip install numpy tqdm pyarrow
fi

# ── sanity check ──
python3 -c "
import torch
print(f'[CHECK] torch {torch.__version__}')
print(f'[CHECK] CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'[CHECK] GPU: {torch.cuda.get_device_name(0)}')
    print(f'[CHECK] CUDA version: {torch.version.cuda}')
"

# ── launch training ──
echo "[SETUP] starting training..."
python3 -m toy_transformers.train "$CONFIG_FILE" "$BUCKET"