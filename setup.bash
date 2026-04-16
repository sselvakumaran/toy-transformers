#!/bin/bash
set -euo pipefail

CONFIG_FILE="${1:?usage: ./setup.bash <config_file> [branch]}"
BRANCH="${2:-main}"
REPO_URL="https://github.com/sselvakumaran/toy-transformers.git"
REPO_DIR="toy-transformers"
RUN_USER="${RUN_USER:-$(id -un)}"
SESSION_NAME="train"

# ── sudo shim: use sudo if we're not root and sudo exists ──
if [[ $EUID -eq 0 ]]; then SUDO=""; else SUDO="${SUDO:-sudo}"; fi

# ── load secrets ──
if [[ ! -f .env ]]; then echo "ERROR: .env not found"; exit 1; fi
set -a; source .env; set +a
: "${AWS_ACCESS_KEY_ID:?missing}" "${AWS_SECRET_ACCESS_KEY:?missing}" "${BUCKET:?missing}"

PY_VER="${PY_VER:-3.10}"

# ── system deps (skip what's already present) ──
echo "[SETUP] checking system dependencies..."
MISSING=()
command -v git    >/dev/null || MISSING+=(git)
command -v tmux   >/dev/null || MISSING+=(tmux)
command -v curl   >/dev/null || MISSING+=(curl)
command -v unzip  >/dev/null || MISSING+=(unzip)

if [[ ${#MISSING[@]} -gt 0 ]]; then
  echo "[SETUP] installing: ${MISSING[*]}"
  $SUDO apt-get update -qq
  DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y -qq "${MISSING[@]}"
else
  echo "[SETUP] base deps present"
fi

# ── python ${PY_VER} via deadsnakes (required for torch 2.8 wheel + project parity) ──
if ! command -v "python${PY_VER}" >/dev/null 2>&1; then
  echo "[SETUP] installing python${PY_VER}..."
  DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y -qq software-properties-common
  $SUDO add-apt-repository -y ppa:deadsnakes/ppa
  $SUDO apt-get update -qq
  DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y -qq \
    "python${PY_VER}" "python${PY_VER}-venv" "python${PY_VER}-dev" "python${PY_VER}-distutils" || \
    DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y -qq \
      "python${PY_VER}" "python${PY_VER}-venv" "python${PY_VER}-dev"
fi

# ensure pip for this interpreter
if ! "python${PY_VER}" -m pip --version >/dev/null 2>&1; then
  echo "[SETUP] bootstrapping pip for python${PY_VER}..."
  curl -fsSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
  "python${PY_VER}" /tmp/get-pip.py --user || $SUDO "python${PY_VER}" /tmp/get-pip.py
  rm -f /tmp/get-pip.py
fi
echo "[SETUP] python: $(command -v python${PY_VER}) ($(python${PY_VER} --version))"

# ── GPU / CUDA probe ──
HAS_GPU=0
GPU_NAME=""
COMPUTE_CAP=""
DRIVER_CUDA=""
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  HAS_GPU=1
  GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 || true)
  COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d ' ' || true)
  DRIVER_CUDA=$(nvidia-smi | grep -oE 'CUDA Version:\s*[0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+' | head -n1 || true)
  echo "[SETUP] GPU: $GPU_NAME  compute_cap=$COMPUTE_CAP  driver_cuda=$DRIVER_CUDA"
else
  echo "[WARN] no GPU detected — will install CPU torch"
fi

# Decide pytorch wheel index based on GPU compute capability + driver CUDA.
# Prefer newest wheel the driver supports; ensure sm is covered.
# torch 2.x stable ships only cu118 / cu126 / cu128 (+ cpu).
# CUDA minor-version compat means cu126 runs on any driver that reports 12.x.
pick_torch_index() {
  if [[ $HAS_GPU -eq 0 ]]; then
    echo "https://download.pytorch.org/whl/cpu"; return
  fi
  local cap="${COMPUTE_CAP%.*}${COMPUTE_CAP#*.}"   # "8.9" -> "89", "12.0" -> "120"
  local dmaj="${DRIVER_CUDA%%.*}"
  # Blackwell (sm_100, sm_120: B100/B200, RTX 5090) requires cu128.
  if [[ -n "$cap" && "$cap" -ge 100 ]]; then
    echo "https://download.pytorch.org/whl/cu128"; return
  fi
  # Any 12.x driver → cu126 (covers Hopper/Ada/Ampere: H100, L40, 4090, A100, 3090, A10, L4).
  if [[ "$dmaj" == "12" || "$dmaj" == "13" ]]; then
    echo "https://download.pytorch.org/whl/cu126"; return
  fi
  # Legacy 11.x drivers.
  echo "https://download.pytorch.org/whl/cu118"
}

# ── pip helpers: install to pinned python${PY_VER} without venv ──
PY="python${PY_VER}"
PIP_FLAGS=(--no-input --disable-pip-version-check)
pip_install() {
  # try normal; on PEP 668 "externally-managed" error, retry with --break-system-packages
  if ! $PY -m pip install "${PIP_FLAGS[@]}" "$@" 2>/tmp/pip_err; then
    if grep -q 'externally-managed-environment' /tmp/pip_err 2>/dev/null; then
      $PY -m pip install "${PIP_FLAGS[@]}" --break-system-packages "$@"
    else
      cat /tmp/pip_err >&2; return 1
    fi
  fi
}

# ── torch: use preinstalled if it works on this GPU; else (re)install ──
TORCH_OK=0
if $PY -c "import torch" 2>/dev/null; then
  echo "[SETUP] torch already installed: $($PY -c 'import torch;print(torch.__version__)')"
  if [[ $HAS_GPU -eq 0 ]]; then
    TORCH_OK=1
  else
    # probe: can we actually run a kernel on this GPU?
    # this catches the "no kernel image is available for execution" sm_mismatch case.
    if $PY - <<'PYEOF' 2>/tmp/torch_probe; then
import os, sys, torch, warnings
warnings.filterwarnings("error")
assert torch.cuda.is_available(), "cuda not available"
x = torch.zeros(8, device="cuda")
y = (x + 1).sum().item()
arch = torch.cuda.get_device_capability(0)
print(f"[PROBE] ok sm={arch[0]}{arch[1]} torch_cuda={torch.version.cuda}")
PYEOF
      TORCH_OK=1
    else
      echo "[SETUP] preinstalled torch failed GPU probe:"
      sed 's/^/  /' /tmp/torch_probe >&2
      echo "[SETUP] will reinstall torch matching this GPU"
    fi
  fi
fi

if [[ $TORCH_OK -eq 0 ]]; then
  TORCH_INDEX=$(pick_torch_index)
  echo "[SETUP] installing torch from $TORCH_INDEX"
  pip_install --force-reinstall --index-url "$TORCH_INDEX" torch==2.8.0 \
    || pip_install --index-url "$TORCH_INDEX" torch
fi

# ── extra deps ──
pip_install --quiet numpy tqdm pyarrow awscli

# make sure the pip-installed aws is on PATH (--user installs land in ~/.local/bin)
if ! command -v aws >/dev/null 2>&1; then
  USER_BIN="$($PY -c 'import site,os;print(os.path.join(site.getuserbase(),"bin"))' 2>/dev/null || echo "$HOME/.local/bin")"
  if [[ -x "$USER_BIN/aws" ]]; then
    export PATH="$USER_BIN:$PATH"
    hash -r
  fi
fi
command -v aws >/dev/null || { echo "ERROR: aws still not on PATH after pip install"; exit 1; }
echo "[SETUP] aws: $(command -v aws) ($(aws --version 2>&1 | head -n1))"

# ── AWS config ──
aws_set() {
  local user="$1"; shift
  if [[ "$user" == "$(id -un)" ]]; then
    aws configure set "$@"
  else
    $SUDO -u "$user" aws configure set "$@"
  fi
}
aws_set "$RUN_USER" aws_access_key_id     "$AWS_ACCESS_KEY_ID"
aws_set "$RUN_USER" aws_secret_access_key "$AWS_SECRET_ACCESS_KEY"
aws_set "$RUN_USER" region                "${AWS_REGION:-us-east-1}"

# ── clone / update repo under RUN_USER home ──
RUN_HOME=$(eval echo "~$RUN_USER")
cd "$RUN_HOME"

if [[ -d "$REPO_DIR/.git" ]]; then
  echo "[SETUP] repo exists, pulling latest..."
  (cd "$REPO_DIR" && git fetch --quiet && git checkout "$BRANCH" && git pull --ff-only)
else
  git clone --single-branch --branch "$BRANCH" "$REPO_URL" "$REPO_DIR"
fi

if [[ "$RUN_USER" != "$(id -un)" ]]; then
  $SUDO chown -R "$RUN_USER":"$RUN_USER" "$RUN_HOME/$REPO_DIR"
fi

# ── final sanity check ──
$PY - <<'PYEOF'
import torch
print(f"[CHECK] torch {torch.__version__}  cuda_build={torch.version.cuda}")
print(f"[CHECK] CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[CHECK] GPU: {torch.cuda.get_device_name(0)}  sm={''.join(map(str,torch.cuda.get_device_capability(0)))}")
    print(f"[CHECK] arch_list: {torch.cuda.get_arch_list()}")
    # final kernel exec probe
    (torch.zeros(1, device='cuda') + 1).cpu()
    print("[CHECK] kernel exec ok")
PYEOF

# ── launch training in tmux ──
# tee into a logfile so output survives if the pane dies, and exec bash at the
# end so the pane stays open after training exits (crash or clean) for attach.
LOG_FILE="$RUN_HOME/$REPO_DIR/train.log"
TRAIN_CMD="cd $RUN_HOME/$REPO_DIR && \
  git fetch --quiet && git checkout $BRANCH && git pull --ff-only && \
  $PY -u -m toy_transformers.train $CONFIG_FILE $BUCKET 2>&1 | tee -a $LOG_FILE; \
  rc=\${PIPESTATUS[0]}; \
  echo; echo \"[train exited rc=\$rc — pane left open, log: $LOG_FILE]\"; \
  exec bash"

if [[ "$RUN_USER" != "$(id -un)" ]]; then
  $SUDO -u "$RUN_USER" tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true
  echo "[SETUP] launching training in tmux session '$SESSION_NAME' as $RUN_USER..."
  $SUDO -u "$RUN_USER" tmux new-session -d -s "$SESSION_NAME" "bash -lc '$TRAIN_CMD'"
else
  tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true
  echo "[SETUP] launching training in tmux session '$SESSION_NAME'..."
  tmux new-session -d -s "$SESSION_NAME" "bash -lc '$TRAIN_CMD'"
fi

echo "[SETUP] done."
echo "  attach:   tmux attach -t $SESSION_NAME"
echo "  log tail: tail -f $LOG_FILE"
