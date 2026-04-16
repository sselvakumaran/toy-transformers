#!/bin/bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
	echo "usage: ./run_train_remote.bash <ip> <branch> <config> [config...]"
	echo "  e.g. ./run_train_remote.bash 123.45.67.89 main configs/gptv4.json configs/gptv5.json"
	exit 1
fi

IP="$1"; shift
BRANCH="$1"; shift
CONFIGS=("$@")
ENV_FILE=".env"
SETUP_FILE="setup.bash"

[[ -f "$ENV_FILE"   ]] || { echo "error: .env not found";       exit 1; }
[[ -f "$SETUP_FILE" ]] || { echo "error: setup.bash not found"; exit 1; }

source "$ENV_FILE"
: "${SSH_PORT:?missing SSH_PORT in .env}"
: "${BUCKET:?missing BUCKET in .env}"
SSH_USER="${SSH_USER:-ubuntu}"
SESSION="train"

SSH="ssh -p $SSH_PORT -o StrictHostKeyChecking=accept-new ${SSH_USER}@${IP}"
SCP="scp -P $SSH_PORT -o StrictHostKeyChecking=accept-new"

echo "[local] uploading .env + setup.bash to ${SSH_USER}@${IP}:${SSH_PORT}..."
$SCP "$ENV_FILE" "$SETUP_FILE" "${SSH_USER}@${IP}:~/"
$SSH "chmod +x ~/setup.bash"

echo "[local] launching: branch=$BRANCH configs=${CONFIGS[*]} (session=$SESSION)"
# setup.bash spawns its own detached tmux session — don't wrap it in one here
# (that used to collide on the $SESSION name and kill itself mid-run).
$SSH "cd ~ && bash ./setup.bash $BRANCH ${CONFIGS[*]}"
echo "[local] setup finished. attaching to tmux session '$SESSION' (detach: Ctrl-b d)..."
exec ssh -p "$SSH_PORT" -o StrictHostKeyChecking=accept-new -t "${SSH_USER}@${IP}" "tmux attach -t $SESSION"
