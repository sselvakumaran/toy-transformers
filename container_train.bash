#!/bin/bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
	echo "usage: ./container_train <ip> <branch> <config>"
	echo "  e.g. ./container_train 123.45.67.89 main configs/gptv4.json"
	exit 1
fi

IP="$1"
BRANCH="$2"
CONFIG="$3"
ENV_FILE=".env"
[[ -f "$ENV_FILE" ]] || { echo "error: .env not found"; exit 1; }
source "$ENV_FILE"
: "${SSH_PORT:?missing SSH_PORT in .env}"
: "${BUCKET:?missing BUCKET in .env}"

echo "[local] uploading .env to $IP:$SSH_PORT..."
scp -P "$SSH_PORT" "$ENV_FILE" "root@${IP}:/root/.env"

echo "[local] launching: branch=$BRANCH config=$CONFIG"
ssh -p "$SSH_PORT" "root@${IP}" -t \
	"tmux kill-session -t train 2>/dev/null; \
	tmux new-session -d -s train ' \
		source /root/.env && \
		cd /workspace/toy-transformers && \
		git fetch --all && \
		git checkout $BRANCH && \
		git pull origin $BRANCH && \
		python -m toy_transformers.train $CONFIG \$BUCKET \
	' && tmux attach -t train"