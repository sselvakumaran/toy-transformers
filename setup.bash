#!/bin/bash
set -euo pipefail

CONFIG_FILE="${1:?usage: ./setup.sh <config_file> [branch]}"
BRANCH="${2:-main}"
REPO="https://github.com/sselvakumaran/toy-transformers.git"
REPO_DIR="toy-transformers"

source .env

aws configure set aws_access_key_id "$AWS_ACCESS_KEY_ID"
aws configure set aws_secret_access_key "$AWS_SECRET_ACCESS_KEY"
aws configure set region "${AWS_REGION:-us-east-1}"

git clone --single-branch --branch "$BRANCH" "$REPO_URL"
cd "$REPO_DIR"

python3 -m venv venv
source venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

python3 -m toy_transformers.train "$CONFIG_FILE" "$BUCKET"
