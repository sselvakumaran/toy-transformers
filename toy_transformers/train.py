import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from toy_transformers.config import TrainingConfig
from toy_transformers.data import S3Sync, S3ShardDownloader, ShardedTokenDataset
from toy_transformers.model_io import MetricsWriter, RunStatus, loadl_model, save_model


# --- constants ---
REPO_ROOT = Path(__file__).parent.parent
RUNS_DIR = REPO_ROOT / "runs"
DATA_DIR = REPO_ROOT / "data"


# --- helpers ---


def train_from_config(cfg: Path, bucket: str, device: str = "cuda"):

	pass

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("config", help="path to config JSON")
	parser.add_argument("bucket", "S3 bucket/base name, can include subdirectories")
	parser.add_argument("--device", default="cuda")
	args = parser.parse_args()
	
	cfg = Path(args.config)
	assert cfg.exists()

	train_from_config(cfg=cfg, bucket=args.bucket, device=args.device)