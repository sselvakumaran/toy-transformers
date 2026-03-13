import argparse
import json
import os
import requests
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pyarrow.parquet as pq
import pyarrow
from tqdm import tqdm

from toy_transformers.tokenization import Vocabulary, TokenizationMode, create_bpe, bulk_encode, shuffle_shards
from toy_transformers.data import S3Sync

REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / "data"

BATCH_SIZE_BYTES = 200 * 1024 * 1024

def load_status(dataset_dir: Path) -> dict:
	status_path = dataset_dir / "status.json"
	if status_path.exists(): return json.loads(status_path.read_text())
	return {"phases_complete": []}

def save_status(dataset_dir: Path, status: dict):
	status_path = dataset_dir / "status.json"
	status_path.write_text(json.dumps(status, indent=2))

def phase_done(status: dict, phase: str) -> bool:
	return phase in status.get("phases_complete", [])

def mark_phase_done(dataset_dir: Path, status: dict, phase: str):
	if "phases_complete" not in status: status["phases_complete"] = []

	if phase not in status["phases_complete"]:
		status["phases_complete"].append(phase)
	save_status(dataset_dir, status)


def get_shard_urls(dataset_id: str, subset: str, split: str) -> list[str]:
	url = f"https://huggingface.co/api/datasets/{dataset_id}/parquet/{subset}/{split}"
	r = requests.get(url)
	r.raise_for_status()
	return r.json()

def download_shard(url: str, dst: Path):
	if dst.exists(): return

	tmp = dst.with_suffix(".tmp")
	try:
		with requests.get(url, stream=True, headers={"User-Agent": "python"}) as r:
			r.raise_for_status()

			total = int(r.headers.get("Content-Length", 0))
			with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dst.name) as bar:
				for chunk in r.iter_content(chunk_size=1024*1024):
					if chunk:
						f.write(chunk)
						bar.update(len(chunk))
		tmp.rename(dst)
		
	except Exception as e:
		if tmp.exists():
			tmp.unlink()
		raise e

def run_download(
	dataset_id: str, subset: str, split: str,
	raw_dir: Path, dataset_dir: Path, status: dict, 
	shard_indices: Optional[list[int]] = None
):
	if phase_done(status, "download"):
		print("[DOWNLOAD]", "skipping download")
	
	print("[DOWNLOAD]", f"{dataset_id} - {subset} - {split}")
	shard_urls = get_shard_urls(dataset_id, subset, split)

	selected = [(i, shard_urls[i]) for i in shard_indices] if shard_indices else list(enumerate(shard_urls))

	status["shard_urls"] = [url for _, url in selected]
	status["num_raw_shards"] = len(selected)
	save_status(dataset_dir, status)

	raw_dir.mkdir(parents=True, exist_ok=True)
	for i, url in tqdm(selected, desc="downloading shards"):
		dst = raw_dir / f"{i:04d}.parquet"
		download_shard(url, dst)
	
	mark_phase_done(dataset_dir, status, "download")

def stream_raw_ds(
	files: list[Path], columns: list[str], 
	batch_size_bytes: int = BATCH_SIZE_BYTES, 
	score_column: Optional[str] = None, min_score: Optional[float] = None
):
	parquet_files = sorted(files)
	filter_docs = score_column and min_score
	cols = list(columns)
	if filter_docs: assert score_column in cols

	batch_tables, batch_bytes = [], 0
	for pf_path in parquet_files:
		pf = pq.ParquetFile(pf_path)
		for rg in range(pf.metadata.num_row_groups):
			table = pf.read_row_group(rg, columns=cols)
			if filter_docs:
				mask = pyarrow.compute.greater_equal(table[score_column], min_score)
				table = table.filter(mask)
			batch_tables.append(table)
			batch_bytes += table.nbytes

			if batch_bytes >= batch_size_bytes:
				yield pyarrow.concat_tables(batch_tables)
				batch_tables, batch_bytes = [], 0
	if batch_tables:
		yield pyarrow.concat_tables(batch_tables)

def stream_texts(
	files: list[Path], bos_token: str, 
	score_column: Optional[str] = None, min_score: Optional[float] = None
):
	for batch in stream_raw_ds(files, columns=["text"], 
		score_column=score_column, min_score=min_score
	):
		yield (bos_token + bos_token.join(batch["text"].to_pylist())).encode('utf-8')

def run_tokenize(
	raw_dir: Path,
	encoded_dir: Path, 
	vocab_path: Path,
	vocab_size: int,
	special_tokens: list[str],
	bos_token: str,
	status: dict,
	dataset_dir: Path,
	vocab_train_shards: Optional[int] = None,
	score_column: Optional[str] = None,
	min_score: Optional[float] = None,
	shard_size: int = 128_000_000
):
	if phase_done(status, "tokenize"):
		print("[TOKENIZE]", "skipping tokenization")
		return

	if not vocab_path.exists():
		print("[TOKENIZE]", f"training vocab (size={vocab_size})...")

		raw_files = sorted(raw_dir.glob("*.parquet"))
		vocab_files = [raw_files[idx] for idx in vocab_train_shards] if vocab_train_shards else raw_files

		vocab = create_bpe(
			data_iter=stream_texts(vocab_files, bos_token, score_column, min_score),
			vocab_size=vocab_size,
			mode=TokenizationMode.BYTES,
			special_tokens=special_tokens
		)

		vocab_path.parent.mkdir(parents=True, exist_ok=True)
		vocab.save(vocab_path)
		print("[TOKENIZE]", f"vocab saved to {vocab_path}")
	else:
		vocab = Vocabulary.load(vocab_path)
		print("[TOKENIZE]", "loading existing vocab from {vocab_path}")
	
	print("[TOKENIZE]", "encoding")
	encoded_dir.mkdir(parents=True, exist_ok=True)
	bulk_encode(
		doc_iter=stream_texts(raw_dir, bos_token, score_column, min_score),
		vocab=vocab,
		vocab_path=vocab_path,
		output_dir=encoded_dir,
		split_token=bos_token,
		shard_size=shard_size,
	)

	mark_phase_done(dataset_dir, status, "tokenize")

		