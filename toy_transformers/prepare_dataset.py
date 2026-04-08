import argparse
import json
import requests
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
		return
	
	print("[DOWNLOAD]", f"{dataset_id} - {subset} - {split}")
	shard_urls = get_shard_urls(dataset_id, subset, split)

	if shard_indices:
		shard_indices = [i for i in shard_indices if i < len(shard_urls)]
	selected = [(i, shard_urls[i]) for i in shard_indices] if shard_indices else list(enumerate(shard_urls))

	status["shard_urls"] = [url for _, url in selected]
	status["num_raw_shards"] = len(selected)
	save_status(dataset_dir, status)

	raw_dir.mkdir(parents=True, exist_ok=True)
	for i, url in tqdm(selected, desc="downloading shards"):
		dst = raw_dir / f"{i:04d}.parquet"
		download_shard(url, dst)
	
	print("[DOWNLOAD]", "complete")
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

	raw_files = sorted(raw_dir.glob("*.parquet"))

	if not vocab_path.exists():
		print("[TOKENIZE]", f"training vocab (size={vocab_size})...")

		vocab_files = raw_files[:vocab_train_shards]

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
		print("[TOKENIZE]", f"loading existing vocab from {vocab_path}")
	
	print("[TOKENIZE]", "encoding")
	encoded_dir.mkdir(parents=True, exist_ok=True)
	bulk_encode(
		doc_iter=stream_texts(raw_files, bos_token, score_column, min_score),
		vocab=vocab,
		vocab_path=vocab_path,
		output_dir=encoded_dir,
		split_token=bos_token,
		shard_size=shard_size,
	)

	print("[TOKENIZE]", "complete")
	mark_phase_done(dataset_dir, status, "tokenize")

def run_shuffle(
	encoded_dir: Path,
	shuffled_dir: Path,
	status: dict,
	dataset_dir: Path,
	n_output_shards: Optional[int] = None,
	val_shards: int = 1,
	seed: int = 42
):
	if phase_done(status, "shuffle"):
		print("[SHUFFLE]", "already complete, skipping")
		return

	with open(encoded_dir / "metadata.json") as f:
		old_meta = json.load(f)

	n_encoded = old_meta["num_shards"]
	n_out = n_output_shards or n_encoded
	shuffled_dir.mkdir(parents=True, exist_ok=True)
	shuffle_shards(
		input_dir=encoded_dir,
		output_dir=shuffled_dir,
		seed=seed,
		n_output_shards=n_out,
		val_shards=val_shards,
	)

	print("[SHUFFLE]", "complete")
	mark_phase_done(dataset_dir, status, "shuffle")

def run_verify(shuffled_dir: Path, vocab_path: Path, bos_token: str):
	print("[VERIFY]", "running checks...")

	vocab = Vocabulary.load(vocab_path)
	bos_key = bos_token.encode('utf-8') if vocab.config.mode == TokenizationMode.BYTES else bos_token
	bos_id = vocab.token_to_idx[bos_key]
	with open(shuffled_dir / "metadata.json") as f:
		meta = json.load(f)

	for shard_name in list(meta["token_counts"].keys())[:3]:
		path = shuffled_dir / shard_name
		raw = np.frombuffer(path.read_bytes(), dtype=np.uint16)
		bos_count = int((raw == bos_id).sum())
		expected_tokens = meta["token_counts"][shard_name]
		print("[VERIFY]", f"{shard_name}: {len(raw):,} tokens (expected {expected_tokens:,}), {bos_count:,} docs")
		decoded = vocab.decode(raw[:200].tolist())
		text = b"".join(decoded).decode("utf-8", errors="replace")
		print("[VERIFY]", f"sample: {text[:120]!r}")

	print("[VERIFY]", "done")

def run_upload(dataset_dir: Path, name: str, vocab_path: Path, s3_remote: str):
	sync = S3Sync(remote_base=s3_remote, local_root=REPO_ROOT)

	print("[UPLOAD]", f"uploading shuffled shards to {s3_remote}/data/datasets/{name}/")

	upload_exts = {".bin", ".json"}
	for path in tqdm(sorted(p for p in dataset_dir.iterdir() if p.suffix in upload_exts), desc="uploading"):
		rel = path.relative_to(REPO_ROOT)
		sync.push(rel)

	# upload vocab
	vocab_rel = vocab_path.relative_to(REPO_ROOT)
	print("[UPLOAD]", f"pushing vocab: {vocab_rel}")
	sync.push(vocab_rel)

	print("[UPLOAD]", "done")

def main():
	parser = argparse.ArgumentParser(description="prepare a (huggingface) dataset for training")

	parser.add_argument("dataset_id", help="HuggingFace dataset ID, e.g. HuggingFaceFW/fineweb-edu")
	parser.add_argument("name", help="local name for this dataset, e.g. fineweb-edu-10BT")
	parser.add_argument("--subset", default="default", help="hf dataset subset/config name")
	parser.add_argument("--split", default="train", help="hf dataset split")
	parser.add_argument("--shard_indices", nargs="+", type=int, default=None,
    help="subset of shard indices to download (default: all)")
	parser.add_argument("--max_shards", type=int, default=None,
    help="download at most N parquet shards from HF (first N)")
	
	parser.add_argument("--vocab_size", type=int, default=32768)
	parser.add_argument("--special_tokens", nargs="+", default=["<BOS>", "<PAD>"],
    help="special tokens in priority order (first = BOS)")
	parser.add_argument("--bos_token", default="<BOS>")
	parser.add_argument("--vocab_path", type=str, default=None,
		help="path to existing or target vocab JSON (default: data/vocabs/<name>/vocab.json)")
	parser.add_argument("--vocab_train_shards", type=int, default=None,
		help="number of raw shards to use for vocab training (default: all)")
	parser.add_argument("--shard_size", type=int, default=128_000_000,
		help="max tokens per encoded shard")
	
	# shuffle
	parser.add_argument("--n_output_shards", type=int, default=None,
		help="number of output shards after shuffle (default: same as encoded)")
	parser.add_argument("--val_shards", type=int, default=1,
		help="number of shards to reserve for validation")
	parser.add_argument("--seed", type=int, default=42)

	# filtering
	parser.add_argument("--min_score", nargs=2, metavar=("COLUMN", "THRESHOLD"), default=None,
		help="filter rows by score column, e.g. --min_score score 3.0")

	# phases
	parser.add_argument("--skip_download", action="store_true")
	parser.add_argument("--skip_tokenize", action="store_true")
	parser.add_argument("--skip_shuffle", action="store_true")
	parser.add_argument("--verify", action="store_true", help="run checks after shuffle")
	parser.add_argument("--force", action="store_true", help="re-run all phases even if marked complete")

	# upload
	parser.add_argument("--s3_remote", type=str, default=None,
		help="S3 remote base, e.g. s3://my-bucket/toy-transformers")

	args = parser.parse_args()

	dataset_dir = DATA_DIR / "datasets" / args.name
	dataset_dir.mkdir(parents=True, exist_ok=True)
	raw_dir = dataset_dir / "_raw"
	encoded_dir = dataset_dir / "_encoded"
	shuffled_dir = dataset_dir
	vocab_path = Path(args.vocab_path) if args.vocab_path \
		else DATA_DIR / f"vocabs/vocab_{args.name}_{args.vocab_size}.json"
	
	score_column, min_score = None, None
	if args.min_score:
		score_column = args.min_score[0]
		min_score = float(args.min_score[1])

	status = {} if args.force else load_status(dataset_dir)

	shard_indices = args.shard_indices
	if shard_indices is None and args.max_shards is not None:
		shard_indices = list(range(args.max_shards))

	if not args.skip_download: run_download(
		args.dataset_id, args.subset, args.split,
		raw_dir, dataset_dir, status,
		shard_indices
	)
	
	if not args.skip_tokenize: run_tokenize(
		raw_dir=raw_dir, 
		encoded_dir=encoded_dir, 
		vocab_path=vocab_path, 
		vocab_size=args.vocab_size, 
		special_tokens=args.special_tokens, 
		bos_token=args.bos_token, 
		status=status, 
		dataset_dir=dataset_dir, 
		vocab_train_shards=args.vocab_train_shards, 
		score_column=score_column, 
		min_score=min_score, 
		shard_size=args.shard_size
	)
	
	if not args.skip_shuffle: run_shuffle(
		encoded_dir=encoded_dir,
		shuffled_dir=shuffled_dir,
		n_output_shards=args.n_output_shards,
		val_shards=args.val_shards,
		status=status,
		dataset_dir=dataset_dir,
		seed=args.seed
	)
	
	if args.verify: run_verify(
		shuffled_dir=shuffled_dir,
		vocab_path=vocab_path,
		bos_token=args.bos_token
	)
		
	if args.s3_remote: run_upload(
			dataset_dir=dataset_dir,
			name=args.name,
			vocab_path=vocab_path,
			s3_remote=args.s3_remote
		)
	
	print("[STATUS]", f"dataset ready at {shuffled_dir}")

if __name__ == "__main__":
	main()