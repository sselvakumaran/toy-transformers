"""
Prepare pes2o-2B and wiki-2B from allenai/dolmino-mix-1124.

The HF auto-converted parquet API only exposes a < 5GB partial-train slice
for this dataset, so we pull the original .json.gz files from the repo's
data/{pes2o,wiki}/ folders, convert text-only to parquet, then reuse
prepare_dataset's tokenize + shuffle pipeline. After tokenization we trim
encoded shards to >=2B tokens. Subsets get fully separate datasets
(separate shuffles, separate metadata).

Subset reference (from the HF dataset card):
  pes2o: 58.6B tokens / 38.8M docs across 26 .json.gz files (~106GB compressed)
  wiki:   3.7B tokens /  6.17M docs across 2 .json.gz files  (~6.5GB compressed)
First file alone is ~4.3GB compressed → ~2.4B tokens, enough for the 2B target.

Usage:
  PYTHONPATH=. python experiments/dolmino-load/prepare.py --subset wiki
  PYTHONPATH=. python experiments/dolmino-load/prepare.py --subset pes2o
  PYTHONPATH=. python experiments/dolmino-load/prepare.py --subset both --verify
  PYTHONPATH=. python experiments/dolmino-load/prepare.py --subset both --s3_remote s3://my-bucket/toy-transformers
"""
import argparse
import gzip
import json
from pathlib import Path
from typing import Optional

import pyarrow as pa
import pyarrow.parquet as pq
import requests
from tqdm import tqdm

from toy_transformers.prepare_dataset import (
	DATA_DIR,
	load_status,
	mark_phase_done,
	phase_done,
	run_shuffle,
	run_tokenize,
	run_upload,
	run_verify,
)

DATASET_REPO = "allenai/dolmino-mix-1124"
TARGET_TOKENS = 2_000_000_000
SHARD_SIZE = 128_000_000  # tokens per encoded shard → ~256MB on disk (uint16)

SUBSETS = {
	"pes2o": {
		"name": "pes2o-2B",
		"file": "data/pes2o/pes2o-0000.json.gz",
	},
	"wiki": {
		"name": "wiki-2B",
		"file": "data/wiki/wiki-0000.json.gz",
	},
}


def file_url(path: str) -> str:
	return f"https://huggingface.co/datasets/{DATASET_REPO}/resolve/main/{path}"


def stream_jsonl_gz_to_parquet(url: str, dst: Path, batch_rows: int = 8_000):
	"""Stream a single .json.gz from HF, write text-only parquet."""
	if dst.exists():
		print(f"[STREAM] {dst.name} already exists, skipping")
		return

	dst.parent.mkdir(parents=True, exist_ok=True)
	tmp = dst.with_suffix(".tmp.parquet")
	schema = pa.schema([("text", pa.string())])
	writer = pq.ParquetWriter(tmp, schema, compression="zstd")

	rows_written = 0
	bytes_seen = 0
	batch: list[str] = []

	def flush():
		nonlocal rows_written
		if not batch:
			return
		writer.write_table(pa.table({"text": pa.array(batch, type=pa.string())}))
		rows_written += len(batch)
		batch.clear()

	print(f"[STREAM] {url}")
	pbar = tqdm(unit="B", unit_scale=True, desc=dst.name)
	try:
		with requests.get(url, stream=True, headers={"User-Agent": "python"}) as r:
			r.raise_for_status()
			with gzip.GzipFile(fileobj=r.raw) as gz:
				for raw_line in gz:
					if not raw_line.strip():
						continue
					try:
						obj = json.loads(raw_line)
					except json.JSONDecodeError:
						continue
					text = obj.get("text")
					if not text:
						continue
					batch.append(text)
					n = len(text.encode("utf-8"))
					bytes_seen += n
					pbar.update(n)
					if len(batch) >= batch_rows:
						flush()
		flush()
	finally:
		writer.close()
		pbar.close()

	tmp.rename(dst)
	print(f"[STREAM] wrote {rows_written:,} docs, {bytes_seen:,} text bytes -> {dst}")


def trim_encoded_to_target(encoded_dir: Path, target_tokens: int):
	"""Keep enough leading shards to cover >= target_tokens; rewrite metadata."""
	meta_path = encoded_dir / "metadata.json"
	with open(meta_path) as f:
		meta = json.load(f)

	counts = meta["token_counts"]
	names = sorted(counts.keys())

	cumulative = 0
	keep: list[str] = []
	for n in names:
		keep.append(n)
		cumulative += counts[n]
		if cumulative >= target_tokens:
			break

	for n in names:
		if n not in keep:
			(encoded_dir / n).unlink(missing_ok=True)

	meta["token_counts"] = {n: counts[n] for n in keep}
	meta["num_shards"] = len(keep)
	with open(meta_path, "w") as f:
		json.dump(meta, f, indent=2)

	total = sum(meta["token_counts"].values())
	print(f"[TRIM] kept {len(keep)}/{len(names)} shards, {total:,} tokens (target {target_tokens:,})")


def prepare_subset(
	subset_key: str,
	vocab_path: Path,
	bos_token: str,
	special_tokens: list[str],
	target_tokens: int,
	force: bool,
	verify: bool,
	s3_remote: Optional[str],
	val_shards: int,
):
	cfg = SUBSETS[subset_key]
	name = cfg["name"]
	dataset_dir = DATA_DIR / "datasets" / name
	dataset_dir.mkdir(parents=True, exist_ok=True)
	raw_dir = dataset_dir / "_raw"
	encoded_dir = dataset_dir / "_encoded"
	shuffled_dir = dataset_dir

	status = {} if force else load_status(dataset_dir)

	if not phase_done(status, "download"):
		stream_jsonl_gz_to_parquet(file_url(cfg["file"]), raw_dir / "0000.parquet")
		mark_phase_done(dataset_dir, status, "download")
	else:
		print("[DOWNLOAD] skipping (already complete)")

	run_tokenize(
		raw_dir=raw_dir,
		encoded_dir=encoded_dir,
		vocab_path=vocab_path,
		vocab_size=32768,
		special_tokens=special_tokens,
		bos_token=bos_token,
		status=status,
		dataset_dir=dataset_dir,
		shard_size=SHARD_SIZE,
	)

	if not phase_done(status, "trim"):
		trim_encoded_to_target(encoded_dir, target_tokens)
		mark_phase_done(dataset_dir, status, "trim")
	else:
		print("[TRIM] skipping (already complete)")

	run_shuffle(
		encoded_dir=encoded_dir,
		shuffled_dir=shuffled_dir,
		status=status,
		dataset_dir=dataset_dir,
		val_shards=val_shards,
	)

	if verify:
		run_verify(shuffled_dir=shuffled_dir, vocab_path=vocab_path, bos_token=bos_token)

	if s3_remote:
		run_upload(
			dataset_dir=dataset_dir,
			name=name,
			vocab_path=vocab_path,
			s3_remote=s3_remote,
		)

	print(f"[STATUS] {name} ready at {shuffled_dir}")


def main():
	parser = argparse.ArgumentParser(description="prepare pes2o-2B / wiki-2B from dolmino-mix-1124")
	parser.add_argument("--subset", choices=["pes2o", "wiki", "both"], default="both")
	parser.add_argument("--target_tokens", type=int, default=TARGET_TOKENS)
	parser.add_argument("--vocab_path", default=str(DATA_DIR / "vocabs/vocab_fineweb_32k.json"))
	parser.add_argument("--bos_token", default="<BOS>")
	parser.add_argument("--special_tokens", nargs="+", default=["<BOS>", "<PAD>"])
	parser.add_argument("--val_shards", type=int, default=1)
	parser.add_argument("--force", action="store_true")
	parser.add_argument("--verify", action="store_true")
	parser.add_argument("--s3_remote", default=None)
	args = parser.parse_args()

	keys = [args.subset] if args.subset != "both" else ["pes2o", "wiki"]
	for k in keys:
		print(f"\n========== preparing {SUBSETS[k]['name']} ==========\n")
		prepare_subset(
			subset_key=k,
			vocab_path=Path(args.vocab_path),
			bos_token=args.bos_token,
			special_tokens=args.special_tokens,
			target_tokens=args.target_tokens,
			force=args.force,
			verify=args.verify,
			s3_remote=args.s3_remote,
			val_shards=args.val_shards,
		)


if __name__ == "__main__":
	main()
