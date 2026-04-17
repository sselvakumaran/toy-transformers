"""
Download + reformat smol-smoltalk into ChatML tokenized shards.

Dataset: HuggingFaceTB/smol-smoltalk / train
Format: "messages" column = list of {"role": str, "content": str} dicts.
  Converted to ChatML: <|im_start|>role\ncontent<|im_end|>

Assumes vocab already has <|im_start|> and <|im_end|> as special tokens.

Usage:
  python experiments/preprocessing/smol-smoltalk/prepare.py \
    --vocab_path data/vocabs/your_vocab.json

  python experiments/preprocessing/smol-smoltalk/prepare.py \
    --vocab_path data/vocabs/your_vocab.json \
    --s3_remote s3://bucket/toy-transformers --skip_existing
"""
import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq
from tqdm import tqdm

from toy_transformers.data import S3Sync
from toy_transformers.tokenization import Vocabulary, bulk_encode, shuffle_shards
from toy_transformers.prepare_dataset import (
	get_shard_urls, download_shard,
	load_status, save_status, mark_phase_done, phase_done,
)

REPO_ROOT = Path(__file__).parent.parent.parent.parent
DATA_DIR = REPO_ROOT / "data"

DATASET_ID = "HuggingFaceTB/smol-smoltalk"
SUBSET = "default"
SPLIT = "train"
NAME = "smol-smoltalk"

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
BOS = "<BOS>"

def messages_to_chatml(messages: list[dict]) -> str:
	parts = []
	for msg in messages:
		role = msg["role"]
		content = msg["content"]
		parts.append(f"{IM_START}{role}\n{content}{IM_END}")
	return "\n".join(parts)


def stream_chatml(raw_dir: Path, max_shards: int | None = None):
	"""Yield BOS-joined ChatML documents as encoded bytes, one chunk per parquet file."""
	parquet_files = sorted(raw_dir.glob("*.parquet"))
	if max_shards is not None:
		parquet_files = parquet_files[:max_shards]

	for pf_path in parquet_files:
		table = pq.read_table(pf_path)
		docs = []
		for i in range(len(table)):
			messages = table["messages"][i].as_py()
			if not messages:
				continue
			docs.append(messages_to_chatml(messages))

		if docs:
			yield (BOS + BOS.join(docs)).encode("utf-8")


def run_download_shards(raw_dir: Path, dataset_dir: Path, status: dict, max_shards: int | None = None):
	if phase_done(status, "download"):
		print("[DOWNLOAD]", "skipping download")
		return

	print("[DOWNLOAD]", f"{DATASET_ID} - {SUBSET} - {SPLIT}")
	shard_urls = get_shard_urls(DATASET_ID, SUBSET, SPLIT)

	if max_shards is not None:
		shard_urls = shard_urls[:max_shards]

	status["shard_urls"] = shard_urls
	status["num_raw_shards"] = len(shard_urls)
	save_status(dataset_dir, status)

	raw_dir.mkdir(parents=True, exist_ok=True)
	for i, url in tqdm(enumerate(shard_urls), total=len(shard_urls), desc="downloading"):
		dst = raw_dir / f"{i:04d}.parquet"
		download_shard(url, dst)

	mark_phase_done(dataset_dir, status, "download")
	print("[DOWNLOAD]", f"complete ({len(shard_urls)} shards)")


def main():
	parser = argparse.ArgumentParser(description="prepare smol-smoltalk as ChatML tokenized shards")
	parser.add_argument("--vocab_path", type=str, required=True,
		help="path to vocab JSON (must include <|im_start|> and <|im_end|>)")
	parser.add_argument("--max_shards", type=int, default=None,
		help="limit number of parquet shards to process")
	parser.add_argument("--shard_size", type=int, default=128_000_000,
		help="tokens per encoded shard")
	parser.add_argument("--n_output_shards", type=int, default=None)
	parser.add_argument("--val_shards", type=int, default=1)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--s3_remote", type=str, default=None)
	parser.add_argument("--skip_existing", action="store_true")
	parser.add_argument("--force", action="store_true")
	args = parser.parse_args()

	vocab_path = Path(args.vocab_path)
	if not vocab_path.is_absolute():
		vocab_path = REPO_ROOT / vocab_path
	vocab = Vocabulary.load(vocab_path)

	# sanity check special tokens
	for tok in [IM_START, IM_END, BOS]:
		assert tok.encode("utf-8") in vocab.token_to_idx, f"missing special token: {tok}"

	dataset_dir = DATA_DIR / "datasets" / NAME
	dataset_dir.mkdir(parents=True, exist_ok=True)
	raw_dir = dataset_dir / "_raw"
	encoded_dir = dataset_dir / "_encoded"
	shuffled_dir = dataset_dir

	status = {} if args.force else load_status(dataset_dir)

	# download
	run_download_shards(raw_dir, dataset_dir, status, args.max_shards)

	# encode
	if not phase_done(status, "tokenize"):
		print("[TOKENIZE]", "encoding ChatML documents")
		encoded_dir.mkdir(parents=True, exist_ok=True)
		bulk_encode(
			doc_iter=stream_chatml(raw_dir, args.max_shards),
			vocab=vocab,
			vocab_path=vocab_path,
			output_dir=encoded_dir,
			split_token=BOS,
			shard_size=args.shard_size,
		)
		mark_phase_done(dataset_dir, status, "tokenize")
		print("[TOKENIZE]", "complete")
	else:
		print("[TOKENIZE]", "skipping")

	# shuffle
	if not phase_done(status, "shuffle"):
		with open(encoded_dir / "metadata.json") as f:
			enc_meta = json.load(f)
		n_out = args.n_output_shards or enc_meta["num_shards"]
		print("[SHUFFLE]", f"{enc_meta['num_shards']} -> {n_out} shards")
		shuffle_shards(
			input_dir=encoded_dir,
			output_dir=shuffled_dir,
			seed=args.seed,
			n_output_shards=n_out,
			val_shards=args.val_shards,
		)
		mark_phase_done(dataset_dir, status, "shuffle")
		print("[SHUFFLE]", "complete")
	else:
		print("[SHUFFLE]", "skipping")

	# upload
	if args.s3_remote:
		sync = S3Sync(remote_base=args.s3_remote, local_root=REPO_ROOT)
		print("[UPLOAD]", f"uploading to {args.s3_remote}")

		upload_exts = {".bin", ".json"}
		for path in tqdm(sorted(p for p in shuffled_dir.iterdir() if p.suffix in upload_exts), desc="uploading"):
			rel = path.relative_to(REPO_ROOT)
			sync.push(rel, skip_existing=args.skip_existing)

		vocab_rel = vocab_path.relative_to(REPO_ROOT)
		sync.push(vocab_rel, skip_existing=args.skip_existing)
		print("[UPLOAD]", "done")

	print("[STATUS]", f"dataset ready at {shuffled_dir}")

if __name__ == "__main__":
	main()
