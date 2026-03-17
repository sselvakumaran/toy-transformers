import argparse
import json
import requests
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from tqdm import tqdm

from toy_transformers.tokenization import Vocabulary
from toy_transformers.data import S3Sync
from toy_transformers.model_io import load_model

REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / "data"


# --- download helpers (same as prepare_dataset.py) ---

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
		if tmp.exists(): tmp.unlink()
		raise e


# --- download, reformat, upload ---

def run_download(dataset_id: str, subset: str, split: str, raw_dir: Path):
	print("[DOWNLOAD]", f"{dataset_id} - {subset} - {split}")
	shard_urls = get_shard_urls(dataset_id, subset, split)

	raw_dir.mkdir(parents=True, exist_ok=True)
	for i, url in enumerate(tqdm(shard_urls, desc="downloading shards")):
		dst = raw_dir / f"{i:04d}.parquet"
		download_shard(url, dst)
	
	print("[DOWNLOAD]", f"complete ({len(shard_urls)} shards)")

def run_reformat(
	raw_dir: Path, out_path: Path,
	question_col: str, answer_col: str, label_col: str
):
	"""Read raw parquets and write a single normalized parquet:
	  question (str), answers (list[str]), label (int)
	"""
	print("[REFORMAT]", f"question={question_col}, answers={answer_col}, label={label_col}")
	
	questions, answers, labels = [], [], []
	skipped = 0

	for pf_path in sorted(raw_dir.glob("*.parquet")):
		table = pq.read_table(pf_path)
		for i in range(len(table)):
			q = table[question_col][i].as_py()
			a = table[answer_col][i].as_py()
			l = table[label_col][i].as_py()

			if l is None or l == "":
				skipped += 1
				continue
			
			questions.append(str(q))
			answers.append([str(x) for x in a])
			labels.append(int(l))
	
	out_table = pa.table({
		"question": pa.array(questions, type=pa.string()),
		"answers": pa.array(answers, type=pa.list_(pa.string())),
		"label": pa.array(labels, type=pa.int32()),
	})

	out_path.parent.mkdir(parents=True, exist_ok=True)
	pq.write_table(out_table, out_path)
	print("[REFORMAT]", f"{len(labels)} examples written, {skipped} skipped -> {out_path}")

def run_upload(eval_dir: Path, s3_remote: str):
	sync = S3Sync(remote_base=s3_remote, local_root=DATA_DIR.parent)

	print("[UPLOAD]", "uploading eval data to S3")
	for path in sorted(eval_dir.iterdir()):
		if path.is_file():
			rel = path.relative_to(DATA_DIR.parent)
			sync.push(rel)
	print("[UPLOAD]", "done")


def eval(
	model: torch.nn.Module,
	vocab: Vocabulary,
	eval_ds: Path,
	block_size: int,
	device: str = "cuda",
	limit: int = 0,
	batch_size: int = 1,
) -> dict:
	table = pq.read_table(eval_ds)
	n = len(table) if limit <= 0 else min(limit, len(table))

	examples = []
	for i in range(n):
		question = table["question"][i].as_py()
		answers = table["answers"][i].as_py()
		label = table["label"][i].as_py()

		q_tokens = vocab.encode(question.encode("utf-8"))

		candidate_tokens = []
		answer_lens = []
		for ans in answers:
			a_tokens = vocab.encode((" " + ans).encode("utf-8"))
			answer_lens.append(len(a_tokens))
			candidate_tokens.append(q_tokens + a_tokens)
		
		# truncate from left if exceeding block_size (preserves answer tokens)
		for j in range(len(candidate_tokens)):
			if len(candidate_tokens[j]) > block_size:
				candidate_tokens[j] = candidate_tokens[j][-block_size:]

		examples.append((candidate_tokens, answer_lens, label))

	print("[EVALUATE]", f"{len(examples)} examples, batch_size={batch_size}")

	num_correct, num_correct_norm, num_total = 0, 0, 0
	for batch_start in tqdm(range(0, len(examples), batch_size), desc="evaluating"):
		batch = examples[batch_start:batch_start + batch_size]

		all_seqs = []
		all_answer_lens = []
		labels = []
		candidates_per_example = []

		for candidate_tokens, answer_lens, label in batch:
			candidates_per_example.append(len(candidate_tokens))
			labels.append(label)
			for j, toks in enumerate(candidate_tokens):
				all_seqs.append(toks)
				all_answer_lens.append(answer_lens[j])
		
		max_len = max(len(s) for s in all_seqs)
		total_seqs = len(all_seqs)

		tokens = torch.zeros(total_seqs, max_len, dtype=torch.long)
		mask = torch.zeros(total_seqs, max_len, dtype=torch.long)
		for j, toks in enumerate(all_seqs):
			L = len(toks)
			tokens[j, :L] = torch.tensor(toks, dtype=torch.long)
			mask[j, L - all_answer_lens[j]:L] = 1

		tokens = tokens.to(device)
		mask = mask.to(device)

		with torch.no_grad(), torch.autocast(device_type=device, dtype=torch.bfloat16):
			logits, _ = model(tokens, eval_logits=True)

		shift_logits = logits[:, :-1, :].float()
		shift_targets = tokens[:, 1:]
		shift_mask = mask[:, 1:].float()

		V = shift_logits.size(-1)
		loss = F.cross_entropy(
			shift_logits.reshape(-1, V),
			shift_targets.reshape(-1),
			reduction="none"
		).reshape(total_seqs, -1)

		masked_loss = loss * shift_mask
		sum_loss = masked_loss.sum(dim=-1)       # (total_seqs,)
		avg_loss = sum_loss / shift_mask.sum(dim=-1).clamp(min=1)

		# split back per example
		offset = 0
		for i, C in enumerate(candidates_per_example):
			ex_sum = sum_loss[offset:offset + C]
			ex_avg = avg_loss[offset:offset + C]
			label = labels[i]

			if ex_sum.argmin().item() == label: num_correct += 1
			if ex_avg.argmin().item() == label: num_correct_norm += 1
			num_total += 1
			offset += C

	return {
		"acc": num_correct / num_total,
		"acc_norm": num_correct_norm / num_total,
		"num_total": num_total,
	}

def main():
	parser = argparse.ArgumentParser(description="download + normalize eval data, and/or evaluate checkpoints")

	parser.add_argument("eval_name", help="eval name (used for directory + defaults lookup)")

	parser.add_argument("--dataset_id", type=str)
	parser.add_argument("--subset", type=str)
	parser.add_argument("--split", type=str)
	parser.add_argument("--question_col", type=str, default="ctx")
	parser.add_argument("--answer_col", type=str, default="endings")
	parser.add_argument("--label_col", type=str, default="label")

	parser.add_argument("--s3_remote", type=str, default=None,
		help="S3 remote base, e.g. s3://my-bucket/toy-transformers")

	parser.add_argument("--config", type=str, default=None,
		help="training config JSON (model arch + vocab)")
	parser.add_argument("--checkpoint", nargs="+", type=str, default=None,
		help="one or more checkpoint directories")
	parser.add_argument("--device", type=str, default="cuda")
	parser.add_argument("--limit", type=int, default=0,
		help="max examples, 0=all")
	parser.add_argument("--batch_size", type=int, default=4,
		help="examples per forward pass (default: 4)")

	args = parser.parse_args()

	dataset_id = args.dataset_id
	subset = args.subset
	split = args.split
	question_col = args.question_col
	answer_col = args.answer_col
	label_col = args.label_col

	eval_dir = DATA_DIR / "evals" / args.eval_name
	eval_dir.mkdir(parents=True, exist_ok=True)
	raw_dir = eval_dir / "_raw"
	eval_parquet = eval_dir / f"{args.eval_name}.parquet"

	run_download(dataset_id, subset, split, raw_dir)
	run_reformat(raw_dir, eval_parquet, question_col, answer_col, label_col)
	if args.s3_remote:
		run_upload(eval_dir, args.s3_remote)

if __name__ == "__main__":
	main()