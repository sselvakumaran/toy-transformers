import argparse
from pathlib import Path

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from tqdm import tqdm

from toy_transformers.config import TrainingConfig
from toy_transformers.data import S3Sync
from toy_transformers.model_io import load_model
from toy_transformers.tokenization import Vocabulary

REPO_ROOT = Path(__file__).parent.parent


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
	parser = argparse.ArgumentParser(description="evaluate a trained model on an eval dataset")
	parser.add_argument("config", help="training config JSON (model arch + tokenizer)")
	parser.add_argument("eval_name", help="eval name (matches directory under data/evals/)")
	parser.add_argument("--bucket", type=str, required=True,
		help="S3 bucket name, e.g. my-bucket")
	parser.add_argument("--device", type=str, default="cuda")
	parser.add_argument("--limit", type=int, default=0, help="max examples, 0=all")
	parser.add_argument("--batch_size", type=int, default=4)
	args = parser.parse_args()

	cfg = TrainingConfig.from_json(args.config)
	sync = S3Sync(remote_base=f"s3://{args.bucket}/toy-transformers", local_root=REPO_ROOT)

	# load tokenizer
	sync.pull_atomic(cfg.tokenizer.path)
	cfg.tokenizer.load(REPO_ROOT)
	vocab = Vocabulary.load(REPO_ROOT / cfg.tokenizer.path)

	# build model
	model = cfg.model.build_model(vocab_size=cfg.tokenizer.vocab_size, device=args.device)
	model.eval()

	# pull + load checkpoint
	ckpt_rel = f"runs/{cfg.run.name}/checkpoints/model/model.pt"
	sync.pull_atomic(ckpt_rel)
	ckpt_dir = REPO_ROOT / f"runs/{cfg.run.name}/checkpoints/model"
	load_model(ckpt_dir, cfg, model, device=args.device)
	print("[EVAL]", f"loaded checkpoint from {ckpt_dir}")

	# pull eval parquet
	eval_rel = f"data/evals/{args.eval_name}/{args.eval_name}.parquet"
	sync.pull_atomic(eval_rel)
	eval_parquet = REPO_ROOT / eval_rel

	block_size = cfg.model.config["block_size"]
	results = eval(model, vocab, eval_parquet, block_size, args.device, args.limit, args.batch_size)

	print(f"\nacc:      {results['acc']:.4f}")
	print(f"acc_norm: {results['acc_norm']:.4f}")
	print(f"n:        {results['num_total']}")

if __name__ == "__main__":
	main()
