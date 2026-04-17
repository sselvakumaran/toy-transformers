"""
Download + reformat ARC-Easy into CF eval format.

Dataset: allenai/ai2_arc / ARC-Easy / test
Format: question text + choices dict with "text" and "label" lists.
  choices["label"] = ["A","B","C","D"], choices["text"] = [str, str, str, str]
  answerKey = "A"/"B"/"C"/"D"

Usage:
  python experiments/preprocessing/arc-easy/prepare.py
  python experiments/preprocessing/arc-easy/prepare.py --bucket my-bucket
"""
import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from toy_transformers.data import S3Sync
from toy_transformers.prepare_eval import run_download, run_upload

REPO_ROOT = Path(__file__).parent.parent.parent.parent
DATA_DIR = REPO_ROOT / "data"

DATASET_ID = "allenai/ai2_arc"
SUBSET = "ARC-Easy"
SPLIT = "train"
EVAL_NAME = "arc-easy"

def run_reformat(raw_dir: Path, out_path: Path):
	print("[REFORMAT]", "arc-easy -> CF format")

	questions, answers, labels = [], [], []
	skipped = 0

	for pf_path in sorted(raw_dir.glob("*.parquet")):
		table = pq.read_table(pf_path)
		for i in range(len(table)):
			question = table["question"][i].as_py()
			choices = table["choices"][i].as_py()
			answer_key = table["answerKey"][i].as_py()

			if answer_key is None or answer_key == "" or choices is None:
				skipped += 1
				continue

			choice_labels = choices["label"]
			choice_texts = choices["text"]

			if answer_key not in choice_labels:
				skipped += 1
				continue

			label_idx = choice_labels.index(answer_key)
			answers_list = [text for text in choice_texts]

			questions.append(str(question))
			answers.append(answers_list)
			labels.append(label_idx)

	out_table = pa.table({
		"question": pa.array(questions, type=pa.string()),
		"answers": pa.array(answers, type=pa.list_(pa.string())),
		"label": pa.array(labels, type=pa.int32()),
	})

	out_path.parent.mkdir(parents=True, exist_ok=True)
	pq.write_table(out_table, out_path)
	print("[REFORMAT]", f"{len(labels)} examples written, {skipped} skipped -> {out_path}")


def main():
	parser = argparse.ArgumentParser(description="prepare arc-easy eval")
	parser.add_argument("--bucket", type=str, default=None, help="S3 bucket name")
	args = parser.parse_args()

	eval_dir = DATA_DIR / "evals" / EVAL_NAME
	eval_dir.mkdir(parents=True, exist_ok=True)
	raw_dir = eval_dir / "_raw"
	eval_parquet = eval_dir / f"{EVAL_NAME}.parquet"

	run_download(DATASET_ID, SUBSET, SPLIT, raw_dir)
	run_reformat(raw_dir, eval_parquet)
	if args.bucket:
		run_upload(eval_dir, args.bucket)

if __name__ == "__main__":
	main()
