import argparse
import requests
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from toy_transformers.data import S3Sync

REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / "data"


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

def run_upload(eval_dir: Path, bucket: str):
	sync = S3Sync(remote_base=f"s3://{bucket}/toy-transformers", local_root=DATA_DIR.parent)

	print("[UPLOAD]", "uploading eval data to S3")
	for path in sorted(eval_dir.iterdir()):
		if path.is_file():
			rel = path.relative_to(DATA_DIR.parent)
			sync.push(rel)
	print("[UPLOAD]", "done")


def main():
	parser = argparse.ArgumentParser(description="download + normalize eval data")

	parser.add_argument("eval_name", help="eval name (used for directory)")

	parser.add_argument("--dataset_id", type=str, required=True)
	parser.add_argument("--subset", type=str, required=True)
	parser.add_argument("--split", type=str, required=True)
	parser.add_argument("--question_col", type=str, default="ctx")
	parser.add_argument("--answer_col", type=str, default="endings")
	parser.add_argument("--label_col", type=str, default="label")

	parser.add_argument("--bucket", type=str, default=None,
		help="S3 bucket name, e.g. my-bucket")

	args = parser.parse_args()

	eval_dir = DATA_DIR / "evals" / args.eval_name
	eval_dir.mkdir(parents=True, exist_ok=True)
	raw_dir = eval_dir / "_raw"
	eval_parquet = eval_dir / f"{args.eval_name}.parquet"

	run_download(args.dataset_id, args.subset, args.split, raw_dir)
	run_reformat(raw_dir, eval_parquet, args.question_col, args.answer_col, args.label_col)
	if args.bucket:
		run_upload(eval_dir, args.bucket)

if __name__ == "__main__":
	main()
