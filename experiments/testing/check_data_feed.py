import argparse
import base64
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from toy_transformers.data import AggregateDataset


def load_vocab_info(path: Path) -> dict:
  obj = json.loads(path.read_text())
  mode = obj["config"]["mode"]
  raw_tokens = obj["tokens"]
  tokens = [
    base64.b64decode(t) for t in raw_tokens
  ] if mode == "bytes" else raw_tokens

  def encode_special(tok: str):
    return tok.encode("utf-8") if mode == "bytes" else tok

  token_to_idx = {tok: i for i, tok in enumerate(tokens)}
  specials = [encode_special(t) for t in obj["config"]["special_tokens"]]
  bos_id = token_to_idx[specials[0]]
  pad_id = token_to_idx[specials[1]] if len(specials) > 1 else -1
  return {
    "mode": mode,
    "tokens": tokens,
    "bos_id": bos_id,
    "pad_id": pad_id,
    "special_tokens": obj["config"]["special_tokens"],
  }


def decode_tokens(vocab: dict, ids: list[int]) -> str:
  pieces = [vocab["tokens"][i] for i in ids]
  if vocab["mode"] == "bytes":
    return b"".join(pieces).decode("utf-8", errors="replace")
  return "".join(pieces)


def local_shards(folder: str, names: list[str], limit: int) -> tuple[list[Path], int]:
  root = REPO_ROOT / "data" / "datasets" / folder
  paths = [root / name for name in names if (root / name).exists()]
  return paths[:limit], len(names) - len(paths)


def summarize_metadata(cfg: dict, vocab: dict, shard_limit: int):
  sources, val_sources = [], []
  folders = cfg["dataset"]["dataset_folders"]
  weights = cfg["dataset"]["dataset_weights"]
  cfg_vocab = cfg["tokenizer"]["path"]

  print("[CONFIG]", f"block_size={cfg['model']['config']['block_size']}",
    f"batch_size={cfg['tokens']['batch_size']}",
    f"weights={dict(zip(folders, weights))}")
  print("[VOCAB]", f"path={cfg_vocab}",
    f"mode={vocab['mode']}",
    f"bos_id={vocab['bos_id']}",
    f"pad_id={vocab['pad_id']}",
    f"specials={vocab['special_tokens']}")

  for folder, weight in zip(folders, weights):
    meta_path = REPO_ROOT / "data" / "datasets" / folder / "metadata.json"
    meta = json.loads(meta_path.read_text())
    train_names = meta.get("train_shards", [])
    val_names = meta.get("val_shards", [])
    train_paths, train_missing = local_shards(folder, train_names, shard_limit)
    val_paths, val_missing = local_shards(folder, val_names, shard_limit)

    vocab_ok = meta.get("vocab_path") == cfg_vocab
    split_ok = meta.get("split_id") in (None, vocab["bos_id"])
    print("[DATASET]", folder,
      f"weight={weight}",
      f"metadata_vocab_ok={vocab_ok}",
      f"split_id_ok={split_ok}",
      f"train_local={len(train_paths)}/{len(train_names)}",
      f"val_local={len(val_paths)}/{len(val_names)}")
    if train_missing:
      print("[DATASET]", folder, f"train shards missing locally: {train_missing}")
    if val_missing:
      print("[DATASET]", folder, f"val shards missing locally: {val_missing}")

    if train_paths:
      sources.append((train_paths, weight))
    if val_paths:
      val_sources.append((val_paths, weight))

  return sources, val_sources


def batch_stats(name: str, loader: DataLoader, vocab: dict, max_batches: int, decode_chars: int):
  bos_id, pad_id = vocab["bos_id"], vocab["pad_id"]
  print(f"[{name.upper()}]", "batch sanity")

  for batch_idx, (x, y, doc_ids, loss_mask) in enumerate(loader):
    if batch_idx >= max_batches:
      break

    supervised = loss_mask.sum(dim=1)
    bos_counts = (x == bos_id).sum(dim=1)
    pad_counts = (x == pad_id).sum(dim=1)
    doc_counts = doc_ids.max(dim=1).values + 1
    bad_shift = ((y[:, :-1] != x[:, 1:]) & loss_mask[:, :-1]).sum().item()
    masked_pad_targets = ((y == pad_id) & loss_mask).sum().item()

    print(f"[{name.upper()}]", f"batch={batch_idx}",
      f"shape={tuple(x.shape)}",
      f"supervised_min/mean/max={int(supervised.min())}/{supervised.float().mean().item():.1f}/{int(supervised.max())}",
      f"bos_min/mean/max={int(bos_counts.min())}/{bos_counts.float().mean().item():.1f}/{int(bos_counts.max())}",
      f"pad_min/mean/max={int(pad_counts.min())}/{pad_counts.float().mean().item():.1f}/{int(pad_counts.max())}",
      f"docs_min/mean/max={int(doc_counts.min())}/{doc_counts.float().mean().item():.1f}/{int(doc_counts.max())}",
      f"bad_shift={bad_shift}",
      f"masked_pad_targets={masked_pad_targets}")

    sample = 0
    real_len = int((x[sample] != pad_id).sum())
    token_ids = x[sample, :real_len].tolist()
    text = decode_tokens(vocab, token_ids)
    text = text.replace("\n", "\\n")
    print(f"[{name.upper()}]", f"sample0_real_tokens={real_len}",
      f"sample0_text={text[:decode_chars]!r}")


def make_loader(sources, cfg: dict, vocab: dict, batch_size: int, seed: int):
  dataset = AggregateDataset.from_shards(
    sources=sources,
    block_size=cfg["model"]["config"]["block_size"],
    bos_id=vocab["bos_id"],
    pad_id=vocab["pad_id"],
    shuffle_docs=True,
    seed=seed,
  )
  return DataLoader(dataset, batch_size=batch_size, num_workers=0, drop_last=True)


def summarize_metrics(run_name: str):
  path = REPO_ROOT / "runs" / run_name / "metrics.jsonl"
  if not path.exists():
    print("[METRICS]", f"no local metrics found at {path.relative_to(REPO_ROOT)}")
    return

  train, val = [], []
  for line in path.read_text().splitlines():
    row = json.loads(line)
    if "t_loss" in row:
      train.append(row)
    if "v_loss" in row:
      val.append(row)

  if train:
    first, last = train[0], train[-1]
    print("[METRICS]", f"train_loss first={first['t_loss']}@{first['step']}",
      f"last={last['t_loss']}@{last['step']}")
  if val:
    first, last = val[0], val[-1]
    print("[METRICS]", f"val_loss first={first['v_loss']}@{first['step']}",
      f"last={last['v_loss']}@{last['step']}")


def main():
  parser = argparse.ArgumentParser(description="Sanity-check local token shards and packed batches.")
  parser.add_argument("--config", default="configs/v4.0.4.dataset_test.json")
  parser.add_argument("--batch-size", type=int, default=8,
    help="diagnostic batch size; use a small value to keep output readable")
  parser.add_argument("--batches", type=int, default=2)
  parser.add_argument("--shards-per-source", type=int, default=8)
  parser.add_argument("--decode-chars", type=int, default=500)
  parser.add_argument("--seed", type=int, default=None)
  args = parser.parse_args()

  cfg_path = Path(args.config)
  if not cfg_path.is_absolute():
    cfg_path = REPO_ROOT / cfg_path
  cfg = json.loads(cfg_path.read_text())
  seed = cfg.get("run", {}).get("seed", 42) if args.seed is None else args.seed

  vocab_path = REPO_ROOT / cfg["tokenizer"]["path"]
  vocab = load_vocab_info(vocab_path)
  train_sources, val_sources = summarize_metadata(cfg, vocab, args.shards_per_source)

  if train_sources:
    train_loader = make_loader(train_sources, cfg, vocab, args.batch_size, seed)
    batch_stats("train", train_loader, vocab, args.batches, args.decode_chars)
  else:
    print("[TRAIN]", "no local train shards found for configured datasets")

  if val_sources:
    val_loader = make_loader(val_sources, cfg, vocab, args.batch_size, seed)
    batch_stats("val", val_loader, vocab, args.batches, args.decode_chars)
  else:
    print("[VAL]", "no local val shards found for configured datasets")

  summarize_metrics(cfg["run"]["name"])


if __name__ == "__main__":
  main()
