import multiprocessing as mp
from pathlib import Path
import random
import subprocess
import time
from typing import Optional

import numpy as np
import torch
from dataclasses import dataclass
from torch.utils.data import IterableDataset

_SENTINEL = None       # signals "no more shards" on the queue
_ERROR_KEY = "__error__"


@dataclass
class S3Sync():
  # wrapper for syncing files between local and S3
  remote_base: str # something like "s3://BUCKET_NAME/**/toy-transformers/"
  local_root: str | Path

  def __post_init__(self):
    self.local_root = Path(self.local_root)
    self.remote_base = self.remote_base.rstrip("/")
    assert self.remote_base.startswith("s3://")

  def _remote(self, rel: str | Path) -> str:
    return f"{self.remote_base}/{Path(rel).as_posix()}"
  
  def _local(self, rel: str | Path) -> Path:
    return self.local_root / rel

  def push(self, rel: str | Path, dry_run = False) -> bool:
    local, remote = self._local(rel), self._remote(rel)
    cmd = ["aws", "s3",
      "cp" if local.is_file() else "sync",
      str(local), str(remote)
    ]
    if dry_run:
      print("dry-run:", " ".join(cmd))
      return True
    
    r = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if r.returncode != 0:
      print(f"push failed: {r.stderr}")
    return r.returncode == 0
  
  def pull(self, rel: str | Path, dry_run=False) -> bool:
    local, remote = self._local(rel), self._remote(rel)
    use_cp = local.suffix != ""
    cmd = ["aws", "s3",
      "cp" if use_cp else "sync", 
      str(remote), str(local)
    ]
    if dry_run:
      print("dry-run: ", " ".join(cmd))
      return True
    
    local.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if r.returncode != 0:
      print(f"pull failed: {r.stderr}")
    return r.returncode == 0

  def pull_atomic(self, rel: str | Path, retries: int = 2) -> Path:
    local = self._local(rel)
    if local.exists():
      return local

    remote = self._remote(rel)
    tmp = local.with_suffix(".tmp")
    local.parent.mkdir(parents=True, exist_ok=True)

    try:
      for attempt in range(retries):
        r = subprocess.run(
          ["aws", "s3", "cp", remote, str(tmp)],
          check=False, capture_output=True, text=True,
        )
        if r.returncode == 0:
          break
        if attempt < retries - 1:
          time.sleep(5)
        else:
          raise RuntimeError(
            f"aws s3 cp failed for {remote} (exit {r.returncode}): {r.stderr}"
          )
      tmp.rename(local)
      return local
    except Exception as e:
      if tmp.exists():
        tmp.unlink()
      raise e
  
  def exists(self, rel: str | Path) -> bool:
    remote = self._remote(rel)
    cmd = ["aws", "s3", "ls", remote]
    r = subprocess.run(cmd, check=False, capture_output=True)
    return r.returncode == 0

  def ls(self, rel: str | Path) -> list[str]:
    remote = self._remote(rel)
    if not remote.endswith("/"):
      remote += "/"
    r = subprocess.run(
      ["aws", "s3", "ls", str(remote)],
      check=False, capture_output=True, text=True,
    )
    if r.returncode != 0:
      return []
    return [line.split(maxsplit=3)[-1].strip() for line in r.stdout.splitlines() if line]


class S3ShardDownloader(mp.Process):
  def __init__(
    self,
    sync: S3Sync,
    shards: list[str], # relative paths
    queue: mp.Queue,
    shuffle: bool = True,
    seed: int = 42,
    skip_shards: int = 0,
  ):
    super().__init__(daemon=True)
    self.sync = sync
    self.shards = list(shards)
    self.queue = queue
    self.shuffle = shuffle
    self.seed = seed
    self.skip_shards = skip_shards

  @property
  def num_shards(self) -> int:
    return len(self.shards)

  def run(self):
    try:
      cycle = 0
      while True:
        order = self.shards.copy()
        if self.shuffle:
          random.Random(self.seed + cycle).shuffle(order)

        skip = self.skip_shards if cycle == 0 else 0
        for i, shard in enumerate(order):
          if i < skip: continue
          local_path = self.sync.pull_atomic(shard)
          self.queue.put(local_path)  # blocking

        cycle += 1
    except Exception as e:
      self.queue.put({_ERROR_KEY: str(e)})


def _iter_docs(
  tokens: np.ndarray, 
  bos_id: int, max_doc_len: int, 
  shuffle: bool, seed: int, 
  shard_idx: int
):
  bos_pos = np.where(tokens == bos_id)[0]
  if len(bos_pos) == 0: return

  boundaries = list(zip(bos_pos, list(bos_pos[1:]) + [len(tokens)]))

  if shuffle:
    random.Random(seed + shard_idx).shuffle(boundaries)
  for start, end in boundaries:
    doc = tokens[start:end]
    if len(doc) > max_doc_len:
      doc = doc[:max_doc_len]
    yield doc

def _make_sample(pack: list[np.ndarray], 
  block_size: int, 
  bos_id: int, pad_id: int
):
  x_np = np.full(block_size, pad_id, dtype=np.uint16)
  y_np = np.full(block_size, pad_id, dtype=np.uint16)
  tokens = np.concatenate(pack)
  n = len(tokens)
  x_np[:n] = tokens[:n]
  if n > 1:
    y_np[:n - 1] = tokens[1:n]
  x = torch.from_numpy(x_np.astype(np.int64))
  y = torch.from_numpy(y_np.astype(np.int64))
  doc_ids = (x == bos_id).cumsum(0) - 1
  doc_ids = doc_ids.clamp(min=0)
  loss_mask = (x != pad_id) & (y != pad_id)
  return x, y, doc_ids, loss_mask

class ShardDataset(IterableDataset):
  def __init__(self,
    shard_paths: list[Path],
    block_size: int, 
    bos_id: int, pad_id: int,
    shuffle: bool = True,
    seed: int = 42
  ):
    self.shard_paths = shard_paths
    self.block_size = block_size
    self.bos_id, self.pad_id = bos_id, pad_id
    self.shuffle = shuffle
    self.seed = seed
  
  def __iter__(self):
    pack, pack_len = [], 0
    for shard_idx, path in enumerate(self.shard_paths):
      raw = np.frombuffer(path.read_bytes(), dtype=np.uint16)
      for doc in _iter_docs(raw, 
        self.bos_id, self.block_size, 
        self.shuffle, self.seed, shard_idx=shard_idx
      ):
        doc_len = len(doc)
        if pack_len + doc_len > self.block_size:
          if pack_len > 0:
            yield _make_sample(pack, self.block_size, self.bos_id, self.pad_id)
          pack, pack_len = [], 0
        pack.append(doc)
        pack_len += doc_len

class AggregateDataset(IterableDataset):
  def __init__(self,
    sources: list[tuple[mp.Queue, float]],
    block_size: int,
    bos_id: int,
    pad_id: int,
    max_doc_len: int = 0,
    shuffle_docs: bool = True,
    seed: int = 42,
    cleanup: bool = False,
  ):
    assert len(sources) > 0
    queues, weights = zip(*sources)
    self.queues: list[mp.Queue] = list(queues)
    total = sum(weights)
    self.probs: list[float] = [w / total for w in weights]
    self.block_size = block_size
    self.bos_id = bos_id
    self.pad_id = pad_id
    self.max_doc_len = max_doc_len or block_size
    self.shuffle_docs = shuffle_docs
    self.seed = seed
    self.cleanup = cleanup
    self.shards_consumed = 0

  def __iter__(self):
    rng = random.Random(self.seed)
    n_sources = len(self.queues)
    doc_iters = [None for _ in range(n_sources)]
    open_paths: list[Path | None] = [None for _ in range(n_sources)]
    shard_counters = [0 for _ in range(n_sources)]

    def next_doc(src: int) -> np.ndarray:
      while True:
        if doc_iters[src] is not None:
          doc = next(doc_iters[src], None)
          if doc is not None: return doc
          if self.cleanup and open_paths[src] is not None:
            open_paths[src].unlink(missing_ok=True)
          self.shards_consumed += 1
        item = self.queues[src].get()
        if isinstance(item, dict) and _ERROR_KEY in item:
          raise RuntimeError(f"downloader error (source {src}): {item[_ERROR_KEY]}")
        open_paths[src] = item
        raw = np.frombuffer(item.read_bytes(), dtype=np.uint16)
        doc_iters[src] = iter(_iter_docs(
          raw, 
          self.bos_id, self.max_doc_len, 
          self.shuffle_docs, self.seed, 
          shard_counters[src]
        ))
        shard_counters[src] += 1

    pack, pack_len = [], 0
    while True:
      src = rng.choices(range(n_sources), weights=self.probs, k=1)[0]
      doc = next_doc(src)
      doc_len = len(doc)
      if pack_len + doc_len > self.block_size:
        if pack_len > 0:
          yield _make_sample(
            pack, 
            self.block_size, self.bos_id, self.pad_id
          )
        pack, pack_len = [], 0
      if doc_len > self.block_size:
        doc = doc[:self.block_size]
        doc_len = self.block_size
      pack.append(doc)
      pack_len += doc_len