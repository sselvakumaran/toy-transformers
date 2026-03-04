import multiprocessing as mp
from pathlib import Path
import random
import subprocess
import time
from typing import Optional

import numpy as np
import torch
from dataclasses import dataclass
from torch.utils.data import IterableDataset, DataLoader

_SENTINEL = None       # signals "no more shards" on the queue
_ERROR_KEY = "__error__"


@dataclass
class S3Sync():
  # wrapper for syncing files between local and S3
  remote_path: str # something like "s3://BUCKET_NAME/**/toy-transformers/"
  local_path: str | Path

  def __post_init__(self):
    self.local_path = Path(self.local_path)
    self.remote_path = self.remote_path.rstrip("/")

  def _local_to_remote(self, local_path: Path | str) -> str:
    local_path = Path(local_path)
    rel = local_path.relative_to(self.local_path)
    return f"{self.remote_path}/{rel}"

  def push(self, local_path: Path | str, dry_run=False) -> bool:
    local_path = Path(local_path)
    remote = self._local_to_remote(local_path)
    cmd = ["aws", "s3",
      "cp" if local_path.is_file() else "sync",
      str(local_path), str(remote)
    ]
    if dry_run:
      print("dry-run: " + " ".join(cmd))
      return True
    
    r = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if r.returncode != 0:
      print(f"push failed: {r.stderr}")
    return r.returncode == 0
  
  def pull(self, local_path: Path | str, dry_run=False) -> bool:
    local_path = Path(local_path)
    remote = self._local_to_remote(local_path)
    is_dir = not local_path.exists() or local_path.is_dir()
    cmd = ["aws", "s3",
      "sync" if is_dir else "cp", 
      str(remote), str(local_path)
    ]
    if dry_run:
      print("dry-run: " + " ".join(cmd))
      return True
    
    local_path.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if r.returncode != 0:
      print(f"pull failed: {r.stderr}")
    return r.returncode == 0

  def pull_atomic(self, local_path: Path | str, retries: int = 2) -> Path:
    local_path = Path(local_path)
    if local_path.exists():
      return local_path

    remote = self._local_to_remote(local_path)
    tmp = local_path.with_suffix(".tmp")
    local_path.parent.mkdir(parents=True, exist_ok=True)

    try:
      for attempt in range(retries):
        r = subprocess.run(
          ["aws", "s3", "cp", str(remote), str(tmp)],
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
      tmp.rename(local_path)
      return local_path
    except Exception as e:
      if tmp.exists():
        tmp.unlink()
      raise e
  def exists(self, local_path: Path | str) -> bool:
    remote = self._local_to_remote(local_path)
    if Path(local_path).is_dir() and not remote.endswith("/"):
      remote += "/"
    cmd = ["aws", "s3", "ls", remote]
    r = subprocess.run(cmd, check=False, capture_output=True)
    return r.returncode == 0

  def ls(self, local_path: Path | str) -> list[str]:
    remote = self._local_to_remote(local_path)
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
    shard_dir: Path,
    queue: mp.Queue,
    num_epochs: int = 1,
    shuffle: bool = True,
    seed: int = 42,
  ):
    super().__init__(daemon=True)
    self.sync = sync
    self.queue = queue
    self.num_epochs = num_epochs
    self.shuffle = shuffle
    self.seed = seed

    self.shard_dir = shard_dir
    names = self.sync.ls(shard_dir)
    self.shards = [n for n in names if not n.endswith("/")]

  def run(self):
    try:
      for epoch in range(self.num_epochs):
        order = list(self.shards)
        if self.shuffle:
          random.Random(self.seed + epoch).shuffle(order)
        for shard in order:
          local_path = self.sync.pull_atomic(self.shard_dir / shard)
          self.queue.put(local_path)  # blocking
      self.queue.put(_SENTINEL)
    except Exception as e:
      self.queue.put({_ERROR_KEY: str(e)})


class ShardedTokenDataset(IterableDataset):
  def __init__(self,
    block_size: int,
    bos_id: int,
    pad_id: int,
    shard_paths: list[Path] | None = None,
    queue: Optional[mp.Queue] = None,
    max_doc_len: int = 0,
    shuffle_docs: bool = True,
    seed: int = 42,
    cleanup: bool = False,
  ):
    assert (shard_paths is None) != (queue is None), \
      "provide exactly one of shard_paths or queue"
    self.shard_paths = shard_paths
    self.queue = queue
    self.block_size = block_size
    self.bos_id = bos_id
    self.pad_id = pad_id
    self.max_doc_len = max_doc_len or block_size
    self.shuffle_docs = shuffle_docs
    self.seed = seed
    self.cleanup = cleanup

  def _iter_docs(self, tokens: np.ndarray, shard_idx: int):
    bos_pos = np.where(tokens == self.bos_id)[0]
    if len(bos_pos) == 0:
      return

    boundaries = list(zip(bos_pos, list(bos_pos[1:]) + [len(tokens)]))

    if self.shuffle_docs:
      rng = random.Random(self.seed + shard_idx)
      rng.shuffle(boundaries)

    for start, end in boundaries:
      doc = tokens[start:end]
      if len(doc) > self.max_doc_len:
        doc = doc[:self.max_doc_len]
      yield doc

  def _make_sample(self, pack: list[np.ndarray]):
    x_np = np.full(self.block_size, self.pad_id, dtype=np.uint16)
    y_np = np.full(self.block_size, self.pad_id, dtype=np.uint16)

    tokens = np.concatenate(pack)
    n = len(tokens)

    x_np[:n] = tokens[:n]
    if n > 1:
      y_np[:n - 1] = tokens[1:n]

    x = torch.from_numpy(x_np.astype(np.int64))
    y = torch.from_numpy(y_np.astype(np.int64))
    doc_ids = (x == self.bos_id).cumsum(0)
    loss_mask = (x != self.pad_id) & (y != self.pad_id)
    return x, y, doc_ids, loss_mask

  def _shard_iter(self):
    if self.queue is None:
      yield from self.shard_paths
      return
    while True:
      item = self.queue.get()
      if isinstance(item, dict) and _ERROR_KEY in item:
        raise RuntimeError(f"S3 downloader failed: {item[_ERROR_KEY]}")
      if item is _SENTINEL:
        return
      yield item

  def __iter__(self):
    pack = []
    pack_len = 0

    for shard_idx, shard_path in enumerate(self._shard_iter()):
      raw = np.frombuffer(shard_path.read_bytes(), dtype=np.uint16)

      for doc in self._iter_docs(raw, shard_idx):
        doc_len = len(doc)

        if pack_len + doc_len > self.block_size:
          if pack_len > 0:
            yield self._make_sample(pack)
          pack = []
          pack_len = 0

        if doc_len > self.block_size:
          doc = doc[:self.block_size]
          doc_len = self.block_size

        pack.append(doc)
        pack_len += doc_len

      if self.cleanup:
        shard_path.unlink(missing_ok=True)