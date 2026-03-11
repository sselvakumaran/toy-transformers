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
    tmp = remote.with_suffix(".tmp")
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
    shard_dir: Path,
    queue: mp.Queue,
    num_epochs: int = 1,
    shuffle: bool = True,
    seed: int = 42,
    start_epoch: int = 0,
    skip_shards: int = 0,
  ):
    super().__init__(daemon=True)
    self.sync = sync
    self.queue = queue
    self.num_epochs = num_epochs
    self.shuffle = shuffle
    self.seed = seed
    self.start_epoch = start_epoch
    self.skip_shards = skip_shards

    self.shard_dir = shard_dir
    names = self.sync.ls(shard_dir)
    self.shards = sorted([n for n in names if not n.endswith("/")])
  
  @classmethod
  def from_remote(
    cls,
    remote_prefix: str,
    local_dir: Path,
    queue: mp.Queue,
    **kwargs,
  ) -> "S3ShardDownloader":
    data_sync = S3Sync(
      remote_base=remote_prefix,
      local_root=local_dir,
    )
    return cls(
      sync=data_sync,
      shard_dir=local_dir,
      queue=queue,
      **kwargs,
    )

  @property
  def num_shards(self) -> int:
    return len(self.shards)

  def run(self):
    try:
      for epoch in range(self.start_epoch, self.num_epochs):
        order = list(self.shards)
        if self.shuffle:
          random.Random(self.seed + epoch).shuffle(order)

        skip = self.skip_shards if epoch == self.start_epoch else 0
        for i, shard in enumerate(order):
          if i < skip: continue
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
    self.shards_consumed = 0

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
    doc_ids = (x == self.bos_id).cumsum(0) - 1
    doc_ids = doc_ids.clamp(min=0)
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
    self.shards_consumed = 0

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

      self.shards_consumed += 1
      if self.cleanup:
        shard_path.unlink(missing_ok=True)