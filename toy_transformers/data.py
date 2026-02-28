import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader

class ShardedTokenDataset(IterableDataset):
  def __init__(self,
    shard_paths: list[Path],
    block_size: int,
    bos_id: int,
    pad_id: int,
    max_doc_len: int = 0,
    shuffle_docs: bool = True,
    seed: int = 42,
  ):
    self.shard_paths = shard_paths
    self.block_size = block_size
    self.bos_id = bos_id
    self.pad_id = pad_id
    self.max_doc_len = max_doc_len or (block_size * 2)
    self.shuffle_docs = shuffle_docs
    self.seed = seed

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
  
  def _make_sample(self, pack: list[np.ndarray], pack_len: int):
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
    loss_mask = (x != self.pad_id)
    return x, y, doc_ids, loss_mask
  
  def __iter__(self):
    pack = []
    pack_len = 0
    
    for shard_idx, shard_path in enumerate(self.shard_paths):
      raw = np.frombuffer(shard_path.read_bytes(), dtype=np.uint16)

      for doc in self._iter_docs(raw, shard_idx):
        doc_len = len(doc)

        if pack_len + doc_len > self.block_size:
          if pack_len > 0:
            yield self._make_sample(pack, pack_len)
          pack = []
          pack_len = 0
        
        if doc_len > self.block_size:
          doc = doc[:self.block_size]
          doc_len = self.block_size 
        
        pack.append(doc)
        pack_len += doc_len

def create_splits(
  shard_dir: Path,
  val_shards: int = 1,
  test_shards: int = 0
):
  shard_dir = Path(shard_dir)

  with open(shard_dir / "metadata.json") as f:
    meta = json.load(f)
  
  shard_names = sorted(meta["shard_tokens"].keys())
  shard_paths = [shard_dir / name for name in shard_names]

  s1 = len(shard_names) - val_shards - test_shards
  s2 = s1 + val_shards

  return shard_paths[:s1], shard_paths[s1:s2], shard_paths[s2:]