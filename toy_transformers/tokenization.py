# handles all tokenization stuff
# has Vocabulary, ProcessedDataset
# note: ProcessedDataset should have a create dataloader fn

import base64
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from functools import cached_property
import heapq
import json
import multiprocessing as mp
from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union
import random

Token = str | bytes
TokenSequence = str | bytes

UNK = "<UNK>"
ENCODING = 'utf-8'
DEFAULT_PATTERN = r"|".join([
  r"'(?i:[sdmt]|ll|ve|re)", # contractions
  r"[^\r\n\w]?[^\W\d_]+", # space + letters
  r"\d{1,3}", # numbers
  r" ?[^\s\w]+[\r\n]*", # space + punctuation (slow)
  # r"\s*[\r\n]+", # newlines (slow)
  r"\s+(?!\S)", # trailing whitespace
  r"\s+" # whitespace
])


class TokenizationMode(Enum):
  STR = "str"
  BYTES = "bytes"

  @property
  def python_type(self):
    return str if self == TokenizationMode.STR else bytes

  def match(self, tok) -> bool:
    return isinstance(tok, self.python_type)


@dataclass
class TokenizationConfig():
  mode: TokenizationMode
  vocab_size: int
  special_tokens_str: List[str]
  pattern_str: str
  num_workers: int = 1
  special_tokens: Optional[List[Token]] = field(init=False, repr=False)
  pattern: Optional[re.Pattern] = field(init=False, repr=False)

  def __post_init__(self):
    raw_pattern = self.pattern_str
    if self.special_tokens_str:
      sorted_special_tokens = sorted(self.special_tokens_str, key=len, reverse=True)
      escaped = [re.escape(s) for s in sorted_special_tokens]
      raw_pattern = "|".join(escaped) + "|" + self.pattern_str
    if self.mode == TokenizationMode.BYTES:
      raw_pattern = raw_pattern.encode(ENCODING)
    self.pattern = re.compile(raw_pattern)

    typed_special_tokens = self.special_tokens_str
    if self.mode == TokenizationMode.BYTES:
      typed_special_tokens = [s.encode(ENCODING) for s in self.special_tokens_str]
    self.special_tokens = typed_special_tokens


@dataclass
class Vocabulary():
  config: TokenizationConfig
  tokens: List[Token]
  merges: Dict[int, Tuple[int, int]]

  @cached_property
  def token_to_idx(self) -> Dict[Token, int]:
    return {t: i for i, t in enumerate(self.tokens)}
  
  def __len__(self):
    return len(self.tokens)
  
  @cached_property
  def _pair_to_rank(self) -> Dict[Tuple[int, int], int]:
    return {pair: idx for idx, pair in self.merges.items()}
  
  def encode(self, text: TokenSequence) -> List[int]:
    if not self.config.mode.match(text):
      raise ValueError(f"cannot use type {type(text).__name__} on {self.config.mode}-based vocab")
    
    out = []
    pair_to_rank: dict = self._pair_to_rank
    cache = {}

    special_set = set(self.config.special_tokens)

    for m in re.finditer(self.config.pattern, text):
      chunk = m.group(0)

      if chunk in special_set:
        out.append(self.token_to_idx[chunk])
        continue

      if chunk in cache:
        out.extend(cache[chunk])
        continue

      tokens = [self.token_to_idx[chunk[i:i+1]] for i in range(len(chunk))]
      INF = float('inf')

      while len(tokens) >= 2:
        best_rank, best_pair, best_idx = INF, None, -1

        for i in range(len(tokens) - 1):
          pair = (tokens[i], tokens[i+1])
          rank = pair_to_rank.get(pair, INF)
          if rank < best_rank:
            best_rank, best_pair, best_idx = rank, pair, i

        if best_pair is None: break
        tokens[best_idx] = best_rank
        tokens.pop(best_idx + 1)

      cache[chunk] = tokens
      out.extend(tokens)
    return out

  def decode(self, idxs: List[int]) -> List[Token]:
    return [self.tokens[idx] for idx in idxs]
  
  def save(self, path: str | Path):
    path = Path(path)
    if path.suffix != '.json':
      path = path.with_suffix(".json")
    path.parent.mkdir(parents=True, exist_ok=True)

    is_bytes = self.config.mode == TokenizationMode.BYTES
    serialized_tokens = \
      [base64.b64encode(t).decode('ascii') for t in self.tokens] \
      if is_bytes else self.tokens
    
    obj = {
      "config": {
        "mode": self.config.mode.value,
        "vocab_size": self.config.vocab_size,
        "special_tokens": self.config.special_tokens_str,
        "pattern": self.config.pattern_str
      },
      "tokens": serialized_tokens,
      "merges": self.merges
    }

    with open(path, 'x') as f:
      json.dump(obj, f, indent=4)

  @classmethod
  def load(cls, path: str | Path):
    path = Path(path)
    if path.suffix != '.json':
      path = path.with_suffix(".json")
    with open(path, 'r') as f:
      obj = json.load(f)
    
    cfg = obj["config"]
    mode = TokenizationMode(cfg["mode"])
    tokens = \
      [base64.b64decode(t) for t in obj["tokens"]] \
      if mode == TokenizationMode.BYTES else obj["tokens"]
    
    return cls(
      config=TokenizationConfig(
        mode=mode,
        vocab_size=cfg["vocab_size"],
        special_tokens_str=cfg["special_tokens"],
        pattern_str=cfg["pattern"]
      ),
      tokens=tokens,
      merges={int(k): tuple(v) for k, v in obj["merges"].items()}
    )


class _TokenMergeTracker():
  word_encodings: Dict[int, List[int]]
  pair_counts: Counter
  word_counts: Counter
  pair_to_words: Dict[Tuple[int, int], Set[int]]
  live_words: Set[int]

  def __init__(self,
    word_counts: Counter,
    token_to_idx: Dict[Token, int],
    special_tokens: List[Token]
  ):
    special_tokens = sorted(special_tokens, key=len, reverse=True)
    
    def to_atomic(word: TokenSequence) -> List[int]:
      tokens = []
      i = 0
      while i < len(word):
        for tok in special_tokens:
          if word.startswith(tok, i):
            tokens.append(token_to_idx[tok])
            i += len(tok)
            break
        else:
          tokens.append(token_to_idx[word[i:i+1]])
          i += 1
      return tokens

    self.word_encodings = dict()
    self.pair_counts = Counter()
    self.word_counts = Counter()
    self.pair_to_words = defaultdict(set)
    self.live_words = set()
    for word_idx, (word, count) in enumerate(word_counts.items()):
      tokens = to_atomic(word)
      self.word_encodings[word_idx] = tokens
      self.word_counts[word_idx] = count
      if len(tokens) >= 2:
        self.live_words.add(word_idx)
      
      for i in range(len(tokens) - 1):
        pair = (tokens[i], tokens[i+1])
        self.pair_counts[pair] += count
        self.pair_to_words[pair].add(word_idx)

  def merge(self, pair: Tuple[int, int], t3: int) -> List[Tuple[int, int]]:
    t1, t2 = pair
    updated_pairs: Set[Tuple[int, int]] = set()
    for word_idx in self.pair_to_words.pop(pair, []):
      count = self.word_counts[word_idx]
      tokens = self.word_encodings[word_idx]
      i = 0
      while i < len(tokens) - 1:
        if tokens[i] == t1 and tokens[i+1] == t2:
          if i > 0:
            old_pair, new_pair = (tokens[i-1], t1), (tokens[i-1], t3)

            self.pair_counts[old_pair] -= count
            if self.pair_counts[old_pair] <= 0:
              del self.pair_counts[old_pair]
              self.pair_to_words[old_pair].discard(word_idx)
              if not self.pair_to_words[old_pair]:
                del self.pair_to_words[old_pair]
            
            self.pair_counts[new_pair] += count
            self.pair_to_words[new_pair].add(word_idx)
            updated_pairs.add(new_pair)
          
          if i + 2 < len(tokens):
            old_pair, new_pair = (t2, tokens[i+2]), (t3, tokens[i+2])

            self.pair_counts[old_pair] -= count
            if self.pair_counts[old_pair] <= 0:
              del self.pair_counts[old_pair]
              self.pair_to_words[old_pair].discard(word_idx)
              if not self.pair_to_words[old_pair]:
                del self.pair_to_words[old_pair]

            self.pair_counts[new_pair] += count
            self.pair_to_words[new_pair].add(word_idx)
            updated_pairs.add(new_pair)
          tokens[i] = t3
          tokens.pop(i + 1)
        else:
          i += 1
      if len(tokens) < 2:
        self.live_words.discard(word_idx)
    self.pair_counts.pop(pair, None)
    return list(updated_pairs)


def _preprocess_shard(args: tuple[TokenSequence, TokenizationConfig]):
  text, pattern_str, mode_value = args
  mode = TokenizationMode(mode_value)
  if mode == TokenizationMode.BYTES:
    pattern_str = pattern_str.encode(ENCODING)
  pattern = re.compile(pattern_str)

  counts = Counter()
  for m in pattern.finditer(text):
    counts[m.group()] += 1
  return counts

def _parallel_preprocess(
  data_iter: Iterable[TokenSequence],
  config: TokenizationConfig
):

  raw = config.pattern.pattern
  if isinstance(raw, bytes):
    raw = raw.decode(ENCODING)
  mode_val = config.mode.value

  def make_args():
    for text in data_iter:
      yield (text, raw, mode_val)
  
  total_counts = Counter()

  from tqdm import tqdm
  pbar = tqdm(total=None, desc="preprocessing", unit="shard")

  if config.num_workers <= 1:
    for args in make_args():
      total_counts += _preprocess_shard(args)
      pbar.update(1)
  else:
    with mp.Pool(config.num_workers) as pool:
      for shard_counts in pool.imap_unordered(_preprocess_shard, make_args()):
        total_counts += shard_counts
        pbar.update(1)

  pbar.close()
  
  return total_counts


def create_bpe(
  data_iter: Iterable[TokenSequence],
  vocab_size: int,
  mode: Optional[TokenizationMode] = TokenizationMode.STR,
  splitting_pattern: Optional[str] = DEFAULT_PATTERN,
  special_tokens: Optional[List[str]] = None,
  num_workers: Optional[int] = None,
  heap_cleanup_ratio: float = 4.0,
) -> Vocabulary:
  from tqdm import tqdm

  if num_workers is None:
    num_workers = max(1, (mp.cpu_count() or 1) - 2)
  
  config = TokenizationConfig(
    mode, 
    vocab_size, 
    special_tokens_str=special_tokens or [],
    pattern_str=splitting_pattern, 
    num_workers=num_workers
  )

  counts = _parallel_preprocess(data_iter, config)

  base_tokens = []
  special_token_set = set(config.special_tokens)

  if mode == TokenizationMode.BYTES:
    base_tokens = [bytes([i]) for i in range(256)
      if bytes([i]) not in special_token_set]
  else:
    chars = set()
    for word in counts:
      chars.update(word)
    base_tokens = sorted(chars - special_token_set)
  vocab = list(config.special_tokens) + base_tokens
  token_to_idx = {c: i for i, c in enumerate(vocab)}

  corpus = _TokenMergeTracker(counts, token_to_idx, config.special_tokens)
  heap = [(-count, pair) for pair, count in corpus.pair_counts.items()]
  heapq.heapify(heap)
  merges = dict()
  num_live_pairs = len(corpus.pair_counts)

  print("starting merging...")
  pbar = tqdm(total=vocab_size - len(vocab), desc="BPE Training")

  while len(vocab) < vocab_size and heap:
    if heap_cleanup_ratio > 0 and len(heap) > heap_cleanup_ratio * max(num_live_pairs, 1):
      heap = [(-c, p) for p, c in corpus.pair_counts.items() if c > 0]
      heapq.heapify(heap)
      num_live_pairs = len(heap)
    neg_count, pair = heapq.heappop(heap)
    actual_count = corpus.pair_counts.get(pair, 0)
    if actual_count != -neg_count:
      if actual_count > 0:
        heapq.heappush(heap, (-actual_count, pair))
      continue
    if actual_count == 0:
      continue
    t3 = len(vocab)
    t1, t2 = pair
    new_token = vocab[t1] + vocab[t2]
    vocab.append(new_token)
    token_to_idx[new_token] = t3
    merges[t3] = (t1, t2)
    updated_pairs = corpus.merge(pair, t3)
    for p in updated_pairs:
      c = corpus.pair_counts.get(p, 0)
      if c > 0:
        heapq.heappush(heap, (-c, p))
    
    pbar.update(1)
  
  pbar.close()

  return Vocabulary(config, vocab, merges)

import numpy as np

def _write_shard(path: Path, tokens: np.ndarray):
  assert tokens.dtype == np.uint16
  with open(path, "wb") as f:
    f.write(tokens.tobytes())

def _read_shard(path: Path) -> np.ndarray:
  with open(path, "rb") as f:
    return np.frombuffer(f.read(), dtype=np.uint16)

_worker_vocab: Vocabulary = None

def _init_encode_worker(vocab_path: str):
  global _worker_vocab
  _worker_vocab = Vocabulary.load(vocab_path)

def _bulk_encode_worker(text: TokenSequence) -> np.ndarray:
  ids = _worker_vocab.encode(text)
  return np.array(ids, dtype=np.uint16)

# ASSUMES doc_iter ALREADY HAS SPLIT_TOKENS WITHIN IT!
# obviously retokenizing the BOS token is less efficient than adding it, 
# will change if meaningful speed improvements later
def bulk_encode(
  doc_iter: Iterable[TokenSequence],
  vocab: Vocabulary,
  vocab_path: Path,
  output_dir: Path,
  split_token: Token,
  shard_size: int = 100_000_000,
  num_workers: Optional[int] = None,
):
  from tqdm import tqdm

  if num_workers is None:
    num_workers = max(1, (mp.cpu_count() or 1) - 2)

  output_dir = Path(output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)

  if vocab.config.mode == TokenizationMode.BYTES:
    split_tok = split_token.encode(ENCODING)
  else:
    split_tok = split_token
  split_id = vocab.token_to_idx[split_tok]

  vocab_path = Path(vocab_path)

  buf = np.empty(shard_size, dtype=np.uint16)
  buf_pos = 0
  shard_idx = 0
  shard_token_counts = []

  def flush_shard(buf_pos, shard_idx, shard_token_counts):
    if buf_pos == 0:
      return buf_pos, shard_idx
    path = output_dir / f"shard_{shard_idx:04d}.bin"
    _write_shard(path, buf[:buf_pos])
    shard_token_counts.append(buf_pos)
    print(f"wrote {buf_pos:,} tokens to {path.name}")
    return 0, shard_idx + 1
  
  def append_tokens(tokens: np.ndarray, buf_pos, shard_idx):
    remaining = tokens
    while len(remaining) > 0:
      space = shard_size - buf_pos
      take = min(space, len(remaining))
      buf[buf_pos:buf_pos + take] = remaining[:take]
      buf_pos += take
      remaining = remaining[take:]
      if buf_pos < shard_size:
        continue
      # writing
      split_positions = np.where(buf[:buf_pos] == split_id)[0]
      
      # if document larger than shard length
      if not (len(split_positions) > 0 and split_positions[-1] > 0):
        buf_pos, shard_idx = flush_shard(buf_pos, shard_idx, shard_token_counts)
        continue
      
      # cut by last document
      cut = int(split_positions[-1])
      leftover = buf_pos - cut
      _, shard_idx = flush_shard(cut, shard_idx, shard_token_counts)
      buf[:leftover] = buf[cut:cut + leftover]
      buf_pos = leftover
    return buf_pos, shard_idx
  
  pbar = tqdm(desc="encoding", unit="chunk")

  if num_workers <= 1:
    _init_encode_worker(str(vocab_path))
    for text in doc_iter:
      buf_pos, shard_idx = append_tokens(_bulk_encode_worker(text), buf_pos, shard_idx)
      pbar.update(1)
  else:
    with mp.Pool(num_workers, initializer=_init_encode_worker, initargs=(vocab_path,)) as pool:
      for encoded in pool.imap(_bulk_encode_worker, doc_iter):
        buf_pos, shard_idx = append_tokens(encoded, buf_pos, shard_idx)
        pbar.update(1)
  
  buf_pos, shard_idx = flush_shard(buf_pos, shard_idx, shard_token_counts)
  pbar.close()

  meta = {
    "num_shards": shard_idx,
    "vocab_path": str(vocab_path),
    "split_id": split_id,
    "shard_tokens": {
      f"shard_{i:04d}.bin": int(count) for i, count in enumerate(shard_token_counts)
    }
  }
  with open(output_dir / "metadata.json", "w") as f:
    json.dump(meta, f, indent=2)
  
def shuffle_shards(
  input_dir: Path,
  output_dir: Path,
  seed: int = 42,
  read_chunk_tokens: int = 1_000_000,
  write_buffer_tokens: int = 1_000_000,   
):
  from tqdm import tqdm

  input_dir = Path(input_dir)
  output_dir = Path(output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)

  with open(input_dir / "metadata.json", "r") as f:
    meta = json.load(f)
  split_id = meta.get("split_id", 0)
  shard_names: list[str] = sorted(meta["shard_tokens"].keys())
  num_shards = meta.get("num_shards", 1)

  rng = random.Random(seed)

  out_bufs: List[List[np.ndarray]] = [[] for _ in range(num_shards)]
  out_buf_sizes = [0 for _ in range(num_shards)]
  out_token_counts = [0 for _ in range(num_shards)]
  out_paths = [output_dir / f"shard_{i:04d}.bin" for i in range(num_shards)]

  for p in out_paths: 
    p.write_bytes(b"")
  
  def flush_output(shard_idx):
    if out_buf_sizes[shard_idx] == 0:
      return
    
    combined = np.concatenate(out_bufs[shard_idx])
    with open(out_paths[shard_idx], 'ab') as f:
      f.write(combined.tobytes())
    
    out_bufs[shard_idx] = []
    out_buf_sizes[shard_idx] = 0
    out_token_counts[shard_idx] += len(combined)
  
  def send_doc(doc: np.ndarray):
    idx = rng.randrange(num_shards)
    out_bufs[idx].append(doc)
    out_buf_sizes[idx] += len(doc)
    if out_buf_sizes[idx] >= write_buffer_tokens:
      flush_output(idx)
  
  total_docs = 0
  pbar = tqdm(desc="shuffling", unit="doc")

  for shard_name in shard_names:
    shard_path = input_dir / shard_name
    file_bytes = shard_path.stat().st_size
    n_tokens = file_bytes // 2
    remainder = np.empty(0, dtype=np.uint16)

    with open(shard_path, 'rb') as f:
      tokens_read = 0

      while tokens_read < n_tokens:
        chunk_size = min(read_chunk_tokens, n_tokens - tokens_read)
        raw = f.read(chunk_size * 2)
        chunk = np.frombuffer(raw, dtype=np.uint16)
        tokens_read += len(chunk)

        if len(remainder) > 0:
          chunk = np.concatenate([remainder, chunk])
          remainder = np.empty(0, dtype=np.uint16)
        
        split_positions = np.where(chunk == split_id)[0]
        if len(split_positions) == 0:
          remainder = chunk
          continue
        
        for i in range(len(split_positions) - 1):
          doc = chunk[split_positions[i]:split_positions[i+1]]
          send_doc(doc)
          total_docs += 1
          pbar.update(1)
        
        remainder = chunk[split_positions[-1]:]

      if len(remainder) > 0:
        send_doc(remainder)
        total_docs += 1
        pbar.update(1)

  for i in range(num_shards):
    flush_output(i)
  
  pbar.close()

  out_meta = {
    "num_shards": num_shards,
    "vocab_path": meta.get("vocab_path"),
    "split_id": split_id,
    "shuffled": True,
    "seed": seed,
    "shard_tokens": {
      f"shard_{i:04d}.bin": int(out_token_counts[i]) for i in range(num_shards)
    }
  }
  with open(output_dir / "metadata.json", 'w') as f:
    json.dump(out_meta, f, indent=2)