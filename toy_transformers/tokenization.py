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

  from tqdm import tqdm
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