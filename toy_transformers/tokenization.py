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
from pathlib import Path
import re
from typing import Dict, List, Optional, Set, Tuple, Union
from io import BufferedIOBase, RawIOBase, TextIOBase

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
  special_tokens: Optional[List[Token]] = field(init=False, repr=False)
  pattern: Optional[re.Pattern] = field(init=False, repr=False)

  def __post_init__(self):
    typed_pattern_str = self.pattern_str
    if self.mode == TokenizationMode.BYTES:
      typed_pattern_str = self.pattern_str.encode(ENCODING)
    self.pattern = re.compile(typed_pattern_str)

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
      raise ValueError(f"cannot use type f{type(text).__name__} on {self.mode}-based vocab")
    
    out = []
    pair_to_rank: dict = self._pair_to_rank
    cache = {}

    for m in re.finditer(self.config.pattern, text):
      chunk = m.group(0)
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
  word_encodings: Dict[int, List[Token]]
  pair_counts: Counter
  word_counts: Counter
  pair_to_words: Dict[Tuple[Token, Token], Set[int]]

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
    for word_idx, (word, count) in enumerate(word_counts.items()):
      tokens = to_atomic(word)
      self.word_encodings[word_idx] = tokens
      self.word_counts[word_idx] = count
      for i in range(len(tokens) - 1):
        pair = (tokens[i], tokens[i+1])
        self.pair_counts[pair] += count
        self.pair_to_words[pair].add(word_idx)

  def merge(self, pair, t3: int):
    t1, t2 = pair
    updated = set()
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
              self.pair_to_words.pop(old_pair, None)
            
            self.pair_counts[new_pair] += count
            self.pair_to_words[new_pair].add(word_idx)
            updated.add(new_pair)
          
          if i + 2 < len(tokens):
            old_pair, new_pair = (t2, tokens[i+2]), (t3, tokens[i+2])

            self.pair_counts[old_pair] -= count
            if self.pair_counts[old_pair] <= 0:
              del self.pair_counts[old_pair]
              self.pair_to_words.pop(old_pair, None)

            self.pair_counts[new_pair] += count
            self.pair_to_words[new_pair].add(word_idx)
            updated.add(new_pair)
          tokens[i] = t3
          tokens.pop(i+1)
        else: i += 1
    self.pair_counts.pop(pair, None)
    return list(updated)


def _stream_chunks(
  data_handle: Union[TextIOBase, BufferedIOBase],
  pattern: re.Pattern,
  mode: TokenizationMode,
  chunk_size: int,
) -> Counter:
  counts = Counter()
  remainder = "" if mode == TokenizationMode.STR else b""
  while True:
    incoming_chunk = data_handle.read(chunk_size)
    if not incoming_chunk:
      if remainder:
        for m in pattern.finditer(remainder):
          counts[m.group()] += 1
      break

    chunk = remainder + incoming_chunk
    last_match = None
    for m in pattern.finditer(chunk):
      if last_match is not None:
        counts[last_match.group()] += 1
      last_match = m

    if last_match is None:
      remainder = chunk
      continue

    remainder = chunk[last_match.end():]
    counts[last_match.group()] += 1

  return counts


def create_bpe(
  data_handle: Union[TextIOBase, BufferedIOBase],
  vocab_size: int,
  mode: Optional[TokenizationMode] = TokenizationMode.STR,
  splitting_pattern: Optional[str] = DEFAULT_PATTERN,
  special_tokens: Optional[List[str]] = None,
  read_chunk_size: int = 10_000_000
) -> Vocabulary:
  is_handle_binary = isinstance(data_handle, (RawIOBase, BufferedIOBase))
  if mode == TokenizationMode.BYTES:
    assert is_handle_binary, \
      "training via bytes, but file was not opened in binary mode"
  else:
    assert not is_handle_binary, \
      "training via text, but file was opened in binary mode"
  
  config = TokenizationConfig(
    mode, vocab_size, special_tokens or [],
    splitting_pattern
  )
  pattern = config.pattern
  special_tokens = config.special_tokens

  counts = _stream_chunks(data_handle, pattern, mode, read_chunk_size)

  base_tokens = []
  special_token_set = set(special_tokens)
  if mode == TokenizationMode.BYTES:
    base_tokens = [bytes([i]) for i in range(256)
      if bytes([i]) not in special_token_set]
  else:
    chars = set()
    for word in counts:
      chars.update(word)
    base_tokens = sorted(chars - special_token_set)
  vocab = special_tokens + base_tokens
  token_to_idx = {c: i for i, c in enumerate(vocab)}

  corpus = _TokenMergeTracker(counts, token_to_idx, special_tokens)
  heap = [(-count, pair) for pair, count in corpus.pair_counts.items()]
  heapq.heapify(heap)
  merges = dict()

  while len(vocab) < vocab_size and heap:
    neg_count, pair = heapq.heappop(heap)
    actual_count = corpus.pair_counts.get(pair, 0)
    if actual_count != -neg_count:
      if actual_count > 0: heapq.heappush(heap, (-actual_count, pair))
      continue
    if actual_count == 0: continue
    t3 = len(vocab)
    t1, t2 = pair
    new_token = vocab[t1] + vocab[t2]
    vocab.append(new_token)
    token_to_idx[new_token] = t3
    merges[t3] = (t1, t2)
    updated = corpus.merge(pair, t3)
    for updated_pair in updated:
      c = corpus.pair_counts.get(updated_pair, 0)
      if c > 0:
        heapq.heappush(heap,
          (-c, updated_pair)
        )

  return Vocabulary(config, vocab, merges)