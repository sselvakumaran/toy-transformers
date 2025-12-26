from collections import Counter, defaultdict
import heapq
from typing import IO, Dict, Iterable, Iterator, List, Literal, Optional, Set, Tuple, Union, ByteString, Sequence, Pattern
from dataclasses import dataclass, field
from enum import Enum
from functools import cached_property
from itertools import chain
import re
import base64

from toy_transformers.utilities import io
from io import TextIOBase, BufferedIOBase, RawIOBase
Token = str | bytes
TokenSequence = str | bytes

UNK = "<UNK>"
ENCODING = 'utf-8'

DEFAULT_PATTERN = r"|".join([
	r"'(?i:[sdmt]|ll|ve|re)", # contractions
	r"[^\r\n\w]?[^\W\d_]+", # space + letters
	r"\d{1,3}", # numbers
	# r" ?[^\s\w]+[\r\n]*",# space + punctuation
	# r"\s*[\r\n]+", # newlines
	r"\s+(?!\S)" # trailing whitespace
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
class BPEConfig():
	mode: TokenizationMode
	vocab_size: int
	special_tokens_str: List[str]
	pattern_str: str
	special_tokens: Optional[List[Token]] = field(init=False, repr=False)
	pattern: Optional[Pattern] = field(init=False, repr=False)

	def __post_init__(self):
		typed_pattern_str = self.pattern_str
		if self.mode == TokenizationMode.BYTES: 
			typed_pattern_str = self.pattern_str.encode(ENCODING)
		self.pattern = re.compile(typed_pattern_str)

		typed_special_tokens = self.special_tokens_str
		if self.mode == TokenizationMode.BYTES:
			typed_special_tokens = [s.encode(ENCODING) for s in self.special_tokens_str]
		self.special_tokens = typed_special_tokens

# _TRIE: required for Vocab encoding
class _Trie():
	@dataclass
	class _TrieNode():
		children: Dict[Token, '_Trie._TrieNode'] = field(default_factory=dict)
		is_token: bool = False

	def __init__(self, 
		mode: TokenizationMode, 
		tokens: Optional[TokenSequence] = None
	):
		self.root = _Trie._TrieNode()
		self.mode = mode
		if not tokens: return

		for token in tokens:
			self.insert(token)
	
	def insert(self, token: Token):
		node = self.root
		for c in token:
			if c not in node.children:
				node.children[c] = _Trie._TrieNode()
			node = node.children[c]
		node.is_token = True
	
	def _match_longest(self, text: TokenSequence, start: int) -> Token:
		node = self.root
		temp = "" if self.mode == TokenizationMode.STR else b""
		out = None
		for i in range(start, len(text)):
			c = text[i]
			if c in node.children:
				node = node.children[c]
				temp += c
				if node.is_token:
					out = temp
			else: break
		if out is None:
			return UNK, i + 1
		return out, start + len(out)

	def tokenize(self, s: TokenSequence):
		tokens = []
		i = 0
		while i < len(s):
			tok, j = self._match_longest(s, i)
			tokens.append(tok)
			i = j
		return tokens

@dataclass
class Vocabulary():
	config: BPEConfig
	tokens: List[Token]
	
	@cached_property
	def token_to_idx(self) -> Dict[Token, int]:
		return {t: i for i, t in enumerate(self.tokens)}
	
	@cached_property
	def _trie(self) -> _Trie:
		return _Trie(self.config.mode, self.tokens)
	
	def encode(self, text: TokenSequence):
		if not self.config.mode.match(text):
			raise ValueError(f"cannot use type f{type(text).__name__} on {self.mode}-based vocab")
		out = []
		iter: Iterator[re.Match] = re.finditer(self.config.pattern, text)
		for match in iter:
			chunk = match.group(0)
			addn = []
			for tok in self._trie.tokenize(chunk):
				addn.append(self.token_to_idx[tok])
			out.extend(addn)
		return out
	
	def __hash__(self):
		h = 0
		for token in self.tokens:
			h ^= hash(token)
		return hash((self.config.mode, h))

	def decode(self, idxs: List[int]):
		return [self.tokens[idx] for idx in idxs]
	
	def to_state_dict(self) -> io.Savable:
		return {
			"metadata": {
				"type": "Vocabulary",
				"version": ""
			},
			"config": {
				"mode": self.config.mode.value,
				"vocab_size": self.config.vocab_size,
				"special_tokens": self.config.special_tokens_str,
				"pattern": self.config.pattern_str
			},
			"tokens": self.tokens if self.config.mode == TokenizationMode.STR else [
				base64.b64encode(token).decode('ascii') for token in self.tokens
			]
		}
	
	@classmethod
	def from_state_dict(cls, obj):
		cfg = obj["config"]
		mode = TokenizationMode(cfg["mode"])
		return cls(
			BPEConfig(
				mode=mode,
				vocab_size = cfg["vocab_size"],
				special_tokens_str=cfg["special_tokens"],
				pattern_str=cfg["pattern"]
			),
			obj["tokens"] if mode == TokenizationMode.STR else [
				base64.b64decode(token) for token in obj["tokens"]
			]
		)

# just has to store the word encodings + pair counts, handle merges
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
				else: # no breaks
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
						self.pair_counts[new_pair] += count
						self.pair_to_words[new_pair].add(word_idx)
						updated.add(new_pair)
					if i + 2 < len(tokens):
						old_pair, new_pair = (t2, tokens[i+2]), (t3, tokens[i+2])
						self.pair_counts[old_pair] -= count
						self.pair_counts[new_pair] += count
						self.pair_to_words[new_pair].add(word_idx)
						updated.add(new_pair)
					tokens[i] = t3
					tokens.pop(i+1)
				else: i += 1
		self.pair_counts[pair] = 0
		return list(updated)

def _stream_chunks(
	data_handle: Union[TextIOBase, BufferedIOBase],
	pattern: Pattern,
	mode: TokenizationMode,
	chunk_size: int,
	verbose: bool = False,
) -> Counter:
	pbar = None
	if verbose:
		from tqdm import tqdm
		data_handle.seek(0, 2)
		total_size = data_handle.tell()
		data_handle.seek(0)
		pbar = tqdm(total=total_size, unit='B', unit_scale=True, desc="Counting words")
	counts = Counter()
	remainder = "" if mode == TokenizationMode.STR else b""
	while True:
		incoming_chunk = data_handle.read(chunk_size)
		if not incoming_chunk:
			if remainder:
				for m in pattern.finditer(remainder):
					counts[m.group()] += 1
			break

		if pbar:
			pbar.update(len(incoming_chunk))
		
		chunk = remainder + incoming_chunk
		last_end = 0
		last_match = None
		for m in pattern.finditer(chunk):
			if last_match is not None:
				counts[last_match.group()] += 1
			last_match = m
		
		if last_match is None:
			remainder = chunk
			continue

		remainder = chunk[last_match.start():]
		counts[last_match.group()] += 1
		
	if pbar:
		pbar.close()
	
	return counts

def create_bpe(
	data_handle: Union[TextIOBase, BufferedIOBase],
	vocab_size: int, 
	mode: Optional[TokenizationMode] = TokenizationMode.STR,
	splitting_pattern: Optional[str] = DEFAULT_PATTERN,
	special_tokens: Optional[List[str]] = None,
	verbose: Optional[bool] = False,
	read_chunk_size: int = 10_000_000
):
	is_handle_binary = isinstance(data_handle, (RawIOBase, BufferedIOBase))
	if mode == TokenizationMode.BYTES:
		assert is_handle_binary, \
			"training via bytes, but file was not opened in binary mode"
	else:
		assert not is_handle_binary, \
			"training via text, but file was opened in binary mode"
	config = BPEConfig(
		mode, vocab_size, special_tokens or [],
		splitting_pattern
	)
	pattern = config.pattern
	special_tokens = config.special_tokens

	if verbose:
		from tqdm import tqdm
		print("counting word frequencies...")

	counts = _stream_chunks(data_handle, pattern, mode, read_chunk_size, verbose=verbose)
	
	if verbose:
		print(f"found {len(counts)} unique words")

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

	if verbose:
		print(f"base vocabulary size: {len(vocab)}")

	corpus = _TokenMergeTracker(counts, token_to_idx, special_tokens)
	heap = [(-count, pair) for pair, count in corpus.pair_counts.items()]
	heapq.heapify(heap)

	pbar = None
	if verbose:
		pbar = tqdm(total=vocab_size, initial=len(vocab), desc="merging tokens")

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
		updated = corpus.merge(pair, t3)
		for updated_pair in updated:
			c = corpus.pair_counts.get(updated_pair, 0)
			if c > 0:
				heapq.heappush(heap, 
					(-c, updated_pair)
				)
		if pbar:
			pbar.update(1)
	
	if pbar:
		pbar.close()

	return Vocabulary(config, vocab)