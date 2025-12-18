from typing import Dict, Iterator, List, Literal, Optional, Union, ByteString, Sequence, Pattern
from dataclasses import dataclass, field
from enum import Enum
from functools import cached_property
import re

from toy_transformers.utilities import io

Token = str | bytes
TokenSequence = str | bytes

UNK = "<UNK>"
ENCODING = 'utf-8'

DEFAULT_PATTERN = r"(\w+'\w+)| ?(\w+)|([^\w\s])|(\s+(?!\S))|(\s+)"

class TokenizationMode(Enum):
	STR = "str"
	BYTES = "bytes"

	_TYPE_MAP = {
		"str": str,
		"bytes": bytes
	}

	def match(self, tok) -> bool:
		return isinstance(tok, self._TYPE_MAP[self.value])

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
class BPEConfig():
	mode: TokenizationMode
	num_merges: int
	special_tokens: List[Token]
	pattern_str: Optional[str] = DEFAULT_PATTERN
	# ADD MORE
	# USE THIS COMPOSITELY TO VOCABULARY.

@dataclass
class Vocabulary():
	mode: TokenizationMode
	tokens: List[Token]
	pattern_str: Optional[str] = DEFAULT_PATTERN
	pattern: Optional[Pattern] = field(init=False)

	def __post_init__(self):
		typed_pattern_str = self.pattern_str
		if self.mode == TokenizationMode.STR: 
			typed_pattern_str = self.pattern_str.encode(ENCODING)
		self.pattern = re.compile(typed_pattern_str)
	
	@cached_property
	def token_to_idx(self) -> Dict[Token, int]:
		return {t: i for i, t in enumerate(self.tokens)}
	
	@cached_property
	def _trie(self) -> _Trie:
		return _Trie(self.mode, self.tokens)
	
	def encode(self, text: TokenSequence):
		if not self.mode.match(text):
			raise ValueError(f"cannot use type f{type(text).__name__} on {self.mode}-based vocab")
		out = []
		iter: Iterator[re.Match] = re.finditer(self.pattern, text)
		for match in iter:
			chunk = match.group(0)
			addn = []
			for tok in self._trie.tokenize(chunk):
				addn.append(self.token_to_idx[tok])
			out.extend(addn)
		return out

	def decode(self, idxs: List[int]):
		return [self.tokens[idx] for idx in idxs]
	
	def to_state_dict(self) -> io.Savable:
		return {
			"mode": self.mode,
			"pattern_str": self.pattern_str,
			"tokens": [
				tok if self.mode == TokenizationMode.STR else tok.decode('utf-8')
				for tok in self.tokens
			]
		}