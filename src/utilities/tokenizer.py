from typing import Dict, Callable, Set, Optional, List
import heapq
from collections import Counter, namedtuple, defaultdict
import numpy as np
import re

TokenDictionary = namedtuple('TokenDictionary', 
	['token_set', 'idx_to_token', 'token_to_idx']
)

class _LazyHeap():
	def __init__(self,
		is_valid: Callable,
		update: Callable,
		heap = None,
	):
		self.heap = heap if heap else []
		heapq.heapify(self.heap)
		self.is_valid = is_valid
		self.update = update
	
	def __bool__(self):
		return len(self.heap) > 0

	def __len__(self):
		return len(self.heap)

	def pop(self):
		item = heapq.heappop(self.heap)
		while not self.is_valid(*item):
			new_item = self.update(*item)
			item = heapq.heappushpop(self.heap, new_item)
		return item
		
	def push(self, item):
		heapq.heappush(self.heap, item)

class _TokenList:
	def __init__(self, chunks: np.ndarray, weights: np.ndarray, boundaries: Optional[Set[int]] = None, verbose=False):
		N = len(chunks)
		self.N = N
		self.tokens = np.copy(chunks).astype(np.int16, copy=False)
		self.weights = weights
		self.prev = np.arange(-1, N - 1)
		self.next = np.arange(1, N + 1)
		self.boundaries = boundaries or set() 
		self.pairs = defaultdict(set)
		self.counts = defaultdict(int)

		iter = range(N - 1)
		if verbose:
			from tqdm import tqdm
			iter = tqdm(iter, desc="initializing tokens")
		
		for i in iter:
			if i in self.boundaries: # boundaries -> blocks (i, i + 1) from merging
				continue
			pair = (int(chunks[i]), int(chunks[i+1]))
			self.pairs[pair].add(i)
			self.counts[pair] += self.weights[i]
	
	def get_key_list(self):
		return [(-count, pair) for pair, count in self.counts.items()]
	
	def get_count(self, pair):
		return self.counts[pair]
	
	def _nullify_index(self, i):
		self.tokens[i] = -1
		self.next[i] = self.N
		self.prev[i] = -1
	
	# merge two tokens together and update counts + positions
	def merge(self, t1, t2, t_new):
		old_pair = (t1, t2)
		modified_pairs = set()

		if old_pair not in self.pairs:
			return modified_pairs
		
		pairs = self.pairs.pop(old_pair)
		self.counts.pop(old_pair)

		for i in pairs:
			if self.tokens[i] != t1 or i in self.boundaries:
				continue
			j = self.next[i]
			if j == self.N or self.tokens[j] != t2:
				continue
			j_was_boundary = (j in self.boundaries)
			k = self.next[j]
			self.tokens[i] = t_new
			self.next[i] = k
			self._nullify_index(j)
			if j_was_boundary:
				self.boundaries.remove(j)
				self.boundaries.add(i)

			if k != self.N:
				self.prev[k] = i
			if self.prev[i] != -1:
				p = self.prev[i]
				if p not in self.boundaries:
					pair = (self.tokens[p], t1)
					new_pair = (self.tokens[p], t_new)
					modified = self.update_count(p, p, pair, new_pair)
					if modified:
						modified_pairs.add(modified)
			if k != self.N:
				if not j_was_boundary:
					pair = (t2, self.tokens[k])
					new_pair = (t_new, self.tokens[k])
					modified = self.update_count(j, i, pair, new_pair)
					if modified:
						modified_pairs.add(modified)
		return modified_pairs
	
	def update_count(self, old_i, new_i, old_pair, new_pair):
		old_pair = (int(old_pair[0]), int(old_pair[1]))
		new_pair = (int(new_pair[0]), int(new_pair[1]))
		if old_pair in self.counts:
			self.counts[old_pair] -= 1
			self.pairs[old_pair].discard(old_i)
			if self.counts[old_pair] <= 0:
				self.counts.pop(old_pair)
				self.pairs.pop(old_pair)
			
		self.counts[new_pair] += 1
		self.pairs[new_pair].add(int(new_i))
		if self.counts[new_pair] == 1:
			return new_pair

def _create_td(token_set: list) -> TokenDictionary:
	idx_to_token = {i: t for i, t in enumerate(token_set)}
	token_to_idx = {t: i for i, t in enumerate(token_set)}
	return TokenDictionary(token_set, idx_to_token, token_to_idx)

def create_tokenizer(
	data: str, 
	num_tokens: int, 
	pattern: Optional[str] = None, 
	predefined: Optional[list[str]] = None,
	verbose: bool = False
):
	pattern = re.compile(
		pattern or r"(\w+'\w+)| ?(\w+)|([^\w\s])|(\s+(?!\S))|(\s+)"
	)

	if verbose:
		from tqdm import tqdm

	word_counts = Counter()

	iter = re.finditer(pattern, data)
	if verbose:
		iter = tqdm(iter, desc="pre-tokenizing data", unit="chunks")
	for match in iter:
		word_counts[match.group(0)] += 1
	
	if verbose: print(f"found {len(word_counts)} unique words")

	base_tokens = set()
	for word in word_counts.keys():
		base_tokens.update(word)
	base_tokens = sorted(list(base_tokens))

	predefined_tokens = ["<UNK>", *predefined] if predefined else ["<UNK>"]
	base_tokens = list(filter(lambda x: x not in base_tokens, predefined_tokens)) + base_tokens

	token_set, id_to_token, token_to_id = _create_td(base_tokens)

	chunks_lst = []
	weights = []
	boundaries = set()

	curr_idx = 0
	for word, count in word_counts.items():
		w_indices = [token_to_id[c] for c in word]
		w_len = len(w_indices)
		chunks_lst.extend(w_indices)
		weights.extend([count] * w_len)
		curr_idx += w_len
		boundaries.add(curr_idx - 1)

	chunks = np.array(chunks_lst, dtype=np.int16)
	weights = np.array(weights, dtype=np.int32)

	tokens = _TokenList(chunks, weights, boundaries=boundaries, verbose=verbose)
	
	heap = _LazyHeap(
		is_valid = lambda neg_c, pair: tokens.get_count(pair) == -neg_c,
		update = lambda _, pair: (-tokens.get_count(pair), pair),
		heap = tokens.get_key_list()
	)

	token_count = len(token_set)

	if verbose:
		from tqdm import tqdm
		pbar = tqdm(
			total=num_tokens, 
			initial=token_count, 
			unit="token",
			desc="merging tokens"
		)

	while token_count < num_tokens and heap:
		_, (t1, t2) = heap.pop()
		new_token = token_count
		new_token_str = id_to_token[t1] + id_to_token[t2]
		token_set.append(new_token_str)
		id_to_token[new_token] = new_token_str
		token_to_id[new_token_str] = new_token

		modified = tokens.merge(t1, t2, new_token)
		for pair in modified:
			heap.push((-tokens.get_count(pair), pair))

		token_count += 1
		if verbose:
			pbar.update(1)
		
	return TokenDictionary(token_set, id_to_token, token_to_id)

class _TrieNode:
	def __init__(self):
		self.children: Dict[str, _TrieNode] = {}
		self.is_token = False

class _Trie:
	def __init__(self):
		self.root = _TrieNode()
	
	def insert(self, token: str):
		node = self.root
		for c in token:
			if c not in node.children:
				node.children[c] = _TrieNode()
			node = node.children[c]
		node.is_token = True
	
	@staticmethod
	def create_from_set(tokens: list):
		out = _Trie()
		for token in tokens:
			out.insert(token)
		return out
	
	def strtok(self, text, start) -> str:
		node = self.root
		s = ""
		longest_s = None
		for i in range(start, len(text)):
			c = text[i]
			if c in node.children:
				node = node.children[c]
				s += c
				if node.is_token:
					longest_s = s
			else:
				break
		if longest_s is None:
			return "<UNK>", i + 1
		return longest_s, start + len(longest_s)
	
	def tokenize(self, s: str):
		tokens = []
		i = 0

		while i < len(s):
			tok, j = self.strtok(s, i)
			tokens.append(tok)
			i = j
		return tokens

def reduce_token_dictionary(td: TokenDictionary, text: str) -> TokenDictionary:
	trie: _Trie = _Trie.create_from_set(td.token_set)
	new_token_set = sorted(set(trie.tokenize(text)))
	return _create_td(new_token_set)

def get_encoder(td: TokenDictionary, pattern: Optional[str] = None, verbose=False) -> Callable:
	trie: _Trie = _Trie.create_from_set(td.token_set)
	pattern = re.compile(
		pattern or r"(\w+'\w+)| ?(\w+)|([^\w\s])|(\s+(?!\S))|(\s+)"
	)
	cache: Dict[str, List[int]] = dict()

	def encode_chunk(chunk: str) -> List[int]:
		return list(map(td.token_to_idx.__getitem__, trie.tokenize(chunk)))
	
	def encode(text: str) -> List[int]:
		out = []
		iter = re.finditer(pattern, text)
		if verbose:
			from tqdm import tqdm
			iter = tqdm(iter, desc="encoding chunks", unit="chunks")
		for match in iter:
			chunk = match.group(0)
			addn = []
			if chunk in cache:
				addn = cache[chunk]
			else:
				addn = encode_chunk(chunk)
				cache[chunk] = addn
			out.extend(addn)
		return out
	return encode

def get_decoder(td: TokenDictionary) -> Callable:
	return lambda b: list(map(td.idx_to_token.__getitem__, b))