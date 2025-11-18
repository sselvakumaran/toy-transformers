The primary bottleneck in your current implementation is that you are running the Linked List BPE algorithm on the raw text stream ($O(N)$ where $N$ is 1GB).

The "State of the Art" optimization used by libraries like SentencePiece, HuggingFace Tokenizers, and OpenAI is **Pre-tokenization with Weighted Frequency**.

Instead of merging "t" and "h" into "th" a million times across the entire text, you:
1.  Split the text into words using your regex (Pre-tokenization).
2.  Count the unique words (e.g., "the": 1,000,000).
3.  Run the Linked List algorithm on the **unique words only**, but give each node a **weight** equal to the word's frequency.

This reduces the problem size from the length of the text (~1,000,000,000 items) to the size of the vocabulary (~50,000 - 200,000 items), effectively making it thousands of times faster.

Here is the optimized implementation. I have modified `_TokenList` to accept `weights` and updated `create_tokenizer` to aggregate the data first.

```python
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
	# Added weights argument to handle frequency aggregation
	def __init__(self, text: np.ndarray, weights: np.ndarray, boundaries: Optional[Set[int]] = None, verbose=False):
		N = len(text)
		self.N = N
		self.tokens = np.copy(text).astype(np.int16, copy=False)
		self.weights = weights # Store weights for statistics
		self.prev = np.arange(-1, N - 1)
		self.next = np.arange(1, N + 1)
		self.boundaries = boundaries or set() 
		self.pairs = defaultdict(set)
		self.counts = defaultdict(int)
		
		iter_range = range(N - 1)
		if verbose:
			from tqdm import tqdm
			iter_range = tqdm(iter_range, desc="initializing tokens")
			
		for i in iter_range:
			if i in self.boundaries:
				continue
			pair = (int(text[i]), int(text[i+1]))
			self.pairs[pair].add(int(i))
			# KEY CHANGE: Add weight instead of 1
			self.counts[pair] += self.weights[i] 
	
	def get_key_list(self):
		return [(-count, pair) for pair, count in self.counts.items()]
	
	def get_count(self, pair):
		return self.counts[pair]
	
	def _nullify_index(self, i):
		self.tokens[i] = -1
		self.next[i] = self.N
		self.prev[i] = -1
		# We do not nullify weights, as the index 'i' might be reused 
		# as the head of the merged token, preserving the word's weight.
	
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
			
			# Perform merge
			self.tokens[i] = t_new
			self.next[i] = k
			self._nullify_index(j)
			
			if j_was_boundary:
				self.boundaries.remove(j)
				self.boundaries.add(i)

			# Update Neighbors
			# Note: We pass 'i' to update_count because 'i' holds the weight 
			# for this specific occurrence (word).
			
			if k != self.N:
				self.prev[k] = i
			
			if self.prev[i] != -1:
				p = self.prev[i]
				if p not in self.boundaries:
					pair = (self.tokens[p], t1)
					new_pair = (self.tokens[p], t_new)
					# The weight comes from 'p', which is the start of the pair (p, i)
					modified = self.update_count(p, pair, new_pair)
					if modified:
						modified_pairs.add(modified)
			
			if k != self.N:
				if not j_was_boundary:
					pair = (t2, self.tokens[k])
					new_pair = (t_new, self.tokens[k])
					# The weight comes from 'i' (formerly t1, now t_new), 
					# which is the start of the pair (i, k)
					modified = self.update_count(i, pair, new_pair)
					if modified:
						modified_pairs.add(modified)
		return modified_pairs
	
	def update_count(self, idx_start, old_pair, new_pair):
		old_pair = (int(old_pair[0]), int(old_pair[1]))
		new_pair = (int(new_pair[0]), int(new_pair[1]))
		
		weight = self.weights[idx_start]

		if old_pair in self.counts:
			self.counts[old_pair] -= weight
			self.pairs[old_pair].discard(idx_start)
			if self.counts[old_pair] <= 0:
				self.counts.pop(old_pair, None)
				self.pairs.pop(old_pair, None)
			
		self.counts[new_pair] += weight
		self.pairs[new_pair].add(int(idx_start))
		
		# Return the pair to update heap priority if it exists or is new
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
	# Default pattern handles contractions, words, punctuation, and whitespace
	regex_pattern = pattern or r"(\w+'\w+)| ?(\w+)|([^\w\s])|(\s+(?!\S))|(\s+)"
	
	# 1. Pre-tokenization and Aggregation
	# Instead of processing 1GB of text, we process ~50k-100k unique words.
	# We use findall to get chunks. (finditer is better for RAM, but Counter needs hashables)
	# To save RAM on 1GB strings, we iterate.
	if verbose:
		print("Pre-tokenizing data...")
	
	word_counts = Counter()
	# Iterate over regex matches to build the frequency map
	for match in re.finditer(regex_pattern, data):
		word_counts[match.group(0)] += 1
	
	if verbose:
		print(f"Found {len(word_counts)} unique words/tokens.")

	# 2. Prepare Vocabulary
	base_tokens = set()
	for word in word_counts.keys():
		base_tokens.update(word)
	base_tokens = sorted(list(base_tokens))
	
	predefined_tokens = predefined or []
	base_tokens = list(filter(lambda x: x not in base_tokens, predefined_tokens)) + base_tokens

	token_set, id_to_token, token_to_id = _create_td(base_tokens)
	
	# 3. Flatten Unique Words into a Weighted Stream
	# We construct a "text" that is just unique words separated by boundaries.
	# But we also construct a "weights" array that maps to their frequency.
	flat_tokens = []
	flat_weights = []
	boundaries = set()
	
	current_idx = 0
	for word, count in word_counts.items():
		word_indices = [token_to_id[c] for c in word]
		flat_tokens.extend(word_indices)
		# Each character in this specific word instance carries the weight of the word count
		flat_weights.extend([count] * len(word_indices))
		
		current_idx += len(word_indices)
		# Mark the end of this word as a boundary so we don't merge across different unique words
		boundaries.add(current_idx - 1)
	
	# Convert to numpy for the TokenList
	text_arr = np.array(flat_tokens, dtype=np.int16)
	weights_arr = np.array(flat_weights, dtype=np.int32) # int32 to hold large counts

	tokens = _TokenList(text_arr, weights_arr, boundaries=boundaries, verbose=verbose)
	
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
		
		# merge in the condensed weighted list
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

def get_encoder(td: TokenDictionary) -> Callable:
	trie: _Trie = _Trie.create_from_set(td.token_set)
	return lambda s: list(map(td.token_to_idx.__getitem__, trie.tokenize(s)))

def get_decoder(td: TokenDictionary) -> Callable:
	return lambda b: list(map(td.idx_to_token.__getitem__, b))
```

### Why this is faster:

1.  **Data Reduction:**
    *   **Old way:** Input string `("the " * 1000)` created a Linked List of length 4000.
    *   **New way:** Input string `("the " * 1000)` creates a Linked List of length 4 (just one instance of "the ") with a `weights` array of `[1000, 1000, 1000, 1000]`.
2.  **Efficient Merging:**
    *   Merging `t+h` -> `th` happens **once** in the new `_TokenList`, but `counts` increases by 1000 immediately.
    *   In the old version, you had to iterate 1000 times to update links and neighbors.
3.  **Memory:**
    *   `Counter` ensures we only store unique words. Even for 1GB of English text, the number of unique words rarely exceeds a few hundred thousand.

### Note on Porting to C++
When you port this to C++, this structure is ideal. You can map `_TokenList` to a `std::vector<int> tokens` and `std::vector<int> weights` alongside a `std::vector<int> next/prev`. This allows for extremely cache-efficient merges compared to pointer-based linked lists.