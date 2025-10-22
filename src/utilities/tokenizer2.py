# BPE CHANGES
# 1. instead of tokens / strings use ints 
# 2. use numpy
# 3. use Counter / counter.top_k(1) instead
# 4. create dict pair -> set(indicies), only update neighboring pair counts
# 5. updating neighboring pair counts:
#      using dict from (4), get all indicies, search left and right for neighboring tokens
#      (replace deleted tokens with -1 for convenience, occasionally clear all -1s)
#      update counter and 

from typing import Dict, Callable
import heapq
from collections import Counter, namedtuple, defaultdict
import numpy as np

TokenDictionary = namedtuple('TokenDictionary', 
	['token_set', 'idx_to_token', 'token_to_idx']
)

class _LazyHeap():
	def __init__(self, 
		refresh: Callable,
		is_valid: Callable,
		heap = None,
		amortize_period = 50,
	):
		self.heap = heap if heap else []
		heapq.heapify(self.heap)

		self.refresh = refresh
		self.is_valid = is_valid
		self.count = 0
		self.limit = amortize_period
	
	def __bool__(self):
		return len(self.heap) > 0

	def __len__(self):
		return len(self.heap)

	def pop(self):
		item = heapq.heappop(self.heap)
		while not self.is_valid(*item):
			self.count += 1
			item = heapq.heappop(self.heap)

			if self.count >= self.limit:
				self.count = 0
				self.heap = self.refresh()
				heapq.heapify(self.heap)
		return item
		
	def push(self, item):
		heapq.heappush(self.heap, item)

class _LazyHeap2():
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
	def __init__(self, text: np.ndarray):
		N = len(text)
		self.N = N
		self.tokens = np.copy(text).astype(np.int16, copy=False)
		self.prev = np.arange(-1, N - 1)
		self.next = np.arange(1, N + 1)

		self.pairs = defaultdict(set)
		self.counts = defaultdict(int)
		for i in range(N - 1):
			pair = (int(text[i]), int(text[i+1]))
			self.pairs[pair].add(int(i))
			self.counts[pair] += 1
	
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
		pairs = self.pairs.pop(old_pair)
		self.counts.pop(old_pair)
		modified_pairs = set()
		for i in pairs:
			if self.tokens[i] != t1:
				continue
			j = self.next[i]
			if j == self.N or self.tokens[j] != t2:
				continue
			k = self.next[j]
			self.tokens[i] = t_new
			self.next[i] = k
			self._nullify_index(j)
			if k != self.N:
				self.prev[k] = i
			if self.prev[i] != -1:
				p = self.prev[i]
				pair = (self.tokens[p], t1)
				new_pair = (self.tokens[p], t_new)
				modified = self.update_count(p, p, pair, new_pair)
				if modified:
					modified_pairs.add(modified)
			if k != self.N:
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
			self.pairs[old_pair].remove(old_i)
			if self.counts[old_pair] <= 0:
				self.counts.pop(old_pair)
				self.pairs.pop(old_pair)
			
		self.counts[new_pair] += 1
		self.pairs[new_pair].add(int(new_i))
		if self.counts[new_pair] == 1:
			return new_pair

def create_td(token_set: list) -> TokenDictionary:
	idx_to_token = {i: t for i, t in enumerate(token_set)}
	token_to_idx = {t: i for i, t in enumerate(token_set)}
	return TokenDictionary(token_set, idx_to_token, token_to_idx)

def create_tokenizer(data: str, num_tokens: int):
	token_set, id_to_token, token_to_id = create_td(sorted(set(data)))
	text = np.array([token_to_id[token] for token in data], dtype=np.int16)

	tokens = _TokenList(text)
	heap = _LazyHeap2(
		is_valid = lambda neg_c, pair: tokens.get_count(pair) == -neg_c,
		update = lambda _, pair: (-tokens.get_count(pair), pair),
		heap = tokens.get_key_list()
	)

	token_count = len(token_set)
	while token_count < num_tokens and heap:
		_, (t1, t2) = heap.pop()
		new_token = token_count
		new_token_str = id_to_token[t1] + id_to_token[t2]
		token_set.append(new_token_str)
		id_to_token[new_token] = new_token_str
		token_to_id[new_token_str] = new_token
		# replace (a, b)
		modified = tokens.merge(t1, t2, new_token)
		for pair in modified:
			heap.push((-tokens.get_count(pair), pair))
		token_count += 1
	return TokenDictionary(token_set, id_to_token, token_to_id)

if __name__ == '__main__':
	data = "banana"
	td = create_tokenizer(data, num_tokens=10)