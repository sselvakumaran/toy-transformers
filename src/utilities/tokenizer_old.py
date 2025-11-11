from typing import Callable, Dict
from collections import namedtuple

TokenDictionary = namedtuple('TokenDictionary', 
  ['token_set', 'idx_to_token', 'token_to_idx']
)

def create_td(token_set: list) -> TokenDictionary:
  idx_to_token = {i: t for i, t in enumerate(token_set)}
  token_to_idx = {t: i for i, t in enumerate(token_set)}
  return TokenDictionary(token_set, idx_to_token, token_to_idx)

def create_character_tokenizer(data: str) -> TokenDictionary:
  token_set = sorted(set(data))
  return create_td(token_set)

def get_character_encoder(td: TokenDictionary) -> Callable:
  return lambda s: list(map(td.token_to_idx.__getitem__, s))

def get_character_decoder(td: TokenDictionary) -> Callable:
  return lambda s: list(map(td.idx_to_token.__getitem__, s))

# VARIABLE TOKENIZER
# uses byte-pair encoding (but unicode lol)

# need heap of pairs
# need map of token name -> all occurences of it in text

# 1. go through data first, update counts of pairs and all locations where it exists
# loop: pop off most used pair, go to all occurences of it, update pair list
# (will require eating tokens)

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
  
  def create_from_set(tokens: list):
    out = _Trie()
    for token in tokens:
      out.insert(token)
    return out
  
  def strtok(self, text, i) -> str:
    node = self.root
    s = ""
    longest_s = None
    for i in range(i, len(text)):
      c = text[i]
      if c in node.children:
        node = node.children[c]
        s += c
        if node.is_token:
          longest_s = s
      else:
        break
    return longest_s, i + len(longest_s)
  
  def tokenize(self, s: str):
    tokens = []
    i = 0
    while i < len(s):
      tok, j = self.strtok(s, i)
      tokens.append(tok)
      i = j
    return tokens

import heapq
from collections import defaultdict

def create_tokenizer(data: str, num_tokens: int, verbose=False) -> TokenDictionary:
  text = list(data)
  N = len(text)
  token_set = sorted(set(text))
  if num_tokens <= len(token_set):
    return create_character_tokenizer(text)
  
  pair_counter = defaultdict(int)
  for i in range(N - 1):
    pair_counter[(text[i], text[i+1])] += 1
  heap = [(-v, k) for k, v in pair_counter.items()]
  heapq.heapify(heap)
  needs_count_update = dict() # t2 tokens which count will be incorrect after updates
  if verbose:
    print("Adding: ", end="")
  while len(token_set) < num_tokens and heap:
    _, pair = heapq.heappop(heap)
    t1, t2 = pair

    if pair in needs_count_update:
      heapq.heappush(heap, (-needs_count_update.pop(pair), pair))
      continue
    if t1 in needs_count_update: # (t1,*) has count 0
      continue

    new_token = t1 + t2
    _update_text(t1, t2, text, N)
    new_token_marginal_counts = _get_marginal_pair_counts(new_token, text, N)
    for new_token_marg, c in new_token_marginal_counts.items():
      heapq.heappush(heap, (-c, (new_token, new_token_marg)))
    t2_token_marginal_counts = _get_marginal_pair_counts(t2, text, N)
    for t2_marg, c in t2_token_marginal_counts.items():
      needs_count_update[(t2, t2_marg)] = c
    needs_count_update[t2] = 0 # for values nonexistant in the set now
    if verbose:
      print(new_token, end=', ')
    token_set.append(new_token)

  return create_td(token_set)

def reduce_token_dictionary(td: TokenDictionary, text: str) -> TokenDictionary:
  trie: _Trie = _Trie.create_from_set(td.token_set)
  new_token_set = sorted(set(trie.tokenize(text)))
  return create_td(new_token_set)

def _update_text(t1, t2, text, N):
  i = 0
  while i < N:
    match text[i]:
      case int(x): # skip
        i += x
        continue
      case str(s) if s != t1:
        i += 1
        continue
    j = _get_next_processed_token(i + 1, text, N)
    if j >= N:
      break
    if text[j] != t2:
      i = j
      continue
    l = text[i+1] if isinstance(text[i+1], int) else 0
    r = text[j+1] if j+1 < N and isinstance(text[j+1], int) else 0
    text[i] = t1 + t2
    text[i+1] = l + r + 1
    text[j] = r + 1
    i += 1
  return None

def _get_marginal_pair_counts(target, text, N):
  counts = defaultdict(int)
  i = 0
  while i < N:
    i1 = _get_next_processed_token(i, text, N)
    i2 = _get_next_processed_token(i1 + 1, text, N)
    if i1 >= N or i2 >= N:
      return counts
    if text[i1] == target:
      counts[text[i2]] += 1
    i = i2
  return counts

def _get_next_processed_token(i, text, N):
  while i < N and isinstance(text[i], int):
    i += text[i]
  return i

def get_encoder(td: TokenDictionary) -> Callable:
  trie: _Trie = _Trie.create_from_set(td.token_set)
  return lambda s: list(map(td.token_to_idx, trie.tokenize(s)))

def get_decoder(td: TokenDictionary) -> Callable:
  return lambda s: list(map(td.idx_to_token.__getitem__, s))