from typing import Callable
import torch
from collections import namedtuple

TokenDictionary = namedtuple('TokenDictionary', 
  ['token_set', 'idx_to_token', 'token_to_idx']
)

def create_character_tokenizer(data: str) -> TokenDictionary:
  token_set = sorted(set(data))
  idx_to_token = {i: t for i, t in enumerate(token_set)}
  token_to_idx = {t: i for i, t in enumerate(token_set)}
  return TokenDictionary(token_set, idx_to_token, token_to_idx)

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

import heapq
from collections import defaultdict

def create_tokenizer(data: str, num_tokens: int) -> TokenDictionary:
  token_set = sorted(set(data))
  if num_tokens <= len(token_set):
    return create_character_tokenizer(data)
  
  pair_counter = defaultdict(list)
  for i in range(len(data) - 1):
    t1, t2 = data[i], data[i+1]
    pair_counter[(t1, t2)] += i
  heap = [(-len(v), k) for k, v in pair_counter.items()]
  pass

def get_encoder(td: TokenDictionary) -> Callable:
  pass

def get_decoder(td: TokenDictionary) -> Callable:
  pass