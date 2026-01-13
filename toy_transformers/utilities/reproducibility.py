import torch
import random
import numpy as np
from typing import Optional
from toy_transformers.artifacts import stable_hash


def set_all_seeds(seed: int, deterministic: bool = True) -> None:
  """ 
  set seeds for python, numpy, pytorch
  deterministic true to enable deterministic algorithms in pytorch
  """
  random.seed(seed)

  np.random.seed(seed)

  torch.manual_seed(seed)

  if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # for multi-gpu

  if torch.backends.mps.is_available():
    torch.mps.manual_seed(seed)

  if deterministic:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # note: use_deterministic_algorithms can break some operations

def worker_init_fn(worker_id: int, base_seed: Optional[int] = None) -> None:
  """
  sets random seeds for DataLoader workers
  """
  if base_seed is None:
    worker_seed = torch.initial_seed() % 2**32
  else:
    # Use stable hash for consistent worker seeding across runs
    H = int(stable_hash("worker_init_fn"), 16) & 0xFFFFFFFF  # Limit to 32-bit
    worker_seed = (H + base_seed + worker_id) % 2**32

  random.seed(worker_seed)
  np.random.seed(worker_seed)


def create_deterministic_generator(seed: int) -> torch.Generator:
  generator = torch.Generator()
  generator.manual_seed(seed)
  return generator
