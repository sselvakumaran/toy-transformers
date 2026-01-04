from dataclasses import dataclass
from typing import Tuple

@dataclass
class AdamWTrainingConfig:
  lr: float
  weight_decay: float
  betas: Tuple[float, float]

