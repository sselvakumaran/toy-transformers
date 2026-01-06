from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class DataConfig:
	"""Data configuration including model input dimensions.

	This config holds all data-related parameters including dimensions
	that the model needs (vocab_size, block_size) and data loading settings.
	"""
	vocab_size: int
	block_size: int
	batch_size: int
	num_workers: int = 0
	pin_memory: bool = False
	drop_last: bool = True
	shuffle: bool = True


@dataclass
class TrainConfig:
	"""Training state and runtime configuration.

	Unlike DataConfig, this is mutable because epoch, step, and
	batches_completed are updated during training.
	"""
	device: str
	seed: int
	max_epochs: Optional[int] = None
	epoch: int = 0
	step: int = 0
	batches_completed: int = 0
