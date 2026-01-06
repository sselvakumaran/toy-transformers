from torch.utils.data import Dataset, DataLoader
import torch
from typing import Optional
from toy_transformers.utilities.hashing import stable_hash


class TextDataset(Dataset):
	def __init__(self, data_tensor: torch.Tensor, block_size: int):
		self.data = data_tensor
		self.block_size = block_size
		self.n_chunks = (len(data_tensor) - 1) // block_size

	def __len__(self):
		return self.n_chunks

	def __getitem__(self, idx):
		chunk = self.data[idx:idx+self.block_size+1]
		x = chunk[:-1]
		y = chunk[1:]
		return x, y


def create_dataloader(
	dataset,
	block_size: Optional[int] = None,
	batch_size: int = 64,
	shuffle: bool = True,
	seed: Optional[int] = None,
	num_workers: int = 0,
	pin_memory: bool = False,
	drop_last: bool = True,
	batches_completed: int = 0,
	**kwargs
) -> DataLoader:
	from toy_transformers.utilities.reproducibility import (
		worker_init_fn,
		create_deterministic_generator
	)

	if hasattr(dataset, 'data') and hasattr(dataset, 'vocab'):
		if block_size is None:
			raise ValueError("block_size required when passing TokenizedData")
		dataset = TextDataset(dataset.data, block_size)

	generator = None
	if seed is not None and shuffle:
		generator = create_deterministic_generator(seed)

	worker_init = None
	if seed is not None and num_workers > 0:
		worker_init = lambda worker_id: worker_init_fn(worker_id, base_seed=seed)

	loader = DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=shuffle,
		num_workers=num_workers,
		pin_memory=pin_memory,
		drop_last=drop_last,
		generator=generator,
		worker_init_fn=worker_init,
		**kwargs
	)

	loader.batches_completed = batches_completed
	loader.seed = seed

	return loader


def get_dataloader_state(loader: DataLoader) -> dict:
	return {
		'seed': getattr(loader, 'seed', None),
		'batches_completed': getattr(loader, 'batches_completed', 0),
		'batch_size': loader.batch_size,
	}


def skip_batches(loader: DataLoader, num_batches: int):
	iterator = iter(loader)
	for _ in range(num_batches):
		try:
			next(iterator)
		except StopIteration:
			break
	return iterator


def create_dataloader_for_epoch(
	dataset,
	epoch: int,
	base_seed: int,
	block_size: Optional[int] = None,
	batch_size: int = 64,
	num_workers: int = 0,
	shuffle: bool = True,
	pin_memory: bool = False,
	drop_last: bool = True,
	**kwargs
) -> DataLoader:
	# kwargs includes some of these
	kwargs.pop('seed', None)
	kwargs.pop('batches_completed', None)

	epoch_seed = base_seed + epoch
	return create_dataloader(
		dataset=dataset,
		block_size=block_size,
		batch_size=batch_size,
		shuffle=shuffle,
		seed=epoch_seed,
		num_workers=num_workers,
		pin_memory=pin_memory,
		drop_last=drop_last,
		batches_completed=0,
		**kwargs
	)


def compute_dataset_hash(dataset) -> str:
	"""
	returns consistent dataset hash between runs
	note stable_hash handles large lists / tensors via sampling
	"""
	if hasattr(dataset, 'data'):
		data_tensor = dataset.data
	else:
		data_tensor = dataset

	# Use stable_hash which already handles tensor shape, dtype, and sampling
	return stable_hash(data_tensor)
