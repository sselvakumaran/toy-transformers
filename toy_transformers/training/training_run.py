from dataclasses import dataclass, asdict, is_dataclass
from typing import Type, Any, Optional, List, Dict, Tuple, Union
import torch.nn as nn
from datetime import datetime, timezone
import importlib

from toy_transformers.utilities import io
from toy_transformers.utilities.io import TorchStateDictRef, TrainingLogRef
from toy_transformers.utilities.version import get_obj_metadata, get_git_version, get_pytorch_version
from toy_transformers.training.optimizer import Optimizer, OptimizerConfig, AdamWConfig, SGDConfig, NoSchedulerConfig, ReduceLROnPlateauConfig, WarmupCosineConfig, CosineOnlyConfig
from toy_transformers.data.dataset import create_dataloader, skip_batches, compute_dataset_hash

# TODO: fix metadata (save time only when loading)


class TrainingRun:
	def __init__(
		self,
		# note: providing TYPES not the actual classes
		model_class: Type[nn.Module],
		config_class: Type,
		model_config: any, # must be dataclass  
		optimizer_config: OptimizerConfig,
		base_seed: int,
		dataset: Any, # only the hash is stored here
		block_size: int,
		batch_size: int,
		vocab_size: Optional[int] = None,
		num_workers: int = 0, # for data loader
		pin_memory: bool = False,
		drop_last: bool = True,
		epoch: int = 0,
		step: int = 0,
		batches_completed: int = 0,
		logs: Optional[List[Dict[str, Union[int, float]]]] = None,
	):
		# Store class references
		self.model_class = model_class
		self.config_class = config_class

		self.model_config = model_config
		self.optimizer_config = optimizer_config
		self.base_seed = base_seed
		self.dataset_hash = compute_dataset_hash(dataset) if dataset is not None else None
		self.vocab_size = vocab_size
		self.block_size = block_size
		self.batch_size = batch_size
		self.num_workers = num_workers
		self.pin_memory = pin_memory
		self.drop_last = drop_last
		self.epoch = epoch
		self.step = step
		self.batches_completed = batches_completed
		self.logs: List[Dict[str, Union[int, float]]] = logs if logs is not None else []
		self.git_hash = get_git_version()
		self.created_at = datetime.now(timezone.utc).isoformat()

	def log_step(self, **metrics: Union[int, float]):
		log_entry = {
			'epoch': self.epoch,
			'step': self.step,
			**metrics
		}
		self.logs.append(log_entry)

	def get_epoch_seed(self, epoch: Optional[int] = None) -> int:
		if epoch is None:
			epoch = self.epoch
		return self.base_seed + epoch

	def create_dataloader(self, dataset: Any, epoch: Optional[int] = None):
		if epoch is None:
			epoch = self.epoch

		epoch_seed = self.get_epoch_seed(epoch)
		loader = create_dataloader(
			dataset=dataset,
			block_size=self.block_size,
			batch_size=self.batch_size,
			shuffle=True,
			seed=epoch_seed,
			num_workers=self.num_workers,
			pin_memory=self.pin_memory,
			drop_last=self.drop_last,
			batches_completed=0
		)
		if self.batches_completed > 0:
			return skip_batches(loader, self.batches_completed)
		else:
			return iter(loader)

	def verify_dataset(self, dataset: Any):
		computed_hash = compute_dataset_hash(dataset)
		assert computed_hash == self.dataset_hash, \
			f"Dataset mismatch! Expected hash {self.dataset_hash}, got {computed_hash}"

	def to_state_dict(self, model: nn.Module, optimizer: Optimizer) -> io.Savable:
		if is_dataclass(self.model_config):
			model_config_dict = asdict(self.model_config)
		else:
			model_config_dict = self.model_config.__dict__
		log_rows = self._logs_to_csv()

		return {
			"metadata": {
				**get_obj_metadata(self, include_timestamp=True, include_pytorch_version=True),
				"created_at": self.created_at,
			},

			# Model versioning
			"model_class_name": self.model_class.__name__,
			"model_class_module": self.model_class.__module__,
			"config_class_name": self.config_class.__name__,
			"config_class_module": self.config_class.__module__,
			"model_config": model_config_dict,
			"vocab_size": self.vocab_size,
			"optimizer_config": self._serialize_optimizer_config(),
			"block_size": self.block_size,
			"batch_size": self.batch_size,
			"num_workers": self.num_workers,
			"pin_memory": self.pin_memory,
			"drop_last": self.drop_last,
			"epoch": self.epoch,
			"step": self.step,
			"batches_completed": self.batches_completed,
			"base_seed": self.base_seed,
			"dataset_hash": self.dataset_hash,
			"model_weights": TorchStateDictRef("model_weights", model.state_dict()),
			"optimizer_state": TorchStateDictRef("optimizer_state", optimizer.state_dict()),
			"logs": TrainingLogRef("training_logs", log_rows) if log_rows else None,
		}

	def _serialize_optimizer_config(self) -> dict:
		if is_dataclass(self.optimizer_config.optimizer_params):
			optimizer_params = asdict(self.optimizer_config.optimizer_params)
			optimizer_params = self._convert_tuples_to_lists(optimizer_params)
		else:
			optimizer_params = self.optimizer_config.optimizer_params
		if is_dataclass(self.optimizer_config.scheduler):
			scheduler_config = asdict(self.optimizer_config.scheduler)
			scheduler_config = self._convert_tuples_to_lists(scheduler_config)
		else:
			scheduler_config = self.optimizer_config.scheduler

		return {
			"optimizer_type": self.optimizer_config.optimizer_type,
			"optimizer_params": optimizer_params,
			"scheduler": scheduler_config,
		}

	@staticmethod
	def _convert_tuples_to_lists(obj):
		if isinstance(obj, tuple):
			return list(obj)
		elif isinstance(obj, dict):
			return {k: TrainingRun._convert_tuples_to_lists(v) for k, v in obj.items()}
		elif isinstance(obj, list):
			return [TrainingRun._convert_tuples_to_lists(item) for item in obj]
		else:
			return obj

	def _logs_to_csv(self) -> List[List[Union[str, int, float]]]:
		if not self.logs:
			return []
		keys = set()
		for log in self.logs:
			keys.update(log.keys())
		keys = sorted(keys)
		rows = [keys]
		for log in self.logs:
			row = [log.get(key, '') for key in keys]
			rows.append(row)

		return rows

	@classmethod
	def from_state_dict(cls, obj: dict) -> Tuple['TrainingRun', dict, dict]:
		model_module = importlib.import_module(obj["model_class_module"])
		model_class = getattr(model_module, obj["model_class_name"])
		config_module = importlib.import_module(obj["config_class_module"])
		config_class = getattr(config_module, obj["config_class_name"])
		model_config = config_class(**obj["model_config"])
		optimizer_config = cls._deserialize_optimizer_config(obj["optimizer_config"])
		logs = cls._csv_to_logs(obj["logs"].logs) if obj.get("logs") else []
		training_run = cls(
			model_class=model_class,
			config_class=config_class,
			model_config=model_config,
			optimizer_config=optimizer_config,
			base_seed=obj["base_seed"],
			dataset=None,
			block_size=obj["block_size"],
			batch_size=obj["batch_size"],
			vocab_size=obj.get("vocab_size"),
			num_workers=obj.get("num_workers", 0),
			pin_memory=obj.get("pin_memory", False),
			drop_last=obj.get("drop_last", True),
			epoch=obj["epoch"],
			step=obj["step"],
			batches_completed=obj["batches_completed"],
			logs=logs,
		)
		training_run.dataset_hash = obj["dataset_hash"]
		training_run.git_hash = obj["metadata"]["git_hash"]
		training_run.created_at = obj["metadata"]["created_at"]
		model_state_dict = obj["model_weights"].state_dict
		optimizer_state_dict = obj["optimizer_state"].state_dict

		return training_run, model_state_dict, optimizer_state_dict

	@classmethod
	def _deserialize_optimizer_config(cls, obj: dict) -> OptimizerConfig:
		match obj["optimizer_type"]:
			case "adamw":
				params = obj["optimizer_params"].copy()
				if "betas" in params and isinstance(params["betas"], list):
					params["betas"] = tuple(params["betas"])
				optimizer_params = AdamWConfig(**params)
			case "sgd":
				optimizer_params = SGDConfig(**obj["optimizer_params"])
			case _:
				raise ValueError(f"Unknown optimizer type: {obj['optimizer_type']}")
		scheduler_dict = obj["scheduler"]
		match scheduler_dict.get("type"):
			case "none":
				scheduler = NoSchedulerConfig()
			case "reduce_on_plateau":
				scheduler = ReduceLROnPlateauConfig(**scheduler_dict)
			case "warmup_cosine":
				scheduler = WarmupCosineConfig(**scheduler_dict)
			case "cosine":
				scheduler = CosineOnlyConfig(**scheduler_dict)
			case _:
				raise ValueError(f"Unknown scheduler type: {scheduler_dict.get('type')}")

		return OptimizerConfig(
			optimizer_type=obj["optimizer_type"],
			optimizer_params=optimizer_params,
			scheduler=scheduler
		)

	@classmethod
	def _csv_to_logs(cls, csv_rows: List[List]) -> List[Dict[str, Union[int, float]]]:
		if not csv_rows:
			return []

		headers = csv_rows[0]
		logs = []

		for row in csv_rows[1:]:
			log = {}
			for i, key in enumerate(headers):
				if i < len(row) and row[i] != '':
					value = row[i]
					try:
						if '.' in str(value):
							log[key] = float(value)
						else:
							log[key] = int(value)
					except (ValueError, TypeError):
						log[key] = value
			logs.append(log)

		return logs

	def create_model(self) -> nn.Module:
		if self.vocab_size is not None:
			return self.model_class(vocab_size=self.vocab_size, config=self.model_config)
		else:
			return self.model_class(config=self.model_config)

	def create_optimizer(self, model: nn.Module) -> Optimizer:
		return Optimizer(config=self.optimizer_config, model_parameters=model.parameters())
