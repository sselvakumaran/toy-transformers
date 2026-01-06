from dataclasses import dataclass, asdict, is_dataclass
from typing import Type, Any, Optional, Tuple, Literal
from enum import Enum
import torch.nn as nn
from datetime import datetime, timezone
import warnings

from toy_transformers.utilities import io
from toy_transformers.utilities.io import TorchStateDictRef, MetricLogRef
from toy_transformers.utilities.version import get_obj_metadata, get_git_version
from toy_transformers.training.optimizer import Optimizer, OptimizerConfig, AdamWConfig, SGDConfig, NoSchedulerConfig, ReduceLROnPlateauConfig, WarmupCosineConfig, CosineOnlyConfig
from toy_transformers.training.configs import DataConfig, TrainConfig
from toy_transformers.data.dataset import create_dataloader, skip_batches, compute_dataset_hash


# Type alias for save modes
SaveMode = Literal['full', 'final', 'archived']


class _SaveModeLevel(Enum):
	"""Internal enum for save mode validation."""
	ARCHIVED = 0
	FINAL = 1
	FULL = 2


# Data-driven component specification for each save mode
_SAVE_MODE_COMPONENTS = {
	'full': {
		'model_weights': True,
		'optimizer_state': True,
		'logs': True,
	},
	'final': {
		'model_weights': True,
		'optimizer_state': False,
		'logs': True,
	},
	'archived': {
		'model_weights': False,
		'optimizer_state': False,
		'logs': True,
	}
}


class TrainingRun:
	"""Training run with support for multiple save modes.

	Save modes:
	- 'full': Complete state including optimizer (for resuming training)
	- 'final': Model weights + logs only (for inference/evaluation)
	- 'archived': Metadata + logs only (for record keeping)
	"""

	def __init__(
		self,
		model_class: Type[nn.Module],
		model_config: Any,  # GPTv1Config, GPTv2Config, etc.
		data_config: DataConfig,
		train_config: TrainConfig,
		optimizer_config: OptimizerConfig,
		dataset_hash: Optional[str] = None,
	):
		"""Initialize TrainingRun.

		Args:
			model_class: Model class (e.g., LanguageModel from gptv1)
			model_config: Model-specific config (GPTv1Config, GPTv2Config, etc.)
			data_config: Data configuration
			train_config: Training configuration
			optimizer_config: Optimizer configuration
			dataset_hash: Optional dataset hash for verification
		"""
		self.model_class = model_class
		self.model_config = model_config
		self.data_config = data_config
		self.train_config = train_config
		self.optimizer_config = optimizer_config
		self.dataset_hash = dataset_hash

		self.logs = MetricLogRef("training_logs")
		self.git_hash = get_git_version()
		self.created_at = datetime.now(timezone.utc).isoformat()
		self.save_mode = 'full'  # Track current save mode

	def log_step(self, **metrics):
		"""Log metrics for current step.

		Automatically includes epoch and step in metrics.
		"""
		self.logs.log(
			epoch=self.train_config.epoch,
			step=self.train_config.step,
			**metrics
		)

	def get_epoch_seed(self, epoch: Optional[int] = None) -> int:
		"""Get seed for specified epoch."""
		if epoch is None:
			epoch = self.train_config.epoch
		return self.train_config.seed + epoch

	def create_dataloader(self, dataset: Any, epoch: Optional[int] = None):
		"""Create dataloader for training.

		Uses DataConfig settings and TrainConfig state.
		"""
		if epoch is None:
			epoch = self.train_config.epoch

		epoch_seed = self.get_epoch_seed(epoch)
		loader = create_dataloader(
			dataset=dataset,
			block_size=self.data_config.block_size,
			batch_size=self.data_config.batch_size,
			shuffle=self.data_config.shuffle,
			seed=epoch_seed,
			num_workers=self.data_config.num_workers,
			pin_memory=self.data_config.pin_memory,
			drop_last=self.data_config.drop_last,
			batches_completed=0
		)

		if self.train_config.batches_completed > 0:
			return skip_batches(loader, self.train_config.batches_completed)
		else:
			return iter(loader)

	def verify_dataset(self, dataset: Any, strict: bool = True) -> bool:
		"""Verify dataset matches stored hash.

		Args:
			dataset: Dataset to verify
			strict: If True, raise error on mismatch. If False, return bool.

		Returns:
			True if match or no hash stored, False if mismatch (when strict=False)
		"""
		if self.dataset_hash is None:
			return True  # No hash stored, can't verify

		computed = compute_dataset_hash(dataset)
		matches = computed == self.dataset_hash

		if not matches and strict:
			raise ValueError(
				f"Dataset mismatch! Expected hash {self.dataset_hash}, got {computed}"
			)

		return matches

	def set_dataset_hash(self, dataset: Any):
		"""Compute and store dataset hash."""
		self.dataset_hash = compute_dataset_hash(dataset)

	def to_state_dict(self, model: nn.Module | None, optimizer: Optimizer | None,
	                  mode: SaveMode = 'full') -> io.Savable:
		"""Save training run with specified mode.

		Args:
			model: Model instance (required for 'full' and 'final' modes)
			optimizer: Optimizer instance (required for 'full' mode only)
			mode: Save mode - 'full', 'final', or 'archived'

		Returns:
			Savable state dict
		"""
		components = _SAVE_MODE_COMPONENTS[mode]

		# Validate required components
		if components['model_weights'] and model is None:
			raise ValueError(f"mode='{mode}' requires model parameter")
		if components['optimizer_state'] and optimizer is None:
			raise ValueError(f"mode='{mode}' requires optimizer parameter")

		# Serialize model config
		if is_dataclass(self.model_config):
			model_config_dict = asdict(self.model_config)
		else:
			model_config_dict = self.model_config.__dict__

		# Build state dict with metadata
		state = {
			"metadata": {
				**get_obj_metadata(self, include_timestamp=True, include_pytorch_version=True),
				"created_at": self.created_at,
				"save_mode": mode,
			},

			# Model versioning - use model_type instead of module/class names
			"model_type": self.model_class.model_type,
			"model_config": model_config_dict,
			"data_config": asdict(self.data_config),
			"train_config": asdict(self.train_config),
			"optimizer_config": self._serialize_optimizer_config(),
			"dataset_hash": self.dataset_hash,
		}

		# Add optional components based on mode
		if components['model_weights']:
			state["model_weights"] = TorchStateDictRef("model_weights", model.state_dict())

		if components['optimizer_state']:
			state["optimizer_state"] = TorchStateDictRef("optimizer_state", optimizer.state_dict())

		if components['logs'] and self.logs.data:
			state["logs"] = self.logs

		return state

	def save(self, path: str, model: nn.Module | None = None,
	         optimizer: Optimizer | None = None, mode: SaveMode = 'full'):
		"""Save training run to disk.

		Args:
			path: Directory path to save to
			model: Model instance (required for 'full' and 'final' modes)
			optimizer: Optimizer instance (required for 'full' mode only)
			mode: Save mode - 'full', 'final', or 'archived'
		"""
		state_dict = self.to_state_dict(model, optimizer, mode)
		io.save(state_dict, path)

	@classmethod
	def from_state_dict(cls, obj: dict, model_class: Type[nn.Module]) -> Tuple['TrainingRun', dict | None, dict | None]:
		"""Load TrainingRun from state dict.

		Args:
			obj: Loaded state dict
			model_class: Model class imported by user (e.g., GPTv1.LanguageModel)

		Returns:
			Tuple of (training_run, model_state_dict, optimizer_state_dict)
			model_state_dict and optimizer_state_dict can be None based on save mode

		Raises:
			Warning if model_type mismatch detected
		"""
		# Verify model_type matches
		stored_model_type = obj["model_type"]
		if model_class.model_type != stored_model_type:
			warnings.warn(
				f"Model type mismatch! Saved as '{stored_model_type}', "
				f"loading with '{model_class.model_type}'. Proceed with caution."
			)

		# Reconstruct configs from dicts
		# Get config class from model class
		if hasattr(model_class, 'config_class'):
			config_class = model_class.config_class
		else:
			# Fallback: try to infer from model module
			# For GPTv1, GPTv2, GPTv3 this should be GPTv1Config, GPTv2Config, GPTv3Config
			module = model_class.__module__
			config_name = model_class.__name__.replace('LanguageModel', 'Config')
			if config_name == 'Config':
				config_name = 'GPTv1Config'  # Fallback default
			import importlib
			mod = importlib.import_module(module)
			config_class = getattr(mod, config_name)

		model_config = config_class(**obj["model_config"])
		data_config = DataConfig(**obj["data_config"])
		train_config = TrainConfig(**obj["train_config"])
		optimizer_config = cls._deserialize_optimizer_config(obj["optimizer_config"])

		training_run = cls(
			model_class=model_class,
			model_config=model_config,
			data_config=data_config,
			train_config=train_config,
			optimizer_config=optimizer_config,
			dataset_hash=obj.get("dataset_hash"),
		)

		# Load logs if present
		training_run.logs = obj.get("logs", MetricLogRef("training_logs"))
		training_run.git_hash = obj["metadata"]["git_hash"]
		training_run.created_at = obj["metadata"]["created_at"]
		training_run.save_mode = obj["metadata"].get("save_mode", "full")

		# Extract state dicts (may be None based on save mode)
		model_state_dict = obj.get("model_weights").state_dict if "model_weights" in obj else None
		optimizer_state_dict = obj.get("optimizer_state").state_dict if "optimizer_state" in obj else None

		return training_run, model_state_dict, optimizer_state_dict

	@classmethod
	def load(cls, path: str, model_class: Type[nn.Module]) -> Tuple['TrainingRun', Optional[nn.Module], Optional[Optimizer]]:
		"""Load training run from disk.

		Args:
			path: Directory to load from
			model_class: Model class to use (e.g., GPTv1.LanguageModel) - imported by user

		Returns:
			Tuple of (training_run, model, optimizer)
			- model is None if save mode was 'archived'
			- optimizer is None if save mode was 'final' or 'archived'

		Example:
			from toy_transformers.models.gptv1 import LanguageModel
			run, model, optimizer = TrainingRun.load("path/to/checkpoint", LanguageModel)
		"""
		obj = io.load(path)
		training_run, model_state_dict, optimizer_state_dict = cls.from_state_dict(obj, model_class)

		# Create model if weights exist
		model = None
		if model_state_dict is not None:
			model = training_run.create_model()
			model.load_state_dict(model_state_dict)

		# Create optimizer if state exists
		optimizer = None
		if optimizer_state_dict is not None and model is not None:
			optimizer = training_run.create_optimizer(model)
			optimizer.load_state_dict(optimizer_state_dict)

		return training_run, model, optimizer

	def convert_save_mode(self, model: nn.Module | None, optimizer: Optimizer | None,
	                      target_mode: SaveMode) -> 'TrainingRun':
		"""Convert to lower save mode (full → final → archived).

		Args:
			model: Model instance (needed if target mode requires it)
			optimizer: Optimizer instance (needed if target mode requires it)
			target_mode: Target save mode

		Returns:
			New TrainingRun instance with target save mode

		Raises:
			ValueError if trying to convert to higher mode
		"""
		current_level = _SaveModeLevel[self.save_mode.upper()].value
		target_level = _SaveModeLevel[target_mode.upper()].value

		if target_level > current_level:
			raise ValueError(
				f"Cannot convert from '{self.save_mode}' to '{target_mode}'. "
				f"Can only convert to lower modes (full → final → archived)."
			)

		# Create new instance with same configs
		new_run = TrainingRun(
			model_class=self.model_class,
			model_config=self.model_config,
			data_config=self.data_config,
			train_config=self.train_config,
			optimizer_config=self.optimizer_config,
			dataset_hash=self.dataset_hash,
		)
		new_run.logs = self.logs
		new_run.git_hash = self.git_hash
		new_run.created_at = self.created_at
		new_run.save_mode = target_mode
		return new_run

	def create_model(self) -> nn.Module:
		"""Create model instance.

		Returns:
			Model instance with configs applied
		"""
		return self.model_class(
			model_config=self.model_config,
			data_config=self.data_config
		)

	def create_optimizer(self, model: nn.Module) -> Optimizer:
		"""Create optimizer instance.

		Args:
			model: Model whose parameters to optimize

		Returns:
			Optimizer instance
		"""
		return Optimizer(config=self.optimizer_config, model_parameters=model.parameters())

	def _serialize_optimizer_config(self) -> dict:
		"""Serialize optimizer config to dict.

		Handles tuple-to-list conversion for JSON compatibility.
		"""
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
		"""Convert tuples to lists recursively for JSON serialization."""
		if isinstance(obj, tuple):
			return list(obj)
		elif isinstance(obj, dict):
			return {k: TrainingRun._convert_tuples_to_lists(v) for k, v in obj.items()}
		elif isinstance(obj, list):
			return [TrainingRun._convert_tuples_to_lists(item) for item in obj]
		else:
			return obj

	@classmethod
	def _deserialize_optimizer_config(cls, obj: dict) -> OptimizerConfig:
		"""Deserialize optimizer config from dict.

		Handles list-to-tuple conversion for betas parameter.
		"""
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
				scheduler = ReduceLROnPlateauConfig(**{k: v for k, v in scheduler_dict.items() if k != 'type'})
			case "warmup_cosine":
				scheduler = WarmupCosineConfig(**{k: v for k, v in scheduler_dict.items() if k != 'type'})
			case "cosine":
				scheduler = CosineOnlyConfig(**{k: v for k, v in scheduler_dict.items() if k != 'type'})
			case _:
				raise ValueError(f"Unknown scheduler type: {scheduler_dict.get('type')}")

		return OptimizerConfig(
			optimizer_type=obj["optimizer_type"],
			optimizer_params=optimizer_params,
			scheduler=scheduler
		)
