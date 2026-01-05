from dataclasses import dataclass, asdict, is_dataclass
from typing import Type, Any, Optional, List, Dict, Tuple, Union
import torch
import torch.nn as nn
from datetime import datetime, timezone
import importlib

from toy_transformers.utilities import io
from toy_transformers.utilities.io import TorchStateDictRef, TrainingLogRef
from toy_transformers.utilities.version import get_obj_metadata, get_git_version, get_pytorch_version
from toy_transformers.training import training_run

class FinalModel:
	def __init__(
		self,
		model_class: Type[nn.Module],
		config_class: Type,
		model_config: Any,
		dataset_hash: int,
		vocab_size: Optional[int] = None,
		logs: Optional[List[Dict[str, Union[int, float]]]] = None,
		final_epoch: Optional[int] = None,
		final_step: Optional[int] = None,
	):
		self.model_class = model_class
		self.config_class = config_class
		self.model_config = model_config
		self.dataset_hash = dataset_hash
		self.vocab_size = vocab_size
		self.logs = logs if logs is not None else []
		self.final_epoch = final_epoch
		self.final_step = final_step

		# Capture version info
		self.git_hash = get_git_version()
		self.created_at = datetime.now(timezone.utc).isoformat()

	def to_state_dict(self, model: nn.Module) -> io.Savable:
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

			# Training provenance
			"dataset_hash": self.dataset_hash,
			"final_epoch": self.final_epoch,
			"final_step": self.final_step,

			# Data
			"model_weights": TorchStateDictRef("model_weights", model.state_dict()),
			"logs": TrainingLogRef("training_logs", log_rows) if log_rows else None,
		}

	def _logs_to_csv(self) -> List[List[Union[str, int, float]]]:
		"""Convert logs to CSV format."""
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
	def from_state_dict(cls, obj: dict) -> Tuple['FinalModel', dict]:
		import importlib

		model_module = importlib.import_module(obj["model_class_module"])
		model_class = getattr(model_module, obj["model_class_name"])

		config_module = importlib.import_module(obj["config_class_module"])
		config_class = getattr(config_module, obj["config_class_name"])

		model_config = config_class(**obj["model_config"])

		logs = cls._csv_to_logs(obj["logs"].logs) if obj.get("logs") else []

		final_model = cls(
			model_class=model_class,
			config_class=config_class,
			model_config=model_config,
			dataset_hash=obj["dataset_hash"],
			vocab_size=obj.get("vocab_size"),
			logs=logs,
			final_epoch=obj.get("final_epoch"),
			final_step=obj.get("final_step"),
		)

		final_model.git_hash = obj["metadata"]["git_hash"]
		final_model.created_at = obj["metadata"]["created_at"]

		model_state_dict = obj["model_weights"].state_dict

		return final_model, model_state_dict

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
					try:
						if '.' in str(row[i]):
							log[key] = float(row[i])
						else:
							log[key] = int(row[i])
					except (ValueError, TypeError):
						log[key] = row[i]
			logs.append(log)

		return logs

	def create_model(self) -> nn.Module:
		if self.vocab_size is not None:
			return self.model_class(
				vocab_size=self.vocab_size,
				config=self.model_config
			)
		else:
			return self.model_class(config=self.model_config)

	@classmethod
	def from_training_run(
		cls,
		training_run: training_run.TrainingRun,
		include_logs: bool = True
	) -> 'FinalModel':
		return cls(
			model_class=training_run.model_class,
			config_class=training_run.config_class,
			model_config=training_run.model_config,
			dataset_hash=training_run.dataset_hash,
			vocab_size=training_run.vocab_size,
			logs=training_run.logs if include_logs else [],
			final_epoch=training_run.epoch,
			final_step=training_run.step,
		)
