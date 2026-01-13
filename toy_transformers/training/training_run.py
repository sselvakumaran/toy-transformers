from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple
import torch.nn as nn
from datetime import datetime, timezone

from toy_transformers.utilities import io
from toy_transformers.utilities.io import TorchStateDictRef, MetricLogRef
from toy_transformers.artifacts import ArtifactType, ArtifactMetadata, stable_hash, get_git_version
from toy_transformers.training.optimizer import Optimizer
from toy_transformers.configs import DataConfig, OptimizerConfig
from toy_transformers.data.dataset import create_dataloader, skip_batches
from toy_transformers.data.tokenization import TokenizedData

# TODO:
# stable_hash() of TrainingRun
# merge load() and from_state_dict


class TrainingRun:
  def __init__(
    self,
    model_name: str,
    model_config: dict,
    data_config: DataConfig,
    optimizer_config: OptimizerConfig,
    tokenized_data_id: Optional[str] = None,
    epoch: int = 0, step: int = 0, batches_completed: int = 0,
  ):
    self.model_name = model_name
    self.model_config = model_config
    self.data_config = data_config
    self.optimizer_config = optimizer_config
    self.tokenized_data_id = tokenized_data_id

    self._epoch = epoch
    self._step = step
    self._batches_completed = batches_completed

    self.logs: Dict[str, List[Any]] = {}

  @property
  def epoch(self) -> int:
    return self._epoch

  @epoch.setter
  def epoch(self, value: int):
    self._epoch = value

  @property
  def step(self) -> int:
    return self._step

  @step.setter
  def step(self, value: int):
    self._step = value

  @property
  def batches_completed(self) -> int:
    return self._batches_completed

  @batches_completed.setter
  def batches_completed(self, value: int):
    self._batches_completed = value

  @property
  def device(self) -> str:
    return self.optimizer_config.device

  @property
  def seed(self) -> int:
    return self.optimizer_config.seed

  @property
  def max_epochs(self) -> Optional[int]:
    return self.optimizer_config.max_epochs

  def log_step(self, **metrics):
    """log metrics for current step"""
    all_metrics = {"epoch": self.epoch, "step": self.step, **metrics}
    for key, value in all_metrics.items():
      if key not in self.logs:
        self.logs[key] = []
      self.logs[key].append(value)

  def get_epoch_seed(self, epoch: Optional[int] = None) -> int:
    """get seed for a specified epoch"""
    if epoch is None:
      epoch = self.epoch
    return self.seed + epoch

  def create_dataloader(self, dataset: Any, epoch: Optional[int] = None):
    """create dataloader for training an epoch"""
    if epoch is None:
      epoch = self.epoch

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

    if self.batches_completed > 0:
      return skip_batches(loader, self.batches_completed)
    else:
      return iter(loader)

  def verify_tokenized_data(self, tokenized_data: TokenizedData):
    """verify tokenized data matches stored artifact ID"""
    if self.tokenized_data_id is None: return
    computed = tokenized_data.stable_hash()
    assert computed == self.tokenized_data_id, \
      f"TokenizedData mismatch! Expected {self.tokenized_data_id[:8]}..., got {computed[:8]}..."

  def to_state_dict(self, model: nn.Module, optimizer: Optimizer) -> io.Savable:
    """save training run state"""
    content_hash = stable_hash({
      "model_type": self.model_name,
      "model_config": self.model_config,
      "training_state": {
        "epoch": self.epoch,
        "step": self.step,
      },
      "created_at": self.created_at,
    })
    metadata = ArtifactMetadata(ArtifactType.TRAINING_RUN, content_hash)

    state = {
      "artifact_metadata": metadata.to_dict(),
      "model_type": self.model_name,
      "model_config": self.model_config,
      "data_config": asdict(self.data_config),
      "optimizer_config": Optimizer.serialize_config(self.optimizer_config),
      "training_state": {
        "epoch": self.epoch,
        "step": self.step,
        "batches_completed": self.batches_completed,
      },
      "tokenized_data_id": self.tokenized_data_id,
      "model_weights": TorchStateDictRef("model_weights", model.state_dict()),
      "optimizer_state": TorchStateDictRef("optimizer_state", optimizer.state_dict()),
    }

    if self.logs:
      state["logs"] = MetricLogRef("training_logs", data=self.logs)

    return state

  def save(self, path: str, model: nn.Module, optimizer: Optimizer):
    """save training run to disk"""
    state_dict = self.to_state_dict(model, optimizer)
    io.save(state_dict, path)

  @classmethod
  def from_state_dict(cls, obj: dict) -> Tuple['TrainingRun', dict, dict]:
    model_name = obj["model_type"]
    model_config = obj["model_config"]
    data_config = DataConfig(**obj["data_config"])
    optimizer_config = Optimizer.deserialize_config(obj["optimizer_config"])

    training_state = obj["training_state"]

    training_run = cls(
      model_name=model_name,
      model_config=model_config,
      data_config=data_config,
      optimizer_config=optimizer_config,
      tokenized_data_id=obj.get("tokenized_data_id"),
      epoch=training_state["epoch"],
      step=training_state["step"],
      batches_completed=training_state["batches_completed"],
    )

    if "logs" in obj and obj["logs"].data is not None:
      training_run.logs = obj["logs"].data
    else:
      training_run.logs = {}

    # Extract state dicts
    model_state_dict = obj["model_weights"].state_dict
    optimizer_state_dict = obj["optimizer_state"].state_dict

    return training_run, model_state_dict, optimizer_state_dict

  @classmethod
  def load(cls, path: str) -> Tuple['TrainingRun', nn.Module, Optimizer]:
    """load training run from disk, returns (training_run, model, optimizer)"""
    obj = io.load(path)
    training_run, model_state_dict, optimizer_state_dict = cls.from_state_dict(obj)

    model = training_run.create_model()
    model.load_state_dict(model_state_dict)

    optimizer = training_run.create_optimizer(model)
    optimizer.load_state_dict(optimizer_state_dict)

    return training_run, model, optimizer

  def create_model(self) -> nn.Module:
    """create model instance using ModelRegistry"""
    from toy_transformers.models import ModelRegistry
    return ModelRegistry.build(self.model_name, self.model_config, self.data_config)

  def create_optimizer(self, model: nn.Module) -> Optimizer:
    """create optimizer instance"""
    return Optimizer(config=self.optimizer_config, model_parameters=model.parameters())
