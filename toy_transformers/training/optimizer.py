from dataclasses import dataclass, field, asdict
from typing import Optional, Literal, Union, List
import torch
import torch.nn as nn

# optimizer configs
@dataclass(frozen=True)
class AdamWConfig:
  lr: float
  weight_decay: float = 0.01
  betas: List[float] = field(default_factory=lambda: [0.9, 0.999])
  eps: float = 1e-8

@dataclass(frozen=True)
class SGDConfig:
  lr: float
  momentum: float = 0.0
  weight_decay: float = 0.0
  dampening: float = 0.0
  nesterov: bool = False

# scheduler configs
@dataclass(frozen=True)
class NoSchedulerConfig:
  type: Literal["none"] = "none"

@dataclass(frozen=True)
class ReduceLROnPlateauConfig:
  type: Literal["reduce_on_plateau"] = "reduce_on_plateau"
  mode: Literal["min", "max"] = "min"
  factor: float = 0.1
  patience: int = 10
  threshold: float = 1e-4
  cooldown: int = 0
  min_lr: float = 0.0

@dataclass(frozen=True)
class WarmupCosineConfig:
  warmup_steps: int
  total_steps: int
  start_factor: float = 0.01
  eta_min: float = 0.0
  type: Literal["warmup_cosine"] = "warmup_cosine"

@dataclass(frozen=True)
class CosineOnlyConfig:
  T_max: int
  eta_min: float = 0.0
  type: Literal["cosine"] = "cosine"

SchedulerConfig = Union[
  NoSchedulerConfig,
  ReduceLROnPlateauConfig,
  WarmupCosineConfig,
  CosineOnlyConfig
]


@dataclass(frozen=True)
class OptimizerConfig:
  optimizer_type: Literal["adamw", "sgd"]
  optimizer_params: Union[AdamWConfig, SGDConfig]
  scheduler: SchedulerConfig
  device: str = "cpu"
  seed: int = 42
  max_epochs: Optional[int] = None


class Optimizer:

  def __init__(self, config: OptimizerConfig, model_parameters):
    self.config = config
    self.optimizer = self._create_optimizer(model_parameters)
    self.scheduler = self._create_scheduler()

  def _create_optimizer(self, model_parameters) -> torch.optim.Optimizer:
    match self.config.optimizer_type:
      case "adamw":
        params = self.config.optimizer_params
        return torch.optim.AdamW(
          model_parameters,
          lr=params.lr,
          betas=params.betas,
          eps=params.eps,
          weight_decay=params.weight_decay
        )
      case "sgd":
        params = self.config.optimizer_params
        return torch.optim.SGD(
          model_parameters,
          lr=params.lr,
          momentum=params.momentum,
          weight_decay=params.weight_decay,
          dampening=params.dampening,
          nesterov=params.nesterov
        )
      case _:
        raise ValueError(f"unknown optimizer type: {self.config.optimizer_type}")

  def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    match self.config.scheduler:
      case NoSchedulerConfig():
        return None
      case ReduceLROnPlateauConfig() as cfg:
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
          self.optimizer,
          mode=cfg.mode,
          factor=cfg.factor,
          patience=cfg.patience,
          threshold=cfg.threshold,
          cooldown=cfg.cooldown,
          min_lr=cfg.min_lr
        )
      case WarmupCosineConfig() as cfg:
        warmup = torch.optim.lr_scheduler.LinearLR(
          self.optimizer,
          start_factor=cfg.start_factor,
          total_iters=cfg.warmup_steps
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
          self.optimizer,
          T_max=cfg.total_steps - cfg.warmup_steps,
          eta_min=cfg.eta_min
        )
        return torch.optim.lr_scheduler.SequentialLR(
          self.optimizer,
          schedulers=[warmup, cosine],
          milestones=[cfg.warmup_steps]
        )
      case CosineOnlyConfig() as cfg:
        return torch.optim.lr_scheduler.CosineAnnealingLR(
          self.optimizer,
          T_max=cfg.T_max,
          eta_min=cfg.eta_min
        )
      case _:
        raise ValueError(f"unknown scheduler type: {self.config.scheduler}")

  def step(self, loss: Optional[float] = None):
    self.optimizer.step()

    if self.scheduler is not None:
      # ReduceLROnPlateau requires loss metric
      if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        if loss is None:
          raise ValueError("ReduceLROnPlateau requires loss argument")
        self.scheduler.step(loss)
      else:
        self.scheduler.step()

  def zero_grad(self, set_to_none: bool = True):
    self.optimizer.zero_grad(set_to_none=set_to_none)

  def get_lr(self) -> float:
    return self.optimizer.param_groups[0]['lr']

  def state_dict(self) -> dict:
    state = {
      'optimizer': self.optimizer.state_dict(),
    }
    if self.scheduler is not None:
      state['scheduler'] = self.scheduler.state_dict()
    return state

  def load_state_dict(self, state_dict: dict):
    self.optimizer.load_state_dict(state_dict['optimizer'])
    if self.scheduler is not None and 'scheduler' in state_dict:
      self.scheduler.load_state_dict(state_dict['scheduler'])

  @classmethod
  def from_config_and_state(
    cls,
    config: OptimizerConfig,
    model_parameters,
    state_dict: Optional[dict] = None
  ) -> 'Optimizer':
    optimizer = cls(config, model_parameters)
    if state_dict is not None:
      optimizer.load_state_dict(state_dict)
    return optimizer

  @classmethod
  def serialize_config(cls, config: OptimizerConfig) -> dict:
    return {
      "optimizer_type": config.optimizer_type,
      "optimizer_params": asdict(config.optimizer_params),
      "scheduler": asdict(config.scheduler),
      "device": config.device,
      "seed": config.seed,
      "max_epochs": config.max_epochs,
    }

  @classmethod
  def deserialize_config(cls, obj: dict) -> OptimizerConfig:
    match obj["optimizer_type"]:
      case "adamw":
        optimizer_params = AdamWConfig(**obj["optimizer_params"])
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
      scheduler=scheduler,
      device=obj["device"],
      seed=obj["seed"],
      max_epochs=obj["max_epochs"],
    )
