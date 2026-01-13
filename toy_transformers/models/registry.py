from dataclasses import dataclass, fields
from typing import Type, Dict, Any, Optional
import torch.nn as nn

@dataclass
class ModelInfo:
  name: str
  model_class: Type[nn.Module]
  config_class: Type
  description: Optional[str] = None


class ModelRegistry:
  # models must register in their respective programs 
  # so importing automatically imports

  _registry: Dict[str, ModelInfo] = {}

  @classmethod
  def register(
    cls,
    name: str,
    model_class: Type[nn.Module],
    config_class: Type,
    description: Optional[str] = None
  ):
    if name in cls._registry:
      raise ValueError(f"model '{name}' already registered")

    if not hasattr(model_class, 'model_type'):
      raise ValueError(
        f"model class {model_class.__name__} must have 'model_type' attribute"
      )

    if model_class.model_type != name:
      raise ValueError(
        f"model type mismatch: registry name '{name}' != "
        f"model_class.model_type '{model_class.model_type}'"
      )

    cls._registry[name] = ModelInfo(
      name=name,
      model_class=model_class,
      config_class=config_class,
      description=description
    )

  @classmethod
  def get(cls, name: str) -> ModelInfo:
    """get model info by name"""
    if name not in cls._registry:
      available = list(cls._registry.keys())
      raise ValueError(
        f"model '{name}' not registered. available models: {available}"
      )
    return cls._registry[name]

  @classmethod
  def describe(cls, name: str) -> str:
    """describe model configuration fields and defaults"""
    info = cls.get(name)
    config_class = info.config_class

    lines = [f"model: {name}"]
    if info.description:
      lines.append(f"description: {info.description}")
    lines.append(f"\nconfiguration ({config_class.__name__}):")

    for field in fields(config_class):
      field_type = field.type
      # Get default value
      if field.default is not field.default_factory:
        default = field.default
      else:
        default = '<required>'

      lines.append(f"  {field.name}: {field_type} = {default}")

    return "\n".join(lines)

  @classmethod
  def build(cls, name: str, model_config: dict, data_config: Any) -> nn.Module:
    """build model instance from config dict"""
    info = cls.get(name)
    config_instance = info.config_class(**model_config)
    return info.model_class(
      model_config=config_instance,
      data_config=data_config
    )

  @classmethod
  def list_models(cls) -> list[str]:
    """list all registered model names"""
    return sorted(cls._registry.keys())

  @classmethod
  def clear(cls):
    """clear the registry (mainly for testing)"""
    cls._registry.clear()
