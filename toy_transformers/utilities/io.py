from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Protocol, Tuple, TypeGuard, Union, runtime_checkable, Set
import os
import json
import torch

# base serializable typing (stored in JSON)
Serializable = Union[
  int, float, bool, str,
  List['Serializable'], 
  Dict[str, 'Serializable'],
]
# saved in independent files (custom savable types)
@runtime_checkable
class SavableProtocol(Protocol):
  def encode(self) -> Serializable:
    ...
  @classmethod
  def decode(cls, obj: Serializable):
    ...
  def write(self, dir: str):
    ...

CUSTOM_SERIALIZABLE_TYPES: Dict[str, type] = dict()

def custom_serializable_type(name: str):
  def wrapper(cls):
    CUSTOM_SERIALIZABLE_TYPES[name] = cls
    cls.__typename__ = name
    return cls
  return wrapper

Savable = Union[Serializable | SavableProtocol]

def encode(obj: Savable) -> Tuple[Serializable, List[SavableProtocol]]:
  match obj:
    case int(v) | float(v) | bool(v) | str(v):
      return v
    case list(v):
      return [encode(el) for el in v]
    case dict(d):
      return {k: encode(v) for k, v in d.items()}
    case SavableProtocol(custom_obj):
      return custom_obj.encode()
    
@custom_serializable_type("TorchTensorRef")
@dataclass(frozen=True)
class TorchTensorRef:
  tensor: torch.Tensor

  def encode(self) -> Tuple[Serializable, List[SavableProtocol]]:
    return ({
      "__type__": self.__typename__,
      "__ref__": "tensor.pt"
    }, self)
  
  def decode(cls, obj: Serializable):
    return cls(tensor=torch.load())
  


# encoding outputs basename
# saving requires basename and dir
# decoding requires basename (and implied dir)