from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Protocol, Tuple, TypeGuard, Union, runtime_checkable, Set, Self
import os
import json
import torch
import functools
import operator

flatten_list = lambda l: functools.reduce(operator.iconcat, l, [])

# note: references must provide basename, write definition, and data
# write definition requires basename, dirname, data

# base serializable typing (stored in JSON)
Serializable = Union[
  int, float, bool, str,
  List['Serializable'], 
  Dict[str, 'Serializable'],
]
# saved in independent files (custom savable types)
@runtime_checkable
class SavableProtocol(Protocol):
  name: str
  def encode(self) -> Serializable:
    ...
  @classmethod
  def decode(cls, obj: Serializable) -> Self:
    ...
  def write(self, dirname: str) -> None:
    ...
  def read(self, dirname: str) -> Self:
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
      return (v, [])
    case list(v):
      vals = zip(*[encode(el) for el in v])
      return (vals[0], flatten_list(vals[1]))
    case dict(d):
      pairs = {k: encode(v) for k, v in d.items()}
      return (
        {k: s for k, (s, _) in pairs.items()},
        flatten_list(r for _, (_, r) in pairs.items())
      )
    case SavableProtocol():
      return obj.encode()
    case _:
      raise TypeError(f"cannot encode object {obj}")

def decode(obj: Serializable) -> Tuple[Savable, SavableProtocol]:
  match obj:
    case int(v) | float(v) | bool(v) | str(v):
      return (v, [])
    case list(v):
      vals = zip(*[decode(el) for el in v])
      return (vals[0], flatten_list(vals[1]))
    case dict(d) if "__type__" not in d:
      pairs = {k: decode(v) for k, v in d.items()}
      return (
        {k: s for k, (s, _) in pairs.items()},
        flatten_list(r for _, (_, r) in pairs.items())
      )
    case dict(d):
      t = obj["__type__"]
      cls = CUSTOM_SERIALIZABLE_TYPES.get(t)
      if cls is None: raise TypeError(f"cannot decode type {t}")
      ref = cls.decode(obj)
      return (ref, [ref])
    case _:
      raise TypeError(f"cannot decode object {obj}")

def save(obj: Savable, path: str) -> None:
  metadata, refs = encode(obj)
  if not os.path.exists(path):
    try: os.makedirs(path, exist_ok=True)
    except OSError as e: raise ValueError(f"error creating directory for {path}") from e

  # write JSON
  with open(os.path.join(path, "metadata.json"), "w+") as file:
    json.dump(metadata, file, indent=4)
  
  # write references
  for ref in refs:
    try:
      ref.write(path)
    except Exception as e:
      raise IOError(f"error writing {type(ref).__name__}") from e

def load(path: str) -> Savable:
  try:
    with open(os.path.join(path, "metadata.json"), 'r') as file:
      obj, refs = decode(json.load(file))
      for ref in refs:
        ref.read(path)
      return obj
  except FileNotFoundError:
    raise
  except Exception as e:
    raise IOError("error reading object") from e

@custom_serializable_type("TorchTensorRef")
@dataclass()
class TorchTensorRef:
  name: str
  tensor: Optional[torch.Tensor] = None

  def encode(self) -> Tuple[Serializable, List[SavableProtocol]]:
    return ({
      "__type__": self.__typename__,
      "__ref__": self.name,
    }, [self])
  
  @classmethod
  def decode(cls, obj: Serializable):
    return TorchTensorRef(name=obj["__ref__"])
  
  def write(self, dirname: str):
    torch.save(self.tensor, os.path.join(dirname, self.name))
    
  def read(self, dirname: str) -> Self:
    self.tensor = torch.load(os.path.join(dirname, self.name))

# class Config(dataclass):
#   vocab_size: int
#   # ...

# class TokenDictionary(NamedTuple):
#   token_set: List[int]
#   idx_to_token: Dict[int, str]
#   token_to_idx: Dict[str, int]

#   def __save__(self):
#     return { "token_list": self.token_set }
  
#   @staticmethod
#   def __load__(state_dict: Dict[str, Any]):
#     S = state_dict['token_set']
#     idx_to_token = dict(enumerate(S))
#     token_to_idx = dict([(x, i) for i, x in enumerate(S)])
#     return TokenDictionary(
#       token_set=state_dict['token_set'],
#       idx_to_token=idx_to_token,
#       token_to_idx=token_to_idx
#     )

# class ProcessedText():
#   vocab_hash: str
#   data: torch.Tensor # SAVE IN SEPARATE FILE

# class Model():
#   vocab_hash: str
#   config: Config
#   state_dict: Dict[str, Any] # use torch.save, torch.load, torch.state_dict, ...

# class TrainingRun():
#   config: Config
#   checkpoints: List[Tuple[int, Dict[str, Any]]]
#   optimizer_state_dict: Dict[str, Any]
#   training_loss: List[Tuple[int, int, int]]