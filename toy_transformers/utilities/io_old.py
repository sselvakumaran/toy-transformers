from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Protocol, Tuple, TypeGuard, Union, Set
import torch
import os
import json
# savable things should have save() function then use hasattr

# Savable: primitive | torch.Tensor | torch.Module


"""
new thing:
each has a SavableProtocol 
"""


Serializable = Union[
  int, float, bool, str,
  List['Serializable'], 
  Dict[str, 'Serializable'],
]

@dataclass(frozen=True)
class TorchTensorRef:
  tensor: torch.Tensor

@dataclass(frozen=True)
class TorchStateDictRef:
  state_dict: Dict[str, Any]

Savable = Union[Serializable | TorchTensorRef | TorchStateDictRef]

class SavableProtocol(Protocol):
  def encode(self) -> Serializable:
    ...
  @classmethod
  def decode(cls, obj: Serializable):
    ...

CUSTOM_SERIALIZABLE_TYPES: Dict[str, type] = dict()

def custom_serializable_type(name: str):
  def wrapper(cls):
    CUSTOM_SERIALIZABLE_TYPES[name] = cls
    cls.__typename__ = name
    return cls
  return wrapper

def parse_spec(
    spec: Dict[str, Savable]
) -> Dict[str, Serializable]:
  if type(spec) != dict:
    raise ValueError("spec formatting incorrect: must be dict")
  
  def recursive_parse_spec(obj: Savable) -> Dict[str, Serializable]:
    match obj:
      case int(v) | float(v) | bool(v) | str(v):
        return v
      case list(lst):
        return list(map(recursive_parse_spec, lst))
      case dict(d):
        if not all(isinstance(k, str) for k in d.keys()):
          failed = list(filter(not isinstance(k, str) for k in d.keys()))
          raise ValueError(f"keys {failed} are not strings")
        return dict((t[0], recursive_parse_spec(t[1])) for t in d.items())
      case _:
        raise ValueError(f"{obj} is not Savable type")
  return recursive_parse_spec(spec)

def unparse_spec(
  spec: Dict[str, Serializable]
) -> Dict[str, Savable]:
  return spec

def save(
  spec: Dict[str, Savable],
  path: str,
) -> None:
  parsed_spec = parse_spec(spec)
  if not path.endswith(".json"):
    raise ValueError(f"filename is not a proper JSON: {path}")

  dir = os.path.dirname(path)
  if not os.path.exists(path) and dir != '':
    try: os.makedirs(dir, exist_ok=True)
    except OSError as e: raise ValueError(f"error creating directory for {dir}: {e}")

  with open(path, "w+") as file:
    json.dump(parsed_spec, file, indent=4)

def load(
  path: str,
) -> Dict[str, Savable]:
  try:
    with open(path, 'r') as file:
      parsed_spec = json.load(file)
      spec = unparse_spec(parsed_spec)
      return spec
  except FileNotFoundError:
    print(f"file not found: {path}")
  except json.decoder.JSONDecodeError as e:
    raise ValueError(f"error decoding JSON: {e}") from e
  except Exception as e:
    raise e  

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

"""
{x: torch.Tensor(...), y: 2}
->
{x: {"ref": "x.pt", "type": "tensor"}, y: 2}
"""