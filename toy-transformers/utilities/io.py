
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple, TypeGuard, Union
import torch
# savable things should have save() function then use hasattr

# Savable: primitive | torch.Tensor | torch.Module

"""
Savable: any object serializable
  __save__: returns serializable format
  __load__: creates Savable object from serialization
obj.__save__ -> returns Dict[str, Savable]
obj.__load__ -> returns new class

POTENTIAL
obj.__save -> returns (Dict[str, Serializable], optionally fn_save)
fn_save takes base path and writes to that location
problem is io.py doesnt handle writing... but ok if torch handles??
or maybe also take in (relative path, InputStream) ?

do i want io.py to have torch.save() / load? YES - so don't have to worry about
it in individual classes
individual classes must SIGNAL to save with torch

input - Savable functions, must have save and load

save should return of str to Serializable | TorchWritable | ExternalWritable...
"""

# io.save(Object.dump())
# Object.load(io.load())

Serializable = Union[
  int, float, str, bool, 
  List['Serializable'], 
  Dict[str, 'Serializable']
]

@dataclass(frozen=True)
class TorchTensorRef:
  tensor: torch.Tensor

@dataclass(frozen=True)
class TorchStateDictRef:
  state_dict: Dict[str, Any]

Savable = Union[Serializable | TorchTensorRef | TorchStateDictRef]



def parse_spec(
    spec: Dict[str, Savable]
) -> Optional[Tuple[Dict[str, Serializable], Callable[[str], None]]]:
  if type(spec) != dict:
    raise ValueError("spec formatting incorrect: must be dict")
  pass

def save(
  spec: Dict[str, Savable],
  path: str,
) -> None:
  """
  Saves an object to path. Either will be a JSON or a directory depending 
  on contents. To save using torch.load, use TorchTensorRef for tensors and 
  TorchModuleStateRef for state refs. 
  This functionality is to store metadata in easier to read formatting and provide
  greater transparency and convenience in saving and loading experiments.
  """
  # a) (parse spec -> json, other files)
  # b) save data
  for key, val in spec.items():
    pass

def load(
  path: str,
) -> Dict[str, Any]:
  pass

class Config(dataclass):
  vocab_size: int
  # ...

class TokenDictionary(NamedTuple):
  token_set: List[int]
  idx_to_token: Dict[int, str]
  token_to_idx: Dict[str, int]

  def __save__(self):
    return { "token_list": self.token_set }
  
  @staticmethod
  def __load__(state_dict: Dict[str, Any]):
    S = state_dict['token_set']
    idx_to_token = dict(enumerate(S))
    token_to_idx = dict([(x, i) for i, x in enumerate(S)])
    return TokenDictionary(
      token_set=state_dict['token_set'],
      idx_to_token=idx_to_token,
      token_to_idx=token_to_idx
    )

class ProcessedText():
  vocab_hash: str
  data: torch.Tensor # SAVE IN SEPARATE FILE

class Model():
  vocab_hash: str
  config: Config
  state_dict: Dict[str, Any] # use torch.save, torch.load, torch.state_dict, ...

class TrainingRun():
  config: Config
  checkpoints: List[Tuple[int, Dict[str, Any]]]
  optimizer_state_dict: Dict[str, Any]
  training_loss: List[Tuple[int, int, int]]