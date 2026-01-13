from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Protocol, Tuple, TypeGuard, Union, runtime_checkable, Set, Self
from pathlib import Path
import json
import torch
import csv
import zlib

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

  def encode(self) -> Tuple[Serializable, List['SavableProtocol']]:
    """return (serializable_representation, list_of_references)"""
    ...

  @classmethod
  def decode(cls, obj: Serializable) -> Self:
    """construct object from serialized form (without loading files)"""
    ...

  def write(self, dirname: str) -> None:
    """write associated file into directory"""
    ...

  def read(self, dirname: str) -> Self:
    """load associated file from directory"""
    ...

CUSTOM_SERIALIZABLE_TYPES: Dict[str, type] = dict()

Savable = Union[Serializable | SavableProtocol]


def encode(obj: Savable) -> Tuple[Serializable, List[SavableProtocol]]:
  match obj:
    case int(v) | float(v) | bool(v) | str(v):
      return (v, [])
    case list(v):
      serialized = []
      refs = []
      for el in v:
        s, r = encode(el)
        serialized.append(s)
        refs.extend(r)
      return (serialized, refs)
    case dict(d):
      serialized = {}
      refs = []
      for k, v in d.items():
        s, r = encode(v)
        serialized[k] = s
        refs.extend(r)
      return (serialized, refs)
    case SavableProtocol():
      return obj.encode()
    case _:
      raise TypeError(f"cannot encode object of type {type(obj)}: {obj}")


def decode(obj: Serializable) -> Tuple[Savable, List[SavableProtocol]]:
  match obj:
    case int(v) | float(v) | bool(v) | str(v):
      return (v, [])
    case list(v):
      decoded = []
      refs = []
      for el in v:
        d, r = decode(el)
        decoded.append(d)
        refs.extend(r)
      return (decoded, refs)
    case dict(d) if "__type__" not in d:
      decoded = {}
      refs = []
      for k, v in d.items():
        dec, r = decode(v)
        decoded[k] = dec
        refs.extend(r)
      return (decoded, refs)
    case dict(d):
      t = obj["__type__"]
      cls = CUSTOM_SERIALIZABLE_TYPES.get(t)
      if cls is None:
        raise TypeError(f"cannot decode type {t}")
      ref = cls.decode(obj)
      return (ref, [ref])
    case _:
      raise TypeError(f"cannot decode object {obj}")


def save(obj: Savable, path: str) -> None:
  metadata, refs = encode(obj)
  unique_refs = {ref.name: ref for ref in refs}

  path_obj = Path(path)
  path_obj.mkdir(parents=True, exist_ok=True)

  # write JSON
  with open(path_obj / "metadata.json", "w") as file:
    json.dump(metadata, file, indent=2)

  # write references
  for ref in unique_refs.values():
    ref.write(str(path_obj))


def load(path: str) -> Savable:
  path_obj = Path(path)
  try:
    with open(path_obj / "metadata.json", 'r') as file:
      obj, refs = decode(json.load(file))
      for ref in refs:
        ref.read(str(path_obj))
      return obj
  except FileNotFoundError:
    raise
  except Exception as e:
    raise IOError("error reading object") from e

# Base FileRef class with auto-registration


@dataclass
class FileRef:
  """base class for file references, subclasses are auto-registered"""
  name: str
  extension: str = ""

  def __init_subclass__(cls, **kwargs):
    super().__init_subclass__(**kwargs)
    # Auto-register subclass using its name
    typename = cls.__name__
    cls.__typename__ = typename
    CUSTOM_SERIALIZABLE_TYPES[typename] = cls

  def __post_init__(self):
    if self.extension and not self.name.endswith(self.extension):
      self.name = f"{self.name}{self.extension}"

  def encode(self) -> Tuple[Serializable, List[SavableProtocol]]:
    self._validate_data()
    return ({
      "__type__": self.__typename__,
      "__ref__": self.name,
    }, [self])

  @classmethod
  def decode(cls, obj: Serializable):
    if not isinstance(obj, dict):
      raise TypeError(f"expected dict, got {type(obj)}")
    return cls(name=obj["__ref__"])

  def _validate_data(self):
    pass

  def write(self, _dirname: str) -> None:
    raise NotImplementedError

  def read(self, _dirname: str) -> Self:
    raise NotImplementedError


@dataclass
class TorchTensorRef(FileRef):
  tensor: Optional[torch.Tensor] = None
  extension: str = field(default='.pt', init=False)

  def _validate_data(self):
    if self.tensor is None:
      raise ValueError(f"cannot encode TorchTensorRef '{self.name}' with None tensor")

  def write(self, dirname: str):
    self._validate_data()
    torch.save(self.tensor, Path(dirname) / self.name)

  def read(self, dirname: str) -> Self:
    self.tensor = torch.load(Path(dirname) / self.name, weights_only=True)
    return self


@dataclass
class TorchStateDictRef(FileRef):
  state_dict: Optional[dict] = None
  extension: str = field(default='.pt', init=False)

  def _validate_data(self):
    if self.state_dict is None:
      raise ValueError(f"cannot encode TorchStateDictRef '{self.name}' with None dict")

  def write(self, dirname: str):
    self._validate_data()
    torch.save(self.state_dict, Path(dirname) / self.name)

  def read(self, dirname: str) -> Self:
    self.state_dict = torch.load(Path(dirname) / self.name, weights_only=True)
    return self


@dataclass
class TrainingLogRef(FileRef):
  logs: Optional[List[List[Union[str, int, float, bool]]]] = None
  extension: str = field(default='.csv', init=False)

  def _validate_data(self):
    if self.logs is None:
      raise ValueError(f"cannot encode TrainingLogRef '{self.name}' with None array")

  def write(self, dirname: str):
    self._validate_data()
    with open(Path(dirname) / self.name, mode='w', newline='') as f:
      writer = csv.writer(f)
      writer.writerows(self.logs)

  def read(self, dirname: str) -> Self:
    with open(Path(dirname) / self.name, mode='r') as f:
      reader = csv.reader(f)
      self.logs = list(reader)
    return self


@dataclass
class MetricLogRef(FileRef):
  data: Optional[Dict[str, List[Any]]] = None
  extension: str = field(default='.json', init=False)

  def write(self, dirname: str):
    with open(Path(dirname) / self.name, 'w') as f:
      json.dump(self.data, f)

  def read(self, dirname: str) -> Self:
    with open(Path(dirname) / self.name, 'r') as f:
      self.data = json.load(f)
    return self

from toy_transformers.data.bpe import TokenizationMode

@dataclass
class CompressedBytesRef(FileRef):
  data: Optional[str | bytes] = None
  mode: Optional[TokenizationMode] = None
  extension: str = field(default='.zlib', init=False)

  def _validate_data(self):
    if self.data is None:
      raise ValueError(f"cannot encode CompressedBytesRef '{self.name}' with None data")

  def write(self, dirname: str):
    self._validate_data()
    if isinstance(self.data, bytes):
      raw_bytes = self.data
    else:
      raw_bytes = self.data.encode('utf-8')
    compressed = zlib.compress(raw_bytes, level=9)
    with open(Path(dirname) / self.name, 'wb') as f:
      f.write(compressed)

  def read(self, dirname: str) -> Self:
    with open(Path(dirname) / self.name, 'rb') as f:
      compressed = f.read()
    raw_bytes = zlib.decompress(compressed)
    if self.mode == TokenizationMode.BYTES:
      self.data = raw_bytes
    else:
      self.data = raw_bytes.decode('utf-8')
    return self

  def encode(self) -> Tuple[Serializable, List[SavableProtocol]]:
    self._validate_data()
    mode_value = self.mode.value if self.mode else TokenizationMode.STR.value
    return ({
      "__type__": self.__typename__,
      "__ref__": self.name,
      "mode": mode_value,
    }, [self])

  @classmethod
  def decode(cls, obj: Serializable):
    if not isinstance(obj, dict):
      raise TypeError(f"expected dict, got {type(obj)}")
    mode = TokenizationMode(obj.get("mode", "str"))
    return cls(name=obj["__ref__"], mode=mode)