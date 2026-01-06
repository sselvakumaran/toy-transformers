from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Protocol, Tuple, TypeGuard, Union, runtime_checkable, Set, Self
import os
import json
import torch
import functools
import operator
import csv

flatten_list = lambda l: functools.reduce(operator.iconcat, l, [])

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
			if not v:
				return ([], [])
			encoded = [encode(el) for el in v]
			return (
				[s for s, _ in encoded],
				flatten_list([r for _, r in encoded])
			)
		case dict(d):
			if not d:
				return ({}, [])
			pairs = {k: encode(v) for k, v in d.items()}
			return (
				{k: s for k, (s, _) in pairs.items()},
				flatten_list(r for _, (_, r) in pairs.items())
			)
		case SavableProtocol():
			return obj.encode()
		case _:
			raise TypeError(f"cannot encode object of type {type(obj)}: {obj}")

def decode(obj: Serializable) -> Tuple[Savable, SavableProtocol]:
	match obj:
		case int(v) | float(v) | bool(v) | str(v):
			return (v, [])
		case list(v):
			vals = list(zip(*[decode(el) for el in v]))
			return (vals[0], flatten_list(vals[1])) if vals else ([], [])
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

	unique_refs = {ref.name: ref for ref in refs}

	os.makedirs(path, exist_ok=True)

	# write JSON
	with open(os.path.join(path, "metadata.json"), "w+") as file:
		json.dump(metadata, file, indent=2)
	
	# write references
	for ref in unique_refs.values():
		ref.write(path)

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
@dataclass
class TorchTensorRef:
	name: str
	tensor: Optional[torch.Tensor] = None

	def __post_init__(self):
		if not self.name.endswith('.pt'):
			self.name = f"{self.name}.pt"

	def encode(self) -> Tuple[Serializable, List[SavableProtocol]]:
		if self.tensor is None:
			raise ValueError(f"cannot encode TorchTensorRef '{self.name}' with None tensor")
		return ({
			"__type__": self.__typename__,
			"__ref__": self.name,
		}, [self])
	
	@classmethod
	def decode(cls, obj: Serializable):
		if not isinstance(obj, dict):
			raise TypeError(f"expected dict, got {type(obj)}")
		return TorchTensorRef(name=obj["__ref__"])
	
	def write(self, dirname: str):
		if self.tensor is None:
			raise ValueError(f"cannot write TorchTensorRef '{self.name}' with None tensor")
		torch.save(self.tensor, os.path.join(dirname, self.name))
		
	def read(self, dirname: str) -> Self:
		self.tensor = torch.load(os.path.join(dirname, self.name), weights_only=True)
		return self

@custom_serializable_type("TorchStateDictRef")
@dataclass
class TorchStateDictRef:
	name: str
	state_dict: Optional[dict] = None

	def __post_init__(self):
		if not self.name.endswith('.pt'):
			self.name = f"{self.name}.pt"

	def encode(self) -> Tuple[Serializable, List[SavableProtocol]]:
		if self.state_dict is None:
			raise ValueError(f"cannot encode TorchStateDictRef '{self.name}' with None dict")
		return ({
			"__type__": self.__typename__,
			"__ref__": self.name,
		}, [self])
	
	@classmethod
	def decode(cls, obj: Serializable):
		if not isinstance(obj, dict):
			raise TypeError(f"expected dict, got {type(obj)}")
		return TorchStateDictRef(name=obj["__ref__"])
	
	def write(self, dirname: str):
		if self.state_dict is None:
			raise ValueError(f"cannot write TorchStateDictRef '{self.name}' with None tensor")
		torch.save(self.state_dict, os.path.join(dirname, self.name))
		
	def read(self, dirname: str) -> Self:
		self.state_dict = torch.load(os.path.join(dirname, self.name), weights_only=True)
		return self

@custom_serializable_type("TrainingLogRef")
@dataclass
class TrainingLogRef:
	name: str
	logs: Optional[List[List[Union[str, int, float, bool]]]] = None

	def __post_init__(self):
		if not self.name.endswith('.csv'):
			self.name = f"{self.name}.csv"

	def encode(self) -> Tuple[Serializable, List[SavableProtocol]]:
		if self.logs is None:
			raise ValueError(f"cannot encode TrainingLogRef '{self.name}' with None array")
		return ({
			"__type__": self.__typename__,
			"__ref__": self.name,
		}, [self])

	@classmethod
	def decode(cls, obj: Serializable):
		if not isinstance(obj, dict):
			raise TypeError(f"expected dict, got {type(obj)}")
		return TrainingLogRef(name=obj["__ref__"])
	
	def write(self, dirname: str):
		if self.logs is None:
			raise ValueError(f"cannot write TrainingLogRef '{self.name}' with None array")
		with open(os.path.join(dirname, self.name), mode='w', newline='') as f:
			writer = csv.writer(f)
			writer.writerows(self.logs)
		
	def read(self, dirname: str) -> Self:
		with open(os.path.join(dirname, self.name), mode='r') as f:
			reader = csv.reader(f)
			self.logs = list(reader)
		return self

@custom_serializable_type("MetricLogRef")
@dataclass
class MetricLogRef(SavableProtocol):
	name: str
	data: Dict[str, List[float]] = field(default_factory=dict)

	def __post_init__(self):
		if not self.name.endswith('.json'):
			self.name = f"{self.name}.json"

	def log(self, **metrics):
		for k, v in metrics.items():
			if k not in self.data:
				self.data[k] = []
			self.data[k].append(float(v))

	def encode(self) -> Tuple[dict, List[SavableProtocol]]:
		return ({
			"__type__": self.__typename__,
			"__ref__": self.name,
		}, [self])

	@classmethod
	def decode(cls, obj: dict):
		return cls(name=obj["__ref__"])

	def write(self, dirname: str):
		with open(os.path.join(dirname, self.name), 'w') as f:
			json.dump(self.data, f)

	def read(self, dirname: str):
		with open(os.path.join(dirname, self.name), 'r') as f:
			self.data = json.load(f)
		return self