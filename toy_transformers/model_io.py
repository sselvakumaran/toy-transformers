import json
import torch
from dataclasses import dataclass, asdict, field
from pathlib import Path

from toy_transformers.config import TrainingConfig

# --- SAVING / LOADING MODELS ---

def save_model(
	path: str | Path, # will save in directory
	model,
	optimizer=None, scheduler=None
):
	path = Path(path)
	path.mkdir(parents=True, exist_ok=True)
	
	ckpt = {"model": model.state_dict()}
	if optimizer is not None:
		ckpt["optimizer"] = optimizer.state_dict()
	if scheduler is not None:
		ckpt["scheduler"] = scheduler.state_dict()
	
	torch.save(ckpt, path / "model.pt")

def load_model(
	path: str | Path,
	cfg: TrainingConfig,
	model, optimizer = None, scheduler = None, 
	vocab_size: int | None = None,
	device: str = "cuda"
):
	path = Path(path)
	ckpt = torch.load(path / "model.pt", map_location=device, weights_only=True)
	model.load_state_dict(ckpt["model"])
	if optimizer: optimizer.load_state_dict(ckpt["optimizer"])
	if scheduler: scheduler.load_state_dict(ckpt["scheduler"])


# --- SAVING / LOADING STATUS ---
_STATUS_FN = "status.json"

@dataclass
class RunStatus:
	step: int = 0
	shards_consumed: int = 0
	best_val_loss: float = float("inf")
	status: str = "running"
	dataset_shards: dict = field(default_factory=dict)

	@classmethod
	def load(cls, run_dir: Path | str):
		path = Path(run_dir) / _STATUS_FN
		if not path.exists():
			return cls()
		raw = json.loads(path.read_text())
		return cls(**raw)
	
	def save(self, run_dir: Path | str):
		path = Path(run_dir) / _STATUS_FN
		path.parent.mkdir(parents=True, exist_ok=True)
		path.write_text(json.dumps(asdict(self), indent=2))
	
	def update(self, 
		run_dir: Path | str, 
		step: int | None = None, 
		shards_consumed: int | None = None, 
		best_val_loss: float | None = None, 
		status: str | None = None,
		dataset_shards: dict | None = None
	):
		if step is not None: self.step = step
		if shards_consumed is not None: self.shards_consumed = shards_consumed
		if best_val_loss is not None: self.best_val_loss = best_val_loss
		if status is not None: self.status = status
		if dataset_shards is not None: self.dataset_shards = dataset_shards
		self.save(run_dir)


# --- SAVING / LOADING METRICS ---
_METRICS_FN = "metrics.jsonl"

class MetricsWriter:
	def __init__(self, run_dir: Path | str):
		self._path = Path(run_dir) / _METRICS_FN
		self._path.parent.mkdir(parents=True, exist_ok=True)
		self._f = open(self._path, "a")
	
	def write(self, row: dict):
		self._f.write(json.dumps(row) + "\n")
	
	def flush(self):
		self._f.flush()
	
	def close(self):
		self._f.close()
	
	def __enter__(self):
		return self
	
	def __exit__(self, *_):
		self.close()

def read_metrics(run_dir: Path | str) -> list[dict]:
	path = Path(run_dir) / _METRICS_FN
	if not path.exists():
		return []
	return [json.loads(line) for line in path.read_text().splitlines() if not line.isspace()]