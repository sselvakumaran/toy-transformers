import json
import torch
from dataclasses import dataclass, asdict
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
	device: str = "cuda"
):
	path = Path(path)
	ckpt = torch.load(path / "model.pt", map_location=device, weights_only=True)
	model = cfg.model.build_model(
		vocab_size=cfg.tokenizer.vocab_size,
		device=device
	)

	model.load_state_dict(ckpt["model"])

	return model, ckpt.get("optimizer"), ckpt.get("scheduler")


# --- SAVING / LOADING STATUS ---
_STATUS_FN = "status.json"

@dataclass
class RunStatus:
	step: int = 0
	shards_consumed: int = 0
	best_val_loss: float = float("inf")
	status: str = "running"

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