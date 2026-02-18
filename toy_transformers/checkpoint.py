import json
import csv
import torch
from dataclasses import asdict
from pathlib import Path

def save(
	path: str | Path, # will save in directory
	model,
	model_config,
	training_config: dict,
	metrics: list[dict],
	optimizer=None, scheduler=None
):
	path = Path(path)
	path.mkdir(parents=True, exist_ok=True)

	meta = {
		"model_type": getattr(model, 'model_type', ''),
		"model_config": asdict(model_config),
		"training": training_config
	}
	with open(path / "metadata.json", "w") as f:
		json.dump(meta, f, indent=4)
	
	ckpt = {"model": model.state_dict()}
	if optimizer is not None:
		ckpt["optimizer"] = optimizer.state_dict()
	if scheduler is not None:
		ckpt["scheduler"] = scheduler.state_dict()
	torch.save(ckpt, path / "model.pt")

	if metrics:
		keys = list(dict.fromkeys(key for row in metrics for key in row.keys()))
		with open(path / "metrics.csv", "w", newline="") as f:
			writer = csv.DictWriter(f, fieldnames=keys)
			writer.writeheader()
			writer.writerows(metrics)


def _default_type_map(row_name, val):
	if not val: return None
	match row_name:
		case "step": return int(val)
		case "train_loss": return float(val)
		case "val_loss": return float(val)
		case _: return float(val)

def load(
	path: str | Path,
	model_cls, 
	config_cls,
	device="cpu",
	metric_type_map: callable = _default_type_map
):
	path = Path(path)

	with open(path / "metadata.json", 'r') as f:
		meta = json.load(f)
	
	config = config_cls(**meta["model_config"])

	ckpt = torch.load(path / "model.pt", map_location=device, weights_only=True)
	model = model_cls(config).to(device)
	model.load_state_dict(ckpt["model"])

	metrics = []
	if (path / "metrics.csv").exists():
		with open(path / "metrics.csv", 'r') as f:
			reader = csv.DictReader(f)
			metrics = [
				{k: metric_type_map(k, v) for k, v in row.items()} 
				for row in reader
			]

	return model, config, meta["training"], metrics, ckpt.get("optimizer"), ckpt.get("scheduler")