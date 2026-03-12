import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from toy_transformers.config import TrainingConfig
from toy_transformers.data import S3Sync, S3ShardDownloader, ShardedTokenDataset
from toy_transformers.model_io import MetricsWriter, RunStatus, load_model, save_model


REPO_ROOT = Path(__file__).parent.parent
RUNS_DIR = REPO_ROOT / "runs"
DATA_DIR = REPO_ROOT / "data"


def setup_data(cfg: TrainingConfig, sync: S3Sync) -> tuple[dict, Path]:
	cfg.tokenizer.load(DATA_DIR)

	metadatas = dict()
	for folder in cfg.dataset.dataset_folders:
		path = sync.pull_atomic(f"data/datasets/{folder}/metadata.json")
		metadatas[folder] = json.loads(path.read_text())
		print("[DATA]", f"{folder}: {len(metadatas[folder]['train_shards'])} training shards")
	
	primary = cfg.dataset.dataset_folders[0]
	val_name = metadatas[primary]["val_shard"]
	print("[DATA]", "pulling val shard...")
	val_path = sync.pull_atomic(f"data/datasets/{primary}/{val_name}")
	return metadatas, val_path

def compute_total_steps(cfg: TrainingConfig) -> int:
	if cfg.tokens.train_tokens <= 0:
		raise ValueError("train_tokens invalid")
	return cfg.tokens.train_tokens // cfg.tokens_per_step

def setup_model(cfg: TrainingConfig, total_steps: int, device: str):
	torch.set_float32_matmul_precision("medium")
	model = cfg.model.build_model(vocab_size=cfg.tokenizer.vocab_size, device=device)
	print("[MODEL]", f"{model.get_num_parameters(as_str=True)} parameters")
	model.compile()
	optimizer = cfg.optimizer.build_optimizer(model)
	scheduler = cfg.optimizer.build_scheduler(optimizer, total_steps)
	return model, optimizer, scheduler

def maybe_resume(run_dir: Path, model, optimizer, scheduler, device: str) -> RunStatus:
	temp_ckpt = run_dir / "checkpoints/temp"
	status = RunStatus.load(run_dir)

	if temp_ckpt.exists() and status.step > 0:
		print("[RESUME]", f"step={status.step}, shards_consumed={status.shards_consumed}")
		_, opt_state, sched_state = load_model(temp_ckpt, cfg, model=model, device=device)
		ckpt = load_model(temp_ckpt, )
		if opt_state: optimizer.load_state_dict(opt_state)
		if sched_state: scheduler.load_state_dict(sched_state)
	else:
		print("[TRAIN]", "starting fresh")
		status = RunStatus()
	
	return status


@torch.no_grad()
def estimate_loss(model, loader, n_batches: int, device: str) -> float:
	model.eval()
	losses = []
	for i, (x, y, doc_ids, loss_mask) in enumerate(loader):
		if i >= n_batches: break
		x, y = x.to(device), y.to(device)
		doc_ids, loss_mask = doc_ids.to(device), loss_mask.to(device)
		with torch.autocast(device_type=device, dtype=torch.bfloat16):
			_, loss = model(x, y, doc_ids=doc_ids, loss_mask=loss_mask)
		losses.append(loss.item())
	model.train()
	return sum(losses) / len(losses) if losses else float("inf")

def train(
	cfg: TrainingConfig,
	model, optimizer, scheduler,
	train_loader: DataLoader,
	val_loader: DataLoader,
	total_steps: int,
	status: RunStatus,
	run_dir: Path,
	sync: S3Sync,
	downloaders: list[S3ShardDownloader],
	device: str
):
	print("[TRAIN]", f"{total_steps:,} total steps, {cfg.tokens_per_step:,} tokens/step")

	def save_checkpoint(name: str):
		path = run_dir / "checkpoints" / name
		save_model(path, model, optimizer, scheduler)
		sync.push(f"runs/{cfg.run.name}/checkpoints/{name}")
		print("[CKPT]", f"saved + synced: {name}")
	
	def stop_downloaders():
		for d in downloaders:
			d.terminate()
	
	step = status.step
	micro_buffer = []
	t0 = time.time()
	metrics = MetricsWriter(run_dir)
	try:
		for x, y, doc_ids, loss_mask in train_loader:
			x, y = x.to(device), y.to(device)
			doc_ids, loss_mask = doc_ids.to(device), loss_mask.to(device)

			micro_buffer.append((x, y, doc_ids, loss_mask))
			if len(micro_buffer) < cfg.tokens.grad_accum_steps:
				continue
			
			optimizer.zero_grad(set_to_none=True)
			loss_accum = 0.0
			for mx, my, mdoc, mmask in micro_buffer:
				with torch.autocast(device_type=device, dtype=torch.bfloat16):
					_, loss = model(mx, my, doc_ids=mdoc, loss_mask=mmask)
				loss = loss / cfg.tokens.grad_accum_steps
				loss.backward()
				loss_accum += loss.item()
			torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
			optimizer.step()
			scheduler.step()
			step += 1
			micro_buffer = []

			# logging
			if step % cfg.run.log_interval == 0:
				dt = time.time() - t0
				tok_per_sec = (cfg.run.log_interval * cfg.tokens_per_step) / dt
				lr = scheduler.get_last_lr()[0]
				metrics.write({
					"step": step, "t_loss": round(loss_accum, 6),
					"lr": lr, "tokens": step * cfg.tokens_per_step
				})
				print("[TRAIN]", " | ".join([
					f"step {step:5d}/{total_steps}",
					f"loss {loss_accum:.4f}",
					f"lr {lr:.2e}",
					f"tok/s {tok_per_sec:.0f}",
					f"shards {train_loader.dataset.shards_consumed}"
				]))
				t0 = time.time()
			
			# eval
			if step % cfg.eval.interval == 0:
				val_loss = estimate_loss(model, val_loader, cfg.eval.batches, device)
				is_best = val_loss < status.best_val_loss
				print("[TRAIN]", f"\tval_loss {val_loss:.4f} {'(best)' if is_best else f'(best {status.best_val_loss:.4f})'}")
				metrics.write({
					"step": step, "v_loss": round(val_loss, 6),
					"tokens": step * cfg.tokens_per_step
				})
				metrics.flush()
				if is_best:
					status.update(run_dir, best_val_loss=min(val_loss, status.best_val_loss))
				t0 = time.time()
			
			# checkpoint
			if step % cfg.run.save_interval == 0:
				status.update(run_dir, step=step, shards_consumed=train_loader.dataset.shards_consumed)
				save_checkpoint(f"step-{step}")
			
			if step >= total_steps: break
	
	except KeyboardInterrupt:
		print("[CLEANUP]", f"interrupted at step {step}")
		status.update(run_dir, step=step, shards_consumed=train_loader.dataset.shards_consumed, status="running")
		save_checkpoint("temp")
		stop_downloaders()
		return
	
	metrics.close()

	for d in downloaders:
		d.join(timeout=5)
		if d.is_alive(): d.terminate()
	
	status.update(run_dir, step=step, shards_consumed=train_loader.dataset.shards_consumed, status="completed")
	save_checkpoint(f"final")
	print("[TRAIN]", "finished!")


def train_from_config(cfg: Path, bucket: str, device: str = "cuda"):
	pass

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("config", help="path to config JSON")
	parser.add_argument("bucket", "S3 bucket/base name, can include subdirectories")
	parser.add_argument("--device", default="cuda")
	args = parser.parse_args()
	
	cfg = Path(args.config)
	assert cfg.exists()

	train_from_config(cfg=cfg, bucket=args.bucket, device=args.device)