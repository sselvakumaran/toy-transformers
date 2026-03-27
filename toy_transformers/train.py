import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from toy_transformers.config import TrainingConfig
from toy_transformers.data import S3Sync, S3ShardDownloader, AggregateDataset, ShardDataset
from toy_transformers.model_io import MetricsWriter, RunStatus, load_model, save_model


REPO_ROOT = Path(__file__).parent.parent
RUNS_DIR = REPO_ROOT / "runs"
DATA_DIR = REPO_ROOT / "data"


def setup_data(cfg: TrainingConfig, sync: S3Sync) -> tuple[dict, Path]:
	sync.pull_atomic(cfg.tokenizer.path)
	cfg.tokenizer.load(REPO_ROOT)

	metadatas = dict()
	for folder in cfg.dataset.dataset_folders:
		path = sync.pull_atomic(f"data/datasets/{folder}/metadata.json")
		metadatas[folder] = json.loads(path.read_text())
		print("[SETUP]", f"{folder}: {len(metadatas[folder]['train_shards'])} training shards")
	
	primary = cfg.dataset.dataset_folders[0]
	val_name = metadatas[primary]["val_shard"]
	print("[SETUP]", "pulling val shard...")
	val_path = sync.pull_atomic(f"data/datasets/{primary}/{val_name}")
	return metadatas, val_path

def compute_total_steps(cfg: TrainingConfig) -> int:
	if cfg.tokens.train_tokens <= 0:
		raise ValueError("train_tokens invalid")
	return cfg.tokens.train_tokens // cfg.tokens_per_step

def setup_model(cfg: TrainingConfig, total_steps: int, device: str):
	torch.set_float32_matmul_precision("medium")
	model = cfg.model.build_model(vocab_size=cfg.tokenizer.vocab_size, device=device)
	print("[SETUP]", f"{model.get_num_parameters(as_str=True)} parameters")
	model.compile()
	optimizer = cfg.optimizer.build_optimizer(model)
	scheduler = cfg.optimizer.build_scheduler(optimizer, total_steps)
	return model, optimizer, scheduler

def maybe_resume(run_dir: Path, cfg, model, optimizer, scheduler, sync: S3Sync, device: str) -> RunStatus:
	temp_ckpt = run_dir / "checkpoints/temp"
	s3_run = f"runs/{cfg.run.name}"

	status_local = (run_dir / "status.json").exists()
	status_s3 = sync.exists(f"{s3_run}/status.json")
	print("[RESUME]", f"status.json local={status_local}, s3={status_s3}")
	if not status_local and status_s3:
		ok = sync.pull(f"{s3_run}/status.json")
		print("[RESUME]", f"pulled status.json: {ok}, exists now: {(run_dir / 'status.json').exists()}")
	status = RunStatus.load(run_dir)
	print("[RESUME]", f"loaded status: step={status.step}, shards_consumed={status.shards_consumed}")

	ckpt_local = temp_ckpt.exists()
	ckpt_s3 = sync.exists(f"{s3_run}/checkpoints/temp")
	print("[RESUME]", f"temp ckpt local={ckpt_local}, s3={ckpt_s3}")
	if not ckpt_local and status.step > 0 and ckpt_s3:
		print("[RESUME]", "pulling temp checkpoint from S3...")
		sync.pull(f"{s3_run}/checkpoints/temp")

	if temp_ckpt.exists() and status.step > 0:
		print("[RESUME]", f"resuming, step={status.step}, shards_consumed={status.shards_consumed}")
		load_model(temp_ckpt, cfg, model=model, optimizer=optimizer, scheduler=scheduler, device=device)
	else:
		print("[RESUME]", "starting fresh")
		status = RunStatus()
	return status


@torch.no_grad()
def estimate_loss(model, loader, n_batches: int, device: str) -> float:
	losses = []
	for i, (x, y, doc_ids, loss_mask) in enumerate(loader):
		if i >= n_batches: break
		x, y = x.to(device), y.to(device)
		doc_ids, loss_mask = doc_ids.to(device), loss_mask.to(device)
		with torch.autocast(device_type=device, dtype=torch.bfloat16):
			_, loss = model(x, y, doc_ids=doc_ids, loss_mask=loss_mask)
		losses.append(loss.item())
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
			d.join(timeout=3)
			if d.is_alive(): d.terminate()
	
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
				metrics.flush()
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
				sync.push(f"runs/{cfg.run.name}/metrics.jsonl")
				if is_best:
					status.update(run_dir, best_val_loss=min(val_loss, status.best_val_loss))
				t0 = time.time()
			
			# checkpoint
			if step % cfg.run.save_interval == 0:
				status.update(
					run_dir, step=step, shards_consumed=train_loader.dataset.shards_consumed,
					dataset_shards={
						folder: train_loader.dataset.source_shards_consumed[i]
						for i, folder in enumerate(cfg.dataset.dataset_folders)
					})
				save_checkpoint(f"step-{step}")
			
			if step >= total_steps: break
	
	except KeyboardInterrupt:
		print("[CLEANUP]", f"interrupted at step {step}")
		status.update(run_dir, step=step, shards_consumed=train_loader.dataset.shards_consumed, status="running", 
			dataset_shards={
				folder: train_loader.dataset.source_shards_consumed[i]
				for i, folder in enumerate(cfg.dataset.dataset_folders)
		})
		save_checkpoint("temp")
		metrics.close()
		sync.push(f"runs/{cfg.run.name}/metrics.jsonl")
		sync.push(f"runs/{cfg.run.name}/status.json")
		stop_downloaders()
		return
	
	metrics.close()
	stop_downloaders()
	
	status.update(run_dir, step=step, shards_consumed=train_loader.dataset.shards_consumed, status="completed",
		dataset_shards={
			folder: train_loader.dataset.source_shards_consumed[i]
			for i, folder in enumerate(cfg.dataset.dataset_folders)
	})
	save_checkpoint(f"final")
	print("[TRAIN]", "finished!")


def train_from_config(cfg: TrainingConfig, bucket: str, device: str = "cuda"):
	run_dir = RUNS_DIR / cfg.run.name
	run_dir.mkdir(parents=True, exist_ok=True)
	# cfg.to_json(run_dir / "config.json")
	
	sync = S3Sync(remote_base=f"s3://{bucket}/toy-transformers", local_root=REPO_ROOT)
	print("[SETUP]", f"connected {sync.remote_base} <-> {REPO_ROOT}")
	
	metadatas, val_path = setup_data(cfg, sync)
	total_steps = compute_total_steps(cfg)
	
	model, optimizer, scheduler = setup_model(cfg, total_steps, device)
	status = maybe_resume(run_dir, cfg, model, optimizer, scheduler, sync, device)

	downloaders = []
	sources = []
	for folder, w in zip(cfg.dataset.dataset_folders, cfg.dataset.dataset_weights):
		shards = [f"data/datasets/{folder}/{s}" for s in metadatas[folder]["train_shards"]]
		q = mp.Queue(maxsize=4)
		sources.append((q, w))
		skip = status.dataset_shards.get(folder, 0)
		downloader = S3ShardDownloader(
			sync=sync, shards=shards,
			queue=q,
			shuffle=True, seed=cfg.run.seed,
			skip_shards=skip
		)
		downloader.start()
		downloaders.append(downloader)
	
	block_size = cfg.model.config["block_size"]
	train_dataset = AggregateDataset(
		sources=sources,
		block_size=block_size,
		bos_id=cfg.tokenizer.bos_id, pad_id=cfg.tokenizer.pad_id,
		shuffle_docs=True, seed=cfg.run.seed,
	)
	val_dataset = ShardDataset(
		shard_paths=[val_path],
		block_size=block_size,
		bos_id=cfg.tokenizer.bos_id, pad_id=cfg.tokenizer.pad_id,
		shuffle=False, seed=cfg.run.seed,
	)

	train_loader = DataLoader(train_dataset, batch_size=cfg.tokens.batch_size, num_workers=0)
	val_loader = DataLoader(val_dataset, batch_size=cfg.tokens.batch_size, num_workers=0, drop_last=True)

	train(
		cfg, model, optimizer, scheduler,
		train_loader, val_loader, total_steps, status,
		run_dir, sync, downloaders, device
	)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("config", help="path to config JSON")
	parser.add_argument("bucket", help="S3 bucket/base name, can include subdirectories")
	parser.add_argument("--device", default="cuda")
	args = parser.parse_args()
	
	cfg = TrainingConfig.from_json(args.config)
	train_from_config(cfg=cfg, bucket=args.bucket, device=args.device)