import subprocess
import sys
import shutil
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

EXPERIMENT_DIR = Path(__file__).parent
REPO_DIR = Path.cwd()
assert (REPO_DIR / ".git").exists()

if str(REPO_DIR) not in sys.path:
  sys.path.insert(0, str(REPO_DIR))

from toy_transformers.models import gptv3
from toy_transformers import tokenization, checkpoint

if len(sys.argv) < 2:
  print("usage: python train.py BUCKET_NAME")
  sys.exit(1)

S3_BUCKET = sys.argv[1]
S3_BASE = f"s3://{S3_BUCKET}/toy-transformers/experiments/{EXPERIMENT_DIR.name}"


# CONFIG
VOCAB_SIZE = 4096
BATCH_SIZE = 16
GRAD_ACCUM_STEPS = 2
MODE = tokenization.TokenizationMode.STR
DEVICE = "cuda"

MAX_LR = 3e-4
MIN_LR = 3e-5
WARMUP_STEPS = 500
NUM_EPOCHS = 10

EVAL_INTERVAL = 1000
EVAL_BATCHES = 20
LOG_INTERVAL = 10

config = gptv3.GPTv3Config(
  vocab_size=VOCAB_SIZE,
  block_size=512,
  device=DEVICE,
  n_heads=8,
  n_embed=512,
  n_layers=8
)

DATA_DIR = EXPERIMENT_DIR / "data"
CKPT_DIR = EXPERIMENT_DIR / "checkpoints"
TEMP_CKPT = CKPT_DIR / "temp"


# s3 helpers
def s3_sync_up(local_path: Path, s3_path: str):
  subprocess.run([
    "aws", "s3", "sync", 
    str(local_path), str(s3_path)],
    check=True
  )

def s3_rm(s3_path: str):
  subprocess.run(["aws", "s3", "rm", s3_path, "--recursive"], check=False)

def checkpoint_save_s3(
  local_path: str | Path,
  model,
  model_config,
  training_config: dict,
  metrics: list[dict],
  optimizer=None, scheduler=None
):
  checkpoint.save(
    local_path, model, 
    model_config, training_config, 
    metrics, 
    optimizer=optimizer, scheduler=scheduler
  )
  s3_sync_up(local_path, f"{S3_BASE}/checkpoints/{local_path.name}/")
  print(f">>> synced checkpoint {local_path.name} to S3")


class TokenChunkDataset(Dataset):
  def __init__(self, tokens: torch.Tensor, block_size: int):
    self.tokens = tokens
    self.block_size = block_size
    self.n_chunks = (len(tokens) - 1) // block_size
  
  def __len__(self):
    return self.n_chunks

  def __getitem__(self, idx):
    offset = idx * self.block_size
    x = self.tokens[offset : offset + self.block_size]
    y = self.tokens[offset + 1 : offset + self.block_size + 1]
    return x, y


@torch.no_grad
def estimate_loss(model, loader, n_batches=EVAL_BATCHES):
  model.eval()
  losses = []
  for i, (X, Y) in enumerate(loader):
    if i >= n_batches: break
    X, Y = X.to(DEVICE), Y.to(DEVICE)
    with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
      _, loss = model(X, Y)
    losses.append(loss.item())
  model.train()
  return sum(losses) / len(losses)


def main():
  vocab = tokenization.Vocabulary.load(DATA_DIR / f"vocab_{VOCAB_SIZE}.json")
  
  train_tokens = torch.load(DATA_DIR / "train.pt", weights_only=True)
  val_tokens = torch.load(DATA_DIR / "valid.pt", weights_only=True)
  test_tokens = torch.load(DATA_DIR / "test.pt", weights_only=True)

  block_size = config.block_size
  train_loader = DataLoader(
    TokenChunkDataset(train_tokens, block_size), 
    batch_size=BATCH_SIZE, shuffle=True, drop_last=True, pin_memory=True
  )
  val_loader = DataLoader(
    TokenChunkDataset(val_tokens, block_size), 
    batch_size=BATCH_SIZE, shuffle=False, drop_last=True, pin_memory=True
  )
  test_loader = DataLoader(
    TokenChunkDataset(test_tokens, block_size), 
    batch_size=BATCH_SIZE, shuffle=False, drop_last=True, pin_memory=True
  )

  steps_per_epoch = len(train_loader) // GRAD_ACCUM_STEPS
  TOTAL_STEPS = steps_per_epoch * NUM_EPOCHS

  TRAINING_CONFIG = {
	  "max_lr": MAX_LR,
	  "min_lr": MIN_LR,
	  "warmup_steps": WARMUP_STEPS,
	  "num_epochs": NUM_EPOCHS,
	  "total_steps": TOTAL_STEPS,
	  "steps_per_epoch": steps_per_epoch,
	  "batch_size": BATCH_SIZE,
	  "grad_accum_steps": GRAD_ACCUM_STEPS,
	  "effective_batch_tokens": BATCH_SIZE * GRAD_ACCUM_STEPS * config.block_size,
	  "vocab_path": str(DATA_DIR / f"vocab_{VOCAB_SIZE}.json"),
  }

  print(f"{len(train_loader):,} batches per epoch")
  print(f"{steps_per_epoch:,} steps per epoch")
  print(f"{TOTAL_STEPS:,} total steps")
  print(f"{BATCH_SIZE * GRAD_ACCUM_STEPS * block_size:,} tokens per step")
  print(f"s3 location: {S3_BASE}")

  torch.set_float32_matmul_precision("medium")
  m = gptv3.LanguageModel(config).to(DEVICE)
  print(f"{m.get_num_parameters(as_str=True)} parameters")
  m.compile()

  optimizer = m.get_optimizer(weight_decay=0.1, lr=MAX_LR)
  scheduler = torch.optim.lr_scheduler.SequentialLR(
	  optimizer,
	  schedulers=[
		  torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1/WARMUP_STEPS, total_iters=WARMUP_STEPS
      ),
		  torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=TOTAL_STEPS - WARMUP_STEPS, eta_min=MIN_LR
      ),
	  ],
	  milestones=[WARMUP_STEPS],
  )

  metrics = []
  step = 0
  best_val_loss = float("inf")
  if TEMP_CKPT.exists():
    print("resuming from temp checkpoint...")
    # this only loads local temp checkpoint!!
    loaded_model, _, _, saved_metrics, opt_state, sched_state = checkpoint.load(
      TEMP_CKPT, gptv3.LanguageModel, gptv3.GPTv3Config, device=DEVICE
    )
    m.load_state_dict(loaded_model.state_dict())
    if opt_state: optimizer.load_state_dict(opt_state)
    if sched_state: scheduler.load_state_dict(sched_state)
    if saved_metrics:
      metrics = saved_metrics
      step = metrics[-1].get("step", 0)
      val_losses = [
        row["val_loss"] 
        for row in metrics if "val_loss" in row and row["val_loss"]
      ]
      if val_losses: 
        best_val_loss = min(val_losses)
  else:
    print("starting fresh")

  m.train()
  t0 = time.time()
  start_epoch = step // steps_per_epoch

  try:
    for epoch in range(start_epoch, NUM_EPOCHS):
      print(f"--- epoch {epoch+1}/{NUM_EPOCHS} (step {step:,}/{TOTAL_STEPS:,}) ---")

      micro_batch_buffer = []
      epoch_step = 0
      batches_into_epoch = (step % steps_per_epoch) * GRAD_ACCUM_STEPS

      for xb, yb in train_loader:
        # skip batches already done this epoch when resuming
        epoch_step += 1
        if epoch_step <= batches_into_epoch: continue

        # accumulate items
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        micro_batch_buffer.append((xb, yb))
        if len(micro_batch_buffer) < GRAD_ACCUM_STEPS: continue

        # train batch
        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0

        for micro_xb, micro_yb in micro_batch_buffer:
          with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
            _, loss = m(micro_xb, micro_yb)
          loss = loss / GRAD_ACCUM_STEPS
          loss.backward()
          loss_accum += loss.item()
        
        # update after batch
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        step += 1
        micro_batch_buffer = []

        # logging
        if step % LOG_INTERVAL == 0:
          dt = time.time() - t0
          tokens_per_sec = (LOG_INTERVAL * BATCH_SIZE * GRAD_ACCUM_STEPS * block_size) / dt
          lr = scheduler.get_last_lr()[0]
          row = {
            "step": step, 
            "epoch": epoch+1, 
            "train_loss": loss_accum, 
            "lr": lr
          }
          print(" | ".join([
            f"step {step:5d}",
            f"epoch {epoch+1:2d}",
            f"loss {loss_accum:.4f}",
            f"lr {lr:.2e}",
            f"tok/s {tokens_per_sec:.0f}"
          ]))
          metrics.append(row)
          t0 = time.time()
        
        if step % EVAL_INTERVAL == 0:
          val_loss = estimate_loss(m, val_loader)
          print(f"\t>>> val_loss: {val_loss:.4f}, best: {best_val_loss:.4f}")
          if metrics:
            metrics[-1]["val_loss"] = val_loss
          if val_loss < best_val_loss:
            best_val_loss = val_loss
            # checkpoint_save_s3(
            #   CKPT_DIR / "best", m, 
            #   config, TRAINING_CONFIG,
            #   metrics,
            #   optimizer=optimizer, scheduler=scheduler
            # )
          t0 = time.time()
    
      checkpoint_save_s3(
        CKPT_DIR / f"epoch-{epoch+1}", 
        m, config, TRAINING_CONFIG, 
        metrics, 
        optimizer=optimizer, scheduler=scheduler
      )

  except KeyboardInterrupt:
    print(f"interrupted at step {step}, saving...")
    checkpoint_save_s3(
      TEMP_CKPT, m, 
      config, TRAINING_CONFIG, 
      metrics, 
      optimizer=optimizer, scheduler=scheduler
    )
    print("saved!")
  
  print("finished!")

if __name__ == "__main__":
  main()