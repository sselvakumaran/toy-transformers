import multiprocessing as mp
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

EXPERIMENT_DIR = Path(__file__).parent
REPO_DIR = Path.cwd()
assert (REPO_DIR / ".git").exists()

if str(REPO_DIR) not in sys.path:
  sys.path.insert(0, str(REPO_DIR))
if str(EXPERIMENT_DIR) not in sys.path:
  sys.path.insert(0, str(EXPERIMENT_DIR))

from toy_transformers.models import gptv4
from toy_transformers import checkpoint
from data_loader import S3Sync, S3ShardDownloader, ShardedTokenDataset

if len(sys.argv) < 2:
  print("usage: python train.py BUCKET_NAME")
  sys.exit(1)

S3_BUCKET = sys.argv[1]
S3_BASE = f"s3://{S3_BUCKET}/toy-transformers/experiments/{EXPERIMENT_DIR.name}"

# CONFIG
VOCAB_SIZE = 32768
BLOCK_SIZE = 1024
BATCH_SIZE = 16
GRAD_ACCUM_STEPS = 4
DEVICE = "cuda"

MAX_LR = 3e-4
MIN_LR = 3e-5
WARMUP_STEPS = 1000
NUM_EPOCHS = 1

EVAL_INTERVAL = 500
EVAL_BATCHES = 20
LOG_INTERVAL = 10
SAVE_INTERVAL = 5000

BOS_ID = 0  # split_id from metadata — BOS token prepended to each document
PAD_ID = 1  # padding token for incomplete packs

# 78 total shards; reserve last two for val/test
NUM_SHARDS = 78
VAL_SHARD  = f"shard_{NUM_SHARDS - 2:04d}.bin"  # shard_0076.bin
TEST_SHARD = f"shard_{NUM_SHARDS - 1:04d}.bin"  # shard_0077.bin (unused during training)
TRAIN_SHARDS = [f"shard_{i:04d}.bin" for i in range(NUM_SHARDS - 2)]

# ~127M tokens/shard, 76 train shards → estimate total train steps
TOKENS_PER_SHARD = 127_000_000
TRAIN_TOKENS = TOKENS_PER_SHARD * len(TRAIN_SHARDS)
TOKENS_PER_STEP = BATCH_SIZE * GRAD_ACCUM_STEPS * BLOCK_SIZE
TOTAL_STEPS = TRAIN_TOKENS // TOKENS_PER_STEP

model_config = gptv4.GPTv4Config(
  vocab_size=VOCAB_SIZE,
  block_size=BLOCK_SIZE,
  device=DEVICE,
  n_heads=12,
  n_embed=768,
  n_layers=12,
  n_kv_heads=4,
)

DATA_DIR = EXPERIMENT_DIR / "data"
SHARD_DIR = DATA_DIR / "shuffled"
CKPT_DIR = EXPERIMENT_DIR / "checkpoints"
TEMP_CKPT = CKPT_DIR / "temp"

TRAINING_CONFIG = {
  "max_lr": MAX_LR,
  "min_lr": MIN_LR,
  "warmup_steps": WARMUP_STEPS,
  "num_epochs": NUM_EPOCHS,
  "total_steps": TOTAL_STEPS,
  "batch_size": BATCH_SIZE,
  "grad_accum_steps": GRAD_ACCUM_STEPS,
  "effective_batch_tokens": TOKENS_PER_STEP,
  "block_size": BLOCK_SIZE,
  "bos_id": BOS_ID,
  "pad_id": PAD_ID,
  "s3_base": S3_BASE,
}


@torch.no_grad()
def estimate_loss(model, loader, n_batches=EVAL_BATCHES):
  model.eval()
  losses = []
  for i, (x, y, doc_ids, loss_mask) in enumerate(loader):
    if i >= n_batches:
      break
    x, y = x.to(DEVICE), y.to(DEVICE)
    doc_ids = doc_ids.to(DEVICE)
    loss_mask = loss_mask.to(DEVICE)
    with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
      _, loss = model(x, y, doc_ids=doc_ids, loss_mask=loss_mask)
    losses.append(loss.item())
  model.train()
  return sum(losses) / len(losses)


def main():
  data_sync = S3Sync(
    remote_base=f"{S3_BASE}/data/shuffled",
    local_root=SHARD_DIR,
  )
  ckpt_sync = S3Sync(
    remote_base=f"{S3_BASE}/checkpoints",
    local_root=CKPT_DIR,
  )

  # pull val shard upfront
  print("[DATA] pulling val shard...")
  SHARD_DIR.mkdir(parents=True, exist_ok=True)
  val_shard_path = data_sync.pull_atomic(SHARD_DIR / VAL_SHARD)
  val_dataset = ShardedTokenDataset(
    shard_paths=[val_shard_path],
    block_size=BLOCK_SIZE,
    bos_id=BOS_ID,
    pad_id=PAD_ID,
    shuffle_docs=False,
  )
  val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=0)

  # model setup
  torch.set_float32_matmul_precision("medium")
  m = gptv4.LanguageModel(model_config).to(DEVICE)
  print(f"{m.get_num_parameters(as_str=True)} parameters")
  m.compile()

  optimizer = m.get_optimizer(weight_decay=0.1, lr=MAX_LR)
  scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer,
    schedulers=[
      torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1 / WARMUP_STEPS, total_iters=WARMUP_STEPS
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
  shards_consumed = 0

  if TEMP_CKPT.exists():
    print("resuming from temp checkpoint...")
    loaded_model, _, saved_training, saved_metrics, opt_state, sched_state = checkpoint.load(
      TEMP_CKPT, gptv4.LanguageModel, gptv4.GPTv4Config, device=DEVICE
    )
    m.load_state_dict(loaded_model.state_dict())
    if opt_state:
      optimizer.load_state_dict(opt_state)
    if sched_state:
      scheduler.load_state_dict(sched_state)
    if saved_metrics:
      metrics = saved_metrics
      step = metrics[-1].get("step", 0)
      val_losses = [
        row["val_loss"]
        for row in metrics if "val_loss" in row and row["val_loss"]
      ]
      if val_losses:
        best_val_loss = min(val_losses)
    shards_consumed = saved_training.get("shards_consumed", 0)
    print(f"resumed at step {step}, shards_consumed={shards_consumed}")
  else:
    print("starting fresh")

  # start background shard downloader
  queue: mp.Queue = mp.Queue(maxsize=4)
  downloader = S3ShardDownloader(
    sync=data_sync,
    shard_dir=SHARD_DIR,
    queue=queue,
    num_epochs=NUM_EPOCHS,
    shuffle=True,
    seed=42,
    start_epoch=0,
    skip_shards=shards_consumed,
  )
  # restrict downloader to training shards only
  downloader.shards = [s for s in downloader.shards if s in TRAIN_SHARDS]
  downloader.start()

  train_dataset = ShardedTokenDataset(
    queue=queue,
    block_size=BLOCK_SIZE,
    bos_id=BOS_ID,
    pad_id=PAD_ID,
    shuffle_docs=True,
    cleanup=True,
  )
  train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=0)

  print(f"~{TOTAL_STEPS:,} total steps estimated")
  print(f"{TOKENS_PER_STEP:,} tokens per step")
  print(f"s3 location: {S3_BASE}")

  m.train()
  t0 = time.time()
  micro_batch_buffer = []

  def save_checkpoint(name):
    path = CKPT_DIR / name
    tc = {**TRAINING_CONFIG, "shards_consumed": train_dataset.shards_consumed}
    checkpoint.save(path, m, model_config, tc, metrics, optimizer=optimizer, scheduler=scheduler)
    ckpt_sync.push(path)
    print(f">>> synced checkpoint {name} to S3")

  try:
    for x, y, doc_ids, loss_mask in train_loader:
      x, y = x.to(DEVICE), y.to(DEVICE)
      doc_ids = doc_ids.to(DEVICE)
      loss_mask = loss_mask.to(DEVICE)

      micro_batch_buffer.append((x, y, doc_ids, loss_mask))
      if len(micro_batch_buffer) < GRAD_ACCUM_STEPS:
        continue

      optimizer.zero_grad(set_to_none=True)
      loss_accum = 0.0

      for micro_x, micro_y, micro_doc_ids, micro_loss_mask in micro_batch_buffer:
        with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
          _, loss = m(micro_x, micro_y, doc_ids=micro_doc_ids, loss_mask=micro_loss_mask)
        loss = loss / GRAD_ACCUM_STEPS
        loss.backward()
        loss_accum += loss.item()

      torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
      optimizer.step()
      scheduler.step()
      step += 1
      micro_batch_buffer = []

      if step % LOG_INTERVAL == 0:
        dt = time.time() - t0
        tokens_per_sec = (LOG_INTERVAL * TOKENS_PER_STEP) / dt
        lr = scheduler.get_last_lr()[0]
        row = {
          "step": step,
          "train_loss": loss_accum,
          "lr": lr,
        }
        print(" | ".join([
          f"step {step:6d}",
          f"loss {loss_accum:.4f}",
          f"lr {lr:.2e}",
          f"tok/s {tokens_per_sec:.0f}",
          f"shards {train_dataset.shards_consumed}",
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
        t0 = time.time()

      if step % SAVE_INTERVAL == 0:
        save_checkpoint(f"step-{step}")

  except KeyboardInterrupt:
    print(f"interrupted at step {step}, saving...")
    save_checkpoint("temp")
    print("saved!")
    downloader.terminate()
    return

  downloader.join()
  save_checkpoint(f"final-step-{step}")
  print("finished!")


if __name__ == "__main__":
  main()
