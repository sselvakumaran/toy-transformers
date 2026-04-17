"""Probe the largest micro batch that fits for a given config.

Usage: python probe_batch.py configs/v4.0.1.bs256.json [--max 64]

Does a doubling search, runs 2 optimizer steps per candidate (AdamW state
only allocates after the first step, so step 2 is the real memory peak).
"""
import argparse
import gc
from pathlib import Path

import torch

from toy_transformers.config import TrainingConfig

ROOT = Path(__file__).parent


def try_batch(cfg: TrainingConfig, micro: int) -> bool:
    torch.cuda.empty_cache()
    gc.collect()
    model = cfg.model.build_model(vocab_size=cfg.tokenizer.vocab_size, device="cuda")
    opt = cfg.optimizer.build_optimizer(model)
    T = cfg.model.config["block_size"]
    V = cfg.tokenizer.vocab_size
    try:
        for _ in range(2):
            x = torch.randint(0, V, (micro, T), device="cuda")
            y = torch.randint(0, V, (micro, T), device="cuda")
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss = model(x, targets=y)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        print(f"  micro={micro:<4d} OK   peak={peak_gb:.1f} GB")
        return True
    except torch.cuda.OutOfMemoryError:
        print(f"  micro={micro:<4d} OOM")
        return False
    finally:
        del model, opt
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("--max", type=int, default=64)
    args = p.parse_args()

    cfg = TrainingConfig.from_json(args.config)
    cfg.tokenizer.load(ROOT)

    m = cfg.model.config
    print(f"{args.config}  d={m['n_embed']} L={m['n_layers']} T={m['block_size']}")

    # doubling phase
    last_ok = 0
    micro = 1
    while micro <= args.max:
        if try_batch(cfg, micro):
            last_ok = micro
            micro *= 2
        else:
            break

    print(f"\nmax safe micro batch (doubling): {last_ok}")
    print("→ refine with linear search between last_ok and next double if you want tighter bound")
