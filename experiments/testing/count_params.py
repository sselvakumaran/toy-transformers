"""Compute a gpt-v4 model's param count analytically from its config.

Usage: python count_params.py configs/v4.0.lr_batch1.json [...]
"""
import sys
from pathlib import Path

from toy_transformers.config import TrainingConfig
from toy_transformers.models.gptv4 import GPTv4Config

ROOT = Path.cwd()

def count_params(cfg: GPTv4Config, vocab_size: int) -> int:
    V = vocab_size
    d = cfg.n_embed
    L = cfg.n_layers
    H = cfg.n_heads
    K = cfg.n_kv_heads
    m = cfg.mlp_mul
    h = d // H  # head_dim

    # per-block: attn (q, kv, out) + 2 rmsnorms + mlp (l1, proj[, l1_gate])
    per_block = (d * d) + (d * 2 * K * h) + (d * d) + 2 * d + (d * m * d) + (m * d * d)
    if cfg.activation_fn == "swiglu":
        per_block += d * m * d

    # token_embed (tied with head) + final rmsnorm + blocks
    return V * d + d + L * per_block


def fmt(n: int) -> str:
    for t, s in [(10 ** 9, "b"), (10 ** 6, "m"), (10 ** 3, "k")]:
        if n > t:
            return f"{n / t:.3f}{s}"
    return str(n)


for path in sys.argv[1:]:
    tcfg = TrainingConfig.from_json(path)
    print(ROOT)
    tcfg.tokenizer.load(ROOT)
    cfg = GPTv4Config(vocab_size=tcfg.tokenizer.vocab_size, **tcfg.model.config)
    n = count_params(cfg, vocab_size=tcfg.tokenizer.vocab_size)
    print(f"{path}")
    print(f"  d={cfg.n_embed} L={cfg.n_layers} H={cfg.n_heads} "
          f"KV={cfg.n_kv_heads} d/L={cfg.n_embed / cfg.n_layers:.1f}")
    print(f"  params: {fmt(n)} ({n:,})\n")
