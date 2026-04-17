"""Instantiate a model from a config and print its param count.

Usage: python count_params.py configs/v4.0.lr_batch1.json [...]
"""
import sys
from pathlib import Path

from toy_transformers.config import TrainingConfig

ROOT = Path(__file__).parent

for path in sys.argv[1:]:
    cfg = TrainingConfig.from_json(path)
    cfg.tokenizer.load(ROOT)
    model = cfg.model.build_model(vocab_size=cfg.tokenizer.vocab_size, device="cpu")
    m = cfg.model.config
    print(f"{path}")
    print(f"  d={m['n_embed']} L={m['n_layers']} H={m['n_heads']} "
          f"KV={m.get('n_kv_heads', m['n_heads'])} d/L={m['n_embed']/m['n_layers']:.1f}")
    print(f"  params: {model.get_num_parameters(as_str=True)} "
          f"({model.get_num_parameters():,})\n")
