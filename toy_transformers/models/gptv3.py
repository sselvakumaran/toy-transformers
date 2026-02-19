from dataclasses import dataclass
from typing import Dict, Literal
import torch
import torch.nn as nn
import torch.nn.functional as F

# Version 3 - RMSNorm, RoPE, ReLU^2, QK-Norm, logit soft-capping

@dataclass(frozen=True)
class GPTv3Config:
  vocab_size: int
  block_size: int
  device: str = "cpu"
  n_heads: int = 8 # number of embedding heads (n_embed / n_heads MUST be an integer)
  n_embed: int = 288 # number of dimensions in embedding vector
  n_layers: int = 6 # number of blocks
  checkpoint: bool = False


class CausalSelfAttention(nn.Module):
  def __init__(self, config: GPTv3Config):
    super().__init__()
    self.n_heads, self.n_embed = config.n_heads, config.n_embed
    if self.n_embed % self.n_heads != 0:
      raise ValueError("n_embed is not a multiple of n_heads")
    # batch k,q,v into one linear block
    # when doing forward pass, each head will be per batch
    # note n_embed = n_heads * <some factor>
    self.qkv_block = nn.Linear(self.n_embed, 3 * self.n_embed)
    self.proj = nn.Linear(config.n_embed, config.n_embed)
    self.proj.__INIT_SCALAR__ = (2 * config.n_layers) ** -0.5

  def forward(self, x: torch.Tensor):
    # get k,q,v linear block
    B, T, _ = x.size() # assuming C == self.n_embed
    n_head_embed = self.n_embed // self.n_heads
    qkv: torch.Tensor = self.qkv_block(x) # dimension (B, T, 3*C)
    q_joined, k_joined, v_joined = qkv.split(self.n_embed, dim=-1)
    # view must be (B, n_heads, T, n_head_embed) needed for scaled_dot_product_attention
    q = q_joined.view(B, T, self.n_heads, n_head_embed).transpose(1, 2)
    k = k_joined.view(B, T, self.n_heads, n_head_embed).transpose(1, 2)
    v = v_joined.view(B, T, self.n_heads, n_head_embed).transpose(1, 2)
    y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    y_joined = y.transpose(1, 2).contiguous().view(B, T, self.n_embed)
    y_proj = self.proj(y_joined)
    return y_proj


class MultiLayerPerceptron(nn.Module):
  def __init__(self, config: GPTv3Config):
    super().__init__()
    self.l1 = nn.Linear(config.n_embed, 4*config.n_embed)
    self.gelu = nn.GELU(approximate='tanh')
    self.proj = nn.Linear(4*config.n_embed, config.n_embed)
    self.proj.__INIT_SCALAR__ = (2 * config.n_layers) ** -0.5

  def forward(self, x: torch.Tensor):
    x = self.l1(x)
    x = self.gelu(x)
    x = self.proj(x)
    return x


class TransformerBlock(nn.Module):
  def __init__(self, config: GPTv3Config):
    super().__init__()
    self.norm1 = nn.RMSNorm(config.n_embed)
    self.attn = CausalSelfAttention(config)
    self.norm2 = nn.RMSNorm(config.n_embed)
    self.mlp = MultiLayerPerceptron(config)

  def forward(self, x: torch.Tensor):
    # residual connections
    attn_add = self.attn(self.norm1(x))
    x = torch.add(x, attn_add)
    mlp_add = self.mlp(self.norm2(x))
    x = torch.add(x, mlp_add)
    return x


class LanguageModel(nn.Module):
  model_type: str = 'gpt-v3'

  def __init__(self, config: GPTv3Config):
    super().__init__()
    self.config = config

    vocab_size = config.vocab_size
    block_size = config.block_size

    self.token_embed = nn.Embedding(vocab_size, config.n_embed)
    self.position_embed = nn.Embedding(block_size, config.n_embed)
    self.blocks = nn.Sequential(*[TransformerBlock(config) for _ in range(config.n_layers)])
    self.ln = nn.RMSNorm(config.n_embed)
    self.head = nn.Linear(config.n_embed, vocab_size, bias=False)

    self.register_buffer("pos", torch.arange(block_size).unsqueeze(0))

    def _init_weights(module, base_std=0.02):
      if isinstance(module, nn.Linear):
        applied_std = base_std * (module.__INIT_SCALAR__ if hasattr(module, "__INIT_SCALAR__") else 1)
        torch.nn.init.normal_(module.weight, mean=0.0, std=applied_std)
        if module.bias is not None:
          torch.nn.init.zeros_(module.bias)
      elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=base_std)
      elif isinstance(module, nn.RMSNorm):
        torch.nn.init.ones_(module.weight)

    self.apply(_init_weights)

    # tie weights after init
    self.token_embed.weight = self.head.weight

  def forward(self, idx: torch.Tensor, targets=None):
    _, T = idx.size() # number of batches, token sequence
    block_size = self.config.block_size

    idx: torch.Tensor = idx if T <= block_size else idx[:, -block_size:]

    pos_e = self.position_embed(self.pos[:, :T])
    tok_e = self.token_embed(idx)
    x = torch.add(pos_e, tok_e)
    if self.config.checkpoint:
      for block in self.blocks:
        x = torch.utils.checkpoint.checkpoint(block, x)
    else:
      x = self.blocks(x)
    x = self.ln(x)
    if targets is None:
      logits = self.head(x[:, [-1], :])
      return logits, None

    logits = self.head(x)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    return logits, loss

  def get_optimizer(self, weight_decay, lr, b1=0.9, b2=0.95, eps=1e-8):
    params: Dict[str, nn.Parameter] = dict(filter(lambda t: t[1].requires_grad, self.named_parameters()))
    decay_params = [p for _, p in params.items() if p.dim() >= 2]
    nodecay_params = [p for _, p in params.items() if p.dim() < 2]
    optimization_groups = [
      {'params': decay_params, 'weight_decay': weight_decay},
      {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    return torch.optim.AdamW(optimization_groups, lr=lr, betas=(b1, b2), eps=eps)

  def get_num_parameters(self, as_str=False):
    n = sum(p.numel() for p in self.parameters())
    if not as_str:
      return n

    sfxs = [(10 ** 9, 'b'), (10 ** 6, 'm'), (10 ** 3, 'k')]
    for threshold, sfx in sfxs:
      if n > threshold:
        frac = n / threshold
        return f"{frac:.3f}{sfx}"
    return f"{n}"

  @torch.no_grad()
  def generate(self, seed: torch.Tensor, max_new_tokens, temperature=1.0, topk=-1):
    device = seed.device  # get device from input tensor
    block_size = self.config.block_size
    idx = seed
    self.eval()
    for _ in range(max_new_tokens):
      idx_cond = idx[:, -block_size:]
      with torch.no_grad():
        device_type = device.type
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
          logits, _ = self(idx_cond)
        logits = logits[:, -1, :]
        probs = F.softmax(logits / temperature, dim = -1)
        if topk <= 0:
          xcol = torch.multinomial(probs, num_samples = 1)
        else:
          top_p, top_i = torch.topk(probs, topk, dim=-1)
          ix = torch.multinomial(top_p, num_samples=1)
          xcol = torch.gather(top_i, -1, ix)
        idx = torch.cat((idx, xcol), dim=1)
        yield xcol
    self.train()