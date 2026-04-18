from dataclasses import dataclass
from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

# Version 4 - document masking, GQA

LigerFusedLinearCrossEntropyLoss = None
flash_attn_varlen_func = None
if torch.cuda.is_available():
  try:
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
  except ImportError:
    pass
  try:
    from flash_attn import flash_attn_varlen_func
  except ImportError:
    pass

@dataclass(frozen=True)
class GPTv4Config:
  vocab_size: int
  block_size: int
  device: str = "cpu"
  n_heads: int = 8 # number of embedding heads (n_embed / n_heads MUST be an integer)
  n_embed: int = 288 # number of dimensions in embedding vector
  n_layers: int = 6 # number of blocks
  mlp_mul: int = 4
  activation_fn: str = "relu2"
  n_kv_heads: int = 2
  rope_base: float = 10000.0
  logit_cap: float = 30.0
  checkpoint: bool = False


class RotaryPositionalEmbedding(nn.Module):
  def __init__(self, config: GPTv4Config):
    super().__init__()
    head_dim = config.n_embed // config.n_heads
    max_seq_len = config.block_size
    base = config.rope_base

    assert head_dim % 2 == 0
    # frequencies
    thetas = torch.pow(base, -torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    # positions
    t = torch.arange(max_seq_len, dtype=torch.float32)
    angles = torch.outer(t, thetas)

    self.register_buffer("cos", angles.cos())
    self.register_buffer("sin", angles.sin())

  def forward(self, q: torch.Tensor, k: torch.Tensor):
    # q, k ~ (B, n_heads, T, head_dim)
    _, _, T, _ = q.shape

    cos = self.cos[:T].view(1, 1, T, -1)
    sin = self.sin[:T].view(1, 1, T, -1)

    q_rotated = self._rotate(q, cos, sin)
    k_rotated = self._rotate(k, cos, sin)

    return q_rotated, k_rotated
  
  @staticmethod
  def _rotate(x: torch.Tensor, cos, sin):
    x1, x2 = x.view(*x.shape[:-1], -1, 2).unbind(dim=-1)
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)

    real = x1 * cos - x2 * sin
    im = x1 * sin + x2 * cos

    return torch.stack((real, im), dim=-1).flatten(-2)


class CausalSelfAttention(nn.Module):
  def __init__(self, config: GPTv4Config):
    super().__init__()
    self.n_heads, self.n_kv_heads, self.n_embed = config.n_heads, config.n_kv_heads, config.n_embed
    self.head_dim = config.n_embed // config.n_heads

    assert self.n_embed % self.n_heads == 0
    assert self.n_heads % self.n_kv_heads == 0

    self.q_proj = nn.Linear(config.n_embed, config.n_embed, bias=False)
    self.kv_proj = nn.Linear(config.n_embed, 2 * (config.n_kv_heads * self.head_dim), bias=False)
    self.rope = RotaryPositionalEmbedding(config=config)
    self.proj = nn.Linear(config.n_embed, config.n_embed, bias=False)
    # self.q_proj.__INIT_SCALAR__ = pow((2 * config.n_layers), -0.5)
    # self.kv_proj.__INIT_SCALAR__ = pow((2 * config.n_layers), -0.5)
    self.proj.__INIT_SCALAR__ = pow((2 * config.n_layers), -0.5)

  def forward(self, x: torch.Tensor, attn_info=None):
    # attn_info is one of:
    #   None                       -> pure causal (no doc mask)
    #   (cu_seqlens, max_seqlen)   -> CUDA + flash-attn varlen path (doc-masked)
    #   bool tensor (B,1,T,T)      -> SDPA fallback path (doc-masked, non-CUDA or no flash-attn)
    B, T, _ = x.size() # assuming C == self.n_embed

    q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
    kv = self.kv_proj(x).view(B, T, self.n_kv_heads, 2 * self.head_dim)
    k, v = kv.split(self.head_dim, dim=-1)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    q, k = self.rope(q, k)

    if isinstance(attn_info, tuple):
      cu_seqlens, max_seqlen = attn_info
      # (B,H,T,D) -> (B,T,H,D) -> (B*T,H,D)
      qf = q.transpose(1, 2).reshape(B * T, self.n_heads, self.head_dim)
      kf = k.transpose(1, 2).reshape(B * T, self.n_kv_heads, self.head_dim)
      vf = v.transpose(1, 2).reshape(B * T, self.n_kv_heads, self.head_dim)
      y = flash_attn_varlen_func(qf, kf, vf, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, causal=True)
      y = y.view(B, T, self.n_heads, self.head_dim)
    else:
      # attn_info is either None (pure causal) or a bool (B,1,T,T) tensor
      y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_info, is_causal=attn_info is None, enable_gqa=True)
      y = y.transpose(1, 2).contiguous()
    return self.proj(y.reshape(B, T, self.n_embed))


class MultiLayerPerceptron(nn.Module):
  def __init__(self, config: GPTv4Config):
    super().__init__()
    self.activation_fn = config.activation_fn
    n_hidden = config.mlp_mul * config.n_embed

    self.l1 = nn.Linear(config.n_embed, n_hidden, bias=False)
    if self.activation_fn == "swiglu":
      self.l1_gate = nn.Linear(config.n_embed, n_hidden, bias=False)

    self.proj = nn.Linear(n_hidden, config.n_embed, bias=False)
    self.proj.__INIT_SCALAR__ = (2 * config.n_layers) ** -0.5

  def forward(self, x: torch.Tensor):
    if self.activation_fn == "relu2":
      x = F.relu(self.l1(x)).square()
    elif self.activation_fn == "swiglu":
      x = self.l1(x) * F.silu(self.l1_gate(x))
    elif self.activation_fn == "gelu":
      x = F.gelu(self.l1(x))
    else: raise ValueError(f"unknown activation function {self.activation_fn}")
    return self.proj(x)


class TransformerBlock(nn.Module):
  def __init__(self, config: GPTv4Config):
    super().__init__()
    self.norm1 = nn.RMSNorm(config.n_embed)
    self.attn = CausalSelfAttention(config)
    self.norm2 = nn.RMSNorm(config.n_embed)
    self.mlp = MultiLayerPerceptron(config)

  def forward(self, x: torch.Tensor, attn_info=None):
    # residual connections
    attn_add = self.attn(self.norm1(x), attn_info)
    x = torch.add(x, attn_add)
    mlp_add = self.mlp(self.norm2(x))
    x = torch.add(x, mlp_add)
    return x


class LanguageModel(nn.Module):
  model_type: str = 'gpt-v4'

  def __init__(self, config: GPTv4Config):
    super().__init__()
    self.config = config

    vocab_size = config.vocab_size

    self.token_embed = nn.Embedding(vocab_size, config.n_embed)
    self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
    self.ln = nn.RMSNorm(config.n_embed)
    self.head = nn.Linear(config.n_embed, vocab_size, bias=False)

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

    self._liger_ce = None
    if config.logit_cap <= 0 and LigerFusedLinearCrossEntropyLoss is not None and config.device == "cuda":
      self._liger_ce = LigerFusedLinearCrossEntropyLoss(reduction='none')

  @staticmethod
  def build_attn_info(doc_ids: torch.Tensor, block_size: int):
    """Build attention metadata from doc_ids.
    CUDA: returns (cu_seqlens, max_seqlen) for flash_attn_varlen_func.
    Non-CUDA: returns a (B, 1, T, T) bool mask for SDPA.
    """
    B, T = doc_ids.shape
    if doc_ids.device.type != "cuda" or flash_attn_varlen_func is None:
      causal = torch.ones(T, T, dtype=torch.bool, device=doc_ids.device).tril()
      doc = doc_ids.unsqueeze(2) == doc_ids.unsqueeze(1)
      return (causal.unsqueeze(0) & doc).unsqueeze(1)

    flat = doc_ids.reshape(-1)
    pos = torch.arange(B * T, device=doc_ids.device)
    sample_start = (pos % T == 0)
    within = torch.zeros_like(flat, dtype=torch.bool)
    within[1:] = flat[1:] != flat[:-1]
    within = within & ~sample_start
    is_start = sample_start | within
    # Fixed-length output for CUDA graph compatibility: pack valid start
    # positions to the front via sort; filler slots hold B*T so trailing
    # segments are zero-length (FA varlen skips them). Total length: B*T+1.
    filler = torch.full_like(pos, B * T)
    packed = torch.where(is_start, pos, filler)
    sorted_pos, _ = torch.sort(packed)
    end = torch.tensor([B * T], device=doc_ids.device, dtype=sorted_pos.dtype)
    cu_seqlens = torch.cat([sorted_pos, end]).to(torch.int32)
    return cu_seqlens, block_size

  def forward(self,
    idx: torch.Tensor, targets=None, attn_info=None,
    loss_mask: Optional[torch.Tensor] = None,
    eval_logits: bool = False
  ):
    _, T = idx.size() # number of batches, token sequence
    block_size = self.config.block_size

    idx: torch.Tensor = idx if T <= block_size else idx[:, -block_size:]

    tok_e = self.token_embed(idx) * pow(self.config.n_embed, 0.5)
    x = tok_e
    if self.config.checkpoint:
      for block in self.blocks:
        x = torch.utils.checkpoint.checkpoint(block, x, attn_info, use_reentrant=False)
    else:
      for block in self.blocks:
        x = block(x, attn_info)
    x = self.ln(x)
    
    if targets is None and not eval_logits: x = x[:, [-1], :]

    use_liger = (
      self._liger_ce is not None and targets is not None
      and not eval_logits and x.is_cuda
    )
    if use_liger:
      loss = self._liger_ce(self.head.weight, x.reshape(-1, x.size(-1)), targets.view(-1))
      logits = None
    else:
      logits = self.head(x)
      if self.config.logit_cap > 0:
        logits = self.config.logit_cap * torch.tanh(logits / self.config.logit_cap)
      if targets is None: return logits, None
      loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction='none')

    if loss_mask is not None:
      loss = (loss * loss_mask.view(-1).float()).sum() / loss_mask.sum().clamp(min=1)
    else:
      loss = loss.mean()
    return logits, loss

  def get_optimizer(self, weight_decay, lr, b1=0.9, b2=0.95, eps=1e-8):
    params: Dict[str, nn.Parameter] = dict(filter(lambda t: t[1].requires_grad, self.named_parameters()))
    decay_params = [p for _, p in params.items() if p.dim() >= 2]
    nodecay_params = [p for _, p in params.items() if p.dim() < 2]
    optimization_groups = [
      {'params': decay_params, 'weight_decay': weight_decay},
      {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    fused = next(iter(params.values())).is_cuda
    return torch.optim.AdamW(optimization_groups, lr=lr, betas=(b1, b2), eps=eps, fused=fused)

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
  def generate(self, 
    seed: torch.Tensor, max_new_tokens, temperature=1.0, topk=-1,
    use_bfloat: bool = True
  ):
    device = seed.device  # get device from input tensor
    block_size = self.config.block_size
    idx = seed
    self.eval()
    for _ in range(max_new_tokens):
      idx_cond = idx[:, -block_size:]
      device_type = device.type
      
      if use_bfloat:
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
          logits, _ = self(idx_cond)
      else:
        logits, _ = self(idx_cond)
      logits = logits[:, -1, :].float()

      probs = F.softmax(logits / temperature, dim = -1)
      if torch.isnan(probs).any() or torch.isinf(probs).any():
        print(f"logits range: [{logits.min():.2f}, {logits.max():.2f}]")
        print(f"probs has nan: {torch.isnan(probs).any()}, inf: {torch.isinf(probs).any()}")
        break
      if topk <= 0:
        xcol = torch.multinomial(probs, num_samples = 1)
      else:
        top_p, top_i = torch.topk(probs, topk, dim=-1)
        ix = torch.multinomial(top_p, num_samples=1)
        xcol = torch.gather(top_i, -1, ix)
      idx = torch.cat((idx, xcol), dim=1)
      yield xcol
    self.train()