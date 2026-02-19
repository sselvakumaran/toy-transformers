from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

# Version 1 of a GPT-like self-attention multi-headed network w/ dropout and layernorm
# Based on Andrej Karpathy's nanoGPT model (2023)

@dataclass(frozen=True)
class GPTv1Config:
  vocab_size: int
  block_size: int # max context length
  device: str = "cpu"
  n_heads: int = 8 # number of embedding heads (n_embed / n_heads MUST be an integer)
  n_embed: int = 288 # number of dimensions in embedding vector
  n_layers: int = 6 # number of blocks
  dropout: float = 0.2 # number of nodes to randomly drop to reduce overfit


class Head(nn.Module):
  def __init__(self, head_size: int, config: GPTv1Config, block_size: int):
    super().__init__()
    n_embed, dropout = config.n_embed, config.dropout

    self.key = nn.Linear(n_embed, head_size, bias=False)
    self.query = nn.Linear(n_embed, head_size, bias=False)
    self.value = nn.Linear(n_embed, head_size, bias=False)
    self.dropout = nn.Dropout(dropout)
    # Create buffer on CPU, will be moved to correct device with model
    self.register_buffer('tril',
      torch.tril(torch.ones(block_size, block_size, device=config.device))
    )
    self.attention_scalar = pow(head_size, -0.5)

  def forward(self, x):
    _, T, _ = x.shape
    k = self.key(x)
    v = self.value(x)
    q = self.query(x)

    # doing attn. formula -> softmax(qK^T / sqrt(d_k)) * V
    weights = q @ k.transpose(-2, -1) * self.attention_scalar
    weights = F.softmax(
      weights.masked_fill(self.tril[:T, :T] == 0, float('-inf')),
      dim=-1
    )
    weights = self.dropout(weights)
    out = weights @ v
    return out


class MultiHeadAttention(nn.Module):
  def __init__(self, head_size: int, config: GPTv1Config, block_size: int):
    super().__init__()
    n_embed, dropout, num_heads = config.n_embed, config.dropout, config.n_heads

    self.heads = nn.ModuleList([Head(head_size, config, block_size) for _ in range(num_heads)])
    self.proj = nn.Linear(n_embed, n_embed)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    out = torch.cat([h(x) for h in self.heads], dim = -1)
    out = self.dropout(self.proj(out))
    return out


class FeedForward(nn.Module):
  def __init__(self, config: GPTv1Config):
    super().__init__()
    n_embed, dropout = config.n_embed, config.dropout

    self.net = nn.Sequential(
      nn.Linear(n_embed, 4 * n_embed),
      nn.ReLU(),
      nn.Linear(4 * n_embed, n_embed),
      nn.Dropout(dropout)
    )

  def forward(self, x):
    return self.net(x)


class Block(nn.Module):
  def __init__(self, model_config: GPTv1Config, block_size: int):
    super().__init__()
    n_embed, n_head = model_config.n_embed, model_config.n_heads
    head_size = n_embed // n_head

    self.sa = MultiHeadAttention(head_size, model_config, block_size)
    self.ffwd = FeedForward(model_config)
    self.ln1 = nn.LayerNorm(n_embed)
    self.ln2 = nn.LayerNorm(n_embed)

  def forward(self, x):
    x = torch.add(x, self.sa(self.ln1(x)))
    x = torch.add(x, self.ffwd(self.ln2(x)))
    return x


class LanguageModel(nn.Module):
  model_type: str = "gpt-v1"

  def __init__(self, config: GPTv1Config):
    super().__init__()
    self.config = config

    vocab_size = config.vocab_size
    block_size = config.block_size
    n_embed, n_layers = config.n_embed, config.n_layers

    self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
    self.position_embedding_table = nn.Embedding(block_size, n_embed)
    self.blocks = nn.Sequential(
      *[Block(config, block_size) for _ in range(n_layers)],
      nn.LayerNorm(n_embed),
    )
    self.lm_head = nn.Linear(n_embed, vocab_size)

    self.apply(self._init_weights)

  @staticmethod
  def _init_weights(module):
    if isinstance(module, nn.Linear):
      nn.init.normal_(module.weight, mean=0.0, std=0.02)
      if module.bias is not None:
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
      nn.init.normal_(module.weight, mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
      nn.init.ones_(module.weight)
      nn.init.zeros_(module.bias)

  def forward(self, idx, targets=None):
    B, T = idx.shape
    device = self.config.device

    tok_embed = self.token_embedding_table(idx)
    pos_embed = self.position_embedding_table(torch.arange(T, device=device))
    x = self.blocks(torch.add(tok_embed, pos_embed))
    logits = self.lm_head(x)

    if targets is None:
      return logits, None

    B, T, C = logits.shape
    logits_shrink = logits.view(B*T, C)
    targets_shrink = targets.view(B*T)
    loss = F.cross_entropy(logits_shrink, targets_shrink)
    return logits_shrink, loss

  @torch.no_grad()
  def generate(self, idx, max_new_tokens):
    block_size = self.config.block_size
    for _ in range(max_new_tokens):
      idx_cond = idx[:, -block_size:]

      _, T = idx_cond.shape
      device = self.config.device  # get device from input tensor
      tok_embed = self.token_embedding_table(idx_cond)
      pos_embed = self.position_embedding_table(torch.arange(T, device=device))
      x = self.blocks(tok_embed + pos_embed)
      logits = self.lm_head(x)

      logits = logits[:, -1, :]
      probs = F.softmax(logits, dim = -1)
      idx_next = torch.multinomial(probs, num_samples = 1)
      idx = torch.cat((idx, idx_next), dim=1)
      yield idx_next