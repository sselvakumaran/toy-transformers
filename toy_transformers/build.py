from dataclasses import dataclass, field, asdict
import json
from pathlib import Path
import torch

@dataclass
class TrainingConfig:
	
	@dataclass
	class RunConfig:
		name: str
		log_interval: int = 10
		save_interval: int = 5000
		seed: int = 42
	run: RunConfig


	@dataclass
	class DatasetConfig:
		dataset_folders: list[str]
		dataset_weights: list[float]

		def __post_init__(self):
			if not self.dataset_weights: pass
			total = sum(self.dataset_weights)
			self.dataset_weights = [w / total for w in self.dataset_weights]
	dataset: DatasetConfig


	@dataclass
	class ModelConfig:
		config: dict[str, any]
		model: str = 'gptv4'

		def build_config(self, vocab_size: int, device: str = "cuda"):
			if self.model == 'gptv4':
				from toy_transformers.models.gptv4 import LanguageModel, GPTv4Config
				cfg = GPTv4Config(
					vocab_size=vocab_size,
					device=device,
					**self.config
				)
				return LanguageModel(cfg).to(device)
	model: ModelConfig


	@dataclass
	class OptimizerConfig:
		min_lr: float
		max_lr: float
		warmup_steps: int
		schedule: str = "cosine"

		def build_scheduler(self, optimizer, total_steps: int):
			warmup = torch.optim.lr_scheduler.LinearLR(
				optimizer, start_factor= 1/self.warmup_steps, total_iters=self.warmup_steps
			)

			if self.schedule == "cosine":
				decay = torch.optim.lr_scheduler.CosineAnnealingLR(
					optimizer, T_max=total_steps - self.warmup_steps, eta_min=self.min_lr
				)
			else: raise ValueError()

			return torch.optim.lr_scheduler.SequentialLR(
				optimizer, schedulers=[warmup, decay], milestones=[self.warmup_steps]
			)
	optimizer: OptimizerConfig


	@dataclass
	class TokenizerConfig:
		path: str
	tokenizer: TokenizerConfig


	@dataclass
	class TokensConfig:
		batch_size: int
		grad_accum_steps: int
		total_steps: int
	tokens: TokensConfig


	@dataclass
	class EvalConfig:
		interval: int
		batches: int
	eval: EvalConfig


	@classmethod
	def from_json(cls, path: str | Path):
		raw = json.loads(Path(path).read_text())
		return cls(
			run=cls.RunConfig(**raw.get("run", {})),
			dataset=cls.DatasetConfig(**raw.get("dataset", {})),
			model=cls.ModelConfig(**raw.get("model", {})),
			optimizer=cls.OptimizerConfig(**raw.get("optimizer", {})),
			tokenizer=cls.TokenizerConfig(**raw.get("tokenizer", {})),
			tokens=cls.TokensConfig(**raw.get("tokens", {})),
			eval=cls.EvalConfig(**raw.get("train", {})),
		)