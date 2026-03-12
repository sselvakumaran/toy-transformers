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

	@dataclass
	class DatasetConfig:
		dataset_folders: list[str]
		dataset_weights: list[float]

		def __post_init__(self):
			if not self.dataset_weights: return
			total = sum(self.dataset_weights)
			self.dataset_weights = [w / total for w in self.dataset_weights]

	@dataclass
	class ModelConfig:
		config: dict
		model: str = 'gptv4'

		def build_model(self, vocab_size: int, device: str = "cuda"):
			if self.model == 'gptv4':
				from toy_transformers.models.gptv4 import LanguageModel, GPTv4Config
				cfg = GPTv4Config(vocab_size=vocab_size, device=device, **self.config)
				return LanguageModel(cfg).to(device)
			else: raise ValueError(f"unknown model: {self.model}")


	@dataclass
	class OptimizerConfig:
		max_lr: float
		min_lr: float
		warmup_steps: int
		weight_decay: float = 0.1
		schedule: str = "cosine"
		adam_b1: float = 0.9
		adam_b2: float = 0.95
		adam_eps: float = 1e-8
		
		def build_optimizer(self, model):
			return model.get_optimizer(
				weight_decay=self.weight_decay,
				lr=self.max_lr,
				b1=self.adam_b1, b2=self.adam_b2, eps=self.adam_eps
			)

		def build_scheduler(self, optimizer, total_steps: int):
			warmup = torch.optim.lr_scheduler.LinearLR(
				optimizer, start_factor= 1/self.warmup_steps, total_iters=self.warmup_steps
			)

			if self.schedule == "cosine":
				decay = torch.optim.lr_scheduler.CosineAnnealingLR(
					optimizer, T_max=total_steps - self.warmup_steps, eta_min=self.min_lr
				)
			elif self.schedule == "linear":
				decay = torch.optim.lr_scheduler.LinearLR(
					optimizer, 
					start_factor=1.0, end_factor=self.min_lr / self.max_lr,
					total_iters=total_steps - self.warmup_steps
				)
			else: raise ValueError(f"unknown schedule {self.schedule}")

			return torch.optim.lr_scheduler.SequentialLR(
				optimizer, schedulers=[warmup, decay], milestones=[self.warmup_steps]
			)


	@dataclass
	class TokenizerConfig:
		path: str
		vocab_size: int = field(init=False)
		bos_id: int = field(init=False)
		pad_id: int = field(init=False)

		def load(self, local_root: Path):
			vocab_path = local_root / self.path
			raw = json.loads(vocab_path.read_text())
			self.vocab_size = raw["vocab_size"]
			self.bos_id = raw.get("bos_id", -1)
			self.pad_id = raw.get("pad_id", -1)


	@dataclass
	class TokensConfig:
		batch_size: int
		grad_accum_steps: int
		train_tokens: int

	@dataclass
	class EvalConfig:
		interval: int = 500
		batches: int = 20

	run: RunConfig
	dataset: DatasetConfig
	model: ModelConfig
	optimizer: OptimizerConfig
	tokenizer: TokenizerConfig
	tokens: TokensConfig
	eval: EvalConfig

	@property
	def tokens_per_step(self):
		return self.tokens.batch_size * self.tokens.grad_accum_steps \
			* self.model.config["block_size"]

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
			eval=cls.EvalConfig(**raw.get("eval", {})),
		)

	def to_json(self, path: str | Path):
		path = Path(path)
		path.parent.mkdir(parents=True, exist_ok=True)
		path.write_text(json.dumps(asdict(self), indent=2))