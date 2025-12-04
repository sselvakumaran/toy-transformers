import unittest
from toy_transformers.utilities import io

class TestIOMethods(unittest.TestCase):

	def test_config_object(self):
		from dataclasses import dataclass, asdict

		@dataclass(frozen=True)
		class Config:
			vocab_size: int
			batch_size: int = 64
			block_size: int = 128
			n_heads: int = 8
			n_embed: int = 288
			n_layers: int = 6
			dropout: float = 0.2
			device: str = "mps"
			checkpoint: bool = False

			def to_savable(self):
				return asdict(self)
			
			@staticmethod
			def from_serializable(d: dict):
				return Config(**d)
		
		obj1 = Config(vocab_size=513)

		io.save(obj1.to_savable(), path="temp/iotests/test1.json")
		obj2 = Config.from_serializable(io.load(path="temp/iotests/test1.json"))

		self.assertEqual(obj1, obj2)

if __name__ == "__main__":
	unittest.main()