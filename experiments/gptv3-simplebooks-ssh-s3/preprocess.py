import sys
from pathlib import Path
import torch

EXPERIMENT_DIR = Path(__file__).parent
REPO_DIR = Path.cwd()
assert (REPO_DIR / ".git").exists()

if str(REPO_DIR) not in sys.path:
  sys.path.insert(0, str(REPO_DIR))

from toy_transformers import tokenization

# CONFIG
VOCAB_SIZE = 4096
MODE = tokenization.TokenizationMode.STR

DATA_DIR = REPO_DIR / "data/raw/simplebooks/simplebooks-92-raw"
TRAIN_PATH = DATA_DIR / "train.txt"
VALID_PATH = DATA_DIR / "valid.txt"
TEST_PATH = DATA_DIR / "test.txt"

OUT_DIR = EXPERIMENT_DIR / "data"
VOCAB_PATH = OUT_DIR / f"vocab_{VOCAB_SIZE}.json"

if not VOCAB_PATH.exists():
  raw_data = open(TRAIN_PATH, "r")
  print("building vocab...")
  vocab = tokenization.create_bpe(
    raw_data, 
    VOCAB_SIZE, MODE
  )
  vocab.save(VOCAB_PATH)
  print(f"saved to {VOCAB_PATH}")
else:
  print(f"loading vocab from {VOCAB_PATH}")
  vocab = tokenization.Vocabulary.load(VOCAB_PATH)

def cache_and_tokenize(text_path: Path, cache_path: Path, vocab: tokenization.Vocabulary) -> torch.Tensor:
	if cache_path.exists():
		print("loading from cache")
		return torch.load(cache_path, weights_only=True)
	print("tokenizing...")
	text = open(text_path, "r").read()
	tokens = torch.tensor(vocab.encode(text), dtype=torch.long)
	torch.save(tokens, cache_path)
	return tokens

for split, path in [
  ("train", TRAIN_PATH), ("valid", VALID_PATH), ("test", TEST_PATH)
]:
  cache_path = OUT_DIR / f"{split}.pt"
  if cache_path.exists():
    print(f"{split}.pt: loaded from cache")
    continue
  print(f"{split}.pt: tokenizing...")
  text = open(path, 'r').read()
  tokens = torch.tensor(vocab.encode(text), dtype=torch.long)
  torch.save(tokens, cache_path)
  print(f"{split}.pt: completed!")