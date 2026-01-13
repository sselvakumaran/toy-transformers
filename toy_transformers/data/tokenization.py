from io import StringIO, BytesIO
from toy_transformers.data import bpe as tokenizer
from typing import Optional
import torch

from toy_transformers.utilities.io import TorchTensorRef
from toy_transformers.artifacts import ArtifactType, ArtifactMetadata, stable_hash
from toy_transformers.data import RawDataset

class TokenizedData:
  data: torch.Tensor
  vocab_id: str
  raw_dataset_id: str

  def __init__(self,
    data: torch.Tensor,
    vocab_id: str,
    raw_dataset_id: str,
  ):
    self.data = data
    self.vocab_id = vocab_id
    self.raw_dataset_id = raw_dataset_id

  def get_artifact_type(self) -> ArtifactType:
    return ArtifactType.TOKENIZED_DATA

  def stable_hash(self) -> str:
    return stable_hash((self.data, self.vocab_id, self.raw_dataset_id))

  def to_state_dict(self) -> dict:
    content_hash = self.stable_hash()
    metadata = ArtifactMetadata(self.get_artifact_type(), content_hash)

    return {
      "artifact_metadata": metadata.to_dict(),
      "vocab_id": self.vocab_id,
      "raw_dataset_id": self.raw_dataset_id,
      "data": TorchTensorRef("data", self.data)
    }

  @classmethod
  def from_state_dict(cls, obj: dict) -> 'TokenizedData':
    return cls(
      data=obj["data"].tensor,
      vocab_id=obj["vocab_id"],
      raw_dataset_id=obj["raw_dataset_id"],
    )


def train_vocabulary(
  raw_dataset: RawDataset,
  vocab_size: int,
  mode: Optional[tokenizer.TokenizationMode] = tokenizer.TokenizationMode.STR,
  **kwargs
) -> tokenizer.Vocabulary:

  if mode == tokenizer.TokenizationMode.BYTES:
    data_handle = BytesIO(raw_dataset.data 
      if isinstance(raw_dataset.data, bytes) 
      else raw_dataset.data.encode('utf-8')
    )
  else:
    data_handle = StringIO(raw_dataset.data 
      if isinstance(raw_dataset.data, str)
      else raw_dataset.data.decode('utf-8')
    )

  return tokenizer.create_bpe(
    data_handle,
    vocab_size,
    mode=mode,
    raw_dataset_id=raw_dataset.stable_hash(),
    **kwargs
  )

def tokenize_dataset(
  vocab: tokenizer.Vocabulary,
  raw_dataset: RawDataset,
) -> TokenizedData:

  token_ids = vocab.encode(raw_dataset.data)
  data_tensor = torch.tensor(token_ids, dtype=torch.long)

  return TokenizedData(
    data=data_tensor,
    vocab_id=vocab.artifact_id,
    raw_dataset_id=raw_dataset.stable_hash()
  )
