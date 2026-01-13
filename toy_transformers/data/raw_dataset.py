from dataclasses import dataclass
from typing import Optional
from toy_transformers.utilities import io
from toy_transformers.utilities.io import CompressedBytesRef
from toy_transformers.artifacts import ArtifactType, ArtifactMetadata, stable_hash
from toy_transformers.data.bpe import TokenizationMode


@dataclass
class RawDataset:
  data: str | bytes
  mode: TokenizationMode

  def get_artifact_type(self) -> ArtifactType:
    return ArtifactType.RAW_DATASET

  def stable_hash(self) -> str:
    if self.mode == TokenizationMode.BYTES:
      return stable_hash(self.data)
    return stable_hash(self.data.encode('utf-8'))

  def to_state_dict(self) -> dict:
    content_hash = self.compute_content_hash()
    metadata = ArtifactMetadata(self.get_artifact_type(), content_hash)

    return {
      "artifact_metadata": metadata.to_dict(),
      "data_hash": content_hash,
      "data": CompressedBytesRef("raw_data", data=self.data, mode=self.mode)
    }

  @classmethod
  def from_state_dict(cls, obj: dict) -> 'RawDataset':
    compressed_ref: CompressedBytesRef = obj["data"]
    data = compressed_ref.data
    mode = compressed_ref.mode

    if mode == TokenizationMode.BYTES:
      computed_hash = stable_hash(data)
    else:
      computed_hash = stable_hash(data.encode('utf-8'))
    stored_hash = obj["data_hash"]
    if computed_hash != stored_hash:
      raise ValueError(f"hashes do not match")

    return cls(data=data, mode=mode)


def register_raw_dataset(data: str | bytes, output_path: str, mode: Optional[TokenizationMode] = None) -> RawDataset:
  if mode is None:
    mode = TokenizationMode.BYTES if isinstance(data, bytes) else TokenizationMode.STR
  dataset = RawDataset(data=data, mode=mode)
  state_dict = dataset.to_state_dict()

  io.save(state_dict, output_path)
  return dataset