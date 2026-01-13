from dataclasses import dataclass
from enum import Enum

class ArtifactType(Enum):
  RAW_DATASET = "raw_ds"
  VOCABULARY = "vocab"
  TOKENIZED_DATA = "data"
  TRAINING_RUN = "run"

  @property
  def prefix(self) -> str:
    return self.value


@dataclass
class ArtifactID:
  type: ArtifactType
  hash_prefix: str  # git like hex hash

  def __str__(self) -> str:
    return f"{self.type.prefix}_{self.hash_prefix}"

  @classmethod
  def from_string(cls, id_string: str) -> 'ArtifactID':
    parts = id_string.split('_', 1)
    if len(parts) != 2: raise ValueError(
      f"invalid artifact ID format: '{id_string}'"
      f"expected format: {{type}}_{{hash}}"
    )

    type_prefix, hash_prefix = parts
    artifact_type = next(
      filter(
        lambda atype: atype.prefix == type_prefix,
        ArtifactType
      ),
      None
    )

    if artifact_type is None: raise ValueError(
      f"unknown artifact type prefix '{type_prefix}'"
    )

    # Validate hash prefix format (should be hex)
    if not all(c in '0123456789abcdef' for c in hash_prefix.lower()):
      raise ValueError(
        f"invalid hexadecimal hash prefix: '{hash_prefix}'"
      )

    return cls(type=artifact_type, hash_prefix=hash_prefix.lower())


def validate_artifact(artifact_path: str) -> dict:
  """validate for metadata structure"""
  from toy_transformers.utilities import io

  obj = io.load(artifact_path)

  if "artifact_metadata" not in obj:
    raise ValueError(f"missing artifact_metadata in {artifact_path}")

  metadata = obj["artifact_metadata"]
  required = ["artifact_id", "artifact_type", "content_hash", "created_at", "git_hash"]
  missing = [f for f in required if f not in metadata]
  if missing:
    raise ValueError(
      f"artifact '{metadata.get('artifact_id', artifact_path)}' "
      f"missing required metadata fields: {missing}"
    )

  try:
    ArtifactID.from_string(metadata["artifact_id"])
  except ValueError as e:
    raise ValueError(f"Invalid artifact ID: {e}")

  return obj
