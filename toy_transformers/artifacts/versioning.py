import hashlib
import subprocess
import torch
from dataclasses import is_dataclass, fields
from datetime import datetime, timezone
from typing import Any, Hashable
from toy_transformers.artifacts.types import ArtifactType
from itertools import zip_longest

def stable_hash(obj: Any) -> str:
  """
  create SHA256 hash of object consistent between python runtimes
  works for most objects including tensors, which uses sampling-based method
  """
  # prioritize object's own stable_hash() method
  if hasattr(obj, 'stable_hash') and callable(obj.stable_hash):
    result = obj.stable_hash()
    if isinstance(result, str):
      return result

  hasher = hashlib.sha256()
  _update_hasher(hasher, obj)
  return hasher.hexdigest()

def _update_hasher(hasher: hashlib.sha256, obj: Any) -> None:
  """update hasher with object content"""
  # check for stable_hash() method first
  if hasattr(obj, 'stable_hash') and callable(obj.stable_hash):
    result = obj.stable_hash()
    if isinstance(result, str):
      hasher.update(result.encode('utf-8'))
      return

  # handle *frozen* dataclasses specially
  if is_dataclass(obj) and obj.__dataclass_fields__:
    hasher.update(b'dataclass:')
    hasher.update(type(obj).__name__.encode('utf-8'))
    hasher.update(b'{')
    for field_obj in fields(obj):
      hasher.update(field_obj.name.encode('utf-8'))
      hasher.update(b':')
      _update_hasher(hasher, getattr(obj, field_obj.name))
      hasher.update(b',')
    hasher.update(b'}')
    return

  if isinstance(obj, str):
    hasher.update(obj.encode('utf-8'))
  elif isinstance(obj, bytes):
    hasher.update(obj)
  elif isinstance(obj, (int, float)):
    hasher.update(str(obj).encode('utf-8'))
  elif isinstance(obj, (tuple, list)):
    hasher.update(b'[')
    for item in obj:
      _update_hasher(hasher, item)
      hasher.update(b',')
    hasher.update(b']')
  elif isinstance(obj, dict):
    hasher.update(b'{')
    for key in sorted(obj.keys(), key=str):
      _update_hasher(hasher, key)
      hasher.update(b':')
      _update_hasher(hasher, obj[key])
      hasher.update(b',')
    hasher.update(b'}')
  elif isinstance(obj, torch.Tensor):
    # Hash shape, dtype, and sample of values
    hasher.update(str(tuple(obj.shape)).encode('utf-8'))
    hasher.update(str(obj.dtype).encode('utf-8'))
    # Sample values for large tensors
    n_samples = min(1000, obj.numel())
    if obj.numel() > n_samples:
      indices = torch.linspace(0, obj.numel() - 1, n_samples, dtype=torch.long)
      samples = obj.flatten()[indices]
    else:
      samples = obj.flatten()
    hasher.update(samples.cpu().numpy().tobytes())
  else:
    hasher.update(str(obj).encode('utf-8'))


def get_git_version() -> str:
  """get current git commit hash"""
  try:
    return subprocess.check_output(
      ["git", "rev-parse", "--short", "HEAD"],
      stderr=subprocess.DEVNULL,
      text=True
    ).strip()
  except (subprocess.CalledProcessError, FileNotFoundError):
    return "unknown"


class ArtifactMetadata:
  """standard metadata for all artifacts"""
  artifact_id: str
  artifact_type: str
  created_at: str
  git_hash: str
  content_hash: str

  def __init__(self, artifact_type: ArtifactType, content_hash: str):
    self.content_hash = content_hash
    self.git_hash = get_git_version()
    self.created_at = datetime.now(timezone.utc).isoformat()
    self.artifact_type = artifact_type.value
    self.artifact_id = f"{artifact_type.value}_{content_hash[:6]}"

  def to_dict(self) -> dict:
    """convert metadata to dictionary for serialization"""
    return {
      "artifact_id": self.artifact_id,
      "artifact_type": self.artifact_type,
      "created_at": self.created_at,
      "git_hash": self.git_hash,
      "content_hash": self.content_hash,
    }
