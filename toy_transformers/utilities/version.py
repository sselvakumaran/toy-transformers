from datetime import datetime, timezone
import subprocess
import sys
from toy_transformers.utilities.hashing import stable_hash

def get_git_version() -> str:
  try: return subprocess.check_output(
    ["git", "rev-parse", "--short", "HEAD"],
    stderr=subprocess.DEVNULL,
    text=True
  ).strip()
  except (subprocess.CalledProcessError, FileNotFoundError):
    return "unknown"

def get_pytorch_version() -> str:
  import torch
  return torch.__version__

def get_obj_metadata(
  obj,
  include_timestamp: bool = True,
  include_hash: bool = False,
  include_pytorch_version: bool = False
):
  obj_type = type(obj)
  out = {
    "type": obj_type.__name__,
    "module": obj_type.__module__,
    "git_hash": get_git_version(),
  }
  if include_timestamp:
    out["timestamp"] = datetime.now(timezone.utc).isoformat()
  if include_hash:
    # Use stable hash if object has stable_hash_value method, else use stable_hash
    if hasattr(obj, 'stable_hash_value'):
      out["hash"] = obj.stable_hash_value()
    else:
      out["hash"] = stable_hash(repr(obj))
  if include_pytorch_version:
    out["pytorch_version"] = get_pytorch_version()
  return out