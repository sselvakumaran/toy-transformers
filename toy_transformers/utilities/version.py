from datetime import datetime, timezone
import subprocess
import sys

def get_git_version() -> str:
  try: return subprocess.check_output(
    ["git", "rev-parse", "--short", "HEAD"],
    stderr=subprocess.DEVNULL,
    text=True
  ).strip()
  except (subprocess.CalledProcessError, FileNotFoundError):
    return "unknown"

def get_obj_metadata(
  obj,
  include_timestamp: bool = True,
  include_hash: bool = False,
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
    out["hash"] = hash(obj)
  return out