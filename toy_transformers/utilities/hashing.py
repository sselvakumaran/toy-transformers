import hashlib
import torch
from typing import Any, Hashable

# TODO: use __hash__ if available for object
# TODO: probably centralize hashing within library (do this with next structural change) like bpe.py
# ie. __hash__ requires int64 but stable_hash doesn't - just enforce ints for all hashes

def stable_hash(obj: Any) -> str:
    """create stable SHA256 hash of object.
    should be consistent between runs of python
    (default hash() is inconsistent between executions)
    handles: str, bytes, int, float, tuple, list, torch.Tensor, 
    __hash__ (typing.Hashable), __str__.
    """
    hasher = hashlib.sha256()
    _update_hasher(hasher, obj)
    return hasher.hexdigest()


def _update_hasher(hasher: hashlib.sha256, obj: Any) -> None:
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
        hasher.update(b']')
    elif isinstance(obj, torch.Tensor):
        # For tensors, hash shape, dtype, and sample of values
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
    # NOTE: IDEALLY INPUT IS STABLE, CHECK BEFORE INPUTTING
    elif isinstance(obj, Hashable): 
        hasher.update(obj.__hash__())
    else:
        hasher.update(str(obj).encode('utf-8'))