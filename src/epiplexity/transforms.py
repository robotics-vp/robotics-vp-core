"""Transform chain hashing for epiplexity caching."""
from __future__ import annotations

from typing import Sequence
import hashlib
import json


def transform_chain_hash(chain: Sequence[str]) -> str:
    payload = json.dumps(list(chain), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


__all__ = ["transform_chain_hash"]
