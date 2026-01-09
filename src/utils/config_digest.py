"""
Configuration digest utility.
Computes a SHA-256 digest of the configuration tree for reproducibility.
"""
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Union


def compute_config_digest(config: Dict[str, Any]) -> str:
    """
    Compute a SHA-256 digest of the configuration dictionary.

    Args:
        config: Configuration dictionary

    Returns:
        First 16 characters of the SHA-256 hex digest
    """
    try:
        # Sort keys to ensure deterministic ordering
        config_str = json.dumps(config, sort_keys=True, default=str)
        digest = hashlib.sha256(config_str.encode("utf-8")).hexdigest()
        return digest[:16]
    except Exception as e:
        return "digest_error"


def sha256_file(path: Union[str, Path]) -> str:
    """Compute SHA-256 of a file.

    Args:
        path: Path to file

    Returns:
        Full SHA-256 hex digest
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(obj: Any) -> str:
    """Compute SHA-256 of a JSON-serializable object (canonicalized).

    Args:
        obj: JSON-serializable object

    Returns:
        Full SHA-256 hex digest
    """
    # Canonicalize with sorted keys
    json_str = json.dumps(obj, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(json_str.encode("utf-8")).hexdigest()

