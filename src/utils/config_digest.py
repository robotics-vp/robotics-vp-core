"""
Configuration digest utility.
Computes a SHA-256 digest of the configuration tree for reproducibility.
"""
import hashlib
import json
from typing import Any, Dict

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
