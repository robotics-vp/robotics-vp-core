"""
Provenance tracking for Scene IR Tracker.

Captures reproducibility information for exports and runs.
"""
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class Provenance:
    """Provenance information for reproducibility.
    
    Attributes:
        timestamp: ISO 8601 timestamp.
        repo_git_hash: Git commit hash of main repo.
        third_party_hashes: Dict of third_party module -> git hash.
        config_json: JSON-serialized config.
        seed: Random seed used.
        device: Device used (cpu/cuda).
        precision: Precision used (float32/float16).
        python_version: Python version.
        torch_version: PyTorch version (if available).
        numpy_version: NumPy version.
    """
    timestamp: str
    repo_git_hash: Optional[str]
    third_party_hashes: Dict[str, Optional[str]]
    config_json: str
    seed: Optional[int]
    device: str
    precision: str
    python_version: str
    torch_version: Optional[str]
    numpy_version: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


def get_git_hash(repo_path: Optional[Path] = None) -> Optional[str]:
    """Get git commit hash for a repository.
    
    Args:
        repo_path: Path to repository root. Uses cwd if None.
    
    Returns:
        Git hash string or None if not a git repo.
    """
    try:
        cmd = ["git", "rev-parse", "HEAD"]
        result = subprocess.run(
            cmd,
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()[:12]
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    return None


def get_third_party_hashes() -> Dict[str, Optional[str]]:
    """Get git hashes for third_party submodules.
    
    Returns:
        Dict mapping module name to git hash.
    """
    hashes = {}
    
    # Find third_party directory
    third_party_dir = Path(__file__).parent.parent.parent.parent / "third_party"
    
    modules = ["sam3d_objects", "sam3d_body", "inrtracker"]
    
    for module in modules:
        module_path = third_party_dir / module
        if module_path.is_dir():
            hashes[module] = get_git_hash(module_path)
        else:
            hashes[module] = None
    
    return hashes


def get_provenance(
    config: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
    device: str = "cpu",
    precision: str = "float32",
) -> Provenance:
    """Collect provenance information.
    
    Args:
        config: Configuration dict to include.
        seed: Random seed used.
        device: Device used.
        precision: Precision used.
    
    Returns:
        Provenance instance.
    """
    import sys
    
    # Get versions
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    numpy_version = np.__version__
    
    torch_version = None
    try:
        import torch
        torch_version = torch.__version__
    except ImportError:
        pass
    
    # Get main repo hash
    # Walk up from this file to find repo root
    repo_root = Path(__file__).parent.parent.parent.parent
    repo_hash = get_git_hash(repo_root)
    
    # Get third_party hashes
    third_party_hashes = get_third_party_hashes()
    
    # Serialize config
    config_json = json.dumps(config or {}, indent=2, default=str)
    
    return Provenance(
        timestamp=datetime.utcnow().isoformat() + "Z",
        repo_git_hash=repo_hash,
        third_party_hashes=third_party_hashes,
        config_json=config_json,
        seed=seed,
        device=device,
        precision=precision,
        python_version=python_version,
        torch_version=torch_version,
        numpy_version=numpy_version,
    )


def provenance_to_npz_dict(provenance: Provenance) -> Dict[str, np.ndarray]:
    """Convert provenance to numpy arrays for npz storage.
    
    Args:
        provenance: Provenance to convert.
    
    Returns:
        Dict suitable for np.savez.
    """
    return {
        "provenance_json": np.array([provenance.to_json()], dtype="U8192"),
    }


def print_provenance(provenance: Provenance) -> None:
    """Print provenance information to stdout."""
    print("=" * 60)
    print("PROVENANCE")
    print("=" * 60)
    print(f"Timestamp: {provenance.timestamp}")
    print(f"Repo Hash: {provenance.repo_git_hash or 'unknown'}")
    print(f"Device: {provenance.device}")
    print(f"Precision: {provenance.precision}")
    print(f"Seed: {provenance.seed}")
    print(f"Python: {provenance.python_version}")
    print(f"NumPy: {provenance.numpy_version}")
    print(f"PyTorch: {provenance.torch_version or 'not installed'}")
    print("-" * 60)
    print("Third Party:")
    for module, hash_val in provenance.third_party_hashes.items():
        print(f"  {module}: {hash_val or 'not installed'}")
    print("=" * 60)


if __name__ == "__main__":
    # Quick test
    prov = get_provenance(
        config={"test": True},
        seed=42,
        device="cpu",
        precision="float32",
    )
    print_provenance(prov)
