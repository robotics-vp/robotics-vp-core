"""Run manifest for provenance tracking.

Captures all provenance information for a closed-loop run including
code commit, plan hash, datapack manifest, seeds, and schema versions.
"""
from __future__ import annotations

import json
import subprocess
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.contracts.schemas import RunManifestV1
from src.utils.config_digest import sha256_json


def get_git_commit() -> Optional[str]:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()[:12]
    except Exception:
        pass
    return None


def compute_datapack_manifest_sha(datapack_ids: List[str]) -> str:
    """Compute SHA-256 of datapack manifest.

    Args:
        datapack_ids: List of datapack identifiers

    Returns:
        SHA-256 of sorted datapack IDs
    """
    return sha256_json(sorted(datapack_ids))


def create_run_manifest(
    run_id: Optional[str] = None,
    plan_sha: str = "",
    audit_suite_id: str = "default_audit",
    audit_seed: int = 42,
    audit_config_sha: str = "",
    datapack_ids: Optional[List[str]] = None,
    seeds: Optional[Dict[str, int]] = None,
    plan_path: Optional[str] = None,
    determinism_config: Optional[Dict[str, Any]] = None,
) -> RunManifestV1:
    """Create a run manifest for provenance.

    Args:
        run_id: Optional run identifier (generated if not provided)
        plan_sha: SHA-256 of the plan
        audit_suite_id: Audit suite identifier
        audit_seed: Audit seed
        audit_config_sha: SHA-256 of audit config
        datapack_ids: List of datapack IDs used
        seeds: Seed values used
        plan_path: Path to plan file
        determinism_config: Determinism configuration

    Returns:
        RunManifestV1 with provenance information
    """
    if run_id is None:
        run_id = str(uuid.uuid4())[:8]

    if seeds is None:
        seeds = {"audit": audit_seed}

    datapack_manifest_sha = compute_datapack_manifest_sha(datapack_ids or [])

    return RunManifestV1(
        run_id=run_id,
        source_commit=get_git_commit(),
        plan_path=plan_path,
        plan_sha=plan_sha,
        audit_suite_id=audit_suite_id,
        audit_seed=audit_seed,
        audit_config_sha=audit_config_sha,
        datapack_manifest_sha=datapack_manifest_sha,
        seeds=seeds,
        determinism_config=determinism_config,
    )


def write_manifest(path: str, manifest: RunManifestV1) -> None:
    """Write manifest to JSON file.

    Args:
        path: Output path
        manifest: Manifest to write
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(manifest.model_dump(mode="json"), f, indent=2)


__all__ = [
    "get_git_commit",
    "compute_datapack_manifest_sha",
    "create_run_manifest",
    "write_manifest",
]
