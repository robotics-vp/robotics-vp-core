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
    # Optional provenance fields
    probe_config_sha: Optional[str] = None,
    probe_report_sha: Optional[str] = None,
    plan_policy_config_sha: Optional[str] = None,
    baseline_weights_sha: Optional[str] = None,
    final_weights_sha: Optional[str] = None,
    plan_applied_events_sha: Optional[str] = None,
    graph_spec_sha: Optional[str] = None,
    graph_summary_sha: Optional[str] = None,
    # Regal provenance (Stage-6 meta-regal)
    regal_config_sha: Optional[str] = None,
    regal_report_sha: Optional[str] = None,
    regal_inputs_sha: Optional[str] = None,
    regal_context_sha: Optional[str] = None,
    # Trajectory audit provenance
    trajectory_audit_sha: Optional[str] = None,
    # Economics provenance
    econ_basis_sha: Optional[str] = None,
    econ_tensor_sha: Optional[str] = None,
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
        probe_config_sha: Optional probe config SHA
        probe_report_sha: Optional probe report SHA
        plan_policy_config_sha: Optional plan policy config SHA
        baseline_weights_sha: Optional baseline weights SHA
        final_weights_sha: Optional final weights SHA
        plan_applied_events_sha: Optional plan applied events SHA
        graph_spec_sha: Optional graph spec SHA
        graph_summary_sha: Optional graph summary SHA
        regal_config_sha: Optional regal config SHA (Stage-6)
        regal_report_sha: Optional regal report SHA (Stage-6)
        regal_inputs_sha: Optional regal inputs SHA (Stage-6)
        regal_context_sha: Optional regal context SHA (Stage-6)
        trajectory_audit_sha: Optional trajectory audit SHA
        econ_basis_sha: Optional econ basis SHA
        econ_tensor_sha: Optional econ tensor SHA

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
        probe_config_sha=probe_config_sha,
        probe_report_sha=probe_report_sha,
        plan_policy_config_sha=plan_policy_config_sha,
        baseline_weights_sha=baseline_weights_sha,
        final_weights_sha=final_weights_sha,
        plan_applied_events_sha=plan_applied_events_sha,
        graph_spec_sha=graph_spec_sha,
        graph_summary_sha=graph_summary_sha,
        regal_config_sha=regal_config_sha,
        regal_report_sha=regal_report_sha,
        regal_inputs_sha=regal_inputs_sha,
        regal_context_sha=regal_context_sha,
        trajectory_audit_sha=trajectory_audit_sha,
        econ_basis_sha=econ_basis_sha,
        econ_tensor_sha=econ_tensor_sha,
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
