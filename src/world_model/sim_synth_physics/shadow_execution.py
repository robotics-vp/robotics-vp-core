"""WM-owned backend shadow execution/materialization helpers."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Optional

from src.envs.physics.isaac_backend import IsaacBackend

from .common import mapping
from .physics_contracts import PhysicsExecutionContract
from .receipts import BackendExecutionBindingReceipt, BackendShadowExecutionReceipt
from .state import SimSynthPhysicsWorldState


def _int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8")


def _derive_isaac_shadow_env_config(
    world_state: SimSynthPhysicsWorldState,
    *,
    output_dir: Optional[str | Path],
) -> dict[str, Any]:
    embodiment_context = _mapping(world_state.input_context.get("embodiment"))
    control_constraints = _mapping(embodiment_context.get("control_constraints"))
    robot_asset_manifest = _mapping(
        embodiment_context.get("robot_asset_manifest")
        or embodiment_context.get("asset_manifest")
        or embodiment_context.get("robot_assets")
    )
    first_job = (
        world_state.simulation_agenda.jobs[0]
        if world_state.simulation_agenda.jobs
        else None
    )
    env_name = (
        _mapping(world_state.input_context.get("semantic")).get("env_name")
        or world_state.simulation_agenda.metadata.get("env_name")
        or "sim_synth_shadow"
    )
    action_dim = _int(
        embodiment_context.get("action_dim", control_constraints.get("action_dim", 12)),
        12,
    )
    task_name = getattr(first_job, "objective", None) or env_name
    return {
        "env_name": str(env_name),
        "task": str(task_name),
        "robot": str(
            embodiment_context.get("primary_embodiment")
            or embodiment_context.get("robot_family")
            or "unitree_shadow"
        ),
        "backend_mode": "shadow_contract",
        "max_steps": max(2, min(4, len(world_state.synthetic_branch_plans) + 1)),
        "dt": float(control_constraints.get("dt", 1.0 / 60.0) or 1.0 / 60.0),
        "action_dim": action_dim,
        "obs_dim": max(action_dim * 4, 32),
        "seed": 0,
        "econ_params": _mapping(world_state.input_context.get("economic")),
        "robot_asset_manifest": robot_asset_manifest,
        "output_root": (
            None
            if output_dir is None
            else str(Path(output_dir) / "backend_shadow_execution" / "isaac")
        ),
    }


def materialize_backend_shadow_execution(
    world_state: SimSynthPhysicsWorldState,
    execution_contract: PhysicsExecutionContract,
    backend_binding_receipt: BackendExecutionBindingReceipt,
    *,
    output_dir: str | Path | None = None,
) -> BackendShadowExecutionReceipt | None:
    """Run an explicit WM-owned shadow execution where a concrete backend is absent."""

    binding = world_state.backend_execution_binding
    if binding is None:
        return None
    if str(binding.backend) != "isaac":
        return None

    output_root = None if output_dir is None else Path(output_dir)
    env_config = _derive_isaac_shadow_env_config(world_state, output_dir=output_root)
    backend = IsaacBackend(
        env_config=env_config,
        num_envs=max(1, min(2, len(world_state.synthetic_branch_plans) or 1)),
        device="cuda:0",
    )
    artifact_refs: list[str] = []
    episode_ids: list[str] = []
    execution_status = "shadow_executed"
    summary_path = None if output_root is None else output_root / "backend_shadow_execution_receipt.json"
    batch_summary_path = None if output_root is None else output_root / "backend_shadow_batch_summary.json"

    try:
        backend.reset()
        zero_action = [0.0 for _ in range(_int(env_config.get("action_dim", 12), 12))]
        for _ in range(max(1, _int(env_config.get("max_steps", 2), 2) - 1)):
            backend.step(zero_action)
        episode_ids = [episode_id for episode_id in [backend.get_current_episode_id(0)] if episode_id]
        if backend.num_envs > 1:
            for env_idx in range(1, backend.num_envs):
                backend.reset_env(env_idx)
                episode_id = backend.get_current_episode_id(env_idx)
                if episode_id:
                    episode_ids.append(episode_id)
        media_ref_sets = [backend.get_media_refs(env_idx) for env_idx in range(backend.num_envs)]
        for media_refs in media_ref_sets:
            artifact_refs.extend(str(value) for value in media_refs.values() if value)
        batch_summaries = [asdict(summary) for summary in backend.get_batch_episode_info()]
        if batch_summary_path is not None:
            _write_json(
                batch_summary_path,
                {
                    "version": "backend_shadow_batch_summary_v1",
                    "backend": "isaac",
                    "execution_mode": "shadow_contract",
                    "batch_summaries": batch_summaries,
                    "binding_status": backend_binding_receipt.binding_status,
                },
            )
            artifact_refs.append(str(batch_summary_path.resolve()))
        if binding.missing_assets:
            execution_status = "shadow_executed_with_asset_gaps"
    except Exception as exc:
        execution_status = "shadow_failed"
        artifact_refs = []
        episode_ids = []
        batch_summaries = []
        failure = {"error": repr(exc)}
        if batch_summary_path is not None:
            _write_json(
                batch_summary_path,
                {
                    "version": "backend_shadow_batch_summary_v1",
                    "backend": "isaac",
                    "execution_mode": "shadow_contract",
                    "binding_status": backend_binding_receipt.binding_status,
                    "failure": failure,
                },
            )
            artifact_refs.append(str(batch_summary_path.resolve()))
    finally:
        backend.close()

    receipt = BackendShadowExecutionReceipt(
        receipt_id=f"backend_shadow_execution_receipt_{world_state.state_id}",
        backend="isaac",
        execution_mode="shadow_contract",
        execution_status=execution_status,
        episode_ids=episode_ids,
        artifact_refs=artifact_refs,
        metadata={
            "world_state_id": world_state.state_id,
            "physics_execution_contract_id": execution_contract.contract_id,
            "backend_execution_binding_receipt_id": backend_binding_receipt.receipt_id,
            "binding_status": backend_binding_receipt.binding_status,
            "requested_backend": execution_contract.requested_backend,
            "resolved_backend": execution_contract.resolved_backend,
            "route_status": execution_contract.route_status,
            "required_assets": list(
                backend_binding_receipt.metadata.get("required_assets", []) or []
            ),
            "available_assets": list(
                backend_binding_receipt.metadata.get("available_assets", []) or []
            ),
            "missing_assets": list(
                backend_binding_receipt.metadata.get("missing_assets", []) or []
            ),
            "env_config": mapping(env_config),
        },
    )
    if summary_path is not None:
        _write_json(summary_path, receipt.to_dict())
    return receipt


__all__ = ["materialize_backend_shadow_execution"]
