"""WM-owned render provider materialization helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional

from .common import mapping
from .physics_contracts import PhysicsExecutionContract
from .receipts import PhysicsAdaptationReceipt, RenderProviderReceipt
from .state import BranchRenderProviderState, SimSynthPhysicsWorldState, SyntheticBranchPlan


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8")


def _provider_output_dir(output_dir: Optional[str | Path], plan_id: str) -> Optional[Path]:
    if output_dir is None:
        return None
    return Path(output_dir) / "render_provider_materializations" / plan_id


def _artifact_ref(path: Path) -> str:
    return str(path.resolve())


def _materialize_lsd_scene(
    *,
    output_root: Optional[Path],
    provider: BranchRenderProviderState,
    plan: SyntheticBranchPlan,
) -> tuple[str, str, list[str], dict[str, Any]]:
    artifact_refs: list[str] = []
    metadata = {
        "provider_truth_class": "scene_materialization",
        "unsatisfied_preconditions": [],
    }
    if output_root is not None:
        contract_path = output_root / "render_provider_contract.json"
        config_path = output_root / "lsd_vector_scene_config.json"
        manifest_path = output_root / "render_materialization_manifest.json"
        _write_json(contract_path, provider.to_dict())
        _write_json(
            config_path,
            mapping(provider.provider_config.get("lsd_vector_scene", provider.provider_config)),
        )
        _write_json(
            manifest_path,
            {
                "version": "render_materialization_manifest_v1",
                "branch_plan_id": plan.plan_id,
                "provider_kind": provider.provider_kind,
                "materialization_mode": "scene_config",
                "materialization_status": "scene_materialized",
                "unsatisfied_preconditions": [],
            },
        )
        artifact_refs.extend(
            [_artifact_ref(contract_path), _artifact_ref(config_path), _artifact_ref(manifest_path)]
        )
    return "scene_materialized", "scene_config", artifact_refs, metadata


def _materialize_nag_counterfactual(
    *,
    output_root: Optional[Path],
    provider: BranchRenderProviderState,
    plan: SyntheticBranchPlan,
) -> tuple[str, str, list[str], dict[str, Any]]:
    artifact_refs: list[str] = []
    metadata = {
        "provider_truth_class": "counterfactual_work_order",
        "unsatisfied_preconditions": ["source_lsd_rollout"],
    }
    if not provider.metadata.get("nag_renderer_available", False):
        metadata["unsatisfied_preconditions"].append("nag_gaussian_renderer")
    if provider.provider_status == "blocked":
        metadata["unsatisfied_preconditions"].append("lsd_scene_provider")
    if output_root is not None:
        contract_path = output_root / "render_provider_contract.json"
        work_order_path = output_root / "nag_counterfactual_work_order.json"
        _write_json(contract_path, provider.to_dict())
        _write_json(
            work_order_path,
            {
                "version": "nag_counterfactual_work_order_v1",
                "branch_plan_id": plan.plan_id,
                "provider_kind": provider.provider_kind,
                "render_mode": provider.render_mode,
                "counterfactual_mode": provider.counterfactual_mode,
                "provider_config": mapping(provider.provider_config),
                "unsatisfied_preconditions": list(metadata["unsatisfied_preconditions"]),
            },
        )
        artifact_refs.extend([_artifact_ref(contract_path), _artifact_ref(work_order_path)])
    status = (
        "work_order_materialized"
        if not metadata["unsatisfied_preconditions"]
        else "work_order_materialized_with_preconditions"
    )
    return status, "counterfactual_work_order", artifact_refs, metadata


def _materialize_ggds_scene(
    *,
    output_root: Optional[Path],
    provider: BranchRenderProviderState,
    plan: SyntheticBranchPlan,
) -> tuple[str, str, list[str], dict[str, Any]]:
    artifact_refs: list[str] = []
    metadata = {
        "provider_truth_class": "ggds_work_order",
        "unsatisfied_preconditions": ["source_lsd_scene"],
    }
    if not provider.metadata.get("ggds_concrete_available", False):
        metadata["unsatisfied_preconditions"].append("ggds_ldm_renderer")
    if output_root is not None:
        contract_path = output_root / "render_provider_contract.json"
        lsd_config_path = output_root / "lsd_vector_scene_config.json"
        ggds_path = output_root / "ggds_work_order.json"
        _write_json(contract_path, provider.to_dict())
        _write_json(
            lsd_config_path,
            mapping(provider.provider_config.get("lsd_vector_scene", {})),
        )
        _write_json(
            ggds_path,
            {
                "version": "ggds_work_order_v1",
                "branch_plan_id": plan.plan_id,
                "provider_kind": provider.provider_kind,
                "render_mode": provider.render_mode,
                "ggds_mode": provider.ggds_mode,
                "provider_config": mapping(provider.provider_config.get("ggds", provider.provider_config)),
                "unsatisfied_preconditions": list(metadata["unsatisfied_preconditions"]),
            },
        )
        artifact_refs.extend(
            [_artifact_ref(contract_path), _artifact_ref(lsd_config_path), _artifact_ref(ggds_path)]
        )
    status = (
        "work_order_materialized"
        if not metadata["unsatisfied_preconditions"]
        else "work_order_materialized_with_preconditions"
    )
    return status, "ggds_work_order", artifact_refs, metadata


def materialize_render_provider_receipts(
    world_state: SimSynthPhysicsWorldState,
    execution_contract: PhysicsExecutionContract,
    adaptation_receipt: PhysicsAdaptationReceipt,
    *,
    output_dir: str | Path | None = None,
) -> list[RenderProviderReceipt]:
    """Materialize WM-owned render-provider artifacts and receipts."""

    receipts: list[RenderProviderReceipt] = []
    for index, plan in enumerate(world_state.synthetic_branch_plans, start=1):
        provider = plan.render_provider
        if provider is None:
            continue
        output_root = _provider_output_dir(output_dir, plan.plan_id)
        if provider.provider_status == "blocked":
            materialization_status = "materialization_blocked"
            materialization_mode = "blocked"
            artifact_refs: list[str] = []
            provider_metadata = {
                "provider_truth_class": "blocked_provider",
                "unsatisfied_preconditions": ["provider_unavailable"],
            }
        elif provider.provider_kind == "lsd_scene_graph":
            (
                materialization_status,
                materialization_mode,
                artifact_refs,
                provider_metadata,
            ) = _materialize_lsd_scene(output_root=output_root, provider=provider, plan=plan)
        elif provider.provider_kind == "nag_lsd_counterfactual":
            (
                materialization_status,
                materialization_mode,
                artifact_refs,
                provider_metadata,
            ) = _materialize_nag_counterfactual(
                output_root=output_root,
                provider=provider,
                plan=plan,
            )
        else:
            (
                materialization_status,
                materialization_mode,
                artifact_refs,
                provider_metadata,
            ) = _materialize_ggds_scene(output_root=output_root, provider=provider, plan=plan)
        receipts.append(
            RenderProviderReceipt(
                receipt_id=f"render_provider_receipt_{world_state.state_id}_{index:03d}",
                branch_plan_id=plan.plan_id,
                provider_id=provider.provider_id,
                provider_kind=provider.provider_kind,
                provider_status=provider.provider_status,
                render_mode=provider.render_mode,
                counterfactual_mode=provider.counterfactual_mode,
                materialization_status=materialization_status,
                materialization_mode=materialization_mode,
                materialization_entrypoint=provider.materialization_entrypoint,
                artifact_refs=artifact_refs,
                metadata={
                    "world_state_id": world_state.state_id,
                    "physics_execution_contract_id": execution_contract.contract_id,
                    "physics_adaptation_receipt_id": adaptation_receipt.receipt_id,
                    "generation_mode": plan.generation_mode,
                    "fallback_provider": provider.fallback_provider,
                    "fallback_reason": provider.fallback_reason,
                    "ggds_mode": provider.ggds_mode,
                    "provider_config": mapping(provider.provider_config),
                    "target_hardware_class": adaptation_receipt.target_hardware_class,
                    "provider_metadata": mapping(provider.metadata),
                    "provider_truth_class": str(provider_metadata.get("provider_truth_class", "")),
                    "unsatisfied_preconditions": list(
                        provider_metadata.get("unsatisfied_preconditions", [])
                    ),
                },
            )
        )
    return receipts


__all__ = ["materialize_render_provider_receipts"]
