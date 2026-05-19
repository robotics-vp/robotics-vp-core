"""WM-owned render provider materialization helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

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


def _render_source_context(
    world_state: SimSynthPhysicsWorldState,
    plan: SyntheticBranchPlan,
) -> dict[str, Any]:
    semantic_context = mapping(world_state.input_context.get("semantic"))
    context = mapping(semantic_context.get("render_source_context"))
    for key in (
        "source_lsd_episode",
        "lsd_backend_episode",
        "source_gaussian_scene",
        "source_gaussian_scene_path",
        "render_prompts",
    ):
        value = semantic_context.get(key)
        if value not in (None, "", [], {}):
            context.setdefault(key, value)
    future_training = mapping(plan.metadata.get("future_training_artifacts"))
    if future_training:
        context.setdefault("future_training_artifacts", future_training)
    scene_hierarchy = (
        {}
        if world_state.scene_hierarchy is None
        else world_state.scene_hierarchy.to_dict()
    )
    if scene_hierarchy:
        context.setdefault("scene_hierarchy", scene_hierarchy)
    plan_scene_ref = mapping(plan.metadata.get("scene_hierarchy_ref"))
    if plan_scene_ref:
        context.setdefault("scene_hierarchy_ref", plan_scene_ref)
    return context


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
                "scene_hierarchy_ref": mapping(plan.metadata.get("scene_hierarchy_ref")),
                "unsatisfied_preconditions": [],
            },
        )
        artifact_refs.extend(
            [_artifact_ref(contract_path), _artifact_ref(config_path), _artifact_ref(manifest_path)]
        )
    return "scene_materialized", "scene_config", artifact_refs, metadata


def _materialize_nag_counterfactual(
    *,
    world_state: SimSynthPhysicsWorldState,
    output_root: Optional[Path],
    provider: BranchRenderProviderState,
    plan: SyntheticBranchPlan,
) -> tuple[str, str, list[str], dict[str, Any]]:
    artifact_refs: list[str] = []
    source_context = _render_source_context(world_state, plan)
    metadata = {
        "provider_truth_class": "counterfactual_work_order",
        "unsatisfied_preconditions": [],
    }
    backend_episode = mapping(
        source_context.get("source_lsd_episode") or source_context.get("lsd_backend_episode")
    )
    if not backend_episode:
        metadata["unsatisfied_preconditions"].append("source_lsd_episode")
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
                "source_context_keys": sorted(source_context),
                "unsatisfied_preconditions": list(metadata["unsatisfied_preconditions"]),
            },
        )
        artifact_refs.extend([_artifact_ref(contract_path), _artifact_ref(work_order_path)])
    if metadata["unsatisfied_preconditions"]:
        return "work_order_materialized_with_preconditions", "counterfactual_work_order", artifact_refs, metadata
    try:
        from src.vision.nag.integration_lsd_backend import (
            NAGEditPolicyConfig,
            NAGFromLSDConfig,
            create_camera_from_lsd_config,
            generate_nag_counterfactuals_for_lsd_episode,
        )

        nag_config = NAGFromLSDConfig(
            **mapping(provider.provider_config.get("nag_from_lsd"))
        )
        if bool(getattr(nag_config, "use_stub_renderer", False)):
            raise RuntimeError(
                "NAG config still requests stub rendering; concrete counterfactual execution requires a real renderer."
            )
        edit_config = NAGEditPolicyConfig(
            **mapping(provider.provider_config.get("nag_edit_policy"))
        )
        camera = create_camera_from_lsd_config(nag_config)
        datapacks = generate_nag_counterfactuals_for_lsd_episode(
            backend_episode=backend_episode,
            camera=camera,
            nag_config=nag_config,
            edit_config=edit_config,
        )
        metadata = {
            "provider_truth_class": "counterfactual_datapacks",
            "unsatisfied_preconditions": [],
            "counterfactual_count": len(datapacks),
        }
        if output_root is not None:
            manifest_rows: list[dict[str, Any]] = []
            for index, datapack in enumerate(datapacks):
                payload_path = output_root / f"nag_counterfactual_{index:03d}.npz"
                summary_path = output_root / f"nag_counterfactual_{index:03d}.json"
                np.savez_compressed(payload_path, frames=datapack.frames)
                _write_json(summary_path, datapack.to_dict())
                artifact_refs.extend([_artifact_ref(payload_path), _artifact_ref(summary_path)])
                manifest_rows.append(
                    {
                        "counterfactual_id": datapack.counterfactual_id,
                        "num_edits": len(datapack.nag_edit_vector),
                        "frames_shape": list(datapack.frames.shape),
                    }
                )
            manifest_path = output_root / "nag_counterfactual_manifest.json"
            _write_json(
                manifest_path,
                {
                    "version": "nag_counterfactual_manifest_v1",
                    "branch_plan_id": plan.plan_id,
                    "count": len(datapacks),
                    "rows": manifest_rows,
                },
            )
            artifact_refs.append(_artifact_ref(manifest_path))
        return "counterfactuals_materialized", "counterfactual_datapacks", artifact_refs, metadata
    except Exception as exc:
        metadata = {
            "provider_truth_class": "counterfactual_generation_failure",
            "unsatisfied_preconditions": [],
            "error": str(exc),
        }
        if output_root is not None:
            failure_path = output_root / "nag_counterfactual_failure.json"
            _write_json(
                failure_path,
                {
                    "version": "nag_counterfactual_failure_v1",
                    "branch_plan_id": plan.plan_id,
                    "error": str(exc),
                },
            )
            artifact_refs.append(_artifact_ref(failure_path))
        return "counterfactual_generation_failed", "counterfactual_datapacks", artifact_refs, metadata


def _materialize_ggds_scene(
    *,
    world_state: SimSynthPhysicsWorldState,
    output_root: Optional[Path],
    provider: BranchRenderProviderState,
    plan: SyntheticBranchPlan,
) -> tuple[str, str, list[str], dict[str, Any]]:
    artifact_refs: list[str] = []
    source_context = _render_source_context(world_state, plan)
    metadata = {
        "provider_truth_class": "ggds_work_order",
        "unsatisfied_preconditions": [],
    }
    source_scene_payload = mapping(source_context.get("source_gaussian_scene"))
    source_scene_path = str(source_context.get("source_gaussian_scene_path", "") or "")
    if not source_scene_payload and source_scene_path:
        try:
            source_scene_payload = json.loads(Path(source_scene_path).read_text(encoding="utf-8"))
        except Exception:
            source_scene_payload = {}
    if not source_scene_payload:
        metadata["unsatisfied_preconditions"].append("source_gaussian_scene")
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
                "source_context_keys": sorted(source_context),
                "unsatisfied_preconditions": list(metadata["unsatisfied_preconditions"]),
            },
        )
        artifact_refs.extend(
            [_artifact_ref(contract_path), _artifact_ref(lsd_config_path), _artifact_ref(ggds_path)]
        )
    if metadata["unsatisfied_preconditions"]:
        return "work_order_materialized_with_preconditions", "ggds_work_order", artifact_refs, metadata
    try:
        from src.envs.lsd3d_env.gaussian_scene import GaussianScene
        from src.envs.lsd3d_env.ggds import CameraRig, create_default_optimizer

        optimizer = create_default_optimizer()
        if not getattr(optimizer, "_is_initialized", False):
            raise RuntimeError("GGDS optimizer is not concretely initialized.")
        scene = GaussianScene.from_dict(source_scene_payload)
        ggds_config = mapping(provider.provider_config.get("ggds", {}))
        prompts = list(ggds_config.get("prompts") or ["a realistic scene"])
        camera_rig = CameraRig.create_orbit(
            center=(0.0, 0.0, 0.0),
            radius=5.0,
            num_views=int(ggds_config.get("num_views", 4) or 4),
            resolution=(64, 64),
        )
        optimized_scene = optimizer.optimize_scene(
            scene,
            camera_rig,
            prompts=prompts,
            num_iterations=int(ggds_config.get("num_iterations", 4) or 4),
        )
        metadata = {
            "provider_truth_class": "ggds_scene_materialization",
            "unsatisfied_preconditions": [],
            "optimized_gaussian_count": int(getattr(optimized_scene, "num_gaussians", 0)),
        }
        if output_root is not None:
            scene_path = output_root / "optimized_gaussian_scene.json"
            manifest_path = output_root / "ggds_scene_manifest.json"
            _write_json(scene_path, optimized_scene.to_dict())
            _write_json(
                manifest_path,
                {
                    "version": "ggds_scene_manifest_v1",
                    "branch_plan_id": plan.plan_id,
                    "provider_kind": provider.provider_kind,
                    "optimized_gaussian_count": int(getattr(optimized_scene, "num_gaussians", 0)),
                },
            )
            artifact_refs.extend([_artifact_ref(scene_path), _artifact_ref(manifest_path)])
        return "ggds_scene_materialized", "ggds_scene_optimization", artifact_refs, metadata
    except Exception as exc:
        metadata = {
            "provider_truth_class": "ggds_generation_failure",
            "unsatisfied_preconditions": [],
            "error": str(exc),
        }
        if output_root is not None:
            failure_path = output_root / "ggds_scene_failure.json"
            _write_json(
                failure_path,
                {
                    "version": "ggds_scene_failure_v1",
                    "branch_plan_id": plan.plan_id,
                    "error": str(exc),
                },
            )
            artifact_refs.append(_artifact_ref(failure_path))
        return "ggds_scene_generation_failed", "ggds_scene_optimization", artifact_refs, metadata


def materialize_render_provider_receipts(
    world_state: SimSynthPhysicsWorldState,
    execution_contract: PhysicsExecutionContract,
    adaptation_receipt: PhysicsAdaptationReceipt | None,
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
                world_state=world_state,
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
            ) = _materialize_ggds_scene(
                world_state=world_state,
                output_root=output_root,
                provider=provider,
                plan=plan,
            )
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
                    "physics_adaptation_receipt_id": (
                        "" if adaptation_receipt is None else adaptation_receipt.receipt_id
                    ),
                    "generation_mode": plan.generation_mode,
                    "fallback_provider": provider.fallback_provider,
                    "fallback_reason": provider.fallback_reason,
                    "ggds_mode": provider.ggds_mode,
                    "provider_config": mapping(provider.provider_config),
                    "target_hardware_class": (
                        execution_contract.target_hardware_class
                        if adaptation_receipt is None
                        else adaptation_receipt.target_hardware_class
                    ),
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
