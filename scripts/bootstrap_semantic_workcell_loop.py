#!/usr/bin/env python3
"""Bootstrap a locally grounded workcell semantic loop end to end."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evidence.belief_state import BeliefState
from src.evidence.scene_tracks_truth import normalize_scene_tracks_truth
from src.constraints.constraint_set import ConstraintSet
from src.economics.functor import ObjectiveEconFunctor
from src.envs.workcell_env import WorkcellEnv
from src.envs.workcell_env.config import WorkcellEnvConfig
from src.envs.workcell_env.observations.mujoco_render import (
    build_segmentation_label_map,
    render_workcell_frames,
)
from src.envs.workcell_env.scene.scene_spec import FixtureSpec, PartSpec, ToolSpec, WorkcellSceneSpec
from src.envs.workcell_env.tasks.peg_in_hole import PegInHoleTask
from src.motor_backend.rollout_capture import EpisodeMetadata, record_episode_rollout, start_rollout_capture
from src.motor_backend.sensor_bundle import SensorBundleData
from src.objectives.profile_loader import load_contract_profile
from src.objectives.runtime_builder import ObjectiveRuntimeBuilder, ObjectiveRuntimeRecord, SourceDomain
from src.orchestrator.coverage_loop import run_coverage_loop
from src.orchestrator.semantic_runtime_learning import (
    build_semantic_runtime_learning_corpus,
    write_semantic_runtime_learning_corpus,
)
from src.replay.dataset import ReplayDatasetBuilder, load_replay_dataset
from src.runtime import (
    DecisionLedgerEntry,
    RuntimeEvent,
    decision_ledger_sidecar_payload,
    event_spine_sidecar_payload,
)
from src.runtime.packets import SchemaRef, runtime_packet_from_record
from src.semantic.runtime_backbone import SemanticRuntimeBackbone
from src.vision.scene_ir_tracker.io.scene_tracks_runner import run_scene_tracks
from src.world_model.semantic_world_model import SemanticWorldModelBuilder


def _mujoco_available() -> bool:
    return importlib.util.find_spec("mujoco") is not None


def _build_scene_spec(seed: int, episode_idx: int) -> WorkcellSceneSpec:
    x_offset = ((seed + episode_idx) % 5 - 2) * 0.01
    y_offset = ((seed + 2 * episode_idx) % 5 - 2) * 0.008
    return WorkcellSceneSpec(
        workcell_id=f"semantic_bootstrap_{episode_idx:03d}",
        fixtures=[
            FixtureSpec(
                id="hole",
                position=(0.03 + x_offset, -0.02 + y_offset, 0.05),
                orientation=(1.0, 0.0, 0.0, 0.0),
                fixture_type="vise",
            )
        ],
        parts=[
            PartSpec(
                id="peg",
                position=(-0.03 + x_offset, 0.01 - y_offset, 0.15),
                orientation=(1.0, 0.0, 0.0, 0.0),
                part_type="peg",
                dimensions_mm=(30.0, 30.0, 60.0),
                material="aluminum",
            )
        ],
        tools=[
            ToolSpec(
                id="end_effector",
                position=(0.0, 0.0, 0.25),
                orientation=(1.0, 0.0, 0.0, 0.0),
                tool_type="gripper",
                precision_mm=0.5,
            )
        ],
        spatial_bounds=(1.0, 1.0, 1.0),
    )


def _action_profiles() -> Sequence[Sequence[Sequence[float]]]:
    return (
        ((0.0, 0.0, -0.01), (0.01, 0.0, -0.01), (0.01, -0.005, -0.01), (0.0, 0.0, -0.005)),
        ((0.01, 0.0, -0.008), (0.01, 0.0, -0.008), (-0.005, 0.008, -0.01), (0.0, 0.0, -0.01)),
        ((-0.008, 0.006, -0.008), (0.012, -0.006, -0.01), (0.0, 0.0, -0.012), (0.0, 0.0, -0.01)),
    )


def _build_scene_object_catalog(scene_spec: WorkcellSceneSpec) -> List[Dict[str, Any]]:
    catalog: List[Dict[str, Any]] = []
    for fixture in list(scene_spec.fixtures):
        catalog.append(
            {
                "object_id": fixture.id,
                "class_name": fixture.fixture_type,
                "category": "fixture",
                "semantic_tags": [fixture.fixture_type, "affordance:align"],
            }
        )
    for part in list(scene_spec.parts):
        catalog.append(
            {
                "object_id": part.id,
                "class_name": part.part_type,
                "category": "workpiece",
                "semantic_tags": [part.part_type, "affordance:insert", "affordance:pick"],
            }
        )
    for tool in list(scene_spec.tools):
        catalog.append(
            {
                "object_id": tool.id,
                "class_name": tool.tool_type,
                "category": "tool",
                "semantic_tags": [tool.tool_type, "affordance:grasp", "affordance:align"],
            }
        )
    return catalog


def _episode_actions(episode_idx: int, steps: int) -> List[Dict[str, Any]]:
    profiles = _action_profiles()
    base_profile = profiles[episode_idx % len(profiles)]
    actions: List[Dict[str, Any]] = []
    for step_idx in range(steps):
        delta = base_profile[min(step_idx, len(base_profile) - 1)]
        actions.append({"object_id": "end_effector", "delta_position": tuple(float(v) for v in delta)})
    return actions


def _episode_timestamp(seed: int, episode_idx: int, *, step_idx: int = 0) -> str:
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    offset_s = int(seed) * 97 + int(episode_idx) * 11 + int(step_idx)
    return (base + timedelta(seconds=offset_s)).isoformat()


def _grounded_data_ready(scene_tracks_truth: Dict[str, Any]) -> bool:
    return bool(
        scene_tracks_truth.get("scene_tracks_non_stub", False)
        and scene_tracks_truth.get("scene_tracks_training_eligible", False)
    )


def _build_bootstrap_schema_refs(
    *,
    states: Sequence[Dict[str, Any]],
    actions: Sequence[Dict[str, Any]],
    camera: str,
    time_step_s: float,
) -> tuple[SchemaRef, SchemaRef]:
    first_state = dict(states[0] or {}) if states else {}
    first_action = dict(actions[0] or {}) if actions else {}
    sample_hz = (1.0 / float(time_step_s)) if float(time_step_s) > 0.0 else 0.0
    observation_schema = SchemaRef(
        schema_id="workcell_bootstrap_observation_v1",
        version="v1",
        shape={
            "state_keys": sorted(first_state.keys()),
            "joint_position_dim": len(list(first_state.get("joint_positions", []) or [])),
            "cameras": [str(camera)],
        },
        timing={"sample_hz": sample_hz, "time_step_s": float(time_step_s)},
        provenance={"source_adapter": "bootstrap_semantic_workcell_loop"},
        metadata={"env_id": "workcell"},
    )
    action_schema = SchemaRef(
        schema_id="workcell_bootstrap_action_v1",
        version="v1",
        shape={
            "action_keys": sorted(first_action.keys()),
            "delta_position_dim": len(list(first_action.get("delta_position", []) or [])),
        },
        timing={"apply_hz": sample_hz, "time_step_s": float(time_step_s)},
        provenance={"source_adapter": "bootstrap_semantic_workcell_loop"},
        metadata={"robot_id": "workcell"},
    )
    return observation_schema, action_schema


def _write_bootstrap_trace_sidecars(
    *,
    episode_dir: Path,
    scenario_id: str,
    episode_id: str,
    seed: int,
    episode_idx: int,
    scene_spec: WorkcellSceneSpec,
    states: Sequence[Dict[str, Any]],
    actions: Sequence[Dict[str, Any]],
    camera: str,
    time_step_s: float,
    metrics: Dict[str, Any],
    grounding_mode: str,
    backend_policy: str,
    backend_selected: str,
    scene_tracks_truth: Dict[str, Any],
    scene_tracks_summary: Dict[str, Any],
    belief_state: BeliefState,
    world_model: Any,
) -> Dict[str, Any]:
    observation_schema, action_schema = _build_bootstrap_schema_refs(
        states=states,
        actions=actions,
        camera=camera,
        time_step_s=time_step_s,
    )
    runtime_timestamp = _episode_timestamp(seed, episode_idx)
    grounded_ready = _grounded_data_ready(scene_tracks_truth)
    runtime_metrics = {
        "reward_total": float(metrics.get("reward", 0.0)),
        "steps": int(len(actions)),
        "time_step_s": float(time_step_s),
        "throughput_units_per_hour": float(max(len(actions), 1) * 12.0),
        "error_rate": float(max(0.0, 1.0 - float(metrics.get("quality_score", 0.0)))),
        "safety_score": float(metrics.get("safety_score", 0.0)),
        "energy_wh_per_unit": float(metrics.get("energy_wh_per_unit", 0.0)),
    }
    runtime_record = ObjectiveRuntimeRecord(
        task_id="peg_in_hole",
        episode_id=episode_id,
        env_id="workcell",
        world_id=str(scene_spec.workcell_id),
        robot_id="workcell",
        source_domain=SourceDomain.SIM_ROLLOUT,
        seed=int(seed + episode_idx),
        run_id=str(scenario_id),
        timestamp=runtime_timestamp,
        episode_metrics=runtime_metrics,
        reward_components={"scalar_reward": float(metrics.get("reward", 0.0))},
        telemetry={
            "trust_score": float(metrics.get("quality_score", 0.0)),
            "uncertainty": float(max(0.0, 1.0 - float(metrics.get("quality_score", 0.0)))),
            "scene_tracks_backend": str(scene_tracks_truth.get("scene_tracks_backend", "")),
            "semantic_density_score": float(scene_tracks_summary.get("semantic_density_score", 0.0) or 0.0),
            "grounded_track_object_count": int(world_model.topology.get("grounded_track_object_count", 0) or 0),
            "semantic_tags": list(belief_state.semantic_tags),
        },
        context={
            "grounding_mode": grounding_mode,
            "backend_policy": backend_policy,
            "backend_selected": backend_selected,
            "sam3d_grounded_data_requires_gpu": True,
            "sam3d_grounded_data_requires_real_backend": True,
        },
    )
    contract_profile = load_contract_profile("balanced_contract")
    runtime_builder = ObjectiveRuntimeBuilder()
    objective_tensor = runtime_builder.build(runtime_record)
    constraint_set = ConstraintSet.from_runtime(
        hard_constraints={"throughput": {"min": 0.0}},
        soft_constraints={"energy": {"max": float(runtime_metrics.get("energy_wh_per_unit", 8.0) or 8.0)}},
        geometry_hints={"source": "bootstrap_semantic_workcell_loop", "camera": camera},
        trust_metadata={"trust_score": float(metrics.get("quality_score", 0.0))},
    )
    constraint_flags = constraint_set.flag_observations(runtime_metrics)
    econ_tensor = ObjectiveEconFunctor(base_price_per_unit=3.0).map(
        objective_tensor,
        constraint_flags=constraint_flags,
        uncertainty=float(max(0.0, 1.0 - float(metrics.get("quality_score", 0.0)))),
        context={"run_id": scenario_id, "episode_id": episode_id, "source_domain": SourceDomain.SIM_ROLLOUT.value},
    )
    runtime_packet = runtime_packet_from_record(
        record=runtime_record,
        contract_id=f"contract.balanced_contract.workcell_bootstrap.{scene_spec.workcell_id}",
        objective_profile_id=contract_profile.profile_id,
        objective_tensor=objective_tensor,
        econ_tensor=econ_tensor,
        constraint_set=constraint_set,
        observation_schema=observation_schema,
        action_schema=action_schema,
        semantic_evidence={
            "semantic_tags": list(belief_state.semantic_tags),
            "belief_state_id": belief_state.belief_id,
            "scene_tracks_backend": str(scene_tracks_truth.get("scene_tracks_backend", "")),
            "scene_tracks_non_stub": bool(scene_tracks_truth.get("scene_tracks_non_stub", False)),
            "scene_tracks_training_eligible": bool(scene_tracks_truth.get("scene_tracks_training_eligible", False)),
            "semantic_grounding_non_heuristic": bool(
                scene_tracks_truth.get("semantic_grounding_non_heuristic", False)
            ),
            "grounded_track_object_count": int(world_model.topology.get("grounded_track_object_count", 0) or 0),
            "sam3d_grounded_data_required": True,
            "grounded_data_ready": grounded_ready,
        },
        uncertainty={
            "runtime": float(max(0.0, 1.0 - float(metrics.get("quality_score", 0.0)))),
            "coverage_gap": float(max(0.0, 1.0 - float(scene_tracks_summary.get("class_label_coverage", 0.0) or 0.0))),
        },
        provenance={
            "source_adapter": "bootstrap_semantic_workcell_loop",
            "grounding_mode": grounding_mode,
            "backend_policy": backend_policy,
            "backend_selected": backend_selected,
        },
        metadata={
            "grounded_data_ready": grounded_ready,
            "sam3d_gpu_required": True,
        },
        semantic_schema_id="bootstrap_workcell_semantic_evidence_v1",
    )
    trace_event = RuntimeEvent.from_components(
        run_id=scenario_id,
        episode_id=episode_id,
        timestamp=runtime_timestamp,
        event_kind="bootstrap_rollout_trace_recorded",
        sequence_idx=0,
        scope={"scope_kind": "episode"},
        runtime_packet_id=runtime_packet.packet_id,
        contract_id=runtime_packet.contract.contract_id,
        artifact_refs={"runtime_packet": f"{episode_id}_runtime_packet_v1.json"},
        provenance={"component": "bootstrap_semantic_workcell_loop"},
        metadata={
            "step_count": int(len(actions)),
            "reward_total": float(metrics.get("reward", 0.0)),
            "scene_tracks_backend": backend_selected,
        },
    )
    grounded_reasons = [f"scene_tracks_backend:{backend_selected}"]
    if not grounded_ready:
        grounded_reasons.append("sam3d_gpu_required_for_grounded_data")
        if backend_selected != "real":
            grounded_reasons.append("real_sam3d_backend_missing")
        if not bool(scene_tracks_truth.get("scene_tracks_training_eligible", False)):
            grounded_reasons.append("scene_tracks_training_ineligible")
    grounding_event = RuntimeEvent.from_components(
        run_id=scenario_id,
        episode_id=episode_id,
        timestamp=_episode_timestamp(seed, episode_idx, step_idx=1),
        event_kind="grounded_data_lane_classified",
        sequence_idx=1,
        scope={"scope_kind": "episode"},
        runtime_packet_id=runtime_packet.packet_id,
        contract_id=runtime_packet.contract.contract_id,
        artifact_refs={"event_spine": f"{episode_id}_event_spine_v1.json"},
        provenance={"component": "bootstrap_semantic_workcell_loop"},
        metadata={
            "grounded_data_ready": grounded_ready,
            "scene_tracks_backend": backend_selected,
            "requires_real_sam3d": True,
            "requires_gpu": True,
        },
    )
    trace_decision = DecisionLedgerEntry.from_components(
        run_id=scenario_id,
        episode_id=episode_id,
        timestamp=runtime_timestamp,
        decision_kind="bootstrap_trace_recorded",
        outcome="trace_complete",
        sequence_idx=0,
        scope={"scope_kind": "episode"},
        reasons=["runtime_packet_event_spine_decision_ledger_emitted"],
        source_event_ids=[trace_event.event_id],
        runtime_packet_id=runtime_packet.packet_id,
        contract_id=runtime_packet.contract.contract_id,
        artifact_refs={"decision_ledger": f"{episode_id}_decision_ledger_v1.json"},
        provenance={"component": "bootstrap_semantic_workcell_loop"},
        metadata={"grounded_data_ready": grounded_ready},
    )
    grounding_decision = DecisionLedgerEntry.from_components(
        run_id=scenario_id,
        episode_id=episode_id,
        timestamp=_episode_timestamp(seed, episode_idx, step_idx=1),
        decision_kind="grounded_data_eligibility",
        outcome="benchmark_candidate" if grounded_ready else "dev_only_passthrough",
        sequence_idx=1,
        scope={"scope_kind": "episode"},
        reasons=grounded_reasons,
        source_event_ids=[grounding_event.event_id],
        runtime_packet_id=runtime_packet.packet_id,
        contract_id=runtime_packet.contract.contract_id,
        artifact_refs={"decision_ledger": f"{episode_id}_decision_ledger_v1.json"},
        provenance={"component": "bootstrap_semantic_workcell_loop"},
        metadata={
            "requires_real_sam3d": True,
            "requires_gpu": True,
            "grounded_data_ready": grounded_ready,
        },
    )
    runtime_packet_path = episode_dir / f"{episode_id}_runtime_packet_v1.json"
    event_spine_path = episode_dir / f"{episode_id}_event_spine_v1.json"
    decision_ledger_path = episode_dir / f"{episode_id}_decision_ledger_v1.json"
    runtime_packet_path.write_text(json.dumps(runtime_packet.to_dict(), indent=2), encoding="utf-8")
    event_spine_path.write_text(
        json.dumps(event_spine_sidecar_payload(run_id=scenario_id, events=[trace_event, grounding_event]), indent=2),
        encoding="utf-8",
    )
    decision_ledger_path.write_text(
        json.dumps(
            decision_ledger_sidecar_payload(
                run_id=scenario_id,
                decisions=[trace_decision, grounding_decision],
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "runtime_packet_path": str(runtime_packet_path),
        "event_spine_path": str(event_spine_path),
        "decision_ledger_path": str(decision_ledger_path),
        "runtime_packet_id": runtime_packet.packet_id,
        "event_refs": [trace_event.event_id, grounding_event.event_id],
        "decision_refs": [trace_decision.decision_id, grounding_decision.decision_id],
        "grounded_data_ready": grounded_ready,
        "grounded_data_mode": "real_sam3d" if grounded_ready else "dev_only_passthrough",
    }


def _run_single_workcell_episode(
    *,
    scenario_id: str,
    episode_idx: int,
    output_root: Path,
    steps: int,
    max_frames: int,
    seed: int,
    camera: str,
    grounding_mode: str,
    backend_policy: str,
) -> Dict[str, Any]:
    scene_spec = _build_scene_spec(seed, episode_idx)
    task = PegInHoleTask(peg_id="peg", hole_id="hole", tolerance_mm=2.0)
    capture_rgb = grounding_mode != "vector_proxy"
    config = WorkcellEnvConfig(
        physics_mode="MUJOCO" if _mujoco_available() else "SIMPLE",
        max_steps=steps,
        time_step_s=0.02,
        capture_rgb_frames=capture_rgb,
        render_width=128,
        render_height=128,
        render_fps=10,
        render_max_frames=max_frames,
    )
    env = WorkcellEnv(config=config, scene_spec=scene_spec, task=task, seed=seed + episode_idx)
    episode_id = f"{scenario_id}_ep_{episode_idx:03d}"
    env.reset(seed=seed + episode_idx, episode_id=episode_id)

    states: List[Dict[str, Any]] = []
    actions = _episode_actions(episode_idx, steps)
    for action in actions:
        env.step(action)
        states.append(env.physics_adapter.get_state())

    frames, depth_frames, seg_frames, camera_params = render_workcell_frames(
        scene_spec=scene_spec,
        states=states,
        camera_name=camera,
        width=config.render_width,
        height=config.render_height,
        max_frames=max_frames,
        seed=seed + episode_idx,
    )
    timestamps = [float(state.get("time_s", idx * config.time_step_s)) for idx, state in enumerate(states[: len(frames)])]
    seg_label_map = build_segmentation_label_map(scene_spec)
    scene_object_catalog = _build_scene_object_catalog(scene_spec)
    sensor_bundle = SensorBundleData(
        cameras=[camera],
        rgb={camera: np.asarray(frames, dtype=np.uint8)} if capture_rgb else {},
        depth={camera: np.asarray(depth_frames, dtype=np.float32)} if depth_frames else {},
        seg={camera: np.asarray(seg_frames, dtype=np.int32)} if seg_frames else {},
        intrinsics={
            camera: {
                "fx": float(camera_params.fx),
                "fy": float(camera_params.fy),
                "cx": float(camera_params.cx),
                "cy": float(camera_params.cy),
                "width": int(camera_params.width),
                "height": int(camera_params.height),
            }
        },
        extrinsics={camera: np.asarray(camera_params.world_from_cam, dtype=np.float32)},
        timestamps_s=timestamps,
        depth_unit="meters",
        segmentation_label_map={
            str(seg_id): {
                "object_id": object_id,
                "class_name": next(
                    (item["class_name"] for item in scene_object_catalog if item["object_id"] == object_id),
                    object_id,
                ),
            }
            for object_id, seg_id in seg_label_map.items()
        },
        scene_object_catalog=scene_object_catalog,
    )

    trajectory_data = {
        "scene_spec": scene_spec.to_dict(),
        "states": states,
        "actions": actions,
        "seed": seed + episode_idx,
    }
    metrics = {
        "reward": float(max(0.0, 5.0 - episode_idx)),
        "quality_score": 0.9,
        "safety_score": 0.95,
        "energy_wh_per_unit": 1.5 + episode_idx * 0.1,
    }

    rollout_root = output_root / "rollouts"
    start_rollout_capture(scenario_id, rollout_root)
    record_episode_rollout(
        scenario_id=scenario_id,
        episode_idx=episode_idx,
        metadata=EpisodeMetadata(
            episode_id=episode_id,
            task_id="peg_in_hole",
            robot_family="workcell",
            seed=seed + episode_idx,
            env_params={"config": config.to_dict(), "scene_spec": scene_spec.to_dict()},
        ),
        trajectory_data=trajectory_data,
        rgb_frames=np.asarray(frames, dtype=np.uint8) if capture_rgb else None,
        depth_frames=np.asarray(depth_frames, dtype=np.float32) if depth_frames else None,
        metrics=metrics,
        base_dir=rollout_root,
        sensor_bundle=sensor_bundle,
    )

    episode_dir = rollout_root / scenario_id / f"episode_{episode_idx:03d}"
    scene_tracks_result = run_scene_tracks(
        datapack_path=episode_dir,
        output_path=episode_dir,
        seed=seed + episode_idx,
        max_frames=max_frames,
        camera=camera,
        mode="rgb" if capture_rgb else "vector_proxy",
        min_quality=0.1,
        allow_low_quality=True,
        backend_policy=backend_policy,
    )

    belief_timestamp = datetime.now(timezone.utc).isoformat()
    scene_tracks_summary = dict(scene_tracks_result.frame_metadata.get("semantic_summary", {}) or {})
    belief_state = BeliefState(
        belief_id=f"belief_{episode_id}",
        episode_id=episode_id,
        timestamp=belief_timestamp,
        semantic_tags=sorted(
            {
                "peg",
                "hole",
                "workcell",
                "affordance:align",
                "affordance:insert",
                *[str(tag) for tag in scene_tracks_result.frame_metadata.get("semantic_tags", []) or []],
            }
        ),
        state_vector={
            "geometry_quality": float(scene_tracks_result.scene_ir_quality),
            "semantic_quality": float(scene_tracks_result.quality.quality_score),
            "coverage": float(scene_tracks_summary.get("class_label_coverage", 0.0) or 0.0),
        },
        uncertainty={
            "epistemic": float(max(0.0, 1.0 - scene_tracks_result.quality.quality_score)),
            "coverage_gap": float(max(0.0, 1.0 - scene_tracks_summary.get("class_label_coverage", 0.0))),
        },
        artifact_refs={
            "scene_tracks_path": str(scene_tracks_result.scene_tracks_path),
            "datapack_path": str(episode_dir),
        },
        metadata={
            "task_family": "workcell_bootstrap",
            "grounding_mode": "scene_tracks_real_rgb" if capture_rgb else "scene_tracks_vector_proxy",
        },
    )
    backend_selected = scene_tracks_result.frame_metadata.get("runner", {}).get("run_config", {}).get(
        "backend_selected",
        "",
    )
    world_model_builder = SemanticWorldModelBuilder()
    world_model = world_model_builder.build_from_runtime_fusion(
        episode_id=episode_id,
        task_id="peg_in_hole",
        objective_preset="balanced_contract",
        belief_state=belief_state,
        semantic_tags=belief_state.semantic_tags,
        scene_tracks_payload=dict(np.load(scene_tracks_result.scene_tracks_path, allow_pickle=False)),
        artifact_refs={
            "scene_tracks_path": str(scene_tracks_result.scene_tracks_path),
            "datapack_path": str(episode_dir),
        },
        metadata={
            "camera": camera,
            "backend_policy": backend_policy,
            "backend_selected": backend_selected,
        },
    )
    backbone = SemanticRuntimeBackbone()
    scene_tracks_truth = normalize_scene_tracks_truth(
        backend=scene_tracks_result.adapter_status.get("overall_mode", ""),
        explicit_non_stub=bool(scene_tracks_result.adapter_status.get("overall_mode") == "real"),
        semantic_grounding_ready=bool(scene_tracks_summary.get("grounding_ready", False)),
        training_eligible=bool(
            scene_tracks_result.frame_metadata.get("execution_preconditions", {}).get("ready", False)
        ),
        explicit_non_heuristic=bool(scene_tracks_result.adapter_status.get("overall_mode") == "real"),
    )
    backbone_result = backbone.build(
        task_id="peg_in_hole",
        objective_preset="balanced_contract",
        semantic_world_model=world_model,
        runtime_metrics={
            "scene_tracks_quality": float(scene_tracks_result.quality.quality_score),
            "scene_ir_quality": float(scene_tracks_result.scene_ir_quality),
            "expected_delta_mpl": 0.15,
            "expected_delta_error": -0.05,
            "expected_delta_energy_Wh": -0.02,
        },
        metadata={
            "scene_tracks_backend": backend_selected,
            "scene_tracks_non_stub": bool(scene_tracks_truth.get("scene_tracks_non_stub", False)),
        },
        backends=[str(scene_tracks_result.adapter_status.get("overall_mode", ""))],
    )
    sidecar_paths = backbone.write_sidecars(
        output_dir=episode_dir,
        file_stem=episode_id,
        result=backbone_result,
    )
    trace_sidecars = _write_bootstrap_trace_sidecars(
        episode_dir=episode_dir,
        scenario_id=scenario_id,
        episode_id=episode_id,
        seed=seed,
        episode_idx=episode_idx,
        scene_spec=scene_spec,
        states=states,
        actions=actions,
        camera=camera,
        time_step_s=float(config.time_step_s),
        metrics=metrics,
        grounding_mode=grounding_mode,
        backend_policy=backend_policy,
        backend_selected=backend_selected,
        scene_tracks_truth=scene_tracks_truth,
        scene_tracks_summary=scene_tracks_summary,
        belief_state=belief_state,
        world_model=world_model,
    )

    metadata_path = episode_dir / "metadata.json"
    metadata_payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata_payload["scene_tracks_non_stub"] = bool(scene_tracks_truth.get("scene_tracks_non_stub", False))
    metadata_payload["semantic_memory_grounded"] = bool(
        world_model.topology.get("grounded_track_object_count", 0) > 0
    )
    metadata_payload["semantic_grounding_non_heuristic"] = bool(
        scene_tracks_truth.get("semantic_grounding_non_heuristic", False)
    )
    metadata_payload["semantic_world_model_path"] = sidecar_paths["semantic_world_model_path"]
    metadata_payload["semantic_snapshot_path"] = sidecar_paths["semantic_snapshot_path"]
    metadata_payload["orchestrator_advisory_path"] = sidecar_paths["orchestrator_advisory_path"]
    metadata_payload["scene_tracks_backend"] = backend_selected
    metadata_payload["scene_tracks_training_eligible"] = bool(
        scene_tracks_truth.get("scene_tracks_training_eligible", False)
    )
    metadata_payload["runtime_packet_path"] = trace_sidecars["runtime_packet_path"]
    metadata_payload["event_spine_path"] = trace_sidecars["event_spine_path"]
    metadata_payload["decision_ledger_path"] = trace_sidecars["decision_ledger_path"]
    metadata_payload["runtime_packet_id"] = trace_sidecars["runtime_packet_id"]
    metadata_payload["event_refs"] = list(trace_sidecars["event_refs"])
    metadata_payload["decision_refs"] = list(trace_sidecars["decision_refs"])
    metadata_payload["grounded_data_ready"] = bool(trace_sidecars["grounded_data_ready"])
    metadata_payload["grounded_data_mode"] = str(trace_sidecars["grounded_data_mode"])
    metadata_payload["grounded_data_requirements"] = {
        "requires_real_sam3d": True,
        "requires_gpu": True,
        "passthrough_dev_only": True,
    }
    metadata_payload["future_training_signals"] = {
        "scene_tracks_non_stub": bool(scene_tracks_truth.get("scene_tracks_non_stub", False)),
        "semantic_memory_grounded": bool(world_model.topology.get("grounded_track_object_count", 0) > 0),
        "semantic_grounding_non_heuristic": bool(
            scene_tracks_truth.get("semantic_grounding_non_heuristic", False)
        ),
        "benchmark_eligible": False,
    }
    metadata_path.write_text(json.dumps(metadata_payload, indent=2), encoding="utf-8")

    return {
        "episode_id": episode_id,
        "episode_dir": str(episode_dir),
        "scene_tracks_path": str(scene_tracks_result.scene_tracks_path),
        "scene_tracks_quality": float(scene_tracks_result.quality.quality_score),
        "scene_ir_quality": float(scene_tracks_result.scene_ir_quality),
        "semantic_world_model_path": sidecar_paths["semantic_world_model_path"],
        "backend_selected": backend_selected,
        "grounded_track_object_count": int(world_model.topology.get("grounded_track_object_count", 0)),
        "semantic_density_score": float(scene_tracks_summary.get("semantic_density_score", 0.0) or 0.0),
        "runtime_packet_ref": trace_sidecars["runtime_packet_path"],
        "event_spine_ref": trace_sidecars["event_spine_path"],
        "decision_ledger_ref": trace_sidecars["decision_ledger_path"],
        "trace_ready": True,
        "grounded_data_ready": bool(trace_sidecars["grounded_data_ready"]),
    }


def run_semantic_workcell_bootstrap(
    *,
    output_root: Path,
    episodes: int,
    steps: int,
    max_frames: int,
    seed: int,
    camera: str,
    grounding_mode: str,
    backend_policy: str,
    sim_limit: int,
    diffusion_limit: int,
) -> Dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    scenario_id = f"semantic_workcell_{seed}"
    episode_summaries = [
        _run_single_workcell_episode(
            scenario_id=scenario_id,
            episode_idx=episode_idx,
            output_root=output_root,
            steps=steps,
            max_frames=max_frames,
            seed=seed,
            camera=camera,
            grounding_mode=grounding_mode,
            backend_policy=backend_policy,
        )
        for episode_idx in range(episodes)
    ]

    replay_dir = output_root / "replay_dataset"
    replay_bundle = ReplayDatasetBuilder().add_rollout_bundle(
        output_root / "rollouts",
        scenario_id=scenario_id,
        source_domain="workcell_semantic_bootstrap",
    ).write(replay_dir)
    loaded_bundle = load_replay_dataset(replay_dir)

    corpus = build_semantic_runtime_learning_corpus(loaded_bundle, max_counterfactuals=2)
    corpus_paths = write_semantic_runtime_learning_corpus(output_root / "semantic_runtime_corpus", corpus)
    coverage_result = run_coverage_loop(
        [row.to_dict() for row in corpus.rows],
        env_names=["workcell"],
        semantic_world_model=None,
        sim_agenda_limit=sim_limit,
        diffusion_limit=diffusion_limit,
        write_artifacts=False,
        artifact_dir=output_root / "coverage_artifacts",
    )

    coverage_paths = coverage_result.write_artifacts(output_root / "coverage_artifacts")
    execution_summary = dict(replay_bundle.manifest.metadata.get("execution_precondition_summary", {}) or {})
    grounded_episode_count = sum(1 for row in episode_summaries if row.get("grounded_data_ready", False))
    passthrough_episode_count = sum(1 for row in episode_summaries if row.get("backend_selected") == "passthrough")
    summary = {
        "scenario_id": scenario_id,
        "episodes": episode_summaries,
        "replay_summary": replay_bundle.to_summary(),
        "replay_execution_preconditions": execution_summary,
        "runtime_corpus_summary": corpus.summary,
        "runtime_corpus_paths": corpus_paths,
        "coverage_summary": coverage_result.coverage_summary,
        "coverage_artifact_paths": coverage_paths,
        "mujoco_available": _mujoco_available(),
        "backend_policy": backend_policy,
        "grounding_mode": grounding_mode,
        "trace_artifact_summary": {
            "runtime_packet_count": len(episode_summaries),
            "event_spine_count": len(episode_summaries),
            "decision_ledger_count": len(episode_summaries),
            "ready_episode_count": int(execution_summary.get("ready_count", 0)),
        },
        "grounded_data_summary": {
            "grounded_episode_count": grounded_episode_count,
            "passthrough_episode_count": passthrough_episode_count,
            "requires_real_sam3d": True,
            "requires_gpu": True,
            "local_passthrough_dev_only": True,
        },
    }
    summary_path = output_root / "bootstrap_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, help="Directory for datapacks, replay data, and coverage artifacts")
    parser.add_argument("--episodes", type=int, default=2, help="Number of workcell episodes to capture")
    parser.add_argument("--steps", type=int, default=5, help="Control steps per episode")
    parser.add_argument("--max-frames", type=int, default=5, help="Rendered frames per episode")
    parser.add_argument("--seed", type=int, default=42, help="Seed for deterministic scene/task variation")
    parser.add_argument("--camera", default="front", help="Camera name for SceneTracks and rendering")
    parser.add_argument(
        "--grounding-mode",
        choices=("auto", "real_rgb", "vector_proxy"),
        default="auto",
        help="Grounding mode: real_rgb forces rendered RGB; vector_proxy skips RGB and uses geometry/masks only",
    )
    parser.add_argument(
        "--backend-policy",
        choices=("auto", "real", "passthrough", "stub"),
        default="auto",
        help="SceneTracks backend selection policy",
    )
    parser.add_argument("--sim-limit", type=int, default=6, help="Coverage-loop simulation agenda cap")
    parser.add_argument("--diffusion-limit", type=int, default=6, help="Coverage-loop diffusion prompt cap")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    grounding_mode = args.grounding_mode
    if grounding_mode == "auto":
        grounding_mode = "real_rgb" if _mujoco_available() else "vector_proxy"
    summary = run_semantic_workcell_bootstrap(
        output_root=Path(args.output_root),
        episodes=max(args.episodes, 1),
        steps=max(args.steps, 1),
        max_frames=max(args.max_frames, 1),
        seed=args.seed,
        camera=args.camera,
        grounding_mode=grounding_mode,
        backend_policy=args.backend_policy,
        sim_limit=max(args.sim_limit, 1),
        diffusion_limit=max(args.diffusion_limit, 1),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
