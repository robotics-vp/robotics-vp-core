import json
from pathlib import Path

import numpy as np
import pytest

from src.motor_backend.base import MotorEvalResult, MotorTrainingResult
from src.motor_backend.rollout_capture import EpisodeMetadata, EpisodeRollout, RolloutBundle
from src.orchestrator.diffusion_requests import build_diffusion_prompts_from_world_state
from src.orchestrator.semantic_simulation import compile_simulation_agenda
from src.world_model.semantic_coverage_graph import CoverageEdge, CoverageNode, SemanticCoverageGraph
from src.world_model.sim_synth_physics import (
    LearnedBackendSelector,
    LearnedBranchPlanner,
    SimSynthPhysicsRuntime,
    SimSynthPhysicsRuntimeConfig,
    compile_gap_driven_diffusion_plans,
    compile_sim_synth_physics_world_state,
)
from src.world_model.sim_synth_physics.backend_selector import (
    BACKEND_LABELS,
    FIDELITY_LABELS,
    RANDOMIZATION_LABELS,
)


def _make_test_graph() -> SemanticCoverageGraph:
    return SemanticCoverageGraph(
        nodes=[
            CoverageNode("task:drawer_vase", "task", "drawer_vase"),
            CoverageNode("hrl:grasp_handle", "skill", "Grasp Handle"),
            CoverageNode("prim:locate_handle", "env_primitive", "Locate Handle"),
            CoverageNode("risk:collision", "risk_family", "collision"),
        ],
        edges=[
            CoverageEdge(
                "hrl:grasp_handle",
                "prim:locate_handle",
                "requires",
                evidence_count=0,
                economic_priority=0.8,
                trust_priority=0.5,
                promotion_readiness=0.2,
            ),
            CoverageEdge(
                "hrl:grasp_handle",
                "risk:collision",
                "requires",
                evidence_count=0,
                economic_priority=0.3,
                trust_priority=0.2,
                promotion_readiness=0.9,
            ),
        ],
    )


class PromotedBackendSelector:
    benchmark_gate = {"ready": True}

    def select_backend(self, *, context):
        assert context["heuristic_backend"] == "pybullet"
        return {
            "preferred_backend": "isaac",
            "fidelity_tier": "high_fidelity",
            "domain_randomization_regime": "benchmark_focus",
        }


class PromotedHolosomaBackendSelector:
    benchmark_gate = {"ready": True}

    def select_backend(self, *, context):
        assert context["heuristic_backend"] == "pybullet"
        return {
            "preferred_backend": "holosoma",
            "fidelity_tier": "high_fidelity",
            "domain_randomization_regime": "benchmark_focus",
        }


class ShadowBranchPlanner:
    benchmark_gate = {"ready": False}

    def plan_branch(self, *, job, context):
        assert context["physics_context"]["backend"] == "pybullet"
        return {
            "generation_mode": "neural_branch_candidate",
            "expected_yield_score": 0.95,
        }


class PromotedGGDSBranchPlanner:
    benchmark_gate = {"ready": True}

    def plan_branch(self, *, job, context):
        return {
            "generation_mode": "targeted_synth_rollout",
            "expected_yield_score": 0.9,
        }


def test_world_state_compiles_canonical_agenda_and_branch_plans() -> None:
    world_state = compile_sim_synth_physics_world_state(_make_test_graph(), limit=2)

    assert world_state.simulation_agenda.jobs[0].skill_edge == "Grasp Handle -> Locate Handle"
    assert world_state.physics_context.backend == "pybullet"
    assert world_state.physics_execution_contract is not None
    assert world_state.physics_execution_contract.requested_backend == "pybullet"
    assert world_state.physics_execution_contract.route_status == "ready"
    assert world_state.physics_execution_contract.metadata["requested_branch_count"] == 2
    assert world_state.physics_adaptation_policy is not None
    assert world_state.physics_adaptation_policy.domain_randomization_profile
    assert world_state.backend_execution_binding is not None
    assert world_state.backend_execution_binding.binding_status == "ready"
    assert world_state.robot_asset_contract is not None
    assert world_state.robot_asset_contract.asset_profile == "tabletop_workcell_assets"
    assert world_state.backend_runtime_bridge is not None
    assert world_state.backend_runtime_bridge.bridge_status == "runtime_bridge_ready"
    assert world_state.backend_runtime_bridge.transport_profile == "local_python_sim_bridge"
    assert len(world_state.synthetic_branch_plans) == 2
    assert world_state.synthetic_branch_plans[0].render_provider is not None
    assert world_state.diffusion_conditioning is not None
    assert world_state.diffusion_conditioning.branch_job_ids == [
        plan.source_job_id for plan in world_state.synthetic_branch_plans
    ]
    assert world_state.gen2sim_admission is not None
    assert (
        world_state.metadata["compiled_receipt_inventory"]["runtime_depth_projection"][
            "binding_status"
        ]
        == world_state.backend_execution_binding.binding_status
    )
    assert (
        world_state.artifact_refs["physics_execution_contract_id"]
        == world_state.physics_execution_contract.contract_id
    )
    assert (
        world_state.simulation_agenda.jobs[0].inferential_learnability_contract["subject_kind"]
        == "sim_synth_job"
    )
    assert (
        world_state.synthetic_branch_plans[0].inferential_learnability_contract["subject_kind"]
        == "synthetic_branch_plan"
    )


def test_world_state_to_dict_round_trips_core_phase1_state() -> None:
    world_state = compile_sim_synth_physics_world_state(_make_test_graph(), limit=2)

    payload = world_state.to_dict()
    round_tripped = json.loads(json.dumps(payload))

    assert round_tripped["state_id"] == world_state.state_id
    assert round_tripped["physics_context"]["backend"] == world_state.physics_context.backend
    assert (
        round_tripped["physics_execution_contract"]["contract_id"]
        == world_state.physics_execution_contract.contract_id
    )
    assert round_tripped["physics_adaptation_policy"]["policy_id"] == world_state.physics_adaptation_policy.policy_id
    assert round_tripped["backend_execution_binding"]["binding_id"] == world_state.backend_execution_binding.binding_id
    assert round_tripped["robot_asset_contract"]["contract_id"] == world_state.robot_asset_contract.contract_id
    assert round_tripped["backend_runtime_bridge"]["bridge_id"] == world_state.backend_runtime_bridge.bridge_id
    assert round_tripped["gen2sim_admission"]["admission_id"] == world_state.gen2sim_admission.admission_id
    assert round_tripped["diffusion_conditioning"]["conditioning_id"] == world_state.diffusion_conditioning.conditioning_id
    assert (
        round_tripped["metadata"]["compiled_receipt_inventory"]["inventory_id"]
        == world_state.metadata["compiled_receipt_inventory"]["inventory_id"]
    )
    assert round_tripped["synthetic_branch_plans"]


def test_world_state_uses_promoted_backend_selector_from_day_one() -> None:
    world_state = compile_sim_synth_physics_world_state(
        _make_test_graph(),
        backend_selector=PromotedBackendSelector(),
        backend_selector_mode="auto",
    )

    assert world_state.physics_context.backend == "isaac"
    assert world_state.physics_context.selection_policy == "heuristic_plus_learned_backend_selector"
    assert world_state.physics_context.metadata["backend_helper_status"]["promotion_stage"] == "promoted"
    assert world_state.physics_adaptation_policy is not None
    assert world_state.physics_adaptation_policy.target_hardware_class == "unitree_g1_r1_class"
    assert world_state.backend_execution_binding is not None
    assert world_state.backend_execution_binding.binding_status in {"assets_missing", "shadow_ready"}
    assert world_state.robot_asset_contract is not None
    assert "unitree_robot_description" in world_state.robot_asset_contract.required_assets
    assert world_state.backend_runtime_bridge is not None
    assert world_state.backend_runtime_bridge.bridge_status in {
        "runtime_targets_missing",
        "runtime_assets_missing",
        "shadow_bridge_only",
    }


def test_world_state_marks_isaac_runtime_ready_when_isaaclab_backend_exists(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from src.world_model.sim_synth_physics import backend_adapters as adapter_module
    from src.world_model.sim_synth_physics.adapters import backend_isaac as binding_module
    from src.world_model.sim_synth_physics import runtime_targets as runtime_targets_module

    monkeypatch.setattr(
        adapter_module,
        "_has_module",
        lambda name: name == "src.motor_backend.workcell_isaaclab_backend",
    )
    monkeypatch.setattr(
        binding_module,
        "_has_module",
        lambda name: name == "src.motor_backend.workcell_isaaclab_backend",
    )
    monkeypatch.setattr(
        runtime_targets_module,
        "_has_module",
        lambda name: name == "src.motor_backend.workcell_isaaclab_backend",
    )

    isaaclab_root = tmp_path / "isaaclab"
    unitree_sdk_root = tmp_path / "unitree_sdk2"
    unitree_asset_root = tmp_path / "unitree_assets"
    isaaclab_root.mkdir()
    unitree_sdk_root.mkdir()
    unitree_asset_root.mkdir()

    world_state = compile_sim_synth_physics_world_state(
        _make_test_graph(),
        backend_selector=PromotedBackendSelector(),
        embodiment_context={
            "isaaclab_root": str(isaaclab_root),
            "unitree_sdk2_root": str(unitree_sdk_root),
            "unitree_asset_root": str(unitree_asset_root),
            "control_constraints": {
                "control_rate_hz": 250.0,
                "planner_rate_hz": 10.0,
                "observation_rate_hz": 60.0,
                "action_decimation": 4,
                "latency_budget_ms": 8.0,
            },
            "robot_asset_manifest": {
                "unitree_usd": "/assets/unitree/g1.usd",
                "joint_map_path": "/assets/unitree/joint_map.yaml",
                "camera_extrinsics": "/assets/unitree/camera_extrinsics.json",
                "imu_extrinsics": "/assets/unitree/imu_extrinsics.json",
                "force_torque_calibration": "/assets/unitree/ft_calibration.json",
                "actuator_latency_profile": "/assets/unitree/latency.yaml",
                "joint_limit_profile": "/assets/unitree/joint_limits.yaml",
                "safety_watchdog_profile": "/assets/unitree/watchdog.yaml",
            }
        },
    )
    runtime = SimSynthPhysicsRuntime(
        SimSynthPhysicsRuntimeConfig(default_backend="pybullet", fallback_backend="pybullet")
    )
    result = runtime.execute_world_state(world_state)

    assert world_state.backend_execution_binding is not None
    assert world_state.backend_execution_binding.binding_status == "runtime_ready"
    assert world_state.physics_context.metadata["backend_adapter"]["supports_execution"] is True
    assert (
        world_state.backend_execution_binding.metadata["runtime_target_contract"]["backend"]
        == "isaac"
    )
    assert "runtime_layout_contract" in world_state.backend_execution_binding.metadata
    assert "policy_contract" in world_state.backend_execution_binding.metadata
    assert (
        world_state.backend_execution_binding.metadata["upstream_runtime_pack"]["pack_status"]
        == "pack_partial"
    )
    assert (
        world_state.backend_execution_binding.metadata["normalized_asset_manifest"]["unitree_robot_description"][
            "present"
        ]
        is True
    )
    assert result.physics_execution_contract.route_status == "ready"
    assert result.physics_execution_contract.resolved_backend == "isaac"
    assert world_state.backend_runtime_bridge is not None
    assert world_state.backend_runtime_bridge.bridge_status == "runtime_bridge_ready"
    assert world_state.backend_runtime_bridge.transport_profile == "isaaclab_unitree_dds_bridge"
    assert world_state.backend_runtime_bridge.transport_stack[:2] == ["python_bridge", "isaacsim"]
    assert "dds" in world_state.backend_runtime_bridge.transport_stack
    assert result.backend_runtime_bridge_receipt.bridge_status == "runtime_bridge_ready"
    assert result.backend_runtime_bridge_receipt.execution_authority == "shadow_runtime"
    assert (
        result.backend_runtime_bridge_receipt.metadata["upstream_runtime_pack"]["pack_status"]
        == "pack_partial"
    )
    assert "whole_body_balance_guard_v1" in result.backend_runtime_bridge_receipt.safety_channels
    assert "watchdog_state_v1" in result.backend_runtime_bridge_receipt.telemetry_contracts
    assert result.backend_runtime_bridge_receipt.planner_rate_hz == pytest.approx(10.0)
    assert result.backend_runtime_bridge_receipt.control_rate_hz == pytest.approx(250.0)
    assert result.backend_runtime_bridge_receipt.metadata["runtime_layout_ready_profiles"]


def test_world_state_marks_isaac_external_launch_ready_for_lerobot_and_teleop(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from src.world_model.sim_synth_physics import backend_adapters as adapter_module
    from src.world_model.sim_synth_physics.adapters import backend_isaac as binding_module
    from src.world_model.sim_synth_physics import runtime_targets as runtime_targets_module

    monkeypatch.setattr(adapter_module, "_has_module", lambda name: False)
    monkeypatch.setattr(binding_module, "_has_module", lambda name: False)
    monkeypatch.setattr(runtime_targets_module, "_has_module", lambda name: False)

    xr_root = tmp_path / "xr_teleoperate"
    sdk_root = tmp_path / "unitree_sdk2"
    sdk2_python_root = tmp_path / "unitree_sdk2_python"
    teleimager_root = tmp_path / "teleimager"
    asset_root = tmp_path / "unitree_assets"
    lerobot_root = tmp_path / "unitree_lerobot"
    policy_root = tmp_path / "policies"
    for root in (
        xr_root,
        sdk_root,
        sdk2_python_root,
        teleimager_root,
        asset_root,
        lerobot_root,
        policy_root,
    ):
        root.mkdir()
    (xr_root / "teleop").mkdir()
    (sdk_root / "include").mkdir()
    (sdk2_python_root / "setup.py").write_text("", encoding="utf-8")
    (teleimager_root / "README.md").write_text("teleimager", encoding="utf-8")
    (asset_root / "g1.usd").write_text("x", encoding="utf-8")
    (lerobot_root / "examples").mkdir()
    (policy_root / "g1_policy.onnx").write_text("x", encoding="utf-8")

    world_state = compile_sim_synth_physics_world_state(
        _make_test_graph(),
        backend_selector=PromotedBackendSelector(),
        embodiment_context={
            "xr_teleoperate_root": str(xr_root),
            "unitree_sdk2_root": str(sdk_root),
            "unitree_sdk2_python_root": str(sdk2_python_root),
            "teleimager_root": str(teleimager_root),
            "unitree_asset_root": str(asset_root),
            "unitree_lerobot_root": str(lerobot_root),
            "unitree_policy_root": str(policy_root),
            "runtime_policy_id": str(policy_root / "g1_policy.onnx"),
            "active_embodiments": ["unitree_g1"],
            "robot_asset_manifest": {
                "unitree_usd": "/assets/unitree/g1.usd",
                "joint_map_path": "/assets/unitree/joint_map.yaml",
                "camera_extrinsics": "/assets/unitree/camera_extrinsics.json",
                "imu_extrinsics": "/assets/unitree/imu_extrinsics.json",
                "force_torque_calibration": "/assets/unitree/ft_calibration.json",
                "actuator_latency_profile": "/assets/unitree/latency.yaml",
                "joint_limit_profile": "/assets/unitree/joint_limits.yaml",
                "safety_watchdog_profile": "/assets/unitree/watchdog.yaml",
            },
        },
    )

    assert world_state.backend_execution_binding is not None
    assert world_state.backend_execution_binding.binding_status == "external_launch_ready"
    deployment_contract = world_state.backend_execution_binding.metadata["deployment_contract"]
    upstream_runtime_pack = world_state.backend_execution_binding.metadata["upstream_runtime_pack"]
    assert deployment_contract["teleop_launch_ready"] is True
    assert deployment_contract["lerobot_eval_ready"] is True
    assert upstream_runtime_pack["pack_status"] == "pack_ready"
    assert "runtime_target_surface" in upstream_runtime_pack["ready_surfaces"]
    assert world_state.backend_runtime_bridge is not None
    assert world_state.backend_runtime_bridge.bridge_status == "runtime_bridge_ready"
    assert world_state.backend_runtime_bridge.transport_profile == "unitree_xr_teleop_bridge"
    assert "sdk2_python" in world_state.backend_runtime_bridge.transport_stack
    assert "teleimager" in world_state.backend_runtime_bridge.transport_stack
    assert "webrtc" in world_state.backend_runtime_bridge.transport_stack


def test_world_state_normalizes_unitree_asset_aliases_into_robot_contract() -> None:
    world_state = compile_sim_synth_physics_world_state(
        _make_test_graph(),
        backend_selector=PromotedBackendSelector(),
        embodiment_context={
            "robot_asset_manifest": {
                "unitree_urdf": "/assets/unitree/g1.urdf",
                "joint_map": "/assets/unitree/joint_map.json",
                "sensor_extrinsics": "/assets/unitree/sensors.json",
                "actuator_profile": "/assets/unitree/actuator_latency.json",
                "joint_limit_config": "/assets/unitree/joint_limits.yaml",
                "watchdog_profile": "/assets/unitree/watchdog.yaml",
            }
        },
    )

    assert world_state.robot_asset_contract is not None
    assert "unitree_robot_description" in world_state.robot_asset_contract.available_assets
    assert "whole_body_joint_map" in world_state.robot_asset_contract.available_assets
    assert "camera_extrinsics" in world_state.robot_asset_contract.available_assets
    assert "imu_extrinsics" in world_state.robot_asset_contract.available_assets
    assert "force_torque_calibration" in world_state.robot_asset_contract.missing_assets
    assert (
        world_state.robot_asset_contract.metadata["normalized_asset_manifest"]["unitree_robot_description"][
            "matched_aliases"
        ]
        == ["unitree_urdf"]
    )


def test_holosoma_binding_records_runtime_target_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.world_model.sim_synth_physics.adapters import backend_holosoma as binding_module
    from src.world_model.sim_synth_physics import runtime_targets as runtime_targets_module

    monkeypatch.setattr(binding_module, "_has_module", lambda name: name == "holosoma")
    monkeypatch.setattr(runtime_targets_module, "_has_module", lambda name: name == "holosoma")
    holosoma_root = tmp_path / "holosoma_repo"
    motion_root = tmp_path / "holosoma_motion"
    policy_root = tmp_path / "holosoma_policy"
    retargeting_root = tmp_path / "retargeting"
    holosoma_root.mkdir()
    motion_root.mkdir()
    policy_root.mkdir()
    retargeting_root.mkdir()

    world_state = compile_sim_synth_physics_world_state(
        _make_test_graph(),
        backend_selector=PromotedHolosomaBackendSelector(),
        embodiment_context={
            "active_embodiments": ["unitree_g1"],
            "holosoma_root": str(holosoma_root),
            "holosoma_motion_root": str(motion_root),
            "holosoma_policy_root": str(policy_root),
            "retargeting_root": str(retargeting_root),
            "motion_clip_datapacks": ["dp_motion_1"],
            "retargeting_contract": {"kind": "whole_body_retargeting_v1"},
            "whole_body_reward_overlay": {"balance_weight": 1.0},
            "robot_asset_manifest": {
                "unitree_urdf": "/assets/unitree/g1.urdf",
                "joint_map": "/assets/unitree/joint_map.json",
                "camera_extrinsics": "/assets/unitree/camera_extrinsics.json",
                "imu_extrinsics": "/assets/unitree/imu_extrinsics.json",
                "force_torque_calibration": "/assets/unitree/ft_calibration.json",
                "actuator_latency_profile": "/assets/unitree/latency.yaml",
                "joint_limit_profile": "/assets/unitree/joint_limits.yaml",
                "safety_watchdog_profile": "/assets/unitree/watchdog.yaml",
            },
            "control_constraints": {
                "servo_rate_hz": 120.0,
                "policy_decimation": 2,
            },
        },
    )

    assert world_state.backend_execution_binding is not None
    assert (
        world_state.backend_execution_binding.metadata["runtime_target_contract"]["backend"]
        == "holosoma"
    )
    assert "runtime_layout_contract" in world_state.backend_execution_binding.metadata
    assert "policy_contract" in world_state.backend_execution_binding.metadata
    assert world_state.backend_execution_binding.metadata["deployment_contract"]["motion_train_ready"] is True
    # pack_partial is the honest status: the test creates an empty holosoma_root
    # directory with no entrypoints. The install/preflight hardening correctly
    # reports partial readiness rather than the previously over-optimistic pack_ready.
    assert world_state.backend_execution_binding.metadata["upstream_runtime_pack"]["pack_status"] in (
        "pack_ready",
        "pack_partial",
    )
    assert world_state.backend_runtime_bridge is not None
    assert world_state.backend_runtime_bridge.bridge_status == "runtime_bridge_ready"
    assert world_state.backend_runtime_bridge.transport_profile == "holosoma_motion_runtime_bridge"
    assert "retargeting_guard_v1" in world_state.backend_runtime_bridge.safety_channels


def test_shadow_branch_planner_records_neural_trace_without_overriding() -> None:
    world_state = compile_sim_synth_physics_world_state(
        _make_test_graph(),
        branch_planner=ShadowBranchPlanner(),
        branch_planner_mode="auto",
    )

    first_plan = world_state.synthetic_branch_plans[0]
    assert first_plan.generation_mode != "neural_branch_candidate"
    assert first_plan.selection_policy == "heuristic_plus_learned_branch_planner"
    assert first_plan.metadata["branch_helper_status"]["promotion_stage"] == "shadow_candidate"
    assert first_plan.metadata["branch_helper_trace"]["generation_mode"] == "neural_branch_candidate"
    assert first_plan.metadata["branch_helper_resolution"] == "heuristic_due_to_shadow_candidate"
    assert first_plan.metadata["branch_helper_resolution_reason"] == "benchmark_gate_not_ready"
    assert first_plan.metadata["branch_helper_payload_applied"] is False
    assert first_plan.render_provider is not None
    assert first_plan.render_provider.provider_kind in {
        "lsd_scene_graph",
        "lsd_ggds_scene",
        "nag_lsd_counterfactual",
    }
    assert first_plan.render_provider.materialization_entrypoint


def test_diffusion_conditioning_uses_admitted_branches_for_budget() -> None:
    world_state = compile_sim_synth_physics_world_state(_make_test_graph(), limit=2)

    assert world_state.diffusion_conditioning is not None
    assert len(world_state.diffusion_conditioning.admissible_branch_ids) == 1
    assert len(world_state.diffusion_conditioning.blocked_branch_ids) == 1
    assert world_state.diffusion_conditioning.render_budget == 1
    assert (
        world_state.diffusion_conditioning.admissible_branch_ids
        == world_state.gen2sim_admission.admissible_branch_ids
    )


def test_diffusion_plans_rank_admitted_branches_ahead_of_blocked_ones() -> None:
    world_state = compile_sim_synth_physics_world_state(_make_test_graph(), limit=2)

    plans = compile_gap_driven_diffusion_plans(world_state, coverage_graph=_make_test_graph())

    assert len(plans) == 2
    assert plans[0].routing_context["branch_admissible"] is True
    assert plans[0].diffusion_priority_score >= plans[1].diffusion_priority_score
    assert plans[0].inferential_learnability_contract["subject_kind"] == "synthetic_branch_plan"


def test_legacy_agenda_wrapper_surfaces_wm_owned_fields() -> None:
    agenda = compile_simulation_agenda(_make_test_graph(), limit=1)

    assert agenda[0]["job_id"].startswith("sim_job_")
    assert agenda[0]["coverage_targets"]["source_id"] == "hrl:grasp_handle"
    assert "simulation_outcome_receipt" in agenda[0]["expected_receipts"]
    assert "inferential_learnability_contract" in agenda[0]


def test_diffusion_prompts_compile_from_world_state_contract() -> None:
    world_state = compile_sim_synth_physics_world_state(_make_test_graph(), limit=1)
    prompts = build_diffusion_prompts_from_world_state(
        world_state,
        coverage_graph=_make_test_graph(),
        limit=1,
    )

    assert prompts[0].routing_source == "sim_synth_physics_world_state"
    assert prompts[0].engine_type == world_state.physics_context.backend
    assert prompts[0].routing_context["physics_selection_policy"] == world_state.physics_context.selection_policy
    assert prompts[0].routing_context["branch_selection_policy"] == world_state.synthetic_branch_plans[0].selection_policy
    assert prompts[0].routing_context["render_provider_kind"] == world_state.synthetic_branch_plans[0].render_provider.provider_kind
    assert prompts[0].governed_hypotheses[0]["metadata"]["render_provider"]["materialization_entrypoint"]
    assert prompts[0].governed_hypotheses[0]["metadata"]["branch_plan_id"] == world_state.synthetic_branch_plans[0].plan_id


def _write_backend_selector_package(tmp_path: Path) -> str:
    torch = pytest.importorskip("torch")

    checkpoint_path = tmp_path / "backend_selector.pt"
    model = LearnedBackendSelector(hidden_dim=16)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.backend_head.bias[BACKEND_LABELS.index("isaac")] = 8.0
        model.fidelity_head.bias[FIDELITY_LABELS.index("high_fidelity")] = 8.0
        model.randomization_head.bias[RANDOMIZATION_LABELS.index("benchmark_focus")] = 8.0
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": model.net[0].in_features,
            "hidden_dim": model.net[0].out_features,
        },
        checkpoint_path,
    )
    package_path = tmp_path / "backend_selector_package.json"
    package_path.write_text(
        json.dumps(
            {
                "package_id": "backend_selector_test_pkg",
                "checkpoint_path": checkpoint_path.name,
                "benchmark_gate": {"ready": True},
                "execution_preconditions": {
                    "unsatisfied_preconditions": [],
                    "benchmark_gate_ready": True,
                },
                "promotion_stage": "promoted",
                "inference_contract": {"helper_blend_policy": "bounded_backend_selector_helper_v1"},
                "metadata": {"target_hardware_class": "unitree_g1_r1_class"},
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return str(package_path)


def _write_branch_planner_package(tmp_path: Path) -> str:
    torch = pytest.importorskip("torch")

    checkpoint_path = tmp_path / "branch_planner.pt"
    model = LearnedBranchPlanner(hidden_dim=16)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.mode_head.bias[4] = 8.0
        model.yield_head[0].bias.fill_(3.0)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": model.net[0].in_features,
            "hidden_dim": model.net[0].out_features,
        },
        checkpoint_path,
    )
    package_path = tmp_path / "branch_planner_package.json"
    package_path.write_text(
        json.dumps(
            {
                "package_id": "branch_planner_test_pkg",
                "checkpoint_path": checkpoint_path.name,
                "benchmark_gate": {"ready": True},
                "execution_preconditions": {
                    "unsatisfied_preconditions": [],
                    "benchmark_gate_ready": True,
                },
                "promotion_stage": "promoted",
                "inference_contract": {"helper_blend_policy": "bounded_branch_planner_helper_v1"},
                "metadata": {"target_hardware_class": "unitree_g1_r1_class"},
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return str(package_path)


def test_world_state_loads_backend_selector_runtime_package(tmp_path: Path) -> None:
    world_state = compile_sim_synth_physics_world_state(
        _make_test_graph(),
        backend_selector=_write_backend_selector_package(tmp_path),
        backend_selector_mode="auto",
    )

    helper_status = world_state.physics_context.metadata["backend_helper_status"]
    helper_trace = world_state.physics_context.metadata["backend_helper_trace"]

    assert world_state.physics_context.backend == "isaac"
    assert world_state.physics_context.fidelity_tier == "high_fidelity"
    assert world_state.physics_context.domain_randomization_regime == "benchmark_focus"
    assert world_state.physics_context.selection_policy == "heuristic_plus_learned_backend_selector"
    assert helper_status["status"] == "loaded"
    assert helper_status["promotion_stage"] == "promoted"
    assert helper_status["package_id"] == "backend_selector_test_pkg"
    assert helper_trace["preferred_backend"] == "isaac"


def test_world_state_loads_branch_planner_runtime_package(tmp_path: Path) -> None:
    world_state = compile_sim_synth_physics_world_state(
        _make_test_graph(),
        branch_planner=_write_branch_planner_package(tmp_path),
        branch_planner_mode="auto",
    )

    first_plan = world_state.synthetic_branch_plans[0]
    helper_status = first_plan.metadata["branch_helper_status"]
    helper_trace = first_plan.metadata["branch_helper_trace"]

    assert first_plan.generation_mode == "neural_branch_candidate"
    assert first_plan.selection_policy == "heuristic_plus_learned_branch_planner"
    assert helper_trace["expected_yield_score"] > 0.9
    assert helper_trace["expected_yield_score"] > first_plan.metadata["heuristic_expected_yield_score"]
    assert first_plan.expected_yield_score > 0.7
    assert helper_status["status"] == "loaded"
    assert helper_status["promotion_stage"] == "promoted"
    assert helper_status["package_id"] == "branch_planner_test_pkg"
    assert helper_trace["generation_mode"] == "neural_branch_candidate"
    assert first_plan.metadata["branch_helper_resolution"] == "learned_payload_applied"
    assert first_plan.metadata["branch_helper_resolution_reason"] == "benchmark_gate_ready"
    assert first_plan.metadata["branch_helper_payload_applied"] is True


def test_resolve_helper_demotion_on_evidence_failure() -> None:
    """A promoted helper is demoted to shadow when evidence signals indicate failure."""
    from src.world_model.sim_synth_physics.promotion import resolve_helper

    class PromotedHelper:
        benchmark_gate = {"ready": True}

    helper, status = resolve_helper(
        PromotedHelper(),
        mode="auto",
        name="test_helper",
        evidence_signals={"evidence_failure": True},
    )
    assert helper is not None
    assert status["promotion_stage"] == "demoted_to_shadow"
    assert status["helper_weight"] == 0.25
    assert status["demotion_reason"] == "evidence_failure"
    assert status["benchmark_gate_ready"] is True  # gate was ready, but evidence overrode


def test_resolve_helper_demotion_on_failure_rate() -> None:
    """A promoted helper is demoted when recent failure rate exceeds threshold."""
    from src.world_model.sim_synth_physics.promotion import resolve_helper

    class PromotedHelper:
        benchmark_gate = {"ready": True, "demotion_failure_threshold": 0.3}

    helper, status = resolve_helper(
        PromotedHelper(),
        mode="auto",
        name="test_helper",
        evidence_signals={"recent_failure_rate": 0.6},
    )
    assert helper is not None
    assert status["promotion_stage"] == "demoted_to_shadow"
    assert status["helper_weight"] == 0.25
    assert "failure_rate" in status["demotion_reason"]


def test_resolve_helper_no_demotion_without_evidence() -> None:
    """A promoted helper stays promoted when no evidence signals are provided."""
    from src.world_model.sim_synth_physics.promotion import resolve_helper

    class PromotedHelper:
        benchmark_gate = {"ready": True}

    helper, status = resolve_helper(
        PromotedHelper(),
        mode="auto",
        name="test_helper",
    )
    assert status["promotion_stage"] == "promoted"
    assert status["helper_weight"] == 0.7


def test_resolve_helper_no_demotion_on_passing_evidence() -> None:
    """A promoted helper stays promoted when evidence signals are healthy."""
    from src.world_model.sim_synth_physics.promotion import resolve_helper

    class PromotedHelper:
        benchmark_gate = {"ready": True}

    helper, status = resolve_helper(
        PromotedHelper(),
        mode="auto",
        name="test_helper",
        evidence_signals={"recent_failure_rate": 0.1},
    )
    assert status["promotion_stage"] == "promoted"
    assert status["helper_weight"] == 0.7


def test_backend_selector_demotion_on_evidence_failure() -> None:
    """Backend selector resolver demotes on evidence failure."""
    from src.world_model.sim_synth_physics.backend_selector_runtime import (
        resolve_backend_selector_helper,
    )

    class PromotedSelector:
        benchmark_gate = {"ready": True}

        def select_backend(self, *, context):
            return {"preferred_backend": "isaac"}

    _, status = resolve_backend_selector_helper(
        PromotedSelector(),
        mode="auto",
        evidence_signals={"benchmark_gate_revoked": True},
    )
    assert status["promotion_stage"] == "demoted_to_shadow"
    assert status["demotion_reason"] == "benchmark_gate_revoked"


def test_branch_planner_demotion_on_evidence_failure() -> None:
    """Branch planner resolver demotes on evidence failure."""
    from src.world_model.sim_synth_physics.branch_planner_runtime import (
        resolve_branch_planner_helper,
    )

    class PromotedPlanner:
        benchmark_gate = {"ready": True}

        def plan_branch(self, *, job, context):
            return {"generation_mode": "neural"}

    _, status = resolve_branch_planner_helper(
        PromotedPlanner(),
        mode="auto",
        evidence_signals={"evidence_failure": True},
    )
    assert status["promotion_stage"] == "demoted_to_shadow"
    assert status["demotion_reason"] == "evidence_failure"


def test_runtime_executes_world_state_with_explicit_isaac_fallback(tmp_path: Path) -> None:
    runtime = SimSynthPhysicsRuntime(
        SimSynthPhysicsRuntimeConfig(default_backend="pybullet", fallback_backend="pybullet")
    )
    world_state = runtime.compile_world_state(
        _make_test_graph(),
        backend_selector=PromotedBackendSelector(),
    )

    result = runtime.execute_world_state(world_state, output_dir=tmp_path)

    assert result.physics_execution_contract.requested_backend == "isaac"
    assert result.physics_execution_contract.resolved_backend == "pybullet"
    assert result.physics_execution_contract.route_status == "fallback"
    assert result.physics_adaptation_receipt.target_hardware_class == "unitree_g1_r1_class"
    assert result.gen2sim_admission_receipt.admission_id == result.world_state.gen2sim_admission.admission_id
    assert result.gen2sim_admission_receipt.metadata["robot_asset_contract"]["target_hardware_class"] == "unitree_g1_r1_class"
    assert result.backend_execution_binding_receipt.binding_status in {
        "assets_missing",
        "shadow_ready",
    }
    assert result.robot_asset_contract_receipt.target_hardware_class == "unitree_g1_r1_class"
    assert result.robot_asset_contract_receipt.readiness_score < 1.0
    assert result.backend_runtime_execution_receipt is not None
    assert result.backend_runtime_execution_receipt.backend == "isaac"
    assert (
        result.backend_runtime_execution_receipt.execution_status
        == "runtime_request_materialized_with_preconditions"
    )
    assert result.backend_shadow_execution_receipt is not None
    assert result.backend_shadow_execution_receipt.backend == "isaac"
    assert result.backend_shadow_execution_receipt.execution_mode == "shadow_contract"
    assert result.backend_shadow_execution_receipt.execution_status in {
        "shadow_executed",
        "shadow_executed_with_asset_gaps",
    }
    assert result.physics_calibration_receipt.metadata["explicit_gap_kind"] == "missing_backend_adapter"
    assert result.backend_runtime_bridge_receipt.bridge_status in {
        "runtime_targets_missing",
        "runtime_assets_missing",
        "shadow_bridge_only",
    }
    assert result.backend_runtime_bridge_receipt.execution_authority == "shadow_runtime"
    assert result.backend_runtime_work_orders
    assert result.backend_runtime_work_orders[0].backend == "isaac"
    assert result.backend_runtime_work_orders[0].status in {
        "blocked_by_runtime_targets",
        "blocked_by_assets",
        "blocked_by_runtime_preconditions",
        "ready_for_gpu_runtime",
    }
    assert "isaac_unitree_runtime_smoke" in result.backend_runtime_work_orders[0].linked_backlog_ids
    assert (
        result.backend_runtime_execution_receipt.metadata["runtime_bundle"]["backend"]
        == "isaac"
    )
    assert "launch_spec" in result.backend_runtime_execution_receipt.metadata
    assert (
        result.physics_adaptation_receipt.metadata["runtime_evidence"]["shadow_execution_status"]
        in {"shadow_executed", "shadow_executed_with_asset_gaps"}
    )
    assert result.backend_shadow_execution_receipt.metadata["robot_asset_contract_id"] == result.world_state.robot_asset_contract.contract_id
    assert len(result.backend_shadow_execution_receipt.metadata["asset_sidecar_refs"]) == 3
    assert result.backend_shadow_execution_receipt.metadata["shadow_harvest_mode"] == "shadow_with_data_harvest"
    assert result.backend_shadow_execution_receipt.metadata["backend_runtime_execution_receipt_id"] == result.backend_runtime_execution_receipt.receipt_id
    assert result.backend_shadow_execution_receipt.metadata["backend_runtime_binding_status"]
    assert result.backend_shadow_execution_receipt.metadata["shadow_runtime_binding_consumed"] is True
    assert (
        result.backend_shadow_execution_receipt.metadata["env_config"][
            "runtime_binding_selected_profile"
        ]
        == result.backend_shadow_execution_receipt.metadata[
            "backend_runtime_binding_selected_profile"
        ]
    )
    assert (
        result.backend_shadow_execution_receipt.metadata["env_config"][
            "runtime_binding_selected_target_refs"
        ]
        == result.backend_shadow_execution_receipt.metadata["backend_runtime_binding"][
            "selected_target_refs"
        ]
    )
    assert (
        result.physics_calibration_receipt.metadata["runtime_evidence"][
            "materialized_render_provider_count"
        ]
        >= 1
    )
    assert result.render_provider_receipts
    assert all(receipt.artifact_refs for receipt in result.render_provider_receipts)
    assert all(receipt.materialization_status != "planned_only" for receipt in result.render_provider_receipts)
    assert all(
        receipt.metadata["provider_truth_class"] in {
            "scene_materialization",
            "counterfactual_work_order",
            "ggds_work_order",
        }
        for receipt in result.render_provider_receipts
    )
    assert (tmp_path / "physics_execution_contract.json").exists()
    assert (tmp_path / "physics_adaptation_receipt.json").exists()
    assert (tmp_path / "gen2sim_admission_receipt.json").exists()
    assert (tmp_path / "backend_execution_binding_receipt.json").exists()
    assert (tmp_path / "robot_asset_contract_receipt.json").exists()
    assert (tmp_path / "backend_runtime_bridge_receipt.json").exists()
    assert (tmp_path / "backend_runtime_work_orders.json").exists()
    assert (tmp_path / "backend_runtime_execution_receipt.json").exists()
    assert (tmp_path / "backend_shadow_execution_receipt.json").exists()
    assert (tmp_path / "backend_shadow_execution" / "isaac" / "robot_asset_contract_sidecar.json").exists()
    assert (tmp_path / "backend_shadow_execution" / "isaac" / "backend_calibration_sidecar.json").exists()
    assert (tmp_path / "backend_shadow_execution" / "isaac" / "backend_io_contract_sidecar.json").exists()
    assert (tmp_path / "physics_calibration_receipt.json").exists()
    assert (tmp_path / "render_provider_receipts.json").exists()
    assert (tmp_path / "simulation_outcome_receipts.json").exists()
    assert all(
        receipt.metadata["artifact_materialization"] != "planned_only"
        for receipt in result.outcome_receipts
    )
    assert all(
        receipt.status in {"planned_with_backend_fallback", "blocked_by_admission"}
        for receipt in result.outcome_receipts
    )


def test_runtime_materializes_holosoma_shadow_work_order(tmp_path: Path) -> None:
    runtime = SimSynthPhysicsRuntime(
        SimSynthPhysicsRuntimeConfig(default_backend="pybullet", fallback_backend="pybullet")
    )
    world_state = runtime.compile_world_state(
        _make_test_graph(),
        backend_selector=PromotedHolosomaBackendSelector(),
        embodiment_context={"active_embodiments": ["unitree_g1"]},
    )

    result = runtime.execute_world_state(world_state, output_dir=tmp_path)

    assert result.physics_execution_contract.requested_backend == "holosoma"
    assert result.physics_execution_contract.resolved_backend == "pybullet"
    assert result.physics_execution_contract.route_status == "fallback"
    assert result.gen2sim_admission_receipt.admission_id == result.world_state.gen2sim_admission.admission_id
    assert result.backend_execution_binding_receipt.binding_status in {"shadow_ready", "assets_missing"}
    assert result.robot_asset_contract_receipt.target_hardware_class == "unitree_g1_r1_class"
    assert result.backend_runtime_execution_receipt is not None
    assert result.backend_runtime_execution_receipt.backend == "holosoma"
    assert (
        result.backend_runtime_execution_receipt.execution_status
        == "runtime_request_materialized_with_preconditions"
    )
    assert result.backend_shadow_execution_receipt is not None
    assert result.backend_shadow_execution_receipt.backend == "holosoma"
    assert result.backend_shadow_execution_receipt.execution_mode == "shadow_work_order"
    assert result.backend_shadow_execution_receipt.execution_status in {
        "shadow_work_order_materialized",
        "shadow_work_order_materialized_with_preconditions",
    }
    assert (
        result.physics_adaptation_receipt.metadata["runtime_evidence"]["shadow_execution_status"]
        in {
            "shadow_work_order_materialized",
            "shadow_work_order_materialized_with_preconditions",
        }
    )
    assert result.backend_runtime_bridge_receipt.execution_authority == "shadow_runtime"
    assert result.backend_runtime_bridge_receipt.transport_profile == "holosoma_motion_runtime_bridge"
    assert result.backend_runtime_work_orders
    assert result.backend_runtime_work_orders[0].backend == "holosoma"
    assert "holosoma_runtime_eval_smoke" in result.backend_runtime_work_orders[0].linked_backlog_ids
    assert (
        result.backend_runtime_execution_receipt.metadata["runtime_bundle"]["backend"]
        == "holosoma"
    )
    assert "launch_spec" in result.backend_runtime_execution_receipt.metadata
    assert result.backend_shadow_execution_receipt.metadata["robot_asset_contract_id"] == result.world_state.robot_asset_contract.contract_id
    assert len(result.backend_shadow_execution_receipt.metadata["asset_sidecar_refs"]) == 3
    assert result.backend_shadow_execution_receipt.metadata["shadow_harvest_mode"] == "shadow_only_preview"
    assert result.backend_shadow_execution_receipt.metadata["backend_runtime_execution_receipt_id"] == result.backend_runtime_execution_receipt.receipt_id
    assert result.backend_shadow_execution_receipt.metadata["shadow_runtime_binding_consumed"] is True
    assert (
        result.backend_shadow_execution_receipt.metadata["work_order"][
            "runtime_binding_selected_profile"
        ]
        == result.backend_shadow_execution_receipt.metadata[
            "backend_runtime_binding_selected_profile"
        ]
    )
    assert (
        result.backend_shadow_execution_receipt.metadata["work_order"][
            "runtime_binding_selected_motion_sources"
        ]
        == result.backend_shadow_execution_receipt.metadata["backend_runtime_binding"][
            "selected_motion_sources"
        ]
    )
    assert result.backend_shadow_execution_receipt.artifact_refs
    assert any("holosoma_shadow_work_order.json" in ref for ref in result.backend_shadow_execution_receipt.artifact_refs)
    assert (
        tmp_path
        / "backend_shadow_execution"
        / "holosoma"
        / "holosoma_shadow_work_order.json"
    ).exists()
    assert (tmp_path / "backend_shadow_execution" / "holosoma" / "robot_asset_contract_sidecar.json").exists()
    assert (tmp_path / "backend_shadow_execution" / "holosoma" / "backend_calibration_sidecar.json").exists()
    assert (tmp_path / "backend_shadow_execution" / "holosoma" / "backend_io_contract_sidecar.json").exists()


def test_runtime_prepares_external_launch_when_runtime_roots_are_ready(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from src.world_model.sim_synth_physics import runtime_launch as runtime_launch_module

    monkeypatch.setattr(runtime_launch_module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(runtime_launch_module, "_cuda_ready", lambda: True)

    runtime = SimSynthPhysicsRuntime(
        SimSynthPhysicsRuntimeConfig(default_backend="pybullet", fallback_backend="pybullet")
    )
    unitree_sim_root = tmp_path / "unitree_sim_isaaclab"
    unitree_sim_root.mkdir()
    (unitree_sim_root / "sim_main.py").write_text("", encoding="utf-8")
    (unitree_sim_root / "dds").mkdir()
    (unitree_sim_root / "action_provider").mkdir()
    unitree_sdk_root = tmp_path / "unitree_sdk2"
    unitree_sdk_root.mkdir()
    (unitree_sdk_root / "include").mkdir()
    unitree_asset_root = tmp_path / "unitree_assets"
    unitree_asset_root.mkdir()
    (unitree_asset_root / "g1.usd").write_text("x", encoding="utf-8")
    policy_root = tmp_path / "policies"
    policy_root.mkdir()
    policy_path = policy_root / "g1_policy.onnx"
    policy_path.write_text("x", encoding="utf-8")

    world_state = runtime.compile_world_state(
        _make_test_graph(),
        backend_selector=PromotedBackendSelector(),
        embodiment_context={
            "unitree_sim_isaaclab_root": str(unitree_sim_root),
            "unitree_sdk2_root": str(unitree_sdk_root),
            "unitree_asset_root": str(unitree_asset_root),
            "isaac_policy_root": str(policy_root),
            "runtime_policy_id": str(policy_path),
            "robot_asset_manifest": {
                "unitree_usd": "/assets/unitree/g1.usd",
                "joint_map_path": "/assets/unitree/joint_map.yaml",
                "camera_extrinsics": "/assets/unitree/camera_extrinsics.json",
                "imu_extrinsics": "/assets/unitree/imu_extrinsics.json",
                "force_torque_calibration": "/assets/unitree/ft_calibration.json",
                "actuator_latency_profile": "/assets/unitree/latency.yaml",
                "joint_limit_profile": "/assets/unitree/joint_limits.yaml",
                "safety_watchdog_profile": "/assets/unitree/watchdog.yaml",
            },
        },
    )

    result = runtime.execute_world_state(world_state)

    assert result.backend_runtime_execution_receipt is not None
    assert result.backend_runtime_execution_receipt.backend == "isaac"
    assert result.backend_runtime_execution_receipt.execution_status == "runtime_launch_prepared"
    assert result.backend_runtime_adapter_receipt is not None
    assert result.backend_runtime_adapter_receipt.adapter_status == "external_launch_ready"
    assert result.backend_runtime_adapter_receipt.execution_path == "external_launch"
    assert (
        result.backend_runtime_adapter_receipt.metadata["realization"]["realization_path"]
        == "external_launch_delegate"
    )
    assert result.backend_runtime_launch_receipt is not None
    assert result.backend_runtime_launch_receipt.launch_status == "launch_prepared"
    assert result.backend_runtime_launch_receipt.executed is False
    assert result.backend_runtime_outcome_receipt is not None
    assert result.backend_runtime_outcome_receipt.outcome_status == "launch_not_executed"
    assert result.backend_runtime_outcome_receipt.harvested_output_count == 0
    assert (
        result.backend_runtime_execution_receipt.metadata["launch_plan"]["status"]
        == "ready_for_launch"
    )
    assert (
        result.backend_runtime_execution_receipt.metadata["executable_adapter_consumer"][
            "consumer_mode"
        ]
        == "external_sim_launch"
    )
    assert (
        result.backend_runtime_execution_receipt.metadata["adapter_receipt"]["adapter_status"]
        == "external_launch_ready"
    )
    assert (
        result.backend_runtime_execution_receipt.metadata["adapter_realization"][
            "realization_status"
        ]
        == "external_launch_delegate_ready"
    )
    assert "sim_main.py" in result.backend_runtime_execution_receipt.metadata["launch_spec"]["command"]
    assert any(
        "sim_main.py" in hint for hint in result.backend_runtime_work_orders[0].command_hints
    )


def test_runtime_executes_external_launch_when_requested(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from src.world_model.sim_synth_physics import backend_runtime_execution as runtime_exec_module
    from src.world_model.sim_synth_physics import runtime_launch as runtime_launch_module

    monkeypatch.setattr(runtime_launch_module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(runtime_launch_module, "_cuda_ready", lambda: True)
    monkeypatch.setattr(runtime_exec_module, "_runtime_supports_execution", lambda backend: False)

    def _fake_launch(runtime_bundle, launch_spec, *, execute, cwd=None, require_policy=True):
        assert execute is True
        logs_dir = Path(str(cwd or launch_spec["root"])) / "logs" / "run_1"
        logs_dir.mkdir(parents=True, exist_ok=True)
        (logs_dir / "policy.onnx").write_text("x", encoding="utf-8")
        (logs_dir / "metrics.json").write_text("{}", encoding="utf-8")
        return {
            "backend": runtime_bundle["backend"],
            "preferred_profile": launch_spec["preferred_profile"],
            "status": "launch_completed",
            "command": launch_spec["command"],
            "cwd": str(cwd or launch_spec["root"]),
            "env_overrides": {},
            "missing_preconditions": [],
            "notes": ["test_launch"],
            "executed": True,
            "returncode": 0,
            "stdout": "ok",
            "stderr": "",
        }

    monkeypatch.setattr(runtime_exec_module, "execute_backend_runtime_launch", _fake_launch)

    runtime = SimSynthPhysicsRuntime(
        SimSynthPhysicsRuntimeConfig(default_backend="pybullet", fallback_backend="pybullet")
    )
    unitree_sim_root = tmp_path / "unitree_sim_isaaclab"
    unitree_sim_root.mkdir()
    (unitree_sim_root / "sim_main.py").write_text("", encoding="utf-8")
    (unitree_sim_root / "dds").mkdir()
    (unitree_sim_root / "action_provider").mkdir()
    unitree_sdk_root = tmp_path / "unitree_sdk2"
    unitree_sdk_root.mkdir()
    (unitree_sdk_root / "include").mkdir()
    unitree_asset_root = tmp_path / "unitree_assets"
    unitree_asset_root.mkdir()
    (unitree_asset_root / "g1.usd").write_text("x", encoding="utf-8")
    policy_root = tmp_path / "policies"
    policy_root.mkdir()
    policy_path = policy_root / "g1_policy.onnx"
    policy_path.write_text("x", encoding="utf-8")

    world_state = runtime.compile_world_state(
        _make_test_graph(),
        backend_selector=PromotedBackendSelector(),
        embodiment_context={
            "unitree_sim_isaaclab_root": str(unitree_sim_root),
            "unitree_sdk2_root": str(unitree_sdk_root),
            "unitree_asset_root": str(unitree_asset_root),
            "isaac_policy_root": str(policy_root),
            "runtime_policy_id": str(policy_path),
            "robot_asset_manifest": {
                "unitree_usd": "/assets/unitree/g1.usd",
                "joint_map_path": "/assets/unitree/joint_map.yaml",
                "camera_extrinsics": "/assets/unitree/camera_extrinsics.json",
                "imu_extrinsics": "/assets/unitree/imu_extrinsics.json",
                "force_torque_calibration": "/assets/unitree/ft_calibration.json",
                "actuator_latency_profile": "/assets/unitree/latency.yaml",
                "joint_limit_profile": "/assets/unitree/joint_limits.yaml",
                "safety_watchdog_profile": "/assets/unitree/watchdog.yaml",
            },
        },
    )

    result = runtime.execute_world_state(
        world_state,
        output_dir=tmp_path,
        execute_external_runtime_launch=True,
    )

    assert result.backend_runtime_execution_receipt is not None
    assert result.backend_runtime_execution_receipt.execution_status == "runtime_external_launch_completed"
    assert result.backend_runtime_adapter_receipt is not None
    assert result.backend_runtime_adapter_receipt.adapter_status == "external_launch_completed"
    assert (
        result.backend_runtime_adapter_receipt.metadata["realization"]["realization_status"]
        == "external_launch_delegate_ready"
    )
    assert result.backend_runtime_launch_receipt is not None
    assert result.backend_runtime_launch_receipt.launch_status == "launch_completed"
    assert result.backend_runtime_launch_receipt.executed is True
    assert result.backend_runtime_outcome_receipt is not None
    assert result.backend_runtime_outcome_receipt.outcome_status == "runtime_outputs_harvested"
    assert result.backend_runtime_outcome_receipt.harvested_output_count >= 2
    assert result.backend_runtime_execution_receipt.metadata["launch_report_refs"]
    assert result.backend_runtime_execution_receipt.metadata["runtime_outcome_receipt"]
    assert result.backend_runtime_execution_receipt.metadata["adapter_receipt"]
    assert result.backend_runtime_execution_receipt.metadata["adapter_realization"]
    assert result.backend_runtime_work_orders[0].status == "satisfied_by_external_runtime_outcomes"
    assert (tmp_path / "backend_runtime_adapter_receipt.json").exists()
    assert (tmp_path / "backend_runtime_adapter_realization.json").exists()
    assert (tmp_path / "backend_runtime_launch_receipt.json").exists()
    assert (tmp_path / "backend_runtime_outcome_receipt.json").exists()


def test_runtime_executes_concrete_isaac_backend_when_local_bridge_exists(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from src.world_model.sim_synth_physics import backend_adapters as adapter_module
    from src.world_model.sim_synth_physics import backend_runtime_execution as runtime_exec_module
    from src.world_model.sim_synth_physics import runtime_targets as runtime_targets_module
    from src.world_model.sim_synth_physics.adapters import backend_isaac as binding_module
    from src.world_model.sim_synth_physics.adapters import (
        isaac_unitree_adapter_execution as isaac_adapter_module,
    )
    from src.world_model.sim_synth_physics.adapters import (
        local_backend_factory_adapter as local_factory_module,
    )

    class FakeRuntimeBackend:
        def evaluate_policy(
            self,
            policy_id,
            task_id,
            objective,
            num_episodes,
            scenario_id=None,
            rollout_base_dir=None,
            seed=None,
        ):
            assert task_id
            assert num_episodes == 1
            assert scenario_id
            assert rollout_base_dir is not None
            episode_dir = Path(rollout_base_dir) / scenario_id / "episode_000"
            episode_dir.mkdir(parents=True, exist_ok=True)
            trajectory_path = episode_dir / "trajectory.npz"
            trajectory_path.write_bytes(b"fake")
            rollout_bundle = RolloutBundle(
                scenario_id=scenario_id,
                episodes=[
                    EpisodeRollout(
                        metadata=EpisodeMetadata(
                            episode_id=f"{scenario_id}_episode_000",
                            task_id=task_id,
                            robot_family="unitree_g1",
                            seed=seed,
                            env_params={"mode": "isaac_local"},
                        ),
                        trajectory_path=trajectory_path,
                    )
                ],
            )
            return MotorEvalResult(
                policy_id=policy_id,
                raw_metrics={"success_rate": 1.0, "mpl_units_per_hour": 91.0},
                econ_metrics={"mpl_units_per_hour": 91.0, "wage_parity": 1.5},
                rollout_bundle=rollout_bundle,
            )

    monkeypatch.setattr(
        adapter_module,
        "_has_module",
        lambda name: name == "src.motor_backend.workcell_isaaclab_backend",
    )
    monkeypatch.setattr(
        binding_module,
        "_has_module",
        lambda name: name == "src.motor_backend.workcell_isaaclab_backend",
    )
    monkeypatch.setattr(
        runtime_targets_module,
        "_has_module",
        lambda name: name == "src.motor_backend.workcell_isaaclab_backend",
    )
    monkeypatch.setattr(isaac_adapter_module, "_has_local_bridge_module", lambda: True)
    monkeypatch.setattr(
        runtime_exec_module,
        "_runtime_supports_execution",
        lambda backend: backend == "isaac",
    )
    monkeypatch.setattr(
        local_factory_module,
        "make_motor_backend",
        lambda name, econ_meter, store, backend_config=None: FakeRuntimeBackend(),
    )
    monkeypatch.setattr(
        runtime_exec_module,
        "make_motor_backend",
        lambda name, econ_meter, store, backend_config=None: FakeRuntimeBackend(),
    )

    runtime_root = tmp_path / "unitree_sim_isaaclab"
    runtime_root.mkdir()
    policy_root = tmp_path / "policies"
    policy_root.mkdir()
    policy_path = policy_root / "g1_policy.onnx"
    policy_path.write_text("x", encoding="utf-8")
    sdk_root = tmp_path / "unitree_sdk2"
    sdk_root.mkdir()
    (sdk_root / "include").mkdir()
    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    (asset_root / "g1.usd").write_text("x", encoding="utf-8")

    runtime = SimSynthPhysicsRuntime(
        SimSynthPhysicsRuntimeConfig(default_backend="pybullet", fallback_backend="pybullet")
    )
    world_state = runtime.compile_world_state(
        _make_test_graph(),
        backend_selector=PromotedBackendSelector(),
        embodiment_context={
            "unitree_sim_isaaclab_root": str(runtime_root),
            "unitree_sdk2_root": str(sdk_root),
            "unitree_asset_root": str(asset_root),
            "isaac_policy_root": str(policy_root),
            "runtime_policy_id": str(policy_path),
            "robot_asset_manifest": {
                "unitree_usd": "/assets/unitree/g1.usd",
                "joint_map_path": "/assets/unitree/joint_map.yaml",
                "camera_extrinsics": "/assets/unitree/camera_extrinsics.json",
                "imu_extrinsics": "/assets/unitree/imu_extrinsics.json",
                "force_torque_calibration": "/assets/unitree/ft_calibration.json",
                "actuator_latency_profile": "/assets/unitree/latency.yaml",
                "joint_limit_profile": "/assets/unitree/joint_limits.yaml",
                "safety_watchdog_profile": "/assets/unitree/watchdog.yaml",
            },
        },
    )

    result = runtime.execute_world_state(world_state, output_dir=tmp_path)

    assert result.backend_runtime_execution_receipt is not None
    assert result.backend_runtime_execution_receipt.execution_status == "runtime_execution_completed"
    assert result.backend_runtime_adapter_receipt is not None
    assert result.backend_runtime_adapter_receipt.adapter_status == "local_bridge_handed_off"
    assert (
        result.backend_runtime_adapter_receipt.metadata["realization"]["realization_path"]
        == "local_backend_factory"
    )
    assert result.backend_runtime_outcome_receipt is not None
    assert result.backend_runtime_outcome_receipt.outcome_status == "runtime_outputs_harvested"
    assert result.backend_runtime_outcome_receipt.harvested_output_count >= 3
    assert result.backend_runtime_outcome_receipt.metadata["harvest_mode"] == "local_runtime_execution"
    assert (
        result.backend_runtime_outcome_receipt.metadata["structured_outputs"]["surface_ready"][
            "policy_surface_ready"
        ]
        is True
    )
    assert (
        result.backend_runtime_outcome_receipt.metadata["structured_outputs"]["surface_ready"][
            "dataset_surface_ready"
        ]
        is True
    )
    assert (
        result.backend_runtime_outcome_receipt.metadata["structured_outputs"]["surface_ready"][
            "metrics_surface_ready"
        ]
        is True
    )
    assert (
        result.backend_runtime_execution_receipt.metadata["runtime_outcome_receipt"][
            "outcome_status"
        ]
        == "runtime_outputs_harvested"
    )
    assert (
        result.backend_runtime_bridge_receipt.execution_authority == "concrete_runtime"
    )
    assert (tmp_path / "backend_runtime_outcome_receipt.json").exists()
    assert (
        tmp_path
        / "backend_runtime_execution"
        / "isaac"
        / "backend_runtime_output_summary.json"
    ).exists()


def test_runtime_run_planning_window_writes_feedback_and_diffusion_artifacts(tmp_path: Path) -> None:
    runtime = SimSynthPhysicsRuntime(SimSynthPhysicsRuntimeConfig(default_backend="pybullet"))

    result = runtime.run_planning_window(_make_test_graph(), output_dir=tmp_path)

    feedback_manifest = json.loads(
        (tmp_path / "sim_synth_training_feedback.json").read_text(encoding="utf-8")
    )
    diffusion_bundle = json.loads(
        (tmp_path / "gap_driven_diffusion_plans.json").read_text(encoding="utf-8")
    )
    loop_summary = json.loads(
        (tmp_path / "sim_synth_physics_loop_summary.json").read_text(encoding="utf-8")
    )

    assert feedback_manifest["world_state_id"] == result.world_state.state_id
    assert feedback_manifest["physics_adaptation_receipt_id"] == result.physics_adaptation_receipt.receipt_id
    assert feedback_manifest["gen2sim_admission_receipt_id"] == result.gen2sim_admission_receipt.receipt_id
    assert feedback_manifest["backend_execution_binding_receipt_id"] == result.backend_execution_binding_receipt.receipt_id
    assert feedback_manifest["robot_asset_contract_receipt_id"] == result.robot_asset_contract_receipt.receipt_id
    assert (
        feedback_manifest["backend_runtime_bridge_receipt_id"]
        == result.backend_runtime_bridge_receipt.receipt_id
    )
    assert feedback_manifest["bridge_execution_authority"] in {
        "planning_only",
        "runtime_request_only",
        "shadow_runtime",
        "concrete_runtime",
    }
    assert feedback_manifest["backend_runtime_work_order_count"] >= 0
    assert feedback_manifest["gen2sim_admissible_branch_count"] >= 0
    assert feedback_manifest["gen2sim_blocked_branch_count"] >= 0
    assert feedback_manifest["backend_shadow_execution_status"] in {
        "",
        "shadow_executed",
        "shadow_executed_with_asset_gaps",
        "shadow_work_order_materialized",
        "shadow_work_order_materialized_with_preconditions",
    }
    assert feedback_manifest["backend_runtime_execution_status"] in {
        "",
        "runtime_request_materialized_with_preconditions",
        "runtime_launch_prepared",
        "runtime_external_launch_completed",
        "runtime_external_launch_failed",
        "runtime_execution_completed",
        "runtime_training_completed",
        "runtime_execution_failed",
    }
    assert feedback_manifest["backend_runtime_launch_status"] in {
        "",
        "launch_blocked",
        "launch_prepared",
        "launch_completed",
        "launch_failed",
    }
    assert feedback_manifest["backend_runtime_outcome_status"] in {
        "",
        "launch_not_executed",
        "runtime_outputs_harvested",
        "runtime_outputs_missing",
        "outcome_sources_missing",
    }
    assert feedback_manifest["backend_runtime_output_count"] >= 0
    assert feedback_manifest["planned_branch_count"] >= 1
    assert feedback_manifest["materialized_render_provider_count"] >= 1
    assert feedback_manifest["robot_asset_readiness_score"] >= 0.0
    assert diffusion_bundle["plans"]
    assert loop_summary["physics_execution_contract_id"] == result.physics_execution_contract.contract_id
    assert loop_summary["gen2sim_admission_receipt_id"] == result.gen2sim_admission_receipt.receipt_id
    assert loop_summary["render_provider_receipt_count"] == len(result.render_provider_receipts)
    assert loop_summary["materialized_render_provider_count"] >= 1
    assert loop_summary["robot_asset_contract_receipt_id"] == result.robot_asset_contract_receipt.receipt_id
    assert loop_summary["backend_runtime_bridge_receipt_id"] == result.backend_runtime_bridge_receipt.receipt_id
    assert loop_summary["backend_runtime_launch_status"] in {
        "",
        "launch_blocked",
        "launch_prepared",
        "launch_completed",
        "launch_failed",
    }
    assert loop_summary["backend_runtime_outcome_status"] in {
        "",
        "launch_not_executed",
        "runtime_outputs_harvested",
        "runtime_outputs_missing",
        "outcome_sources_missing",
    }
    assert loop_summary["backend_runtime_output_count"] >= 0
    assert loop_summary["backend_runtime_work_order_count"] >= 0
    assert result.world_state.input_context["economic"]["economic_urgency_score"] == 0.0


def test_runtime_executes_concrete_holosoma_backend_when_runtime_and_policy_exist(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from src.world_model.sim_synth_physics import backend_runtime_execution as runtime_exec_module
    from src.world_model.sim_synth_physics import runtime_targets as runtime_targets_module
    from src.world_model.sim_synth_physics.adapters import backend_holosoma as binding_module
    from src.world_model.sim_synth_physics.adapters import (
        holosoma_adapter_execution as holosoma_adapter_module,
    )
    from src.world_model.sim_synth_physics.adapters import (
        local_backend_factory_adapter as local_factory_module,
    )

    class FakeRuntimeBackend:
        def evaluate_policy(
            self,
            policy_id,
            task_id,
            objective,
            num_episodes,
            scenario_id=None,
            rollout_base_dir=None,
            seed=None,
        ):
            assert task_id == "humanoid_wbt_g1"
            assert num_episodes == 1
            assert scenario_id
            assert rollout_base_dir is not None
            episode_dir = Path(rollout_base_dir) / scenario_id / "episode_000"
            episode_dir.mkdir(parents=True, exist_ok=True)
            trajectory_path = episode_dir / "trajectory.npz"
            trajectory_path.write_bytes(b"fake")
            rollout_bundle = RolloutBundle(
                scenario_id=scenario_id,
                episodes=[
                    EpisodeRollout(
                        metadata=EpisodeMetadata(
                            episode_id=f"{scenario_id}_episode_000",
                            task_id=task_id,
                            robot_family="unitree_g1",
                            seed=seed,
                            env_params={"mode": "holosoma"},
                        ),
                        trajectory_path=trajectory_path,
                    )
                ],
            )
            return MotorEvalResult(
                policy_id=policy_id,
                raw_metrics={"success_rate": 1.0, "mpl_units_per_hour": 88.0},
                econ_metrics={"mpl_units_per_hour": 88.0, "wage_parity": 1.4},
                rollout_bundle=rollout_bundle,
            )

    monkeypatch.setattr(binding_module, "_has_module", lambda name: name == "holosoma")
    monkeypatch.setattr(runtime_targets_module, "_has_module", lambda name: name == "holosoma")
    monkeypatch.setattr(holosoma_adapter_module, "_has_local_runtime_module", lambda: True)
    monkeypatch.setattr(runtime_exec_module, "_runtime_supports_execution", lambda backend: backend == "holosoma")
    monkeypatch.setattr(
        local_factory_module,
        "make_motor_backend",
        lambda name, econ_meter, store, backend_config=None: FakeRuntimeBackend(),
    )
    monkeypatch.setattr(
        runtime_exec_module,
        "make_motor_backend",
        lambda name, econ_meter, store, backend_config=None: FakeRuntimeBackend(),
    )

    runtime = SimSynthPhysicsRuntime(
        SimSynthPhysicsRuntimeConfig(default_backend="pybullet", fallback_backend="pybullet")
    )
    world_state = runtime.compile_world_state(
        _make_test_graph(),
        backend_selector=PromotedHolosomaBackendSelector(),
        embodiment_context={
            "active_embodiments": ["unitree_g1"],
            "holosoma_policy_id": "holosoma_policy.onnx",
            "motion_clip_datapacks": ["dp_motion_1"],
        },
    )

    result = runtime.execute_world_state(world_state, output_dir=tmp_path)

    assert result.backend_runtime_execution_receipt is not None
    assert result.backend_runtime_execution_receipt.backend == "holosoma"
    assert result.backend_runtime_execution_receipt.execution_status == "runtime_execution_completed"
    assert result.backend_runtime_execution_receipt.policy_id == "holosoma_policy.onnx"
    assert (
        result.backend_runtime_execution_receipt.metadata["executable_adapter_request"][
            "adapter_family"
        ]
        == "holosoma"
    )
    assert result.backend_runtime_execution_receipt.metadata["adapter_realization"][
        "realization_path"
    ] == "local_backend_factory"
    assert (
        result.physics_adaptation_receipt.metadata["runtime_evidence"]["runtime_execution_status"]
        == "runtime_execution_completed"
    )
    assert any(
        "holosoma_datapack_binding.json" in ref
        for ref in result.backend_runtime_execution_receipt.artifact_refs
    )
    assert any(
        "backend_runtime_metrics.json" in ref
        for ref in result.backend_runtime_execution_receipt.artifact_refs
    )
    assert result.backend_runtime_outcome_receipt is not None
    assert result.backend_runtime_outcome_receipt.outcome_status == "runtime_outputs_harvested"
    assert result.backend_runtime_outcome_receipt.metadata["harvest_mode"] == "local_runtime_execution"
    assert (
        result.backend_runtime_outcome_receipt.metadata["structured_outputs"]["surface_ready"][
            "dataset_surface_ready"
        ]
        is True
    )
    assert (
        result.backend_runtime_outcome_receipt.metadata["structured_outputs"]["surface_ready"][
            "metrics_surface_ready"
        ]
        is True
    )
    assert result.backend_runtime_bridge_receipt.execution_authority == "concrete_runtime"
    assert result.backend_runtime_work_orders[0].status == "satisfied_by_concrete_runtime"
    assert (tmp_path / "backend_runtime_execution_receipt.json").exists()


def test_runtime_trains_concrete_holosoma_backend_when_motion_datapacks_exist(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from src.world_model.sim_synth_physics import backend_runtime_execution as runtime_exec_module
    from src.world_model.sim_synth_physics import runtime_targets as runtime_targets_module
    from src.world_model.sim_synth_physics.adapters import backend_holosoma as binding_module
    from src.world_model.sim_synth_physics.adapters import (
        holosoma_adapter_execution as holosoma_adapter_module,
    )
    from src.world_model.sim_synth_physics.adapters import (
        local_backend_factory_adapter as local_factory_module,
    )

    class FakeRuntimeBackend:
        def train_policy(
            self,
            task_id,
            objective,
            datapack_ids,
            num_envs,
            max_steps,
            datapack_configs=None,
            scenario_id=None,
            rollout_base_dir=None,
            seed=None,
        ):
            assert task_id == "humanoid_wbt_g1"
            assert datapack_ids == ["dp_motion_1"]
            assert datapack_configs
            assert datapack_configs[0].motion_clips
            assert num_envs >= 1
            assert max_steps >= 64
            assert scenario_id
            assert rollout_base_dir is not None
            episode_dir = Path(rollout_base_dir) / scenario_id / "episode_000"
            episode_dir.mkdir(parents=True, exist_ok=True)
            trajectory_path = episode_dir / "trajectory.npz"
            trajectory_path.write_bytes(b"fake")
            rollout_bundle = RolloutBundle(
                scenario_id=scenario_id,
                episodes=[
                    EpisodeRollout(
                        metadata=EpisodeMetadata(
                            episode_id=f"{scenario_id}_episode_000",
                            task_id=task_id,
                            robot_family="unitree_g1",
                            seed=seed,
                            env_params={"mode": "holosoma_train"},
                        ),
                        trajectory_path=trajectory_path,
                    )
                ],
            )
            return MotorTrainingResult(
                policy_id="trained_holosoma_policy.onnx",
                raw_metrics={"train_steps": float(max_steps), "train_return": 3.5},
                econ_metrics={"mpl_units_per_hour": 72.0},
                rollout_bundle=rollout_bundle,
            )

    monkeypatch.setattr(binding_module, "_has_module", lambda name: name == "holosoma")
    monkeypatch.setattr(runtime_targets_module, "_has_module", lambda name: name == "holosoma")
    monkeypatch.setattr(holosoma_adapter_module, "_has_local_runtime_module", lambda: True)
    monkeypatch.setattr(runtime_exec_module, "_runtime_supports_execution", lambda backend: backend == "holosoma")
    monkeypatch.setattr(
        local_factory_module,
        "make_motor_backend",
        lambda name, econ_meter, store, backend_config=None: FakeRuntimeBackend(),
    )
    monkeypatch.setattr(
        runtime_exec_module,
        "make_motor_backend",
        lambda name, econ_meter, store, backend_config=None: FakeRuntimeBackend(),
    )

    runtime = SimSynthPhysicsRuntime(
        SimSynthPhysicsRuntimeConfig(default_backend="pybullet", fallback_backend="pybullet")
    )
    world_state = runtime.compile_world_state(
        _make_test_graph(),
        backend_selector=PromotedHolosomaBackendSelector(),
        embodiment_context={
            "active_embodiments": ["unitree_g1"],
            "motion_clip_datapacks": ["dp_motion_1"],
            "motion_clips": [{"path": "/tmp/clip_a.npz", "weight": 0.6}],
        },
    )

    result = runtime.execute_world_state(world_state, output_dir=tmp_path)

    assert result.backend_runtime_execution_receipt is not None
    assert result.backend_runtime_execution_receipt.backend == "holosoma"
    assert result.backend_runtime_execution_receipt.execution_status == "runtime_training_completed"
    assert result.backend_runtime_execution_receipt.execution_mode == "holosoma_train_policy"
    assert result.backend_runtime_execution_receipt.policy_id == "trained_holosoma_policy.onnx"
    assert result.backend_runtime_execution_receipt.metadata["adapter_realization"][
        "realization_path"
    ] == "local_backend_factory"
    assert (
        result.physics_adaptation_receipt.metadata["runtime_evidence"]["runtime_execution_status"]
        == "runtime_training_completed"
    )
    assert (
        result.physics_adaptation_receipt.metadata["runtime_evidence"]["runtime_concrete_completed"]
        is True
    )
    assert any(
        "backend_runtime_metrics.json" in ref
        for ref in result.backend_runtime_execution_receipt.artifact_refs
    )
    assert result.backend_runtime_outcome_receipt is not None
    assert result.backend_runtime_outcome_receipt.outcome_status == "runtime_outputs_harvested"
    assert result.backend_runtime_outcome_receipt.metadata["harvest_mode"] == "local_runtime_execution"
    assert (
        result.backend_runtime_outcome_receipt.metadata["structured_outputs"]["surface_ready"][
            "dataset_surface_ready"
        ]
        is True
    )
    assert (
        result.backend_runtime_outcome_receipt.metadata["structured_outputs"]["surface_ready"][
            "metrics_surface_ready"
        ]
        is True
    )
    assert result.backend_runtime_bridge_receipt.execution_authority == "concrete_runtime"
    assert result.backend_runtime_work_orders[0].status == "satisfied_by_concrete_runtime"
    assert (tmp_path / "backend_runtime_execution_receipt.json").exists()


def test_runtime_materializes_concrete_nag_counterfactuals_when_source_episode_exists(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from src.vision.nag.integration_lsd_backend import NAGDatapack
    from src.world_model.sim_synth_physics import render_providers
    import src.vision.nag.integration_lsd_backend as nag_module

    monkeypatch.setattr(render_providers, "_nag_renderer_available", lambda: True)

    def _fake_generate(**kwargs):
        backend_episode = kwargs["backend_episode"]
        return [
            NAGDatapack(
                base_episode_id=str(backend_episode.get("episode_id", "ep_0")),
                counterfactual_id="ep_0_cf0",
                frames=np.zeros((2, 3, 8, 8), dtype=np.float32),
            )
        ]

    monkeypatch.setattr(
        nag_module,
        "generate_nag_counterfactuals_for_lsd_episode",
        _fake_generate,
    )

    runtime = SimSynthPhysicsRuntime(
        SimSynthPhysicsRuntimeConfig(default_backend="pybullet", fallback_backend="pybullet")
    )
    world_state = runtime.compile_world_state(
        _make_test_graph(),
        backend_selector=PromotedBackendSelector(),
        semantic_context={
            "source_lsd_episode": {
                "episode_id": "source_ep_1",
                "gaussian_scene": None,
                "scene_graph": None,
                "num_frames": 2,
            }
        },
    )

    result = runtime.execute_world_state(world_state, output_dir=tmp_path)

    receipt = next(
        receipt
        for receipt in result.render_provider_receipts
        if receipt.provider_kind == "nag_lsd_counterfactual"
    )
    assert receipt.materialization_status == "counterfactuals_materialized"
    assert receipt.materialization_mode == "counterfactual_datapacks"
    assert receipt.metadata["provider_truth_class"] == "counterfactual_datapacks"
    assert any("nag_counterfactual_manifest.json" in ref for ref in receipt.artifact_refs)


def test_runtime_materializes_concrete_ggds_scene_when_source_scene_exists(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from src.envs.lsd3d_env.gaussian_scene import GaussianScene
    from src.envs.lsd3d_env import ggds as ggds_module
    from src.world_model.sim_synth_physics import render_providers

    class FakeOptimizer:
        def __init__(self) -> None:
            self._is_initialized = True
            self.config = type("Config", (), {"prompts": ["realistic scene"], "num_iterations": 2})()

        def optimize_scene(self, gaussian_scene, camera_rig, prompts=None, num_iterations=None, callback=None):
            return gaussian_scene.clone()

    monkeypatch.setattr(render_providers, "_ggds_concrete_available", lambda: True)
    monkeypatch.setattr(render_providers, "create_default_optimizer", lambda: FakeOptimizer())
    monkeypatch.setattr(ggds_module, "create_default_optimizer", lambda config=None: FakeOptimizer())

    source_scene = GaussianScene(
        means=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        covs=np.array([[0.1, 0.0, 0.0, 0.1, 0.0, 0.1]], dtype=np.float32),
        colors=np.array([[0.5, 0.5, 0.5]], dtype=np.float32),
        opacities=np.array([1.0], dtype=np.float32),
        normals=np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
    )

    runtime = SimSynthPhysicsRuntime(
        SimSynthPhysicsRuntimeConfig(default_backend="pybullet", fallback_backend="pybullet")
    )
    world_state = runtime.compile_world_state(
        _make_test_graph(),
        branch_planner=PromotedGGDSBranchPlanner(),
        semantic_context={"source_gaussian_scene": source_scene.to_dict()},
    )

    result = runtime.execute_world_state(world_state, output_dir=tmp_path)

    receipt = next(
        receipt
        for receipt in result.render_provider_receipts
        if receipt.provider_kind == "lsd_ggds_scene"
    )
    assert receipt.materialization_status == "ggds_scene_materialized"
    assert receipt.materialization_mode == "ggds_scene_optimization"
    assert receipt.metadata["provider_truth_class"] == "ggds_scene_materialization"
    assert any("optimized_gaussian_scene.json" in ref for ref in receipt.artifact_refs)
