"""Phase 3.5 humanoid capacity and environment refit surfaces.

These artifacts close the local Phase 3.5 contract/refit layer only. They do
not run Unitree sim, hardware, providers, training, promotion, live policy
control, or reward-math mutation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

from src.world_model.humanoid_readiness.common import (
    denied_gate_map,
    load_json,
    load_jsonl,
    mapping,
    stable_id,
    strings,
    write_json,
    write_jsonl,
)

PHASE35_REFIT_REPORT_VERSION = "humanoid_phase35_refit_report_v1"
PHASE35_CAPACITY_BAND_VERSION = "humanoid_phase35_capacity_band_contract_v1"
PHASE35_SCHEMA_DELTA_VERSION = "humanoid_phase35_schema_delta_contract_v1"
PHASE35_ENV_TAXONOMY_VERSION = "humanoid_phase35_env_taxonomy_receipt_v1"
PHASE35_BENCHMARK_VERSION = "humanoid_phase35_benchmark_taxonomy_v1"

PHASE35_REMAINING_BLOCKERS = (
    "unitree_sim_assets_and_backend_runtime_missing",
    "live_streams_and_measured_control_timing_missing",
    "hardware_or_hil_evidence_missing",
    "trained_whole_body_models_missing",
    "promotion_grade_humanoid_benchmarks_missing",
)


def _bipedal_chassis_payload(payload: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    return mapping(payload)


@dataclass(frozen=True)
class HumanoidCapacityBandContract:
    """Placement, timing, and resource contract for a humanoid capacity band."""

    band_id: str
    band_name: str
    compute_placement: str
    control_rate_class: str
    intended_work: list[str] = field(default_factory=list)
    excluded_authority: list[str] = field(default_factory=list)
    battery_reserve_class: str = "unknown"
    thermal_headroom_class: str = "unknown"
    comms_qos_class: str = "unknown"
    degraded_mode_allowed: bool = False
    replay_training_awareness: list[str] = field(default_factory=list)
    authority_class: str = "phase35_capacity_contract_only"
    version: str = PHASE35_CAPACITY_BAND_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "band_id": self.band_id,
            "version": self.version,
            "band_name": self.band_name,
            "compute_placement": self.compute_placement,
            "control_rate_class": self.control_rate_class,
            "intended_work": list(self.intended_work),
            "excluded_authority": list(self.excluded_authority),
            "battery_reserve_class": self.battery_reserve_class,
            "thermal_headroom_class": self.thermal_headroom_class,
            "comms_qos_class": self.comms_qos_class,
            "degraded_mode_allowed": bool(self.degraded_mode_allowed),
            "replay_training_awareness": list(self.replay_training_awareness),
            "authority_class": self.authority_class,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "HumanoidCapacityBandContract":
        return cls(
            band_id=str(payload.get("band_id", "")),
            band_name=str(payload.get("band_name", "")),
            compute_placement=str(payload.get("compute_placement", "unknown")),
            control_rate_class=str(payload.get("control_rate_class", "unknown")),
            intended_work=strings(payload.get("intended_work")),
            excluded_authority=strings(payload.get("excluded_authority")),
            battery_reserve_class=str(
                payload.get("battery_reserve_class", "unknown")
            ),
            thermal_headroom_class=str(
                payload.get("thermal_headroom_class", "unknown")
            ),
            comms_qos_class=str(payload.get("comms_qos_class", "unknown")),
            degraded_mode_allowed=bool(payload.get("degraded_mode_allowed", False)),
            replay_training_awareness=strings(
                payload.get("replay_training_awareness")
            ),
            authority_class=str(
                payload.get("authority_class", "phase35_capacity_contract_only")
            ),
            version=str(payload.get("version", PHASE35_CAPACITY_BAND_VERSION)),
        )


@dataclass(frozen=True)
class HumanoidSchemaDeltaContract:
    """Observation/action schema delta for bipedal humanoid readiness."""

    delta_id: str
    schema_family: str
    surface_name: str
    posture_scope: str
    required_fields: list[str] = field(default_factory=list)
    schema_refs: dict[str, Any] = field(default_factory=dict)
    replay_training_awareness: list[str] = field(default_factory=list)
    promotion_posture: str = "planning_only"
    authority_class: str = "phase35_schema_delta_contract_only"
    version: str = PHASE35_SCHEMA_DELTA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "delta_id": self.delta_id,
            "version": self.version,
            "schema_family": self.schema_family,
            "surface_name": self.surface_name,
            "posture_scope": self.posture_scope,
            "required_fields": list(self.required_fields),
            "schema_refs": mapping(self.schema_refs),
            "replay_training_awareness": list(self.replay_training_awareness),
            "promotion_posture": self.promotion_posture,
            "authority_class": self.authority_class,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "HumanoidSchemaDeltaContract":
        return cls(
            delta_id=str(payload.get("delta_id", "")),
            schema_family=str(payload.get("schema_family", "")),
            surface_name=str(payload.get("surface_name", "")),
            posture_scope=str(payload.get("posture_scope", "unknown")),
            required_fields=strings(payload.get("required_fields")),
            schema_refs=mapping(payload.get("schema_refs")),
            replay_training_awareness=strings(
                payload.get("replay_training_awareness")
            ),
            promotion_posture=str(payload.get("promotion_posture", "planning_only")),
            authority_class=str(
                payload.get("authority_class", "phase35_schema_delta_contract_only")
            ),
            version=str(payload.get("version", PHASE35_SCHEMA_DELTA_VERSION)),
        )


@dataclass(frozen=True)
class HumanoidEnvTaxonomyReceipt:
    """Posture-aware environment taxonomy receipt."""

    receipt_id: str
    env_family: str
    posture_tag: str
    role: str
    promotion_limit: str
    required_artifacts: list[str] = field(default_factory=list)
    replay_export_posture: str = "planning_only"
    promotion_posture: str = "not_promotable"
    version: str = PHASE35_ENV_TAXONOMY_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "version": self.version,
            "env_family": self.env_family,
            "posture_tag": self.posture_tag,
            "role": self.role,
            "promotion_limit": self.promotion_limit,
            "required_artifacts": list(self.required_artifacts),
            "replay_export_posture": self.replay_export_posture,
            "promotion_posture": self.promotion_posture,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "HumanoidEnvTaxonomyReceipt":
        return cls(
            receipt_id=str(payload.get("receipt_id", "")),
            env_family=str(payload.get("env_family", "")),
            posture_tag=str(payload.get("posture_tag", "unknown")),
            role=str(payload.get("role", "")),
            promotion_limit=str(payload.get("promotion_limit", "")),
            required_artifacts=strings(payload.get("required_artifacts")),
            replay_export_posture=str(
                payload.get("replay_export_posture", "planning_only")
            ),
            promotion_posture=str(payload.get("promotion_posture", "not_promotable")),
            version=str(payload.get("version", PHASE35_ENV_TAXONOMY_VERSION)),
        )


@dataclass(frozen=True)
class HumanoidBenchmarkTarget:
    """Benchmark target taxonomy for future G1/R1-class evidence."""

    benchmark_id: str
    benchmark_class: str
    posture_tag: str
    current_status: str
    required_evidence: list[str] = field(default_factory=list)
    future_closure_evidence: list[str] = field(default_factory=list)
    promotion_posture: str = "blocked_until_evidence"
    version: str = PHASE35_BENCHMARK_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "version": self.version,
            "benchmark_class": self.benchmark_class,
            "posture_tag": self.posture_tag,
            "current_status": self.current_status,
            "required_evidence": list(self.required_evidence),
            "future_closure_evidence": list(self.future_closure_evidence),
            "promotion_posture": self.promotion_posture,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "HumanoidBenchmarkTarget":
        return cls(
            benchmark_id=str(payload.get("benchmark_id", "")),
            benchmark_class=str(payload.get("benchmark_class", "")),
            posture_tag=str(payload.get("posture_tag", "unknown")),
            current_status=str(payload.get("current_status", "missing")),
            required_evidence=strings(payload.get("required_evidence")),
            future_closure_evidence=strings(payload.get("future_closure_evidence")),
            promotion_posture=str(
                payload.get("promotion_posture", "blocked_until_evidence")
            ),
            version=str(payload.get("version", PHASE35_BENCHMARK_VERSION)),
        )


@dataclass(frozen=True)
class HumanoidPhase35RefitReport:
    """Top-level local Phase 3.5 refit report."""

    report_id: str
    status: str
    capacity_band_count: int
    schema_delta_count: int
    env_taxonomy_count: int
    benchmark_target_count: int
    bipedal_chassis_report_id: str = ""
    bipedal_chassis_joint_count: int = 0
    bipedal_chassis_frame_count: int = 0
    bipedal_chassis_joint_limit_envelope_count: int = 0
    bipedal_balance_receipt_count: int = 0
    canonical_bipedal_chassis_present: bool = False
    limb_frame_tree_present: bool = False
    joint_limit_envelope_present: bool = False
    whole_body_observation_schema_present: bool = False
    whole_body_action_schema_present: bool = False
    balance_envelope_present: bool = False
    bipedal_chassis_local_scaffold_complete: bool = False
    local_structural_refit_complete: bool = False
    ready_for_phase4_local_sweep: bool = False
    ready_for_unitree_runtime: bool = False
    ready_for_training: bool = False
    training_executed: bool = False
    weights_written: bool = False
    provider_executed: bool = False
    hardware_executed: bool = False
    unitree_sim_runtime_executed: bool = False
    live_policy_control: bool = False
    reward_math_mutation: bool = False
    promotion_eligible: bool = False
    denied_gates: dict[str, bool] = field(default_factory=denied_gate_map)
    remaining_blockers: list[str] = field(default_factory=list)
    artifact_refs: dict[str, Any] = field(default_factory=dict)
    version: str = PHASE35_REFIT_REPORT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "version": self.version,
            "status": self.status,
            "capacity_band_count": int(self.capacity_band_count),
            "schema_delta_count": int(self.schema_delta_count),
            "env_taxonomy_count": int(self.env_taxonomy_count),
            "benchmark_target_count": int(self.benchmark_target_count),
            "bipedal_chassis_report_id": self.bipedal_chassis_report_id,
            "bipedal_chassis_joint_count": int(self.bipedal_chassis_joint_count),
            "bipedal_chassis_frame_count": int(self.bipedal_chassis_frame_count),
            "bipedal_chassis_joint_limit_envelope_count": int(
                self.bipedal_chassis_joint_limit_envelope_count
            ),
            "bipedal_balance_receipt_count": int(
                self.bipedal_balance_receipt_count
            ),
            "canonical_bipedal_chassis_present": bool(
                self.canonical_bipedal_chassis_present
            ),
            "limb_frame_tree_present": bool(self.limb_frame_tree_present),
            "joint_limit_envelope_present": bool(
                self.joint_limit_envelope_present
            ),
            "whole_body_observation_schema_present": bool(
                self.whole_body_observation_schema_present
            ),
            "whole_body_action_schema_present": bool(
                self.whole_body_action_schema_present
            ),
            "balance_envelope_present": bool(self.balance_envelope_present),
            "bipedal_chassis_local_scaffold_complete": bool(
                self.bipedal_chassis_local_scaffold_complete
            ),
            "local_structural_refit_complete": bool(
                self.local_structural_refit_complete
            ),
            "ready_for_phase4_local_sweep": bool(self.ready_for_phase4_local_sweep),
            "ready_for_unitree_runtime": bool(self.ready_for_unitree_runtime),
            "ready_for_training": bool(self.ready_for_training),
            "training_executed": bool(self.training_executed),
            "weights_written": bool(self.weights_written),
            "provider_executed": bool(self.provider_executed),
            "hardware_executed": bool(self.hardware_executed),
            "unitree_sim_runtime_executed": bool(
                self.unitree_sim_runtime_executed
            ),
            "live_policy_control": bool(self.live_policy_control),
            "reward_math_mutation": bool(self.reward_math_mutation),
            "promotion_eligible": bool(self.promotion_eligible),
            "denied_gates": denied_gate_map(self.denied_gates),
            "remaining_blockers": list(self.remaining_blockers),
            "artifact_refs": mapping(self.artifact_refs),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "HumanoidPhase35RefitReport":
        return cls(
            report_id=str(payload.get("report_id", "")),
            status=str(payload.get("status", "blocked")),
            capacity_band_count=int(payload.get("capacity_band_count", 0) or 0),
            schema_delta_count=int(payload.get("schema_delta_count", 0) or 0),
            env_taxonomy_count=int(payload.get("env_taxonomy_count", 0) or 0),
            benchmark_target_count=int(
                payload.get("benchmark_target_count", 0) or 0
            ),
            bipedal_chassis_report_id=str(
                payload.get("bipedal_chassis_report_id", "")
            ),
            bipedal_chassis_joint_count=int(
                payload.get("bipedal_chassis_joint_count", 0) or 0
            ),
            bipedal_chassis_frame_count=int(
                payload.get("bipedal_chassis_frame_count", 0) or 0
            ),
            bipedal_chassis_joint_limit_envelope_count=int(
                payload.get("bipedal_chassis_joint_limit_envelope_count", 0) or 0
            ),
            bipedal_balance_receipt_count=int(
                payload.get("bipedal_balance_receipt_count", 0) or 0
            ),
            canonical_bipedal_chassis_present=bool(
                payload.get("canonical_bipedal_chassis_present", False)
            ),
            limb_frame_tree_present=bool(
                payload.get("limb_frame_tree_present", False)
            ),
            joint_limit_envelope_present=bool(
                payload.get("joint_limit_envelope_present", False)
            ),
            whole_body_observation_schema_present=bool(
                payload.get("whole_body_observation_schema_present", False)
            ),
            whole_body_action_schema_present=bool(
                payload.get("whole_body_action_schema_present", False)
            ),
            balance_envelope_present=bool(
                payload.get("balance_envelope_present", False)
            ),
            bipedal_chassis_local_scaffold_complete=bool(
                payload.get("bipedal_chassis_local_scaffold_complete", False)
            ),
            local_structural_refit_complete=bool(
                payload.get("local_structural_refit_complete", False)
            ),
            ready_for_phase4_local_sweep=bool(
                payload.get("ready_for_phase4_local_sweep", False)
            ),
            ready_for_unitree_runtime=bool(
                payload.get("ready_for_unitree_runtime", False)
            ),
            ready_for_training=bool(payload.get("ready_for_training", False)),
            training_executed=bool(payload.get("training_executed", False)),
            weights_written=bool(payload.get("weights_written", False)),
            provider_executed=bool(payload.get("provider_executed", False)),
            hardware_executed=bool(payload.get("hardware_executed", False)),
            unitree_sim_runtime_executed=bool(
                payload.get("unitree_sim_runtime_executed", False)
            ),
            live_policy_control=bool(payload.get("live_policy_control", False)),
            reward_math_mutation=bool(payload.get("reward_math_mutation", False)),
            promotion_eligible=bool(payload.get("promotion_eligible", False)),
            denied_gates=denied_gate_map(payload.get("denied_gates")),
            remaining_blockers=strings(payload.get("remaining_blockers")),
            artifact_refs=mapping(payload.get("artifact_refs")),
            version=str(payload.get("version", PHASE35_REFIT_REPORT_VERSION)),
        )


def build_phase35_humanoid_refit(
    artifact_refs: Mapping[str, Any] | None = None,
    bipedal_chassis_report: Mapping[str, Any] | None = None,
) -> tuple[
    HumanoidPhase35RefitReport,
    list[HumanoidCapacityBandContract],
    list[HumanoidSchemaDeltaContract],
    list[HumanoidEnvTaxonomyReceipt],
    list[HumanoidBenchmarkTarget],
]:
    """Build deterministic local Phase 3.5 readiness artifacts."""

    capacity_specs = [
        (
            "onboard_reflex_reserve",
            "onboard",
            "servo_reflex",
            ["servo loops", "watchdogs", "physical safety checks"],
            ["provider_calls", "economic_planning", "transport_training"],
            False,
        ),
        (
            "onboard_low_rate_state",
            "onboard",
            "whole_body_fast",
            ["compact proprioception", "IMU/contact summaries"],
            ["bipedal_promotion_without_timing", "large_perception"],
            True,
        ),
        (
            "companion_realtime_assist",
            "companion",
            "wm_slow",
            ["perception fusion", "retargeting prechecks", "proposal scoring"],
            ["hard_servo_authority_without_phase4a"],
            True,
        ),
        (
            "companion_heavy_inference",
            "companion",
            "wm_slow",
            ["segmentation", "sim preview", "transport critics"],
            ["unbounded_latency_control_loop"],
            True,
        ),
        (
            "offline_gpu_training",
            "offline_gpu",
            "offline",
            ["bridge training", "whole-body training", "benchmark sweeps"],
            ["live_policy_control", "on_robot_authority"],
            False,
        ),
    ]
    capacity_bands = [
        HumanoidCapacityBandContract(
            band_id=f"phase35_capacity_{name}",
            band_name=name,
            compute_placement=placement,
            control_rate_class=rate,
            intended_work=intended,
            excluded_authority=excluded,
            battery_reserve_class="reserve_first",
            thermal_headroom_class="receipt_required",
            comms_qos_class="receipt_required",
            degraded_mode_allowed=degraded,
            replay_training_awareness=[
                "capacity_receipt",
                "timing_receipt",
                "resource_training_slot",
            ],
        )
        for name, placement, rate, intended, excluded, degraded in capacity_specs
    ]

    schema_specs = [
        (
            "observation",
            "whole_body_proprioception",
            "bipedal_whole_body",
            ["floating_base_pose", "joint_velocity", "joint_torque", "joint_temp"],
        ),
        (
            "observation",
            "imu_and_support_state",
            "bipedal_whole_body",
            ["imu_orientation", "support_phase", "slip_estimate", "balance_margin"],
        ),
        (
            "observation",
            "contact_and_force_state",
            "bipedal_whole_body",
            ["hand_contact", "foot_contact", "force_torque", "contact_normals"],
        ),
        (
            "observation",
            "egocentric_perception",
            "bipedal_whole_body",
            ["camera_refs", "depth_refs", "calibration_refs", "body_relative_scene"],
        ),
        (
            "observation",
            "resource_and_timing",
            "bipedal_whole_body",
            ["compute_placement", "latency_ms", "battery_state", "comms_qos"],
        ),
        (
            "action",
            "whole_body_action_chunk",
            "bipedal_whole_body",
            ["base_action", "torso_action", "arm_action", "hand_action"],
        ),
        (
            "action",
            "balance_preserving_reach",
            "bipedal_whole_body",
            ["reach_target", "balance_margin", "fallback_envelope"],
        ),
        (
            "action",
            "bimanual_dexterous_manipulation",
            "bipedal_whole_body",
            ["dual_arm_targets", "contact_plan", "force_limits", "tool_state"],
        ),
        (
            "action",
            "stable_base_fallback_action",
            "stable_base_mobile_manipulator",
            ["stable_base_envelope", "recovery_mode", "task_continuity_ref"],
        ),
        (
            "action",
            "operator_recovery_action",
            "bipedal_whole_body",
            ["handoff_request", "teleop_authority", "recovery_trace_ref"],
        ),
    ]
    schema_deltas = [
        HumanoidSchemaDeltaContract(
            delta_id=f"phase35_schema_{family}_{surface}",
            schema_family=family,
            surface_name=surface,
            posture_scope=posture,
            required_fields=fields,
            schema_refs={
                "observation_schema_ref": "humanoid_observation_schema_v1",
                "action_schema_ref": "humanoid_action_schema_v1",
            },
            replay_training_awareness=[
                "posture_tag_required",
                "event_spine_ref_required",
                "replay_export_ref_required",
            ],
            promotion_posture="contract_delta_only",
        )
        for family, surface, posture, fields in schema_specs
    ]

    env_specs = [
        (
            "fixed_base_tabletop_*",
            "fixed_base_tabletop",
            "curriculum_regression",
            "cannot_close_g1_r1_whole_body_readiness",
            "curriculum_only",
        ),
        (
            "stable_base_mobile_manipulator_*",
            "stable_base_mobile_manipulator",
            "safety_fallback_degraded_mode",
            "cannot_replace_bipedal_readiness",
            "fallback_only",
        ),
        (
            "bipedal_whole_body_*",
            "bipedal_whole_body",
            "primary_humanoid_readiness",
            "required_for_g1_r1_promotion",
            "benchmark_candidate_after_evidence",
        ),
    ]
    env_receipts = [
        HumanoidEnvTaxonomyReceipt(
            receipt_id=f"phase35_env_{posture}",
            env_family=family,
            posture_tag=posture,
            role=role,
            promotion_limit=limit,
            required_artifacts=[
                "posture_tag",
                "backend_truth",
                "robot_asset_ref",
                "observation_schema_ref",
                "action_schema_ref",
                "replay_export_ref",
            ],
            promotion_posture=promotion,
        )
        for family, posture, role, limit, promotion in env_specs
    ]

    benchmark_specs = [
        "balance_stability",
        "locomotion_manipulation",
        "bimanual_dexterous_task",
        "disturbance_recovery",
        "degraded_sensing",
        "stable_base_fallback",
        "tabletop_curriculum",
    ]
    benchmarks = [
        HumanoidBenchmarkTarget(
            benchmark_id=f"phase35_benchmark_{name}",
            benchmark_class=name,
            posture_tag=(
                "stable_base_mobile_manipulator"
                if name == "stable_base_fallback"
                else "fixed_base_tabletop"
                if name == "tabletop_curriculum"
                else "bipedal_whole_body"
            ),
            current_status="planning_only"
            if name in {"stable_base_fallback", "tabletop_curriculum"}
            else "missing_runtime_evidence",
            required_evidence=[
                "runtime_packet",
                "event_spine_ref",
                "governance_trace_ref",
                "posture_tagged_replay_export",
            ],
            future_closure_evidence=[
                "unitree_sim_or_hardware_receipts",
                "measured_timing_and_recovery_traces",
                "promotion_grade_benchmark_report",
            ],
        )
        for name in benchmark_specs
    ]

    chassis_payload = _bipedal_chassis_payload(bipedal_chassis_report)
    chassis_complete = bool(
        chassis_payload.get("local_structural_scaffold_complete", False)
    )
    complete = (
        len(capacity_bands) >= 5
        and len(schema_deltas) >= 10
        and len(env_receipts) == 3
        and len(benchmarks) >= 7
        and chassis_complete
    )
    report_payload = {
        "capacity_band_count": len(capacity_bands),
        "schema_delta_count": len(schema_deltas),
        "env_taxonomy_count": len(env_receipts),
        "benchmark_target_count": len(benchmarks),
        "bipedal_chassis_report_id": chassis_payload.get("report_id", ""),
        "artifact_refs": mapping(artifact_refs),
    }
    report = HumanoidPhase35RefitReport(
        report_id=stable_id("phase35_refit", report_payload),
        status="ok" if complete else "blocked",
        capacity_band_count=len(capacity_bands),
        schema_delta_count=len(schema_deltas),
        env_taxonomy_count=len(env_receipts),
        benchmark_target_count=len(benchmarks),
        bipedal_chassis_report_id=str(chassis_payload.get("report_id", "")),
        bipedal_chassis_joint_count=int(
            chassis_payload.get("controlled_joint_count", 0) or 0
        ),
        bipedal_chassis_frame_count=int(chassis_payload.get("frame_count", 0) or 0),
        bipedal_chassis_joint_limit_envelope_count=int(
            chassis_payload.get("joint_limit_envelope_count", 0) or 0
        ),
        bipedal_balance_receipt_count=int(
            chassis_payload.get("balance_receipt_count", 0) or 0
        ),
        canonical_bipedal_chassis_present=bool(
            chassis_payload.get("canonical_bipedal_chassis_present", False)
        ),
        limb_frame_tree_present=bool(
            chassis_payload.get("limb_frame_tree_present", False)
        ),
        joint_limit_envelope_present=bool(
            chassis_payload.get("joint_limit_envelope_present", False)
        ),
        whole_body_observation_schema_present=bool(
            chassis_payload.get("whole_body_observation_schema_present", False)
        ),
        whole_body_action_schema_present=bool(
            chassis_payload.get("whole_body_action_schema_present", False)
        ),
        balance_envelope_present=bool(
            chassis_payload.get("balance_envelope_present", False)
        ),
        bipedal_chassis_local_scaffold_complete=chassis_complete,
        local_structural_refit_complete=complete,
        ready_for_phase4_local_sweep=complete,
        denied_gates=denied_gate_map(),
        remaining_blockers=list(PHASE35_REMAINING_BLOCKERS),
        artifact_refs=mapping(artifact_refs),
    )
    return report, capacity_bands, schema_deltas, env_receipts, benchmarks


def save_phase35_humanoid_refit(
    output_dir: str | Path,
    report: HumanoidPhase35RefitReport,
    capacity_bands: list[HumanoidCapacityBandContract],
    schema_deltas: list[HumanoidSchemaDeltaContract],
    env_receipts: list[HumanoidEnvTaxonomyReceipt],
    benchmarks: list[HumanoidBenchmarkTarget],
) -> dict[str, str]:
    output = Path(output_dir)
    paths = {
        "report_path": output / "humanoid_phase35_refit_report_v1.json",
        "capacity_bands_path": output
        / "humanoid_phase35_capacity_band_contracts_v1.jsonl",
        "schema_deltas_path": output / "humanoid_phase35_schema_delta_contracts_v1.jsonl",
        "env_taxonomy_path": output / "humanoid_phase35_env_taxonomy_receipts_v1.jsonl",
        "benchmarks_path": output / "humanoid_phase35_benchmark_taxonomy_v1.jsonl",
    }
    write_json(paths["report_path"], report.to_dict())
    write_jsonl(paths["capacity_bands_path"], [item.to_dict() for item in capacity_bands])
    write_jsonl(paths["schema_deltas_path"], [item.to_dict() for item in schema_deltas])
    write_jsonl(paths["env_taxonomy_path"], [item.to_dict() for item in env_receipts])
    write_jsonl(paths["benchmarks_path"], [item.to_dict() for item in benchmarks])
    return {key: str(value) for key, value in paths.items()}


def load_phase35_humanoid_refit_report(
    path: str | Path,
) -> HumanoidPhase35RefitReport:
    return HumanoidPhase35RefitReport.from_dict(load_json(path))


def load_phase35_capacity_bands(
    path: str | Path,
) -> list[HumanoidCapacityBandContract]:
    return [HumanoidCapacityBandContract.from_dict(row) for row in load_jsonl(path)]


def load_phase35_schema_deltas(
    path: str | Path,
) -> list[HumanoidSchemaDeltaContract]:
    return [HumanoidSchemaDeltaContract.from_dict(row) for row in load_jsonl(path)]


def load_phase35_env_taxonomy_receipts(
    path: str | Path,
) -> list[HumanoidEnvTaxonomyReceipt]:
    return [HumanoidEnvTaxonomyReceipt.from_dict(row) for row in load_jsonl(path)]


def load_phase35_benchmark_targets(path: str | Path) -> list[HumanoidBenchmarkTarget]:
    return [HumanoidBenchmarkTarget.from_dict(row) for row in load_jsonl(path)]
