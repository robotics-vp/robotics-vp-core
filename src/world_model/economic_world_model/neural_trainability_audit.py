"""Neural trainability audit for local WM and trainer surfaces.

The audit produced here is a planning and receipt surface. It inventories
neural/seam/encoder/policy/head/bridge/receiver/trainer components, records
what local scaffolding exists, and emits executable follow-up rows for missing
code, rows, losses, data, provider, GPU, hardware, and benchmark proof.

It does not train, write weights, run providers, launch RunPod, execute
hardware, mutate reward/controller math, grant Phase 7 authority, or make
promotion claims.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from src.utils.config_digest import sha256_json
from src.utils.json_safe import to_json_safe

NEURAL_TRAINABILITY_COMPONENT_VERSION = "neural_trainability_component_v1"
NEURAL_TRAINABILITY_FOLLOWUP_VERSION = "neural_trainability_followup_v1"
NEURAL_TRAINABILITY_AUDIT_REPORT_VERSION = "neural_trainability_audit_report_v1"

FOLLOWUP_PLANES = (
    "local",
    "codex",
    "runpod_provider",
    "runpod_train",
    "hardware_runtime",
)

FOLLOWUP_BLOCKERS = (
    "code",
    "loss",
    "row",
    "data",
    "provider",
    "gpu",
    "hardware",
    "benchmark_missing",
)

DENIED_NEURAL_AUDIT_AUTHORITIES = (
    "training_executed",
    "weights_written",
    "provider_executed",
    "gpu_executed",
    "runpod_launched",
    "hardware_executed",
    "ros2_publish_attempted",
    "unitree_sdk2_write_enabled",
    "live_policy_control",
    "reward_math_mutation",
    "phase7_authority_granted",
    "promotion_eligible",
)


def _mapping(payload: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    return dict(to_json_safe(dict(payload or {})))


def _strings(values: Optional[Iterable[Any]]) -> list[str]:
    return [str(value) for value in list(values or []) if str(value)]


def _stable_id(prefix: str, payload: Mapping[str, Any]) -> str:
    return f"{prefix}_{sha256_json(_mapping(payload))[:16]}"


def _denied_gates() -> dict[str, bool]:
    return {key: False for key in DENIED_NEURAL_AUDIT_AUTHORITIES}


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    return slug or "unknown"


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_training_migration_backlog(path: str | Path) -> list[dict[str, Any]]:
    """Load backlog rows as dictionaries, returning [] when unavailable."""

    target = Path(path)
    if not target.exists():
        return []
    payload = _load_json(target)
    return [
        _mapping(row)
        for row in list(payload.get("backlog", []) or [])
        if isinstance(row, Mapping)
    ]


@dataclass(frozen=True)
class NeuralTrainabilityComponent:
    """One audited neural/seam/policy/trainer component."""

    component_id: str
    owner: str
    component_type: str
    surface_roles: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    train_scripts: list[str] = field(default_factory=list)
    backlog_scripts: list[str] = field(default_factory=list)
    source_backlog_rows: list[dict[str, Any]] = field(default_factory=list)
    smoke_commands: list[str] = field(default_factory=list)
    loss_refs: list[str] = field(default_factory=list)
    row_refs: list[str] = field(default_factory=list)
    trainer_refs: list[str] = field(default_factory=list)
    data_ideas: list[str] = field(default_factory=list)
    provider_dependencies: list[str] = field(default_factory=list)
    gpu_dependencies: list[str] = field(default_factory=list)
    hardware_dependencies: list[str] = field(default_factory=list)
    receipt_refs: list[str] = field(default_factory=list)
    promotion_status: str = "not_promotion_eligible"
    blockers: list[str] = field(default_factory=list)
    tests: list[str] = field(default_factory=list)
    local_static_ready: bool = False
    ready_for_training: bool = False
    promotion_eligible: bool = False
    missing_item_count: int = 0
    denied_gates: dict[str, bool] = field(default_factory=_denied_gates)
    metadata: dict[str, Any] = field(default_factory=dict)
    version: str = NEURAL_TRAINABILITY_COMPONENT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "component_id": self.component_id,
            "version": self.version,
            "owner": self.owner,
            "component_type": self.component_type,
            "surface_roles": list(self.surface_roles),
            "files": list(self.files),
            "train_scripts": list(self.train_scripts),
            "backlog_scripts": list(self.backlog_scripts),
            "source_backlog_rows": [_mapping(row) for row in self.source_backlog_rows],
            "smoke_commands": list(self.smoke_commands),
            "loss_refs": list(self.loss_refs),
            "row_refs": list(self.row_refs),
            "trainer_refs": list(self.trainer_refs),
            "data_ideas": list(self.data_ideas),
            "provider_dependencies": list(self.provider_dependencies),
            "gpu_dependencies": list(self.gpu_dependencies),
            "hardware_dependencies": list(self.hardware_dependencies),
            "receipt_refs": list(self.receipt_refs),
            "promotion_status": self.promotion_status,
            "blockers": list(self.blockers),
            "tests": list(self.tests),
            "local_static_ready": bool(self.local_static_ready),
            "ready_for_training": bool(self.ready_for_training),
            "promotion_eligible": bool(self.promotion_eligible),
            "missing_item_count": int(self.missing_item_count),
            "denied_gates": dict(self.denied_gates),
            "metadata": _mapping(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "NeuralTrainabilityComponent":
        return cls(
            component_id=str(payload.get("component_id", "")),
            owner=str(payload.get("owner", "")),
            component_type=str(payload.get("component_type", "")),
            surface_roles=_strings(payload.get("surface_roles")),
            files=_strings(payload.get("files")),
            train_scripts=_strings(payload.get("train_scripts")),
            backlog_scripts=_strings(payload.get("backlog_scripts")),
            source_backlog_rows=[
                _mapping(row)
                for row in list(payload.get("source_backlog_rows", []) or [])
                if isinstance(row, Mapping)
            ],
            smoke_commands=_strings(payload.get("smoke_commands")),
            loss_refs=_strings(payload.get("loss_refs")),
            row_refs=_strings(payload.get("row_refs")),
            trainer_refs=_strings(payload.get("trainer_refs")),
            data_ideas=_strings(payload.get("data_ideas")),
            provider_dependencies=_strings(payload.get("provider_dependencies")),
            gpu_dependencies=_strings(payload.get("gpu_dependencies")),
            hardware_dependencies=_strings(payload.get("hardware_dependencies")),
            receipt_refs=_strings(payload.get("receipt_refs")),
            promotion_status=str(
                payload.get("promotion_status", "not_promotion_eligible")
            ),
            blockers=_strings(payload.get("blockers")),
            tests=_strings(payload.get("tests")),
            local_static_ready=bool(payload.get("local_static_ready", False)),
            ready_for_training=bool(payload.get("ready_for_training", False)),
            promotion_eligible=bool(payload.get("promotion_eligible", False)),
            missing_item_count=int(payload.get("missing_item_count", 0) or 0),
            denied_gates={
                **_denied_gates(),
                **{
                    str(key): bool(value)
                    for key, value in dict(
                        payload.get("denied_gates", {}) or {}
                    ).items()
                },
            },
            metadata=_mapping(payload.get("metadata")),
            version=str(payload.get("version", NEURAL_TRAINABILITY_COMPONENT_VERSION)),
        )


@dataclass(frozen=True)
class NeuralTrainabilityFollowupRow:
    """Executable follow-up for one incomplete trainability item."""

    missing_item_id: str
    component_id: str
    owner: str
    action: str
    target: str
    plane: str
    blocker: str
    verify_receipt: str
    status: str = "blocked_followup"
    promotion_eligible: bool = False
    source_refs: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    version: str = NEURAL_TRAINABILITY_FOLLOWUP_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "missing_item_id": self.missing_item_id,
            "version": self.version,
            "component_id": self.component_id,
            "owner": self.owner,
            "action": self.action,
            "target": self.target,
            "plane": self.plane,
            "blocker": self.blocker,
            "verify_receipt": self.verify_receipt,
            "status": self.status,
            "promotion_eligible": bool(self.promotion_eligible),
            "source_refs": _mapping(self.source_refs),
            "metadata": _mapping(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "NeuralTrainabilityFollowupRow":
        return cls(
            missing_item_id=str(payload.get("missing_item_id", "")),
            component_id=str(payload.get("component_id", "")),
            owner=str(payload.get("owner", "")),
            action=str(payload.get("action", "")),
            target=str(payload.get("target", "")),
            plane=str(payload.get("plane", "")),
            blocker=str(payload.get("blocker", "")),
            verify_receipt=str(payload.get("verify_receipt", "")),
            status=str(payload.get("status", "blocked_followup")),
            promotion_eligible=bool(payload.get("promotion_eligible", False)),
            source_refs=_mapping(payload.get("source_refs")),
            metadata=_mapping(payload.get("metadata")),
            version=str(payload.get("version", NEURAL_TRAINABILITY_FOLLOWUP_VERSION)),
        )


@dataclass(frozen=True)
class NeuralTrainabilityAuditReport:
    """Top-level trainability audit receipt."""

    audit_id: str
    status: str
    component_count: int
    followup_count: int
    ready_for_training_count: int
    promotion_eligible_count: int
    local_static_ready_count: int
    plane_counts: dict[str, int]
    blocker_counts: dict[str, int]
    surface_role_counts: dict[str, int]
    artifact_refs: dict[str, Any] = field(default_factory=dict)
    denied_gates: dict[str, bool] = field(default_factory=_denied_gates)
    training_executed: bool = False
    weights_written: bool = False
    provider_executed: bool = False
    gpu_executed: bool = False
    runpod_launched: bool = False
    hardware_executed: bool = False
    live_policy_control: bool = False
    reward_math_mutation: bool = False
    phase7_authority_granted: bool = False
    promotion_eligible: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    version: str = NEURAL_TRAINABILITY_AUDIT_REPORT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "audit_id": self.audit_id,
            "version": self.version,
            "status": self.status,
            "component_count": int(self.component_count),
            "followup_count": int(self.followup_count),
            "ready_for_training_count": int(self.ready_for_training_count),
            "promotion_eligible_count": int(self.promotion_eligible_count),
            "local_static_ready_count": int(self.local_static_ready_count),
            "plane_counts": {
                str(key): int(value) for key, value in self.plane_counts.items()
            },
            "blocker_counts": {
                str(key): int(value) for key, value in self.blocker_counts.items()
            },
            "surface_role_counts": {
                str(key): int(value) for key, value in self.surface_role_counts.items()
            },
            "artifact_refs": _mapping(self.artifact_refs),
            "denied_gates": dict(self.denied_gates),
            "training_executed": bool(self.training_executed),
            "weights_written": bool(self.weights_written),
            "provider_executed": bool(self.provider_executed),
            "gpu_executed": bool(self.gpu_executed),
            "runpod_launched": bool(self.runpod_launched),
            "hardware_executed": bool(self.hardware_executed),
            "live_policy_control": bool(self.live_policy_control),
            "reward_math_mutation": bool(self.reward_math_mutation),
            "phase7_authority_granted": bool(self.phase7_authority_granted),
            "promotion_eligible": bool(self.promotion_eligible),
            "metadata": _mapping(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "NeuralTrainabilityAuditReport":
        return cls(
            audit_id=str(payload.get("audit_id", "")),
            status=str(payload.get("status", "blocked")),
            component_count=int(payload.get("component_count", 0) or 0),
            followup_count=int(payload.get("followup_count", 0) or 0),
            ready_for_training_count=int(
                payload.get("ready_for_training_count", 0) or 0
            ),
            promotion_eligible_count=int(
                payload.get("promotion_eligible_count", 0) or 0
            ),
            local_static_ready_count=int(
                payload.get("local_static_ready_count", 0) or 0
            ),
            plane_counts={
                str(key): int(value)
                for key, value in dict(payload.get("plane_counts", {}) or {}).items()
            },
            blocker_counts={
                str(key): int(value)
                for key, value in dict(payload.get("blocker_counts", {}) or {}).items()
            },
            surface_role_counts={
                str(key): int(value)
                for key, value in dict(
                    payload.get("surface_role_counts", {}) or {}
                ).items()
            },
            artifact_refs=_mapping(payload.get("artifact_refs")),
            denied_gates={
                **_denied_gates(),
                **{
                    str(key): bool(value)
                    for key, value in dict(
                        payload.get("denied_gates", {}) or {}
                    ).items()
                },
            },
            training_executed=bool(payload.get("training_executed", False)),
            weights_written=bool(payload.get("weights_written", False)),
            provider_executed=bool(payload.get("provider_executed", False)),
            gpu_executed=bool(payload.get("gpu_executed", False)),
            runpod_launched=bool(payload.get("runpod_launched", False)),
            hardware_executed=bool(payload.get("hardware_executed", False)),
            live_policy_control=bool(payload.get("live_policy_control", False)),
            reward_math_mutation=bool(payload.get("reward_math_mutation", False)),
            phase7_authority_granted=bool(
                payload.get("phase7_authority_granted", False)
            ),
            promotion_eligible=bool(payload.get("promotion_eligible", False)),
            metadata=_mapping(payload.get("metadata")),
            version=str(
                payload.get("version", NEURAL_TRAINABILITY_AUDIT_REPORT_VERSION)
            ),
        )


def _component_specs() -> list[dict[str, Any]]:
    return [
        {
            "component_id": "perception_evidence_fusion_seam",
            "owner": "perception_grounding",
            "component_type": "seam",
            "surface_roles": [
                "lower-WM producer",
                "trainer/runtime lane",
                "receipt substrate",
            ],
            "files": [
                "src/world_model/perception_grounding/neural_seams.py",
                "src/training/perception_seam_trainer.py",
                "src/training/perception_seam_losses.py",
                "src/training/perception_seam_data.py",
            ],
            "train_scripts": ["scripts/smoke_test_perception_seam_training.py"],
            "loss_refs": ["src/training/perception_seam_losses.py"],
            "row_refs": ["src/training/perception_seam_data.py"],
            "trainer_refs": ["src/training/perception_seam_trainer.py"],
            "data_ideas": [
                "LeRobot video-receipt replay/perception samples",
                "governed-video Stage-1 replay rows",
            ],
            "provider_dependencies": ["DINO/SigLIP features", "V-JEPA temporal tokens"],
            "gpu_dependencies": ["perception seam training run"],
            "receipt_refs": [
                "lerobot_video_replay_perception_receipts",
                "perception_seam_training_receipt",
            ],
            "tests": [
                "tests/test_lerobot_video_receipt_adapter.py",
                "tests/test_lerobot_perception_adapter.py",
            ],
            "blockers": ["provider", "gpu", "benchmark_missing"],
            "missing_items": [
                {
                    "action": "run provider-backed perception feature receipt pass",
                    "target": "provider_bringup:dino_siglip_vision_backbone",
                    "plane": "runpod_provider",
                    "blocker": "provider",
                    "verify_receipt": "provider_bringup_ledger_entry:dino_siglip_vision_backbone",
                },
                {
                    "action": "train perception seam on receipt-backed rows",
                    "target": "src/training/perception_seam_trainer.py",
                    "plane": "runpod_train",
                    "blocker": "gpu",
                    "verify_receipt": "perception_seam_training_receipt_v1",
                },
            ],
        },
        {
            "component_id": "perception_vjepa_temporal_alignment",
            "owner": "perception_grounding",
            "component_type": "temporal_seam",
            "surface_roles": ["lower-WM producer", "trainer/runtime lane"],
            "files": [
                "src/world_model/perception_grounding/neural_seams.py",
                "src/dataset_bridges/lerobot_video_receipt_adapter.py",
            ],
            "train_scripts": ["scripts/smoke_test_vjepa_temporal_seam.py"],
            "backlog_scripts": ["train_vjepa2_perception_grounding.py"],
            "row_refs": ["src/dataset_bridges/lerobot_video_receipt_adapter.py"],
            "data_ideas": ["receipt-backed temporal windows with action/event refs"],
            "provider_dependencies": ["V-JEPA2 provider/runtime"],
            "gpu_dependencies": ["V-JEPA2 fine-tuning"],
            "receipt_refs": ["vjepa_temporal_sample_count"],
            "tests": ["tests/test_lerobot_video_receipt_adapter.py"],
            "blockers": ["provider", "gpu", "data"],
            "missing_items": [
                {
                    "action": "bring up V-JEPA2 temporal provider receipt",
                    "target": "train_vjepa2_perception_grounding.py",
                    "plane": "runpod_provider",
                    "blocker": "provider",
                    "verify_receipt": "vjepa2_perception_provider_receipt_v1",
                },
                {
                    "action": "train temporal alignment using receipt-backed windows",
                    "target": "train_vjepa2_perception_grounding.py",
                    "plane": "runpod_train",
                    "blocker": "gpu",
                    "verify_receipt": "vjepa2_perception_training_receipt_v1",
                },
            ],
        },
        {
            "component_id": "vision_backbone_projection_head",
            "owner": "perception_grounding",
            "component_type": "encoder_head",
            "surface_roles": [
                "lower-WM producer",
                "provider/hardware adapter",
                "trainer/runtime lane",
            ],
            "files": [
                "src/policies/vision_encoder.py",
                "src/vision/encoder_with_heads.py",
                "src/dataset_bridges/lerobot_perception_adapter.py",
            ],
            "train_scripts": [
                "scripts/train_vision_backbone.py",
                "scripts/train_vision_backbone_real.py",
            ],
            "data_ideas": ["real provider features paired with replay/econ refs"],
            "provider_dependencies": ["DINO/SigLIP", "SAM/SAM3D"],
            "gpu_dependencies": ["vision backbone fine-tuning"],
            "receipt_refs": ["vision_backbone_projection_sample_count"],
            "tests": ["tests/test_lerobot_perception_adapter.py"],
            "blockers": ["provider", "gpu", "benchmark_missing"],
            "missing_items": [
                {
                    "action": "replace placeholder/flattened features with provider receipts",
                    "target": "provider_bringup:dino_siglip_vision_backbone",
                    "plane": "runpod_provider",
                    "blocker": "provider",
                    "verify_receipt": "vision_provider_feature_receipts_v1",
                }
            ],
        },
        {
            "component_id": "sim_synth_predictive_vjepa_component",
            "owner": "sim_synth_physics",
            "component_type": "predictive_model",
            "surface_roles": [
                "lower-WM producer",
                "trainer/runtime lane",
                "provider/hardware adapter",
            ],
            "files": [
                "src/world_model/sim_synth_physics/training_corpus.py",
                "src/world_model/sim_synth_physics/runtime_bridge.py",
            ],
            "backlog_scripts": ["train_vjepa2_sim_synth_predictor.py"],
            "data_ideas": [
                "Isaac/Unitree/Holosoma runtime receipts",
                "governed video windows",
            ],
            "provider_dependencies": ["V-JEPA2", "Isaac/Unitree", "Holosoma"],
            "gpu_dependencies": ["predictive-state training"],
            "hardware_dependencies": ["Unitree sim/runtime traces"],
            "receipt_refs": ["sim_synth_predictive_receipt_v1"],
            "tests": ["tests/test_sim_synth_phase1x_subsystems.py"],
            "blockers": ["provider", "gpu", "hardware", "data"],
            "missing_items": [
                {
                    "action": "materialize provider/runtime predictive receipts",
                    "target": "provider_bringup:isaac_unitree_runtime",
                    "plane": "runpod_provider",
                    "blocker": "provider",
                    "verify_receipt": "isaac_unitree_runtime_receipt_v1",
                },
                {
                    "action": "collect policy-controlled Unitree trace labels",
                    "target": "Phase4 Unitree runtime traces",
                    "plane": "hardware_runtime",
                    "blocker": "hardware",
                    "verify_receipt": "unitree_policy_controlled_trace_receipt_v1",
                },
            ],
        },
        {
            "component_id": "embodiment_phase34_neural_architecture",
            "owner": "embodiment_actuation",
            "component_type": "whole_body_policy_scaffold",
            "surface_roles": [
                "lower-WM producer",
                "trainer/runtime lane",
                "provider/hardware adapter",
            ],
            "files": [
                "src/world_model/embodiment_actuation/neural_architectures.py",
                "src/world_model/embodiment_actuation/neural_seams.py",
                "src/world_model/embodiment_actuation/training_corpus.py",
            ],
            "backlog_scripts": ["train_embodiment_phase34_neural_architectures.py"],
            "data_ideas": [
                "G1 bipedal whole-body replay rows",
                "Phase 4 Unitree receipts",
            ],
            "provider_dependencies": ["Isaac/Unitree sim runtime"],
            "gpu_dependencies": ["whole-body policy training"],
            "hardware_dependencies": ["Unitree G1 stream/control evidence"],
            "receipt_refs": ["phase4_unitree_runtime_evidence_bridge"],
            "tests": [
                "tests/test_embodiment_actuation_phase34.py",
                "tests/test_humanoid_phase35_bipedal_chassis.py",
            ],
            "blockers": ["hardware", "gpu", "data", "benchmark_missing"],
            "missing_items": [
                {
                    "action": "record G1 whole-body hardware or calibrated sim trace receipts",
                    "target": "Phase4 Unitree runtime bridge",
                    "plane": "hardware_runtime",
                    "blocker": "hardware",
                    "verify_receipt": "unitree_g1_whole_body_trace_receipt_v1",
                },
                {
                    "action": "train embodiment neural architecture from receipt-backed corpus",
                    "target": "train_embodiment_phase34_neural_architectures.py",
                    "plane": "runpod_train",
                    "blocker": "gpu",
                    "verify_receipt": "embodiment_phase34_training_receipt_v1",
                },
            ],
        },
        {
            "component_id": "economic_world_model_neural_components",
            "owner": "economic_world_model",
            "component_type": "multi_head_trainer_scaffold",
            "surface_roles": ["trainer/runtime lane", "receipt substrate"],
            "files": [
                "src/world_model/economic_world_model/neural_architecture_manifest.py",
                "src/world_model/economic_world_model/training_rows.py",
                "scripts/train_economic_world_model_v0.py",
            ],
            "backlog_scripts": ["train_economic_world_model_v0.py"],
            "loss_refs": ["scripts/train_economic_world_model_v0.py"],
            "row_refs": ["src/world_model/economic_world_model/training_rows.py"],
            "trainer_refs": ["scripts/train_economic_world_model_v0.py"],
            "data_ideas": [
                "Phase-5 datapack rows",
                "lower-WM maturity rows",
                "shadow outcomes",
            ],
            "provider_dependencies": ["teacher/runtime evidence"],
            "gpu_dependencies": ["Economic WM trainer run"],
            "receipt_refs": ["economic_wm_trainer_scaffold_manifest_v1"],
            "tests": [
                "tests/test_economic_wm_trainer_scaffold.py",
                "tests/test_economic_wm_phase5_local_prep.py",
            ],
            "blockers": ["gpu", "provider", "benchmark_missing"],
            "missing_items": [
                {
                    "action": "run Economic WM trainer against non-stub receipt corpus",
                    "target": "scripts/train_economic_world_model_v0.py",
                    "plane": "runpod_train",
                    "blocker": "gpu",
                    "verify_receipt": "economic_wm_training_receipt_v1",
                }
            ],
        },
        {
            "component_id": "wm_transport_bridge_receiver_trainer",
            "owner": "wm_transport",
            "component_type": "bridge_receiver_trainer",
            "surface_roles": ["trainer/runtime lane", "receipt substrate"],
            "files": [
                "src/world_model/transport/neural_manifest.py",
                "src/world_model/transport/training_rows.py",
                "src/world_model/transport/training.py",
                "scripts/train_wm_transport_bridge_v0.py",
            ],
            "backlog_scripts": ["train_wm_transport_bridge_v0.py"],
            "loss_refs": ["src/world_model/transport/training.py"],
            "row_refs": ["src/world_model/transport/training_rows.py"],
            "trainer_refs": ["scripts/train_wm_transport_bridge_v0.py"],
            "data_ideas": ["transport rows plus Unitree event-spine join labels"],
            "gpu_dependencies": ["bridge/receiver training"],
            "provider_dependencies": ["provider/hardware transport evidence"],
            "receipt_refs": ["wm_transport_unitree_event_spine_joins_v1"],
            "tests": [
                "tests/test_wm_transport_phase63_neural_scaffold.py",
                "tests/test_wm_transport_phase64_runtime_eval.py",
            ],
            "blockers": ["gpu", "data", "benchmark_missing"],
            "missing_items": [
                {
                    "action": "train bridge/receiver with topology and event-spine labels",
                    "target": "scripts/train_wm_transport_bridge_v0.py",
                    "plane": "runpod_train",
                    "blocker": "gpu",
                    "verify_receipt": "wm_transport_training_receipt_v1",
                },
                {
                    "action": "run topology/latency benchmark receipts",
                    "target": "phase6_transport_latency_benchmarks",
                    "plane": "runpod_train",
                    "blocker": "benchmark_missing",
                    "verify_receipt": "wm_transport_latency_benchmark_receipt_v1",
                },
            ],
        },
        {
            "component_id": "phase65_meta_node_trainer",
            "owner": "humanoid_meta_node",
            "component_type": "meta_node_trainer",
            "surface_roles": ["trainer/runtime lane", "receipt substrate"],
            "files": [
                "src/world_model/humanoid_readiness/phase65_trainer.py",
                "scripts/economic_world_model/build_phase65_meta_node_trainer_scaffold.py",
            ],
            "backlog_scripts": ["train_meta_node_neuralization_v0.py"],
            "data_ideas": [
                "MetaNodeState rows",
                "transport evals",
                "intervention receipts",
            ],
            "gpu_dependencies": ["meta-node training"],
            "provider_dependencies": ["heldout robustness benchmark evidence"],
            "receipt_refs": ["phase65_meta_node_trainer_scaffold_manifest_v1"],
            "tests": ["tests/test_humanoid_phase65_meta_node_trainer_scaffold.py"],
            "blockers": ["gpu", "data", "benchmark_missing"],
            "missing_items": [
                {
                    "action": "train meta-node once counterfactual corpus exists",
                    "target": "train_meta_node_neuralization_v0.py",
                    "plane": "runpod_train",
                    "blocker": "gpu",
                    "verify_receipt": "phase65_meta_node_training_receipt_v1",
                }
            ],
        },
        {
            "component_id": "phase7_signal_adapter_consumer",
            "owner": "phase7_meta_regal",
            "component_type": "bounded_signal_adapter",
            "surface_roles": ["receipt substrate", "trainer/runtime lane"],
            "files": [
                "src/world_model/humanoid_readiness/phase7_signal_adapters.py",
                "src/world_model/humanoid_readiness/phase7_eval.py",
                "src/world_model/humanoid_readiness/phase7_hypernetwork.py",
            ],
            "data_ideas": [
                "Unitree Phase 6.4 event-spine joins",
                "LeRobot perception receipts",
                "bio/neuro receipt joins",
                "provider ledger entries",
                "neural trainability rows",
            ],
            "gpu_dependencies": ["future Phase 7 composition training"],
            "receipt_refs": [
                "provider_bringup_ledger_entries_v1",
                "lerobot_video_receipt_rows",
                "wm_transport_unitree_event_spine_joins_v1",
                "bio_neuro_receipt_join_rows_v1",
                "neural_trainability_components_v1",
                "phase7_governance_node_signal_receipts_v1",
            ],
            "tests": [
                "tests/test_humanoid_phase7_signal_adapters.py",
                "tests/test_humanoid_phase7_shadow_runtime_wiring.py",
            ],
            "blockers": ["data", "gpu", "benchmark_missing"],
            "missing_items": [
                {
                    "action": "collect labeled governance signal outcomes for future composition training",
                    "target": "phase7_governance_signal_outcome_corpus",
                    "plane": "runpod_train",
                    "blocker": "data",
                    "verify_receipt": "phase7_governance_outcome_corpus_receipt_v1",
                }
            ],
        },
        {
            "component_id": "bio_neuro_trainability_bundle",
            "owner": "bio_neuro_cross_wm",
            "component_type": "bio_neuro_trainer_family",
            "surface_roles": [
                "lower-WM producer",
                "trainer/runtime lane",
                "receipt substrate",
            ],
            "files": [
                "src/world_model/embodiment_actuation/bio_neuro_surfaces.py",
                "src/world_model/perception_grounding/bio_neuro_receipts.py",
                "src/world_model/economic_world_model/bio_neuro_receipt_join.py",
                "src/regal/bio_neuro_anomaly.py",
            ],
            "backlog_scripts": [
                "train_self_motion_expectation_v0.py",
                "train_active_sensing_policy_v0.py",
                "train_economic_regime_broadcast_v0.py",
                "train_embodiment_synergy_interoception_v0.py",
                "train_regal_anomaly_governance_v0.py",
                "train_plasticity_consolidation_gates_v0.py",
            ],
            "data_ideas": [
                "synchronized embodiment/perception trajectories",
                "active sensing outcomes",
            ],
            "provider_dependencies": ["temporal grounding provider"],
            "gpu_dependencies": ["bio/neuro trainer family"],
            "hardware_dependencies": ["real interoceptive telemetry"],
            "receipt_refs": ["bio_neuro_receipt_join_rows_v1"],
            "tests": ["tests/test_bio_neuro_substrate.py"],
            "blockers": ["data", "provider", "gpu", "hardware"],
            "missing_items": [
                {
                    "action": "collect synchronized self-motion and disturbance trajectories",
                    "target": "bio_neuro_self_motion_training_corpus",
                    "plane": "hardware_runtime",
                    "blocker": "hardware",
                    "verify_receipt": "self_motion_observed_corpus_receipt_v1",
                },
                {
                    "action": "train bio/neuro substrate models from receipt joins",
                    "target": "bio_neuro_training_family",
                    "plane": "runpod_train",
                    "blocker": "gpu",
                    "verify_receipt": "bio_neuro_training_receipt_v1",
                },
            ],
        },
        {
            "component_id": "orchestrator_semantic_runtime_trainers",
            "owner": "orchestrator_semantic_runtime",
            "component_type": "policy_and_scorer_trainer",
            "surface_roles": ["trainer/runtime lane", "receipt substrate"],
            "files": [
                "src/orchestrator/semantic_runtime_scorer_training.py",
                "src/orchestrator/semantic_policy.py",
                "src/orchestrator/meta_transformer_training.py",
                "src/world_model/semantic_state_encoder.py",
            ],
            "backlog_scripts": [
                "train_orchestration_transformer_v1_curriculum.py",
                "train_semantic_gap_conditioned_world_models.py",
            ],
            "data_ideas": [
                "semantic feedback packets",
                "queue decisions",
                "trajectory audits",
            ],
            "gpu_dependencies": ["semantic-conditioned model training"],
            "receipt_refs": ["semantic_runtime_scorer_manifest_v1"],
            "tests": ["tests/test_semantic_runtime_scorers.py"],
            "blockers": ["row", "gpu", "benchmark_missing"],
            "missing_items": [
                {
                    "action": "materialize unified semantic runtime training rows",
                    "target": "src/orchestrator/training_dataset.py",
                    "plane": "codex",
                    "blocker": "row",
                    "verify_receipt": "semantic_runtime_training_rows_v1",
                }
            ],
        },
        {
            "component_id": "vla_openvla_recap_heads",
            "owner": "vla_advisory",
            "component_type": "vla_provider_adapter",
            "surface_roles": ["provider/hardware adapter", "trainer/runtime lane"],
            "files": [
                "src/vla/openvla_encoder.py",
                "src/vla/vla_trainer.py",
                "src/vla/recap_heads.py",
            ],
            "backlog_scripts": ["train_vla_recap_offline.py"],
            "data_ideas": ["OpenVLA advisory labels joined to replay/econ refs"],
            "provider_dependencies": ["OpenVLA provider"],
            "gpu_dependencies": ["VLA adapter training"],
            "receipt_refs": ["openvla_provider_receipt_v1"],
            "tests": ["tests/test_vla_semantic_evidence.py"],
            "blockers": ["provider", "gpu", "benchmark_missing"],
            "missing_items": [
                {
                    "action": "run OpenVLA provider bring-up receipt before adapter training",
                    "target": "provider_bringup:openvla_semantic_teacher",
                    "plane": "runpod_provider",
                    "blocker": "provider",
                    "verify_receipt": "openvla_provider_receipt_v1",
                }
            ],
        },
        {
            "component_id": "rl_hrl_curriculum_policy_family",
            "owner": "rl_hrl_curriculum",
            "component_type": "curriculum_policy_trainer",
            "surface_roles": ["curriculum/regression source", "trainer/runtime lane"],
            "files": [
                "src/hrl/hrl_trainer.py",
                "src/hrl/low_level_policy.py",
                "src/rl/sampler_policy_training.py",
                "src/rl/hydra_heads.py",
            ],
            "backlog_scripts": [
                "train_high_level_controller.py",
                "train_motion_hierarchy_node.py",
            ],
            "data_ideas": ["fixed-base curriculum rows posture-tagged as non-G1 proof"],
            "gpu_dependencies": ["policy training"],
            "receipt_refs": ["curriculum_training_receipt_v1"],
            "tests": ["tests/test_runpod_launch_profiles.py"],
            "blockers": ["data", "gpu"],
            "missing_items": [
                {
                    "action": "posture-tag HRL/RL trainer rows as curriculum or G1-targeted",
                    "target": "src/hrl src/rl",
                    "plane": "codex",
                    "blocker": "row",
                    "verify_receipt": "curriculum_posture_training_rows_v1",
                }
            ],
        },
    ]


def _script_path(script: str) -> str:
    return script if script.startswith("scripts/") else f"scripts/{script}"


def _text_for_backlog(row: Mapping[str, Any]) -> str:
    return " ".join(
        str(row.get(key, ""))
        for key in (
            "script",
            "blocked_by",
            "notes",
            "priority",
            "downstream_dependents",
        )
    ).lower()


def _blocker_for_text(text: str) -> str:
    if "benchmark" in text or "promotion" in text:
        return "benchmark_missing"
    if "hardware" in text or "unitree" in text or "ros2" in text or "sdk2" in text:
        return "hardware"
    if "provider" in text or "openvla" in text or "vjepa" in text or "sam" in text:
        return "provider"
    if "gpu" in text or "cuda" in text or "training capacity" in text:
        return "gpu"
    if "loss" in text:
        return "loss"
    if "row" in text or "corpus" in text or "data" in text:
        return "data"
    return "code"


def _plane_for_blocker(blocker: str, text: str) -> str:
    if blocker == "hardware":
        return "hardware_runtime"
    if blocker == "provider":
        return "runpod_provider"
    if blocker in {"gpu", "benchmark_missing"}:
        return "runpod_train"
    if "script scaffold exists" in text:
        return "runpod_train"
    return "codex"


def _component_from_backlog_row(
    row: Mapping[str, Any],
) -> tuple[NeuralTrainabilityComponent, NeuralTrainabilityFollowupRow]:
    script = str(row.get("script", "unknown_training_script.py"))
    component_id = f"training_backlog_{_slug(script.removesuffix('.py'))}"
    text = _text_for_backlog(row)
    blocker = _blocker_for_text(text)
    plane = _plane_for_blocker(blocker, text)
    script_ref = _script_path(script)
    exists = Path(script_ref).exists()
    action = (
        f"complete receipt-safe trainer migration for {script}"
        if exists
        else f"materialize missing trainer scaffold or explicitly retire {script}"
    )
    component = NeuralTrainabilityComponent(
        component_id=component_id,
        owner=str(row.get("owner", "unknown")),
        component_type="training_migration_backlog_entry",
        surface_roles=["trainer/runtime lane", "legacy/dev-only tool"],
        files=[script_ref] if exists else [],
        train_scripts=[script_ref],
        backlog_scripts=[script],
        source_backlog_rows=[_mapping(row)],
        smoke_commands=[f"python3 {_script_path(script)} --help"] if exists else [],
        data_ideas=[str(row.get("notes", ""))],
        promotion_status="not_promotion_eligible",
        blockers=["training_migration_backlog_pending", blocker],
        tests=[],
        local_static_ready=exists,
        ready_for_training=False,
        promotion_eligible=False,
        missing_item_count=1,
        metadata={
            "auto_backlog_component": True,
            "priority": str(row.get("priority", "")),
        },
    )
    missing_item_id = _stable_id(
        "neural_trainability_missing",
        {"component_id": component_id, "script": script, "blocker": blocker},
    )
    followup = NeuralTrainabilityFollowupRow(
        missing_item_id=missing_item_id,
        component_id=component_id,
        owner=component.owner,
        action=action,
        target=script_ref,
        plane=plane,
        blocker=blocker,
        verify_receipt=f"training_migration_receipt:{script}",
        promotion_eligible=False,
        source_refs={"training_migration_backlog_script": script},
        metadata={"auto_backlog_followup": True},
    )
    return component, followup


def _backlog_rows_by_script(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("script", "")): _mapping(row) for row in rows if row.get("script")
    }


def _all_refs_exist(refs: Iterable[str]) -> bool:
    paths = [
        Path(ref) for ref in refs if ref and not ref.startswith("provider_bringup:")
    ]
    return bool(paths) and all(path.exists() for path in paths)


def _curated_component_from_spec(
    spec: Mapping[str, Any],
    *,
    backlog_by_script: Mapping[str, Mapping[str, Any]],
) -> tuple[NeuralTrainabilityComponent, list[NeuralTrainabilityFollowupRow]]:
    train_scripts = _strings(spec.get("train_scripts"))
    backlog_scripts = _strings(spec.get("backlog_scripts"))
    source_rows = [
        _mapping(backlog_by_script[script])
        for script in backlog_scripts
        if script in backlog_by_script
    ]
    files = _strings(spec.get("files"))
    missing_specs = [
        _mapping(row)
        for row in list(spec.get("missing_items", []) or [])
        if isinstance(row, Mapping)
    ]
    blockers = sorted(
        {
            *set(_strings(spec.get("blockers"))),
            *{
                _blocker_for_text(_text_for_backlog(row))
                for row in source_rows
                if row.get("blocked_by")
            },
        }
    )
    component = NeuralTrainabilityComponent(
        component_id=str(spec["component_id"]),
        owner=str(spec["owner"]),
        component_type=str(spec["component_type"]),
        surface_roles=_strings(spec.get("surface_roles")),
        files=files,
        train_scripts=train_scripts,
        backlog_scripts=backlog_scripts,
        source_backlog_rows=source_rows,
        smoke_commands=_strings(spec.get("smoke_commands")),
        loss_refs=_strings(spec.get("loss_refs")),
        row_refs=_strings(spec.get("row_refs")),
        trainer_refs=_strings(spec.get("trainer_refs")),
        data_ideas=_strings(spec.get("data_ideas")),
        provider_dependencies=_strings(spec.get("provider_dependencies")),
        gpu_dependencies=_strings(spec.get("gpu_dependencies")),
        hardware_dependencies=_strings(spec.get("hardware_dependencies")),
        receipt_refs=_strings(spec.get("receipt_refs")),
        promotion_status="not_promotion_eligible",
        blockers=blockers,
        tests=_strings(spec.get("tests")),
        local_static_ready=_all_refs_exist(files),
        ready_for_training=False,
        promotion_eligible=False,
        missing_item_count=len(missing_specs),
        metadata={"curated_audit_component": True},
    )
    followups: list[NeuralTrainabilityFollowupRow] = []
    for row in missing_specs:
        plane = str(row.get("plane", "codex"))
        blocker = str(row.get("blocker", "code"))
        missing_item_id = _stable_id(
            "neural_trainability_missing",
            {
                "component_id": component.component_id,
                "action": str(row.get("action", "")),
                "target": str(row.get("target", "")),
            },
        )
        followups.append(
            NeuralTrainabilityFollowupRow(
                missing_item_id=missing_item_id,
                component_id=component.component_id,
                owner=component.owner,
                action=str(row.get("action", "")),
                target=str(row.get("target", "")),
                plane=plane if plane in FOLLOWUP_PLANES else "codex",
                blocker=blocker if blocker in FOLLOWUP_BLOCKERS else "code",
                verify_receipt=str(row.get("verify_receipt", "")),
                promotion_eligible=False,
                source_refs={
                    "files": files,
                    "backlog_scripts": backlog_scripts,
                    "receipt_refs": component.receipt_refs,
                },
                metadata={"curated_followup": True},
            )
        )
    return component, followups


def _counts(values: Iterable[str], allowed: Iterable[str] = ()) -> dict[str, int]:
    counts = {key: 0 for key in allowed}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def build_neural_trainability_audit(
    *,
    training_backlog_path: str | Path = "scripts/TRAINING_MIGRATION_BACKLOG.json",
    artifact_refs: Optional[Mapping[str, Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> tuple[
    NeuralTrainabilityAuditReport,
    list[NeuralTrainabilityComponent],
    list[NeuralTrainabilityFollowupRow],
]:
    """Build a local, non-training neural trainability audit."""

    backlog_rows = load_training_migration_backlog(training_backlog_path)
    backlog_by_script = _backlog_rows_by_script(backlog_rows)
    components: list[NeuralTrainabilityComponent] = []
    followups: list[NeuralTrainabilityFollowupRow] = []
    covered_scripts: set[str] = set()

    for spec in _component_specs():
        component, rows = _curated_component_from_spec(
            spec, backlog_by_script=backlog_by_script
        )
        components.append(component)
        followups.extend(rows)
        covered_scripts.update(component.backlog_scripts)

    for row in backlog_rows:
        script = str(row.get("script", ""))
        if not script or script in covered_scripts:
            continue
        component, followup = _component_from_backlog_row(row)
        components.append(component)
        followups.append(followup)

    component_ids = [component.component_id for component in components]
    payload = {
        "component_ids": sorted(component_ids),
        "followup_ids": sorted(row.missing_item_id for row in followups),
        "training_backlog_path": str(training_backlog_path),
    }
    plane_counts = _counts((row.plane for row in followups), FOLLOWUP_PLANES)
    blocker_counts = _counts((row.blocker for row in followups), FOLLOWUP_BLOCKERS)
    surface_roles = [
        role for component in components for role in component.surface_roles
    ]
    report = NeuralTrainabilityAuditReport(
        audit_id=_stable_id("neural_trainability_audit", payload),
        status="ok_neural_trainability_audit_non_training",
        component_count=len(components),
        followup_count=len(followups),
        ready_for_training_count=sum(
            1 for component in components if component.ready_for_training
        ),
        promotion_eligible_count=sum(
            1 for component in components if component.promotion_eligible
        ),
        local_static_ready_count=sum(
            1 for component in components if component.local_static_ready
        ),
        plane_counts=plane_counts,
        blocker_counts=blocker_counts,
        surface_role_counts=_counts(surface_roles),
        artifact_refs=_mapping(artifact_refs),
        metadata={
            "training_backlog_path": str(training_backlog_path),
            "boundary": "non_training_trainability_audit_only",
            "g1_bipedal_whole_body_primary": True,
            "fixed_base_curriculum_only": True,
            **_mapping(metadata),
        },
    )
    return report, components, followups


def validate_neural_trainability_audit(
    *,
    report: NeuralTrainabilityAuditReport,
    components: Iterable[NeuralTrainabilityComponent],
    followups: Iterable[NeuralTrainabilityFollowupRow],
) -> dict[str, Any]:
    component_rows = list(components)
    followup_rows = list(followups)
    component_ids = {component.component_id for component in component_rows}
    errors: list[str] = []
    warnings: list[str] = []
    if report.component_count != len(component_rows):
        errors.append("component_count_mismatch")
    if report.followup_count != len(followup_rows):
        errors.append("followup_count_mismatch")
    if len(component_ids) != len(component_rows):
        errors.append("duplicate_component_id")
    if any(component.promotion_eligible for component in component_rows):
        errors.append("component_promotion_eligible")
    if any(component.ready_for_training for component in component_rows):
        warnings.append("component_marked_ready_for_training")
    if any(row.promotion_eligible for row in followup_rows):
        errors.append("followup_promotion_eligible")
    for row in followup_rows:
        if row.component_id not in component_ids:
            errors.append(f"missing_component_for_followup:{row.missing_item_id}")
        if row.plane not in FOLLOWUP_PLANES:
            errors.append(f"invalid_plane:{row.missing_item_id}:{row.plane}")
        if row.blocker not in FOLLOWUP_BLOCKERS:
            errors.append(f"invalid_blocker:{row.missing_item_id}:{row.blocker}")
        if not row.verify_receipt:
            errors.append(f"missing_verify_receipt:{row.missing_item_id}")
    if any(report.denied_gates.values()):
        errors.append("report_denied_gate_true")
    if report.promotion_eligible:
        errors.append("report_promotion_eligible")
    return {
        "status": "ok" if not errors else "blocked",
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors,
        "warnings": warnings,
        "component_count": len(component_rows),
        "followup_count": len(followup_rows),
        "safe_for_training": False,
        "safe_for_promotion": False,
    }


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(_mapping(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "".join(json.dumps(_mapping(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def save_neural_trainability_audit(
    *,
    report_path: str | Path,
    report: NeuralTrainabilityAuditReport,
    components_path: str | Path,
    components: Iterable[NeuralTrainabilityComponent],
    followups_path: str | Path,
    followups: Iterable[NeuralTrainabilityFollowupRow],
) -> None:
    _write_json(report_path, report.to_dict())
    _write_jsonl(components_path, [component.to_dict() for component in components])
    _write_jsonl(followups_path, [row.to_dict() for row in followups])


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_neural_trainability_audit_report(
    path: str | Path,
) -> NeuralTrainabilityAuditReport:
    return NeuralTrainabilityAuditReport.from_dict(_load_json(path))


def load_neural_trainability_components(
    path: str | Path,
) -> list[NeuralTrainabilityComponent]:
    return [NeuralTrainabilityComponent.from_dict(row) for row in _load_jsonl(path)]


def load_neural_trainability_followups(
    path: str | Path,
) -> list[NeuralTrainabilityFollowupRow]:
    return [NeuralTrainabilityFollowupRow.from_dict(row) for row in _load_jsonl(path)]


__all__ = [
    "DENIED_NEURAL_AUDIT_AUTHORITIES",
    "FOLLOWUP_BLOCKERS",
    "FOLLOWUP_PLANES",
    "NEURAL_TRAINABILITY_AUDIT_REPORT_VERSION",
    "NEURAL_TRAINABILITY_COMPONENT_VERSION",
    "NEURAL_TRAINABILITY_FOLLOWUP_VERSION",
    "NeuralTrainabilityAuditReport",
    "NeuralTrainabilityComponent",
    "NeuralTrainabilityFollowupRow",
    "build_neural_trainability_audit",
    "load_neural_trainability_audit_report",
    "load_neural_trainability_components",
    "load_neural_trainability_followups",
    "load_training_migration_backlog",
    "save_neural_trainability_audit",
    "validate_neural_trainability_audit",
]
