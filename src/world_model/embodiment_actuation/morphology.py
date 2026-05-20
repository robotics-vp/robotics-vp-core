"""Morphology and OSS evidence surfaces for Phase 3 Embodiment / Actuation.

This module turns public/local Unitree and sim-stack learnings into typed,
receipt-emitting local contracts. It does not claim hardware calibration or
provider execution; it records what can be known from repository/config/model
surfaces and leaves latency/watchdog/drift as external evidence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from src.embodiment.registry import CapabilityProfile, EmbodimentRegistryEntry

from .common import mapping, safe_float, safe_int, stable_id, strings

UNITREE_RL_GYM_G1_CONFIG_URL = (
    "https://github.com/unitreerobotics/unitree_rl_gym/blob/main/legged_gym/envs/g1/g1_config.py"
)
UNITREE_RL_LAB_URL = "https://github.com/unitreerobotics/unitree_rl_lab"
NVIDIA_SIM2REAL_COTRAINING_URL = (
    "https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/13-strategy2-cotraining.html"
)

G1_LOCOMOTION_12DOF_JOINTS: tuple[str, ...] = (
    "left_hip_yaw_joint",
    "left_hip_roll_joint",
    "left_hip_pitch_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_yaw_joint",
    "right_hip_roll_joint",
    "right_hip_pitch_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
)

G1_WAIST_3DOF_JOINTS: tuple[str, ...] = (
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
)

G1_ARM_14DOF_JOINTS: tuple[str, ...] = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

G1_DEX3_HAND_JOINTS: tuple[str, ...] = (
    "left_hand_index_0_joint",
    "left_hand_middle_0_joint",
    "left_hand_thumb_0_joint",
    "left_hand_index_1_joint",
    "left_hand_middle_1_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "right_hand_index_0_joint",
    "right_hand_middle_0_joint",
    "right_hand_thumb_0_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_1_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
)

G1_29DOF_JOINTS: tuple[str, ...] = (
    *G1_LOCOMOTION_12DOF_JOINTS,
    *G1_WAIST_3DOF_JOINTS,
    *G1_ARM_14DOF_JOINTS,
)

G1_VARIANT_JOINTS: dict[str, tuple[str, ...]] = {
    "g1_12dof_locomotion": G1_LOCOMOTION_12DOF_JOINTS,
    "g1_29dof": G1_29DOF_JOINTS,
    "g1_29dof_dex3": (*G1_29DOF_JOINTS, *G1_DEX3_HAND_JOINTS),
}


@dataclass(frozen=True)
class MorphologyJointSpec:
    joint_name: str
    group: str
    default_angle_rad: float = 0.0
    lower_rad: Optional[float] = None
    upper_rad: Optional[float] = None
    effort_limit: Optional[float] = None
    velocity_limit: Optional[float] = None
    source: str = "oss_pattern"
    metadata: dict[str, Any] = field(default_factory=dict)
    version: str = "morphology_joint_spec_v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "joint_name": self.joint_name,
            "group": self.group,
            "default_angle_rad": float(self.default_angle_rad),
            "lower_rad": self.lower_rad,
            "upper_rad": self.upper_rad,
            "effort_limit": self.effort_limit,
            "velocity_limit": self.velocity_limit,
            "source": self.source,
            "metadata": mapping(self.metadata),
            "version": self.version,
        }


@dataclass(frozen=True)
class G1MorphologyProfile:
    profile_id: str
    variant: str
    joint_specs: list[MorphologyJointSpec]
    action_dimension: int
    observation_dimension: int = 0
    privileged_observation_dimension: int = 0
    morphology_truth_class: str = "oss_config_pattern"
    source_refs: dict[str, Any] = field(default_factory=dict)
    unresolved_evidence: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    version: str = "g1_morphology_profile_v1"

    @property
    def joint_count(self) -> int:
        return len(self.joint_specs)

    def joint_names(self) -> list[str]:
        return [joint.joint_name for joint in self.joint_specs]

    def group_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for joint in self.joint_specs:
            counts[joint.group] = counts.get(joint.group, 0) + 1
        return counts

    def to_capability_profile(self) -> CapabilityProfile:
        return CapabilityProfile(
            profile_id=self.profile_id,
            robot_family="unitree_g1",
            sensor_modalities=["proprioception", "imu", "camera_optional"],
            action_spaces=[f"{self.variant}_joint_position"],
            workspace_bounds={},
            skill_capabilities={
                "locomotion": 0.75 if self.action_dimension >= 12 else 0.0,
                "whole_body_reach": 0.65 if self.action_dimension >= 29 else 0.0,
                "dexterous_manipulation": 0.45 if "dex" in self.variant else 0.0,
            },
            timing={"policy_action_dim": float(self.action_dimension)},
            safety_envelopes={
                "unresolved_evidence": list(self.unresolved_evidence),
                "requires_latency_profile": True,
                "requires_watchdog_profile": True,
            },
            metadata={
                "morphology_profile_id": self.profile_id,
                "variant": self.variant,
                "joint_count": self.joint_count,
                "group_counts": self.group_counts(),
                "source_refs": self.source_refs,
            },
        )

    def to_registry_entry(self, embodiment_id: str = "unitree_g1_shadow") -> EmbodimentRegistryEntry:
        return EmbodimentRegistryEntry(
            embodiment_id=embodiment_id,
            robot_id="unitree_g1",
            robot_family="unitree_g1",
            capability_profile=self.to_capability_profile(),
            observation_schema_id=f"{self.variant}_observation_v1",
            action_schema_id=f"{self.variant}_action_v1",
            translator_refs={"retarget": f"retarget://unitree/{self.variant}/shadow"},
            provenance={"morphology_profile_id": self.profile_id, **self.source_refs},
            metadata={"authority_level": "none", "hardware_calibrated": False},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "variant": self.variant,
            "joint_count": self.joint_count,
            "action_dimension": int(self.action_dimension),
            "observation_dimension": int(self.observation_dimension),
            "privileged_observation_dimension": int(self.privileged_observation_dimension),
            "joint_specs": [joint.to_dict() for joint in self.joint_specs],
            "group_counts": self.group_counts(),
            "morphology_truth_class": self.morphology_truth_class,
            "source_refs": mapping(self.source_refs),
            "unresolved_evidence": strings(self.unresolved_evidence),
            "metadata": mapping(self.metadata),
            "version": self.version,
        }


@dataclass(frozen=True)
class MorphologyEvidenceReceipt:
    receipt_id: str
    profile_id: str
    source_id: str
    evidence_kind: str
    status: str
    extracted_fields: dict[str, Any] = field(default_factory=dict)
    missing_evidence: list[str] = field(default_factory=list)
    source_refs: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    version: str = "morphology_evidence_receipt_v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "profile_id": self.profile_id,
            "source_id": self.source_id,
            "evidence_kind": self.evidence_kind,
            "status": self.status,
            "extracted_fields": mapping(self.extracted_fields),
            "missing_evidence": strings(self.missing_evidence),
            "source_refs": mapping(self.source_refs),
            "metadata": mapping(self.metadata),
            "version": self.version,
        }


def _joint_group(joint_name: str) -> str:
    if "hip" in joint_name or "knee" in joint_name or "ankle" in joint_name:
        return "legs"
    if "waist" in joint_name or "torso" in joint_name:
        return "waist"
    if "shoulder" in joint_name or "elbow" in joint_name or "wrist" in joint_name:
        return "arms"
    if "hand" in joint_name or "thumb" in joint_name or "index" in joint_name or "middle" in joint_name:
        return "hands"
    return "other"


def build_g1_morphology_profile(
    variant: str = "g1_29dof",
    *,
    observation_dimension: int = 0,
    privileged_observation_dimension: int = 0,
    source_refs: Optional[Mapping[str, Any]] = None,
) -> G1MorphologyProfile:
    joints = G1_VARIANT_JOINTS.get(variant, G1_29DOF_JOINTS)
    refs = {
        "unitree_rl_gym_g1_config": UNITREE_RL_GYM_G1_CONFIG_URL,
        "unitree_rl_lab": UNITREE_RL_LAB_URL,
        **mapping(source_refs),
    }
    profile_id = stable_id(
        "g1_morphology_profile",
        {"variant": variant, "joint_count": len(joints), "source_refs": refs},
    )
    return G1MorphologyProfile(
        profile_id=profile_id,
        variant=variant,
        joint_specs=[
            MorphologyJointSpec(joint_name=joint, group=_joint_group(joint))
            for joint in joints
        ],
        action_dimension=len(joints),
        observation_dimension=int(observation_dimension),
        privileged_observation_dimension=int(privileged_observation_dimension),
        source_refs=refs,
        unresolved_evidence=[
            "actuator_latency_profile",
            "safety_watchdog_profile",
            "hardware_joint_limit_validation",
            "sim_real_drift_measurement",
        ],
        metadata={"source_posture": "oss_pattern_not_hardware_calibration"},
    )


def _read_text(path: Path) -> str:
    try:
        return path.read_text()
    except Exception:
        return ""


def _extract_int_assignment(text: str, name: str) -> int:
    match = re.search(rf"\b{name}\s*=\s*([0-9]+)", text)
    return safe_int(match.group(1), 0) if match else 0


def _extract_list_assignment(text: str, name: str) -> list[float]:
    match = re.search(rf"\b{name}\s*=\s*\[([^\]]+)\]", text)
    if not match:
        return []
    values = []
    for raw in match.group(1).split(","):
        raw = raw.strip()
        if raw:
            values.append(safe_float(raw, 0.0))
    return values


def scan_unitree_g1_public_evidence(
    roots: Iterable[str | Path],
    *,
    variant: str = "g1_29dof",
) -> tuple[G1MorphologyProfile, list[MorphologyEvidenceReceipt]]:
    """Scan local public-repo clones for G1 morphology/config evidence.

    This scan intentionally records repository/config visibility. It does not
    treat public configs as hardware calibration.
    """
    roots = [Path(root).expanduser() for root in roots]
    observed: dict[str, Any] = {
        "g1_config_paths": [],
        "urdf_paths": [],
        "xml_paths": [],
        "usd_paths": [],
        "isaac_task_paths": [],
    }
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            lower = str(path).lower()
            if "g1" not in lower:
                continue
            if path.name == "g1_config.py":
                observed["g1_config_paths"].append(str(path))
            elif path.suffix == ".urdf":
                observed["urdf_paths"].append(str(path))
            elif path.suffix == ".xml":
                observed["xml_paths"].append(str(path))
            elif path.suffix == ".usd":
                observed["usd_paths"].append(str(path))
            elif path.suffix == ".py" and "tasks" in lower:
                observed["isaac_task_paths"].append(str(path))

    obs_dim = 0
    priv_dim = 0
    action_dim = 0
    friction_range: list[float] = []
    base_mass_range: list[float] = []
    if observed["g1_config_paths"]:
        text = _read_text(Path(observed["g1_config_paths"][0]))
        obs_dim = _extract_int_assignment(text, "num_observations")
        priv_dim = _extract_int_assignment(text, "num_privileged_obs")
        action_dim = _extract_int_assignment(text, "num_actions")
        friction_range = _extract_list_assignment(text, "friction_range")
        base_mass_range = _extract_list_assignment(text, "added_mass_range")

    detected_variant = variant
    if observed["urdf_paths"] or observed["usd_paths"]:
        all_paths = "\n".join(observed["urdf_paths"] + observed["usd_paths"])
        if "29dof" in all_paths.lower():
            detected_variant = "g1_29dof"
        if "dex3" in all_paths.lower():
            detected_variant = "g1_29dof_dex3"
    if action_dim == 12 and variant == "g1_12dof_locomotion":
        detected_variant = "g1_12dof_locomotion"

    profile = build_g1_morphology_profile(
        detected_variant,
        observation_dimension=obs_dim,
        privileged_observation_dimension=priv_dim,
        source_refs={
            "local_g1_config_paths": observed["g1_config_paths"][:5],
            "local_g1_urdf_paths": observed["urdf_paths"][:10],
            "local_g1_xml_paths": observed["xml_paths"][:10],
            "local_g1_usd_paths": observed["usd_paths"][:10],
            "nvidia_sim2real_cotraining": NVIDIA_SIM2REAL_COTRAINING_URL,
        },
    )
    receipts = [
        MorphologyEvidenceReceipt(
            receipt_id=stable_id("morphology_evidence_receipt", {"profile_id": profile.profile_id, "kind": "unitree_rl_gym_config"}),
            profile_id=profile.profile_id,
            source_id="unitree_rl_gym_g1_config",
            evidence_kind="locomotion_config",
            status="observed" if observed["g1_config_paths"] else "missing",
            extracted_fields={
                "num_observations": obs_dim,
                "num_privileged_obs": priv_dim,
                "num_actions": action_dim,
                "friction_range": friction_range,
                "added_mass_range": base_mass_range,
            },
            missing_evidence=[] if observed["g1_config_paths"] else ["g1_config.py"],
            source_refs={"paths": observed["g1_config_paths"], "public_url": UNITREE_RL_GYM_G1_CONFIG_URL},
        ),
        MorphologyEvidenceReceipt(
            receipt_id=stable_id("morphology_evidence_receipt", {"profile_id": profile.profile_id, "kind": "model_assets"}),
            profile_id=profile.profile_id,
            source_id="unitree_g1_model_assets",
            evidence_kind="morphology_asset_visibility",
            status="observed" if (observed["urdf_paths"] or observed["usd_paths"]) else "missing",
            extracted_fields={
                "urdf_count": len(observed["urdf_paths"]),
                "xml_count": len(observed["xml_paths"]),
                "usd_count": len(observed["usd_paths"]),
                "detected_variant": detected_variant,
            },
            missing_evidence=[] if (observed["urdf_paths"] or observed["usd_paths"]) else ["g1_urdf_or_usd"],
            source_refs={
                "urdf_paths": observed["urdf_paths"][:10],
                "xml_paths": observed["xml_paths"][:10],
                "usd_paths": observed["usd_paths"][:10],
                "unitree_rl_lab": UNITREE_RL_LAB_URL,
            },
        ),
        MorphologyEvidenceReceipt(
            receipt_id=stable_id("morphology_evidence_receipt", {"profile_id": profile.profile_id, "kind": "external_blockers"}),
            profile_id=profile.profile_id,
            source_id="phase3_external_evidence",
            evidence_kind="remaining_calibration_blockers",
            status="external_blocked",
            extracted_fields={"unresolved_evidence": profile.unresolved_evidence},
            missing_evidence=list(profile.unresolved_evidence),
            source_refs={"nvidia_sim2real_cotraining": NVIDIA_SIM2REAL_COTRAINING_URL},
        ),
    ]
    return profile, receipts
