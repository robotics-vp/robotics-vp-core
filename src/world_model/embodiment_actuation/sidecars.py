"""Sidecar emission for Phase 3 Embodiment / Actuation WM surfaces.

These helpers materialize the additive Phase 3 state/receipt/training surfaces
beside existing advisory embodiment artifacts. They are intentionally shadow
only: no output carries runtime authority, and provider/hardware evidence stays
explicitly blocked unless real refs are supplied by the caller.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from src.embodiment.registry import EmbodimentRegistryEntry
from src.runtime.action_adapter_v2 import ActionAdapterV2
from src.runtime.observation_adapter_v2 import ObservationAdapterV2
from src.utils.json_safe import to_json_safe

from .common import clip01, mapping, safe_float, stable_id, strings
from .compiler import EmbodimentActuationCompilationResult, compile_embodiment_actuation_with_receipts
from .consumers import (
    build_economic_embodiment_receipt_bundle,
    build_perception_embodiment_feedback,
    build_runtime_adapter_validation_context,
    build_sim_embodiment_transfer_context,
)
from .morphology import G1MorphologyProfile, MorphologyEvidenceReceipt, build_g1_morphology_profile
from .neural_architectures import (
    EmbodimentNeuralArchitectureManifest,
    build_embodiment_neural_architecture_manifest,
)
from .provider_contracts import (
    EmbodimentProviderContract,
    EmbodimentRuntimeResourceSurface,
    holosoma_contract,
    isaac_contract,
    unitree_g1_contract,
)
from .training_corpus import (
    EmbodimentSeamTrainingRow,
    EmbodimentTrainingManifest,
    build_phase34_training_manifest,
    build_phase34_training_rows_from_state,
    write_phase34_training_rows_jsonl,
)

PHASE3_STATE_FILENAME_SUFFIX = "embodiment_actuation_state_v1.json"
PHASE3_RECEIPTS_FILENAME_SUFFIX = "embodiment_actuation_receipts_v1.json"
PHASE3_CONSUMERS_FILENAME_SUFFIX = "embodiment_actuation_consumers_v1.json"
PHASE3_MORPHOLOGY_FILENAME_SUFFIX = "embodiment_morphology_profile_v1.json"
PHASE3_MORPHOLOGY_RECEIPTS_FILENAME_SUFFIX = "embodiment_morphology_receipts_v1.json"
PHASE34_ROWS_FILENAME_SUFFIX = "embodiment_phase34_training_rows_v1.jsonl"
PHASE34_MANIFEST_FILENAME_SUFFIX = "embodiment_phase34_training_manifest_v1.json"
PHASE3_NEURAL_ARCHITECTURE_MANIFEST_FILENAME_SUFFIX = (
    "embodiment_neural_architecture_manifest_v1.json"
)


@dataclass(frozen=True)
class EmbodimentActuationSidecarBundle:
    """In-memory Phase 3 sidecar payloads for one episode."""

    compilation: EmbodimentActuationCompilationResult
    training_rows: list[EmbodimentSeamTrainingRow]
    training_manifest: EmbodimentTrainingManifest
    neural_architecture_manifest: EmbodimentNeuralArchitectureManifest
    consumer_payloads: dict[str, Any]
    morphology_profile: Optional[G1MorphologyProfile] = None
    morphology_receipts: list[MorphologyEvidenceReceipt] | None = None

    def summary(self) -> dict[str, Any]:
        state = self.compilation.state
        morphology = self.morphology_profile
        return {
            "schema_version": "embodiment_actuation_sidecar_summary_v1",
            "state_id": state.state_id,
            "state_version": state.version,
            "authority_level": state.authority_level,
            "compilation_mode": state.compilation_mode,
            "robot_family": state.capability.robot_family,
            "embodiment_id": state.capability.embodiment_id,
            "action_schema_id": state.action_space.schema_id,
            "action_dimension": state.action_space.dimension,
            "observation_schema_id": state.observation_interface.schema_id,
            "safety_status": state.safety_envelope.status,
            "missing_safety_evidence": list(state.safety_envelope.missing_evidence),
            "retargeting_readiness_score": state.inverse_retarget_trace.readiness_score,
            "action_feasibility_score": state.action_proposal_bundle.action_feasibility_score,
            "drift_score": state.drift_summary.drift_score,
            "receipt_count": len(self.compilation.receipts),
            "phase34_row_count": len(self.training_rows),
            "phase34_promotion_eligible": self.training_manifest.promotion_eligible,
            "phase34_blocker_reasons": list(self.training_manifest.blocker_reasons),
            "neural_architecture_count": len(self.neural_architecture_manifest.architecture_specs),
            "neural_architecture_promotion_eligible": (
                self.neural_architecture_manifest.promotion_eligible
            ),
            "neural_architecture_blocker_reasons": list(
                self.neural_architecture_manifest.blocker_reasons
            ),
            "morphology_variant": morphology.variant if morphology else "unknown",
            "morphology_joint_count": morphology.joint_count if morphology else 0,
            "morphology_truth_class": morphology.morphology_truth_class if morphology else "unavailable",
        }


@dataclass(frozen=True)
class EmbodimentActuationSidecarWriteResult:
    """Filesystem refs returned after writing Phase 3 sidecars."""

    artifact_paths: dict[str, str]
    summary: dict[str, Any]


def build_embodiment_actuation_sidecar_bundle(
    *,
    episode_id: str,
    advisory_embodiment_result: Any,
    artifact_refs: Mapping[str, Any] | None = None,
    backend_tags: Mapping[str, Any] | None = None,
    joint_state: Any = None,
    perception_shadow_surface: Any = None,
) -> EmbodimentActuationSidecarBundle:
    """Build canonical Phase 3 state, consumers, and training rows for one episode."""

    tags = mapping(backend_tags)
    refs = mapping(artifact_refs)
    morphology_profile = _resolve_morphology_profile(tags)
    morphology_receipts = _synthetic_morphology_receipts(morphology_profile) if morphology_profile else []
    registry_entry = _registry_entry(morphology_profile, tags)
    action_adapter = _action_adapter(morphology_profile, registry_entry, tags)
    observation_adapter = _observation_adapter(morphology_profile, registry_entry, tags, refs)
    provider_contracts = _provider_contracts(tags, morphology_profile)
    runtime_surface = _runtime_resource_surface(tags, provider_contracts)
    source_refs = {
        **refs,
        "morphology_profile_id": morphology_profile.profile_id if morphology_profile else "",
        "phase3_sidecar_source": "embodiment_runner",
    }

    compilation = compile_embodiment_actuation_with_receipts(
        episode_id=episode_id,
        frame_index=0,
        embodiment_registry_entry=registry_entry,
        advisory_embodiment_result=advisory_embodiment_result,
        action_adapter=action_adapter,
        observation_adapter=observation_adapter,
        perception_shadow_surface=perception_shadow_surface,
        provider_contracts=provider_contracts,
        runtime_resource_surface=runtime_surface,
        joint_state=_normalize_joint_state(joint_state, morphology_profile),
        source_refs=source_refs,
        metadata={
            **tags,
            "source_action_space": str(tags.get("source_action_space", "task_space")),
            "phase": "phase3",
            "phase_tranche": "3.1-3.5_local_sidecars",
            "authority_level": "none",
        },
    )
    rows = build_phase34_training_rows_from_state(compilation.state, compilation.receipts)
    manifest = build_phase34_training_manifest(
        rows,
        source_refs={
            "state_id": compilation.state.state_id,
            "episode_id": episode_id,
            "source": "embodiment_runner_phase3_sidecar",
        },
    )
    neural_manifest = build_embodiment_neural_architecture_manifest(
        compilation.state,
        source_refs={
            "episode_id": episode_id,
            "training_manifest_id": manifest.manifest_id,
            "source": "embodiment_runner_phase3_sidecar",
        },
    )
    consumer_payloads = {
        "schema_version": "embodiment_actuation_consumers_v1",
        "sim_synth_transfer_context": build_sim_embodiment_transfer_context(compilation.state).to_dict(),
        "perception_feedback_surface": build_perception_embodiment_feedback(compilation.state).to_dict(),
        "runtime_adapter_validation_context": build_runtime_adapter_validation_context(compilation.state).to_dict(),
        "economic_embodiment_receipt_bundle": build_economic_embodiment_receipt_bundle(
            compilation.state,
            compilation.receipts,
        ).to_dict(),
    }
    return EmbodimentActuationSidecarBundle(
        compilation=compilation,
        training_rows=rows,
        training_manifest=manifest,
        neural_architecture_manifest=neural_manifest,
        consumer_payloads=consumer_payloads,
        morphology_profile=morphology_profile,
        morphology_receipts=morphology_receipts,
    )


def write_embodiment_actuation_sidecars(
    bundle: EmbodimentActuationSidecarBundle,
    *,
    output_dir: str | Path,
    episode_id: str,
) -> EmbodimentActuationSidecarWriteResult:
    """Write Phase 3 sidecars and return artifact-path refs for metadata."""

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = _safe_stem(episode_id)
    paths = {
        "embodiment_actuation_state_path": out_dir / f"{stem}_{PHASE3_STATE_FILENAME_SUFFIX}",
        "embodiment_actuation_receipts_path": out_dir / f"{stem}_{PHASE3_RECEIPTS_FILENAME_SUFFIX}",
        "embodiment_actuation_consumers_path": out_dir / f"{stem}_{PHASE3_CONSUMERS_FILENAME_SUFFIX}",
        "embodiment_phase34_training_rows_path": out_dir / f"{stem}_{PHASE34_ROWS_FILENAME_SUFFIX}",
        "embodiment_phase34_training_manifest_path": out_dir / f"{stem}_{PHASE34_MANIFEST_FILENAME_SUFFIX}",
        "embodiment_neural_architecture_manifest_path": (
            out_dir / f"{stem}_{PHASE3_NEURAL_ARCHITECTURE_MANIFEST_FILENAME_SUFFIX}"
        ),
    }
    _write_json(paths["embodiment_actuation_state_path"], bundle.compilation.state.to_dict())
    _write_json(
        paths["embodiment_actuation_receipts_path"],
        {
            "schema_version": "embodiment_actuation_receipts_sidecar_v1",
            "state_id": bundle.compilation.state.state_id,
            "receipt_count": len(bundle.compilation.receipts),
            "authority_level": "none",
            "receipts": [receipt.to_dict() for receipt in bundle.compilation.receipts],
        },
    )
    _write_json(paths["embodiment_actuation_consumers_path"], bundle.consumer_payloads)
    write_phase34_training_rows_jsonl(bundle.training_rows, paths["embodiment_phase34_training_rows_path"])
    _write_json(paths["embodiment_phase34_training_manifest_path"], bundle.training_manifest.to_dict())
    _write_json(
        paths["embodiment_neural_architecture_manifest_path"],
        bundle.neural_architecture_manifest.to_dict(),
    )

    if bundle.morphology_profile is not None:
        morph_path = out_dir / f"{stem}_{PHASE3_MORPHOLOGY_FILENAME_SUFFIX}"
        receipts_path = out_dir / f"{stem}_{PHASE3_MORPHOLOGY_RECEIPTS_FILENAME_SUFFIX}"
        _write_json(morph_path, bundle.morphology_profile.to_dict())
        _write_json(
            receipts_path,
            {
                "schema_version": "embodiment_morphology_receipts_v1",
                "profile_id": bundle.morphology_profile.profile_id,
                "receipt_count": len(bundle.morphology_receipts or []),
                "receipts": [receipt.to_dict() for receipt in bundle.morphology_receipts or []],
            },
        )
        paths["embodiment_morphology_profile_path"] = morph_path
        paths["embodiment_morphology_receipts_path"] = receipts_path

    path_strings = {key: str(value) for key, value in paths.items()}
    summary = {
        **bundle.summary(),
        "artifact_paths": path_strings,
    }
    return EmbodimentActuationSidecarWriteResult(artifact_paths=path_strings, summary=summary)


def _safe_stem(value: str) -> str:
    safe = str(value or "episode").replace("/", "_").replace("\\", "_")
    return safe or "episode"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_json_safe(payload), indent=2, sort_keys=True))


def _resolve_morphology_profile(tags: Mapping[str, Any]) -> Optional[G1MorphologyProfile]:
    if not _looks_like_unitree_g1(tags):
        return None
    variant = str(
        tags.get("g1_variant")
        or tags.get("robot_variant")
        or tags.get("embodiment_variant")
        or tags.get("unitree_variant")
        or tags.get("variant")
        or "g1_29dof"
    )
    if variant in {"g1", "unitree_g1", "G1"}:
        variant = "g1_29dof"
    obs_dim = _optional_int(tags, "observation_dimension", "num_observations", "obs_dim")
    priv_obs_dim = _optional_int(
        tags,
        "privileged_observation_dimension",
        "num_privileged_obs",
        "priv_obs_dim",
    )
    return build_g1_morphology_profile(
        variant,
        observation_dimension=obs_dim,
        privileged_observation_dimension=priv_obs_dim,
        source_refs={"backend_tags": _compact_source_tags(tags)},
    )


def _looks_like_unitree_g1(tags: Mapping[str, Any]) -> bool:
    values = " ".join(str(value).lower() for value in tags.values())
    keys = " ".join(str(key).lower() for key in tags.keys())
    text = f"{keys} {values}"
    return "g1" in text or "unitree" in text


def _compact_source_tags(tags: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {
        "robot_family",
        "robot_variant",
        "g1_variant",
        "embodiment_variant",
        "unitree_variant",
        "task_id",
        "env_name",
        "backend",
        "provider_family",
        "num_observations",
        "num_privileged_obs",
    }
    return {str(key): to_json_safe(value) for key, value in tags.items() if key in allowed}


def _optional_int(tags: Mapping[str, Any], *keys: str) -> int:
    for key in keys:
        value = tags.get(key)
        if value is not None:
            try:
                return int(value)
            except Exception:
                continue
    return 0


def _registry_entry(
    morphology_profile: Optional[G1MorphologyProfile],
    tags: Mapping[str, Any],
) -> Optional[EmbodimentRegistryEntry]:
    if morphology_profile is None:
        return None
    return morphology_profile.to_registry_entry(
        embodiment_id=str(tags.get("embodiment_id") or "unitree_g1_shadow")
    )


def _action_adapter(
    morphology_profile: Optional[G1MorphologyProfile],
    registry_entry: Optional[EmbodimentRegistryEntry],
    tags: Mapping[str, Any],
) -> Optional[ActionAdapterV2]:
    if registry_entry is None or morphology_profile is None:
        return None
    control_hz = _first_float(tags, ("control_hz", "policy_hz", "action_hz"), 50.0)
    latency_ms = _first_float(tags, ("actuator_latency_ms", "latency_ms"), 0.0)
    return ActionAdapterV2(
        schema_id=registry_entry.action_schema_id,
        channel_order=morphology_profile.joint_names(),
        control_hz=control_hz,
        latency_ms=latency_ms,
        translator_ref=registry_entry.translator_refs.get("retarget"),
        embodiment_id=registry_entry.embodiment_id,
        bounds={},
        provenance={"source": "phase3_sidecar_morphology_profile"},
        metadata={"authority_level": "none"},
    )


def _observation_adapter(
    morphology_profile: Optional[G1MorphologyProfile],
    registry_entry: Optional[EmbodimentRegistryEntry],
    tags: Mapping[str, Any],
    artifact_refs: Mapping[str, Any],
) -> Optional[ObservationAdapterV2]:
    if registry_entry is None or morphology_profile is None:
        return None
    sample_hz = _first_float(tags, ("sample_hz", "observation_hz", "control_hz", "policy_hz"), 50.0)
    latency_ms = _first_float(tags, ("observation_latency_ms", "latency_ms"), 0.0)
    sensor_refs = ["proprio://unitree_g1", "imu://unitree_g1"]
    semantic_ref = artifact_refs.get("semantic_fusion_ref")
    if semantic_ref:
        sensor_refs.append(str(semantic_ref))
    return ObservationAdapterV2(
        schema_id=registry_entry.observation_schema_id,
        proprio_fields=[f"q_{joint}" for joint in morphology_profile.joint_names()],
        sensor_refs=sensor_refs,
        sample_hz=sample_hz,
        latency_ms=latency_ms,
        translator_ref=str(tags.get("observation_translator_ref") or "obs://unitree/g1/shadow"),
        embodiment_id=registry_entry.embodiment_id,
        provenance={"source": "phase3_sidecar_morphology_profile"},
        metadata={"authority_level": "none"},
    )


def _provider_contracts(
    tags: Mapping[str, Any],
    morphology_profile: Optional[G1MorphologyProfile],
) -> list[EmbodimentProviderContract]:
    contracts: list[EmbodimentProviderContract] = []
    if morphology_profile is not None or _looks_like_unitree_g1(tags):
        contracts.append(
            unitree_g1_contract(
                policy_ref=str(tags.get("unitree_policy_ref") or tags.get("policy_ref") or ""),
                runtime_ref=str(tags.get("unitree_runtime_ref") or tags.get("runtime_ref") or ""),
                actuator_latency_profile_ref=str(tags.get("actuator_latency_profile_ref") or ""),
                safety_watchdog_profile_ref=str(tags.get("safety_watchdog_profile_ref") or ""),
                metadata={"source": "phase3_sidecar", "authority_level": "none"},
            )
        )
    if tags.get("holosoma_policy_ref") or tags.get("holosoma_runtime_ref"):
        contracts.append(
            holosoma_contract(
                policy_ref=str(tags.get("holosoma_policy_ref") or ""),
                runtime_ref=str(tags.get("holosoma_runtime_ref") or ""),
                metadata={"source": "phase3_sidecar", "authority_level": "none"},
            )
        )
    if tags.get("isaac_runtime_ref") or tags.get("isaac_asset_ref"):
        contracts.append(
            isaac_contract(
                runtime_ref=str(tags.get("isaac_runtime_ref") or ""),
                asset_ref=str(tags.get("isaac_asset_ref") or ""),
                metadata={"source": "phase3_sidecar", "authority_level": "none"},
            )
        )
    return contracts


def _runtime_resource_surface(
    tags: Mapping[str, Any],
    provider_contracts: Sequence[EmbodimentProviderContract],
) -> EmbodimentRuntimeResourceSurface:
    battery = _first_float(tags, ("battery_fraction", "battery_state_fraction"), 0.0)
    thermal = _first_float(tags, ("thermal_margin_fraction", "thermal_headroom_fraction"), 0.0)
    latency_budget = _first_float(tags, ("latency_budget_ms", "control_latency_budget_ms"), 0.0)
    onboard_compute = _truthy(tags.get("onboard_compute_available"))
    companion_compute = _truthy(tags.get("companion_compute_available"))
    missing = [component for contract in provider_contracts for component in contract.missing_components]
    if not onboard_compute and not companion_compute:
        missing.append("compute_placement_evidence")
    if battery <= 0.0:
        missing.append("battery_state_evidence")
    if thermal <= 0.0:
        missing.append("thermal_state_evidence")
    if latency_budget <= 0.0:
        missing.append("latency_budget_evidence")
    return EmbodimentRuntimeResourceSurface(
        surface_id=stable_id(
            "embodiment_runtime_resource_surface",
            {"provider_ids": [contract.provider_id for contract in provider_contracts], "tags": _compact_source_tags(tags)},
        ),
        provider_contracts=list(provider_contracts),
        onboard_compute_available=onboard_compute,
        companion_compute_available=companion_compute,
        battery_fraction=clip01(battery),
        thermal_margin_fraction=clip01(thermal),
        latency_budget_ms=max(0.0, latency_budget),
        missing_components=sorted(set(strings(missing))),
        metadata={"source": "phase3_sidecar", "authority_level": "none"},
    )


def _normalize_joint_state(joint_state: Any, morphology_profile: Optional[G1MorphologyProfile]) -> dict[str, Any]:
    joint_names = morphology_profile.joint_names() if morphology_profile else []
    payload = _select_joint_payload(joint_state)
    if not payload:
        if not joint_names:
            return {}
        return {
            "joint_names": joint_names,
            "positions": [0.0 for _ in joint_names],
            "velocities": [0.0 for _ in joint_names],
            "metadata": {"source": "morphology_default_zero_shadow"},
        }
    if isinstance(payload, Mapping):
        named = mapping(payload)
        positions = _vector_from_keys(named, ("positions", "position", "q", "joint_positions"))
        velocities = _vector_from_keys(named, ("velocities", "velocity", "dq", "joint_velocities"))
        efforts = _vector_from_keys(named, ("efforts", "torques", "tau", "joint_efforts"))
        names = strings(named.get("joint_names") or named.get("names")) or joint_names
        if not positions and names:
            positions = [0.0 for _ in names]
        return {
            "joint_names": names,
            "positions": _fit_vector(positions, len(names)),
            "velocities": _fit_vector(velocities, len(names)),
            "efforts": _fit_vector(efforts, len(names)),
            "timestamp_s": safe_float(named.get("timestamp_s", named.get("t", 0.0))),
            "metadata": {"source": "trajectory_joint_state"},
        }
    values = _as_float_vector(payload)
    names = joint_names or [f"joint_{idx}" for idx in range(len(values))]
    return {
        "joint_names": names,
        "positions": _fit_vector(values, len(names)),
        "velocities": [0.0 for _ in names],
        "metadata": {"source": "trajectory_joint_state_sequence"},
    }


def _select_joint_payload(joint_state: Any) -> Any:
    if joint_state is None:
        return None
    if isinstance(joint_state, Mapping):
        return joint_state
    if isinstance(joint_state, Sequence) and not isinstance(joint_state, (str, bytes)):
        items = list(joint_state)
        if not items:
            return None
        for item in reversed(items):
            if isinstance(item, Mapping):
                return item
        return items[-1]
    return joint_state


def _vector_from_keys(payload: Mapping[str, Any], keys: Sequence[str]) -> list[float]:
    for key in keys:
        if key in payload:
            return _as_float_vector(payload.get(key))
    return []


def _as_float_vector(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        return [safe_float(v) for _, v in sorted(value.items())]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [safe_float(item) for item in value]
    return [safe_float(value)]


def _fit_vector(values: Sequence[float], size: int) -> list[float]:
    if size <= 0:
        return [safe_float(value) for value in values]
    fitted = [safe_float(value) for value in values[:size]]
    if len(fitted) < size:
        fitted.extend([0.0] * (size - len(fitted)))
    return fitted


def _first_float(tags: Mapping[str, Any], keys: Sequence[str], default: float) -> float:
    for key in keys:
        value = tags.get(key)
        if value is not None:
            return safe_float(value, default)
    return float(default)


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y", "available"}
    return bool(value)


def _synthetic_morphology_receipts(profile: G1MorphologyProfile) -> list[MorphologyEvidenceReceipt]:
    return [
        MorphologyEvidenceReceipt(
            receipt_id=stable_id(
                "morphology_evidence_receipt",
                {"profile_id": profile.profile_id, "kind": "runner_morphology_profile"},
            ),
            profile_id=profile.profile_id,
            source_id="embodiment_runner_g1_profile",
            evidence_kind="morphology_profile_materialized",
            status="observed",
            extracted_fields={
                "variant": profile.variant,
                "joint_count": profile.joint_count,
                "action_dimension": profile.action_dimension,
            },
            source_refs=profile.source_refs,
            metadata={"authority_level": "none", "hardware_calibrated": False},
        ),
        MorphologyEvidenceReceipt(
            receipt_id=stable_id(
                "morphology_evidence_receipt",
                {"profile_id": profile.profile_id, "kind": "runner_external_blockers"},
            ),
            profile_id=profile.profile_id,
            source_id="phase3_runner_external_evidence",
            evidence_kind="remaining_calibration_blockers",
            status="external_blocked",
            extracted_fields={"unresolved_evidence": profile.unresolved_evidence},
            missing_evidence=list(profile.unresolved_evidence),
            source_refs=profile.source_refs,
            metadata={"authority_level": "none"},
        ),
    ]


__all__ = [
    "EmbodimentActuationSidecarBundle",
    "EmbodimentActuationSidecarWriteResult",
    "PHASE34_MANIFEST_FILENAME_SUFFIX",
    "PHASE34_ROWS_FILENAME_SUFFIX",
    "PHASE3_CONSUMERS_FILENAME_SUFFIX",
    "PHASE3_MORPHOLOGY_FILENAME_SUFFIX",
    "PHASE3_MORPHOLOGY_RECEIPTS_FILENAME_SUFFIX",
    "PHASE3_NEURAL_ARCHITECTURE_MANIFEST_FILENAME_SUFFIX",
    "PHASE3_RECEIPTS_FILENAME_SUFFIX",
    "PHASE3_STATE_FILENAME_SUFFIX",
    "build_embodiment_actuation_sidecar_bundle",
    "write_embodiment_actuation_sidecars",
]
