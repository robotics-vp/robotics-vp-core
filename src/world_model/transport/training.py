"""Non-training Phase-6.3 transport trainer scaffold helpers."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

from src.utils.config_digest import sha256_json
from src.utils.json_safe import to_json_safe
from src.world_model.transport.losses import WMTransportLossLedger
from src.world_model.transport.neural_manifest import (
    WMTransportNeuralArchitectureManifest,
)
from src.world_model.transport.training_rows import (
    WMTransportTrainingManifest,
    WMTransportTrainingRow,
)

WM_TRANSPORT_TRAINER_DATASET_CONTRACT_VERSION = (
    "wm_transport_trainer_dataset_contract_v1"
)
WM_TRANSPORT_MODEL_COMPONENT_CONFIG_VERSION = "wm_transport_model_component_config_v1"
WM_TRANSPORT_CPU_SMOKE_FORWARD_VERSION = "wm_transport_cpu_smoke_forward_v1"
WM_TRANSPORT_TRAINER_SCAFFOLD_MANIFEST_VERSION = (
    "wm_transport_trainer_scaffold_manifest_v1"
)

TRAINER_BLOCKERS = (
    "gpu_transport_training_not_run",
    "provider_or_hardware_transport_evidence_missing",
    "cross_wm_corpus_density_not_proven",
    "topology_latency_evaluation_not_run",
    "promotion_grade_transport_benchmark_missing",
    "weights_not_written",
)


def _mapping(payload: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    return dict(to_json_safe(dict(payload or {})))


def _unique(values: Iterable[str]) -> list[str]:
    return sorted({str(value) for value in values if str(value)})


def _float_dict(payload: Mapping[str, Any]) -> Dict[str, float]:
    values: Dict[str, float] = {}
    for key, value in dict(payload or {}).items():
        try:
            values[str(key)] = float(value)
        except Exception:
            continue
    return values


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@dataclass(frozen=True)
class WMTransportTrainerDatasetContract:
    """Dataset shape contract for a future transport trainer."""

    dataset_contract_id: str
    neural_manifest_id: str
    training_manifest_id: str
    row_count: int
    row_family_counts: Dict[str, int]
    feature_keys: list[str]
    target_keys: list[str]
    feature_dim: int
    target_dim: int
    authority_class: str = "transport_trainer_dataset_contract_only"
    ready_for_cpu_smoke_forward: bool = False
    ready_for_training: bool = False
    training_executed: bool = False
    weights_written: bool = False
    promotion_eligible: bool = False
    reward_math_mutation: bool = False
    blockers: list[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = WM_TRANSPORT_TRAINER_DATASET_CONTRACT_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_contract_id": self.dataset_contract_id,
            "version": self.version,
            "neural_manifest_id": self.neural_manifest_id,
            "training_manifest_id": self.training_manifest_id,
            "row_count": int(self.row_count),
            "row_family_counts": {
                str(key): int(value) for key, value in self.row_family_counts.items()
            },
            "feature_keys": list(self.feature_keys),
            "target_keys": list(self.target_keys),
            "shape_contracts": {
                "feature_dim": int(self.feature_dim),
                "target_dim": int(self.target_dim),
                "row_count": int(self.row_count),
            },
            "authority_class": self.authority_class,
            "ready_for_cpu_smoke_forward": bool(self.ready_for_cpu_smoke_forward),
            "ready_for_training": bool(self.ready_for_training),
            "training_executed": bool(self.training_executed),
            "weights_written": bool(self.weights_written),
            "promotion_eligible": bool(self.promotion_eligible),
            "reward_math_mutation": bool(self.reward_math_mutation),
            "blockers": list(self.blockers),
            "metadata": _mapping(self.metadata),
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "WMTransportTrainerDatasetContract":
        shapes = dict(payload.get("shape_contracts", {}) or {})
        return cls(
            dataset_contract_id=str(payload.get("dataset_contract_id", "")),
            neural_manifest_id=str(payload.get("neural_manifest_id", "")),
            training_manifest_id=str(payload.get("training_manifest_id", "")),
            row_count=int(payload.get("row_count", shapes.get("row_count", 0)) or 0),
            row_family_counts={
                str(key): int(value)
                for key, value in dict(
                    payload.get("row_family_counts", {}) or {}
                ).items()
            },
            feature_keys=[
                str(item) for item in list(payload.get("feature_keys", []) or [])
            ],
            target_keys=[
                str(item) for item in list(payload.get("target_keys", []) or [])
            ],
            feature_dim=int(shapes.get("feature_dim", 0) or 0),
            target_dim=int(shapes.get("target_dim", 0) or 0),
            authority_class=str(
                payload.get(
                    "authority_class", "transport_trainer_dataset_contract_only"
                )
            ),
            ready_for_cpu_smoke_forward=bool(
                payload.get("ready_for_cpu_smoke_forward", False)
            ),
            ready_for_training=bool(payload.get("ready_for_training", False)),
            training_executed=bool(payload.get("training_executed", False)),
            weights_written=bool(payload.get("weights_written", False)),
            promotion_eligible=bool(payload.get("promotion_eligible", False)),
            reward_math_mutation=bool(payload.get("reward_math_mutation", False)),
            blockers=[str(item) for item in list(payload.get("blockers", []) or [])],
            metadata=_mapping(payload.get("metadata")),
            version=str(
                payload.get("version", WM_TRANSPORT_TRAINER_DATASET_CONTRACT_VERSION)
            ),
        )


@dataclass(frozen=True)
class WMTransportTrainerScaffoldManifest:
    """Top-level non-training trainer scaffold manifest for Phase 6.3."""

    trainer_scaffold_id: str
    neural_manifest_id: str
    loss_ledger_id: str
    dataset_contract_id: str
    model_config_id: str
    cpu_smoke_forward_id: str
    authority_class: str = "transport_trainer_scaffold_only"
    dataset_contract_ready: bool = False
    losses_defined: bool = False
    cpu_smoke_forward_passed: bool = False
    ready_for_training: bool = False
    ready_for_gpu_training: bool = False
    training_executed: bool = False
    weights_written: bool = False
    provider_executed: bool = False
    hardware_executed: bool = False
    live_policy_control: bool = False
    reward_math_mutation: bool = False
    promotion_eligible: bool = False
    blockers: list[str] = field(default_factory=list)
    aggregate_counts: Dict[str, float] = field(default_factory=dict)
    artifact_refs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = WM_TRANSPORT_TRAINER_SCAFFOLD_MANIFEST_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trainer_scaffold_id": self.trainer_scaffold_id,
            "version": self.version,
            "neural_manifest_id": self.neural_manifest_id,
            "loss_ledger_id": self.loss_ledger_id,
            "dataset_contract_id": self.dataset_contract_id,
            "model_config_id": self.model_config_id,
            "cpu_smoke_forward_id": self.cpu_smoke_forward_id,
            "authority_class": self.authority_class,
            "dataset_contract_ready": bool(self.dataset_contract_ready),
            "losses_defined": bool(self.losses_defined),
            "cpu_smoke_forward_passed": bool(self.cpu_smoke_forward_passed),
            "ready_for_training": bool(self.ready_for_training),
            "ready_for_gpu_training": bool(self.ready_for_gpu_training),
            "training_executed": bool(self.training_executed),
            "weights_written": bool(self.weights_written),
            "provider_executed": bool(self.provider_executed),
            "hardware_executed": bool(self.hardware_executed),
            "live_policy_control": bool(self.live_policy_control),
            "reward_math_mutation": bool(self.reward_math_mutation),
            "promotion_eligible": bool(self.promotion_eligible),
            "blockers": list(self.blockers),
            "aggregate_counts": {
                str(key): float(value) for key, value in self.aggregate_counts.items()
            },
            "artifact_refs": _mapping(self.artifact_refs),
            "metadata": _mapping(self.metadata),
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "WMTransportTrainerScaffoldManifest":
        return cls(
            trainer_scaffold_id=str(payload.get("trainer_scaffold_id", "")),
            neural_manifest_id=str(payload.get("neural_manifest_id", "")),
            loss_ledger_id=str(payload.get("loss_ledger_id", "")),
            dataset_contract_id=str(payload.get("dataset_contract_id", "")),
            model_config_id=str(payload.get("model_config_id", "")),
            cpu_smoke_forward_id=str(payload.get("cpu_smoke_forward_id", "")),
            authority_class=str(
                payload.get("authority_class", "transport_trainer_scaffold_only")
            ),
            dataset_contract_ready=bool(payload.get("dataset_contract_ready", False)),
            losses_defined=bool(payload.get("losses_defined", False)),
            cpu_smoke_forward_passed=bool(
                payload.get("cpu_smoke_forward_passed", False)
            ),
            ready_for_training=bool(payload.get("ready_for_training", False)),
            ready_for_gpu_training=bool(payload.get("ready_for_gpu_training", False)),
            training_executed=bool(payload.get("training_executed", False)),
            weights_written=bool(payload.get("weights_written", False)),
            provider_executed=bool(payload.get("provider_executed", False)),
            hardware_executed=bool(payload.get("hardware_executed", False)),
            live_policy_control=bool(payload.get("live_policy_control", False)),
            reward_math_mutation=bool(payload.get("reward_math_mutation", False)),
            promotion_eligible=bool(payload.get("promotion_eligible", False)),
            blockers=[str(item) for item in list(payload.get("blockers", []) or [])],
            aggregate_counts={
                str(key): float(value)
                for key, value in dict(
                    payload.get("aggregate_counts", {}) or {}
                ).items()
            },
            artifact_refs=_mapping(payload.get("artifact_refs")),
            metadata=_mapping(payload.get("metadata")),
            version=str(
                payload.get("version", WM_TRANSPORT_TRAINER_SCAFFOLD_MANIFEST_VERSION)
            ),
        )


def build_wm_transport_trainer_dataset_contract(
    *,
    neural_manifest: WMTransportNeuralArchitectureManifest,
    training_manifest: WMTransportTrainingManifest,
    training_rows: Iterable[WMTransportTrainingRow],
    metadata: Optional[Mapping[str, Any]] = None,
) -> WMTransportTrainerDatasetContract:
    rows = list(training_rows)
    feature_keys = _unique(key for row in rows for key in row.feature_vector)
    target_keys = _unique(key for row in rows for key in row.target_vector)
    family_counts: Dict[str, int] = {}
    for row in rows:
        family_counts[row.row_family] = family_counts.get(row.row_family, 0) + 1
    payload = {
        "neural_manifest_id": neural_manifest.manifest_id,
        "training_manifest_id": training_manifest.manifest_id,
        "feature_keys": feature_keys,
        "target_keys": target_keys,
        "row_count": len(rows),
    }
    return WMTransportTrainerDatasetContract(
        dataset_contract_id=f"wm_transport_dataset_contract_{sha256_json(payload)[:16]}",
        neural_manifest_id=neural_manifest.manifest_id,
        training_manifest_id=training_manifest.manifest_id,
        row_count=len(rows),
        row_family_counts=family_counts,
        feature_keys=feature_keys,
        target_keys=target_keys,
        feature_dim=len(feature_keys),
        target_dim=len(target_keys),
        ready_for_cpu_smoke_forward=bool(rows and feature_keys and target_keys),
        blockers=list(TRAINER_BLOCKERS),
        metadata={
            "phase": "6.3_trainer_dataset_contract",
            "training_claim": False,
            **_mapping(metadata),
        },
    )


def _vector_from_keys(payload: Mapping[str, Any], keys: list[str]) -> list[float]:
    vector: list[float] = []
    for key in keys:
        try:
            vector.append(float(payload.get(key, 0.0)))
        except Exception:
            vector.append(0.0)
    return vector


def build_wm_transport_model_component_config(
    *,
    dataset_contract: WMTransportTrainerDatasetContract,
    neural_manifest: WMTransportNeuralArchitectureManifest,
) -> Dict[str, Any]:
    input_dim = max(1, dataset_contract.feature_dim)
    components = []
    for component in neural_manifest.components:
        output_dim = max(1, len(component.output_surfaces), dataset_contract.target_dim)
        hidden_base = min(256, max(16, input_dim * 2))
        components.append(
            {
                "component_key": component.component_key,
                "component_id": component.component_id,
                "model_family": component.model_family,
                "architecture_pattern": component.architecture_pattern,
                "runtime_plane": component.runtime_plane,
                "input_dim": input_dim,
                "hidden_dims": [hidden_base, max(8, hidden_base // 2)],
                "output_dim": output_dim,
                "input_surfaces": list(component.input_surfaces),
                "output_surfaces": list(component.output_surfaces),
                "loss_families": list(component.loss_families),
                "training_enabled": False,
                "weights_initialized": False,
                "weights_written": False,
                "promotion_eligible": False,
                "authority_class": "transport_model_component_config_only",
            }
        )
    payload = {
        "version": WM_TRANSPORT_MODEL_COMPONENT_CONFIG_VERSION,
        "model_config_id": "",
        "dataset_contract_id": dataset_contract.dataset_contract_id,
        "neural_manifest_id": neural_manifest.manifest_id,
        "component_count": len(components),
        "components": components,
        "training_executed": False,
        "weights_initialized": False,
        "weights_written": False,
        "ready_for_gpu_training": False,
        "promotion_eligible": False,
        "reward_math_mutation": False,
        "blockers": list(TRAINER_BLOCKERS),
    }
    payload["model_config_id"] = (
        f"wm_transport_model_config_{sha256_json(payload)[:16]}"
    )
    return payload


def _component_forward(
    component: Mapping[str, Any], vector: list[float]
) -> list[float]:
    output_dim = int(component.get("output_dim", 1) or 1)
    component_key = str(component.get("component_key", "component"))
    scale = (sum(ord(char) for char in component_key) % 29 + 5) / 100.0
    weighted_sum = sum((idx + 1) * value for idx, value in enumerate(vector))
    denom = max(1.0, float(len(vector)))
    return [weighted_sum * scale / (denom * (idx + 1)) for idx in range(output_dim)]


def _loss_smoke_value(*, loss_key: str, rows: list[WMTransportTrainingRow]) -> float:
    if not rows:
        return 0.0
    values = []
    for row in rows:
        features = _float_dict(row.feature_vector)
        targets = _float_dict(row.target_vector)
        feature_mean = sum(features.values()) / max(1, len(features))
        target_mean = sum(targets.values()) / max(1, len(targets))
        if "uncertainty" in loss_key:
            values.append(
                abs(1.0 - targets.get("target_calibration_score", target_mean))
            )
        elif "topology" in loss_key:
            values.append(
                abs(1.0 - targets.get("target_topology_preservation", target_mean))
            )
        elif "receiver" in loss_key or "actionability" in loss_key:
            values.append(
                abs(1.0 - targets.get("target_receiver_actionability", target_mean))
            )
        elif (
            "yield" in loss_key or "counterfactual" in loss_key or "ranking" in loss_key
        ):
            values.append(
                abs(
                    targets.get("target_downstream_yield_proxy", target_mean)
                    - feature_mean
                )
            )
        else:
            values.append(abs(target_mean - feature_mean))
    return sum(values) / max(1, len(values))


def build_wm_transport_cpu_smoke_forward_report(
    *,
    dataset_contract: WMTransportTrainerDatasetContract,
    model_config: Mapping[str, Any],
    loss_ledger: WMTransportLossLedger,
    training_rows: Iterable[WMTransportTrainingRow],
) -> Dict[str, Any]:
    rows = list(training_rows)
    vector = (
        _vector_from_keys(rows[0].feature_vector, dataset_contract.feature_keys)
        if rows
        else [0.0]
    ) or [0.0]
    component_reports = []
    finite = True
    for component in list(model_config.get("components", []) or []):
        output = _component_forward(component, vector)
        component_finite = all(math.isfinite(value) for value in output)
        shape_passed = len(vector) == int(component.get("input_dim", 0)) and len(
            output
        ) == int(component.get("output_dim", 0))
        finite = finite and component_finite and shape_passed
        component_reports.append(
            {
                "component_key": component["component_key"],
                "input_dim": int(component["input_dim"]),
                "output_dim": int(component["output_dim"]),
                "observed_input_dim": len(vector),
                "observed_output_dim": len(output),
                "output_sample": output[: min(4, len(output))],
                "shape_check_passed": shape_passed,
                "finite_check_passed": component_finite,
            }
        )
    loss_reports = []
    for definition in loss_ledger.definitions:
        smoke_value = _loss_smoke_value(loss_key=definition.loss_key, rows=rows)
        loss_reports.append(
            {
                "loss_key": definition.loss_key,
                "smoke_value": smoke_value,
                "default_weight": definition.default_weight,
                "finite_check_passed": math.isfinite(smoke_value),
                "direct_policy_rl": definition.direct_policy_rl,
                "uses_rl_style_signal": definition.uses_rl_style_signal,
            }
        )
        finite = finite and math.isfinite(smoke_value)
    payload = {
        "version": WM_TRANSPORT_CPU_SMOKE_FORWARD_VERSION,
        "cpu_smoke_forward_id": "",
        "dataset_contract_id": dataset_contract.dataset_contract_id,
        "model_config_id": model_config["model_config_id"],
        "loss_ledger_id": loss_ledger.ledger_id,
        "component_reports": component_reports,
        "loss_reports": loss_reports,
        "input_vector_dim": len(vector),
        "component_count": len(component_reports),
        "loss_count": len(loss_reports),
        "cpu_smoke_forward_passed": bool(component_reports and loss_reports and finite),
        "training_executed": False,
        "weights_initialized": False,
        "weights_written": False,
        "provider_executed": False,
        "hardware_executed": False,
        "live_policy_control": False,
        "ready_for_gpu_training": False,
        "promotion_eligible": False,
        "reward_math_mutation": False,
        "blockers": list(TRAINER_BLOCKERS),
    }
    payload["cpu_smoke_forward_id"] = (
        f"wm_transport_cpu_smoke_{sha256_json(payload)[:16]}"
    )
    return payload


def build_wm_transport_trainer_scaffold_manifest(
    *,
    neural_manifest: WMTransportNeuralArchitectureManifest,
    loss_ledger: WMTransportLossLedger,
    dataset_contract: WMTransportTrainerDatasetContract,
    model_config: Mapping[str, Any],
    cpu_smoke_forward: Mapping[str, Any],
    artifact_refs: Mapping[str, Any],
    metadata: Optional[Mapping[str, Any]] = None,
) -> WMTransportTrainerScaffoldManifest:
    payload = {
        "neural_manifest_id": neural_manifest.manifest_id,
        "loss_ledger_id": loss_ledger.ledger_id,
        "dataset_contract_id": dataset_contract.dataset_contract_id,
        "model_config_id": model_config["model_config_id"],
        "cpu_smoke_forward_id": cpu_smoke_forward["cpu_smoke_forward_id"],
    }
    return WMTransportTrainerScaffoldManifest(
        trainer_scaffold_id=f"wm_transport_trainer_scaffold_{sha256_json(payload)[:16]}",
        neural_manifest_id=neural_manifest.manifest_id,
        loss_ledger_id=loss_ledger.ledger_id,
        dataset_contract_id=dataset_contract.dataset_contract_id,
        model_config_id=str(model_config["model_config_id"]),
        cpu_smoke_forward_id=str(cpu_smoke_forward["cpu_smoke_forward_id"]),
        dataset_contract_ready=dataset_contract.ready_for_cpu_smoke_forward,
        losses_defined=bool(loss_ledger.definitions),
        cpu_smoke_forward_passed=bool(
            cpu_smoke_forward.get("cpu_smoke_forward_passed")
        ),
        blockers=list(TRAINER_BLOCKERS),
        aggregate_counts={
            "component_count": float(model_config.get("component_count", 0.0)),
            "loss_count": float(loss_ledger.loss_count),
            "training_row_count": float(dataset_contract.row_count),
        },
        artifact_refs=_mapping(artifact_refs),
        metadata={
            "boundary": "phase6.3 non-training trainer scaffold only",
            "cpu_smoke_only": True,
            "transport_is_policy": False,
            **_mapping(metadata),
        },
    )


def save_wm_transport_trainer_dataset_contract(
    path: str | Path, dataset_contract: WMTransportTrainerDatasetContract
) -> None:
    _write_json(path, dataset_contract.to_dict())


def save_wm_transport_trainer_scaffold_manifest(
    path: str | Path, manifest: WMTransportTrainerScaffoldManifest
) -> None:
    _write_json(path, manifest.to_dict())


def load_wm_transport_trainer_dataset_contract(
    path: str | Path,
) -> WMTransportTrainerDatasetContract:
    return WMTransportTrainerDatasetContract.from_dict(_load_json(path))


def load_wm_transport_trainer_scaffold_manifest(
    path: str | Path,
) -> WMTransportTrainerScaffoldManifest:
    return WMTransportTrainerScaffoldManifest.from_dict(_load_json(path))


__all__ = [
    "TRAINER_BLOCKERS",
    "WM_TRANSPORT_CPU_SMOKE_FORWARD_VERSION",
    "WM_TRANSPORT_MODEL_COMPONENT_CONFIG_VERSION",
    "WM_TRANSPORT_TRAINER_DATASET_CONTRACT_VERSION",
    "WM_TRANSPORT_TRAINER_SCAFFOLD_MANIFEST_VERSION",
    "WMTransportTrainerDatasetContract",
    "WMTransportTrainerScaffoldManifest",
    "build_wm_transport_cpu_smoke_forward_report",
    "build_wm_transport_model_component_config",
    "build_wm_transport_trainer_dataset_contract",
    "build_wm_transport_trainer_scaffold_manifest",
    "load_wm_transport_trainer_dataset_contract",
    "load_wm_transport_trainer_scaffold_manifest",
    "save_wm_transport_trainer_dataset_contract",
    "save_wm_transport_trainer_scaffold_manifest",
]
