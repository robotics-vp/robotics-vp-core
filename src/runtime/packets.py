"""Canonical runtime and contract packets for additive middleware scaffolding."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence

from src.constraints.constraint_set import ConstraintSet
from src.economics.econ_tensor import EconTensor
from src.objectives.runtime_builder import ObjectiveRuntimeRecord
from src.objectives.tensor import ObjectiveTensor
from src.utils.config_digest import sha256_json
from src.utils.json_safe import to_json_safe


def _mapping(payload: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    return dict(to_json_safe(dict(payload or {})))


def _float_mapping(payload: Optional[Mapping[str, Any]]) -> Dict[str, float]:
    values: Dict[str, float] = {}
    for key, value in dict(payload or {}).items():
        try:
            values[str(key)] = float(value)
        except Exception:
            continue
    return values


def _source_domain_value(value: Any) -> str:
    return str(getattr(value, "value", value))


def _objective_payload(objective_tensor: ObjectiveTensor | Mapping[str, Any]) -> Dict[str, Any]:
    if isinstance(objective_tensor, ObjectiveTensor):
        return _mapping(objective_tensor.to_dict())
    return _mapping(objective_tensor)


def _econ_payload(econ_tensor: EconTensor | Mapping[str, Any]) -> Dict[str, Any]:
    if isinstance(econ_tensor, EconTensor):
        return _mapping(econ_tensor.to_dict())
    return _mapping(econ_tensor)


def _constraint_payload(constraint_set: ConstraintSet | Mapping[str, Any]) -> Dict[str, Any]:
    if isinstance(constraint_set, ConstraintSet):
        return _mapping(constraint_set.to_dict())
    return _mapping(constraint_set)


def _schema_ref_payload(schema: Any) -> "SchemaRef":
    if isinstance(schema, SchemaRef):
        return schema
    if hasattr(schema, "to_schema_ref") and callable(schema.to_schema_ref):
        resolved = schema.to_schema_ref()
        if isinstance(resolved, SchemaRef):
            return resolved
    if isinstance(schema, Mapping):
        return SchemaRef.from_dict(schema)
    raise TypeError("Expected SchemaRef-compatible object")


@dataclass(frozen=True)
class SchemaRef:
    """Canonical reference to an action, observation, or evidence schema."""

    schema_id: str
    version: str = "v1"
    shape: Dict[str, Any] = field(default_factory=dict)
    timing: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_id": self.schema_id,
            "version": self.version,
            "shape": _mapping(self.shape),
            "timing": _mapping(self.timing),
            "provenance": _mapping(self.provenance),
            "metadata": _mapping(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SchemaRef":
        return cls(
            schema_id=str(payload.get("schema_id", "")),
            version=str(payload.get("version", "v1")),
            shape=_mapping(payload.get("shape")),
            timing=_mapping(payload.get("timing")),
            provenance=_mapping(payload.get("provenance")),
            metadata=_mapping(payload.get("metadata")),
        )


@dataclass(frozen=True)
class ContractPacket:
    """Static contract surface shared across runtime, replay, and deployment."""

    contract_id: str
    task_id: str
    objective_profile_id: str
    embodiment_id: str
    source_domain: str
    observation_schema: SchemaRef
    action_schema: SchemaRef
    objective_schema_id: str
    econ_schema_id: str
    constraint_schema_id: str
    semantic_schema_id: str
    provenance: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "contract_packet_v1"

    @property
    def contract_hash(self) -> str:
        return sha256_json(self.to_dict())

    def summary(self) -> Dict[str, Any]:
        return {
            "contract_id": self.contract_id,
            "task_id": self.task_id,
            "objective_profile_id": self.objective_profile_id,
            "embodiment_id": self.embodiment_id,
            "source_domain": self.source_domain,
            "contract_hash": self.contract_hash,
            "observation_schema_id": self.observation_schema.schema_id,
            "action_schema_id": self.action_schema.schema_id,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "contract_id": self.contract_id,
            "task_id": self.task_id,
            "objective_profile_id": self.objective_profile_id,
            "embodiment_id": self.embodiment_id,
            "source_domain": self.source_domain,
            "observation_schema": self.observation_schema.to_dict(),
            "action_schema": self.action_schema.to_dict(),
            "objective_schema_id": self.objective_schema_id,
            "econ_schema_id": self.econ_schema_id,
            "constraint_schema_id": self.constraint_schema_id,
            "semantic_schema_id": self.semantic_schema_id,
            "provenance": _mapping(self.provenance),
            "metadata": _mapping(self.metadata),
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ContractPacket":
        return cls(
            contract_id=str(payload.get("contract_id", "")),
            task_id=str(payload.get("task_id", "")),
            objective_profile_id=str(payload.get("objective_profile_id", "")),
            embodiment_id=str(payload.get("embodiment_id", "")),
            source_domain=str(payload.get("source_domain", "")),
            observation_schema=SchemaRef.from_dict(payload.get("observation_schema", {}) or {}),
            action_schema=SchemaRef.from_dict(payload.get("action_schema", {}) or {}),
            objective_schema_id=str(payload.get("objective_schema_id", "")),
            econ_schema_id=str(payload.get("econ_schema_id", "")),
            constraint_schema_id=str(payload.get("constraint_schema_id", "")),
            semantic_schema_id=str(payload.get("semantic_schema_id", "")),
            provenance=_mapping(payload.get("provenance")),
            metadata=_mapping(payload.get("metadata")),
            version=str(payload.get("version", "contract_packet_v1")),
        )


@dataclass(frozen=True)
class RuntimePacket:
    """Dynamic packet binding tensors, evidence, and provenance to a contract."""

    packet_id: str
    contract: ContractPacket
    run_id: str
    episode_id: str
    timestamp: str
    objective_tensor: Dict[str, Any]
    econ_tensor: Dict[str, Any]
    constraint_set: Dict[str, Any]
    semantic_evidence: Dict[str, Any] = field(default_factory=dict)
    uncertainty: Dict[str, float] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "runtime_packet_v1"

    @classmethod
    def from_components(
        cls,
        *,
        contract: ContractPacket,
        run_id: str,
        episode_id: str,
        timestamp: str,
        objective_tensor: ObjectiveTensor | Mapping[str, Any],
        econ_tensor: EconTensor | Mapping[str, Any],
        constraint_set: ConstraintSet | Mapping[str, Any],
        semantic_evidence: Optional[Mapping[str, Any]] = None,
        uncertainty: Optional[Mapping[str, Any]] = None,
        provenance: Optional[Mapping[str, Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        packet_id: Optional[str] = None,
        version: str = "runtime_packet_v1",
    ) -> "RuntimePacket":
        resolved_run_id = str(run_id)
        resolved_episode_id = str(episode_id)
        resolved_timestamp = str(timestamp)
        resolved_objective_tensor = _objective_payload(objective_tensor)
        resolved_econ_tensor = _econ_payload(econ_tensor)
        resolved_constraint_set = _constraint_payload(constraint_set)
        resolved_semantic_evidence = _mapping(semantic_evidence)
        resolved_uncertainty = _float_mapping(uncertainty)
        resolved_provenance = _mapping(provenance)
        resolved_metadata = _mapping(metadata)
        resolved_version = str(version)
        payload: Dict[str, Any] = {
            "contract": contract.to_dict(),
            "run_id": resolved_run_id,
            "episode_id": resolved_episode_id,
            "timestamp": resolved_timestamp,
            "objective_tensor": resolved_objective_tensor,
            "econ_tensor": resolved_econ_tensor,
            "constraint_set": resolved_constraint_set,
            "semantic_evidence": resolved_semantic_evidence,
            "uncertainty": resolved_uncertainty,
            "provenance": resolved_provenance,
            "metadata": resolved_metadata,
            "version": resolved_version,
        }
        resolved_packet_id = packet_id or f"runtime_{sha256_json(payload)[:16]}"
        return cls(
            packet_id=resolved_packet_id,
            contract=contract,
            run_id=resolved_run_id,
            episode_id=resolved_episode_id,
            timestamp=resolved_timestamp,
            objective_tensor=resolved_objective_tensor,
            econ_tensor=resolved_econ_tensor,
            constraint_set=resolved_constraint_set,
            semantic_evidence=resolved_semantic_evidence,
            uncertainty=resolved_uncertainty,
            provenance=resolved_provenance,
            metadata=resolved_metadata,
            version=resolved_version,
        )

    def summary(self) -> Dict[str, Any]:
        return {
            "packet_id": self.packet_id,
            "contract_id": self.contract.contract_id,
            "task_id": self.contract.task_id,
            "objective_profile_id": self.contract.objective_profile_id,
            "embodiment_id": self.contract.embodiment_id,
            "run_id": self.run_id,
            "episode_id": self.episode_id,
            "source_domain": self.contract.source_domain,
            "uncertainty": dict(self.uncertainty),
            "packet_hash": sha256_json(self.to_dict()),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "packet_id": self.packet_id,
            "contract": self.contract.to_dict(),
            "run_id": self.run_id,
            "episode_id": self.episode_id,
            "timestamp": self.timestamp,
            "objective_tensor": _mapping(self.objective_tensor),
            "econ_tensor": _mapping(self.econ_tensor),
            "constraint_set": _mapping(self.constraint_set),
            "semantic_evidence": _mapping(self.semantic_evidence),
            "uncertainty": _float_mapping(self.uncertainty),
            "provenance": _mapping(self.provenance),
            "metadata": _mapping(self.metadata),
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RuntimePacket":
        return cls(
            packet_id=str(payload.get("packet_id", "")),
            contract=ContractPacket.from_dict(payload.get("contract", {}) or {}),
            run_id=str(payload.get("run_id", "")),
            episode_id=str(payload.get("episode_id", "")),
            timestamp=str(payload.get("timestamp", "")),
            objective_tensor=_mapping(payload.get("objective_tensor")),
            econ_tensor=_mapping(payload.get("econ_tensor")),
            constraint_set=_mapping(payload.get("constraint_set")),
            semantic_evidence=_mapping(payload.get("semantic_evidence")),
            uncertainty=_float_mapping(payload.get("uncertainty")),
            provenance=_mapping(payload.get("provenance")),
            metadata=_mapping(payload.get("metadata")),
            version=str(payload.get("version", "runtime_packet_v1")),
        )


def build_contract_packet(
    *,
    contract_id: str,
    task_id: str,
    objective_profile_id: str,
    embodiment_id: str,
    source_domain: str,
    observation_schema: Any,
    action_schema: Any,
    objective_tensor: ObjectiveTensor | Mapping[str, Any],
    econ_tensor: EconTensor | Mapping[str, Any],
    constraint_set: ConstraintSet | Mapping[str, Any],
    semantic_schema_id: str = "semantic_evidence_sidecar_v1",
    provenance: Optional[Mapping[str, Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> ContractPacket:
    objective_payload = _objective_payload(objective_tensor)
    econ_payload = _econ_payload(econ_tensor)
    constraint_payload = _constraint_payload(constraint_set)
    return ContractPacket(
        contract_id=str(contract_id),
        task_id=str(task_id),
        objective_profile_id=str(objective_profile_id),
        embodiment_id=str(embodiment_id),
        source_domain=str(source_domain),
        observation_schema=_schema_ref_payload(observation_schema),
        action_schema=_schema_ref_payload(action_schema),
        objective_schema_id=str(
            objective_payload.get("schema_id")
            or objective_payload.get("schema", {}).get("schema_id", "")
        ),
        econ_schema_id=str(econ_payload.get("schema_id", "")),
        constraint_schema_id=str(constraint_payload.get("version", "")),
        semantic_schema_id=str(semantic_schema_id),
        provenance=_mapping(provenance),
        metadata=_mapping(metadata),
    )


def runtime_packet_from_record(
    *,
    record: ObjectiveRuntimeRecord,
    contract_id: str,
    objective_profile_id: str,
    objective_tensor: ObjectiveTensor | Mapping[str, Any],
    econ_tensor: EconTensor | Mapping[str, Any],
    constraint_set: ConstraintSet | Mapping[str, Any],
    observation_schema: Any,
    action_schema: Any,
    semantic_evidence: Optional[Mapping[str, Any]] = None,
    uncertainty: Optional[Mapping[str, Any]] = None,
    provenance: Optional[Mapping[str, Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    embodiment_id: Optional[str] = None,
    semantic_schema_id: str = "semantic_evidence_sidecar_v1",
) -> RuntimePacket:
    contract = build_contract_packet(
        contract_id=contract_id,
        task_id=record.task_id,
        objective_profile_id=objective_profile_id,
        embodiment_id=embodiment_id or record.robot_id,
        source_domain=_source_domain_value(record.source_domain),
        observation_schema=observation_schema,
        action_schema=action_schema,
        objective_tensor=objective_tensor,
        econ_tensor=econ_tensor,
        constraint_set=constraint_set,
        semantic_schema_id=semantic_schema_id,
        provenance={
            "runtime_record_hash": sha256_json(record.to_dict()),
            **_mapping(provenance),
        },
        metadata=metadata,
    )
    return RuntimePacket.from_components(
        contract=contract,
        run_id=record.run_id,
        episode_id=record.episode_id,
        timestamp=record.timestamp,
        objective_tensor=objective_tensor,
        econ_tensor=econ_tensor,
        constraint_set=constraint_set,
        semantic_evidence=semantic_evidence,
        uncertainty=uncertainty,
        provenance={
            "runtime_record_hash": sha256_json(record.to_dict()),
            **_mapping(provenance),
        },
        metadata=metadata,
    )


def runtime_packet_sidecar_payload(
    *,
    run_id: str,
    packets: Sequence[RuntimePacket],
    schema_version: str = "runtime_packet_sidecar_v1",
) -> Dict[str, Any]:
    """Serialize a deterministic run-level runtime packet sidecar payload."""

    ordered_packets = sorted(
        list(packets),
        key=lambda packet: (packet.run_id, packet.episode_id, packet.packet_id),
    )
    return {
        "schema_version": str(schema_version),
        "run_id": str(run_id),
        "packet_count": len(ordered_packets),
        "episodes": [
            {
                "episode_id": packet.episode_id,
                "packet_id": packet.packet_id,
                "contract_id": packet.contract.contract_id,
                "runtime_packet": packet.to_dict(),
            }
            for packet in ordered_packets
        ],
    }


__all__ = [
    "ContractPacket",
    "RuntimePacket",
    "SchemaRef",
    "build_contract_packet",
    "runtime_packet_from_record",
    "runtime_packet_sidecar_payload",
]
