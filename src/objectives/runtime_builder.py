"""Runtime builders for ObjectiveTensor-first shadow execution."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

from src.objectives.compiler import ObjectiveCompiler
from src.objectives.profile import ObjectiveProfile
from src.objectives.schema import ObjectiveTensorSchema
from src.objectives.tensor import ObjectiveTensor, objective_tensor_from_axes
from src.utils.config_digest import sha256_json


class SourceDomain(str, Enum):
    """Canonical source domains for shadow and future real-robot runtime paths."""

    PYBULLET = "pybullet"
    ISAAC = "isaac"
    SYNTHETIC = "synthetic"
    REPLAY = "replay"
    REAL_LAB = "real_lab"


@dataclass(frozen=True)
class ObjectiveRuntimeWindow:
    """A bounded telemetry slice used for pseudo-real-time pricing ticks."""

    window_id: str
    start_step: int
    end_step: int
    metrics: Dict[str, Any]
    reward_components: Dict[str, Any] = field(default_factory=dict)
    telemetry: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window_id": self.window_id,
            "start_step": int(self.start_step),
            "end_step": int(self.end_step),
            "metrics": dict(self.metrics),
            "reward_components": dict(self.reward_components),
            "telemetry": dict(self.telemetry),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ObjectiveRuntimeRecord:
    """Typed input record for ObjectiveTensor runtime construction."""

    task_id: str
    episode_id: str
    env_id: str
    world_id: str
    robot_id: str
    source_domain: SourceDomain | str
    seed: int
    run_id: str
    timestamp: str
    episode_metrics: Dict[str, Any]
    reward_components: Dict[str, Any] = field(default_factory=dict)
    telemetry: Dict[str, Any] = field(default_factory=dict)
    windows: Sequence[ObjectiveRuntimeWindow] = field(default_factory=tuple)
    policy_checkpoint: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "episode_id": self.episode_id,
            "env_id": self.env_id,
            "world_id": self.world_id,
            "robot_id": self.robot_id,
            "source_domain": str(self.source_domain),
            "seed": int(self.seed),
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "episode_metrics": dict(self.episode_metrics),
            "reward_components": dict(self.reward_components),
            "telemetry": dict(self.telemetry),
            "windows": [window.to_dict() for window in self.windows],
            "policy_checkpoint": self.policy_checkpoint,
            "context": dict(self.context),
            "provenance": dict(self.provenance),
        }


@dataclass(frozen=True)
class ObjectiveCompileResult:
    """Explicit contract-boundary scalarization artifact."""

    objective_profile_id: str
    scalarizer: str
    scalar_reward: float
    constraint_flags: list[Dict[str, Any]]
    scalarization_boundary: str
    profile_hash: str
    objective_summary: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "objective_profile_id": self.objective_profile_id,
            "scalarizer": self.scalarizer,
            "scalar_reward": float(self.scalar_reward),
            "constraint_flags": list(self.constraint_flags),
            "scalarization_boundary": self.scalarization_boundary,
            "profile_hash": self.profile_hash,
            "objective_summary": dict(self.objective_summary),
            "metadata": dict(self.metadata),
        }


def build_runtime_schema(
    *,
    throughput_capacity_per_hour: float = 24.0,
    energy_budget_wh_per_unit: float = 8.0,
) -> ObjectiveTensorSchema:
    """Create a runtime schema with explicit normalization for pricing and governance."""

    return ObjectiveTensorSchema(
        schema_id="objective_tensor_runtime_v1",
        normalization={
            "throughput": {"mode": "minmax", "min": 0.0, "max": float(throughput_capacity_per_hour)},
            "error": {"mode": "clip", "min": 0.0, "max": 1.0},
            "safety": {"mode": "clip", "min": 0.0, "max": 1.0},
            "energy": {"mode": "minmax", "min": 0.0, "max": float(energy_budget_wh_per_unit)},
        },
    )


class ObjectiveRuntimeBuilder:
    """Build runtime ObjectiveTensor artifacts from episode metrics and telemetry."""

    def __init__(
        self,
        *,
        schema: Optional[ObjectiveTensorSchema] = None,
        throughput_capacity_per_hour: float = 24.0,
        energy_budget_wh_per_unit: float = 8.0,
    ) -> None:
        self.schema = schema or build_runtime_schema(
            throughput_capacity_per_hour=throughput_capacity_per_hour,
            energy_budget_wh_per_unit=energy_budget_wh_per_unit,
        )

    def build(self, record: ObjectiveRuntimeRecord) -> ObjectiveTensor:
        axes = self._extract_axes(
            record.episode_metrics,
            reward_components=record.reward_components,
            telemetry=record.telemetry,
        )
        schema_hash = sha256_json(self.schema.to_dict())
        record_hash = sha256_json(record.to_dict())
        context = {
            "task_id": record.task_id,
            "episode_id": record.episode_id,
            "env_id": record.env_id,
            "world_id": record.world_id,
            "robot_id": record.robot_id,
            "source_domain": _source_domain_value(record.source_domain),
            "seed": int(record.seed),
            "run_id": record.run_id,
            "policy_checkpoint": record.policy_checkpoint,
            "timestamp": record.timestamp,
            "schema_version_hash": schema_hash,
            "schema_shape_signature": self.schema.shape_signature(),
        }
        context.update(dict(record.context))
        provenance = {
            "builder": "objective_runtime_builder_v1",
            "runtime_record_hash": record_hash,
            "schema_hash": schema_hash,
            "metrics_hash": sha256_json(record.episode_metrics),
            "reward_hash": sha256_json(record.reward_components),
            "telemetry_hash": sha256_json(record.telemetry),
        }
        provenance.update(dict(record.provenance))
        return objective_tensor_from_axes(
            axes,
            schema=self.schema,
            context=context,
            provenance=provenance,
        )

    def build_window_tensors(self, record: ObjectiveRuntimeRecord) -> list[Dict[str, Any]]:
        tensors: list[Dict[str, Any]] = []
        for window in record.windows:
            window_axes = self._extract_axes(
                window.metrics,
                reward_components=window.reward_components,
                telemetry=window.telemetry,
            )
            tensor = objective_tensor_from_axes(
                window_axes,
                schema=self.schema,
                context={
                    "task_id": record.task_id,
                    "episode_id": record.episode_id,
                    "env_id": record.env_id,
                    "world_id": record.world_id,
                    "robot_id": record.robot_id,
                    "source_domain": _source_domain_value(record.source_domain),
                    "seed": int(record.seed),
                    "run_id": record.run_id,
                    "window_id": window.window_id,
                    "start_step": int(window.start_step),
                    "end_step": int(window.end_step),
                    "timestamp": record.timestamp,
                },
                provenance={
                    "builder": "objective_runtime_builder_v1",
                    "window_hash": sha256_json(window.to_dict()),
                },
            )
            tensors.append(
                {
                    "window": window.to_dict(),
                    "objective_tensor": tensor.to_dict(),
                }
            )
        return tensors

    def compile_contract(
        self,
        objective_tensor: ObjectiveTensor,
        profile: ObjectiveProfile,
        *,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> ObjectiveCompileResult:
        compiler = ObjectiveCompiler(profile)
        flags = compiler.constraint_flags(objective_tensor)
        scalar_reward = compiler.scalarize(objective_tensor)
        summary = {
            axis: float(objective_tensor.mean_vector()[index])
            for index, axis in enumerate(objective_tensor.schema.axes)
        }
        return ObjectiveCompileResult(
            objective_profile_id=profile.profile_id,
            scalarizer=profile.scalarizer,
            scalar_reward=float(scalar_reward),
            constraint_flags=list(flags),
            scalarization_boundary="contract_boundary",
            profile_hash=sha256_json(profile.to_dict()),
            objective_summary=summary,
            metadata=dict(metadata or {}),
        )

    def _extract_axes(
        self,
        metrics: Mapping[str, Any],
        *,
        reward_components: Optional[Mapping[str, Any]] = None,
        telemetry: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, float]:
        reward_components = reward_components or {}
        telemetry = telemetry or {}
        throughput = _first_float(
            metrics,
            (
                "throughput",
                "throughput_units_per_hour",
                "mpl_units_per_hour",
                "items_per_hour",
                "effective_units_per_hour",
            ),
        )
        if throughput is None:
            throughput = _derive_throughput(metrics)

        error = _first_float(
            metrics,
            ("error", "error_rate", "constraint_error_rate", "failure_rate"),
        )
        if error is None:
            errors = _first_float(metrics, ("errors", "constraint_violations"), 0.0) or 0.0
            steps = max(1.0, _first_float(metrics, ("steps", "num_steps"), 1.0) or 1.0)
            error = errors / steps

        energy = _first_float(
            metrics,
            ("energy", "energy_wh_per_unit", "energy_Wh_per_unit", "energy_per_unit"),
        )
        if energy is None:
            total_energy = _first_float(metrics, ("energy_wh", "energy_Wh", "energy_total_wh"), 0.0) or 0.0
            completed = max(1.0, _first_float(metrics, ("items_completed", "units_completed"), 1.0) or 1.0)
            energy = total_energy / completed

        safety = _first_float(metrics, ("safety", "safety_score"))
        if safety is None:
            collision_rate = _first_float(metrics, ("collision_rate",), 0.0) or 0.0
            constraint_rate = _first_float(metrics, ("constraint_error_rate",), 0.0) or 0.0
            reward_safety = _first_float(reward_components, ("safety_bonus",), 0.0) or 0.0
            telemetry_trust = _first_float(telemetry, ("trust_score", "confidence"), 1.0) or 1.0
            safety = 1.0 - min(1.0, 0.65 * max(error, 0.0) + 0.20 * collision_rate + 0.15 * constraint_rate)
            safety = max(0.0, min(1.0, safety + 0.05 * reward_safety + 0.05 * telemetry_trust))

        return {
            "throughput": max(0.0, float(throughput)),
            "error": max(0.0, min(1.0, float(error))),
            "safety": max(0.0, min(1.0, float(safety))),
            "energy": max(0.0, float(energy)),
        }


def summarize_objective_tensor(objective_tensor: ObjectiveTensor) -> Dict[str, Any]:
    """Create a compact, stable summary suitable for ledgers and reports."""

    return {
        "schema_id": objective_tensor.schema.schema_id,
        "schema_hash": sha256_json(objective_tensor.schema.to_dict()),
        "axes": {
            axis: float(objective_tensor.mean_vector()[index])
            for index, axis in enumerate(objective_tensor.schema.axes)
        },
        "normalized_axes": {
            axis: float(objective_tensor.mean_vector(normalize=True)[index])
            for index, axis in enumerate(objective_tensor.schema.axes)
        },
        "context": dict(objective_tensor.context),
        "provenance": dict(objective_tensor.provenance),
    }


def _derive_throughput(metrics: Mapping[str, Any]) -> float:
    completed = _first_float(metrics, ("items_completed", "units_completed", "throughput_count"), 0.0) or 0.0
    duration_seconds = _first_float(metrics, ("duration_s", "elapsed_s", "time_s"), None)
    if duration_seconds is None:
        steps = _first_float(metrics, ("steps", "num_steps"), 1.0) or 1.0
        seconds_per_step = _first_float(metrics, ("time_step_s", "seconds_per_step"), 1.0) or 1.0
        duration_seconds = steps * seconds_per_step
    duration_hours = max(float(duration_seconds) / 3600.0, 1e-6)
    return float(completed) / duration_hours


def _first_float(
    payload: Mapping[str, Any],
    keys: Iterable[str],
    default: Optional[float] = None,
) -> Optional[float]:
    for key in keys:
        if key in payload:
            value = _safe_float(payload.get(key), default)
            if value is not None:
                return value
    return default


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return default


def _source_domain_value(source_domain: SourceDomain | str) -> str:
    if isinstance(source_domain, SourceDomain):
        return source_domain.value
    return str(source_domain)
