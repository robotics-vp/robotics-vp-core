"""Config-driven loaders for customer objective contract profiles."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import yaml

from src.objectives.profile import ObjectiveProfile
from src.utils.config_digest import sha256_json


@dataclass(frozen=True)
class ObjectiveContractProfile:
    """Wrapper around ObjectiveProfile with shadow-runtime contract metadata."""

    profile: ObjectiveProfile
    soft_constraints: Dict[str, Dict[str, float]] = field(default_factory=dict)
    penalty_weights: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_path: Optional[str] = None
    schema_version: str = "objective_contract_profile_v1"

    @property
    def profile_hash(self) -> str:
        return sha256_json(self.to_dict())

    @property
    def profile_id(self) -> str:
        return self.profile.profile_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_path": self.source_path,
            "profile": self.profile.to_dict(),
            "soft_constraints": dict(self.soft_constraints),
            "penalty_weights": dict(self.penalty_weights),
            "metadata": dict(self.metadata),
        }


def load_contract_profile(
    identifier: str | Path,
    *,
    config_dir: str | Path = "config/contracts",
) -> ObjectiveContractProfile:
    """Load a contract profile by path or short name."""

    path = _resolve_contract_path(identifier, config_dir=config_dir)
    payload = _load_mapping(path)
    profile_payload = {
        "profile_id": payload.get("profile_id", path.stem),
        "scalarizer": payload.get("scalarizer", "weighted_sum"),
        "weights": dict(payload.get("weights", {}) or {}),
        "maximize": dict(payload.get("maximize", {}) or {}),
        "constraints": dict(payload.get("constraints", {}) or {}),
        "lexicographic_order": list(payload.get("lexicographic_order", []) or []),
        "epsilon": dict(payload.get("epsilon", {}) or {}),
        "chebyshev_target": dict(payload.get("chebyshev_target", {}) or {}),
        "penalty_weight": float(
            payload.get("penalty_weight", (payload.get("penalty_weights", {}) or {}).get("hard", 10.0))
        ),
        "metadata": dict(payload.get("metadata", {}) or {}),
    }
    return ObjectiveContractProfile(
        profile=ObjectiveProfile.from_dict(profile_payload),
        soft_constraints=_coerce_constraints(payload.get("soft_constraints", {})),
        penalty_weights={
            "hard": float((payload.get("penalty_weights", {}) or {}).get("hard", profile_payload["penalty_weight"])),
            "soft": float((payload.get("penalty_weights", {}) or {}).get("soft", 1.0)),
        },
        metadata=dict(payload.get("metadata", {}) or {}),
        source_path=str(path),
    )


def load_all_contract_profiles(
    *,
    config_dir: str | Path = "config/contracts",
) -> Dict[str, ObjectiveContractProfile]:
    """Load all contract profiles in a config directory."""

    directory = Path(config_dir)
    profiles: Dict[str, ObjectiveContractProfile] = {}
    if not directory.exists():
        return profiles
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() not in {".yaml", ".yml", ".json"}:
            continue
        profile = load_contract_profile(path, config_dir=config_dir)
        profiles[profile.profile_id] = profile
    return profiles


def _resolve_contract_path(identifier: str | Path, *, config_dir: str | Path) -> Path:
    candidate = Path(identifier)
    if candidate.exists():
        return candidate
    base = Path(config_dir)
    if candidate.suffix:
        resolved = base / candidate.name
        if resolved.exists():
            return resolved
    for suffix in (".yaml", ".yml", ".json"):
        resolved = base / f"{candidate.name}{suffix}"
        if resolved.exists():
            return resolved
    raise FileNotFoundError(f"Could not resolve contract profile '{identifier}' under {base}")


def _load_mapping(path: Path) -> Mapping[str, Any]:
    raw = path.read_text()
    if path.suffix.lower() == ".json":
        payload = json.loads(raw)
    else:
        payload = yaml.safe_load(raw)
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"Contract profile at {path} must be a mapping")
    return payload


def _coerce_constraints(payload: Any) -> Dict[str, Dict[str, float]]:
    constraints: Dict[str, Dict[str, float]] = {}
    for axis, spec in dict(payload or {}).items():
        if not isinstance(spec, Mapping):
            continue
        axis_spec: Dict[str, float] = {}
        if "min" in spec:
            axis_spec["min"] = float(spec["min"])
        if "max" in spec:
            axis_spec["max"] = float(spec["max"])
        if axis_spec:
            constraints[str(axis)] = axis_spec
    return constraints
