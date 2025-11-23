"""
Canonical observation dataclasses used by ObservationAdapter.

All models are JSON-safe, deterministic (sorted keys), and avoid mutating
callers' inputs.
"""
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from src.utils.json_safe import to_json_safe


def _sorted_json_safe(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deterministically convert nested structures into a JSON-safe dict.
    """
    try:
        return json.loads(json.dumps(to_json_safe(payload), sort_keys=True))
    except Exception:
        # Best-effort fallback
        return to_json_safe(payload)


@dataclass
class VisionSlice:
    backend_id: str
    state_digest: str
    intrinsics: Dict[str, float]
    extrinsics: Dict[str, float]
    latent: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _sorted_json_safe(asdict(self))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VisionSlice":
        return cls(**data)


@dataclass
class SemanticSlice:
    tags: Dict[str, float]
    ood_score: Optional[float] = None
    recovery_score: Optional[float] = None
    trust_scores: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _sorted_json_safe(asdict(self))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SemanticSlice":
        return cls(**data)


@dataclass
class EconSlice:
    mpl: float
    wage_parity: float
    energy_wh: float
    damage_cost: float
    reward_scalar: float
    components: Dict[str, float] = field(default_factory=dict)
    domain_name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _sorted_json_safe(asdict(self))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EconSlice":
        return cls(**data)


@dataclass
class RecapSlice:
    advantage_bin_probs: List[float] = field(default_factory=list)
    metric_expectations: Dict[str, float] = field(default_factory=dict)
    recap_goodness_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _sorted_json_safe(asdict(self))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RecapSlice":
        return cls(**data)


@dataclass
class ControlSlice:
    curriculum_phase: Optional[str] = None
    sampler_strategy: Optional[str] = None
    objective_preset: Optional[str] = None
    task_id: Optional[str] = None
    episode_id: Optional[str] = None
    pack_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _sorted_json_safe(asdict(self))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ControlSlice":
        return cls(**data)


@dataclass
class Observation:
    vision: Optional[VisionSlice] = None
    semantics: Optional[SemanticSlice] = None
    econ: Optional[EconSlice] = None
    recap: Optional[RecapSlice] = None
    control: Optional[ControlSlice] = None
    raw_env_obs: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        if self.vision is not None:
            payload["vision"] = self.vision.to_dict()
        if self.semantics is not None:
            payload["semantics"] = self.semantics.to_dict()
        if self.econ is not None:
            payload["econ"] = self.econ.to_dict()
        if self.recap is not None:
            payload["recap"] = self.recap.to_dict()
        if self.control is not None:
            payload["control"] = self.control.to_dict()
        if self.raw_env_obs is not None:
            payload["raw_env_obs"] = to_json_safe(self.raw_env_obs)
        return _sorted_json_safe(payload)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Observation":
        pieces = dict(data)
        vision = pieces.get("vision")
        semantics = pieces.get("semantics")
        econ = pieces.get("econ")
        recap = pieces.get("recap")
        control = pieces.get("control")
        raw = pieces.get("raw_env_obs")
        return cls(
            vision=VisionSlice.from_dict(vision) if vision else None,
            semantics=SemanticSlice.from_dict(semantics) if semantics else None,
            econ=EconSlice.from_dict(econ) if econ else None,
            recap=RecapSlice.from_dict(recap) if recap else None,
            control=ControlSlice.from_dict(control) if control else None,
            raw_env_obs=raw,
        )
