"""
Deterministic resolver for skill_mode selection across sampler/curriculum/ConditionVector.

Inputs are JSON-safe, tolerant to missing fields, and never mutate callers. All
new modes are gated by the provided mode_order/default_mode so defaults stay
unchanged unless explicitly configured.
"""
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence
import hashlib


DEFAULT_MODE_ORDER = [
    "frontier_exploration",
    "safety_critical",
    "efficiency_throughput",
    "recovery_heavy",
    "default",
]

MODE_ALIASES = {
    "default": "default",
    "safety_skill": "safety_critical",
    "frontier_skill": "frontier_exploration",
    "efficiency_skill": "efficiency_throughput",
    "novelty_skill": "novelty_skill",
    "econ_priority_skill": "econ_priority_skill",
}


def _hash_to_unit(payload: str) -> float:
    digest = hashlib.sha256(str(payload).encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12)


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val)
    except Exception:
        try:
            return float(default)
        except Exception:
            return 0.0


def _normalize_tags(tags: Optional[Any]) -> Dict[str, float]:
    if tags is None:
        return {}
    if isinstance(tags, dict):
        return {str(k): _safe_float(v, 1.0) for k, v in tags.items()}
    normalized: Dict[str, float] = {}
    for tag in tags or []:
        normalized[str(tag)] = normalized.get(str(tag), 0.0) + 1.0
    return normalized


def _mean_trust(trust_matrix: Optional[Any]) -> float:
    if trust_matrix is None:
        return 0.0
    if isinstance(trust_matrix, (int, float)):
        return _safe_float(trust_matrix)
    if isinstance(trust_matrix, dict) and trust_matrix:
        vals = []
        for v in trust_matrix.values():
            if isinstance(v, (int, float)):
                vals.append(float(v))
        if vals:
            return float(sum(vals) / len(vals))
    return 0.0


@dataclass(frozen=True)
class SkillModeResolver:
    """
    Pure resolver for skill_mode with deterministic, JSON-safe inputs.

    The caller controls which modes are allowed through `mode_order`; any
    candidate not in that list falls back to `default_mode`.
    """

    default_mode: str = "efficiency_throughput"
    mode_order: Sequence[str] = tuple(DEFAULT_MODE_ORDER)

    def resolve(
        self,
        *,
        tags: Optional[Dict[str, float]] = None,
        trust_matrix: Optional[Dict[str, Any]] = None,
        curriculum_phase: Optional[str] = None,
        strategy: Optional[str] = None,
        advisory: Optional[Dict[str, Any]] = None,
        condition_vector: Optional[Any] = None,
        econ_slice: Optional[Dict[str, Any]] = None,
        recap_bucket: Optional[str] = None,
        use_condition_vector: bool = True,
    ) -> str:
        advisory = advisory or {}
        if not use_condition_vector:
            return self._coerce(self.default_mode)

        # Explicit override wins.
        if advisory.get("skill_mode"):
            return self._coerce(advisory["skill_mode"])
        if condition_vector is not None and getattr(condition_vector, "skill_mode", None):
            return self._coerce(getattr(condition_vector, "skill_mode"))

        normalized_tags = _normalize_tags(tags)
        phase = str(curriculum_phase or getattr(condition_vector, "curriculum_phase", "") or advisory.get("phase") or "warmup").lower()
        strat = str(strategy or advisory.get("strategy") or advisory.get("sampler_strategy") or "").lower()
        recap_bucket = str(
            recap_bucket
            or getattr(condition_vector, "recap_goodness_bucket", "")
            or advisory.get("recap_goodness_bucket", "")
        ).lower()

        trust_score = _mean_trust(trust_matrix)
        if advisory.get("trust_score") is not None:
            trust_score = _safe_float(advisory.get("trust_score"), trust_score)
        if econ_slice is not None and econ_slice.get("trust_score") is not None:
            trust_score = _safe_float(econ_slice.get("trust_score"), trust_score)
        if condition_vector is not None and getattr(condition_vector, "sima2_trust_score", None) is not None:
            trust_score = _safe_float(getattr(condition_vector, "sima2_trust_score"), trust_score)

        frontier_hint = bool(
            advisory.get("frontier")
            or advisory.get("is_frontier")
            or "frontier" in strat
            or phase == "frontier"
            or any("frontier" in k for k in normalized_tags.keys())
        )
        safety_hint = bool(
            advisory.get("safety_critical")
            or any("fragile" in k or "damage" in k or "safety" in k for k in normalized_tags.keys())
        )
        recovery_hint = bool(
            trust_score < 0.5
            or any("recovery" in k for k in normalized_tags.keys())
            or (econ_slice is not None and _safe_float(econ_slice.get("damage_cost")) > 0.0)
        )
        novelty_hint = bool(
            any("novel" in k for k in normalized_tags.keys())
            or recap_bucket in {"gold", "platinum"}
        )
        econ_hint = bool(
            "econ" in strat
            or advisory.get("priority") in {"high", "critical"}
        )

        if frontier_hint:
            return self._coerce("frontier_exploration")
        if safety_hint:
            return self._coerce("safety_critical")
        if econ_hint:
            candidate = "econ_priority_skill"
            coerced = self._coerce(candidate, allow_default_on_miss=True)
            if coerced:
                return coerced
        if trust_score > 0.8 and phase in {"skill_building", "frontier", "fine_tuning"}:
            return self._coerce("efficiency_throughput")
        if recovery_hint:
            return self._coerce("recovery_heavy")
        if novelty_hint:
            candidate = "novelty_skill"
            coerced = self._coerce(candidate, allow_default_on_miss=True)
            if coerced:
                return coerced
        return self._coerce(self.default_mode)

    def _coerce(self, candidate: Any, allow_default_on_miss: bool = False) -> str:
        candidate_str = MODE_ALIASES.get(str(candidate), str(candidate))
        order = list(self.mode_order) if self.mode_order else []
        if candidate_str in order:
            return candidate_str
        if self.default_mode in order:
            return self.default_mode
        if allow_default_on_miss:
            return self.default_mode
        return candidate_str


def resolve_skill_mode(**kwargs: Any) -> str:
    """
    Functional wrapper for callers that don't want to instantiate the resolver.
    """
    resolver = SkillModeResolver(
        default_mode=str(kwargs.get("default_mode", "efficiency_throughput")),
        mode_order=kwargs.get("mode_order", DEFAULT_MODE_ORDER),
    )
    kwargs.pop("default_mode", None)
    kwargs.pop("mode_order", None)
    return resolver.resolve(**kwargs)
