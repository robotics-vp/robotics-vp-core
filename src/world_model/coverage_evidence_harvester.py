"""Coverage evidence harvester — extracts real evidence counts from replay data.

This module walks replay and semantic-runtime-learning data to produce
``evidence_counts`` that the ``SemanticCoverageGraph.build()`` method
consumes, plus per-edge ``economic_priority`` and ``trust_priority``
scalars derived from the broader regal/econ/trust state.

This is **Section F** of the handoff: feeding broader regal nodes into
the semantic coverage graph so it reflects observed reality, not just
the typed skill/primitive skeleton.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _safe_str(v: Any, default: str = "") -> str:
    return str(v) if v is not None else default


def _get_nested(d: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = d
    for k in keys:
        if isinstance(current, Mapping):
            current = current.get(k, default)
        else:
            return default
    return current


# ---------------------------------------------------------------------------
# Edge key helpers
# ---------------------------------------------------------------------------

def _task_to_skill_key(task_id: str, skill_id: str) -> Tuple[str, str]:
    return (f"task:{task_id}", _canonical_skill_id(skill_id))


def _skill_to_primitive_key(skill_id: str, prim_id: str) -> Tuple[str, str]:
    return (_canonical_skill_id(skill_id), f"prim:{prim_id}")


def _task_to_risk_key(task_id: str, risk_id: str) -> Tuple[str, str]:
    return (f"task:{task_id}", f"risk:{risk_id}")


def _task_to_env_key(task_id: str, env_id: str) -> Tuple[str, str]:
    return (f"task:{task_id}", f"env:{env_id}")


# ---------------------------------------------------------------------------
# HarvestResult
# ---------------------------------------------------------------------------

@dataclass
class EvidenceHarvestResult:
    """Output of the evidence harvesting pass."""

    evidence_counts: Dict[Tuple[str, str], int] = field(default_factory=dict)
    economic_priorities: Dict[Tuple[str, str], float] = field(default_factory=dict)
    trust_priorities: Dict[Tuple[str, str], float] = field(default_factory=dict)
    promotion_readiness: Dict[Tuple[str, str], float] = field(default_factory=dict)
    episodes_processed: int = 0
    rows_processed: int = 0
    edges_discovered: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        def _key_to_str(k: Tuple[str, str]) -> str:
            return f"{k[0]}||{k[1]}"
        return {
            "evidence_counts": {_key_to_str(k): v for k, v in self.evidence_counts.items()},
            "economic_priorities": {_key_to_str(k): v for k, v in self.economic_priorities.items()},
            "trust_priorities": {_key_to_str(k): v for k, v in self.trust_priorities.items()},
            "promotion_readiness": {_key_to_str(k): v for k, v in self.promotion_readiness.items()},
            "episodes_processed": self.episodes_processed,
            "rows_processed": self.rows_processed,
            "edges_discovered": self.edges_discovered,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EvidenceHarvestResult":
        def _str_to_key(s: str) -> Tuple[str, str]:
            parts = s.split("||", 1)
            return (parts[0], parts[1]) if len(parts) == 2 else (s, "")
        return cls(
            evidence_counts={_str_to_key(k): v for k, v in (payload.get("evidence_counts") or {}).items()},
            economic_priorities={_str_to_key(k): v for k, v in (payload.get("economic_priorities") or {}).items()},
            trust_priorities={_str_to_key(k): v for k, v in (payload.get("trust_priorities") or {}).items()},
            promotion_readiness={_str_to_key(k): v for k, v in (payload.get("promotion_readiness") or {}).items()},
            episodes_processed=int(payload.get("episodes_processed", 0)),
            rows_processed=int(payload.get("rows_processed", 0)),
            edges_discovered=int(payload.get("edges_discovered", 0)),
            metadata=dict(payload.get("metadata") or {}),
        )


# ---------------------------------------------------------------------------
# Harvesting functions
# ---------------------------------------------------------------------------

# Mapping from known skill modes to approximate skill IDs exercised
_SKILL_MODE_MAP: Dict[str, List[str]] = {
    "efficiency_throughput": ["locate_drawer", "grasp_handle", "open_with_clearance", "retract_safe"],
    "safety_constrained": ["locate_vase", "plan_safe_approach", "grasp_handle", "open_with_clearance", "retract_safe"],
    "hrl_full": ["locate_drawer", "locate_vase", "plan_safe_approach", "grasp_handle", "open_with_clearance", "retract_safe"],
}

_KNOWN_HRL_SKILLS = {
    skill_id
    for values in _SKILL_MODE_MAP.values()
    for skill_id in values
}

_WORKCELL_TASK_SKILL_MAP: Dict[str, List[str]] = {
    "peg_in_hole": [
        "workcell:pick_part",
        "workcell:align_peg",
        "workcell:insert_peg",
    ]
}

_WORKCELL_TOKEN_SKILL_MAP: Dict[str, str] = {
    "affordance:pick": "workcell:pick_part",
    "affordance:align": "workcell:align_peg",
    "affordance:insert": "workcell:insert_peg",
    "object:peg": "workcell:align_peg",
    "object:hole": "workcell:insert_peg",
    "peg": "workcell:align_peg",
    "hole": "workcell:insert_peg",
}

# Mapping from env_id prefixes to approximate risk families present
_ENV_RISK_MAP: Dict[str, List[str]] = {
    "drawer_vase": ["collision", "fragile_contact"],
    "dishwashing": ["slip", "splash", "thermal"],
    "workcell": ["collision", "occlusion", "misalignment"],
}


def _canonical_task_id(task_id: str) -> str:
    return _safe_str(task_id).strip().lower().replace(" ", "_")


def _normalize_env_id(env_id: str) -> str:
    canonical = _safe_str(env_id).strip().lower()
    if not canonical:
        return ""
    if "drawer_vase" in canonical or ("drawer" in canonical and "vase" in canonical):  # fixed-base curriculum source
        return "drawer_vase"
    if "dishwashing" in canonical or "dishwash" in canonical:  # fixed-base curriculum source
        return "dishwashing"
    if "workcell" in canonical:  # fixed-base curriculum source
        return "workcell"
    return canonical


def _canonical_skill_id(skill_id: str) -> str:
    normalized = _safe_str(skill_id).strip()
    if not normalized:
        return ""
    if normalized.startswith("skill:"):
        normalized = normalized.split("skill:", 1)[1]
    if normalized.startswith(("hrl:", "sima:", "vla:", "stage2:", "workcell:")):
        return normalized
    if normalized in _KNOWN_HRL_SKILLS:
        return f"hrl:{normalized}"
    return f"skill:{normalized}"


def _ordered_unique(values: Sequence[str]) -> List[str]:
    ordered: List[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = str(value)
        if normalized and normalized not in seen:
            seen.add(normalized)
            ordered.append(normalized)
    return ordered


def _infer_skills_from_row(
    row: Mapping[str, Any],
) -> List[str]:
    """Infer which skill IDs are exercised by a runtime learning row."""
    task_id = _canonical_task_id(_safe_str(row.get("task_id")))
    env_id = _normalize_env_id(_safe_str(row.get("env_id")))
    skill_mode = _safe_str(row.get("skill_mode") or row.get("metadata", {}).get("skill_mode"))
    tokens: List[str] = list(row.get("semantic_tokens") or [])
    if env_id == "workcell":
        inferred = [
            _WORKCELL_TOKEN_SKILL_MAP[str(token)]
            for token in tokens
            if str(token) in _WORKCELL_TOKEN_SKILL_MAP
        ]
        if inferred:
            return _ordered_unique(inferred)
        if task_id in _WORKCELL_TASK_SKILL_MAP:
            return list(_WORKCELL_TASK_SKILL_MAP[task_id])

    if skill_mode in _SKILL_MODE_MAP and env_id != "workcell":
        return _SKILL_MODE_MAP[skill_mode]

    # Fall back to semantic tokens that look like skill references
    skills = [t.replace("skill:", "") for t in tokens if t.startswith("skill:")]
    if skills:
        return _ordered_unique(skills)

    # Default: assume all skills for the task
    if task_id in _WORKCELL_TASK_SKILL_MAP:
        return list(_WORKCELL_TASK_SKILL_MAP[task_id])
    if "drawer" in task_id or "vase" in task_id:
        return _SKILL_MODE_MAP.get("hrl_full", [])
    return []


def _infer_risks_from_env(env_id: str) -> List[str]:
    """Infer risk families from the environment ID."""
    for prefix, risks in _ENV_RISK_MAP.items():
        if prefix in env_id:
            return risks
    return []


def harvest_evidence_counts(
    runtime_rows: Iterable[Mapping[str, Any]],
    *,
    econ_signals: Optional[Mapping[str, Any]] = None,
    trust_state: Optional[Mapping[str, Any]] = None,
    governance_traces: Optional[Sequence[Mapping[str, Any]]] = None,
    env_primitive_map: Optional[Mapping[str, List[str]]] = None,
) -> EvidenceHarvestResult:
    """Harvest evidence counts from replay / runtime learning data.

    Walks ``SemanticRuntimeLearningRow`` dicts (or ``ReplayEpisodeRecord``
    dicts) and increments edge counts for every task→skill, skill→primitive,
    and task→risk combination observed.

    Parameters
    ----------
    runtime_rows
        Iterable of row dicts. Each should have at minimum ``task_id``,
        ``env_id``, and ideally ``semantic_tokens`` and ``skill_mode``.
    econ_signals
        Optional econ signal dict. If present, ``urgency`` flows into
        per-edge ``economic_priority``.
    trust_state
        Optional trust state dict. If present, ``calibration_score``
        flows into per-edge ``trust_priority``.
    governance_traces
        Optional governance trace dicts. Presence of traces for a
        (task, node) pair increases ``promotion_readiness``.
    env_primitive_map
        Optional mapping from env_id → list of primitive IDs. If not
        provided, the function attempts to load from the registry.
    """
    counts: Dict[Tuple[str, str], int] = defaultdict(int)
    econ_pri: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    trust_pri: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    # Parse global signals once
    global_urgency = _safe_float(_get_nested(econ_signals or {}, "urgency"), 0.5)
    global_econ_weight = _safe_float(_get_nested(econ_signals or {}, "w_econ"), 0.5)
    global_trust = _safe_float(_get_nested(trust_state or {}, "calibration_score"), 0.5)

    # Parse governance trace presence
    gov_node_set: set = set()
    for trace in (governance_traces or []):
        node_id = _safe_str(trace.get("node_id"))
        if node_id:
            gov_node_set.add(node_id)

    # Load env_primitive_map from registry if not provided
    if env_primitive_map is None:
        try:
            from src.envs.primitive_inventory import list_registered_env_ids, for_env
            env_primitive_map = {}
            for eid in list_registered_env_ids():
                inv = for_env(eid)
                env_primitive_map[eid] = [p.primitive_id for p in inv.primitives]
        except Exception:
            env_primitive_map = {}

    rows_processed = 0
    for row in runtime_rows:
        rows_processed += 1
        task_id = _canonical_task_id(_safe_str(row.get("task_id")))
        env_id = _normalize_env_id(_safe_str(row.get("env_id")))
        if not task_id:
            continue

        # Extract per-row economic signal (from econ_tensor_summary if present)
        row_econ = _safe_float(
            _get_nested(row, "econ_tensor_summary", "net_value"),
            global_econ_weight,
        )

        # 1. Task → env edge
        if env_id:
            key = _task_to_env_key(task_id, env_id)
            counts[key] += 1
            econ_pri[key].append(global_urgency * row_econ)
            trust_pri[key].append(global_trust)

        # 2. Task → skill edges (from inferred skills)
        skills = _infer_skills_from_row(row)
        for skill_id in skills:
            key = _task_to_skill_key(task_id, skill_id)
            counts[key] += 1
            econ_pri[key].append(global_urgency * row_econ)
            trust_pri[key].append(global_trust)

            # 3. Skill → env primitive edges
            prims = env_primitive_map.get(env_id, [])
            for prim_id in prims:
                pkey = _skill_to_primitive_key(skill_id, prim_id)
                counts[pkey] += 1
                econ_pri[pkey].append(global_urgency * row_econ * 0.7)
                trust_pri[pkey].append(global_trust * 0.9)

        # 4. Task → risk family edges
        risks = _infer_risks_from_env(env_id)
        for risk_id in risks:
            key = _task_to_risk_key(task_id, risk_id)
            counts[key] += 1
            econ_pri[key].append(global_urgency * row_econ * 1.2)  # risk edges valued higher
            trust_pri[key].append(global_trust * 0.8)

    # Compute averages for priority scalars
    def _mean(values: List[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    economic_priorities = {k: min(1.0, _mean(v)) for k, v in econ_pri.items()}
    trust_priorities = {k: min(1.0, _mean(v)) for k, v in trust_pri.items()}

    # Promotion readiness: boosted if governance traces cover the source node
    promotion = {}
    for key in counts:
        src_node = key[0]
        base_readiness = 0.5
        if src_node in gov_node_set or src_node.split(":")[-1] in gov_node_set:
            base_readiness = 0.8
        promotion[key] = base_readiness * trust_priorities.get(key, 0.5)

    return EvidenceHarvestResult(
        evidence_counts=dict(counts),
        economic_priorities=economic_priorities,
        trust_priorities=trust_priorities,
        promotion_readiness=promotion,
        episodes_processed=rows_processed,
        rows_processed=rows_processed,
        edges_discovered=len(counts),
        metadata={
            "global_urgency": global_urgency,
            "global_trust": global_trust,
            "governance_nodes_present": len(gov_node_set),
            "env_ids_with_primitives": list(env_primitive_map.keys()),
        },
    )


__all__ = [
    "EvidenceHarvestResult",
    "harvest_evidence_counts",
]
