from dataclasses import asdict, dataclass, field
import json
import os
from typing import Any, Dict, List


@dataclass
class SemanticMetrics:
    """High-level semantic health + econ-alignment metrics."""

    ontology_version: str
    task_graph_version: str

    # Drift / cohesion
    task_cluster_purity: float  # [0,1]
    concept_drift_score: float  # higher = more drift
    label_conflict_rate: float  # fraction of conflicting tags

    # Cross-module agreement
    vla_vs_sima_agreement: float  # [0,1]
    vla_vs_diffusion_agreement: float
    sim_vs_real_agreement: float

    # Econ alignment
    econ_relevant_task_fraction: (
        float  # % of semantic mass on high-MPL / high-spread regions
    )
    econ_ignored_task_fraction: float  # semantically active but econ-irrelevant

    # Coverage
    underrepresented_tasks: List[str]
    overrepresented_tasks: List[str]

    extra: Dict[str, float] = field(default_factory=dict)

    # NEW: Round-trip semantic feedback fields (v2)
    # These are computed by SemanticOrchestrator to close the econ/semantic feedback loop
    high_priority_task_fraction: float = (
        0.0  # fraction of tasks marked priority=high/critical
    )
    critical_priority_task_fraction: float = (
        0.0  # fraction of tasks marked priority=critical
    )
    fragile_object_count: int = 0  # number of fragile objects in ontology
    fragility_multiplier_active: bool = False  # whether fragility awareness is active
    safety_tag_fraction: float = 0.0  # fraction of tags related to safety
    energy_tag_fraction: float = 0.0  # fraction of tags related to energy efficiency
    novelty_tag_fraction: float = 0.0  # fraction of tags related to novelty/frontier
    semantic_drift_warnings: int = 0  # number of drift warnings emitted
    consistency_score: float = 1.0  # overall semantic consistency (0-1)

    @classmethod
    def from_raw_dict(cls, raw: Dict[str, Any]) -> "SemanticMetrics":
        """Lenient constructor that clamps fractions/agreements and defaults missing fields."""
        return cls(
            ontology_version=_coerce_str(raw.get("ontology_version")),
            task_graph_version=_coerce_str(raw.get("task_graph_version")),
            task_cluster_purity=_coerce_fraction(raw.get("task_cluster_purity")),
            concept_drift_score=_coerce_fraction(
                raw.get("concept_drift_score"), clamp_max=False
            ),
            label_conflict_rate=_coerce_fraction(raw.get("label_conflict_rate")),
            vla_vs_sima_agreement=_coerce_fraction(raw.get("vla_vs_sima_agreement")),
            vla_vs_diffusion_agreement=_coerce_fraction(
                raw.get("vla_vs_diffusion_agreement")
            ),
            sim_vs_real_agreement=_coerce_fraction(raw.get("sim_vs_real_agreement")),
            econ_relevant_task_fraction=_coerce_fraction(
                raw.get("econ_relevant_task_fraction")
            ),
            econ_ignored_task_fraction=_coerce_fraction(
                raw.get("econ_ignored_task_fraction")
            ),
            underrepresented_tasks=_coerce_str_list(raw.get("underrepresented_tasks")),
            overrepresented_tasks=_coerce_str_list(raw.get("overrepresented_tasks")),
            extra=_coerce_float_dict(raw.get("extra")),
            high_priority_task_fraction=_coerce_fraction(
                raw.get("high_priority_task_fraction")
            ),
            critical_priority_task_fraction=_coerce_fraction(
                raw.get("critical_priority_task_fraction")
            ),
            fragile_object_count=_coerce_int(raw.get("fragile_object_count")),
            fragility_multiplier_active=_coerce_bool(
                raw.get("fragility_multiplier_active")
            ),
            safety_tag_fraction=_coerce_fraction(raw.get("safety_tag_fraction")),
            energy_tag_fraction=_coerce_fraction(raw.get("energy_tag_fraction")),
            novelty_tag_fraction=_coerce_fraction(raw.get("novelty_tag_fraction")),
            semantic_drift_warnings=_coerce_int(raw.get("semantic_drift_warnings")),
            consistency_score=_coerce_fraction(raw.get("consistency_score")),
        )


@dataclass
class SemanticEconSuggestion:
    """
    Semantic-aware econ suggestion emitted by EconomicController.

    This captures the contract: econ/datapack say X -> semantic orchestrator suggests Y.
    Stored as JSONL for analysis and transformer training.
    """

    timestamp: float
    econ_context: Dict[str, float]  # Econ signals summary
    datapack_context: Dict[str, float]  # Datapack signals summary
    semantic_metrics: Dict[str, float]  # SemanticMetrics summary
    suggested_objective_adjustment: Dict[
        str, float
    ]  # e.g., {"w_safety": 1.2, "w_energy": 0.9}
    suggested_sampling_override: Dict[str, float]  # e.g., {"tag:fragile": 1.5}
    suggested_profile: str  # "SAFE", "SAVER", "BASE", "BOOST"
    rationale: str  # Human-readable explanation


def semantic_econ_suggestion_to_dict(s: SemanticEconSuggestion) -> Dict[str, Any]:
    return asdict(s)


def write_semantic_econ_suggestions(
    suggestions: List[SemanticEconSuggestion], path: str
) -> None:
    """Append semantic-aware econ suggestions to JSONL file."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "a") as f:
        for s in suggestions:
            f.write(json.dumps(semantic_econ_suggestion_to_dict(s)) + "\n")


def load_semantic_econ_suggestions(path: str) -> List[Dict[str, Any]]:
    """Load semantic-aware econ suggestions from JSONL file."""
    if not os.path.exists(path):
        return []
    suggestions = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                suggestions.append(json.loads(line))
    return suggestions


def semantic_metrics_to_dict(m: SemanticMetrics) -> Dict[str, Any]:
    return asdict(m)


def write_semantic_metrics(m: SemanticMetrics, path: str) -> None:
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(semantic_metrics_to_dict(m)) + "\n")


def load_semantic_metrics(path: str) -> List[SemanticMetrics]:
    if not os.path.exists(path):
        return []
    metrics = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                metrics.append(SemanticMetrics.from_raw_dict(d))
    return metrics


def _coerce_str(value: Any) -> str:
    return "" if value is None else str(value)


def _coerce_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _coerce_fraction(value: Any, *, clamp_max: bool = True) -> float:
    try:
        coerced = float(value)
    except (TypeError, ValueError):
        coerced = 0.0
    if coerced != coerced:
        coerced = 0.0
    coerced = max(0.0, coerced)
    if clamp_max:
        coerced = min(1.0, coerced)
    return coerced


def _coerce_str_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _coerce_float_dict(value: Any) -> Dict[str, float]:
    if not isinstance(value, dict):
        return {}
    coerced: Dict[str, float] = {}
    for key, raw_value in value.items():
        try:
            coerced[str(key)] = float(raw_value)
        except (TypeError, ValueError):
            continue
    return coerced
