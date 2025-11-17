from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional
import json
import os
import time


@dataclass
class SemanticMetrics:
    """High-level semantic health + econ-alignment metrics."""
    ontology_version: str
    task_graph_version: str

    # Drift / cohesion
    task_cluster_purity: float      # [0,1]
    concept_drift_score: float      # higher = more drift
    label_conflict_rate: float      # fraction of conflicting tags

    # Cross-module agreement
    vla_vs_sima_agreement: float    # [0,1]
    vla_vs_diffusion_agreement: float
    sim_vs_real_agreement: float

    # Econ alignment
    econ_relevant_task_fraction: float   # % of semantic mass on high-MPL / high-spread regions
    econ_ignored_task_fraction: float    # semantically active but econ-irrelevant

    # Coverage
    underrepresented_tasks: List[str]
    overrepresented_tasks: List[str]

    extra: Dict[str, float] = None

    # NEW: Round-trip semantic feedback fields (v2)
    # These are computed by SemanticOrchestrator to close the econ/semantic feedback loop
    high_priority_task_fraction: float = 0.0  # fraction of tasks marked priority=high/critical
    critical_priority_task_fraction: float = 0.0  # fraction of tasks marked priority=critical
    fragile_object_count: int = 0  # number of fragile objects in ontology
    fragility_multiplier_active: bool = False  # whether fragility awareness is active
    safety_tag_fraction: float = 0.0  # fraction of tags related to safety
    energy_tag_fraction: float = 0.0  # fraction of tags related to energy efficiency
    novelty_tag_fraction: float = 0.0  # fraction of tags related to novelty/frontier
    semantic_drift_warnings: int = 0  # number of drift warnings emitted
    consistency_score: float = 1.0  # overall semantic consistency (0-1)

    @classmethod
    def from_raw_dict(cls, raw: Dict[str, any]) -> "SemanticMetrics":
        """Lenient constructor that clamps fractions/agreements and defaults missing fields."""
        clamped = {}
        frac_fields = [
            "task_cluster_purity",
            "concept_drift_score",
            "label_conflict_rate",
            "vla_vs_sima_agreement",
            "vla_vs_diffusion_agreement",
            "sim_vs_real_agreement",
            "econ_relevant_task_fraction",
            "econ_ignored_task_fraction",
            "high_priority_task_fraction",
            "critical_priority_task_fraction",
            "safety_tag_fraction",
            "energy_tag_fraction",
            "novelty_tag_fraction",
            "consistency_score",
        ]
        for field in cls.__dataclass_fields__.keys():
            val = raw.get(field)
            if field in frac_fields:
                try:
                    val = float(val)
                    if val != val:  # NaN check
                        val = 0.0
                except Exception:
                    val = 0.0
                val = max(0.0, val)
                if field != "concept_drift_score":
                    val = min(1.0, val)
            if val is None:
                if field in ["underrepresented_tasks", "overrepresented_tasks"]:
                    val = []
                elif field == "extra":
                    val = {}
                elif field in frac_fields:
                    val = 0.0
            clamped[field] = val
        return cls(**clamped)


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
    suggested_objective_adjustment: Dict[str, float]  # e.g., {"w_safety": 1.2, "w_energy": 0.9}
    suggested_sampling_override: Dict[str, float]  # e.g., {"tag:fragile": 1.5}
    suggested_profile: str  # "SAFE", "SAVER", "BASE", "BOOST"
    rationale: str  # Human-readable explanation


def semantic_econ_suggestion_to_dict(s: SemanticEconSuggestion) -> Dict[str, any]:
    return asdict(s)


def write_semantic_econ_suggestions(suggestions: List[SemanticEconSuggestion], path: str) -> None:
    """Append semantic-aware econ suggestions to JSONL file."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "a") as f:
        for s in suggestions:
            f.write(json.dumps(semantic_econ_suggestion_to_dict(s)) + "\n")


def load_semantic_econ_suggestions(path: str) -> List[Dict[str, any]]:
    """Load semantic-aware econ suggestions from JSONL file."""
    if not os.path.exists(path):
        return []
    suggestions = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                suggestions.append(json.loads(line))
    return suggestions


def semantic_metrics_to_dict(m: SemanticMetrics) -> Dict[str, any]:
    return asdict(m)


def write_semantic_metrics(m: SemanticMetrics, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
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
