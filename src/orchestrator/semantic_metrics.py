from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
import json
import os


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
                metrics.append(SemanticMetrics(**d))
    return metrics
