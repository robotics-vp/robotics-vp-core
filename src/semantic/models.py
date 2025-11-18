"""
Semantic spine models unifying semantic/econ/meta signals.
"""
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List
from time import time

from src.sima2.ontology_proposals import OntologyUpdateProposal
from src.sima2.task_graph_proposals import TaskGraphRefinementProposal
from src.sima2.tags.semantic_tags import SemanticEnrichmentProposal as SemanticTag  # alias
from src.orchestrator.meta_transformer import MetaTransformerOutputs
from src.utils.json_safe import to_json_safe


def _sorted_by_id(items, key_name: str = "proposal_id"):
    try:
        return sorted(items, key=lambda x: getattr(x, key_name, str(x)))
    except Exception:
        return list(items)


@dataclass
class EconSlice:
    task_id: str
    avg_mpl_units_per_hour: float
    avg_wage_parity: float
    avg_energy_cost: float
    avg_error_rate: float
    frontier_episodes: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return to_json_safe(asdict(self))

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EconSlice":
        return cls(**d)


@dataclass
class MetaTransformerSlice:
    task_id: str
    objective_vectors: List[Dict[str, Any]] = field(default_factory=list)
    presets: List[str] = field(default_factory=list)
    expected_deltas: Dict[str, float] = field(default_factory=dict)
    backends: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return to_json_safe(asdict(self))

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MetaTransformerSlice":
        return cls(**d)


@dataclass
class SemanticSnapshot:
    task_id: str
    ontology_proposals: List[OntologyUpdateProposal]
    task_refinements: List[TaskGraphRefinementProposal]
    semantic_tags: List[Any]
    econ_slice: EconSlice
    meta_slice: MetaTransformerSlice
    timestamp: float = field(default_factory=lambda: time())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def sorted_copy(self) -> "SemanticSnapshot":
        return SemanticSnapshot(
            task_id=self.task_id,
            ontology_proposals=_sorted_by_id(self.ontology_proposals),
            task_refinements=_sorted_by_id(self.task_refinements),
            semantic_tags=_sorted_by_id(self.semantic_tags, key_name="proposal_id"),
            econ_slice=self.econ_slice,
            meta_slice=self.meta_slice,
            timestamp=self.timestamp,
            metadata=self.metadata,
        )

    def to_dict(self) -> Dict[str, Any]:
        snap = self.sorted_copy()
        return to_json_safe(
            {
                "task_id": snap.task_id,
                "ontology_proposals": [p.to_dict() for p in snap.ontology_proposals],
                "task_refinements": [r.to_dict() for r in snap.task_refinements],
                "semantic_tags": [t.to_dict() if hasattr(t, "to_dict") else t for t in snap.semantic_tags],
                "econ_slice": snap.econ_slice.to_dict(),
                "meta_slice": snap.meta_slice.to_dict(),
                "timestamp": snap.timestamp,
                "metadata": snap.metadata,
            }
        )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SemanticSnapshot":
        return SemanticSnapshot(
            task_id=d["task_id"],
            ontology_proposals=[OntologyUpdateProposal.from_dict(p) for p in d.get("ontology_proposals", [])],
            task_refinements=[TaskGraphRefinementProposal.from_dict(r) for r in d.get("task_refinements", [])],
            semantic_tags=[t for t in d.get("semantic_tags", [])],
            econ_slice=EconSlice.from_dict(d["econ_slice"]),
            meta_slice=MetaTransformerSlice.from_dict(d["meta_slice"]),
            timestamp=d.get("timestamp", time()),
            metadata=d.get("metadata", {}),
        )
