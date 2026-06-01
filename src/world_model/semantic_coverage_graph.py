"""Semantic coverage graph for unified task × skill × env-primitive gap analysis.

The ``SemanticCoverageGraph`` is the typed substrate that answers:

    *"What task–skill–env-primitive combinations are missing, which are
    economically meaningful, and which are ready to pursue?"*

It is built from:
  - Semantic world-model objects / relations / meta-nodes
  - Skill graph (``src.hrl.skill_graph``)
  - Env primitive inventories (``src.envs.primitive_inventory``)
  - Economic / trust / readiness signals

The graph is **read-only after construction** and exposes gap-ranking
utilities consumed by diffusion prompt compilation and simulation agenda
compilation.

Purely additive — no existing world-model code is modified.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Core types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CoverageNode:
    """Single node in the semantic coverage graph."""

    node_id: str
    node_type: str  # task | skill | env_primitive | backend | object_family |
    # risk_family | affordance_family
    label: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CoverageEdge:
    """Directed edge between coverage nodes."""

    source_id: str
    target_id: str
    edge_type: str  # requires | realizes | covers | supports
    evidence_count: int = 0
    evidence_strength: float = 0.0
    economic_priority: float = 0.0
    trust_priority: float = 0.0
    promotion_readiness: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_missing(self) -> bool:
        return self.evidence_count == 0

    def gap_score(
        self,
        *,
        economic_weight: float = 1.0,
        trust_weight: float = 1.0,
        readiness_weight: float = 1.0,
    ) -> float:
        """Composite gap urgency score (higher = more urgent to fill)."""
        if not self.is_missing:
            return 0.0
        return (
            economic_weight * self.economic_priority
            + trust_weight * self.trust_priority
            + readiness_weight * self.promotion_readiness
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_type": self.edge_type,
            "evidence_count": self.evidence_count,
            "evidence_strength": self.evidence_strength,
            "economic_priority": self.economic_priority,
            "trust_priority": self.trust_priority,
            "promotion_readiness": self.promotion_readiness,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CoverageEdge":
        return cls(
            source_id=str(payload["source_id"]),
            target_id=str(payload["target_id"]),
            edge_type=str(payload["edge_type"]),
            evidence_count=int(payload.get("evidence_count", 0)),
            evidence_strength=float(payload.get("evidence_strength", 0.0)),
            economic_priority=float(payload.get("economic_priority", 0.0)),
            trust_priority=float(payload.get("trust_priority", 0.0)),
            promotion_readiness=float(payload.get("promotion_readiness", 0.0)),
            metadata=dict(payload.get("metadata", {})),
        )


# ---------------------------------------------------------------------------
# SemanticCoverageGraph
# ---------------------------------------------------------------------------


@dataclass
class SemanticCoverageGraph:
    """Typed graph of task × skill × env-primitive coverage and gaps."""

    nodes: List[CoverageNode] = field(default_factory=list)
    edges: List[CoverageEdge] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # -- queries ----------------------------------------------------------------

    @property
    def missing_edges(self) -> List[CoverageEdge]:
        """Edges with zero evidence."""
        return [e for e in self.edges if e.is_missing]

    @property
    def covered_edges(self) -> List[CoverageEdge]:
        return [e for e in self.edges if not e.is_missing]

    def node_by_id(self, node_id: str) -> Optional[CoverageNode]:
        for n in self.nodes:
            if n.node_id == node_id:
                return n
        return None

    def nodes_by_type(self, node_type: str) -> List[CoverageNode]:
        return [n for n in self.nodes if n.node_type == node_type]

    def edges_from(self, node_id: str) -> List[CoverageEdge]:
        return [e for e in self.edges if e.source_id == node_id]

    def edges_to(self, node_id: str) -> List[CoverageEdge]:
        return [e for e in self.edges if e.target_id == node_id]

    def rank_gaps(
        self,
        *,
        economic_weight: float = 1.0,
        trust_weight: float = 1.0,
        readiness_weight: float = 1.0,
        limit: Optional[int] = None,
        gap_ranker: Any = None,
    ) -> List[CoverageEdge]:
        """Return missing edges ranked by composite gap urgency (desc).

        Parameters
        ----------
        gap_ranker : LearnedGapRanker, optional
            When provided, uses learned marginal-value predictions instead
            of the heuristic ``gap_score()``.  Falls back to heuristic
            scoring when ``None``.
        """
        gaps = [e for e in self.edges if e.is_missing]

        if gap_ranker is not None:
            # Use learned ranking
            try:
                ranked_pairs = gap_ranker.rank_edges(gaps, self)
                scored = [pair[0] for pair in ranked_pairs]
            except Exception:
                # Fall back to heuristic on any failure
                scored = sorted(
                    gaps,
                    key=lambda e: e.gap_score(
                        economic_weight=economic_weight,
                        trust_weight=trust_weight,
                        readiness_weight=readiness_weight,
                    ),
                    reverse=True,
                )
        else:
            scored = sorted(
                gaps,
                key=lambda e: e.gap_score(
                    economic_weight=economic_weight,
                    trust_weight=trust_weight,
                    readiness_weight=readiness_weight,
                ),
                reverse=True,
            )

        if limit is not None:
            return scored[:limit]
        return scored

    def coverage_summary(self) -> Dict[str, Any]:
        """Return aggregate coverage statistics."""
        total = len(self.edges)
        covered = len(self.covered_edges)
        missing = len(self.missing_edges)
        governance_blocked = sum(
            1
            for edge in self.edges
            if bool(edge.metadata.get("governance_blocked", False))
        )
        return {
            "total_edges": total,
            "covered_edges": covered,
            "missing_edges": missing,
            "coverage_ratio": covered / max(total, 1),
            "governance_blocked_edges": governance_blocked,
            "node_count": len(self.nodes),
            "node_type_counts": {
                nt: len(self.nodes_by_type(nt))
                for nt in sorted({n.node_type for n in self.nodes})
            },
        }

    # -- serialisation ----------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": [
                {
                    "node_id": n.node_id,
                    "node_type": n.node_type,
                    "label": n.label,
                    "metadata": dict(n.metadata),
                }
                for n in self.nodes
            ],
            "edges": [e.to_dict() for e in self.edges],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SemanticCoverageGraph":
        nodes = [
            CoverageNode(
                node_id=n["node_id"],
                node_type=n["node_type"],
                label=n["label"],
                metadata=dict(n.get("metadata", {})),
            )
            for n in payload.get("nodes", [])
        ]
        edges = [CoverageEdge.from_dict(e) for e in payload.get("edges", [])]
        return cls(nodes=nodes, edges=edges, metadata=dict(payload.get("metadata", {})))

    # -- builder ----------------------------------------------------------------

    @classmethod
    def build(
        cls,
        *,
        skill_graph: Any = None,
        env_inventories: Optional[Sequence[Any]] = None,
        semantic_wm: Any = None,
        economic_priorities: Optional[Mapping[Any, float]] = None,
        trust_priorities: Optional[Mapping[Any, float]] = None,
        readiness_signals: Optional[Mapping[Any, float]] = None,
        evidence_counts: Optional[Mapping[Tuple[str, str], int]] = None,
        edge_metadata: Optional[Mapping[Tuple[str, str], Mapping[str, Any]]] = None,
    ) -> "SemanticCoverageGraph":
        """Assemble a coverage graph from available sources.

        Parameters
        ----------
        skill_graph : SkillGraph, optional
            From ``src.hrl.skill_graph``.
        env_inventories : list of EnvPrimitiveInventory, optional
            From ``src.envs.primitive_inventory``.
        semantic_wm : SemanticWorldModelState, optional
            Provides object/relation/meta-node nodes.
        economic_priorities : dict mapping node_id -> priority float
        trust_priorities : dict mapping node_id -> trust float
        readiness_signals : dict mapping node_id -> readiness float
        evidence_counts : dict mapping (source_id, target_id) -> count
        """
        nodes: Dict[str, CoverageNode] = {}
        edges: List[CoverageEdge] = []
        econ = dict(economic_priorities or {})
        trust = dict(trust_priorities or {})
        readiness = dict(readiness_signals or {})
        ev_counts = dict(evidence_counts or {})
        edge_meta = {
            tuple(key): dict(value)
            for key, value in dict(edge_metadata or {}).items()
            if isinstance(key, tuple) and len(key) == 2
        }

        def _resolve_edge_signal(
            values: Mapping[Any, float], src: str, tgt: str
        ) -> float:
            edge_key = (src, tgt)
            if edge_key in values:
                try:
                    return float(values[edge_key])
                except Exception:
                    return 0.0
            if src in values:
                try:
                    return float(values[src])
                except Exception:
                    return 0.0
            if tgt in values:
                try:
                    return float(values[tgt])
                except Exception:
                    return 0.0
            return 0.0

        # -- helper to add a node if not present ------
        def _add(node_id: str, node_type: str, label: str, **kw: Any) -> None:
            if node_id not in nodes:
                nodes[node_id] = CoverageNode(
                    node_id=node_id,
                    node_type=node_type,
                    label=label,
                    metadata=kw,
                )

        # -- helper to add an edge ---------------------
        def _edge(src: str, tgt: str, etype: str, **kw: Any) -> None:
            count = ev_counts.get((src, tgt), 0)
            metadata = dict(kw.pop("metadata", {}) or {})
            metadata.update(edge_meta.get((src, tgt), {}))
            edges.append(
                CoverageEdge(
                    source_id=src,
                    target_id=tgt,
                    edge_type=etype,
                    evidence_count=count,
                    economic_priority=_resolve_edge_signal(econ, src, tgt),
                    trust_priority=_resolve_edge_signal(trust, src, tgt),
                    promotion_readiness=_resolve_edge_signal(readiness, src, tgt),
                    metadata=metadata,
                    **kw,
                )
            )

        # ── Skill-graph nodes + transition edges ────────────────────────
        if skill_graph is not None:
            for sn in getattr(skill_graph, "nodes", []):
                _add(sn.skill_id, "skill", sn.label)
                # skill → env-primitive requirements
                for prim in getattr(sn, "env_primitive_requirements", []):
                    prim_id = f"prim:{prim}"
                    _add(prim_id, "env_primitive", prim)
                    _edge(sn.skill_id, prim_id, "requires")
                # skill → object-family requirements
                for obj in getattr(sn, "object_family_requirements", []):
                    obj_id = f"obj:{obj}"
                    _add(obj_id, "object_family", obj)
                    _edge(sn.skill_id, obj_id, "requires")
                # skill → risk families
                for risk in getattr(sn, "risk_families", []):
                    risk_id = f"risk:{risk}"
                    _add(risk_id, "risk_family", risk)
                    _edge(sn.skill_id, risk_id, "requires")

            # skill transitions as "covers" edges
            for te in getattr(skill_graph, "transitions", []):
                task_id = f"task:{te.task_id}"
                _add(task_id, "task", te.task_id)
                _edge(task_id, te.from_skill, "covers")

        # ── Env primitive inventories ───────────────────────────────────
        for inv in env_inventories or []:
            backend_id = f"backend:{inv.env_id}"
            _add(backend_id, "backend", inv.env_id)
            for prim in getattr(inv, "primitives", []):
                prim_id = f"prim:{prim.primitive_id}"
                _add(prim_id, "env_primitive", prim.label)
                _edge(backend_id, prim_id, "realizes")
            for obj in getattr(inv, "object_families", []):
                obj_id = f"obj:{obj}"
                _add(obj_id, "object_family", obj)
                _edge(backend_id, obj_id, "supports")

        # ── Semantic WM meta-nodes ──────────────────────────────────────
        if semantic_wm is not None:
            for mn in getattr(semantic_wm, "meta_nodes", []):
                mn_id = f"meta:{getattr(mn, 'node_type', 'unknown')}"
                _add(mn_id, "affordance_family", getattr(mn, "node_type", "unknown"))
            for obj in getattr(semantic_wm, "objects", []):
                obj_id = f"obj:{getattr(obj, 'object_id', 'unknown')}"
                _add(obj_id, "object_family", getattr(obj, "object_id", "unknown"))

        return cls(
            nodes=list(nodes.values()),
            edges=edges,
            metadata={"builder": "SemanticCoverageGraph.build"},
        )


__all__ = [
    "CoverageNode",
    "CoverageEdge",
    "SemanticCoverageGraph",
]
