"""Repo-level skill graph unifying HRL, SIMA, VLA, and Stage 2 skill assets.

The graph represents:
  - *SkillNode*s — individual skills from any source family
  - *SkillTransitionEdge*s — observed or expected transitions between skills
  - *SkillGraph* — the aggregate graph with missing-transition detection

The coverage graph consumes this to determine which skill-edges are covered,
missing, or under-evidenced.

Purely additive — does not modify ``skills.py``, ``co_agent.py``, or any
existing HRL/SIMA code.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from src.hrl.skills import SkillID


# ---------------------------------------------------------------------------
# Core types
# ---------------------------------------------------------------------------

_WORKCELL_SKILL_SPECS = (
    {
        "skill_id": "workcell:pick_part",
        "label": "Pick Part",
        "description": "Acquire the peg/workpiece and lift it into the assembly lane.",
        "env_primitive_requirements": ["pick"],
        "object_family_requirements": ["part", "tool"],
        "risk_families": ["collision"],
        "metadata": {"task_ids": ["peg_in_hole"]},
    },
    {
        "skill_id": "workcell:align_peg",
        "label": "Align Peg",
        "description": "Align the peg relative to the target hole before insertion.",
        "env_primitive_requirements": ["align"],
        "object_family_requirements": ["part", "fixture"],
        "risk_families": ["misalignment", "occlusion"],
        "metadata": {"task_ids": ["peg_in_hole"]},
    },
    {
        "skill_id": "workcell:insert_peg",
        "label": "Insert Peg",
        "description": "Perform the bounded insertion into the target fixture.",
        "env_primitive_requirements": ["insert", "align"],
        "object_family_requirements": ["part", "fixture"],
        "risk_families": ["collision", "misalignment"],
        "metadata": {"task_ids": ["peg_in_hole"]},
    },
    {
        "skill_id": "workcell:verify_insertion",
        "label": "Verify Insertion",
        "description": "Check that the peg seated correctly and remained stable.",
        "env_primitive_requirements": [],
        "object_family_requirements": ["part", "fixture"],
        "risk_families": ["misalignment"],
        "metadata": {"task_ids": ["peg_in_hole"]},
    },
)

_WORKCELL_TASK_SEQUENCES = {
    "peg_in_hole": [
        "workcell:pick_part",
        "workcell:align_peg",
        "workcell:insert_peg",
        "workcell:verify_insertion",
    ]
}

@dataclass(frozen=True)
class SkillNode:
    """Single skill in the global graph."""

    skill_id: str
    skill_family: str  # hrl | sima | vla | stage2
    label: str
    description: str = ""
    env_primitive_requirements: List[str] = field(default_factory=list)
    object_family_requirements: List[str] = field(default_factory=list)
    risk_families: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillTransitionEdge:
    """Directed edge between two skills within a task."""

    from_skill: str
    to_skill: str
    task_id: str
    coverage_count: int = 0
    evidence_strength: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillGraph:
    """Global skill graph spanning all families."""

    nodes: List[SkillNode] = field(default_factory=list)
    transitions: List[SkillTransitionEdge] = field(default_factory=list)

    # -- queries --

    @property
    def missing_transitions(self) -> List[SkillTransitionEdge]:
        """Transitions with zero evidence."""
        return [t for t in self.transitions if t.coverage_count == 0]

    def node_by_id(self, skill_id: str) -> Optional[SkillNode]:
        for n in self.nodes:
            if n.skill_id == skill_id:
                return n
        return None

    def skill_ids(self) -> List[str]:
        return [n.skill_id for n in self.nodes]

    def edges_for_task(self, task_id: str) -> List[SkillTransitionEdge]:
        return [e for e in self.transitions if e.task_id == task_id]

    def uncovered_edges(self) -> List[SkillTransitionEdge]:
        return [e for e in self.transitions if e.coverage_count == 0]

    # -- serialisation --

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": [
                {
                    "skill_id": n.skill_id,
                    "skill_family": n.skill_family,
                    "label": n.label,
                    "description": n.description,
                    "env_primitive_requirements": list(n.env_primitive_requirements),
                    "object_family_requirements": list(n.object_family_requirements),
                    "risk_families": list(n.risk_families),
                    "metadata": dict(n.metadata),
                }
                for n in self.nodes
            ],
            "transitions": [
                {
                    "from_skill": t.from_skill,
                    "to_skill": t.to_skill,
                    "task_id": t.task_id,
                    "coverage_count": t.coverage_count,
                    "evidence_strength": t.evidence_strength,
                    "metadata": dict(t.metadata),
                }
                for t in self.transitions
            ],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SkillGraph":
        nodes = [
            SkillNode(
                skill_id=n["skill_id"],
                skill_family=n["skill_family"],
                label=n["label"],
                description=n.get("description", ""),
                env_primitive_requirements=list(n.get("env_primitive_requirements", [])),
                object_family_requirements=list(n.get("object_family_requirements", [])),
                risk_families=list(n.get("risk_families", [])),
                metadata=dict(n.get("metadata", {})),
            )
            for n in payload.get("nodes", [])
        ]
        transitions = [
            SkillTransitionEdge(
                from_skill=t["from_skill"],
                to_skill=t["to_skill"],
                task_id=t["task_id"],
                coverage_count=int(t.get("coverage_count", 0)),
                evidence_strength=float(t.get("evidence_strength", 0.0)),
                metadata=dict(t.get("metadata", {})),
            )
            for t in payload.get("transitions", [])
        ]
        return cls(nodes=nodes, transitions=transitions)

    # -- builders --

    @classmethod
    def build_from_registry(
        cls,
        *,
        hrl_skills: bool = True,
        include_workcell_skills: bool = False,
        sima_sequences: Optional[Sequence[Mapping[str, Any]]] = None,
        vla_hints: Optional[Sequence[Mapping[str, Any]]] = None,
        stage2_refinements: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> "SkillGraph":
        """Build a canonical skill graph from available skill sources.

        Parameters
        ----------
        hrl_skills : bool
            If True, include all ``SkillID`` entries from ``src.hrl.skills``.
        sima_sequences : list of dicts, optional
            Each dict should contain ``skill_ids`` (list of str) and ``task_id``.
        vla_hints : list of dicts, optional
            Each dict should have ``skill_id``, ``label``, and optional fields.
        stage2_refinements : list of dicts, optional
            Each dict should have ``skill_id``, ``label``, ``task_id``.
        """
        nodes: Dict[str, SkillNode] = {}
        transitions: List[SkillTransitionEdge] = []

        # ── HRL skills ──────────────────────────────────────────────────
        if hrl_skills:
            _PRIMITIVE_MAP = {
                SkillID.LOCATE_DRAWER: (["locate_handle"], ["drawer", "handle"]),
                SkillID.LOCATE_VASE: (["detect_fragile_obstacle"], ["vase"]),
                SkillID.PLAN_SAFE_APPROACH: (["plan_safe_approach", "collision_avoidance"], []),
                SkillID.GRASP_HANDLE: (["grasp_handle"], ["handle", "gripper"]),
                SkillID.OPEN_WITH_CLEARANCE: (["open_with_clearance", "collision_avoidance"], ["drawer", "vase"]),
                SkillID.RETRACT_SAFE: (["retract_safe"], ["gripper"]),
            }
            for sid in SkillID.all_ids():
                name = SkillID.name(sid)
                canonical_id = f"hrl:{name.lower()}"
                prims, objs = _PRIMITIVE_MAP.get(sid, ([], []))
                risk = ["collision_avoidance"] if sid in (
                    SkillID.PLAN_SAFE_APPROACH,
                    SkillID.OPEN_WITH_CLEARANCE,
                ) else []
                nodes[canonical_id] = SkillNode(
                    skill_id=canonical_id,
                    skill_family="hrl",
                    label=name,
                    description=SkillID.description(sid),
                    env_primitive_requirements=prims,
                    object_family_requirements=objs,
                    risk_families=risk,
                )
            # default fixed-base curriculum drawer-vase task transition chain
            ordered = [f"hrl:{SkillID.name(i).lower()}" for i in SkillID.all_ids()]
            for a, b in zip(ordered, ordered[1:]):
                transitions.append(SkillTransitionEdge(
                    from_skill=a, to_skill=b, task_id="drawer_vase",
                ))

        # ── Workcell skills ─────────────────────────────────────────────
        if include_workcell_skills:
            for spec in _WORKCELL_SKILL_SPECS:
                skill_id = str(spec["skill_id"])
                metadata = spec.get("metadata", {})
                nodes[skill_id] = SkillNode(
                    skill_id=skill_id,
                    skill_family="workcell",
                    label=str(spec["label"]),
                    description=str(spec.get("description", "")),
                    env_primitive_requirements=list(spec.get("env_primitive_requirements", [])),
                    object_family_requirements=list(spec.get("object_family_requirements", [])),
                    risk_families=list(spec.get("risk_families", [])),
                    metadata=dict(metadata) if isinstance(metadata, Mapping) else {},
                )
            for task_id, ordered in _WORKCELL_TASK_SEQUENCES.items():
                for a, b in zip(ordered, ordered[1:]):
                    transitions.append(
                        SkillTransitionEdge(
                            from_skill=a,
                            to_skill=b,
                            task_id=task_id,
                        )
                    )

        # ── SIMA sequences ──────────────────────────────────────────────
        for seq in (sima_sequences or []):
            task_id = str(seq.get("task_id", "sima_unknown"))
            skill_ids = list(seq.get("skill_ids", []))
            for sid in skill_ids:
                canonical = f"sima:{sid}" if not sid.startswith("sima:") else sid
                if canonical not in nodes:
                    nodes[canonical] = SkillNode(
                        skill_id=canonical,
                        skill_family="sima",
                        label=sid,
                    )
            for a, b in zip(skill_ids, skill_ids[1:]):
                ca = f"sima:{a}" if not a.startswith("sima:") else a
                cb = f"sima:{b}" if not b.startswith("sima:") else b
                transitions.append(SkillTransitionEdge(
                    from_skill=ca, to_skill=cb, task_id=task_id,
                ))

        # ── VLA semantic-action hints ────────────────────────────────────
        for hint in (vla_hints or []):
            sid = str(hint.get("skill_id", ""))
            canonical = f"vla:{sid}" if not sid.startswith("vla:") else sid
            if canonical not in nodes:
                nodes[canonical] = SkillNode(
                    skill_id=canonical,
                    skill_family="vla",
                    label=str(hint.get("label", sid)),
                    env_primitive_requirements=list(hint.get("env_primitive_requirements", [])),
                    object_family_requirements=list(hint.get("object_family_requirements", [])),
                )

        # ── Stage 2 task-refinement proposals ────────────────────────────
        for ref in (stage2_refinements or []):
            sid = str(ref.get("skill_id", ""))
            canonical = f"stage2:{sid}" if not sid.startswith("stage2:") else sid
            if canonical not in nodes:
                nodes[canonical] = SkillNode(
                    skill_id=canonical,
                    skill_family="stage2",
                    label=str(ref.get("label", sid)),
                )
            task_id = str(ref.get("task_id", "stage2_unknown"))
            # optional from/to edges
            prev = ref.get("from_skill")
            if prev:
                transitions.append(SkillTransitionEdge(
                    from_skill=str(prev), to_skill=canonical, task_id=task_id,
                ))

        return cls(nodes=list(nodes.values()), transitions=transitions)


__all__ = [
    "SkillNode",
    "SkillTransitionEdge",
    "SkillGraph",
]
