"""
OntologyUpdateEngine: Advisory ontology proposal generator.

Consumes SemanticPrimitives and produces OntologyUpdateProposals without
mutating ontology or reward logic. All outputs are JSON-safe and respect
economic/datapack/task-graph constraints.
"""

import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.sima2.semantic_primitive_extractor import SemanticPrimitive
from src.sima2.ontology_proposals import (
    OntologyUpdateProposal,
    ProposalPriority,
    ProposalType,
)
from src.orchestrator.ontology import AffordanceType, EnvironmentOntology, ObjectCategory
from src.orchestrator.task_graph import TaskGraph
from src.orchestrator.economic_controller import EconSignals
from src.orchestrator.datapack_engine import DatapackSignals


class OntologyUpdateEngine:
    """
    Generates ontology update proposals from semantic primitives.

    Advisory-only; does not mutate ontology state.
    """

    def __init__(
        self,
        ontology: EnvironmentOntology,
        task_graph: Optional[TaskGraph] = None,
        econ_signals: Optional[EconSignals] = None,
        datapack_signals: Optional[DatapackSignals] = None,
    ):
        """
        Initialize OntologyUpdateEngine.

        Args:
            ontology: Current ontology (read-only)
            task_graph: Current task graph (read-only)
            econ_signals: Economic constraints (optional)
            datapack_signals: Data constraints (optional)
        """
        self.ontology = ontology
        self.task_graph = task_graph
        self.econ_signals = econ_signals or EconSignals()
        self.datapack_signals = datapack_signals or DatapackSignals()

        # Proposal generation state (for determinism)
        self._proposal_counter = 0

    def generate_proposals(
        self, primitives: List[SemanticPrimitive]
    ) -> List[OntologyUpdateProposal]:
        """
        Generate ontology update proposals from semantic primitives.

        Args:
            primitives: List of SemanticPrimitives from Stage 2.1

        Returns:
            List of OntologyUpdateProposals (advisory-only)
        """
        proposals: List[OntologyUpdateProposal] = []

        for prim in primitives:
            # 1. Affordance proposals
            proposals.extend(self._propose_affordances(prim))

            # 2. Risk adjustment proposals
            proposals.extend(self._propose_risk_adjustments(prim))

            # 3. Fragility inference proposals
            proposals.extend(self._propose_fragility_inference(prim))

            # 4. Skill gating proposals
            proposals.extend(self._propose_skill_gates(prim))

            # 5. Energy heuristic proposals
            proposals.extend(self._propose_energy_heuristics(prim))

            # 6. Semantic tag proposals
            proposals.extend(self._propose_semantic_tags(prim))

        return proposals

    def _make_proposal_id(self) -> str:
        """Generate unique proposal ID."""
        self._proposal_counter += 1
        return f"prop_{self._proposal_counter:06d}_{uuid.uuid4().hex[:6]}"

    def _propose_affordances(
        self, prim: SemanticPrimitive
    ) -> List[OntologyUpdateProposal]:
        """
        Propose new affordances based on primitive actions.

        Rules:
        - "grasp" tags → graspable affordance
        - "lift" tags → liftable affordance
        - "pull" tags → pullable affordance
        - etc.
        """
        proposals: List[OntologyUpdateProposal] = []

        # Map tags to affordances
        tag_to_affordance = {
            "grasp": AffordanceType.GRASPABLE,
            "lift": AffordanceType.LIFTABLE,
            "pull": AffordanceType.PULLABLE,
            "push": AffordanceType.PUSHABLE,
            "place": AffordanceType.PLACEABLE,
            "open": AffordanceType.OPENABLE,
        }

        for tag, aff_type in tag_to_affordance.items():
            if tag in prim.tags:
                proposal = OntologyUpdateProposal(
                    proposal_id=self._make_proposal_id(),
                    proposal_type=ProposalType.ADD_AFFORDANCE,
                    priority=ProposalPriority.MEDIUM,
                    source_primitive_id=prim.primitive_id,
                    source=prim.source,
                    target_affordance_type=aff_type.value,
                    proposed_changes={
                        "affordance_type": aff_type.value,
                        "confidence": prim.success_rate,
                        "activation_skill_ids": [],
                        "energy_cost_estimate": prim.energy_intensity,
                        "risk_level": self._risk_level_to_float(prim.risk_level),
                    },
                    rationale=f"Primitive '{prim.primitive_id}' demonstrated '{tag}' action",
                    confidence=prim.success_rate,
                    tags=prim.tags,
                )

                proposals.append(proposal)

        return proposals

    def _propose_risk_adjustments(
        self, prim: SemanticPrimitive
    ) -> List[OntologyUpdateProposal]:
        """
        Propose risk level adjustments based on primitive risk.

        Rules:
        - High risk primitives → elevate affordance risk
        - Econ error_urgency > 0.5 → further risk elevation
        """
        proposals: List[OntologyUpdateProposal] = []

        if prim.risk_level in {"medium", "high"}:
            risk_multiplier = 1.0
            if self.econ_signals.error_urgency > 0.5:
                risk_multiplier = 1.5
            if self.econ_signals.error_urgency > 0.7:
                risk_multiplier = 2.0

            old_risk = self._risk_level_to_float(prim.risk_level)
            new_risk = min(1.0, old_risk * risk_multiplier)

            if new_risk > old_risk:
                proposal = OntologyUpdateProposal(
                    proposal_id=self._make_proposal_id(),
                    proposal_type=ProposalType.ADJUST_RISK,
                    priority=(
                        ProposalPriority.CRITICAL
                        if self.econ_signals.error_urgency > 0.7
                        else ProposalPriority.HIGH
                    ),
                    source_primitive_id=prim.primitive_id,
                    source=prim.source,
                    proposed_changes={
                        "old_risk_level": old_risk,
                        "new_risk_level": new_risk,
                        "adjustment_factor": risk_multiplier,
                        "trigger": f"error_urgency={self.econ_signals.error_urgency:.2f}",
                    },
                    rationale=(
                        f"Primitive risk '{prim.risk_level}' + "
                        f"error urgency {self.econ_signals.error_urgency:.2f} "
                        f"→ elevate risk to {new_risk:.2f}"
                    ),
                    confidence=0.8,
                    tags=prim.tags + ["risk_adjustment"],
                )

                proposals.append(proposal)

        return proposals

    def _propose_fragility_inference(
        self, prim: SemanticPrimitive
    ) -> List[OntologyUpdateProposal]:
        """
        Infer object fragility from primitive tags.

        Rules:
        - "fragile" tag → high fragility (0.9)
        - "glass" tag → high fragility (0.85)
        - High damage cost from econ → medium fragility (0.6)
        """
        proposals: List[OntologyUpdateProposal] = []

        fragility_rules: Dict[str, float] = {
            "fragile": 0.9,
            "glass": 0.85,
            "delicate": 0.8,
            "vase": 0.7,
        }

        for tag, fragility in fragility_rules.items():
            if tag in prim.tags:
                if self.econ_signals.damage_cost_total > 50.0:
                    fragility = min(1.0, fragility * 1.1)

                proposal = OntologyUpdateProposal(
                    proposal_id=self._make_proposal_id(),
                    proposal_type=ProposalType.INFER_FRAGILITY,
                    priority=ProposalPriority.CRITICAL,
                    source_primitive_id=prim.primitive_id,
                    source=prim.source,
                    proposed_changes={
                        "inferred_fragility": fragility,
                        "evidence": [tag, f"task_type={prim.task_type}"],
                        "damage_cost_estimate": self.econ_signals.damage_cost_per_error,
                    },
                    rationale=f"Tag '{tag}' indicates fragility {fragility:.2f}",
                    confidence=0.85,
                    tags=prim.tags + ["fragility_inference"],
                )

                proposals.append(proposal)

        return proposals

    def _propose_skill_gates(
        self, prim: SemanticPrimitive
    ) -> List[OntologyUpdateProposal]:
        """
        Propose skill gating based on risk/safety.

        Rules:
        - High-risk primitives → gate with safety check
        - "fragile" tags → gate pull/move/lift skills
        """
        proposals: List[OntologyUpdateProposal] = []

        if prim.risk_level == "high" or "fragile" in prim.tags:
            gated_skill_ids = [2, 5, 7]

            for skill_id in gated_skill_ids:
                proposal = OntologyUpdateProposal(
                    proposal_id=self._make_proposal_id(),
                    proposal_type=ProposalType.ADD_SKILL_GATE,
                    priority=ProposalPriority.HIGH,
                    source_primitive_id=prim.primitive_id,
                    source=prim.source,
                    target_skill_id=skill_id,
                    proposed_changes={
                        "gated_skill_id": skill_id,
                        "preconditions": [
                            "fragility_check_passed",
                            "collision_avoidance_active",
                        ],
                        "safety_threshold": 0.8,
                        "fallback_skill_id": 0,
                    },
                    rationale=(
                        f"Skill {skill_id} involves high-risk action "
                        f"near fragile object (risk={prim.risk_level})"
                    ),
                    confidence=0.9,
                    tags=prim.tags + ["skill_gating", "safety"],
                )

                proposals.append(proposal)

        return proposals

    def _propose_energy_heuristics(
        self, prim: SemanticPrimitive
    ) -> List[OntologyUpdateProposal]:
        """
        Propose energy efficiency heuristics.

        Rules:
        - Low energy_intensity primitives → prefer_efficient_path heuristic
        - High energy_urgency → boost energy-saving heuristics
        """
        proposals: List[OntologyUpdateProposal] = []

        if prim.energy_intensity < 0.5 and self.econ_signals.energy_urgency > 0.3:
            proposal = OntologyUpdateProposal(
                proposal_id=self._make_proposal_id(),
                proposal_type=ProposalType.ADD_ENERGY_HEURISTIC,
                priority=ProposalPriority.MEDIUM,
                source_primitive_id=prim.primitive_id,
                source=prim.source,
                proposed_changes={
                    "heuristic_type": "prefer_efficient_path",
                    "energy_multiplier": 0.8,
                    "conditions": [
                        "short_reach",
                        f"energy_intensity<{prim.energy_intensity}",
                    ],
                },
                rationale=(
                    f"Primitive demonstrates low energy use ({prim.energy_intensity:.2f}); "
                    f"energy urgency={self.econ_signals.energy_urgency:.2f}"
                ),
                confidence=0.7,
                tags=prim.tags + ["energy_optimization"],
            )

            proposals.append(proposal)

        return proposals

    def _propose_semantic_tags(
        self, prim: SemanticPrimitive
    ) -> List[OntologyUpdateProposal]:
        """
        Propose semantic tag additions for consistency.

        Rules:
        - Unify tags across primitives (e.g., "fragile" vs "delicate")
        - Propagate safety tags to related skills
        """
        proposals: List[OntologyUpdateProposal] = []

        tag_unification: Dict[tuple, str] = {
            ("fragile", "delicate"): "fragile_glassware",
            ("drawer", "open"): "drawer_manipulation",
        }

        for tag_set, unified_tag in tag_unification.items():
            if all(t in prim.tags for t in tag_set):
                proposal = OntologyUpdateProposal(
                    proposal_id=self._make_proposal_id(),
                    proposal_type=ProposalType.ADD_SEMANTIC_TAG,
                    priority=ProposalPriority.LOW,
                    source_primitive_id=prim.primitive_id,
                    source=prim.source,
                    proposed_changes={
                        "tag": unified_tag,
                        "applies_to_objects": [],
                        "applies_to_skills": [],
                        "propagate_to_subtasks": True,
                    },
                    rationale=f"Unify tags {tag_set} → '{unified_tag}'",
                    confidence=0.6,
                    tags=[unified_tag],
                )

                proposals.append(proposal)

        return proposals

    @staticmethod
    def _risk_level_to_float(risk_level: str) -> float:
        """Convert categorical risk to float."""
        mapping = {"low": 0.1, "medium": 0.5, "high": 0.9}
        return mapping.get(risk_level, 0.5)

    def validate_proposals(
        self, proposals: List[OntologyUpdateProposal]
    ) -> List[OntologyUpdateProposal]:
        """
        Validate proposals against constraints.

        Filters out proposals that violate econ/datapack/task-graph constraints.

        Args:
            proposals: List of proposals to validate

        Returns:
            List of valid proposals
        """
        valid: List[OntologyUpdateProposal] = []

        for prop in proposals:
            if not self._check_econ_constraints(prop):
                prop.respects_econ_constraints = False
                continue

            if not self._check_datapack_constraints(prop):
                prop.respects_datapack_constraints = False
                continue

            if not self._check_task_graph_constraints(prop):
                prop.respects_task_graph = False
                continue

            valid.append(prop)

        return valid

    def _check_econ_constraints(self, prop: OntologyUpdateProposal) -> bool:
        """
        Check if proposal respects economic constraints.

        Forbidden:
        - Modifying price_per_unit
        - Modifying damage_cost
        - Modifying reward weights
        """
        forbidden_keys = {
            "price_per_unit",
            "damage_cost",
            "wage_parity",
            "alpha",
            "beta",
            "gamma",
        }

        if any(k in prop.proposed_changes for k in forbidden_keys):
            return False

        return True

    def _check_datapack_constraints(self, prop: OntologyUpdateProposal) -> bool:
        """
        Check if proposal respects datapack constraints.

        Forbidden:
        - Modifying tier classification
        - Modifying novelty scoring
        """
        forbidden_keys = {"tier", "novelty_score", "data_premium"}

        if any(k in prop.proposed_changes for k in forbidden_keys):
            return False

        return True

    def _check_task_graph_constraints(self, prop: OntologyUpdateProposal) -> bool:
        """
        Check if proposal respects task graph constraints.

        Forbidden:
        - Deleting tasks
        - Modifying task dependencies directly
        """
        if prop.proposal_type == ProposalType.ADD_SKILL_GATE:
            return True

        if "delete_task" in prop.proposed_changes:
            return False

        return True
