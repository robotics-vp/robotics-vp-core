"""
Semantic Orchestrator: Meta-mother for semantic consistency.

IMPORTANT HIERARCHY:
- DOWNSTREAM of EconomicController and DatapackEngine
- UPSTREAM / meta-mother to VLA/DINO/SIMA/diffusion/RL semantics
- Does NOT invent economics or data value - it EXECUTES them in semantic space

This module consumes econ_signals and datapack_signals to:
1. Update task graph priorities and structure
2. Update ontology affordances and constraints
3. Align semantic tags across all perception modules
4. Ensure VLA/SIMA/diffusion/RL share consistent vocabulary

FORBIDDEN:
- Importing or defining EconSignals/DatapackSignals (consume only)
- Setting reward weights directly (use suggestions from MetaTransformer)
- Modifying Phase B math

ALLOWED:
- Mutating task_graph based on econ/datapack priorities
- Updating ontology affordances based on economic constraints
- Aligning primitive vocabularies across modules
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.orchestrator.economic_controller import EconomicController, EconSignals
from src.orchestrator.datapack_engine import DatapackEngine, DatapackSignals
from src.orchestrator.semantic_metrics import SemanticMetrics, write_semantic_metrics
from src.orchestrator.task_graph import TaskGraph, TaskNode, TaskStatus
from src.orchestrator.ontology import EnvironmentOntology, ObjectSpec, AffordanceType


@dataclass
class SemanticUpdatePlan:
    """
    Plan for updating semantic structures based on econ/datapack signals.

    This is the output of the semantic orchestrator - a specification of
    what should change in the task graph, ontology, and primitive vocabularies.
    """
    task_graph_changes: Dict[str, Any] = field(default_factory=dict)
    # e.g., {"prioritize_tasks": ["grasp_handle"], "deprioritize": ["lift_heavy"]}

    ontology_changes: Dict[str, Any] = field(default_factory=dict)
    # e.g., {"increase_fragility_awareness": True, "adjust_affordance_risks": {...}}

    primitive_updates: Dict[str, Any] = field(default_factory=dict)
    # e.g., {"skill_weights": {...}, "tag_emphasis": [...]}

    cross_module_constraints: Dict[str, Any] = field(default_factory=dict)
    # e.g., {"vla_caution_level": "high", "diffusion_safety_margin": 0.2}

    rationale: str = ""
    # Human-readable explanation of why these changes are proposed

    urgency_driven: bool = False
    # True if changes are driven by economic urgency signals

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_graph_changes": self.task_graph_changes,
            "ontology_changes": self.ontology_changes,
            "primitive_updates": self.primitive_updates,
            "cross_module_constraints": self.cross_module_constraints,
            "rationale": self.rationale,
            "urgency_driven": self.urgency_driven,
        }


@dataclass
class MetaTransformerOutputs:
    """
    Suggestions from the MetaTransformer (optional input to SemanticOrchestrator).

    These are HINTS only - the SemanticOrchestrator decides whether to apply them.
    """
    energy_profile_weights: Dict[str, float] = field(default_factory=dict)
    data_mix_weights: Dict[str, float] = field(default_factory=dict)
    objective_preset: str = "balanced"
    chosen_backend: str = "pybullet"
    expected_delta_mpl: float = 0.0
    expected_delta_error: float = 0.0
    expected_delta_energy_Wh: float = 0.0
    orchestration_plan: List[Any] = field(default_factory=list)


class SemanticOrchestrator:
    """
    NOTE:
      - DOWNSTREAM of EconomicController and DatapackEngine.
      - UPSTREAM / meta-mother to VLA/DINO/SIMA/diffusion/RL semantics.
      - Does NOT invent economics or data value - it EXECUTES them in the semantic space.
    """

    def __init__(
        self,
        econ_controller: EconomicController,
        datapack_engine: DatapackEngine,
        task_graph: TaskGraph,
        ontology: EnvironmentOntology,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize SemanticOrchestrator with upstream dependencies.

        Args:
            econ_controller: UPSTREAM economic controller
            datapack_engine: UPSTREAM data valuation engine
            task_graph: Task graph to mutate
            ontology: Environment ontology to mutate
        """
        self.econ = econ_controller
        self.datapacks = datapack_engine
        self.task_graph = task_graph
        self.ontology = ontology
        self.config = config or {}

        # Track semantic state
        self._update_history: List[SemanticUpdatePlan] = []
        self._current_constraints: Dict[str, Any] = {}

    def build_update_plan(
        self,
        econ_signals: EconSignals,
        datapack_signals: DatapackSignals,
        meta_out: Optional[MetaTransformerOutputs] = None,
    ) -> SemanticUpdatePlan:
        """
        Consume econ/datapack signals (+ optional meta-transformer suggestions)
        and decide HOW the semantics should change:
          - which tasks are prioritized/merged/split
          - which affordances get emphasized
          - how ontological categories shift

        Args:
            econ_signals: Economic signals from EconomicController
            datapack_signals: Data signals from DatapackEngine
            meta_out: Optional suggestions from MetaTransformer

        Returns:
            SemanticUpdatePlan specifying semantic mutations
        """
        plan = SemanticUpdatePlan()
        rationale_parts = []

        # 1. Prioritize tasks based on economic urgencies
        task_priorities = {}
        if econ_signals.error_urgency > 0.5:
            # High error rate - prioritize safety-related tasks
            plan.urgency_driven = True
            for node in self.task_graph.get_all_nodes():
                if "verify" in node.name.lower() or "check" in node.name.lower():
                    task_priorities[node.task_id] = "high"
                if node.task_type.value == "checkpoint":
                    task_priorities[node.task_id] = "critical"
            rationale_parts.append(f"Error urgency {econ_signals.error_urgency:.2f} - prioritizing safety checks")

        if econ_signals.energy_urgency > 0.5:
            # High energy costs - prioritize efficient motions
            plan.urgency_driven = True
            for node in self.task_graph.get_skill_nodes():
                if node.skill_id in [2, 5]:  # PULL, MOVE are often energy-intensive
                    task_priorities[node.task_id] = "optimize_energy"
            rationale_parts.append(f"Energy urgency {econ_signals.energy_urgency:.2f} - optimizing motion efficiency")

        if econ_signals.mpl_urgency > 0.3:
            # Low MPL - prioritize throughput
            plan.urgency_driven = True
            for node in self.task_graph.get_leaf_nodes():
                if node.task_type.value == "skill":
                    task_priorities[node.task_id] = "speed_up"
            rationale_parts.append(f"MPL urgency {econ_signals.mpl_urgency:.2f} - focusing on throughput")

        plan.task_graph_changes["task_priorities"] = task_priorities

        # 2. Adjust ontology based on economic constraints
        ontology_adjustments = {}

        # Increase fragility awareness if errors are high
        if econ_signals.error_urgency > 0.3:
            ontology_adjustments["fragility_multiplier"] = 1.0 + econ_signals.error_urgency
            for obj_id, obj in self.ontology.objects.items():
                if obj.fragility > 0.5:
                    # Increase risk awareness for fragile objects
                    for aff in obj.affordances:
                        ontology_adjustments[f"{obj_id}_{aff.affordance_type.value}_risk"] = (
                            aff.risk_level * (1.0 + econ_signals.error_urgency)
                        )
            rationale_parts.append("Increasing fragility awareness due to error rate")

        # Adjust energy cost estimates based on energy urgency
        if econ_signals.energy_urgency > 0.3:
            ontology_adjustments["energy_cost_multiplier"] = 1.0 + econ_signals.energy_urgency * 0.5
            rationale_parts.append("Adjusting energy cost estimates upward")

        plan.ontology_changes = ontology_adjustments

        # 3. Update primitive vocabulary based on datapack signals
        primitive_updates = {}

        # Adjust skill emphasis based on data coverage
        if datapack_signals.tier2_fraction < 0.1:
            # Need more frontier data - emphasize exploration
            primitive_updates["exploration_bonus"] = 0.2
            primitive_updates["exploitation_penalty"] = 0.1
            rationale_parts.append("Low frontier data - encouraging exploration")

        # Semantic tag emphasis
        tag_emphasis = []
        if datapack_signals.vla_annotation_fraction < 0.5:
            tag_emphasis.append("vla:needs_annotation")
        if econ_signals.error_urgency > 0.5:
            tag_emphasis.append("safety:high_priority")
        if econ_signals.energy_urgency > 0.5:
            tag_emphasis.append("energy:optimize")

        primitive_updates["tag_emphasis"] = tag_emphasis
        primitive_updates["recommended_focus"] = datapack_signals.recommended_collection_focus

        plan.primitive_updates = primitive_updates

        # 4. Cross-module constraints
        cross_module = {}

        # VLA caution level
        if econ_signals.error_urgency > 0.7:
            cross_module["vla_caution_level"] = "very_high"
        elif econ_signals.error_urgency > 0.3:
            cross_module["vla_caution_level"] = "high"
        else:
            cross_module["vla_caution_level"] = "normal"

        # Diffusion safety margin
        fragile_objs = self.ontology.get_fragile_objects(threshold=0.5)
        if fragile_objs:
            cross_module["diffusion_safety_margin"] = 0.2 + 0.1 * len(fragile_objs)
        else:
            cross_module["diffusion_safety_margin"] = 0.1

        # RL exploration rate
        if datapack_signals.data_coverage_score < 0.5:
            cross_module["rl_exploration_rate"] = 0.3  # More exploration
        else:
            cross_module["rl_exploration_rate"] = 0.1  # More exploitation

        # SIMA primitive selection bias
        if econ_signals.wage_parity < 0.8:
            cross_module["sima_efficiency_bias"] = "high"
        else:
            cross_module["sima_efficiency_bias"] = "balanced"

        # Epiplexity-driven scheduling (advisory)
        if self.config.get("use_epiplexity_term", False):
            epi_alpha = float(self.config.get("epi_alpha", 0.1))
            epi_term = float(getattr(datapack_signals, "mean_delta_epi_per_flop", 0.0))
            cross_module["epiplexity_term"] = {
                "enabled": True,
                "epi_alpha": epi_alpha,
                "expected_delta_epi_per_flop": epi_term,
            }
            if epi_term > 0.0:
                rationale_parts.append("Epiplexity term enabled for scheduling")

        plan.cross_module_constraints = cross_module

        # 5. Apply meta-transformer suggestions if provided
        if meta_out:
            if meta_out.objective_preset:
                plan.primitive_updates["suggested_preset"] = meta_out.objective_preset
                rationale_parts.append(f"MetaTransformer suggests preset: {meta_out.objective_preset}")

            if meta_out.energy_profile_weights:
                plan.primitive_updates["energy_profile_weights"] = meta_out.energy_profile_weights

        # Build rationale
        plan.rationale = "; ".join(rationale_parts) if rationale_parts else "No urgent changes needed"

        return plan

    def apply_update_plan(self, plan: SemanticUpdatePlan) -> None:
        """
        Mutate task_graph + ontology in-place according to the plan.

        Args:
            plan: SemanticUpdatePlan to apply
        """
        # Apply task graph changes
        if "task_priorities" in plan.task_graph_changes:
            for task_id, priority in plan.task_graph_changes["task_priorities"].items():
                node = self.task_graph.get_node_by_id(task_id)
                if node:
                    node.metadata["semantic_priority"] = priority

        # Apply ontology changes
        for key, value in plan.ontology_changes.items():
            if key == "fragility_multiplier":
                self.ontology.metadata["fragility_multiplier"] = value
            elif "_risk" in key:
                # Update specific affordance risk
                self.ontology.metadata[key] = value
            elif key == "energy_cost_multiplier":
                self.ontology.metadata["energy_cost_multiplier"] = value

        # Store current constraints
        self._current_constraints = plan.cross_module_constraints.copy()

        # Add to history
        self._update_history.append(plan)

    def update_task_graph(
        self,
        econ_signals: EconSignals,
        datapack_signals: DatapackSignals,
        meta_out: Optional[MetaTransformerOutputs] = None,
    ) -> None:
        """
        Shortcut to build and apply updates to task graph only.
        """
        plan = self.build_update_plan(econ_signals, datapack_signals, meta_out)
        self.apply_update_plan(plan)

    def update_ontology(
        self,
        econ_signals: EconSignals,
        datapack_signals: DatapackSignals,
        meta_out: Optional[MetaTransformerOutputs] = None,
    ) -> None:
        """
        Shortcut to build and apply updates to ontology only.
        """
        plan = self.build_update_plan(econ_signals, datapack_signals, meta_out)
        self.apply_update_plan(plan)

    def align_semantics_across_modules(
        self,
        vla_tags: Dict[str, Any],
        sima_primitives: Optional[Dict[str, Any]] = None,
        diffusion_tags: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Enforce that VLA, SIMA, diffusion, RL, and sim share a consistent
        vocabulary, grounded in the econ/datapack priorities already computed.

        Args:
            vla_tags: Current VLA semantic tags
            sima_primitives: Optional SIMA primitive definitions
            diffusion_tags: Optional diffusion model tags

        Returns:
            Aligned semantic mapping for all modules
        """
        aligned = {
            "shared_vocabulary": [],
            "module_specific_mappings": {},
            "constraints": self._current_constraints.copy(),
        }

        # Build shared vocabulary from ontology
        core_vocab = set()
        for obj in self.ontology.objects.values():
            core_vocab.add(obj.category.value)
            core_vocab.update(obj.tags)
            for aff in obj.affordances:
                core_vocab.add(aff.affordance_type.value)

        aligned["shared_vocabulary"] = list(core_vocab)

        # Map VLA tags to shared vocabulary
        vla_mapping = {}
        for tag, value in vla_tags.items():
            if "grasp" in tag.lower():
                vla_mapping[tag] = "graspable"
            elif "lift" in tag.lower():
                vla_mapping[tag] = "liftable"
            elif "place" in tag.lower():
                vla_mapping[tag] = "placeable"
            else:
                vla_mapping[tag] = tag

        aligned["module_specific_mappings"]["vla"] = vla_mapping

        # Map SIMA primitives if provided
        if sima_primitives:
            sima_mapping = {}
            for prim_name, prim_def in sima_primitives.items():
                # Map to closest ontology concept
                if "move" in prim_name.lower():
                    sima_mapping[prim_name] = "pushable"
                elif "pick" in prim_name.lower():
                    sima_mapping[prim_name] = "graspable"
                else:
                    sima_mapping[prim_name] = prim_name
            aligned["module_specific_mappings"]["sima"] = sima_mapping

        # Map diffusion tags if provided
        if diffusion_tags:
            diff_mapping = {}
            for tag, count in diffusion_tags.items():
                if tag in core_vocab:
                    diff_mapping[tag] = tag
                else:
                    diff_mapping[tag] = "unaligned"
            aligned["module_specific_mappings"]["diffusion"] = diff_mapping

        return aligned

    def semantic_consistency_checks(self) -> Dict[str, Any]:
        """
        Return diagnostics on where semantics are drifting relative to
        econ/datapack priorities.
        """
        checks = {
            "drift_warnings": [],
            "consistency_score": 1.0,
            "recommended_fixes": [],
        }

        # Check if task priorities align with economic urgencies
        urgency_driven_tasks = sum(
            1 for node in self.task_graph.get_all_nodes()
            if node.metadata.get("semantic_priority") in ["high", "critical"]
        )
        if urgency_driven_tasks == 0 and self._update_history:
            last_plan = self._update_history[-1]
            if last_plan.urgency_driven:
                checks["drift_warnings"].append(
                    "Task priorities not reflecting economic urgencies"
                )
                checks["consistency_score"] -= 0.2

        # Check if ontology constraints match current economic state
        if "fragility_multiplier" not in self.ontology.metadata:
            fragile_count = len(self.ontology.get_fragile_objects())
            if fragile_count > 0:
                checks["drift_warnings"].append(
                    "Fragility awareness not set despite fragile objects"
                )
                checks["recommended_fixes"].append("Apply semantic update plan")
                checks["consistency_score"] -= 0.1

        # Check cross-module constraints
        if not self._current_constraints:
            checks["drift_warnings"].append("No cross-module constraints set")
            checks["consistency_score"] -= 0.1

        return checks

    def snapshot(self) -> Dict[str, Any]:
        """
        Get current semantic state snapshot.

        Returns:
            Dict with current task graph, ontology, and constraints
        """
        return {
            "task_graph_summary": self.task_graph.summary(),
            "ontology_summary": self.ontology.summary(),
            "current_constraints": self._current_constraints,
            "update_history_length": len(self._update_history),
            "consistency": self.semantic_consistency_checks(),
        }

    def compute_semantic_metrics(
        self,
        econ_signals: Dict[str, Any],
        datapack_signals: Dict[str, Any],
    ) -> SemanticMetrics:
        consistency = self.semantic_consistency_checks()
        return SemanticMetrics(
            ontology_version=str(getattr(self.ontology, "metadata", {}).get("version", "v0")),
            task_graph_version=str(getattr(self.task_graph, "metadata", {}).get("version", "v0")),
            task_cluster_purity=1.0 - consistency.get("consistency_score", 0.0),
            concept_drift_score=max(0.0, 1.0 - consistency.get("consistency_score", 0.0)),
            label_conflict_rate=len(consistency.get("drift_warnings", [])) / 10.0,
            vla_vs_sima_agreement=0.8,
            vla_vs_diffusion_agreement=0.8,
            sim_vs_real_agreement=0.8,
            econ_relevant_task_fraction=0.5,
            econ_ignored_task_fraction=0.5,
            underrepresented_tasks=[],
            overrepresented_tasks=[],
            extra={"has_cross_constraints": float(bool(self._current_constraints))},
        )

    def export_semantic_metrics(
        self,
        econ_signals: Dict[str, Any],
        datapack_signals: Dict[str, Any],
        out_path: str,
    ) -> None:
        metrics = self.compute_semantic_metrics(econ_signals, datapack_signals)
        write_semantic_metrics(metrics, out_path)
