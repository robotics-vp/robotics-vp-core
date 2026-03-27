"""
SemanticOrchestratorV2: semantic routing shell that can emit bounded activation plans.
"""
from dataclasses import dataclass, field, asdict, replace
from typing import Any, Dict, List, Optional
import json

from src.evidence.preconditions import build_execution_work_order
from src.orchestrator.shell_activation import (
    evaluate_shell_activation_backlog,
    get_shell_activation_assessment,
)
from src.semantic.models import SemanticSnapshot
from src.utils.json_safe import to_json_safe


@dataclass
class OrchestratorAdvisory:
    task_id: str
    focus_objective_presets: List[str]
    sampler_strategy_overrides: Dict[str, float]
    datapack_priority_tags: List[str]
    safety_emphasis: float
    execution_mode: str = "advisory"
    policy_source: str = "heuristic_fallback"
    promotion_stage: Optional[str] = None
    meta_node_weights: Dict[str, float] = field(default_factory=dict)
    activation_plan: Dict[str, Any] = field(default_factory=dict)
    activation_work_order: Optional[Dict[str, Any]] = None
    helper_trace: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        return to_json_safe(asdict(self))


class SemanticOrchestratorV2:
    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        self.output_dir = self.config.get("output_dir", "results/orchestrator")
        self.write_to_file = self.config.get("write_to_file", True)
        self.trust_matrix = self.config.get("trust_matrix", {}) or {}
        self.shell_policy_helper_mode = str(
            self.config.get("shell_policy_helper_mode", "disabled") or "disabled"
        )
        self._shell_policy_helper = None
        if self.shell_policy_helper_mode != "disabled":
            from src.orchestrator.orchestrator_shell_policy_runtime import (
                resolve_orchestrator_shell_policy_helper,
            )

            self._shell_policy_helper = resolve_orchestrator_shell_policy_helper(
                helper_mode=self.shell_policy_helper_mode,
                package=self.config.get("shell_policy_package"),
                package_path=self.config.get("shell_policy_package_path"),
            )

    def _normalize_strategy_overrides(self, strategy_overrides: Dict[str, float]) -> Dict[str, float]:
        clean = {str(key): float(max(0.0, value)) for key, value in strategy_overrides.items()}
        total = sum(clean.values())
        if total <= 0.0:
            return {"balanced": 1.0}
        return {key: value / total for key, value in clean.items()}

    def propose(self, snapshot: SemanticSnapshot) -> OrchestratorAdvisory:
        econ = snapshot.econ_slice
        meta = snapshot.meta_slice
        focus_presets = list(meta.presets or ["balanced"])
        execution_summary = {}
        if snapshot.metadata:
            execution_summary = (
                snapshot.metadata.get("execution_precondition_summary")
                or snapshot.metadata.get("execution_preconditions")
                or {}
            )

        strategy_overrides = {"balanced": 0.5, "frontier_prioritized": 0.3, "econ_urgency": 0.2}
        if econ.avg_wage_parity < 1.0:
            strategy_overrides["econ_urgency"] = 0.5
            strategy_overrides["frontier_prioritized"] = 0.3
            strategy_overrides["balanced"] = 0.2

        safety_emphasis = 0.3
        priority_tags: List[str] = []
        recap = snapshot.metadata.get("recap", {}) if snapshot.metadata else {}
        world_model = getattr(snapshot, "semantic_world_model", None)
        if world_model is None and snapshot.metadata:
            payload = snapshot.metadata.get("semantic_world_model")
            if isinstance(payload, dict):
                from src.world_model.semantic_world_model import SemanticWorldModelState

                world_model = SemanticWorldModelState.from_dict(payload)
        for tag in snapshot.semantic_tags:
            try:
                enrichment = tag.to_dict()
                supervision = enrichment.get("supervision_hints", {})
                if supervision.get("safety_critical"):
                    safety_emphasis = 0.8
                if enrichment.get("fragility_tags"):
                    priority_tags.append("fragility_tags")
                if enrichment.get("risk_tags"):
                    priority_tags.append("risk_tags")
            except Exception:
                continue
        priority_tags = sorted(list(set(priority_tags)))
        if recap:
            mean_good = float(recap.get("mean_goodness", 0.0))
            if mean_good > 0:
                strategy_overrides["frontier_prioritized"] = min(1.0, strategy_overrides.get("frontier_prioritized", 0.3) + 0.1)
            if mean_good < 0:
                strategy_overrides["balanced"] = min(1.0, strategy_overrides.get("balanced", 0.5) + 0.1)
            if recap.get("top_episodes"):
                priority_tags.append("recap_top")
            priority_tags = sorted(list(set(priority_tags)))

        meta_node_weights: Dict[str, float] = {}
        capability_scores: Dict[str, float] = {}
        topology: Dict[str, Any] = {}
        if world_model is not None:
            meta_node_weights = {
                node.node_type: float(node.score)
                for node in world_model.meta_nodes
            }
            capability_scores = dict(world_model.capability_scores or {})
            topology = dict(world_model.topology or {})
            if meta_node_weights.get("risk_triage", 0.0) > 0.0:
                safety_emphasis = max(
                    safety_emphasis,
                    min(1.0, 0.35 + 0.65 * float(meta_node_weights["risk_triage"])),
                )
                priority_tags.append("risk_triage")
            if meta_node_weights.get("recovery_router", 0.0) >= 0.45:
                priority_tags.append("recovery_router")
                strategy_overrides["frontier_prioritized"] = strategy_overrides.get("frontier_prioritized", 0.3) + 0.05
            if meta_node_weights.get("semantic_memory_refresh", 0.0) >= 0.45:
                priority_tags.append("semantic_memory_refresh")
                strategy_overrides["frontier_prioritized"] = strategy_overrides.get("frontier_prioritized", 0.3) + 0.1
            if meta_node_weights.get("fusion_bridge", 0.0) >= 0.45:
                priority_tags.append("fusion_backbone")
            if meta_node_weights.get("ontology_router", 0.0) >= 0.4:
                priority_tags.append("ontology_router")
            if meta_node_weights.get("task_graph_router", 0.0) >= 0.4:
                priority_tags.append("task_graph_router")
            if meta_node_weights.get("efficiency_router", 0.0) >= 0.45:
                priority_tags.append("efficiency_router")
                if "energy_saver" not in focus_presets:
                    focus_presets.append("energy_saver")
            if capability_scores.get("risk_reasoning", 0.0) < 0.5 and "safety" not in focus_presets:
                focus_presets.append("safety")
            if capability_scores.get("stage2_bridge", 0.0) < 0.35:
                priority_tags.append("stage2_bridge_gap")
            if capability_scores.get("fusion_bridge", 0.0) < 0.35:
                priority_tags.append("fusion_bridge_gap")
            priority_tags = sorted(list(set(priority_tags)))

        # Trust-aware safety emphasis
        ood_trust = float(self.trust_matrix.get("OODTag", {}).get("trust_score", 0.0))
        max_ood_sev = 0.0
        if snapshot.metadata:
            max_ood_sev = float(snapshot.metadata.get("max_ood_severity", snapshot.metadata.get("ood_severity", 0.0)))
        if ood_trust > 0.8 and max_ood_sev > 0.9:
            safety_emphasis = 1.0
            priority_tags.append("safety_stop")
        elif ood_trust > 0.5 and max_ood_sev > 0.9:
            priority_tags.append("ood_warning")
        priority_tags = sorted(list(set(priority_tags)))
        if isinstance(execution_summary, dict) and execution_summary:
            blocked_count = int(execution_summary.get("blocked_count", 0) or 0)
            ready_count = int(execution_summary.get("ready_count", 0) or 0)
            if blocked_count > 0:
                priority_tags.append("precondition_repair")
                safety_emphasis = max(safety_emphasis, 0.7)
            if ready_count == 0:
                strategy_overrides["balanced"] = min(1.0, strategy_overrides.get("balanced", 0.5) + 0.1)
            priority_tags = sorted(list(set(priority_tags)))

        segmentation_meta = {
            "num_segments": getattr(snapshot, "num_segments", 0),
            "segment_types": getattr(snapshot, "segment_types", {}),
            "subtask_label_histogram": getattr(snapshot, "subtask_label_histogram", {}),
            "mobility_drift_rate": getattr(snapshot, "mobility_drift_rate", 0.0),
            "recovery_segment_fraction": getattr(snapshot, "recovery_segment_fraction", 0.0),
        }
        mobility_priority = "MEDIUM"
        if segmentation_meta["recovery_segment_fraction"] > 0.2 or segmentation_meta["mobility_drift_rate"] > 0.5:
            mobility_priority = "HIGH"
        if segmentation_meta["mobility_drift_rate"] < 0.1:
            mobility_priority = "LOW"

        shell_activation = evaluate_shell_activation_backlog(
            execution_summary if isinstance(execution_summary, dict) else {},
            module_keys=["semantic_orchestrator_v2"],
            subject_prefix=f"task:{snapshot.task_id}",
        )
        routing_activation = get_shell_activation_assessment(
            shell_activation,
            "semantic_orchestrator_preconditioned_routing",
        )
        execution_mode = "advisory"
        activation_plan: Dict[str, Any] = {}
        activation_work_order: Optional[Dict[str, Any]] = None
        if routing_activation and routing_activation.get("state") == "activated":
            execution_mode = str(routing_activation.get("target_mode", "preconditioned_routing"))
            priority_tags.append("precondition_ready")
            priority_tags = sorted(list(set(priority_tags)))
            activation_plan = {
                "activation_id": routing_activation.get("activation_id"),
                "mode": execution_mode,
                "apply_sampler_strategy_overrides": self._normalize_strategy_overrides(strategy_overrides),
                "apply_priority_tags": priority_tags,
                "bounded_actions": list(routing_activation.get("bounded_actions", []) or []),
                "repair_backlog": list(
                    shell_activation.get("pending", [])
                ),
            }
            activation_work_order = build_execution_work_order(
                order_type="shell_activation",
                subject_id=snapshot.task_id,
                subject_kind="semantic_orchestrator_v2",
                decision=str(routing_activation.get("activation_decision", "activate_semantic_routing")),
                priority=float(routing_activation.get("readiness", {}).get("readiness_score", 1.0)),
                recommended_mode=str(routing_activation.get("recommended_mode", "bounded_execution")),
                readiness=dict(routing_activation.get("readiness", {}) or {}),
                reasons=list(routing_activation.get("bounded_actions", []) or ["activate_semantic_routing"]),
                metadata={
                    "activation_id": routing_activation.get("activation_id"),
                    "task_id": snapshot.task_id,
                },
            ).to_dict()

        advisory = OrchestratorAdvisory(
            task_id=snapshot.task_id,
            focus_objective_presets=sorted(list(set(focus_presets))),
            sampler_strategy_overrides=self._normalize_strategy_overrides(strategy_overrides),
            datapack_priority_tags=priority_tags,
            safety_emphasis=float(min(max(safety_emphasis, 0.0), 1.0)),
            execution_mode=execution_mode,
            meta_node_weights=dict(sorted(meta_node_weights.items(), key=lambda item: item[0])),
            activation_plan=activation_plan,
            activation_work_order=activation_work_order,
            metadata={
                "frontier_eps": econ.frontier_episodes,
                "recap": recap,
                "segmentation": segmentation_meta,
                "mobility_priority": mobility_priority,
                "execution_preconditions": execution_summary,
                "shell_activation": shell_activation,
                "semantic_world_model_topology": topology,
                "semantic_capabilities": capability_scores,
            },
        )
        if self._shell_policy_helper is not None:
            helper_update = self._shell_policy_helper.apply_to_advisory(
                snapshot=snapshot,
                heuristic_advisory=advisory.to_json(),
                trust_matrix=self.trust_matrix,
                helper_mode=self.shell_policy_helper_mode,
            )
            metadata = dict(advisory.metadata)
            metadata["shell_policy_helper"] = dict(helper_update.get("helper_trace", {}) or {})
            metadata["shell_policy_helper_mode"] = self.shell_policy_helper_mode
            advisory = replace(
                advisory,
                focus_objective_presets=list(helper_update.get("focus_objective_presets", advisory.focus_objective_presets) or advisory.focus_objective_presets),
                sampler_strategy_overrides=self._normalize_strategy_overrides(
                    dict(helper_update.get("sampler_strategy_overrides", advisory.sampler_strategy_overrides) or advisory.sampler_strategy_overrides)
                ),
                safety_emphasis=float(
                    helper_update.get("safety_emphasis", advisory.safety_emphasis)
                ),
                execution_mode=str(helper_update.get("execution_mode", advisory.execution_mode) or advisory.execution_mode),
                policy_source=str(helper_update.get("policy_source", "heuristic_plus_learned_helper") or "heuristic_plus_learned_helper"),
                promotion_stage=str(helper_update.get("promotion_stage", "shadow_candidate") or "shadow_candidate"),
                activation_plan=dict(helper_update.get("activation_plan", advisory.activation_plan) or {}),
                activation_work_order=helper_update.get("activation_work_order", advisory.activation_work_order),
                helper_trace=dict(helper_update.get("helper_trace", {}) or {}),
                metadata=metadata,
            )
        if self.write_to_file:
            self._write_advisory(advisory)
        return advisory

    def _write_advisory(self, advisory: OrchestratorAdvisory) -> None:
        import os

        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, f"advisories_{advisory.task_id}.jsonl")
        with open(path, "a") as f:
            f.write(json.dumps(advisory.to_json(), sort_keys=True))
            f.write("\n")

def load_latest_advisory(task_id: str, output_dir: str = "results/orchestrator") -> Optional[OrchestratorAdvisory]:
    import os
    path = os.path.join(output_dir, f"advisories_{task_id}.jsonl")
    if not os.path.exists(path):
        return None
    last = None
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                last = json.loads(line)
    if not last:
        return None
    return OrchestratorAdvisory(**last)
