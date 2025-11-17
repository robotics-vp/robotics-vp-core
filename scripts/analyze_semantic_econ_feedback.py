#!/usr/bin/env python3
"""
Run one offline semantic feedback pass and print advisory outputs.
"""
import json
import os

from src.orchestrator.economic_controller import EconomicController
from src.orchestrator.datapack_engine import DatapackEngine
from src.orchestrator.semantic_orchestrator import SemanticOrchestrator
from src.orchestrator.pipeline_manager import run_semantic_feedback_pass
from src.valuation.datapack_repo import DataPackRepo
from src.valuation.datapack_schema import ObjectiveProfile
from src.config.econ_params import EconParams


def main():
    econ = EconomicController.from_econ_params(EconParams(
        price_per_unit=0.3,
        damage_cost=1.0,
        energy_Wh_per_attempt=0.05,
        time_step_s=60.0,
        base_rate=2.0,
        p_min=0.02,
        k_err=0.12,
        q_speed=1.2,
        q_care=1.5,
        care_cost=0.25,
        max_steps=240,
        max_catastrophic_errors=3,
        max_error_rate_sla=0.12,
        min_steps_for_sla=5,
        zero_throughput_patience=10,
        preset="toy",
    ))
    datapacks = DatapackEngine(DataPackRepo(base_dir="data/datapacks"))
    # Minimal dummy graph/ontology
    from src.orchestrator.task_graph import TaskGraph, TaskNode
    from src.orchestrator.ontology import EnvironmentOntology
    sem = SemanticOrchestrator(
        econ_controller=econ,
        datapack_engine=datapacks,
        task_graph=TaskGraph(root=TaskNode("root", "root", "root task")),
        ontology=EnvironmentOntology(ontology_id="ont1", name="default"),
    )
    out_path = "results/semantic_feedback_preview.jsonl"
    run_semantic_feedback_pass(econ, datapacks, sem, out_path)

    from src.orchestrator.semantic_metrics import load_semantic_metrics
    metrics_list = load_semantic_metrics(out_path)
    metrics = metrics_list[-1] if metrics_list else None
    print("Latest SemanticMetrics:", metrics.__dict__ if metrics else None)
    print("Objective adjustments:", econ.suggest_objective_adjustments_from_semantics(metrics))
    print("Sampling overrides:", datapacks.update_sampling_from_semantics(metrics) if metrics else {})

    os.makedirs("results", exist_ok=True)
    with open("results/semantic_feedback_preview.json", "w") as f:
        json.dump({
            "metrics": metrics.__dict__ if metrics else None,
            "objective_adjustments": econ.suggest_objective_adjustments_from_semantics(metrics),
            "sampling_overrides": datapacks.update_sampling_from_semantics(metrics) if metrics else {},
        }, f, indent=2)


if __name__ == "__main__":
    main()
