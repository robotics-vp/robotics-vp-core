#!/usr/bin/env python3
"""
Deterministic semantic feedback loop smoke test.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.orchestrator.semantic_orchestrator import SemanticOrchestrator
from src.orchestrator.economic_controller import EconomicController
from src.orchestrator.datapack_engine import DatapackEngine
from src.orchestrator.pipeline_manager import run_semantic_feedback_pass
from src.valuation.datapack_repo import DataPackRepo
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
    from src.orchestrator.task_graph import TaskGraph, TaskNode
    from src.orchestrator.ontology import EnvironmentOntology
    sem = SemanticOrchestrator(
        econ_controller=econ,
        datapack_engine=datapacks,
        task_graph=TaskGraph(root=TaskNode("root", "root", "root task")),
        ontology=EnvironmentOntology(ontology_id="ont1", name="default"),
    )
    run_semantic_feedback_pass(econ, datapacks, sem, "results/semantic_feedback_loop.jsonl")
    from src.orchestrator.semantic_metrics import load_semantic_metrics
    metrics = load_semantic_metrics("results/semantic_feedback_loop.jsonl")[-1]
    print("Metrics:", metrics.__dict__)
    print("Objective adjustments:", econ.suggest_objective_adjustments_from_semantics(metrics))
    print("Sampling overrides:", datapacks.update_sampling_from_semantics(metrics))


if __name__ == "__main__":
    main()
