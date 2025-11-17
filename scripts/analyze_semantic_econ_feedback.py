#!/usr/bin/env python3
"""
Run one offline semantic feedback pass and print advisory outputs.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json

from src.orchestrator.economic_controller import EconomicController
from src.orchestrator.datapack_engine import DatapackEngine
from src.orchestrator.semantic_orchestrator import SemanticOrchestrator
from src.orchestrator.pipeline_manager import run_semantic_feedback_pass
from src.valuation.datapack_repo import DataPackRepo
from src.valuation.datapack_schema import ObjectiveProfile
from src.config.econ_params import EconParams
from src.valuation.datapack_validators import validate_datapack_meta


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
    try:
        sample_warnings = []
        for env_name in ["drawer_vase", "dishwashing", "dishwashing_arm"]:
            for dp in datapacks.repo.iter_all(env_name) or []:
                sample_warnings.extend(validate_datapack_meta(dp))
                if len(sample_warnings) > 5:
                    break
            if len(sample_warnings) > 5:
                break
        if sample_warnings:
            print(f"[validation] Datapack warnings (sample): {sample_warnings[:5]}")
    except Exception:
        # Repo may be empty; keep advisory-only
        pass
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

    from src.orchestrator.semantic_metrics import load_semantic_metrics, load_semantic_econ_suggestions
    metrics_list = load_semantic_metrics(out_path)
    metrics = metrics_list[-1] if metrics_list else None
    print("Latest SemanticMetrics:", metrics.__dict__ if metrics else None)
    print("Objective adjustments:", econ.suggest_objective_adjustments_from_semantics(metrics))
    print("Sampling overrides:", datapacks.update_sampling_from_semantics(metrics) if metrics else {})

    # Generate and save semantic-econ suggestions (NEW: closes feedback loop)
    print("\n--- Generating Semantic-Econ Suggestions ---")
    suggestions = []

    # Aggregate econ signals from datapacks
    from src.orchestrator.economic_controller import EconSignals
    econ_signals = econ.compute_signals([])  # Empty for demo - uses default signals
    datapack_signals = datapacks.compute_datapack_stats()  # Returns coverage/tier signals

    # Generate a suggestion
    suggestion = econ.generate_semantic_econ_suggestion(
        econ_signals=econ_signals,
        datapack_signals=datapack_signals,
        semantic_metrics=metrics,
    )
    suggestions.append(suggestion)

    print(f"Suggested profile: {suggestion.suggested_profile}")
    print(f"Suggested adjustments: {suggestion.suggested_objective_adjustment}")
    print(f"Suggested sampling: {suggestion.suggested_sampling_override}")
    print(f"Rationale: {suggestion.rationale}")

    # Save suggestions to JSONL (for transformer training)
    suggestion_path = "results/semantic_econ_suggestions.jsonl"
    econ.dump_semantic_econ_suggestions(suggestions, suggestion_path)

    # Load and display contract
    print("\n--- Contract: Econ/Datapack say X -> Semantic Orchestrator did Y ---")
    loaded_suggestions = load_semantic_econ_suggestions(suggestion_path)
    if loaded_suggestions:
        latest = loaded_suggestions[-1]
        print(f"Econ context: {latest.get('econ_context', {})}")
        print(f"Datapack context: {latest.get('datapack_context', {})}")
        print(f"Semantic metrics: {latest.get('semantic_metrics', {})}")
        print(f"Suggestion: profile={latest.get('suggested_profile')}, adjustments={latest.get('suggested_objective_adjustment')}")

    os.makedirs("results", exist_ok=True)
    with open("results/semantic_feedback_preview.json", "w") as f:
        json.dump({
            "metrics": metrics.__dict__ if metrics else None,
            "objective_adjustments": econ.suggest_objective_adjustments_from_semantics(metrics),
            "sampling_overrides": datapacks.update_sampling_from_semantics(metrics) if metrics else {},
            "semantic_econ_suggestions": [s.__dict__ if hasattr(s, "__dict__") else s for s in suggestions],
        }, f, indent=2, default=str)


if __name__ == "__main__":
    main()
