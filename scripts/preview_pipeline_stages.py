#!/usr/bin/env python3
"""
Preview the 5-stage learning pipeline.

Demonstrates:
1. Task graph decomposition
2. Environment ontology
3. Pipeline manager with advisory stages
4. Vision backbone embeddings
5. End-to-end flow visualization

ADVISORY ONLY: This script shows what the pipeline looks like but does not
execute actual training or modify Phase B.
"""

import argparse
import json
import os
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(description="Preview the 5-stage learning pipeline")
    parser.add_argument("--env", type=str, default="drawer_vase", choices=["drawer_vase", "grasp_place"])
    parser.add_argument("--iterations", type=int, default=3, help="Number of iterations to simulate")
    parser.add_argument("--out-dir", type=str, default="results/pipeline_preview")
    parser.add_argument("--verbose", action="store_true", help="Show detailed output")
    args = parser.parse_args()

    print("=" * 70)
    print("5-STAGE LEARNING PIPELINE PREVIEW")
    print("=" * 70)
    print()
    print("This is an ADVISORY preview showing pipeline structure and flow.")
    print("No actual training is executed. Phase B remains frozen.")
    print()

    # Stage 1: Task Graph
    print("=" * 70)
    print("1. TASK GRAPH - Semantic Task Decomposition")
    print("=" * 70)

    from src.orchestrator.task_graph import build_drawer_vase_task_graph, build_grasp_place_task_graph

    if args.env == "drawer_vase":
        task_graph = build_drawer_vase_task_graph()
    else:
        task_graph = build_grasp_place_task_graph()

    print(f"Task Graph: {task_graph.name}")
    print(f"  Graph ID: {task_graph.graph_id}")
    print(f"  Description: {task_graph.description}")
    print()

    summary = task_graph.summary()
    print("Graph Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print()

    print("Task Nodes:")
    for i, node in enumerate(task_graph.get_all_nodes()[:10]):  # Show first 10
        skill_info = f" [skill_id={node.skill_id}]" if node.skill_id is not None else ""
        print(f"  {i+1}. {node.name} ({node.task_type.value}){skill_info}")
        if node.preconditions:
            print(f"     Preconditions: {node.preconditions}")
        if node.postconditions:
            print(f"     Postconditions: {node.postconditions}")
        if node.affordances:
            print(f"     Affordances: {node.affordances}")

    print()
    print(f"Critical Path: {task_graph.get_critical_path()}")
    print(f"Execution Order: {task_graph.get_execution_order()}")
    print()

    # Stage 2: Environment Ontology
    print("=" * 70)
    print("2. ENVIRONMENT ONTOLOGY - Objects and Affordances")
    print("=" * 70)

    from src.orchestrator.ontology import build_drawer_vase_ontology, build_simple_grasp_ontology

    if args.env == "drawer_vase":
        ontology = build_drawer_vase_ontology()
    else:
        ontology = build_simple_grasp_ontology()

    print(f"Ontology: {ontology.name}")
    print(f"  Ontology ID: {ontology.ontology_id}")
    print(f"  Description: {ontology.description}")
    print()

    ont_summary = ontology.summary()
    print("Ontology Summary:")
    for key, value in ont_summary.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k2, v2 in value.items():
                print(f"    {k2}: {v2}")
        else:
            print(f"  {key}: {value}")
    print()

    print("Objects:")
    for obj_id, obj in ontology.objects.items():
        print(f"  {obj.name} ({obj.category.value})")
        print(f"    Material: {obj.material.value}")
        print(f"    Fragility: {obj.fragility:.2f}")
        print(f"    Affordances: {obj.list_affordances()}")
        if obj.value_usd > 0:
            print(f"    Value: ${obj.value_usd:.2f}, Damage Cost: ${obj.damage_cost_usd:.2f}")
        print()

    print("Global Constraints:")
    for constraint in ontology.global_constraints:
        print(f"  - {constraint['type']} (priority: {constraint['priority']})")
        print(f"    {constraint['description']}")
    print()

    # Stage 3: Vision Backbone Preview
    print("=" * 70)
    print("3. VISION BACKBONE - Episode Embeddings for Novelty/Regime")
    print("=" * 70)

    from src.vla.backbones.dummy_backbone import DummyBackbone
    from src.valuation.embedding_utils import compute_embedding_novelty, cluster_embeddings_kmeans

    backbone = DummyBackbone(embedding_dim=384)
    print(f"Vision Backbone: {backbone.name}")
    print(f"  Embedding Dimension: {backbone.embedding_dim}")
    print()

    # Simulate episode embeddings
    from PIL import Image
    import numpy as np

    print("Simulating 5 episodes with different visual contexts...")
    episode_embeddings = []
    for i in range(5):
        # Create dummy frames with different colors
        colors = ["red", "green", "blue", "yellow", "purple"]
        frames = [Image.new("RGB", (64, 64), color=colors[i]) for _ in range(10)]
        emb = backbone.encode_sequence(frames)
        episode_embeddings.append(emb)
        print(f"  Episode {i+1}: embedding norm={np.linalg.norm(emb):.4f}")

    # Compute novelty
    print()
    print("Computing embedding novelty scores:")
    reference_embs = episode_embeddings[:3]
    for i in range(3, 5):
        novelty = compute_embedding_novelty(episode_embeddings[i], reference_embs)
        print(f"  Episode {i+1} novelty (vs first 3): {novelty:.4f}")

    # Cluster for regime detection
    assignments, centroids = cluster_embeddings_kmeans(episode_embeddings, n_clusters=2)
    print()
    print(f"Regime clustering (k=2): {assignments}")
    print()

    # Stage 4: Pipeline Manager
    print("=" * 70)
    print("4. PIPELINE MANAGER - 5-Stage Learning Loop")
    print("=" * 70)

    from src.orchestrator.pipeline_manager import (
        create_default_pipeline_manager,
        simulate_pipeline_iteration,
        PipelineStage,
    )

    manager = create_default_pipeline_manager()
    print(f"Pipeline: {manager.name}")
    print(f"  Pipeline ID: {manager.pipeline_id[:8]}...")
    print(f"  Description: {manager.description}")
    print()

    print("Configuration:")
    for key, value in manager.config.items():
        print(f"  {key}: {value}")
    print()

    print("Pipeline Stages:")
    for stage in PipelineStage:
        print(f"  {stage.value}")
    print()

    # Simulate iterations
    print(f"Simulating {args.iterations} pipeline iterations...")
    print()

    for i in range(args.iterations):
        print(f"--- Iteration {i+1} ---")
        iteration = simulate_pipeline_iteration(manager)
        print(f"  Iteration ID: {iteration.iteration_id[:8]}...")
        print(f"  Status: {'Complete' if iteration.is_complete else 'In Progress'}")
        print(f"  Summary Metrics:")
        for key, value in iteration.summary_metrics.items():
            print(f"    {key}: {value:.4f}")
        print()

        if args.verbose:
            print("  Stage Results:")
            for stage_name, result in iteration.stage_results.items():
                print(f"    {stage_name}:")
                print(f"      Status: {result.status.value}")
                print(f"      Duration: {result.duration_seconds:.2f}s")
                if result.metrics:
                    print(f"      Metrics: {result.metrics}")
                if result.recommendations:
                    print(f"      Recommendations: {result.recommendations[:2]}")
            print()

    # Generate advisory report
    print("=" * 70)
    print("5. ADVISORY REPORT - Pipeline Insights")
    print("=" * 70)

    report = manager.generate_advisory_report()
    print(f"Total Iterations: {report['total_iterations']}")
    print()

    print("Progress Overview:")
    progress = report["progress"]
    print(f"  Completed: {progress['completed_iterations']}/{progress['iterations']}")
    print(f"  Failed: {progress['failed_iterations']}")
    print()

    if progress.get("trends"):
        print("Metric Trends (over all iterations):")
        for key, value in progress["trends"].items():
            direction = "↑" if value > 0 else "↓" if value < 0 else "→"
            print(f"  {key}: {direction} {value:.4f}")
        print()

    print("Action Items:")
    for item in report.get("action_items", []):
        print(f"  • {item}")
    print()

    print("Stage Performance Summary:")
    for stage_name, stats in report["stage_summaries"].items():
        if stats["total_runs"] > 0:
            print(f"  {stage_name}:")
            print(f"    Success Rate: {stats['successful']}/{stats['total_runs']}")
            print(f"    Avg Duration: {stats['avg_duration_s']:.2f}s")
            if stats["common_recommendations"]:
                print(f"    Recommendations: {stats['common_recommendations'][0]}")
    print()

    # Preview next iteration
    print("Next Iteration Preview:")
    next_preview = manager.preview_next_iteration()
    print(f"  Iteration Number: {next_preview['iteration_number']}")
    print(f"  Suggested Config Adjustments:")
    for key, value in next_preview["suggested_config"].items():
        if key not in manager.config or manager.config[key] != value:
            print(f"    {key}: {value}")
    print()

    # Save outputs
    os.makedirs(args.out_dir, exist_ok=True)

    # Save task graph
    with open(os.path.join(args.out_dir, "task_graph.json"), "w") as f:
        json.dump(task_graph.to_dict(), f, indent=2)

    # Save ontology
    with open(os.path.join(args.out_dir, "ontology.json"), "w") as f:
        json.dump(ontology.to_dict(), f, indent=2)

    # Save pipeline state
    with open(os.path.join(args.out_dir, "pipeline_state.json"), "w") as f:
        json.dump(manager.to_dict(), f, indent=2)

    # Save advisory report
    with open(os.path.join(args.out_dir, "advisory_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print("=" * 70)
    print("PREVIEW COMPLETE")
    print("=" * 70)
    print()
    print(f"Outputs saved to: {args.out_dir}")
    print("  - task_graph.json: Task decomposition graph")
    print("  - ontology.json: Environment object/affordance ontology")
    print("  - pipeline_state.json: Pipeline manager state with iteration history")
    print("  - advisory_report.json: Advisory insights and recommendations")
    print()
    print("IMPORTANT NOTES:")
    print("  - This is ADVISORY infrastructure only")
    print("  - No actual training was executed")
    print("  - Phase B math and RL loops remain unchanged")
    print("  - Vision backbone uses DummyBackbone (no Meta/HF dependencies required)")
    print("  - All components soft-fail gracefully if optional dependencies missing")
    print()


if __name__ == "__main__":
    main()
