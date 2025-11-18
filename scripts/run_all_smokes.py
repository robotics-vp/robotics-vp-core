#!/usr/bin/env python3
"""
Run a standard bundle of smoke tests for the meta/orchestrator layer.

This is read-only and does not modify Phase B math or reward behavior.
"""
import subprocess
import sys


SMOKES = [
    ["python3", "scripts/smoke_test_dependency_hierarchy.py"],
    ["python3", "scripts/smoke_test_pareto_frontier.py"],
    ["python3", "scripts/smoke_test_semantic_feedback_loop.py"],
    ["python3", "scripts/smoke_test_reward_builder.py"],
    ["python3", "scripts/smoke_test_reward_heads.py"],
    ["python3", "scripts/smoke_test_datapack_rl_ingestion.py"],
    ["python3", "scripts/smoke_test_stage1_pipeline.py"],
    ["python3", "scripts/smoke_test_stage1_to_rl_sampling.py"],
    ["python3", "scripts/smoke_test_sima2_semantic_extraction.py"],
    ["python3", "scripts/smoke_test_ontology_update_engine.py"],
    ["python3", "scripts/smoke_test_task_graph_refiner.py"],
    ["python3", "scripts/smoke_test_stage2_e2e_pipeline.py"],
    ["python3", "scripts/smoke_test_stage2_4_semantic_tag_propagation.py"],
    ["python3", "scripts/smoke_test_stage3_sampler.py"],
    ["python3", "scripts/smoke_test_stage3_curriculum.py"],
    ["python3", "scripts/smoke_test_stage3_training_integration.py"],
    ["python3", "scripts/smoke_test_ontology_store.py"],
    ["python3", "scripts/smoke_test_ontology_adapters.py"],
    ["python3", "scripts/smoke_test_episode_logging_and_econ_vector.py"],
]

OPTIONAL = [
    ["python3", "scripts/smoke_test_vision_backbone.py"],
]


def main():
    failures = []
    for cmd in SMOKES:
        print(f"[run_all_smokes] Running {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError:
            failures.append(" ".join(cmd))
    for cmd in OPTIONAL:
        try:
            subprocess.run(cmd, check=True)
        except FileNotFoundError:
            print(f"[run_all_smokes] Optional script missing: {' '.join(cmd)}")
        except subprocess.CalledProcessError:
            failures.append(" ".join(cmd))
    if failures:
        print("[run_all_smokes] Failures detected:")
        for f in failures:
            print(f"  {f}")
        sys.exit(1)
    print("[run_all_smokes] All smokes passed.")


if __name__ == "__main__":
    main()
