#!/usr/bin/env python3
"""
Semantic run CLI wrapper for orchestrator-driven simulations.
"""
import argparse
import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from src.ontology.store import OntologyStore
from src.orchestrator.semantic_simulation import run_semantic_simulation


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a semantic simulation loop via the orchestrator API.")
    parser.add_argument("--ontology-root", type=str, default="data/ontology")
    parser.add_argument("--intent", type=str, default="")
    parser.add_argument("--tags", nargs="*", default=[], help="Semantic tags for datapack selection")
    parser.add_argument("--robot-family", type=str, default=None)
    parser.add_argument("--objective-hint", type=str, default=None)
    parser.add_argument("--objective-config", type=str, default=None)
    parser.add_argument("--task-id", type=str, default=None)
    parser.add_argument("--notes", type=str, default=None)
    parser.add_argument("--motor-backend", type=str, default="holosoma")
    parser.add_argument("--datapack-limit", type=int, default=1)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--eval-episodes", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--rollout-dir", type=str, default="")
    parser.add_argument("--datapack-output-dir", type=str, default="configs/datapacks")
    parser.add_argument("--run-log", type=str, default="data/logs/semantic_runs.jsonl")
    args = parser.parse_args()

    store = OntologyStore(root_dir=args.ontology_root)
    rollout_dir = Path(args.rollout_dir) if args.rollout_dir else None

    result = run_semantic_simulation(
        store=store,
        intent=args.intent or None,
        tags=args.tags,
        robot_family=args.robot_family,
        objective_hint=args.objective_hint,
        notes=args.notes,
        task_id=args.task_id,
        motor_backend=args.motor_backend,
        objective_config=args.objective_config,
        datapack_limit=args.datapack_limit,
        num_envs=args.num_envs,
        max_steps=args.max_steps,
        eval_episodes=args.eval_episodes,
        seed=args.seed,
        rollout_base_dir=rollout_dir,
        datapack_output_dir=args.datapack_output_dir,
        run_log_path=args.run_log,
    )

    print(f"[semantic_run] status={result.status}")
    if result.scenario:
        print(f"[semantic_run] scenario_id={result.scenario.scenario_id}")
    if result.reason:
        print(f"[semantic_run] reason={result.reason}")
    if result.simulation:
        train_metrics = result.simulation.train_result.econ_metrics
        print(f"[semantic_run] mpl_units_per_hour={train_metrics.get('mpl_units_per_hour', 0):.2f}")
        print(f"[semantic_run] anti_reward_hacking={train_metrics.get('anti_reward_hacking_suspicious', 0.0)}")
        print(f"[semantic_run] new_datapacks={len(result.simulation.labeled_datapacks)}")


if __name__ == "__main__":
    main()
