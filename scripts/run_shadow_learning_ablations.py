#!/usr/bin/env python3
"""Compare heuristic shadow runtime against replay-learning advisory modes."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.learning.replay_policy_trainer import evaluate_replay_policy, train_replay_policy
from src.orchestrator.shadow_advisory import build_shadow_advisory_output
from src.replay.dataset import ReplayDatasetBuilder, load_replay_dataset
from src.shadow_runtime.advisors import AdvisorMode, DataValueAdvisor, PolicyAdvisor, PricingAdvisor, RegalSupportAdvisor
from src.shadow_runtime.control_plane import run_shadow_control_plane
from src.utils.config_digest import sha256_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Run shadow replay-learning ablations")
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--objective-profile", type=str, default="balanced_contract")
    args = parser.parse_args()

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    shadow_dir = output_root / "shadow_run"
    replay_dir = output_root / "replay_dataset"
    policy_dir = output_root / "replay_policy"
    policy_eval_dir = output_root / "replay_policy_eval"
    model_dir = output_root / "shadow_models"

    run_shadow_control_plane(
        output_dir=shadow_dir,
        seed=args.seed,
        episodes=args.episodes,
        objective_profile_id=args.objective_profile,
        include_regal=True,
        timestamp_base="2026-01-01T00:00:00+00:00",
    )
    ReplayDatasetBuilder().add_shadow_run(shadow_dir).write(replay_dir)
    policy_train = train_replay_policy(
        dataset_dir=replay_dir,
        config_path="configs/replay_policy/cpu_smoke.yaml",
        output_dir=policy_dir,
    )
    policy_eval = evaluate_replay_policy(
        dataset_dir=replay_dir,
        checkpoint_path=policy_train.best_checkpoint_path,
        output_dir=policy_eval_dir,
    )

    # Train learned episode models via the same helpers used by the standalone script.
    from src.learning.data_value_models import train_data_value_model
    from src.learning.pricing_models import train_pricing_delta_model
    from src.learning.regal_support_models import train_regal_support_model
    import yaml

    model_dir.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(Path("configs/shadow_models/cpu_smoke.yaml").read_text(encoding="utf-8")) or {}
    dataset = load_replay_dataset(replay_dir)
    hidden_dim = int(config.get("training", {}).get("hidden_dim", 64))
    epochs = int(config.get("training", {}).get("epochs", 8))
    lr = float(config.get("training", {}).get("lr", 1e-3))
    seed = int(config.get("training", {}).get("seed", 42))
    pricing_model, pricing_metrics = train_pricing_delta_model(dataset.episodes, seed=seed, epochs=epochs, lr=lr, hidden_dim=hidden_dim)
    data_model, data_metrics = train_data_value_model(dataset.episodes, seed=seed, epochs=epochs, lr=lr, hidden_dim=hidden_dim)
    regal_model, regal_metrics = train_regal_support_model(dataset.episodes, seed=seed, epochs=epochs, lr=lr, hidden_dim=hidden_dim)
    pricing_ckpt = model_dir / "pricing_delta.pt"
    data_ckpt = model_dir / "data_value.pt"
    regal_ckpt = model_dir / "regal_support.pt"
    _save_residual_checkpoint(pricing_ckpt, pricing_model, pricing_metrics, config, dataset.manifest.dataset_digest, hidden_dim)
    _save_residual_checkpoint(data_ckpt, data_model, data_metrics, config, dataset.manifest.dataset_digest, hidden_dim)
    _save_residual_checkpoint(regal_ckpt, regal_model, regal_metrics, config, dataset.manifest.dataset_digest, hidden_dim)

    mode_payloads = {
        "mode_a_heuristic_only": build_shadow_advisory_output(
            replay_dataset_dir=str(replay_dir),
            policy_advisor=PolicyAdvisor(mode=AdvisorMode.HEURISTIC_ONLY),
            pricing_advisor=PricingAdvisor(mode=AdvisorMode.HEURISTIC_ONLY),
            data_value_advisor=DataValueAdvisor(mode=AdvisorMode.HEURISTIC_ONLY),
            regal_support_advisor=RegalSupportAdvisor(mode=AdvisorMode.HEURISTIC_ONLY),
        ),
        "mode_b_replay_bc_eval": build_shadow_advisory_output(
            replay_dataset_dir=str(replay_dir),
            policy_advisor=PolicyAdvisor(
                mode=AdvisorMode.HEURISTIC_LEARNED_COMPARE_ONLY,
                checkpoint_path=policy_train.best_checkpoint_path,
            ),
            pricing_advisor=PricingAdvisor(mode=AdvisorMode.HEURISTIC_ONLY),
            data_value_advisor=DataValueAdvisor(mode=AdvisorMode.HEURISTIC_ONLY),
            regal_support_advisor=RegalSupportAdvisor(mode=AdvisorMode.HEURISTIC_ONLY),
        ),
        "mode_c_pricing_value_compare": build_shadow_advisory_output(
            replay_dataset_dir=str(replay_dir),
            policy_advisor=PolicyAdvisor(mode=AdvisorMode.HEURISTIC_LEARNED_COMPARE_ONLY, checkpoint_path=policy_train.best_checkpoint_path),
            pricing_advisor=PricingAdvisor(
                mode=AdvisorMode.HEURISTIC_LEARNED_COMPARE_ONLY,
                checkpoint_path=pricing_ckpt,
            ),
            data_value_advisor=DataValueAdvisor(
                mode=AdvisorMode.HEURISTIC_LEARNED_COMPARE_ONLY,
                checkpoint_path=data_ckpt,
            ),
            regal_support_advisor=RegalSupportAdvisor(mode=AdvisorMode.HEURISTIC_ONLY),
        ),
        "mode_d_pricing_residual_regal_support": build_shadow_advisory_output(
            replay_dataset_dir=str(replay_dir),
            policy_advisor=PolicyAdvisor(mode=AdvisorMode.HEURISTIC_LEARNED_COMPARE_ONLY, checkpoint_path=policy_train.best_checkpoint_path),
            pricing_advisor=PricingAdvisor(
                mode=AdvisorMode.HEURISTIC_LEARNED_RESIDUAL,
                checkpoint_path=pricing_ckpt,
            ),
            data_value_advisor=DataValueAdvisor(
                mode=AdvisorMode.HEURISTIC_LEARNED_COMPARE_ONLY,
                checkpoint_path=data_ckpt,
            ),
            regal_support_advisor=RegalSupportAdvisor(
                mode=AdvisorMode.HEURISTIC_LEARNED_COMPARE_ONLY,
                checkpoint_path=regal_ckpt,
            ),
        ),
    }

    for mode_name, payload in mode_payloads.items():
        mode_dir = output_root / mode_name
        mode_dir.mkdir(parents=True, exist_ok=True)
        (mode_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        (mode_dir / "summary.md").write_text(_mode_markdown(mode_name, payload), encoding="utf-8")

    comparison = {
        "dataset_digest": dataset.manifest.dataset_digest,
        "policy_eval": policy_eval["metrics"],
        "modes": {
            mode_name: {
                "summary": payload["summary"],
                "sample_episode": payload["episodes"][0] if payload["episodes"] else {},
            }
            for mode_name, payload in mode_payloads.items()
        },
        "comparison": {
            "replay_ingestion_runs": dataset.manifest.num_steps > 0,
            "bc_policy_runs": policy_eval["metrics"]["count"] > 0,
            "learned_outputs_inspectable": True,
            "pricing_behavior_changes": (
                mode_payloads["mode_c_pricing_value_compare"]["episodes"][0]["pricing_advisor"]["learned_output"]
                != {}
                and mode_payloads["mode_d_pricing_residual_regal_support"]["episodes"][0]["pricing_advisor"]["applied_output"]
                != mode_payloads["mode_a_heuristic_only"]["episodes"][0]["pricing_advisor"]["applied_output"]
            ),
            "regal_support_changes": (
                mode_payloads["mode_d_pricing_residual_regal_support"]["episodes"][0]["regal_support_advisor"]["learned_output"]
                != {}
            ),
        },
    }
    comparison_json = output_root / "shadow_learning_ablation_comparison.json"
    comparison_md = output_root / "shadow_learning_ablation_comparison.md"
    comparison_json.write_text(json.dumps(comparison, indent=2, sort_keys=True), encoding="utf-8")
    comparison_md.write_text(_comparison_markdown(comparison), encoding="utf-8")
    print(json.dumps(comparison["comparison"], indent=2, sort_keys=True))


def _mode_markdown(mode_name: str, payload: dict) -> str:
    lines = [
        f"# {mode_name}",
        "",
        f"- Episodes: {payload['summary']['episodes']}",
        f"- Sampling priorities: {payload['summary']['sampling_priorities']}",
        f"- Collect more data: {payload['summary']['collect_more_data_count']}",
        f"- Retrain: {payload['summary']['retrain_count']}",
        "",
    ]
    for episode in payload["episodes"]:
        lines.extend(
            [
                f"## {episode['episode_id']}",
                f"- Sampling priority: {episode['sampling_priority']}",
                f"- Slice weight multiplier: {episode['slice_weight_multiplier']:.2f}",
                f"- Replay tags: {', '.join(episode['replay_queue_tags'])}",
                "",
            ]
        )
    return "\n".join(lines)


def _comparison_markdown(comparison: dict) -> str:
    lines = [
        "# Shadow Learning Ablations",
        "",
        "## Summary",
        f"- Replay ingestion runs: {comparison['comparison']['replay_ingestion_runs']}",
        f"- BC policy runs: {comparison['comparison']['bc_policy_runs']}",
        f"- Learned outputs inspectable: {comparison['comparison']['learned_outputs_inspectable']}",
        f"- Pricing behavior changes: {comparison['comparison']['pricing_behavior_changes']}",
        f"- Regal support changes: {comparison['comparison']['regal_support_changes']}",
        "",
        "## Policy Eval",
        f"- Count: {comparison['policy_eval']['count']}",
        f"- MSE: {comparison['policy_eval']['mse']:.4f}",
        f"- MAE: {comparison['policy_eval']['mae']:.4f}",
    ]
    return "\n".join(lines)


def _save_residual_checkpoint(path: Path, model, metrics: dict, config: dict, dataset_digest: str, hidden_dim: int) -> None:
    import torch

    first_weight = next(iter(model.state_dict().values()))
    input_dim = int(first_weight.shape[-1])
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metrics": metrics,
            "training_config": config,
            "dataset_digest": dataset_digest,
            "config_digest": sha256_json(config),
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "model_version": metrics.get("model_version"),
        },
        path,
    )


if __name__ == "__main__":
    main()
