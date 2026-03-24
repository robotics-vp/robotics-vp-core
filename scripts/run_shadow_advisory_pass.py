#!/usr/bin/env python3
"""Emit trainer/orchestrator advisory sidecars from shadow replay data."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.orchestrator.shadow_advisory import build_shadow_advisory_output
from src.replay.dataset import ReplayDatasetBuilder
from src.shadow_runtime.advisors import AdvisorMode, DataValueAdvisor, PolicyAdvisor, PricingAdvisor, RegalSupportAdvisor
from src.shadow_runtime.control_plane import run_shadow_control_plane


def main() -> None:
    parser = argparse.ArgumentParser(description="Run advisory-only shadow pass for trainer/orchestrator wiring")
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument("--replay-dataset-dir", type=str, default=None)
    parser.add_argument("--shadow-run-dir", type=str, default=None)
    parser.add_argument("--generate-shadow-run", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--policy-mode", type=str, default=AdvisorMode.HEURISTIC_ONLY.value)
    parser.add_argument("--pricing-mode", type=str, default=AdvisorMode.HEURISTIC_ONLY.value)
    parser.add_argument("--data-value-mode", type=str, default=AdvisorMode.HEURISTIC_ONLY.value)
    parser.add_argument("--regal-support-mode", type=str, default=AdvisorMode.HEURISTIC_ONLY.value)
    parser.add_argument("--policy-checkpoint", type=str, default=None)
    parser.add_argument("--pricing-checkpoint", type=str, default=None)
    parser.add_argument("--data-value-checkpoint", type=str, default=None)
    parser.add_argument("--regal-support-checkpoint", type=str, default=None)
    parser.add_argument("--promotion-policy", type=str, default="configs/regality/promotion_default.yaml")
    parser.add_argument("--receipt-label-dir", type=str, default=None)
    parser.add_argument("--receipt-label-mode", type=str, default="synthetic_shadow")
    parser.add_argument("--epiplexity-overlays", type=str, default=None)
    args = parser.parse_args()

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    replay_dir = Path(args.replay_dataset_dir) if args.replay_dataset_dir else output_root / "replay_dataset"
    if args.generate_shadow_run or args.shadow_run_dir:
        shadow_dir = Path(args.shadow_run_dir) if args.shadow_run_dir else output_root / "shadow_run"
        if args.generate_shadow_run:
            run_shadow_control_plane(
                output_dir=shadow_dir,
                seed=args.seed,
                episodes=args.episodes,
                objective_profile_id="balanced_contract",
                include_regal=True,
                timestamp_base="2026-01-01T00:00:00+00:00",
            )
        if not replay_dir.exists():
            ReplayDatasetBuilder().add_shadow_run(shadow_dir).write(replay_dir)

    advisory = build_shadow_advisory_output(
        replay_dataset_dir=str(replay_dir),
        policy_advisor=PolicyAdvisor(mode=args.policy_mode, checkpoint_path=args.policy_checkpoint),
        pricing_advisor=PricingAdvisor(mode=args.pricing_mode, checkpoint_path=args.pricing_checkpoint),
        data_value_advisor=DataValueAdvisor(mode=args.data_value_mode, checkpoint_path=args.data_value_checkpoint),
        regal_support_advisor=RegalSupportAdvisor(mode=args.regal_support_mode, checkpoint_path=args.regal_support_checkpoint),
        promotion_policy_path=args.promotion_policy,
        receipt_label_dir=args.receipt_label_dir,
        receipt_label_mode=args.receipt_label_mode,
        epiplexity_overlay_path=args.epiplexity_overlays,
    )

    json_path = output_root / "shadow_advisory.json"
    md_path = output_root / "shadow_advisory.md"
    queue_json_path = output_root / "live_queue_selection.json"
    budget_json_path = output_root / "adaptation_budget.json"
    json_path.write_text(json.dumps(advisory, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(_advisory_markdown(advisory), encoding="utf-8")
    queue_json_path.write_text(json.dumps(advisory["live_queue_selection"], indent=2, sort_keys=True), encoding="utf-8")
    budget_json_path.write_text(json.dumps(advisory["adaptation_budget"], indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(advisory["summary"], indent=2, sort_keys=True))


def _advisory_markdown(advisory: dict) -> str:
    lines = [
        "# Shadow Advisory Pass",
        "",
        f"- Episodes: {advisory['summary']['episodes']}",
        f"- Collect more data: {advisory['summary']['collect_more_data_count']}",
        f"- Retrain: {advisory['summary']['retrain_count']}",
        f"- Receipt labels: {advisory['summary']['receipt_label_coverage']['total_labels']}",
        "",
        "## Episode Decisions",
    ]
    for episode in advisory["episodes"]:
        lines.extend(
            [
                f"### {episode['episode_id']}",
                f"- Sampling priority: {episode['sampling_priority']} ({episode['sampling_priority_score']:.2f})",
                f"- Slice weight multiplier: {episode['slice_weight_multiplier']:.2f}",
                f"- Replay queue tags: {', '.join(episode['replay_queue_tags'])}",
                f"- Replay action: {episode['replay_action']}",
                f"- Collect more data: {episode['collect_more_data']}",
                f"- Retrain: {episode['retrain']}",
                f"- Inferential budget decision: {episode['inferential_budget_decision']['decision']}",
                "",
            ]
        )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
