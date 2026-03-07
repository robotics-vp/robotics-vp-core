#!/usr/bin/env python3
"""Runnable demo for economically gated inferential training decisions."""
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
from src.shadow_runtime.control_plane import run_shadow_control_plane


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the inferential budget gate demo")
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument("--replay-dataset-dir", type=str, default=None)
    parser.add_argument("--shadow-run-dir", type=str, default=None)
    parser.add_argument("--generate-shadow-run", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--promotion-policy", default="configs/regality/promotion_default.yaml", type=str)
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
        promotion_policy_path=args.promotion_policy,
    )
    payload = {
        "summary": advisory["adaptation_budget"]["summary"],
        "decisions": advisory["adaptation_budget"]["decisions"],
        "live_queue_selection": advisory["live_queue_selection"],
        "promotion_policy": advisory["promotion_policy"],
    }
    json_path = output_root / "inferential_budget_gate_demo.json"
    md_path = output_root / "inferential_budget_gate_demo.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))


def _markdown(payload: dict) -> str:
    lines = [
        "# Inferential Budget Gate Demo",
        "",
        f"- Adapt now: {payload['summary']['adapt_now']}",
        f"- Collect more data: {payload['summary']['collect_more_data']}",
        f"- Require review: {payload['summary']['require_review']}",
        f"- No-op: {payload['summary']['no_op']}",
        "",
        "## Decisions",
    ]
    for decision in payload["decisions"]:
        lines.extend(
            [
                f"### {decision['artifact_summary'].get('episode_id')}",
                f"- Decision: {decision['decision']}",
                f"- Training mode: {decision['recommended_training_mode']}",
                f"- Net benefit: {decision['net_benefit']:.4f}",
                f"- Reasons: {', '.join(decision['reasons'])}",
                "",
            ]
        )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
