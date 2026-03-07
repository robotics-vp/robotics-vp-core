#!/usr/bin/env python3
"""Run baseline vs shadow vs shadow+regal ablations on one deterministic input batch."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.shadow_runtime.control_plane import run_shadow_control_plane
from src.shadow_runtime.demo_source import ShadowEpisodeTrace, generate_workcell_shadow_batch


def main() -> None:
    parser = argparse.ArgumentParser(description="Run shadow economic control plane ablations")
    parser.add_argument("--output-dir", type=str, default="artifacts/shadow_econ_ablations")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--objective-profile", type=str, default="balanced_contract")
    parser.add_argument("--pricing-policy", type=str, default="config/pricing/default.yaml")
    parser.add_argument("--timestamp-base", type=str, default="2026-01-01T00:00:00+00:00")
    args = parser.parse_args()

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    traces = generate_workcell_shadow_batch(
        run_id="shadow_ablation_shared",
        seed=args.seed,
        episodes=args.episodes,
        timestamp_base=args.timestamp_base,
    )

    baseline_dir = output_root / "mode_a_baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    baseline_summary = _baseline_summary(traces)
    (baseline_dir / "baseline_summary.json").write_text(json.dumps(baseline_summary, indent=2, sort_keys=True), encoding="utf-8")
    (baseline_dir / "baseline_summary.md").write_text(_baseline_markdown(baseline_summary), encoding="utf-8")

    mode_b = run_shadow_control_plane(
        output_dir=output_root / "mode_b_shadow",
        seed=args.seed,
        episodes=args.episodes,
        objective_profile_id=args.objective_profile,
        pricing_policy_path=args.pricing_policy,
        include_regal=False,
        timestamp_base=args.timestamp_base,
        run_id="shadow_ablation_shared",
        episode_traces=traces,
    )
    mode_c = run_shadow_control_plane(
        output_dir=output_root / "mode_c_shadow_regal",
        seed=args.seed,
        episodes=args.episodes,
        objective_profile_id=args.objective_profile,
        pricing_policy_path=args.pricing_policy,
        include_regal=True,
        timestamp_base=args.timestamp_base,
        run_id="shadow_ablation_shared",
        episode_traces=traces,
    )

    comparison = _comparison_report(
        baseline_summary=baseline_summary,
        mode_b=mode_b.to_dict(),
        mode_c=mode_c.to_dict(),
    )
    comparison_json = output_root / "ablation_comparison.json"
    comparison_md = output_root / "ablation_comparison.md"
    comparison_json.write_text(json.dumps(comparison, indent=2, sort_keys=True), encoding="utf-8")
    comparison_md.write_text(_comparison_markdown(comparison), encoding="utf-8")

    print(f"[shadow_econ_ablations] report={comparison_json}")
    print(f"[shadow_econ_ablations] markdown={comparison_md}")


def _baseline_summary(traces: Sequence[ShadowEpisodeTrace]) -> Dict[str, Any]:
    episodes = [trace.baseline_summary for trace in traces]
    success_rate = sum(1 for episode in episodes if episode.get("success")) / max(len(episodes), 1)
    return {
        "mode": "baseline",
        "episodes": len(episodes),
        "success_rate": success_rate,
        "mean_reward_total": sum(float(episode.get("reward_total", 0.0)) for episode in episodes) / max(len(episodes), 1),
        "mean_throughput_units_per_hour": sum(float(episode.get("throughput_units_per_hour", 0.0)) for episode in episodes) / max(len(episodes), 1),
        "mean_error_rate": sum(float(episode.get("error_rate", 0.0)) for episode in episodes) / max(len(episodes), 1),
        "mean_energy_wh_per_unit": sum(float(episode.get("energy_wh_per_unit", 0.0)) for episode in episodes) / max(len(episodes), 1),
        "mean_quality_score": sum(float(episode.get("quality_score", 0.0)) for episode in episodes) / max(len(episodes), 1),
        "episode_summaries": episodes,
    }


def _comparison_report(
    *,
    baseline_summary: Mapping[str, Any],
    mode_b: Mapping[str, Any],
    mode_c: Mapping[str, Any],
) -> Dict[str, Any]:
    mode_b_summary = dict(mode_b.get("summary", {}) or {})
    mode_c_summary = dict(mode_c.get("summary", {}) or {})
    return {
        "modes": {
            "mode_a_baseline": dict(baseline_summary),
            "mode_b_shadow": mode_b_summary,
            "mode_c_shadow_regal": mode_c_summary,
        },
        "comparison": {
            "shadow_loop_runs": bool(mode_b_summary.get("episodes")) and bool(mode_c_summary.get("episodes")),
            "pricing_traceable": mode_b_summary.get("mean_net_customer_rate", 0.0) > 0.0,
            "regal_changes_interpretable": bool(mode_c_summary.get("deploy_recommendations", {})),
            "baseline_success_rate": baseline_summary.get("success_rate", 0.0),
            "shadow_mean_net_customer_rate": mode_b_summary.get("mean_net_customer_rate", 0.0),
            "shadow_regal_mean_net_customer_rate": mode_c_summary.get("mean_net_customer_rate", 0.0),
            "shadow_regal_pricing_recommendations": mode_c_summary.get("pricing_recommendations", {}),
            "shadow_regal_datapack_recommendations": mode_c_summary.get("datapack_recommendations", {}),
        },
    }


def _baseline_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Mode A: Baseline",
        "",
        f"- Episodes: `{summary['episodes']}`",
        f"- Success rate: `{summary['success_rate']:.2%}`",
        f"- Mean reward total: `{summary['mean_reward_total']:.2f}`",
        f"- Mean throughput: `{summary['mean_throughput_units_per_hour']:.2f}` units/hour",
        f"- Mean error rate: `{summary['mean_error_rate']:.3f}`",
        f"- Mean energy: `{summary['mean_energy_wh_per_unit']:.3f}` Wh/unit",
    ]
    return "\n".join(lines) + "\n"


def _comparison_markdown(report: Mapping[str, Any]) -> str:
    comparison = report["comparison"]
    lines = [
        "# Shadow Econ Ablations",
        "",
        "| Mode | Success Rate | Mean Net Rate | Pricing Recs | Datapack Recs |",
        "|---|---:|---:|---|---|",
        "| A baseline | {0:.2%} | n/a | n/a | n/a |".format(report["modes"]["mode_a_baseline"]["success_rate"]),
        "| B shadow | {0:.2%} | {1:.2f} | n/a | n/a |".format(
            report["modes"]["mode_b_shadow"]["success_rate"],
            report["modes"]["mode_b_shadow"]["mean_net_customer_rate"],
        ),
        "| C shadow+regal | {0:.2%} | {1:.2f} | {2} | {3} |".format(
            report["modes"]["mode_c_shadow_regal"]["success_rate"],
            report["modes"]["mode_c_shadow_regal"]["mean_net_customer_rate"],
            report["modes"]["mode_c_shadow_regal"]["pricing_recommendations"],
            report["modes"]["mode_c_shadow_regal"]["datapack_recommendations"],
        ),
        "",
        f"- Shadow loop runs: `{comparison['shadow_loop_runs']}`",
        f"- Pricing traceable: `{comparison['pricing_traceable']}`",
        f"- Regal changes interpretable: `{comparison['regal_changes_interpretable']}`",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
