#!/usr/bin/env python3
"""
Process Reward Ablation Report.

Answers four key questions to justify the module's existence:
1. Does confidence predict success? corr(conf_p10, success), corr(conf_mean, success)
2. Does disagreement/entropy predict failure? corr(disagreement_mean, success), corr(entropy_mean, success)
3. If we gate low-confidence episodes, does MPL uplift improve?
4. Does "process_reward_quality" sampling outperform baselines?

Usage:
    python scripts/report_process_reward_ablation.py \
        --episodes-path results/demo_sim/episodes.jsonl \
        --output-dir results/process_reward_ablation

    # Or infer episodes/output from a run directory:
    python scripts/report_process_reward_ablation.py \
        --run-dir results/demo_sim

Inputs: JSONL with process reward fields (phi_star_*, conf_*, disagreement_*, entropy_*, etc.)
Outputs: JSON report + CSV summary + console analysis
"""
import argparse
import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))


@dataclass
class EpisodeRecord:
    """Parsed episode with process reward fields."""
    episode_id: str
    success: bool
    total_reward: float
    mpl_estimate: float = 0.0
    # Process reward fields
    phi_star_mean: float = 0.0
    phi_star_final: float = 0.0
    phi_star_delta: float = 0.0
    conf_mean: float = 0.5
    conf_p10: float = 0.5
    conf_min: float = 0.5
    r_shape_sum: float = 0.0
    disagreement_mean: float = 0.0
    disagreement_max: float = 0.0
    entropy_mean: float = 0.0
    entropy_max: float = 0.0
    phi_B_disabled: bool = False
    # Upstream quality
    scene_ir_quality: float = 0.0
    motion_quality: float = 0.0
    mhn_plausibility: float = 1.0
    # Extra metadata
    raw: Dict[str, Any] = field(default_factory=dict)


def load_episodes(path: Path) -> List[EpisodeRecord]:
    """Load episodes from JSONL file."""
    if not path.exists():
        return []
    episodes = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                ep = EpisodeRecord(
                    episode_id=str(data.get("episode_id", "")),
                    success=bool(data.get("success", False)),
                    total_reward=float(data.get("total_reward", 0.0)),
                    mpl_estimate=float(data.get("mpl_estimate", 0.0)),
                    phi_star_mean=float(data.get("phi_star_mean", 0.0)),
                    phi_star_final=float(data.get("phi_star_final", 0.0)),
                    phi_star_delta=float(data.get("phi_star_delta", 0.0)),
                    conf_mean=float(data.get("conf_mean", 0.5)),
                    conf_p10=float(data.get("conf_p10", 0.5)),
                    conf_min=float(data.get("conf_min", 0.5)),
                    r_shape_sum=float(data.get("r_shape_sum", 0.0)),
                    disagreement_mean=float(data.get("disagreement_mean", 0.0)),
                    disagreement_max=float(data.get("disagreement_max", 0.0)),
                    entropy_mean=float(data.get("entropy_mean", 0.0)),
                    entropy_max=float(data.get("entropy_max", 0.0)),
                    phi_B_disabled=bool(data.get("phi_B_disabled", False)),
                    scene_ir_quality=float(data.get("scene_ir_quality", 0.0)),
                    motion_quality=float(data.get("motion_quality", 0.0)),
                    mhn_plausibility=float(data.get("mhn_plausibility", 1.0)),
                    raw=data,
                )
                episodes.append(ep)
            except (json.JSONDecodeError, ValueError):
                continue
    return episodes


def has_process_reward_data(episodes: List[EpisodeRecord]) -> bool:
    """Check if episodes have process reward fields populated."""
    if not episodes:
        return False
    # Check if at least some episodes have non-default values
    non_default = 0
    for ep in episodes:
        if ep.conf_mean != 0.5 or ep.phi_star_delta != 0.0 or ep.disagreement_mean != 0.0:
            non_default += 1
    return non_default > len(episodes) * 0.1  # At least 10% have data


def safe_corr(x: np.ndarray, y: np.ndarray) -> Tuple[float, str]:
    """Compute correlation with safety checks. Returns (corr, interpretation)."""
    if len(x) < 5:
        return 0.0, "insufficient_data"
    if np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return 0.0, "no_variance"
    corr = float(np.corrcoef(x, y)[0, 1])
    if np.isnan(corr):
        return 0.0, "nan"
    # Interpretation
    abs_corr = abs(corr)
    if abs_corr >= 0.7:
        strength = "strong"
    elif abs_corr >= 0.4:
        strength = "moderate"
    elif abs_corr >= 0.2:
        strength = "weak"
    else:
        strength = "negligible"
    direction = "positive" if corr > 0 else "negative"
    return corr, f"{strength}_{direction}"


def analyze_confidence_vs_success(episodes: List[EpisodeRecord]) -> Dict[str, Any]:
    """Q1: Does confidence predict success?"""
    successes = np.array([float(ep.success) for ep in episodes])
    conf_means = np.array([ep.conf_mean for ep in episodes])
    conf_p10s = np.array([ep.conf_p10 for ep in episodes])
    conf_mins = np.array([ep.conf_min for ep in episodes])

    corr_mean, interp_mean = safe_corr(conf_means, successes)
    corr_p10, interp_p10 = safe_corr(conf_p10s, successes)
    corr_min, interp_min = safe_corr(conf_mins, successes)

    # Segment analysis: success rate by confidence bucket
    high_conf_mask = conf_means >= 0.6
    mid_conf_mask = (conf_means >= 0.3) & (conf_means < 0.6)
    low_conf_mask = conf_means < 0.3

    def success_rate(mask: np.ndarray) -> float:
        if mask.sum() == 0:
            return 0.0
        return float(successes[mask].mean())

    return {
        "question": "Does confidence predict success?",
        "correlations": {
            "conf_mean_vs_success": {"value": corr_mean, "interpretation": interp_mean},
            "conf_p10_vs_success": {"value": corr_p10, "interpretation": interp_p10},
            "conf_min_vs_success": {"value": corr_min, "interpretation": interp_min},
        },
        "segmentation": {
            "high_conf_success_rate": success_rate(high_conf_mask),
            "mid_conf_success_rate": success_rate(mid_conf_mask),
            "low_conf_success_rate": success_rate(low_conf_mask),
            "high_conf_count": int(high_conf_mask.sum()),
            "mid_conf_count": int(mid_conf_mask.sum()),
            "low_conf_count": int(low_conf_mask.sum()),
        },
        "verdict": "POSITIVE" if corr_mean > 0.2 or corr_p10 > 0.2 else "NEUTRAL" if corr_mean > 0 else "NEGATIVE",
    }


def analyze_disagreement_entropy_vs_success(episodes: List[EpisodeRecord]) -> Dict[str, Any]:
    """Q2: Does disagreement/entropy predict failure?"""
    successes = np.array([float(ep.success) for ep in episodes])
    disagreements = np.array([ep.disagreement_mean for ep in episodes])
    entropies = np.array([ep.entropy_mean for ep in episodes])

    corr_disag, interp_disag = safe_corr(disagreements, successes)
    corr_entropy, interp_entropy = safe_corr(entropies, successes)

    # We expect NEGATIVE correlation (high disagreement/entropy → failure)
    # Segment by disagreement
    high_disag_mask = disagreements >= np.percentile(disagreements, 75) if len(disagreements) > 4 else np.zeros(len(disagreements), dtype=bool)
    low_disag_mask = disagreements <= np.percentile(disagreements, 25) if len(disagreements) > 4 else np.zeros(len(disagreements), dtype=bool)

    def success_rate(mask: np.ndarray) -> float:
        if mask.sum() == 0:
            return 0.0
        return float(successes[mask].mean())

    return {
        "question": "Does disagreement/entropy predict failure?",
        "correlations": {
            "disagreement_vs_success": {"value": corr_disag, "interpretation": interp_disag},
            "entropy_vs_success": {"value": corr_entropy, "interpretation": interp_entropy},
        },
        "segmentation": {
            "high_disagreement_success_rate": success_rate(high_disag_mask),
            "low_disagreement_success_rate": success_rate(low_disag_mask),
            "high_disagreement_count": int(high_disag_mask.sum()),
            "low_disagreement_count": int(low_disag_mask.sum()),
        },
        "verdict": "POSITIVE" if corr_disag < -0.2 or corr_entropy < -0.2 else "NEUTRAL" if corr_disag < 0 else "NEGATIVE",
        "note": "Negative correlation is expected (high disagreement → lower success)",
    }


def analyze_gating_mpl_uplift(episodes: List[EpisodeRecord]) -> Dict[str, Any]:
    """Q3: If we gate low-confidence episodes, does MPL uplift improve?"""
    if not episodes:
        return {"question": "Gating analysis", "error": "no_data"}

    mpls = np.array([ep.mpl_estimate for ep in episodes])
    conf_p10s = np.array([ep.conf_p10 for ep in episodes])
    disagreements = np.array([ep.disagreement_mean for ep in episodes])

    all_mpl = float(np.mean(mpls)) if len(mpls) > 0 else 0.0

    # Gate by conf_p10
    thresholds = [0.1, 0.2, 0.3]
    conf_gating_results = {}
    for thresh in thresholds:
        mask = conf_p10s >= thresh
        if mask.sum() > 0:
            gated_mpl = float(np.mean(mpls[mask]))
            uplift = (gated_mpl - all_mpl) / (all_mpl + 1e-8) * 100
            conf_gating_results[f"gate_conf_p10>={thresh}"] = {
                "gated_mpl": gated_mpl,
                "uplift_pct": uplift,
                "episodes_retained": int(mask.sum()),
                "episodes_dropped": int((~mask).sum()),
            }
        else:
            conf_gating_results[f"gate_conf_p10>={thresh}"] = {"error": "no_episodes_above_threshold"}

    # Gate by disagreement (drop high disagreement)
    disag_percentiles = [90, 80]
    disag_gating_results = {}
    for pct in disag_percentiles:
        if len(disagreements) > 4:
            thresh = np.percentile(disagreements, pct)
            mask = disagreements <= thresh
            if mask.sum() > 0:
                gated_mpl = float(np.mean(mpls[mask]))
                uplift = (gated_mpl - all_mpl) / (all_mpl + 1e-8) * 100
                disag_gating_results[f"drop_top_{100-pct}pct_disagreement"] = {
                    "gated_mpl": gated_mpl,
                    "uplift_pct": uplift,
                    "episodes_retained": int(mask.sum()),
                    "episodes_dropped": int((~mask).sum()),
                }

    # Best gating strategy
    best_strategy = None
    best_uplift = 0.0
    for strategy, result in {**conf_gating_results, **disag_gating_results}.items():
        if isinstance(result, dict) and "uplift_pct" in result:
            if result["uplift_pct"] > best_uplift:
                best_uplift = result["uplift_pct"]
                best_strategy = strategy

    return {
        "question": "If we gate low-confidence episodes, does MPL uplift improve?",
        "baseline_mpl": all_mpl,
        "total_episodes": len(episodes),
        "confidence_gating": conf_gating_results,
        "disagreement_gating": disag_gating_results,
        "best_strategy": best_strategy,
        "best_uplift_pct": best_uplift,
        "verdict": "POSITIVE" if best_uplift > 5.0 else "NEUTRAL" if best_uplift > 0 else "NEGATIVE",
    }


def analyze_sampling_strategy_comparison(episodes: List[EpisodeRecord]) -> Dict[str, Any]:
    """Q4: Does process_reward_quality sampling outperform baselines?

    Simulates different sampling strategies and compares expected outcomes.
    """
    if len(episodes) < 10:
        return {"question": "Sampling strategy comparison", "error": "insufficient_data"}

    # Compute weights for each strategy
    weights_uniform = np.ones(len(episodes))
    weights_conf = np.array([max(ep.conf_mean, 0.1) for ep in episodes])
    weights_progress = np.array([max(1.0 + ep.phi_star_delta, 0.1) for ep in episodes])
    weights_quality = np.array([
        max(ep.conf_mean * (1.0 + max(0.0, ep.phi_star_delta)), 0.1)
        for ep in episodes
    ])

    # Normalize weights
    def normalize(w: np.ndarray) -> np.ndarray:
        return w / (w.sum() + 1e-8)

    weights_uniform = normalize(weights_uniform)
    weights_conf = normalize(weights_conf)
    weights_progress = normalize(weights_progress)
    weights_quality = normalize(weights_quality)

    # Compute weighted outcomes
    successes = np.array([float(ep.success) for ep in episodes])
    mpls = np.array([ep.mpl_estimate for ep in episodes])

    def weighted_outcome(weights: np.ndarray) -> Dict[str, float]:
        return {
            "expected_success_rate": float(np.sum(weights * successes)),
            "expected_mpl": float(np.sum(weights * mpls)),
        }

    strategies = {
        "uniform": weighted_outcome(weights_uniform),
        "process_reward_conf": weighted_outcome(weights_conf),
        "process_reward_progress": weighted_outcome(weights_progress),
        "process_reward_quality": weighted_outcome(weights_quality),
    }

    # Find best strategy
    baseline_success = strategies["uniform"]["expected_success_rate"]
    baseline_mpl = strategies["uniform"]["expected_mpl"]

    best_success_strategy = max(strategies.items(), key=lambda kv: kv[1]["expected_success_rate"])[0]
    best_mpl_strategy = max(strategies.items(), key=lambda kv: kv[1]["expected_mpl"])[0]

    quality_vs_uniform_success = (
        (strategies["process_reward_quality"]["expected_success_rate"] - baseline_success)
        / (baseline_success + 1e-8) * 100
    )
    quality_vs_uniform_mpl = (
        (strategies["process_reward_quality"]["expected_mpl"] - baseline_mpl)
        / (baseline_mpl + 1e-8) * 100
    )

    return {
        "question": "Does process_reward_quality sampling outperform baselines?",
        "strategies": strategies,
        "best_for_success": best_success_strategy,
        "best_for_mpl": best_mpl_strategy,
        "quality_vs_uniform": {
            "success_uplift_pct": quality_vs_uniform_success,
            "mpl_uplift_pct": quality_vs_uniform_mpl,
        },
        "verdict": "POSITIVE" if quality_vs_uniform_success > 5 or quality_vs_uniform_mpl > 5 else "NEUTRAL" if quality_vs_uniform_success > 0 else "NEGATIVE",
    }


def extract_worst_episodes(
    episodes: List[EpisodeRecord],
    top_n: int = 20,
) -> Dict[str, List[Dict[str, Any]]]:
    """Extract worst episodes by various metrics for inspection.

    Returns lists of episode IDs + metrics for:
    - Lowest conf_p10 (unreliable)
    - Highest disagreement (conflicting perspectives)
    - Negative delta (regression)
    - Stagnant (high conf + zero delta)
    """
    if len(episodes) < 5:
        return {}

    def ep_summary(ep: EpisodeRecord) -> Dict[str, Any]:
        return {
            "episode_id": ep.episode_id,
            "success": ep.success,
            "conf_mean": round(ep.conf_mean, 3),
            "conf_p10": round(ep.conf_p10, 3),
            "phi_star_delta": round(ep.phi_star_delta, 3),
            "disagreement_mean": round(ep.disagreement_mean, 3),
            "mpl_estimate": round(ep.mpl_estimate, 3),
        }

    # Lowest conf_p10
    by_conf = sorted(episodes, key=lambda e: e.conf_p10)[:top_n]

    # Highest disagreement
    by_disag = sorted(episodes, key=lambda e: -e.disagreement_mean)[:top_n]

    # Most negative delta (regression)
    by_delta = sorted(episodes, key=lambda e: e.phi_star_delta)[:top_n]

    # Stagnant: high conf + near-zero delta
    stagnant = [
        ep for ep in episodes
        if ep.conf_p10 >= 0.4 and abs(ep.phi_star_delta) <= 0.05
    ]
    stagnant = sorted(stagnant, key=lambda e: -e.conf_p10)[:top_n]

    return {
        "lowest_confidence": [ep_summary(ep) for ep in by_conf],
        "highest_disagreement": [ep_summary(ep) for ep in by_disag],
        "most_negative_delta": [ep_summary(ep) for ep in by_delta],
        "stagnant_high_conf": [ep_summary(ep) for ep in stagnant],
    }


def generate_report(episodes: List[EpisodeRecord]) -> Dict[str, Any]:
    """Generate full ablation report."""
    has_pr_data = has_process_reward_data(episodes)

    report = {
        "summary": {
            "total_episodes": len(episodes),
            "has_process_reward_data": has_pr_data,
            "success_rate": float(np.mean([ep.success for ep in episodes])) if episodes else 0.0,
            "mean_mpl": float(np.mean([ep.mpl_estimate for ep in episodes])) if episodes else 0.0,
        },
        "analyses": {},
        "worst_episodes": {},
        "overall_verdict": "UNKNOWN",
        "recommendations": [],
    }

    if not has_pr_data:
        report["warning"] = "Episodes lack process reward data. Run process_reward_episode() first."
        return report

    # Run all analyses
    q1 = analyze_confidence_vs_success(episodes)
    q2 = analyze_disagreement_entropy_vs_success(episodes)
    q3 = analyze_gating_mpl_uplift(episodes)
    q4 = analyze_sampling_strategy_comparison(episodes)

    report["analyses"] = {
        "q1_confidence_predicts_success": q1,
        "q2_disagreement_predicts_failure": q2,
        "q3_gating_improves_mpl": q3,
        "q4_quality_sampling_outperforms": q4,
    }

    # Extract worst episodes for inspection
    report["worst_episodes"] = extract_worst_episodes(episodes)

    # Overall verdict
    verdicts = [q1.get("verdict", "NEUTRAL"), q2.get("verdict", "NEUTRAL"),
                q3.get("verdict", "NEUTRAL"), q4.get("verdict", "NEUTRAL")]
    positive_count = verdicts.count("POSITIVE")
    negative_count = verdicts.count("NEGATIVE")

    if positive_count >= 3:
        report["overall_verdict"] = "STRONG_POSITIVE"
        report["recommendations"].append("Process reward module is earning its keep. Consider increasing confidence gating thresholds.")
    elif positive_count >= 2:
        report["overall_verdict"] = "MODERATE_POSITIVE"
        report["recommendations"].append("Process reward shows promise. Consider A/B testing gating strategies in production.")
    elif negative_count >= 2:
        report["overall_verdict"] = "NEGATIVE"
        report["recommendations"].append("Process reward may need calibration. Check hop labels and FusionNet training.")
    else:
        report["overall_verdict"] = "NEUTRAL"
        report["recommendations"].append("Inconclusive results. Gather more data or run controlled experiment.")

    # Specific recommendations
    if q1.get("verdict") == "POSITIVE":
        report["recommendations"].append("Confidence is predictive - use conf_p10 for episode filtering.")
    if q2.get("verdict") == "POSITIVE":
        report["recommendations"].append("Disagreement signals failure - consider logging high-disagreement episodes for review.")
    if q3.get("best_uplift_pct", 0) > 10:
        report["recommendations"].append(f"Gating strategy '{q3.get('best_strategy')}' shows {q3['best_uplift_pct']:.1f}% MPL uplift - deploy it.")
    if q4.get("verdict") == "POSITIVE":
        report["recommendations"].append("Use 'process_reward_quality' as default sampling strategy.")

    return report


def write_csv_summary(report: Dict[str, Any], path: Path) -> None:
    """Write CSV summary of key metrics."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value", "interpretation"])

        # Summary
        writer.writerow(["total_episodes", report["summary"]["total_episodes"], ""])
        writer.writerow(["success_rate", f"{report['summary']['success_rate']:.3f}", ""])
        writer.writerow(["mean_mpl", f"{report['summary']['mean_mpl']:.3f}", ""])
        writer.writerow(["overall_verdict", report["overall_verdict"], ""])

        # Q1 correlations
        if "q1_confidence_predicts_success" in report.get("analyses", {}):
            q1 = report["analyses"]["q1_confidence_predicts_success"]
            for key, data in q1.get("correlations", {}).items():
                writer.writerow([f"q1_{key}", f"{data['value']:.3f}", data["interpretation"]])

        # Q2 correlations
        if "q2_disagreement_predicts_failure" in report.get("analyses", {}):
            q2 = report["analyses"]["q2_disagreement_predicts_failure"]
            for key, data in q2.get("correlations", {}).items():
                writer.writerow([f"q2_{key}", f"{data['value']:.3f}", data["interpretation"]])

        # Q3 best strategy
        if "q3_gating_improves_mpl" in report.get("analyses", {}):
            q3 = report["analyses"]["q3_gating_improves_mpl"]
            writer.writerow(["q3_best_strategy", q3.get("best_strategy", "none"), ""])
            writer.writerow(["q3_best_uplift_pct", f"{q3.get('best_uplift_pct', 0):.1f}", ""])

        # Q4 quality vs uniform
        if "q4_quality_sampling_outperforms" in report.get("analyses", {}):
            q4 = report["analyses"]["q4_quality_sampling_outperforms"]
            qvu = q4.get("quality_vs_uniform", {})
            writer.writerow(["q4_quality_success_uplift_pct", f"{qvu.get('success_uplift_pct', 0):.1f}", ""])
            writer.writerow(["q4_quality_mpl_uplift_pct", f"{qvu.get('mpl_uplift_pct', 0):.1f}", ""])


def print_report(report: Dict[str, Any]) -> None:
    """Print human-readable report to console."""
    print("=" * 80)
    print("PROCESS REWARD ABLATION REPORT")
    print("=" * 80)
    print()

    summary = report["summary"]
    print(f"Total episodes: {summary['total_episodes']}")
    print(f"Has process reward data: {summary['has_process_reward_data']}")
    print(f"Overall success rate: {summary['success_rate']:.1%}")
    print(f"Mean MPL: {summary['mean_mpl']:.3f}")
    print()

    if "warning" in report:
        print(f"WARNING: {report['warning']}")
        return

    analyses = report.get("analyses", {})

    # Q1
    if "q1_confidence_predicts_success" in analyses:
        q1 = analyses["q1_confidence_predicts_success"]
        print("-" * 40)
        print(f"Q1: {q1['question']}")
        print(f"    Verdict: {q1['verdict']}")
        for key, data in q1.get("correlations", {}).items():
            print(f"    {key}: {data['value']:.3f} ({data['interpretation']})")
        seg = q1.get("segmentation", {})
        print(f"    High conf success rate: {seg.get('high_conf_success_rate', 0):.1%} (n={seg.get('high_conf_count', 0)})")
        print(f"    Low conf success rate: {seg.get('low_conf_success_rate', 0):.1%} (n={seg.get('low_conf_count', 0)})")
        print()

    # Q2
    if "q2_disagreement_predicts_failure" in analyses:
        q2 = analyses["q2_disagreement_predicts_failure"]
        print("-" * 40)
        print(f"Q2: {q2['question']}")
        print(f"    Verdict: {q2['verdict']}")
        for key, data in q2.get("correlations", {}).items():
            print(f"    {key}: {data['value']:.3f} ({data['interpretation']})")
        print()

    # Q3
    if "q3_gating_improves_mpl" in analyses:
        q3 = analyses["q3_gating_improves_mpl"]
        print("-" * 40)
        print(f"Q3: {q3['question']}")
        print(f"    Verdict: {q3['verdict']}")
        print(f"    Baseline MPL: {q3.get('baseline_mpl', 0):.3f}")
        print(f"    Best strategy: {q3.get('best_strategy', 'none')}")
        print(f"    Best uplift: {q3.get('best_uplift_pct', 0):.1f}%")
        print()

    # Q4
    if "q4_quality_sampling_outperforms" in analyses:
        q4 = analyses["q4_quality_sampling_outperforms"]
        print("-" * 40)
        print(f"Q4: {q4['question']}")
        print(f"    Verdict: {q4['verdict']}")
        print(f"    Best for success: {q4.get('best_for_success', 'unknown')}")
        print(f"    Best for MPL: {q4.get('best_for_mpl', 'unknown')}")
        qvu = q4.get("quality_vs_uniform", {})
        print(f"    Quality vs uniform (success): {qvu.get('success_uplift_pct', 0):+.1f}%")
        print(f"    Quality vs uniform (MPL): {qvu.get('mpl_uplift_pct', 0):+.1f}%")
        print()

    # Overall
    print("=" * 80)
    print(f"OVERALL VERDICT: {report['overall_verdict']}")
    print()
    print("RECOMMENDATIONS:")
    for rec in report.get("recommendations", []):
        print(f"  • {rec}")
    print("=" * 80)


def run_report(
    episodes_path: str,
    output_dir: str,
) -> Dict[str, Any]:
    """Run the ablation report."""
    episodes = load_episodes(Path(episodes_path))
    report = generate_report(episodes)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write JSON
    json_path = out_dir / "ablation.json"
    with json_path.open("w") as f:
        json.dump(report, f, indent=2, sort_keys=True)

    # Write CSV
    write_csv_summary(report, out_dir / "summary.csv")

    # Print to console
    print_report(report)

    print(f"\nReport saved to: {out_dir}")
    return report


def parse_args():
    parser = argparse.ArgumentParser(description="Process Reward Ablation Report")
    parser.add_argument(
        "--episodes-path",
        default=None,
        help="Path to episodes JSONL with process reward fields",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Run directory containing episodes.jsonl (defaults output to <run_dir>/artifacts/process_reward)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for report (defaults to results/process_reward_ablation or run-dir artifacts)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.episodes_path and not args.run_dir:
        raise SystemExit("Must provide --episodes-path or --run-dir")

    output_dir = args.output_dir
    episodes_path = args.episodes_path
    if args.run_dir:
        run_dir = Path(args.run_dir)
        if episodes_path is None:
            episodes_path = str(run_dir / "episodes.jsonl")
        if output_dir is None:
            output_dir = str(run_dir / "artifacts" / "process_reward")
    if output_dir is None:
        output_dir = "results/process_reward_ablation"
    run_report(
        episodes_path=episodes_path,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
