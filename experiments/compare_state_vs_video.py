"""
State vs Video Mode Comparison Experiment

Runs both state and video modes for equal episodes and compares:
- Marginal product (MPL_r)
- Error rate
- Wage parity
- Consumer surplus
- Spread allocation

Generates comparison table and markdown summary.
"""
import os
import sys
import subprocess
import pandas as pd
import numpy as np

def run_training(config_path, episodes, log_suffix):
    """Run training and return log path."""
    print(f"\n{'='*60}")
    print(f"Running: {config_path}")
    print(f"Episodes: {episodes}")
    print(f"{'='*60}\n")

    # Run training
    cmd = [
        sys.executable,
        "train_sac_v2.py",
        config_path,
        "--episodes", str(episodes)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"❌ Training failed:")
        print(result.stderr)
        return None

    # Determine log path from config
    if "video" in config_path:
        log_path = "logs/sac_video_train.csv"
    else:
        log_path = "logs/sac_train.csv"

    print(f"✅ Training complete: {log_path}\n")
    return log_path


def analyze_logs(log_path, mode_name, last_n=20):
    """Analyze training logs and compute summary statistics."""
    df = pd.read_csv(log_path)

    # Take last N episodes
    df_tail = df.tail(last_n)

    summary = {
        'mode': mode_name,
        'episodes': len(df),
        'mpl_r_mean': df_tail['mp_r'].mean(),
        'mpl_r_std': df_tail['mp_r'].std(),
        'error_rate_mean': df_tail['err_rate'].mean() * 100,  # As percentage
        'error_rate_std': df_tail['err_rate'].std() * 100,
        'wage_parity_mean': df_tail['wage_parity'].mean(),
        'wage_parity_std': df_tail['wage_parity'].std(),
        'consumer_surplus_mean': df_tail['consumer_surplus'].mean(),
        'consumer_surplus_std': df_tail['consumer_surplus'].std(),
        'spread_mean': df_tail['spread_value'].mean(),
        'spread_std': df_tail['spread_value'].std(),
    }

    return summary


def generate_comparison_table(state_summary, video_summary):
    """Generate comparison table."""
    print("\n" + "="*80)
    print("STATE VS VIDEO COMPARISON")
    print("="*80)
    print(f"Analysis window: Last 20 episodes")
    print("-"*80)
    print(f"{'Metric':<30} {'State Mode':<25} {'Video Mode':<25}")
    print("-"*80)

    print(f"{'MPL (dishes/hr)':<30} {state_summary['mpl_r_mean']:>8.1f} ± {state_summary['mpl_r_std']:>5.1f}      "
          f"{video_summary['mpl_r_mean']:>8.1f} ± {video_summary['mpl_r_std']:>5.1f}")

    print(f"{'Error Rate (%)':<30} {state_summary['error_rate_mean']:>8.2f} ± {state_summary['error_rate_std']:>5.2f}      "
          f"{video_summary['error_rate_mean']:>8.2f} ± {video_summary['error_rate_std']:>5.2f}")

    print(f"{'Wage Parity (ŵᵣ/wₕ)':<30} {state_summary['wage_parity_mean']:>8.3f} ± {state_summary['wage_parity_std']:>5.3f}      "
          f"{video_summary['wage_parity_mean']:>8.3f} ± {video_summary['wage_parity_std']:>5.3f}")

    print(f"{'Consumer Surplus ($/hr)':<30} {state_summary['consumer_surplus_mean']:>8.2f} ± {state_summary['consumer_surplus_std']:>5.2f}      "
          f"{video_summary['consumer_surplus_mean']:>8.2f} ± {video_summary['consumer_surplus_std']:>5.2f}")

    print(f"{'Spread Value ($/hr)':<30} {state_summary['spread_mean']:>8.2f} ± {state_summary['spread_std']:>5.2f}      "
          f"{video_summary['spread_mean']:>8.2f} ± {video_summary['spread_std']:>5.2f}")

    print("="*80)


def generate_markdown_summary(state_summary, video_summary, output_path):
    """Generate markdown summary file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        f.write("# State vs Video Mode Comparison\n\n")
        f.write(f"**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Episodes**: {state_summary['episodes']} (each mode)\n\n")
        f.write(f"**Analysis Window**: Last 20 episodes\n\n")

        f.write("## Summary\n\n")
        f.write("| Metric | State Mode | Video Mode |\n")
        f.write("|--------|------------|------------|\n")
        f.write(f"| MPL (dishes/hr) | {state_summary['mpl_r_mean']:.1f} ± {state_summary['mpl_r_std']:.1f} | "
                f"{video_summary['mpl_r_mean']:.1f} ± {video_summary['mpl_r_std']:.1f} |\n")
        f.write(f"| Error Rate (%) | {state_summary['error_rate_mean']:.2f} ± {state_summary['error_rate_std']:.2f} | "
                f"{video_summary['error_rate_mean']:.2f} ± {video_summary['error_rate_std']:.2f} |\n")
        f.write(f"| Wage Parity (ŵᵣ/wₕ) | {state_summary['wage_parity_mean']:.3f} ± {state_summary['wage_parity_std']:.3f} | "
                f"{video_summary['wage_parity_mean']:.3f} ± {video_summary['wage_parity_std']:.3f} |\n")
        f.write(f"| Consumer Surplus ($/hr) | {state_summary['consumer_surplus_mean']:.2f} ± {state_summary['consumer_surplus_std']:.2f} | "
                f"{video_summary['consumer_surplus_mean']:.2f} ± {video_summary['consumer_surplus_std']:.2f} |\n")
        f.write(f"| Spread Value ($/hr) | {state_summary['spread_mean']:.2f} ± {state_summary['spread_std']:.2f} | "
                f"{video_summary['spread_mean']:.2f} ± {video_summary['spread_std']:.2f} |\n")

        f.write("\n## Interpretation\n\n")

        # Compute relative differences
        mpl_diff = ((video_summary['mpl_r_mean'] - state_summary['mpl_r_mean']) / state_summary['mpl_r_mean']) * 100
        err_diff = ((video_summary['error_rate_mean'] - state_summary['error_rate_mean']) / state_summary['error_rate_mean']) * 100
        wage_diff = ((video_summary['wage_parity_mean'] - state_summary['wage_parity_mean']) / state_summary['wage_parity_mean']) * 100

        f.write(f"- **MPL Difference**: {mpl_diff:+.1f}% (video vs state)\n")
        f.write(f"- **Error Rate Difference**: {err_diff:+.1f}% (video vs state)\n")
        f.write(f"- **Wage Parity Difference**: {wage_diff:+.1f}% (video vs state)\n\n")

        f.write("### Sanity Checks\n\n")

        # Check if both converged
        state_converged = state_summary['mpl_r_mean'] > 70  # Reasonable performance
        video_converged = video_summary['mpl_r_mean'] > 70

        f.write(f"- State mode convergence: {'✅ Yes' if state_converged else '❌ No'} "
                f"(MPL = {state_summary['mpl_r_mean']:.1f})\n")
        f.write(f"- Video mode convergence: {'✅ Yes' if video_converged else '❌ No'} "
                f"(MPL = {video_summary['mpl_r_mean']:.1f})\n")

        # Check if economics are sane
        state_cs_ok = state_summary['consumer_surplus_mean'] >= 0
        video_cs_ok = video_summary['consumer_surplus_mean'] >= 0

        f.write(f"- State mode consumer surplus non-negative: {'✅ Yes' if state_cs_ok else '❌ No'}\n")
        f.write(f"- Video mode consumer surplus non-negative: {'✅ Yes' if video_cs_ok else '❌ No'}\n")

        # Check if performance is similar
        similar = abs(mpl_diff) < 20  # Within 20%

        f.write(f"- Performance similarity (within 20%): {'✅ Yes' if similar else '❌ No'}\n\n")

        f.write("### Conclusion\n\n")
        if state_converged and video_converged and similar:
            f.write("✅ **Both modes converge to similar performance.** ")
            f.write("Economics layer successfully operates on both state and video observations.\n")
        elif state_converged and video_converged:
            f.write("⚠️ **Both modes converge but with different performance.** ")
            f.write("Video may be noisier or require longer training.\n")
        else:
            f.write("❌ **One or both modes failed to converge.** ")
            f.write("Further debugging needed.\n")

    print(f"\n✅ Markdown summary saved: {output_path}")


def main():
    """Run comparison experiment."""
    import argparse

    parser = argparse.ArgumentParser(description='Compare state vs video modes')
    parser.add_argument('--episodes', type=int, default=100,
                       help='Number of episodes per mode')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip training and analyze existing logs')
    args = parser.parse_args()

    if not args.skip_training:
        # Run state mode
        state_log = run_training(
            "configs/dishwashing_feasible.yaml",
            args.episodes,
            "state"
        )

        if state_log is None:
            print("❌ State mode training failed. Exiting.")
            return 1

        # Run video mode
        video_log = run_training(
            "configs/dishwashing_video.yaml",
            args.episodes,
            "video"
        )

        if video_log is None:
            print("❌ Video mode training failed. Exiting.")
            return 1
    else:
        print("Skipping training, analyzing existing logs...")
        state_log = "logs/sac_train.csv"
        video_log = "logs/sac_video_train.csv"

    # Analyze logs
    state_summary = analyze_logs(state_log, "State", last_n=20)
    video_summary = analyze_logs(video_log, "Video", last_n=20)

    # Generate comparison
    generate_comparison_table(state_summary, video_summary)

    # Generate markdown summary
    generate_markdown_summary(
        state_summary,
        video_summary,
        "artifacts/state_vs_video_summary.md"
    )

    print("\n✅ Comparison complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
