"""
Compare All Three Modes: State, Synthetic Video, Physics

Compares:
- State mode (baseline)
- Synthetic video (colored bars)
- Physics simulation (PyBullet rendered)

Metrics:
- MPL, error rate, wage parity
- Consumer surplus, spread allocation
- Training stability
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_and_analyze(log_path, mode_name, last_n=20):
    """Load logs and compute summary statistics."""
    if not os.path.exists(log_path):
        print(f"⚠️  Log file not found: {log_path}")
        return None

    df = pd.read_csv(log_path)

    if len(df) == 0:
        print(f"⚠️  Empty log file: {log_path}")
        return None

    # Take last N episodes
    df_tail = df.tail(last_n)

    summary = {
        'mode': mode_name,
        'episodes_total': len(df),
        'episodes_analyzed': len(df_tail),
        'mpl_r_mean': df_tail['mp_r'].mean(),
        'mpl_r_std': df_tail['mp_r'].std(),
        'mpl_r_min': df_tail['mp_r'].min(),
        'mpl_r_max': df_tail['mp_r'].max(),
        'error_rate_mean': df_tail['err_rate'].mean() * 100,
        'error_rate_std': df_tail['err_rate'].std() * 100,
        'wage_parity_mean': df_tail['wage_parity'].mean(),
        'wage_parity_std': df_tail['wage_parity'].std(),
        'consumer_surplus_mean': df_tail['consumer_surplus'].mean(),
        'consumer_surplus_std': df_tail['consumer_surplus'].std(),
        'spread_mean': df_tail['spread_value'].mean(),
        'spread_std': df_tail['spread_value'].std(),
        'episode_reward_mean': df_tail['episode_reward'].mean(),
        'episode_reward_std': df_tail['episode_reward'].std(),
    }

    return summary


def print_comparison_table(summaries):
    """Print comparison table."""
    print("\n" + "="*100)
    print("THREE-WAY COMPARISON: State vs Synthetic Video vs Physics")
    print("="*100)
    print(f"Analysis window: Last 20 episodes (or fewer if training incomplete)")
    print("-"*100)

    # Header
    print(f"{'Metric':<30} {'State':<25} {'Synthetic Video':<25} {'Physics':<25}")
    print("-"*100)

    # Episodes
    print(f"{'Episodes Completed':<30} "
          f"{summaries['state']['episodes_total']:>8}                "
          f"{summaries['synthetic']['episodes_total']:>8}                "
          f"{summaries['physics']['episodes_total']:>8}")

    # MPL
    print(f"{'MPL (dishes/hr)':<30} "
          f"{summaries['state']['mpl_r_mean']:>8.1f} ± {summaries['state']['mpl_r_std']:>5.1f}      "
          f"{summaries['synthetic']['mpl_r_mean']:>8.1f} ± {summaries['synthetic']['mpl_r_std']:>5.1f}      "
          f"{summaries['physics']['mpl_r_mean']:>8.1f} ± {summaries['physics']['mpl_r_std']:>5.1f}")

    # Error rate
    print(f"{'Error Rate (%)':<30} "
          f"{summaries['state']['error_rate_mean']:>8.2f} ± {summaries['state']['error_rate_std']:>5.2f}      "
          f"{summaries['synthetic']['error_rate_mean']:>8.2f} ± {summaries['synthetic']['error_rate_std']:>5.2f}      "
          f"{summaries['physics']['error_rate_mean']:>8.2f} ± {summaries['physics']['error_rate_std']:>5.2f}")

    # Wage parity
    print(f"{'Wage Parity (ŵᵣ/wₕ)':<30} "
          f"{summaries['state']['wage_parity_mean']:>8.3f} ± {summaries['state']['wage_parity_std']:>5.3f}      "
          f"{summaries['synthetic']['wage_parity_mean']:>8.3f} ± {summaries['synthetic']['wage_parity_std']:>5.3f}      "
          f"{summaries['physics']['wage_parity_mean']:>8.3f} ± {summaries['physics']['wage_parity_std']:>5.3f}")

    # Consumer surplus
    print(f"{'Consumer Surplus ($/hr)':<30} "
          f"{summaries['state']['consumer_surplus_mean']:>8.2f} ± {summaries['state']['consumer_surplus_std']:>5.2f}      "
          f"{summaries['synthetic']['consumer_surplus_mean']:>8.2f} ± {summaries['synthetic']['consumer_surplus_std']:>5.2f}      "
          f"{summaries['physics']['consumer_surplus_mean']:>8.2f} ± {summaries['physics']['consumer_surplus_std']:>5.2f}")

    # Spread
    print(f"{'Spread Value ($/hr)':<30} "
          f"{summaries['state']['spread_mean']:>8.2f} ± {summaries['state']['spread_std']:>5.2f}      "
          f"{summaries['synthetic']['spread_mean']:>8.2f} ± {summaries['synthetic']['spread_std']:>5.2f}      "
          f"{summaries['physics']['spread_mean']:>8.2f} ± {summaries['physics']['spread_std']:>5.2f}")

    print("="*100)


def create_comparison_plots(summaries, output_path):
    """Create comparison plots."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    modes = ['state', 'synthetic', 'physics']
    mode_labels = ['State', 'Synthetic Video', 'Physics']
    colors = ['#2ecc71', '#3498db', '#e74c3c']

    # Plot 1: MPL
    ax = axes[0, 0]
    mpl_means = [summaries[m]['mpl_r_mean'] for m in modes]
    mpl_stds = [summaries[m]['mpl_r_std'] for m in modes]
    ax.bar(mode_labels, mpl_means, yerr=mpl_stds, color=colors, alpha=0.7)
    ax.set_ylabel('MPL (dishes/hr)')
    ax.set_title('Marginal Product of Labor')
    ax.grid(axis='y', alpha=0.3)

    # Plot 2: Error Rate
    ax = axes[0, 1]
    err_means = [summaries[m]['error_rate_mean'] for m in modes]
    err_stds = [summaries[m]['error_rate_std'] for m in modes]
    ax.bar(mode_labels, err_means, yerr=err_stds, color=colors, alpha=0.7)
    ax.set_ylabel('Error Rate (%)')
    ax.set_title('Error Rate')
    ax.grid(axis='y', alpha=0.3)

    # Plot 3: Wage Parity
    ax = axes[0, 2]
    wage_means = [summaries[m]['wage_parity_mean'] for m in modes]
    wage_stds = [summaries[m]['wage_parity_std'] for m in modes]
    ax.bar(mode_labels, wage_means, yerr=wage_stds, color=colors, alpha=0.7)
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1, label='Parity')
    ax.set_ylabel('Wage Parity (ŵᵣ/wₕ)')
    ax.set_title('Wage Parity')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Plot 4: Consumer Surplus
    ax = axes[1, 0]
    cs_means = [summaries[m]['consumer_surplus_mean'] for m in modes]
    cs_stds = [summaries[m]['consumer_surplus_std'] for m in modes]
    ax.bar(mode_labels, cs_means, yerr=cs_stds, color=colors, alpha=0.7)
    ax.set_ylabel('Consumer Surplus ($/hr)')
    ax.set_title('Consumer Surplus')
    ax.grid(axis='y', alpha=0.3)

    # Plot 5: Spread Value
    ax = axes[1, 1]
    spread_means = [summaries[m]['spread_mean'] for m in modes]
    spread_stds = [summaries[m]['spread_std'] for m in modes]
    ax.bar(mode_labels, spread_means, yerr=spread_stds, color=colors, alpha=0.7)
    ax.set_ylabel('Spread Value ($/hr)')
    ax.set_title('Spread Allocation')
    ax.grid(axis='y', alpha=0.3)

    # Plot 6: Episode Reward
    ax = axes[1, 2]
    reward_means = [summaries[m]['episode_reward_mean'] for m in modes]
    reward_stds = [summaries[m]['episode_reward_std'] for m in modes]
    ax.bar(mode_labels, reward_means, yerr=reward_stds, color=colors, alpha=0.7)
    ax.set_ylabel('Episode Reward')
    ax.set_title('Episode Reward')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n✅ Comparison plots saved: {output_path}")


def generate_markdown_report(summaries, output_path):
    """Generate markdown report."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        f.write("# Three-Way Mode Comparison\n\n")
        f.write(f"**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Training Episodes\n\n")
        f.write("| Mode | Episodes |\n")
        f.write("|------|----------|\n")
        f.write(f"| State | {summaries['state']['episodes_total']} |\n")
        f.write(f"| Synthetic Video | {summaries['synthetic']['episodes_total']} |\n")
        f.write(f"| Physics | {summaries['physics']['episodes_total']} |\n\n")

        f.write("## Performance Metrics (Last 20 Episodes)\n\n")
        f.write("| Metric | State | Synthetic Video | Physics |\n")
        f.write("|--------|-------|-----------------|--------|\n")

        f.write(f"| MPL (dishes/hr) | "
                f"{summaries['state']['mpl_r_mean']:.1f} ± {summaries['state']['mpl_r_std']:.1f} | "
                f"{summaries['synthetic']['mpl_r_mean']:.1f} ± {summaries['synthetic']['mpl_r_std']:.1f} | "
                f"{summaries['physics']['mpl_r_mean']:.1f} ± {summaries['physics']['mpl_r_std']:.1f} |\n")

        f.write(f"| Error Rate (%) | "
                f"{summaries['state']['error_rate_mean']:.2f} ± {summaries['state']['error_rate_std']:.2f} | "
                f"{summaries['synthetic']['error_rate_mean']:.2f} ± {summaries['synthetic']['error_rate_std']:.2f} | "
                f"{summaries['physics']['error_rate_mean']:.2f} ± {summaries['physics']['error_rate_std']:.2f} |\n")

        f.write(f"| Wage Parity | "
                f"{summaries['state']['wage_parity_mean']:.3f} ± {summaries['state']['wage_parity_std']:.3f} | "
                f"{summaries['synthetic']['wage_parity_mean']:.3f} ± {summaries['synthetic']['wage_parity_std']:.3f} | "
                f"{summaries['physics']['wage_parity_mean']:.3f} ± {summaries['physics']['wage_parity_std']:.3f} |\n")

        f.write(f"| Consumer Surplus ($/hr) | "
                f"{summaries['state']['consumer_surplus_mean']:.2f} ± {summaries['state']['consumer_surplus_std']:.2f} | "
                f"{summaries['synthetic']['consumer_surplus_mean']:.2f} ± {summaries['synthetic']['consumer_surplus_std']:.2f} | "
                f"{summaries['physics']['consumer_surplus_mean']:.2f} ± {summaries['physics']['consumer_surplus_std']:.2f} |\n")

        f.write("\n## Analysis\n\n")

        # Compute relative differences vs state (baseline)
        state_mpl = summaries['state']['mpl_r_mean']
        synth_mpl_diff = ((summaries['synthetic']['mpl_r_mean'] - state_mpl) / state_mpl) * 100
        phys_mpl_diff = ((summaries['physics']['mpl_r_mean'] - state_mpl) / state_mpl) * 100

        f.write(f"- **Synthetic Video vs State**: MPL difference = {synth_mpl_diff:+.1f}%\n")
        f.write(f"- **Physics vs State**: MPL difference = {phys_mpl_diff:+.1f}%\n")
        f.write(f"- **Physics vs Synthetic**: MPL difference = "
                f"{((summaries['physics']['mpl_r_mean'] - summaries['synthetic']['mpl_r_mean']) / summaries['synthetic']['mpl_r_mean']) * 100:+.1f}%\n\n")

        f.write("### Observations\n\n")

        # Convergence check
        for mode in ['state', 'synthetic', 'physics']:
            converged = summaries[mode]['mpl_r_mean'] > 70
            f.write(f"- {mode.capitalize()} mode: ")
            f.write(f"{'✅ Converged' if converged else '❌ Not converged'} ")
            f.write(f"(MPL = {summaries[mode]['mpl_r_mean']:.1f})\n")

        f.write("\n### Conclusion\n\n")
        all_converged = all(summaries[m]['mpl_r_mean'] > 70 for m in modes)
        if all_converged:
            f.write("✅ **All three modes achieve reasonable performance.** ")
            f.write("Economics layer successfully operates across state, synthetic video, and physics-rendered observations.\n")
        else:
            f.write("⚠️ **Some modes have not converged yet.** May need longer training or parameter tuning.\n")

    print(f"✅ Markdown report saved: {output_path}")


def main():
    """Run three-way comparison."""
    print("\n" + "#"*100)
    print("# THREE-WAY MODE COMPARISON")
    print("#"*100)

    # Load all three modes
    print("\nLoading training logs...")

    state_summary = load_and_analyze("logs/sac_train.csv", "state")
    synthetic_summary = load_and_analyze("logs/sac_video_train.csv", "synthetic")
    physics_summary = load_and_analyze("logs/sac_physics_train.csv", "physics")

    if not all([state_summary, synthetic_summary, physics_summary]):
        print("\n❌ Some log files missing or empty. Cannot complete comparison.")
        print("Please ensure all three modes have completed training:")
        print("  - State: logs/sac_train.csv")
        print("  - Synthetic: logs/sac_video_train.csv")
        print("  - Physics: logs/sac_physics_train.csv")
        return 1

    summaries = {
        'state': state_summary,
        'synthetic': synthetic_summary,
        'physics': physics_summary
    }

    # Print comparison
    print_comparison_table(summaries)

    # Generate plots
    create_comparison_plots(summaries, "artifacts/mode_comparison_plots.png")

    # Generate markdown report
    generate_markdown_report(summaries, "artifacts/three_way_comparison.md")

    print("\n✅ Three-way comparison complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
