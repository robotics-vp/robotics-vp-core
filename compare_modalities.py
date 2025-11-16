"""
Three-Way Modality Comparison: State vs Synthetic Video vs Physics

Validates the economics theorem across all observation modalities:
- State mode: Direct task state (t, completed, attempts, errors)
- Synthetic video mode: Rendered synthetic video observations
- Physics mode: PyBullet physics simulation with camera rendering

Research Question: Does the wage parity and economics layer remain consistent
across observation modalities?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Set plotting style
plt.style.use('default')
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

def load_training_data(log_path):
    """Load training CSV and return DataFrame."""
    df = pd.read_csv(log_path)
    print(f"Loaded {log_path}: {len(df)} episodes")
    return df

def compute_summary_stats(df, name):
    """Compute summary statistics for a training run."""
    # Last 20 episodes (steady state)
    df_last = df.tail(20)

    stats = {
        'name': name,
        'n_episodes': len(df),

        # Marginal Product (MP)
        'mp_mean': df_last['mp_r'].mean(),
        'mp_std': df_last['mp_r'].std(),
        'mp_final': df['mp_r'].iloc[-1],

        # Error rate
        'err_mean': df_last['err_rate'].mean(),
        'err_std': df_last['err_rate'].std(),
        'err_final': df['err_rate'].iloc[-1],

        # Wage parity
        'wage_parity_mean': df_last['wage_parity'].mean(),
        'wage_parity_std': df_last['wage_parity'].std(),
        'wage_parity_final': df['wage_parity'].iloc[-1],

        # Profit
        'profit_mean': df_last['profit'].mean(),
        'profit_std': df_last['profit'].std(),
        'profit_final': df['profit'].iloc[-1],

        # Consumer surplus (if available)
        'consumer_surplus_mean': df_last['consumer_surplus'].mean() if 'consumer_surplus' in df.columns else 0,
        'consumer_surplus_final': df['consumer_surplus'].iloc[-1] if 'consumer_surplus' in df.columns else 0,
    }

    return stats

def plot_comparison(df_state, df_video, df_physics, output_dir):
    """Create comprehensive comparison plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use first 100 episodes for fair comparison
    n_episodes = min(len(df_state), len(df_video), len(df_physics), 100)

    df_state = df_state.head(n_episodes)
    df_video = df_video.head(n_episodes)
    df_physics = df_physics.head(n_episodes)

    # Create figure with 3x2 subplots
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle('Three-Way Modality Comparison: State vs Synthetic Video vs Physics',
                 fontsize=16, fontweight='bold')

    # 1. Marginal Product (MP) over time
    ax = axes[0, 0]
    ax.plot(df_state['episode'], df_state['mp_r'], label='State', alpha=0.7, linewidth=1.5)
    ax.plot(df_video['episode'], df_video['mp_r'], label='Synthetic Video', alpha=0.7, linewidth=1.5)
    ax.plot(df_physics['episode'], df_physics['mp_r'], label='Physics', alpha=0.7, linewidth=1.5)
    ax.axhline(y=60, color='red', linestyle='--', label='Human MP', alpha=0.5)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Marginal Product (units/hr)')
    ax.set_title('A. Marginal Product Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Error rate over time
    ax = axes[0, 1]
    ax.plot(df_state['episode'], df_state['err_rate'], label='State', alpha=0.7, linewidth=1.5)
    ax.plot(df_video['episode'], df_video['err_rate'], label='Synthetic Video', alpha=0.7, linewidth=1.5)
    ax.plot(df_physics['episode'], df_physics['err_rate'], label='Physics', alpha=0.7, linewidth=1.5)
    ax.axhline(y=df_state['err_target'].iloc[-1], color='orange', linestyle='--', label='Target', alpha=0.5)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Error Rate')
    ax.set_title('B. Error Rate Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Wage parity over time
    ax = axes[1, 0]
    ax.plot(df_state['episode'], df_state['wage_parity'], label='State', alpha=0.7, linewidth=1.5)
    ax.plot(df_video['episode'], df_video['wage_parity'], label='Synthetic Video', alpha=0.7, linewidth=1.5)
    ax.plot(df_physics['episode'], df_physics['wage_parity'], label='Physics', alpha=0.7, linewidth=1.5)
    ax.axhline(y=1.0, color='green', linestyle='--', label='Parity (w_r = w_h)', alpha=0.5)
    ax.axhspan(0.9, 1.1, alpha=0.1, color='green', label='±10% tolerance')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Wage Parity (w_r / w_h)')
    ax.set_title('C. Wage Parity Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Profit over time
    ax = axes[1, 1]
    ax.plot(df_state['episode'], df_state['profit'], label='State', alpha=0.7, linewidth=1.5)
    ax.plot(df_video['episode'], df_video['profit'], label='Synthetic Video', alpha=0.7, linewidth=1.5)
    ax.plot(df_physics['episode'], df_physics['profit'], label='Physics', alpha=0.7, linewidth=1.5)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Profit ($/hr)')
    ax.set_title('D. Profit Over Training')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. Consumer surplus over time (if available)
    ax = axes[2, 0]
    if 'consumer_surplus' in df_state.columns:
        ax.plot(df_state['episode'], df_state['consumer_surplus'], label='State', alpha=0.7, linewidth=1.5)
    if 'consumer_surplus' in df_video.columns:
        ax.plot(df_video['episode'], df_video['consumer_surplus'], label='Synthetic Video', alpha=0.7, linewidth=1.5)
    if 'consumer_surplus' in df_physics.columns:
        ax.plot(df_physics['episode'], df_physics['consumer_surplus'], label='Physics', alpha=0.7, linewidth=1.5)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Consumer Surplus ($/hr)')
    ax.set_title('E. Consumer Surplus Over Training')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. Final comparison bar chart (last 20 episodes average)
    ax = axes[2, 1]

    metrics = ['MP (units/hr)', 'Error Rate', 'Wage Parity', 'Profit ($/hr)']
    state_vals = [
        df_state.tail(20)['mp_r'].mean(),
        df_state.tail(20)['err_rate'].mean(),
        df_state.tail(20)['wage_parity'].mean(),
        df_state.tail(20)['profit'].mean()
    ]
    video_vals = [
        df_video.tail(20)['mp_r'].mean(),
        df_video.tail(20)['err_rate'].mean(),
        df_video.tail(20)['wage_parity'].mean(),
        df_video.tail(20)['profit'].mean()
    ]
    physics_vals = [
        df_physics.tail(20)['mp_r'].mean(),
        df_physics.tail(20)['err_rate'].mean(),
        df_physics.tail(20)['wage_parity'].mean(),
        df_physics.tail(20)['profit'].mean()
    ]

    x = np.arange(len(metrics))
    width = 0.25

    # Normalize for visualization (except profit which is already in $/hr)
    state_norm = [state_vals[0]/60, state_vals[1]*10, state_vals[2], state_vals[3]]
    video_norm = [video_vals[0]/60, video_vals[1]*10, video_vals[2], video_vals[3]]
    physics_norm = [physics_vals[0]/60, physics_vals[1]*10, physics_vals[2], physics_vals[3]]

    rects1 = ax.bar(x - width, state_norm, width, label='State', alpha=0.8)
    rects2 = ax.bar(x, video_norm, width, label='Synthetic Video', alpha=0.8)
    rects3 = ax.bar(x + width, physics_norm, width, label='Physics', alpha=0.8)

    ax.set_ylabel('Normalized Value')
    ax.set_title('F. Final Performance Comparison (last 20 episodes)')
    ax.set_xticks(x)
    ax.set_xticklabels(['MP\n(norm)', 'Err\n(×10)', 'Wage\nParity', 'Profit\n($/hr)'], fontsize=9)
    ax.legend()
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    def autolabel(rects, values, ax):
        for rect, val in zip(rects, values):
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., height,
                   f'{val:.2f}',
                   ha='center', va='bottom', fontsize=7)

    autolabel(rects1, state_vals, ax)
    autolabel(rects2, video_vals, ax)
    autolabel(rects3, physics_vals, ax)

    plt.tight_layout()

    # Save plot
    output_path = output_dir / 'modality_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot: {output_path}")

    plt.close()

def generate_report(stats_state, stats_video, stats_physics, output_dir):
    """Generate text report with comparison statistics."""
    output_dir = Path(output_dir)
    report_path = output_dir / 'modality_comparison_report.txt'

    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("THREE-WAY MODALITY COMPARISON REPORT\n")
        f.write("State vs Synthetic Video vs Physics\n")
        f.write("=" * 80 + "\n\n")

        f.write("RESEARCH QUESTION:\n")
        f.write("Does the economics theorem (wage parity, profit maximization) hold\n")
        f.write("consistently across different observation modalities?\n\n")

        f.write("=" * 80 + "\n")
        f.write("SUMMARY STATISTICS (Last 20 Episodes)\n")
        f.write("=" * 80 + "\n\n")

        # Table header
        f.write(f"{'Metric':<30} {'State':<15} {'Video':<15} {'Physics':<15}\n")
        f.write("-" * 80 + "\n")

        # Marginal Product
        f.write(f"{'Marginal Product (units/hr)':<30} "
               f"{stats_state['mp_mean']:>6.1f} ± {stats_state['mp_std']:<5.1f} "
               f"{stats_video['mp_mean']:>6.1f} ± {stats_video['mp_std']:<5.1f} "
               f"{stats_physics['mp_mean']:>6.1f} ± {stats_physics['mp_std']:<5.1f}\n")

        # Error Rate
        f.write(f"{'Error Rate':<30} "
               f"{stats_state['err_mean']:>6.3f} ± {stats_state['err_std']:<5.3f} "
               f"{stats_video['err_mean']:>6.3f} ± {stats_video['err_std']:<5.3f} "
               f"{stats_physics['err_mean']:>6.3f} ± {stats_physics['err_std']:<5.3f}\n")

        # Wage Parity
        f.write(f"{'Wage Parity (w_r / w_h)':<30} "
               f"{stats_state['wage_parity_mean']:>6.3f} ± {stats_state['wage_parity_std']:<5.3f} "
               f"{stats_video['wage_parity_mean']:>6.3f} ± {stats_video['wage_parity_std']:<5.3f} "
               f"{stats_physics['wage_parity_mean']:>6.3f} ± {stats_physics['wage_parity_std']:<5.3f}\n")

        # Profit
        f.write(f"{'Profit ($/hr)':<30} "
               f"{stats_state['profit_mean']:>6.2f} ± {stats_state['profit_std']:<5.2f} "
               f"{stats_video['profit_mean']:>6.2f} ± {stats_video['profit_std']:<5.2f} "
               f"{stats_physics['profit_mean']:>6.2f} ± {stats_physics['profit_std']:<5.2f}\n")

        # Consumer Surplus
        f.write(f"{'Consumer Surplus ($/hr)':<30} "
               f"{stats_state['consumer_surplus_mean']:>6.2f}         "
               f"{stats_video['consumer_surplus_mean']:>6.2f}         "
               f"{stats_physics['consumer_surplus_mean']:>6.2f}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("KEY FINDINGS\n")
        f.write("=" * 80 + "\n\n")

        # Check convergence criteria
        human_mp = 60
        human_wage = 18.0

        f.write("1. WAGE PARITY ANALYSIS:\n")
        f.write(f"   Target: w_r ≈ w_h (parity ≈ 1.0, within ±10% tolerance)\n\n")

        for name, stats in [('State', stats_state), ('Video', stats_video), ('Physics', stats_physics)]:
            parity = stats['wage_parity_mean']
            within_tolerance = 0.9 <= parity <= 1.1
            status = "✓ CONVERGED" if within_tolerance else "✗ NOT CONVERGED"
            f.write(f"   {name:<15}: {parity:.3f}  {status}\n")

        f.write("\n2. PRODUCTIVITY ANALYSIS:\n")
        f.write(f"   Human benchmark: {human_mp} units/hr\n\n")

        for name, stats in [('State', stats_state), ('Video', stats_video), ('Physics', stats_physics)]:
            mp = stats['mp_mean']
            prod_parity = mp / human_mp
            status = "✓ EXCEEDS HUMAN" if prod_parity >= 1.0 else f"{prod_parity*100:.0f}% of human"
            f.write(f"   {name:<15}: {mp:>5.1f} units/hr  ({status})\n")

        f.write("\n3. ERROR RATE ANALYSIS:\n\n")

        for name, stats in [('State', stats_state), ('Video', stats_video), ('Physics', stats_physics)]:
            err = stats['err_mean']
            err_pct = err * 100
            f.write(f"   {name:<15}: {err:.3f} ({err_pct:.1f}% error rate)\n")

        f.write("\n4. ECONOMICS THEOREM VALIDATION:\n\n")

        f.write("   Core hypothesis: Economics layer (wage convergence, profit, consumer surplus)\n")
        f.write("   should be consistent across observation modalities.\n\n")

        # Compute coefficient of variation for key metrics
        wage_parities = [stats_state['wage_parity_mean'],
                         stats_video['wage_parity_mean'],
                         stats_physics['wage_parity_mean']]
        wage_parity_cv = np.std(wage_parities) / np.mean(wage_parities) * 100

        profits = [stats_state['profit_mean'],
                   stats_video['profit_mean'],
                   stats_physics['profit_mean']]
        profit_cv = np.std(profits) / np.mean(profits) * 100

        f.write(f"   Wage Parity CV: {wage_parity_cv:.1f}%  ")
        if wage_parity_cv < 10:
            f.write("(✓ CONSISTENT across modalities)\n")
        else:
            f.write("(✗ INCONSISTENT across modalities)\n")

        f.write(f"   Profit CV:      {profit_cv:.1f}%  ")
        if profit_cv < 20:
            f.write("(✓ CONSISTENT across modalities)\n")
        else:
            f.write("(✗ INCONSISTENT across modalities)\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("CONCLUSION\n")
        f.write("=" * 80 + "\n\n")

        # Overall assessment
        all_converged = all(0.9 <= stats['wage_parity_mean'] <= 1.1
                           for stats in [stats_state, stats_video, stats_physics])

        if all_converged and wage_parity_cv < 10:
            f.write("✅ THEOREM VALIDATED: Economics layer is consistent across all modalities.\n")
            f.write("   All three observation types (state, synthetic video, physics) converge to\n")
            f.write("   wage parity with low inter-modality variance.\n")
        elif all_converged:
            f.write("⚠️  PARTIAL VALIDATION: All modalities converge to wage parity, but with\n")
            f.write("   moderate variance. Economics layer is mostly consistent.\n")
        else:
            f.write("❌ THEOREM NOT VALIDATED: Not all modalities achieve wage parity convergence.\n")
            f.write("   Economics layer shows inconsistency across observation types.\n")

        f.write("\n" + "=" * 80 + "\n")

    print(f"Saved comparison report: {report_path}")

    # Print summary to console
    with open(report_path, 'r') as f:
        print("\n" + f.read())

def main():
    # Paths
    log_dir = Path('logs')
    output_dir = Path('reports/modality_comparison')

    # Load training data
    print("Loading training data...")
    df_state = load_training_data(log_dir / 'sac_train.csv')
    df_video = load_training_data(log_dir / 'sac_video_train.csv')
    df_physics = load_training_data(log_dir / 'sac_physics_train.csv')

    # Compute summary statistics
    print("\nComputing summary statistics...")
    stats_state = compute_summary_stats(df_state, 'State')
    stats_video = compute_summary_stats(df_video, 'Synthetic Video')
    stats_physics = compute_summary_stats(df_physics, 'Physics')

    # Generate plots
    print("\nGenerating comparison plots...")
    plot_comparison(df_state, df_video, df_physics, output_dir)

    # Generate report
    print("\nGenerating comparison report...")
    generate_report(stats_state, stats_video, stats_physics, output_dir)

    print("\n✅ Modality comparison complete!")
    print(f"📊 Plots: {output_dir}/modality_comparison.png")
    print(f"📄 Report: {output_dir}/modality_comparison_report.txt")

if __name__ == '__main__':
    main()
