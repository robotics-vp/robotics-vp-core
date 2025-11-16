"""
Consumer surplus validation and visualization.

Verifies:
1. Consumer surplus guarantee holds (customer_cost <= w_h_indexed)
2. Consumer surplus distribution
3. Time evolution of pricing vs wage benchmark
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def validate_consumer_surplus_guarantee(df, tolerance=1e-6):
    """
    Verify that customer_cost <= w_h_indexed for all episodes.

    Args:
        df: Training log dataframe
        tolerance: Numerical tolerance

    Returns:
        dict with validation results
    """
    violations = df[df['customer_cost'] > df['w_h_indexed'] + tolerance]
    n_violations = len(violations)
    n_total = len(df)

    return {
        'n_violations': n_violations,
        'n_total': n_total,
        'pct_valid': 100.0 * (1 - n_violations / n_total) if n_total > 0 else 100.0,
        'violations': violations if n_violations > 0 else None
    }


def plot_consumer_surplus_time(df, output_path='plots/consumer_surplus_time.png'):
    """
    Plot consumer surplus evolution over time.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Consumer Surplus Analysis', fontsize=16)

    # 1. Wage comparison
    ax = axes[0, 0]
    ax.plot(df['episode'], df['w_h_indexed'], label='Human Wage (indexed)', alpha=0.7, linewidth=2)
    ax.plot(df['episode'], df['w_hat_r'], label='Robot Wage', alpha=0.7)
    ax.plot(df['episode'], df['customer_cost'], label='Customer Cost', alpha=0.7, linestyle='--')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Wage/Cost ($/hr)')
    ax.set_title('Wage Indexing & Customer Cost')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Consumer surplus over time
    ax = axes[0, 1]
    ax.plot(df['episode'], df['consumer_surplus'], alpha=0.7, color='green')
    ax.fill_between(df['episode'], 0, df['consumer_surplus'], alpha=0.3, color='green')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Consumer Surplus ($/hr)')
    ax.set_title('Consumer Surplus (Customer Savings)')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='red', linestyle='--', alpha=0.5)

    # 3. Guarantee verification (customer_cost vs w_h)
    ax = axes[1, 0]
    ax.scatter(df['w_h_indexed'], df['customer_cost'], alpha=0.3, s=10)
    # Add diagonal line (where customer_cost = w_h_indexed)
    min_val = min(df['w_h_indexed'].min(), df['customer_cost'].min())
    max_val = max(df['w_h_indexed'].max(), df['customer_cost'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='customer_cost = w_h (boundary)', linewidth=2)
    ax.set_xlabel('Human Wage (indexed) ($/hr)')
    ax.set_ylabel('Customer Cost ($/hr)')
    ax.set_title('Consumer Surplus Guarantee Verification')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Spread decomposition
    ax = axes[1, 1]
    # Spread = w_hat_r - w_h_indexed
    spread = df['w_hat_r'] - df['w_h_indexed']
    ax.plot(df['episode'], spread, label='Spread (robot - human)', alpha=0.7)
    ax.plot(df['episode'], df['rebate'] / df['time_h'], label='Rebate ($/hr)', alpha=0.7)
    ax.plot(df['episode'], df['consumer_surplus'], label='Consumer Surplus ($/hr)', alpha=0.7)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Amount ($/hr)')
    ax.set_title('Spread Decomposition')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='black', linestyle='-', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"✅ Saved: {output_path}")


def plot_consumer_surplus_histogram(df, output_path='plots/consumer_surplus_hist.png'):
    """
    Plot distribution of consumer surplus.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 1. Consumer surplus histogram
    ax = axes[0]
    ax.hist(df['consumer_surplus'], bins=30, alpha=0.7, color='green', edgecolor='black')
    ax.axvline(df['consumer_surplus'].mean(), color='red', linestyle='--',
               linewidth=2, label=f'Mean: ${df["consumer_surplus"].mean():.2f}/hr')
    ax.set_xlabel('Consumer Surplus ($/hr)')
    ax.set_ylabel('Frequency')
    ax.set_title('Consumer Surplus Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Customer cost histogram
    ax = axes[1]
    ax.hist(df['customer_cost'], bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax.axvline(df['customer_cost'].mean(), color='red', linestyle='--',
               linewidth=2, label=f'Mean: ${df["customer_cost"].mean():.2f}/hr')
    ax.axvline(df['w_h_indexed'].mean(), color='orange', linestyle='--',
               linewidth=2, label=f'Mean wₕ: ${df["w_h_indexed"].mean():.2f}/hr')
    ax.set_xlabel('Customer Cost ($/hr)')
    ax.set_ylabel('Frequency')
    ax.set_title('Customer Cost Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"✅ Saved: {output_path}")


def print_consumer_surplus_summary(df):
    """
    Print summary statistics for consumer surplus.
    """
    print("\n" + "=" * 70)
    print("CONSUMER SURPLUS ANALYSIS")
    print("=" * 70)

    # Validate guarantee
    validation = validate_consumer_surplus_guarantee(df)
    print(f"\n[CONSUMER SURPLUS GUARANTEE]")
    print(f"  Total episodes: {validation['n_total']}")
    print(f"  Valid episodes (cost <= wage): {validation['n_total'] - validation['n_violations']}")
    print(f"  Violations: {validation['n_violations']}")
    print(f"  % Valid: {validation['pct_valid']:.2f}%")

    if validation['n_violations'] > 0:
        print("\n  ⚠️  WARNING: Consumer surplus guarantee violated!")
        print(f"  First violation at episode: {validation['violations']['episode'].iloc[0]}")
    else:
        print("\n  ✅ Consumer surplus guarantee holds for ALL episodes")

    # Consumer surplus stats
    print(f"\n[CONSUMER SURPLUS STATISTICS]")
    print(f"  Mean: ${df['consumer_surplus'].mean():.2f}/hr")
    print(f"  Std: ${df['consumer_surplus'].std():.2f}/hr")
    print(f"  Min: ${df['consumer_surplus'].min():.2f}/hr")
    print(f"  Max: ${df['consumer_surplus'].max():.2f}/hr")
    print(f"  Median: ${df['consumer_surplus'].median():.2f}/hr")

    # Customer cost stats
    print(f"\n[CUSTOMER COST STATISTICS]")
    print(f"  Mean: ${df['customer_cost'].mean():.2f}/hr")
    print(f"  Std: ${df['customer_cost'].std():.2f}/hr")
    print(f"  Min: ${df['customer_cost'].min():.2f}/hr")
    print(f"  Max: ${df['customer_cost'].max():.2f}/hr")

    # Wage indexing
    print(f"\n[WAGE INDEXING]")
    print(f"  Initial wₕ: ${df['w_h_indexed'].iloc[0]:.2f}/hr")
    print(f"  Final wₕ: ${df['w_h_indexed'].iloc[-1]:.2f}/hr")
    print(f"  Change: ${df['w_h_indexed'].iloc[-1] - df['w_h_indexed'].iloc[0]:.2f}/hr ({100*(df['w_h_indexed'].iloc[-1]/df['w_h_indexed'].iloc[0] - 1):.2f}%)")
    print(f"  Mean wₕ: ${df['w_h_indexed'].mean():.2f}/hr")

    # Customer savings
    avg_savings_pct = 100 * df['consumer_surplus'].mean() / df['w_h_indexed'].mean()
    print(f"\n[CUSTOMER SAVINGS]")
    print(f"  Avg savings vs human: {avg_savings_pct:.2f}%")
    print(f"  Total savings (all episodes): ${df['consumer_surplus'].sum():.2f}")

    print("\n" + "=" * 70)


def main(log_path='logs/sac_train.csv'):
    """
    Main analysis function.
    """
    print(f"Loading logs: {log_path}")
    df = pd.read_csv(log_path)

    # Check for required columns
    required_cols = ['w_h_indexed', 'customer_cost', 'consumer_surplus']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"\n❌ Missing columns: {missing}")
        print("   Make sure train_sac.py has wage indexer and pricing integrated.")
        return

    # Print summary
    print_consumer_surplus_summary(df)

    # Generate plots
    print("\nGenerating plots...")
    Path('plots').mkdir(exist_ok=True)
    plot_consumer_surplus_time(df)
    plot_consumer_surplus_histogram(df)

    print("\n✅ Consumer surplus validation complete")


if __name__ == "__main__":
    import sys

    log_path = 'logs/sac_train.csv'
    if len(sys.argv) > 1:
        log_path = sys.argv[1]

    main(log_path)
