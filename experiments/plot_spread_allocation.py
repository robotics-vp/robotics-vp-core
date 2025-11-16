"""
Plot spread allocation over training.

Visualizes:
1. Spread value over time
2. Rebate vs captured spread
3. Customer and platform contribution shares (s_cust, s_plat)
4. Verification that rebate + captured ≈ spread_value
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_spread_allocation(log_path='logs/sac_train.csv',
                           output_path='plots/spread_allocation.png'):
    """
    Read SAC training logs and plot spread allocation metrics.
    """
    print(f"Loading training logs: {log_path}")
    df = pd.read_csv(log_path)

    print(f"  Episodes: {len(df)}")
    print(f"  Columns: {list(df.columns)}")

    # Check for spread allocation columns
    required_cols = ['spread', 'spread_value', 's_cust', 's_plat',
                     'rebate', 'captured_spread']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"\n❌ Missing columns: {missing}")
        print("   Make sure train_sac.py is logging spread allocation fields.")
        return

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Mechanistic Spread Allocation (Causal Mode)', fontsize=16)

    # 1. Spread value over time
    ax = axes[0, 0]
    ax.plot(df['episode'], df['spread_value'], alpha=0.7, linewidth=1)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Spread Value ($)')
    ax.set_title('Total Spread Value Over Time')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='red', linestyle='--', alpha=0.5)

    # 2. Rebate vs captured
    ax = axes[0, 1]
    ax.plot(df['episode'], df['rebate'], label='Rebate (Customer)', alpha=0.7)
    ax.plot(df['episode'], df['captured_spread'], label='Captured (Platform)', alpha=0.7)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Amount ($)')
    ax.set_title('Spread Allocation: Customer vs Platform')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Contribution shares
    ax = axes[0, 2]
    ax.plot(df['episode'], df['s_cust'], label='s_cust (Customer Share)', alpha=0.7)
    ax.plot(df['episode'], df['s_plat'], label='s_plat (Platform Share)', alpha=0.7)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Share')
    ax.set_title('Contribution Shares (ΔMPL-based)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.1)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.3)

    # 4. Verification: rebate + captured ≈ spread_value
    ax = axes[1, 0]
    sum_allocated = df['rebate'] + df['captured_spread']
    error = sum_allocated - df['spread_value']
    ax.plot(df['episode'], error, alpha=0.7, color='red')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Error ($)')
    ax.set_title('Allocation Error: (Rebate + Captured) - Spread')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='black', linestyle='-', linewidth=2, alpha=0.5)

    # 5. Wage parity context
    ax = axes[1, 1]
    ax.plot(df['episode'], df['wage_parity'], alpha=0.7, color='green')
    ax.axhline(1.0, color='red', linestyle='--', alpha=0.5, label='Parity = 1')
    ax.axhline(1.05, color='orange', linestyle='--', alpha=0.5, label='Parity = 1.05 (threshold)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Wage Parity')
    ax.set_title('Wage Parity (Spread triggers above 1.05)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. ΔMPL context
    ax = axes[1, 2]
    ax.plot(df['episode'], df['delta_mpl_total'], label='ΔMPL_total', alpha=0.7)
    ax.plot(df['episode'], df['delta_mpl_cust'], label='ΔMPL_cust', alpha=0.7)
    ax.set_xlabel('Episode')
    ax.set_ylabel('ΔMPL (units/hr)')
    ax.set_title('MPL Changes (Contribution Attribution)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\n✅ Saved plot: {output_path}")

    # Print summary statistics
    print("\n" + "=" * 60)
    print("SPREAD ALLOCATION SUMMARY")
    print("=" * 60)

    # Episodes with spread
    has_spread = df['spread_value'] > 0
    n_with_spread = has_spread.sum()
    n_total = len(df)

    print(f"\nEpisodes with spread: {n_with_spread}/{n_total} ({100*n_with_spread/n_total:.1f}%)")

    if n_with_spread > 0:
        spread_df = df[has_spread]

        print(f"\nSpread value:")
        print(f"  Mean: ${spread_df['spread_value'].mean():.4f}")
        print(f"  Std:  ${spread_df['spread_value'].std():.4f}")
        print(f"  Min:  ${spread_df['spread_value'].min():.4f}")
        print(f"  Max:  ${spread_df['spread_value'].max():.4f}")

        print(f"\nCustomer rebate:")
        print(f"  Mean: ${spread_df['rebate'].mean():.4f}")
        print(f"  Total: ${spread_df['rebate'].sum():.2f}")

        print(f"\nPlatform captured:")
        print(f"  Mean: ${spread_df['captured_spread'].mean():.4f}")
        print(f"  Total: ${spread_df['captured_spread'].sum():.2f}")

        print(f"\nContribution shares (when spread > 0):")
        print(f"  s_cust mean: {spread_df['s_cust'].mean():.4f}")
        print(f"  s_plat mean: {spread_df['s_plat'].mean():.4f}")

        # Check allocation accuracy
        sum_allocated = spread_df['rebate'] + spread_df['captured_spread']
        error = (sum_allocated - spread_df['spread_value']).abs()
        print(f"\nAllocation accuracy:")
        print(f"  Mean absolute error: ${error.mean():.6f}")
        print(f"  Max absolute error:  ${error.max():.6f}")

        if error.mean() < 1e-4:
            print("  ✅ Allocation is exact (mechanistic split working)")
        else:
            print("  ⚠️  Some allocation error detected")

    else:
        print("\n⚠️  No episodes with spread > 0")
        print("   This is expected if wage parity never exceeded 1.05")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    import sys

    log_path = 'logs/sac_train.csv'
    output_path = 'plots/spread_allocation.png'

    if len(sys.argv) > 1:
        log_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_path = sys.argv[2]

    # Ensure plots directory exists
    Path('plots').mkdir(exist_ok=True)

    plot_spread_allocation(log_path, output_path)
