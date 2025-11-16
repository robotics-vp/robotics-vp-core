#!/usr/bin/env python3
"""
Generate scatter plots from physics training CSV to diagnose environment behavior.

Usage:
    python scripts/plot_physics_scatter.py --csv logs/sac_physics_train.csv
    python scripts/plot_physics_scatter.py --csv logs/sac_physics_train.csv --output reports/
"""

import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os


def plot_physics_scatter(csv_path, output_dir='reports'):
    """Generate scatter plots from physics training CSV"""
    os.makedirs(output_dir, exist_ok=True)

    # Load CSV
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)

    # Check for column names (handle both formats)
    mpl_col = 'mp_r' if 'mp_r' in df.columns else 'mpl'
    err_col = 'err_rate' if 'err_rate' in df.columns else 'error_rate'

    # Drop rows with NA in key columns
    df_clean = df.dropna(subset=[mpl_col, 'attempts', err_col])

    print(f"Loaded {len(df_clean)} valid episodes (dropped {len(df) - len(df_clean)} with NA)")

    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: MPL vs Attempts
    axes[0].scatter(df_clean['attempts'], df_clean[mpl_col], alpha=0.6, s=30)
    axes[0].set_xlabel('Attempts per Episode', fontsize=11)
    axes[0].set_ylabel('MPL (dishes/hr)', fontsize=11)
    axes[0].set_title('Marginal Product vs Attempts', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Add correlation
    corr_attempts_mpl = df_clean['attempts'].corr(df_clean[mpl_col])
    axes[0].text(0.05, 0.95, f'ρ = {corr_attempts_mpl:.3f}',
                 transform=axes[0].transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 2: MPL vs Error Rate
    axes[1].scatter(df_clean[err_col], df_clean[mpl_col], alpha=0.6, s=30)
    axes[1].set_xlabel('Error Rate', fontsize=11)
    axes[1].set_ylabel('MPL (dishes/hr)', fontsize=11)
    axes[1].set_title('Marginal Product vs Error Rate', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    # Add correlation
    corr_err_mpl = df_clean[err_col].corr(df_clean[mpl_col])
    axes[1].text(0.05, 0.95, f'ρ = {corr_err_mpl:.3f}',
                 transform=axes[1].transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # Save plots
    plot_path = os.path.join(output_dir, 'physics_scatter_plots.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n✅ Saved plots to {plot_path}")

    # Print summary statistics
    print(f"\n{'='*60}")
    print(f"Summary Statistics ({len(df_clean)} episodes)")
    print(f"{'='*60}")
    print(f"\nMPL (dishes/hr):")
    print(f"  Mean:  {df_clean[mpl_col].mean():>8.1f}")
    print(f"  Std:   {df_clean[mpl_col].std():>8.1f}")
    print(f"  Range: [{df_clean[mpl_col].min():.1f}, {df_clean[mpl_col].max():.1f}]")

    print(f"\nAttempts:")
    print(f"  Mean:  {df_clean['attempts'].mean():>8.2f}")
    print(f"  Std:   {df_clean['attempts'].std():>8.2f}")
    print(f"  Range: [{df_clean['attempts'].min():.0f}, {df_clean['attempts'].max():.0f}]")

    print(f"\nError Rate:")
    print(f"  Mean:  {df_clean[err_col].mean():>8.3f}")
    print(f"  Std:   {df_clean[err_col].std():>8.3f}")
    print(f"  Range: [{df_clean[err_col].min():.3f}, {df_clean[err_col].max():.3f}]")

    print(f"\nCorrelations:")
    print(f"  Attempts ↔ MPL:    {corr_attempts_mpl:>6.3f}")
    print(f"  Error Rate ↔ MPL:  {corr_err_mpl:>6.3f}")

    print(f"\n{'='*60}")

    # Sanity checks
    print("\nSanity Checks:")

    # Check if MPL is stuck at extremes
    mpl_zero_pct = (df_clean[mpl_col] == 0).sum() / len(df_clean) * 100
    mpl_high_pct = (df_clean[mpl_col] > 150).sum() / len(df_clean) * 100

    print(f"  MPL = 0:     {mpl_zero_pct:>5.1f}% of episodes")
    print(f"  MPL > 150:   {mpl_high_pct:>5.1f}% of episodes")

    if mpl_zero_pct > 80:
        print("  ⚠️  WARNING: >80% episodes have MPL=0 (environment may be broken)")
    elif mpl_high_pct > 80:
        print("  ⚠️  WARNING: >80% episodes have MPL>150 (unrealistically high)")
    else:
        print("  ✅ MPL distribution looks reasonable")

    # Check variance
    if df_clean[mpl_col].std() < 1.0:
        print("  ⚠️  WARNING: MPL variance very low (std < 1.0)")
    else:
        print("  ✅ MPL shows healthy variance")

    print(f"{'='*60}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate scatter plots from physics training CSV'
    )
    parser.add_argument(
        '--csv',
        type=str,
        default='logs/sac_physics_train.csv',
        help='Path to training CSV'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='reports',
        help='Output directory for plots'
    )
    args = parser.parse_args()

    plot_physics_scatter(args.csv, args.output)
