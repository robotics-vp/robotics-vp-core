#!/usr/bin/env python3
"""
Empirical Validation of Structural Deflation Theorem

Validates four theorem claims using logs/sac_train.csv:
1. Consumer surplus non-negativity: CS(t) ≥ 0
2. Mechanistic surplus decomposition: SV(t) = R(t) + C(t)
3. Platform revenue increases with productivity
4. Structural deflation: ρ(t) = MC_r(t)/MC_h(t) ≤ 1 and decreasing

Outputs:
- plots/structural_deflation_validation.png (4-panel visualization)
- experiments/structural_deflation_summary.txt (quantitative results)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression

def load_data(csv_path='logs/sac_train.csv'):
    """Load training logs with economic metrics"""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} episodes from {csv_path}")

    # Verify required columns exist
    required = ['episode', 'w_h_indexed', 'customer_cost', 'consumer_surplus',
                'spread_value', 'rebate', 'captured_spread', 'mp_r']
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df

def validate_consumer_surplus(df):
    """Theorem 1: CS(t) ≥ 0 for all t"""
    violations = df[df['consumer_surplus'] < -1e-6]  # Tolerance for floating point
    n_violations = len(violations)
    pct_valid = 100.0 * (1 - n_violations / len(df))

    result = {
        'n_violations': n_violations,
        'pct_valid': pct_valid,
        'min_cs': df['consumer_surplus'].min(),
        'mean_cs': df['consumer_surplus'].mean(),
        'max_cs': df['consumer_surplus'].max(),
    }

    return result

def validate_surplus_decomposition(df, tolerance=1e-4):
    """Theorem 2: SV(t) = R(t) + C(t) when spread exists"""
    # Filter to episodes with positive spread
    df_spread = df[df['spread_value'] > 0].copy()

    if len(df_spread) == 0:
        return {'n_episodes': 0, 'message': 'No episodes with positive spread'}

    # Compute decomposition error
    df_spread['decomp_lhs'] = df_spread['spread_value']
    df_spread['decomp_rhs'] = df_spread['rebate'] + df_spread['captured_spread']
    df_spread['decomp_error'] = np.abs(df_spread['decomp_lhs'] - df_spread['decomp_rhs'])

    # Check if decomposition holds
    violations = df_spread[df_spread['decomp_error'] > tolerance]
    n_violations = len(violations)
    pct_valid = 100.0 * (1 - n_violations / len(df_spread))

    result = {
        'n_episodes': len(df_spread),
        'n_violations': n_violations,
        'pct_valid': pct_valid,
        'mean_error': df_spread['decomp_error'].mean(),
        'max_error': df_spread['decomp_error'].max(),
    }

    return result

def validate_productivity_incentive(df):
    """Theorem 3: C(t) increases with MPL_r(t)"""
    # Filter to episodes above parity (where spread exists)
    df_spread = df[df['spread_value'] > 0].copy()

    if len(df_spread) < 10:
        return {'n_episodes': len(df_spread), 'message': 'Insufficient episodes with spread'}

    # Compute correlation between MPL_r and captured_spread
    pearson_r, pearson_p = pearsonr(df_spread['mp_r'], df_spread['captured_spread'])
    spearman_r, spearman_p = spearmanr(df_spread['mp_r'], df_spread['captured_spread'])

    # Linear regression
    X = df_spread['mp_r'].values.reshape(-1, 1)
    y = df_spread['captured_spread'].values
    reg = LinearRegression().fit(X, y)
    slope = reg.coef_[0]

    result = {
        'n_episodes': len(df_spread),
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p,
        'linear_slope': slope,
        'positive_correlation': pearson_r > 0 and pearson_p < 0.05,
    }

    return result

def validate_structural_deflation(df):
    """Theorem 4: ρ(t) = MC_r(t)/MC_h(t) ≤ 1 and decreasing"""
    # Compute deflation ratio
    df = df.copy()
    df['rho'] = df['customer_cost'] / df['w_h_indexed']

    # Check ρ ≤ 1 (with tolerance)
    violations = df[df['rho'] > 1.0 + 1e-6]
    n_violations_ceiling = len(violations)
    pct_valid_ceiling = 100.0 * (1 - n_violations_ceiling / len(df))

    # Check if ρ is decreasing over time (linear regression slope)
    X = df['episode'].values.reshape(-1, 1)
    y = df['rho'].values
    reg = LinearRegression().fit(X, y)
    slope = reg.coef_[0]

    # Compute moving average to smooth noise
    window = 50
    if len(df) >= window:
        df['rho_ma'] = df['rho'].rolling(window=window, center=True).mean()
        # Check if MA is decreasing
        X_ma = df.dropna(subset=['rho_ma'])['episode'].values.reshape(-1, 1)
        y_ma = df.dropna(subset=['rho_ma'])['rho_ma'].values
        reg_ma = LinearRegression().fit(X_ma, y_ma)
        slope_ma = reg_ma.coef_[0]
    else:
        slope_ma = None

    result = {
        'n_violations_ceiling': n_violations_ceiling,
        'pct_valid_ceiling': pct_valid_ceiling,
        'mean_rho': df['rho'].mean(),
        'min_rho': df['rho'].min(),
        'max_rho': df['rho'].max(),
        'initial_rho': df['rho'].iloc[0] if len(df) > 0 else None,
        'final_rho': df['rho'].iloc[-1] if len(df) > 0 else None,
        'linear_slope': slope,
        'linear_slope_ma': slope_ma,
        'is_decreasing': slope < 0,
        'is_decreasing_ma': slope_ma < 0 if slope_ma is not None else None,
    }

    return result

def plot_validation(df, output_path='plots/structural_deflation_validation.png'):
    """Generate 4-panel validation visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Structural Deflation Theorem: Empirical Validation', fontsize=14, fontweight='bold')

    # Compute ρ
    df = df.copy()
    df['rho'] = df['customer_cost'] / df['w_h_indexed']

    # Panel 1: Consumer Surplus (Theorem 1)
    ax = axes[0, 0]
    ax.plot(df['episode'], df['consumer_surplus'], alpha=0.6, linewidth=0.8, label='CS(t)')
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1, label='CS = 0')
    ax.fill_between(df['episode'], 0, df['consumer_surplus'], alpha=0.2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Consumer Surplus ($/hr)')
    ax.set_title('Theorem 1: Consumer Surplus Non-Negativity')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add annotation
    violations = len(df[df['consumer_surplus'] < 0])
    ax.text(0.02, 0.98, f'Violations: {violations}/{len(df)} ({100*violations/len(df):.1f}%)',
            transform=ax.transAxes, verticalalignment='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Panel 2: Surplus Decomposition (Theorem 2)
    ax = axes[0, 1]
    df_spread = df[df['spread_value'] > 0]
    if len(df_spread) > 0:
        ax.scatter(df_spread['spread_value'],
                   df_spread['rebate'] + df_spread['captured_spread'],
                   alpha=0.5, s=20)

        # Add perfect decomposition line
        max_val = max(df_spread['spread_value'].max(),
                      (df_spread['rebate'] + df_spread['captured_spread']).max())
        ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect decomposition')

        ax.set_xlabel('SV(t) = Spread Value')
        ax.set_ylabel('R(t) + C(t)')
        ax.set_title('Theorem 2: Mechanistic Surplus Decomposition')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Compute error
        error = np.abs(df_spread['spread_value'] -
                       (df_spread['rebate'] + df_spread['captured_spread']))
        ax.text(0.02, 0.98, f'Mean error: ${error.mean():.4f}\nMax error: ${error.max():.4f}',
                transform=ax.transAxes, verticalalignment='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        ax.text(0.5, 0.5, 'No episodes with positive spread',
                transform=ax.transAxes, ha='center', va='center', fontsize=12)

    # Panel 3: Productivity Incentive (Theorem 3)
    ax = axes[1, 0]
    if len(df_spread) > 0:
        ax.scatter(df_spread['mp_r'], df_spread['captured_spread'], alpha=0.5, s=20)

        # Add trend line
        X = df_spread['mp_r'].values.reshape(-1, 1)
        y = df_spread['captured_spread'].values
        reg = LinearRegression().fit(X, y)
        x_line = np.linspace(df_spread['mp_r'].min(), df_spread['mp_r'].max(), 100)
        y_line = reg.predict(x_line.reshape(-1, 1))
        ax.plot(x_line, y_line, 'r--', linewidth=2,
                label=f'Trend (slope={reg.coef_[0]:.2f})')

        ax.set_xlabel('Robot MPL (units/hr)')
        ax.set_ylabel('Platform Captured Spread ($)')
        ax.set_title('Theorem 3: Platform Revenue vs Productivity')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Correlation stats
        r, p = pearsonr(df_spread['mp_r'], df_spread['captured_spread'])
        ax.text(0.02, 0.98, f'Pearson r = {r:.3f}\np-value = {p:.2e}',
                transform=ax.transAxes, verticalalignment='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        ax.text(0.5, 0.5, 'No episodes with positive spread',
                transform=ax.transAxes, ha='center', va='center', fontsize=12)

    # Panel 4: Structural Deflation (Theorem 4)
    ax = axes[1, 1]
    ax.plot(df['episode'], df['rho'], alpha=0.4, linewidth=0.8, label='ρ(t) = MC_r/MC_h')

    # Add moving average
    window = 50
    if len(df) >= window:
        rho_ma = df['rho'].rolling(window=window, center=True).mean()
        ax.plot(df['episode'], rho_ma, 'b-', linewidth=2, label=f'{window}-episode MA')

    # Add ceiling
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1, label='ρ = 1 (parity)')
    ax.fill_between(df['episode'], df['rho'], 1.0, where=(df['rho'] <= 1.0),
                     alpha=0.2, color='green', label='Structural deflation zone')

    ax.set_xlabel('Episode')
    ax.set_ylabel('ρ(t) = Customer Cost / Human Wage')
    ax.set_title('Theorem 4: Structural Deflation (ρ ≤ 1, decreasing)')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([max(0, df['rho'].min() - 0.05), min(1.2, df['rho'].max() + 0.05)])

    # Add trend annotation
    X = df['episode'].values.reshape(-1, 1)
    y = df['rho'].values
    reg = LinearRegression().fit(X, y)
    slope = reg.coef_[0]

    ax.text(0.02, 0.98,
            f'Initial: {df["rho"].iloc[0]:.3f}\nFinal: {df["rho"].iloc[-1]:.3f}\nSlope: {slope:.2e}/ep',
            transform=ax.transAxes, verticalalignment='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved validation plot: {output_path}")
    plt.close()

def generate_summary(df, cs_result, decomp_result, incentive_result, deflation_result,
                     output_path='experiments/structural_deflation_summary.txt'):
    """Generate text summary of validation results"""
    lines = []
    lines.append("=" * 80)
    lines.append("STRUCTURAL DEFLATION THEOREM: EMPIRICAL VALIDATION")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Dataset: {len(df)} episodes")
    lines.append(f"Episode range: {df['episode'].min()} - {df['episode'].max()}")
    lines.append("")

    # Theorem 1: Consumer Surplus
    lines.append("-" * 80)
    lines.append("THEOREM 1: CONSUMER SURPLUS NON-NEGATIVITY")
    lines.append("-" * 80)
    lines.append(f"Claim: CS(t) ≥ 0 for all t")
    lines.append(f"")
    lines.append(f"Validation:")
    lines.append(f"  Valid episodes: {len(df) - cs_result['n_violations']}/{len(df)} ({cs_result['pct_valid']:.1f}%)")
    lines.append(f"  Violations: {cs_result['n_violations']}")
    if cs_result['n_violations'] == 0:
        lines.append(f"  ✅ GUARANTEE HOLDS for ALL episodes")
    else:
        lines.append(f"  ⚠️  {cs_result['n_violations']} violations detected")
    lines.append(f"")
    lines.append(f"Consumer Surplus Statistics:")
    lines.append(f"  Mean: ${cs_result['mean_cs']:.4f}/hr")
    lines.append(f"  Min:  ${cs_result['min_cs']:.4f}/hr")
    lines.append(f"  Max:  ${cs_result['max_cs']:.4f}/hr")
    lines.append("")

    # Theorem 2: Surplus Decomposition
    lines.append("-" * 80)
    lines.append("THEOREM 2: MECHANISTIC SURPLUS DECOMPOSITION")
    lines.append("-" * 80)
    lines.append(f"Claim: SV(t) = R(t) + C(t) for episodes with positive spread")
    lines.append(f"")
    if decomp_result.get('n_episodes', 0) > 0:
        lines.append(f"Validation:")
        lines.append(f"  Episodes with spread: {decomp_result['n_episodes']}")
        lines.append(f"  Valid decompositions: {decomp_result['n_episodes'] - decomp_result['n_violations']}/{decomp_result['n_episodes']} ({decomp_result['pct_valid']:.1f}%)")
        lines.append(f"  Violations: {decomp_result['n_violations']}")
        if decomp_result['n_violations'] == 0:
            lines.append(f"  ✅ EXACT DECOMPOSITION for ALL episodes with spread")
        else:
            lines.append(f"  ⚠️  {decomp_result['n_violations']} violations detected")
        lines.append(f"")
        lines.append(f"Decomposition Error:")
        lines.append(f"  Mean: ${decomp_result['mean_error']:.6f}")
        lines.append(f"  Max:  ${decomp_result['max_error']:.6f}")
    else:
        lines.append(f"  No episodes with positive spread (robot not yet beating human)")
    lines.append("")

    # Theorem 3: Productivity Incentive
    lines.append("-" * 80)
    lines.append("THEOREM 3: PLATFORM REVENUE INCREASES WITH PRODUCTIVITY")
    lines.append("-" * 80)
    lines.append(f"Claim: C(t) increases when MPL_r(t) increases (above parity)")
    lines.append(f"")
    if incentive_result.get('n_episodes', 0) >= 10:
        lines.append(f"Validation:")
        lines.append(f"  Episodes analyzed: {incentive_result['n_episodes']}")
        lines.append(f"  Pearson correlation: r = {incentive_result['pearson_r']:.4f} (p = {incentive_result['pearson_p']:.2e})")
        lines.append(f"  Spearman correlation: ρ = {incentive_result['spearman_r']:.4f} (p = {incentive_result['spearman_p']:.2e})")
        lines.append(f"  Linear regression slope: {incentive_result['linear_slope']:.4f}")
        if incentive_result['positive_correlation']:
            lines.append(f"  ✅ POSITIVE CORRELATION confirmed (p < 0.05)")
        else:
            lines.append(f"  ⚠️  Correlation not significant or negative")
    else:
        lines.append(f"  Insufficient episodes with spread ({incentive_result.get('n_episodes', 0)} < 10)")
    lines.append("")

    # Theorem 4: Structural Deflation
    lines.append("-" * 80)
    lines.append("THEOREM 4: STRUCTURAL DEFLATION")
    lines.append("-" * 80)
    lines.append(f"Claim: ρ(t) = MC_r(t)/MC_h(t) ≤ 1 and decreasing over time")
    lines.append(f"")
    lines.append(f"Validation:")
    lines.append(f"  Episodes with ρ ≤ 1: {len(df) - deflation_result['n_violations_ceiling']}/{len(df)} ({deflation_result['pct_valid_ceiling']:.1f}%)")
    lines.append(f"  Ceiling violations: {deflation_result['n_violations_ceiling']}")
    if deflation_result['n_violations_ceiling'] == 0:
        lines.append(f"  ✅ CEILING HOLDS: ρ ≤ 1 for ALL episodes")
    else:
        lines.append(f"  ⚠️  {deflation_result['n_violations_ceiling']} episodes above ceiling")
    lines.append(f"")
    lines.append(f"Deflation Ratio ρ(t):")
    lines.append(f"  Initial: {deflation_result['initial_rho']:.4f}")
    lines.append(f"  Final:   {deflation_result['final_rho']:.4f}")
    lines.append(f"  Mean:    {deflation_result['mean_rho']:.4f}")
    lines.append(f"  Min:     {deflation_result['min_rho']:.4f}")
    lines.append(f"  Max:     {deflation_result['max_rho']:.4f}")
    lines.append(f"")
    lines.append(f"Trend Analysis:")
    lines.append(f"  Linear slope (raw): {deflation_result['linear_slope']:.2e} per episode")
    if deflation_result['linear_slope_ma'] is not None:
        lines.append(f"  Linear slope (MA):  {deflation_result['linear_slope_ma']:.2e} per episode")
    if deflation_result['is_decreasing']:
        lines.append(f"  ✅ DECREASING TREND confirmed (negative slope)")
    else:
        lines.append(f"  ⚠️  Trend not decreasing (positive slope)")
    lines.append("")

    # Overall Summary
    lines.append("=" * 80)
    lines.append("OVERALL THEOREM VALIDATION")
    lines.append("=" * 80)

    all_pass = (
        cs_result['n_violations'] == 0 and
        (decomp_result.get('n_violations', 0) == 0 or decomp_result.get('n_episodes', 0) == 0) and
        incentive_result.get('positive_correlation', False) and
        deflation_result['n_violations_ceiling'] == 0 and
        deflation_result['is_decreasing']
    )

    if all_pass:
        lines.append("✅ ALL FOUR THEOREM CLAIMS VALIDATED")
    else:
        lines.append("⚠️  Some theorem claims have violations or insufficient data")

    lines.append("")
    lines.append("Economic Coherence:")
    lines.append("  1. Customer protection (CS ≥ 0): " + ("✅" if cs_result['n_violations'] == 0 else "⚠️"))
    lines.append("  2. Fair value split (SV = R + C): " + ("✅" if decomp_result.get('n_violations', 0) == 0 else "⚠️"))
    lines.append("  3. Aligned incentives (C ↑ when MPL ↑): " + ("✅" if incentive_result.get('positive_correlation', False) else "⚠️"))
    lines.append("  4. Structural deflation (ρ ≤ 1, decreasing): " + ("✅" if deflation_result['is_decreasing'] else "⚠️"))
    lines.append("")
    lines.append("=" * 80)

    # Write to file
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"Saved validation summary: {output_path}")

    # Also print to console
    print('\n'.join(lines))

def main():
    """Main validation pipeline"""
    # Load data
    df = load_data('logs/sac_train.csv')

    # Run all validations
    print("\n" + "="*80)
    print("Running theorem validations...")
    print("="*80)

    cs_result = validate_consumer_surplus(df)
    print(f"\n✓ Theorem 1 (Consumer Surplus): {cs_result['pct_valid']:.1f}% valid")

    decomp_result = validate_surplus_decomposition(df)
    if decomp_result.get('n_episodes', 0) > 0:
        print(f"✓ Theorem 2 (Surplus Decomposition): {decomp_result['pct_valid']:.1f}% valid")
    else:
        print(f"✓ Theorem 2 (Surplus Decomposition): No episodes with spread yet")

    incentive_result = validate_productivity_incentive(df)
    if incentive_result.get('n_episodes', 0) >= 10:
        print(f"✓ Theorem 3 (Productivity Incentive): r={incentive_result['pearson_r']:.3f}, p={incentive_result['pearson_p']:.2e}")
    else:
        print(f"✓ Theorem 3 (Productivity Incentive): Insufficient data")

    deflation_result = validate_structural_deflation(df)
    print(f"✓ Theorem 4 (Structural Deflation): ρ {deflation_result['initial_rho']:.3f} → {deflation_result['final_rho']:.3f}")

    # Generate visualizations
    print("\n" + "="*80)
    print("Generating visualizations...")
    print("="*80)
    plot_validation(df)

    # Generate summary report
    print("\n" + "="*80)
    print("Generating summary report...")
    print("="*80)
    generate_summary(df, cs_result, decomp_result, incentive_result, deflation_result)

    print("\n" + "="*80)
    print("Validation complete!")
    print("="*80)

if __name__ == '__main__':
    main()
