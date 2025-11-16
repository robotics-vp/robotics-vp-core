"""
Extract key metrics from training logs to validate claims in documentation.

Backs quantitative statements in:
- ECON_ARCHITECTURE.md
- V2P_TECHNICAL_OVERVIEW.md
- INVESTOR_STORY.md
"""
import pandas as pd
import numpy as np
from pathlib import Path


def load_logs(log_path='logs/sac_train.csv'):
    """Load training logs."""
    return pd.read_csv(log_path)


def compute_summary_statistics(df, window=100):
    """
    Compute key metrics for documentation.

    Args:
        df: Training log dataframe
        window: Rolling window for averages (final metrics)
    """
    # Final window metrics (last N episodes)
    final_window = df.tail(window)

    summary = {
        'training': {
            'total_episodes': len(df),
            'final_episode': int(df['episode'].max()),
        },
        'performance': {
            'final_mp_mean': final_window['mp_r'].mean(),
            'final_mp_std': final_window['mp_r'].std(),
            'final_err_mean': final_window['err_rate'].mean(),
            'final_err_std': final_window['err_rate'].std(),
            'err_target': df['err_target'].iloc[-1],
            'err_margin': (df['err_target'].iloc[-1] - final_window['err_rate'].mean()) / df['err_target'].iloc[-1],
        },
        'economics': {
            'final_wage_parity_mean': final_window['wage_parity'].mean(),
            'final_wage_parity_std': final_window['wage_parity'].std(),
            'final_profit_mean': final_window['profit'].mean(),
            'final_profit_std': final_window['profit'].std(),
            'human_wage': float(final_window['w_h'].iloc[0]),
            'robot_wage_mean': final_window['w_hat_r'].mean(),
        },
        'spread_allocation': {},
        'convergence': {
            'episodes_above_parity': (df['wage_parity'] > 1.0).sum(),
            'episodes_meeting_sla': (df['err_rate'] <= df['err_target']).sum(),
            'pct_episodes_viable': 100 * ((df['wage_parity'] > 1.0) & (df['err_rate'] <= df['err_target'])).sum() / len(df),
        }
    }

    # Spread allocation (if columns exist)
    if 'spread' in df.columns:
        spread_df = df[df['spread'] > 0]
        if len(spread_df) > 0:
            summary['spread_allocation'] = {
                'episodes_with_spread': len(spread_df),
                'pct_with_spread': 100 * len(spread_df) / len(df),
                'spread_mean': spread_df['spread'].mean(),
                'spread_std': spread_df['spread'].std(),
                'spread_value_mean': spread_df['spread_value'].mean(),
                's_cust_mean': spread_df['s_cust'].mean(),
                's_plat_mean': spread_df['s_plat'].mean(),
                'rebate_total': spread_df['rebate'].sum(),
                'captured_total': spread_df['captured_spread'].sum(),
                'rebate_mean': spread_df['rebate'].mean(),
                'captured_mean': spread_df['captured_spread'].mean(),
            }

    return summary


def format_summary(summary):
    """Format summary as readable text."""
    lines = []
    lines.append("=" * 70)
    lines.append("TRAINING SUMMARY STATISTICS")
    lines.append("=" * 70)

    # Training
    lines.append("\n[TRAINING]")
    lines.append(f"  Total episodes: {summary['training']['total_episodes']}")
    lines.append(f"  Final episode: {summary['training']['final_episode']}")

    # Performance
    perf = summary['performance']
    lines.append("\n[PERFORMANCE] (last 100 episodes)")
    lines.append(f"  Marginal Product: {perf['final_mp_mean']:.1f} ± {perf['final_mp_std']:.1f} units/hr")
    lines.append(f"  Error Rate: {100*perf['final_err_mean']:.2f}% ± {100*perf['final_err_std']:.2f}%")
    lines.append(f"  Error Target: {100*perf['err_target']:.2f}%")
    lines.append(f"  Error Margin: {100*perf['err_margin']:.1f}% below target")

    # Economics
    econ = summary['economics']
    lines.append("\n[ECONOMICS] (last 100 episodes)")
    lines.append(f"  Robot Wage: ${econ['robot_wage_mean']:.2f}/hr ± ${econ['final_profit_std']:.2f}")
    lines.append(f"  Human Wage: ${econ['human_wage']:.2f}/hr")
    lines.append(f"  Wage Parity: {econ['final_wage_parity_mean']:.3f} ± {econ['final_wage_parity_std']:.3f}")
    lines.append(f"  Profit: ${econ['final_profit_mean']:.2f}/hr ± ${econ['final_profit_std']:.2f}")

    wage_premium_pct = 100 * (econ['final_wage_parity_mean'] - 1.0)
    lines.append(f"  Wage Premium: {wage_premium_pct:.1f}% above human")

    # Spread allocation
    if summary['spread_allocation']:
        spread = summary['spread_allocation']
        lines.append("\n[SPREAD ALLOCATION] (episodes with spread > 0)")
        lines.append(f"  Episodes with spread: {spread['episodes_with_spread']}/{summary['training']['total_episodes']} ({spread['pct_with_spread']:.1f}%)")
        lines.append(f"  Spread: ${spread['spread_mean']:.2f}/hr ± ${spread['spread_std']:.2f}")
        lines.append(f"  Spread value: ${spread['spread_value_mean']:.2f}/episode")
        lines.append(f"  Customer share (s_cust): {100*spread['s_cust_mean']:.1f}%")
        lines.append(f"  Platform share (s_plat): {100*spread['s_plat_mean']:.1f}%")
        lines.append(f"  Total rebate: ${spread['rebate_total']:.2f}")
        lines.append(f"  Total captured: ${spread['captured_total']:.2f}")
        lines.append(f"  Mean rebate: ${spread['rebate_mean']:.2f}/episode")
        lines.append(f"  Mean captured: ${spread['captured_mean']:.2f}/episode")

    # Convergence
    conv = summary['convergence']
    lines.append("\n[CONVERGENCE]")
    lines.append(f"  Episodes above parity: {conv['episodes_above_parity']}/{summary['training']['total_episodes']} ({100*conv['episodes_above_parity']/summary['training']['total_episodes']:.1f}%)")
    lines.append(f"  Episodes meeting SLA: {conv['episodes_meeting_sla']}/{summary['training']['total_episodes']} ({100*conv['episodes_meeting_sla']/summary['training']['total_episodes']:.1f}%)")
    lines.append(f"  Episodes viable (parity>1 AND err<=target): {conv['pct_episodes_viable']:.1f}%")

    lines.append("\n" + "=" * 70)

    return "\n".join(lines)


def compute_investor_metrics(summary, deployments=1000, hours_per_year=1000):
    """
    Compute investor-relevant metrics.

    Args:
        summary: Summary statistics from compute_summary_statistics
        deployments: Number of robot deployments
        hours_per_year: Operating hours per robot per year
    """
    spread = summary['spread_allocation']
    if not spread:
        return None

    # Revenue calculations
    spread_per_hr = spread['spread_mean']
    platform_share = spread['s_plat_mean']
    revenue_per_hr = platform_share * spread_per_hr
    revenue_per_robot_year = revenue_per_hr * hours_per_year
    total_annual_revenue = revenue_per_robot_year * deployments

    # Customer economics
    human_wage = summary['economics']['human_wage']
    customer_rebate_per_hr = spread['rebate_mean']
    customer_net_cost = human_wage - customer_rebate_per_hr
    customer_savings_pct = 100 * customer_rebate_per_hr / human_wage

    # Cost assumptions (from INVESTOR_STORY.md)
    cost_per_robot_year = 1700  # Compute + hardware + support
    gross_profit_per_robot = revenue_per_robot_year - cost_per_robot_year
    gross_margin = 100 * gross_profit_per_robot / revenue_per_robot_year

    investor = {
        'revenue': {
            'spread_per_hr': spread_per_hr,
            'platform_share_pct': 100 * platform_share,
            'revenue_per_hr': revenue_per_hr,
            'revenue_per_robot_year': revenue_per_robot_year,
            'total_annual_revenue': total_annual_revenue,
        },
        'customer': {
            'human_wage': human_wage,
            'rebate_per_hr': customer_rebate_per_hr,
            'net_cost_per_hr': customer_net_cost,
            'savings_pct': customer_savings_pct,
            'cost_per_year': customer_net_cost * hours_per_year,
        },
        'unit_economics': {
            'cost_per_robot_year': cost_per_robot_year,
            'gross_profit_per_robot': gross_profit_per_robot,
            'gross_margin_pct': gross_margin,
            'total_gross_profit': gross_profit_per_robot * deployments,
        },
        'assumptions': {
            'deployments': deployments,
            'hours_per_year': hours_per_year,
        }
    }

    return investor


def format_investor_metrics(investor):
    """Format investor metrics as readable text."""
    if not investor:
        return "No spread allocation data available for investor metrics."

    lines = []
    lines.append("\n" + "=" * 70)
    lines.append("INVESTOR METRICS")
    lines.append("=" * 70)

    # Revenue
    rev = investor['revenue']
    lines.append("\n[PLATFORM REVENUE]")
    lines.append(f"  Spread (robot vs human): ${rev['spread_per_hr']:.2f}/hr")
    lines.append(f"  Platform share: {rev['platform_share_pct']:.1f}%")
    lines.append(f"  Revenue per robot-hour: ${rev['revenue_per_hr']:.2f}/hr")
    lines.append(f"  Revenue per robot-year: ${rev['revenue_per_robot_year']:,.0f}/year")
    lines.append(f"  Total annual revenue ({investor['assumptions']['deployments']} robots): ${rev['total_annual_revenue']:,.0f}/year")

    # Customer
    cust = investor['customer']
    lines.append("\n[CUSTOMER ECONOMICS]")
    lines.append(f"  Human wage: ${cust['human_wage']:.2f}/hr")
    lines.append(f"  Rebate per hour: ${cust['rebate_per_hr']:.2f}/hr")
    lines.append(f"  Net cost per hour: ${cust['net_cost_per_hr']:.2f}/hr")
    lines.append(f"  Savings vs human: {cust['savings_pct']:.1f}%")
    lines.append(f"  Annual cost per robot: ${cust['cost_per_year']:,.0f}/year")

    # Unit economics
    unit = investor['unit_economics']
    lines.append("\n[UNIT ECONOMICS]")
    lines.append(f"  Cost per robot-year: ${unit['cost_per_robot_year']:,.0f}")
    lines.append(f"  Gross profit per robot: ${unit['gross_profit_per_robot']:,.0f}")
    lines.append(f"  Gross margin: {unit['gross_margin_pct']:.1f}%")
    lines.append(f"  Total gross profit ({investor['assumptions']['deployments']} robots): ${unit['total_gross_profit']:,.0f}/year")

    lines.append("\n" + "=" * 70)

    return "\n".join(lines)


def main(log_path='logs/sac_train.csv', output_path='experiments/summary_snapshot.txt'):
    """Generate summary snapshot."""
    print(f"Loading logs: {log_path}")
    df = load_logs(log_path)

    print(f"Computing summary statistics...")
    summary = compute_summary_statistics(df, window=100)

    # Format and print
    text = format_summary(summary)
    print(text)

    # Investor metrics
    print("\nComputing investor metrics...")
    investor = compute_investor_metrics(summary, deployments=1000, hours_per_year=1000)
    investor_text = format_investor_metrics(investor)
    print(investor_text)

    # Save to file
    Path(output_path).parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        f.write("QUANTITATIVE VALIDATION FOR DOCUMENTATION\n")
        f.write(f"Source: {log_path}\n")
        f.write(f"Generated: {pd.Timestamp.now()}\n\n")
        f.write(text)
        f.write("\n")
        f.write(investor_text)

    print(f"\n✅ Summary saved: {output_path}")


if __name__ == "__main__":
    import sys

    log_path = 'logs/sac_train.csv'
    output_path = 'experiments/summary_snapshot.txt'

    if len(sys.argv) > 1:
        log_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_path = sys.argv[2]

    main(log_path, output_path)
