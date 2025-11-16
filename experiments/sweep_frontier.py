"""
Feasibility sweep: validate SLA is reachable.

Grid over (speed, care) ∈ [0,1]² and plot:
- Iso-error contours
- Iso-MP contours
- Verify 6% SLA intersects MP ≥ 80/hr
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.envs.dishwashing_env import DishwashingEnv, DishwashingParams


def compute_frontier(grid_size=50, episodes_per_point=100, steps_per_episode=60):
    """
    Sweep (speed, care) grid and compute expected MP and error rate.

    Returns:
        speed_grid, care_grid, mp_grid, err_grid
    """
    params = DishwashingParams()
    env = DishwashingEnv(params)

    speed_vals = np.linspace(0, 1, grid_size)
    care_vals = np.linspace(0, 1, grid_size)

    mp_grid = np.zeros((grid_size, grid_size))
    err_grid = np.zeros((grid_size, grid_size))

    print(f"Computing frontier: {grid_size}x{grid_size} grid, {episodes_per_point} episodes/point")

    for i, speed in enumerate(speed_vals):
        for j, care in enumerate(care_vals):
            # Run multiple episodes at this (speed, care)
            mp_samples = []
            err_samples = []

            for ep in range(episodes_per_point):
                env.reset()
                action = np.array([speed, care])

                for step in range(steps_per_episode):
                    obs, info, done = env.step(action)

                # Episode stats
                time_hours = obs['t'] / 3600.0
                mp = obs['completed'] / time_hours if time_hours > 0 else 0
                err_rate = obs['errors'] / max(1, obs['attempts'])

                mp_samples.append(mp)
                err_samples.append(err_rate)

            mp_grid[j, i] = np.mean(mp_samples)
            err_grid[j, i] = np.mean(err_samples)

        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{grid_size} speed values")

    return speed_vals, care_vals, mp_grid, err_grid


def plot_frontier(speed_vals, care_vals, mp_grid, err_grid):
    """Plot iso-error and iso-MP contours."""

    os.makedirs('plots', exist_ok=True)

    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Error rate heatmap with contours
    ax = axes[0]
    im = ax.contourf(speed_vals, care_vals, err_grid, levels=20, cmap='RdYlGn_r')
    contours = ax.contour(speed_vals, care_vals, err_grid, levels=[0.06, 0.08, 0.10, 0.15, 0.20],
                          colors='black', linewidths=1.5, linestyles='--')
    ax.clabel(contours, inline=True, fontsize=9, fmt='%.2f')

    # Highlight 6% SLA
    ax.contour(speed_vals, care_vals, err_grid, levels=[0.06],
              colors='red', linewidths=3, linestyles='-')

    ax.set_xlabel('Speed', fontsize=12)
    ax.set_ylabel('Care', fontsize=12)
    ax.set_title('Error Rate (6% SLA in red)', fontsize=13, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Error Rate', fontsize=11)
    ax.grid(True, alpha=0.3)

    # Plot 2: MP heatmap with contours
    ax = axes[1]
    im = ax.contourf(speed_vals, care_vals, mp_grid, levels=20, cmap='viridis')
    contours = ax.contour(speed_vals, care_vals, mp_grid, levels=[60, 80, 100, 120],
                          colors='white', linewidths=1.5, linestyles='--')
    ax.clabel(contours, inline=True, fontsize=9, fmt='%.0f')

    # Highlight 80/hr threshold
    ax.contour(speed_vals, care_vals, mp_grid, levels=[80],
              colors='cyan', linewidths=3, linestyles='-')

    ax.set_xlabel('Speed', fontsize=12)
    ax.set_ylabel('Care', fontsize=12)
    ax.set_title('Marginal Product (80/hr in cyan)', fontsize=13, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('MP (dishes/hr)', fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('plots/feasibility_frontier.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: plots/feasibility_frontier.png")

    # Combined overlay plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # MP contours (background)
    im = ax.contourf(speed_vals, care_vals, mp_grid, levels=15, cmap='viridis', alpha=0.6)
    mp_contours = ax.contour(speed_vals, care_vals, mp_grid, levels=[60, 80, 100, 120],
                             colors='white', linewidths=1.5, linestyles='--', alpha=0.8)
    ax.clabel(mp_contours, inline=True, fontsize=9, fmt='MP=%.0f')

    # Error contours (overlay)
    err_contours = ax.contour(speed_vals, care_vals, err_grid, levels=[0.06, 0.08, 0.10, 0.15],
                              colors='red', linewidths=2, linestyles='-', alpha=0.9)
    ax.clabel(err_contours, inline=True, fontsize=10, fmt='err=%.2f')

    ax.set_xlabel('Speed', fontsize=12)
    ax.set_ylabel('Care', fontsize=12)
    ax.set_title('Feasibility Frontier: MP (background) + Error (red contours)',
                 fontsize=13, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('MP (dishes/hr)', fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('plots/feasibility_combined.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: plots/feasibility_combined.png")


def analyze_feasibility(speed_vals, care_vals, mp_grid, err_grid, sla_target=0.06, mp_target=80):
    """Check if SLA is feasible and find optimal operating points."""

    print("\n" + "="*60)
    print("FEASIBILITY ANALYSIS")
    print("="*60)

    # Find points that meet SLA
    sla_mask = err_grid <= sla_target
    feasible_count = np.sum(sla_mask)

    print(f"\nSLA Target: {sla_target:.1%} error rate")
    print(f"Feasible points (err ≤ {sla_target:.1%}): {feasible_count}/{sla_mask.size}")

    if feasible_count > 0:
        # Find best MP among feasible points
        mp_feasible = mp_grid.copy()
        mp_feasible[~sla_mask] = -np.inf

        best_idx = np.unravel_index(np.argmax(mp_feasible), mp_feasible.shape)
        best_mp = mp_grid[best_idx]
        best_err = err_grid[best_idx]
        best_speed = speed_vals[best_idx[1]]
        best_care = care_vals[best_idx[0]]

        print(f"\n✅ FEASIBLE: SLA can be met!")
        print(f"\nBest feasible operating point:")
        print(f"  speed = {best_speed:.2f}")
        print(f"  care = {best_care:.2f}")
        print(f"  MP = {best_mp:.1f} dishes/hr")
        print(f"  Error rate = {best_err:.1%}")

        # Check if MP target is also met
        viable_mask = sla_mask & (mp_grid >= mp_target)
        viable_count = np.sum(viable_mask)

        print(f"\nViable points (err ≤ {sla_target:.1%} AND MP ≥ {mp_target}/hr): {viable_count}")

        if viable_count > 0:
            print(f"✅ VIABLE: SLA + MP target both achievable!")

            # Find middle-ground viable point
            viable_indices = np.argwhere(viable_mask)
            mid_idx = viable_indices[len(viable_indices)//2]
            mid_speed = speed_vals[mid_idx[1]]
            mid_care = care_vals[mid_idx[0]]
            mid_mp = mp_grid[mid_idx[0], mid_idx[1]]
            mid_err = err_grid[mid_idx[0], mid_idx[1]]

            print(f"\nSample viable point:")
            print(f"  speed = {mid_speed:.2f}")
            print(f"  care = {mid_care:.2f}")
            print(f"  MP = {mid_mp:.1f} dishes/hr")
            print(f"  Error rate = {mid_err:.1%}")
        else:
            print(f"⚠️  SLA is feasible but MP < {mp_target}/hr")
            print(f"   Consider increasing BASE_RATE or reducing k_err")

    else:
        print(f"\n❌ INFEASIBLE: No (speed, care) achieves err ≤ {sla_target:.1%}")
        print(f"   Minimum error rate: {err_grid.min():.1%}")
        print(f"   Increase p_min or reduce k_err to fix")

    # Find unconstrained optimum (max MP, ignoring errors)
    max_mp_idx = np.unravel_index(np.argmax(mp_grid), mp_grid.shape)
    max_mp = mp_grid[max_mp_idx]
    max_mp_err = err_grid[max_mp_idx]
    max_mp_speed = speed_vals[max_mp_idx[1]]
    max_mp_care = care_vals[max_mp_idx[0]]

    print(f"\nUnconstrained maximum MP:")
    print(f"  speed = {max_mp_speed:.2f}")
    print(f"  care = {max_mp_care:.2f}")
    print(f"  MP = {max_mp:.1f} dishes/hr")
    print(f"  Error rate = {max_mp_err:.1%}")

    print("\n" + "="*60)


if __name__ == "__main__":
    print("Running feasibility sweep...")
    speed_vals, care_vals, mp_grid, err_grid = compute_frontier(
        grid_size=50,
        episodes_per_point=50,
        steps_per_episode=60
    )

    plot_frontier(speed_vals, care_vals, mp_grid, err_grid)
    analyze_feasibility(speed_vals, care_vals, mp_grid, err_grid)

    print("\n✅ Feasibility sweep complete!")
