def wage_parity_reward(mp_r, mp_r_prev, mp_h, w_hat_r, w_h, err_rate,
                       alpha=1.0, beta=1.0, gamma=1.0, target=1.0,
                       lam_down=1.0, lam_up=0.3):
    # normalized productivity improvement
    delta_mp_norm = (mp_r - mp_r_prev) / mp_h if mp_h > 0 else 0.0
    # asymmetric wage convergence penalty
    gap = (w_hat_r / w_h) - target if w_h > 0 else 0.0
    lam = lam_down if gap < 0 else lam_up
    wage_loss = lam * (gap ** 2)
    return alpha * delta_mp_norm - beta * err_rate - gamma * wage_loss

def econ_lagrangian_reward(mp_r, err_rate, price_per_unit, damage_cost,
                           lam, err_target, energy_cost_per_hour=0.0):
    """
    Lagrangian-based economic reward with quality constraint.

    Maximizes: Profit/hr = p·MPᵣ - cₐ·(e·MPᵣ) - energy_cost
    Subject to: e ≤ e* (SLA)

    Args:
        mp_r: Robot marginal product (units/hr)
        err_rate: Error rate (fraction)
        price_per_unit: Revenue per unit ($/unit)
        damage_cost: Cost per error ($/error)
        lam: Lagrangian multiplier (updated via dual ascent)
        err_target: Target error rate e* (SLA)
        energy_cost_per_hour: Fixed cost per hour ($/hr)

    Returns:
        reward: Profit - λ·max(0, e - e*)
    """
    # Economic profit
    revenue = price_per_unit * mp_r
    error_cost = damage_cost * (err_rate * mp_r)
    profit = revenue - error_cost - energy_cost_per_hour

    # Constraint penalty (only when violating SLA)
    penalty = lam * max(0.0, err_rate - err_target)

    return profit - penalty
