"""
Mechanistic spread allocation between customer and platform.

Split based on ΔMPL contributions (causal shares), not arbitrary percentages.

Key principle: If customer's data drove ΔMPL_cust of the total ΔMPL_total improvement,
they receive that same share of the wage spread vs human.
"""


def compute_contribution_shares(delta_mpl_cust: float,
                                delta_mpl_total: float,
                                eps: float = 1e-6) -> tuple[float, float]:
    """
    Compute customer vs platform contribution shares based on ΔMPL.

    Args:
        delta_mpl_cust: MPL improvement attributable to customer's data
        delta_mpl_total: Total MPL improvement this period
        eps: Small constant to avoid division by zero

    Returns:
        (s_cust, s_plat): Customer and platform shares, sum to 1.0
            s_cust = ΔMPL_cust / (ΔMPL_total + eps), clipped to [0,1]
            s_plat = 1 - s_cust
    """
    denom = delta_mpl_total + eps
    s_cust = 0.0 if denom <= 0.0 else delta_mpl_cust / denom
    s_cust = max(0.0, min(1.0, s_cust))
    s_plat = 1.0 - s_cust
    return s_cust, s_plat


def compute_spread_allocation(w_robot: float,
                              w_human: float,
                              hours: float,
                              delta_mpl_cust: float,
                              delta_mpl_total: float,
                              eps_parity: float = 0.05) -> dict:
    """
    Mechanistic allocation of spread vs human between customer and platform.

    Logic:
    - If parity <= 1 + eps_parity: no spread, all amounts = 0.
    - Otherwise:
        spread_value = (w_robot - w_human) * hours
        s_cust, s_plat from ΔMPL contributions
        rebate  = s_cust * spread_value (customer gets this)
        captured = s_plat * spread_value (platform retains this)

    Args:
        w_robot: Robot implied wage ($/hr)
        w_human: Human wage benchmark ($/hr)
        hours: Time period for allocation (hours)
        delta_mpl_cust: MPL improvement from customer data
        delta_mpl_total: Total MPL improvement
        eps_parity: Parity threshold (no spread below 1 + eps_parity)

    Returns:
        dict with keys:
            spread: Wage spread ($/hr) = w_robot - w_human
            spread_value: Total spread value ($) = spread * hours
            s_cust: Customer contribution share [0, 1]
            s_plat: Platform contribution share [0, 1]
            rebate: Amount allocated to customer ($)
            captured: Amount retained by platform ($)
    """
    if w_human <= 0.0:
        return {
            "spread": 0.0,
            "spread_value": 0.0,
            "s_cust": 0.0,
            "s_plat": 0.0,
            "rebate": 0.0,
            "captured": 0.0,
        }

    parity = w_robot / w_human
    if parity <= 1.0 + eps_parity:
        # Below parity threshold: no spread to allocate
        return {
            "spread": 0.0,
            "spread_value": 0.0,
            "s_cust": 0.0,
            "s_plat": 0.0,
            "rebate": 0.0,
            "captured": 0.0,
        }

    # Compute spread
    spread = max(0.0, w_robot - w_human)
    spread_value = spread * hours

    # Compute contribution shares
    s_cust, s_plat = compute_contribution_shares(delta_mpl_cust, delta_mpl_total)

    # Allocate spread value
    rebate = s_cust * spread_value
    captured = s_plat * spread_value

    return {
        "spread": spread,
        "spread_value": spread_value,
        "s_cust": s_cust,
        "s_plat": s_plat,
        "rebate": rebate,
        "captured": captured,
    }
