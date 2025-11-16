"""
Customer pricing with consumer surplus guarantee.

Ensures customers never pay more than human wage benchmark, even as robots improve.
This is an accounting/pricing layer only - does NOT affect RL behavior.
"""


def compute_customer_cost_per_hour(
    w_robot: float,
    w_human: float,
    rebate: float,
    base_fee: float = 0.0,
    floor_margin: float = 0.0,
) -> float:
    """
    Compute effective cost per hour charged to customer.

    Args:
        w_robot: Robot implied wage ($/hr) before rebates
        w_human: Indexed human wage benchmark ($/hr)
        rebate: Rebate credited to customer from spread allocation ($/hr)
        base_fee: Optional fixed fee component ($/hr)
        floor_margin: Optional minimum margin to preserve ($/hr)

    Returns:
        customer_cost: Effective cost ($/hr)

    Guarantee:
        customer_cost <= w_human (consumer surplus guarantee)

    Logic:
        1. Compute raw cost: w_robot - rebate + base_fee
        2. Enforce floor margin if specified
        3. Cap at w_human (never charge more than human alternative)
    """
    # Baseline robot charge before cap
    raw_cost = max(0.0, w_robot - rebate) + base_fee

    # Enforce minimum margin if desired
    if floor_margin > 0.0:
        raw_cost = max(raw_cost, floor_margin)

    # Consumer surplus guarantee: never charge more than human wage
    customer_cost = min(raw_cost, w_human)

    return customer_cost


def compute_consumer_surplus(w_human: float, customer_cost: float) -> float:
    """
    Compute consumer surplus per effective labor hour.

    Args:
        w_human: Human wage benchmark ($/hr)
        customer_cost: Actual cost charged to customer ($/hr)

    Returns:
        consumer_surplus: w_human - customer_cost ($/hr)

    Interpretation:
        This is the value customers capture by using robot vs human.
        Always >= 0 due to consumer surplus guarantee.
    """
    return max(0.0, w_human - customer_cost)


def validate_consumer_surplus_guarantee(customer_cost: float, w_human: float,
                                        tolerance: float = 1e-6) -> bool:
    """
    Validate that consumer surplus guarantee holds.

    Args:
        customer_cost: Charged cost ($/hr)
        w_human: Human wage benchmark ($/hr)
        tolerance: Numerical tolerance for floating point comparison

    Returns:
        True if customer_cost <= w_human + tolerance

    This should always return True if pricing logic is correct.
    """
    return customer_cost <= w_human + tolerance
