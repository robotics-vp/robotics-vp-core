def implied_robot_wage(price_per_unit: float, mp_r: float, error_rate: float, damage_cost: float) -> float:
    return price_per_unit * mp_r - damage_cost * (error_rate * mp_r)
