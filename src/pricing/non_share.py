def non_sharing_premium(unit_price, expected_delta_mpl, hours_horizon, scale_mult=1.0, kappa=0.8):
    return unit_price * kappa * expected_delta_mpl * hours_horizon * scale_mult
