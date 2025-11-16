def value_per_datapoint(price_per_unit, delta_mpl_i, delta_mpl_bar):
    # only reward net-positive contribution versus baseline
    return max(0.0, price_per_unit * (delta_mpl_i - delta_mpl_bar))
