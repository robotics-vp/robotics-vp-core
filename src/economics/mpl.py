def mpl(units_completed: float, time_hours: float) -> float:
    return 0.0 if time_hours <= 0 else units_completed / time_hours
