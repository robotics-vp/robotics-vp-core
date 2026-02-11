from src.objectives.frontier import ParetoFrontierTracker
from src.objectives.tensor import objective_tensor_from_axes


def test_frontier_tracker_promotes_non_dominated():
    tracker = ParetoFrontierTracker(
        maximize={"throughput": True, "error": False, "safety": True, "energy": False}
    )

    a = objective_tensor_from_axes({"throughput": 0.6, "error": 0.4, "safety": 0.6, "energy": 0.4})
    b = objective_tensor_from_axes({"throughput": 0.5, "error": 0.5, "safety": 0.5, "energy": 0.5})
    c = objective_tensor_from_axes({"throughput": 0.9, "error": 0.2, "safety": 0.8, "energy": 0.2})

    gain_a = tracker.add(a, task_id="t", env_id="e", profile_id="p")
    gain_b = tracker.marginal_gain(b, task_id="t", env_id="e", profile_id="p")
    gain_c = tracker.add(c, task_id="t", env_id="e", profile_id="p")

    assert gain_a > 0.0
    assert gain_b == 0.0
    assert gain_c > 0.0
    assert len(tracker.frontier(task_id="t", env_id="e", profile_id="p")) >= 1
