from src.objectives.compiler import ObjectiveCompiler
from src.objectives.profile import ObjectiveProfile
from src.objectives.tensor import objective_tensor_from_axes


def _sample_tensor():
    return objective_tensor_from_axes(
        {
            "throughput": 0.8,
            "error": 0.2,
            "safety": 0.9,
            "energy": 0.1,
        }
    )


def test_weighted_sum_scalarizer():
    tensor = _sample_tensor()
    profile = ObjectiveProfile(
        scalarizer="weighted_sum",
        weights={"throughput": 2.0, "error": 1.0, "safety": 1.0, "energy": 1.0},
        maximize={"throughput": True, "error": False, "safety": True, "energy": False},
    )
    score = ObjectiveCompiler(profile).scalarize(tensor)
    assert score > 0.0


def test_constrained_scalarizer_penalizes_violations():
    tensor = _sample_tensor()
    profile = ObjectiveProfile(
        scalarizer="constrained",
        weights={"throughput": 1.0, "error": 1.0, "safety": 1.0, "energy": 1.0},
        maximize={"throughput": True, "error": False, "safety": True, "energy": False},
        constraints={"error": {"max": 0.1}},
        penalty_weight=100.0,
    )
    compiler = ObjectiveCompiler(profile)
    score = compiler.scalarize(tensor)
    flags = compiler.constraint_flags(tensor)
    assert flags
    assert score < 0.0


def test_lexicographic_scalarizer_deterministic():
    tensor = _sample_tensor()
    profile = ObjectiveProfile(
        scalarizer="lexicographic",
        maximize={"throughput": True, "error": False, "safety": True, "energy": False},
        lexicographic_order=["safety", "throughput", "error", "energy"],
    )
    compiler = ObjectiveCompiler(profile)
    score1 = compiler.scalarize(tensor)
    score2 = compiler.scalarize(tensor)
    assert score1 == score2
