import numpy as np

from src.constraints.constraint_set import ConstraintSet


def test_constraint_set_flagging_and_roundtrip(tmp_path):
    constraint_set = ConstraintSet.from_runtime(
        hard_constraints={
            "throughput": {"min": 4.0},
            "collision_rate": {"max": 0.1},
        },
        soft_constraints={"constraint_error_rate": {"max": 0.05}},
        safety_invariants={"respect_fragility": True},
        uncertainty={"runtime_uncertainty": 0.2},
    )

    observations = {
        "throughput": 3.0,
        "collision_rate": 0.2,
        "constraint_error_rate": 0.08,
        "respect_fragility": False,
    }
    flags = constraint_set.flag_observations(observations)
    assert [flag["constraint_id"] for flag in flags] == sorted(flag["constraint_id"] for flag in flags)
    assert any(flag["severity"] == "hard" for flag in flags)
    assert any(flag["severity"] == "soft" for flag in flags)

    payload = constraint_set.to_npz_dict()
    out = tmp_path / "constraint_set.npz"
    np.savez_compressed(out, **payload)
    restored = ConstraintSet.from_npz_dict(dict(np.load(out, allow_pickle=False)))
    assert restored.soft_bounds["constraint_error_rate"]["max"] == 0.05
    assert restored.safety_invariants["respect_fragility"] is True
