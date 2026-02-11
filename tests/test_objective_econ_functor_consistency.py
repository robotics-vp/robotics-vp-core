import numpy as np

from src.economics.functor import ObjectiveEconFunctor
from src.objectives.tensor import objective_tensor_from_axes


def test_objective_econ_functor_is_deterministic():
    tensor = objective_tensor_from_axes(
        {"throughput": 0.7, "error": 0.2, "safety": 0.8, "energy": 0.3}
    )
    functor = ObjectiveEconFunctor(base_price_per_unit=2.0)

    out1 = functor.map(tensor, constraint_flags=[], uncertainty=0.1)
    out2 = functor.map(tensor, constraint_flags=[], uncertainty=0.1)

    assert out1.schema.schema_id == out2.schema.schema_id
    assert np.allclose(out1.values, out2.values)
