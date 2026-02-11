import numpy as np

from src.objectives.serialization import (
    objective_tensor_from_npz_dict,
    objective_tensor_to_npz_dict,
)
from src.objectives.tensor import objective_tensor_from_axes


def test_objective_tensor_npz_roundtrip():
    tensor = objective_tensor_from_axes(
        {"throughput": 0.9, "error": 0.1, "safety": 0.8, "energy": 0.2},
        context={"episode_id": "ep1"},
    )
    payload = objective_tensor_to_npz_dict(tensor)
    restored = objective_tensor_from_npz_dict(payload)
    assert np.allclose(tensor.to_numpy(), restored.to_numpy())
    assert restored.context["episode_id"] == "ep1"
