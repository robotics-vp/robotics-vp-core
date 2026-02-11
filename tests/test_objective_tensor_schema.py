import numpy as np

from src.objectives.schema import ObjectiveTensorSchema


def test_objective_tensor_schema_shape_signature_stable():
    schema = ObjectiveTensorSchema()
    assert schema.shape_signature() == schema.shape_signature()
    assert len(schema.axes) == 4


def test_objective_tensor_schema_normalization_modes():
    schema = ObjectiveTensorSchema(
        normalization={
            "throughput": {"mode": "minmax", "min": 0.0, "max": 10.0},
            "error": {"mode": "clip", "min": 0.0, "max": 1.0},
            "safety": {"mode": "zscore", "mean": 0.5, "std": 0.25},
        }
    )
    values = np.asarray([10.0, 2.0, 0.75, 3.0], dtype=np.float32)
    out = schema.normalize_values(values)
    assert np.isclose(out[0], 1.0)
    assert np.isclose(out[1], 1.0)
    assert np.isclose(out[2], 1.0)
    assert np.isclose(out[3], 3.0)
