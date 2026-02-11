import numpy as np

from src.provenance import (
    extract_stamped_metadata,
    stamp_objective_tensor_metadata,
    stamp_regal_decision_metadata,
)


def test_provenance_npz_roundtrip(tmp_path):
    payload = {
        "dummy": np.array([1.0], dtype=np.float32),
    }
    payload = stamp_objective_tensor_metadata(
        payload,
        objective_tensor_metadata={
            "schema_id": "objective_tensor_v1",
            "shape_signature": "abc123",
            "episode_id": "ep_1",
        },
    )
    payload = stamp_regal_decision_metadata(
        payload,
        regal_decision_metadata={
            "decision": "BLOCK",
            "reason_codes": ["early_scalarization"],
        },
    )

    out = tmp_path / "sample.npz"
    np.savez_compressed(out, **payload)

    loaded = dict(np.load(out, allow_pickle=False))
    restored = extract_stamped_metadata(loaded)

    assert restored["objective_tensor"]["schema_id"] == "objective_tensor_v1"
    assert restored["objective_tensor"]["episode_id"] == "ep_1"
    assert restored["regal_decision"]["decision"] == "BLOCK"
    assert restored["regal_decision"]["reason_codes"] == ["early_scalarization"]
