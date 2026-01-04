from __future__ import annotations

import numpy as np
import pytest

from src.vision.map_first_supervision.semantics import parse_vla_semantic_evidence

pytestmark = pytest.mark.mapfirst


def test_vla_semantic_evidence_alignment() -> None:
    scene_track_ids = np.array(["track_a", "track_b"], dtype="U32")
    class_probs = np.array([
        [
            [0.0, 1.0],  # track_b
            [1.0, 0.0],  # track_a
        ]
    ], dtype=np.float32)
    payload = {
        "vla_semantic_evidence_v1/version": np.array(["v1"], dtype="U8"),
        "vla_semantic_evidence_v1/class_probs": class_probs,
        "vla_semantic_evidence_v1/track_ids": np.array(["track_b", "track_a"], dtype="U32"),
        "vla_semantic_evidence_v1/provenance_json": np.array(["{\"source\":\"test\"}"], dtype="U64"),
    }

    evidence = parse_vla_semantic_evidence(payload, scene_track_ids=scene_track_ids)
    assert evidence is not None
    assert evidence.class_probs is not None
    assert evidence.provenance == {"source": "test"}
    assert np.allclose(evidence.class_probs[0, 0], np.array([1.0, 0.0], dtype=np.float32))
    assert np.allclose(evidence.class_probs[0, 1], np.array([0.0, 1.0], dtype=np.float32))
