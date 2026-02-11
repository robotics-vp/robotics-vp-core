import numpy as np

from src.constraints.constraint_set import ConstraintSet


def test_constraint_set_roundtrip_npz(tmp_path):
    cs = ConstraintSet.from_artifacts(
        semantic_evidence={"semantic_tags": ["fragile"], "fragile": True, "safety_critical": True},
        map_first_summary={"map_first_quality_score": 0.7},
        fusion_metrics={"semantic_fusion_confidence_mean": 0.8, "semantic_disagreement_vla_vs_map": 0.1},
    )
    payload = cs.to_npz_dict()
    out = tmp_path / "constraint_set.npz"
    np.savez_compressed(out, **payload)
    loaded = dict(np.load(out, allow_pickle=False))
    restored = ConstraintSet.from_npz_dict(loaded)
    assert restored.hard_bounds["map_first_quality_score"]["min"] > 0.0
    assert restored.geometry_priors["depth_consistency"] >= 0.0
