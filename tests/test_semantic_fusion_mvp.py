from __future__ import annotations

import numpy as np

from src.orchestrator.semantic_fusion import fuse_semantic_evidence_mvp


def test_fusion_shapes_and_normalization() -> None:
    vla = np.array(
        [
            [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]],
            [[0.6, 0.3, 0.1], [0.2, 0.3, 0.5]],
        ],
        dtype=np.float32,
    )
    map_sem = np.array(
        [
            [[0.2, 0.7, 0.1], [0.3, 0.3, 0.4]],
            [[0.2, 0.6, 0.2], [0.2, 0.5, 0.3]],
        ],
        dtype=np.float32,
    )
    result = fuse_semantic_evidence_mvp(
        vla_class_probs=vla,
        vla_confidence=np.ones((2, 2), dtype=np.float32),
        map_semantics=map_sem,
        map_stability=np.ones((2, 2), dtype=np.float32),
    )
    sums = np.sum(result.fused_class_probs, axis=-1)
    assert result.fused_class_probs.shape == (2, 2, 3) and np.allclose(sums, 1.0, atol=1e-4)


def test_fusion_prefers_map_when_residual_high() -> None:
    vla = np.array([[[0.9, 0.1]]], dtype=np.float32)
    map_sem = np.array([[[0.1, 0.9]]], dtype=np.float32)
    result = fuse_semantic_evidence_mvp(
        vla_class_probs=vla,
        vla_confidence=np.ones((1, 1), dtype=np.float32),
        map_semantics=map_sem,
        map_stability=np.ones((1, 1), dtype=np.float32),
        geom_residual=np.array([[5.0]], dtype=np.float32),
    )
    assert int(np.argmax(result.fused_class_probs[0, 0])) == 1


def test_fusion_missing_vla_fallbacks() -> None:
    map_sem = np.array([[[0.2, 0.8]]], dtype=np.float32)
    res_map = fuse_semantic_evidence_mvp(
        vla_class_probs=None,
        vla_confidence=None,
        map_semantics=map_sem,
        map_stability=np.ones((1, 1), dtype=np.float32),
    )
    res_unknown = fuse_semantic_evidence_mvp(
        vla_class_probs=None,
        vla_confidence=None,
        map_semantics=None,
        map_stability=None,
        geom_residual=np.zeros((1, 1), dtype=np.float32),
        num_classes=3,
    )
    uniform = np.full((1, 1, 3), 1.0 / 3.0, dtype=np.float32)
    assert (
        np.allclose(res_map.fused_class_probs, map_sem, atol=1e-4)
        and np.allclose(res_unknown.fused_class_probs, uniform, atol=1e-4)
        and np.max(res_unknown.fused_confidence) < 0.2
    )
