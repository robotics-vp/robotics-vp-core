"""Tests for semantic_state_encoder.py (Phase 4)."""
import unittest

import numpy as np

from src.world_model.semantic_state_encoder import (
    FLAT_EMBED_DIM,
    OBJECT_FEATURE_DIM,
    RELATION_FEATURE_DIM,
    META_NODE_FEATURE_DIM,
    encode_object,
    encode_relation,
    encode_meta_node,
    encode_wm_state_flat,
)
from src.world_model.semantic_world_model import (
    SemanticObjectState,
    SemanticRelationState,
    SemanticMetaNode,
    SemanticWorldModelState,
)


def _make_object(obj_id="drawer", category="container"):
    return SemanticObjectState(
        object_id=obj_id, label=obj_id, category=category,
        confidence=0.9, salience=0.7,
        affordances=["open", "close"], risk_tags=["fragility"],
        state_tags=["occluding"],
    )


def _make_relation():
    return SemanticRelationState(
        relation_id="r1", subject_id="drawer", relation_type="spatial_near",
        object_id="vase", confidence=0.8,
    )


def _make_meta_node():
    return SemanticMetaNode(
        node_id="mn1", node_type="risk_alert", priority="high",
        score=0.85, rationale="fragile object nearby",
        target_refs=["vase"], suggested_actions=["reduce_speed"],
    )


def _make_wm_state():
    return SemanticWorldModelState(
        world_model_id="wm1", episode_id="ep1", task_id="open_drawer",
        objective_preset="safety_first",
        semantic_tags=["drawer", "vase", "fragile"],
        objects=[_make_object("drawer"), _make_object("vase", "fragile_object")],
        relations=[_make_relation()],
        meta_nodes=[_make_meta_node()],
        capability_scores={"grasp": 0.8, "release": 0.9},
    )


class TestElementEncoders(unittest.TestCase):

    def test_object_encoding_shape(self):
        obj = _make_object()
        enc = encode_object(obj)
        self.assertEqual(enc.shape, (OBJECT_FEATURE_DIM,))

    def test_object_confidence_encoded(self):
        obj = _make_object()
        enc = encode_object(obj)
        # confidence should be first scalar
        self.assertAlmostEqual(enc[0], 0.9)

    def test_relation_encoding_shape(self):
        rel = _make_relation()
        enc = encode_relation(rel)
        self.assertEqual(enc.shape, (RELATION_FEATURE_DIM,))

    def test_meta_node_encoding_shape(self):
        mn = _make_meta_node()
        enc = encode_meta_node(mn)
        self.assertEqual(enc.shape, (META_NODE_FEATURE_DIM,))


class TestFlatEncoding(unittest.TestCase):

    def test_flat_embedding_shape(self):
        state = _make_wm_state()
        emb = encode_wm_state_flat(state)
        self.assertEqual(emb.shape, (FLAT_EMBED_DIM,))

    def test_empty_state_embedding(self):
        state = SemanticWorldModelState(
            world_model_id="empty", episode_id="ep", task_id="t",
            objective_preset="default", semantic_tags=[],
        )
        emb = encode_wm_state_flat(state)
        self.assertEqual(emb.shape, (FLAT_EMBED_DIM,))

    def test_different_states_different_embeddings(self):
        s1 = _make_wm_state()
        s2 = SemanticWorldModelState(
            world_model_id="wm2", episode_id="ep2", task_id="pick_object",
            objective_preset="speed", semantic_tags=["workpiece"],
            objects=[_make_object("workpiece", "manipulated_object")],
        )
        e1 = encode_wm_state_flat(s1)
        e2 = encode_wm_state_flat(s2)
        self.assertFalse(np.allclose(e1, e2))


class TestEmbedMethod(unittest.TestCase):

    def test_embed_fallback(self):
        """embed() without encoder should use flat encoding."""
        state = _make_wm_state()
        emb = state.embed()
        self.assertEqual(emb.shape, (FLAT_EMBED_DIM,))

    def test_embed_with_none_encoder(self):
        state = _make_wm_state()
        emb = state.embed(encoder=None)
        self.assertEqual(emb.shape, (FLAT_EMBED_DIM,))


if __name__ == "__main__":
    unittest.main()
