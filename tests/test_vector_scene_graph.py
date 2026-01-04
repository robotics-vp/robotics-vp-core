"""
Tests for vector scene graph representation and encoding.
"""

import numpy as np
import pytest

from src.scene.vector_scene.graph import (
    EdgeType,
    NodeType,
    ObjectClass,
    SceneEdge,
    SceneGraph,
    SceneNode,
    SceneObject,
)
from src.scene.vector_scene.encoding import (
    deterministic_node_order,
    deterministic_object_order,
    ordered_scene_tensors,
    sinusoidal_positional_encoding,
)


class TestSceneNode:
    def test_node_creation(self):
        polyline = np.array([[0, 0], [10, 0], [10, 10]], dtype=np.float32)
        node = SceneNode(
            id=0,
            polyline=polyline,
            node_type=NodeType.CORRIDOR,
        )
        assert node.id == 0
        assert node.node_type == NodeType.CORRIDOR
        assert len(node.polyline) == 3
        assert node.width is not None

    def test_node_bounding_box(self):
        polyline = np.array([[0, 0], [10, 5]], dtype=np.float32)
        node = SceneNode(id=0, polyline=polyline)
        bbox = node.bounding_box
        assert bbox == (0.0, 0.0, 10.0, 5.0)

    def test_node_centroid(self):
        polyline = np.array([[0, 0], [10, 10]], dtype=np.float32)
        node = SceneNode(id=0, polyline=polyline)
        centroid = node.centroid
        assert centroid == (5.0, 5.0)

    def test_node_length(self):
        polyline = np.array([[0, 0], [3, 4]], dtype=np.float32)
        node = SceneNode(id=0, polyline=polyline)
        assert abs(node.length - 5.0) < 1e-6

    def test_node_to_feature_vector(self):
        polyline = np.array([[0, 0], [10, 0]], dtype=np.float32)
        node = SceneNode(id=0, polyline=polyline, node_type=NodeType.AISLE)
        feat = node.to_feature_vector(max_polyline_points=10)
        assert feat.ndim == 1
        assert len(feat) > 0


class TestSceneObject:
    def test_object_creation(self):
        obj = SceneObject(
            id=0,
            class_id=ObjectClass.HUMAN,
            x=5.0,
            y=10.0,
            z=0.0,
            heading=np.pi / 4,
            speed=1.5,
        )
        assert obj.id == 0
        assert obj.class_id == ObjectClass.HUMAN
        assert obj.speed == 1.5

    def test_object_position(self):
        obj = SceneObject(id=0, x=1.0, y=2.0, z=3.0)
        assert obj.position == (1.0, 2.0, 3.0)

    def test_object_bounding_box(self):
        obj = SceneObject(id=0, x=5.0, y=5.0, length=2.0, width=2.0)
        bbox = obj.bounding_box_2d
        assert bbox == (4.0, 4.0, 6.0, 6.0)

    def test_object_to_feature_vector(self):
        obj = SceneObject(
            id=0,
            class_id=ObjectClass.ROBOT,
            x=1.0,
            y=2.0,
            heading=0.5,
        )
        feat = obj.to_feature_vector()
        assert feat.ndim == 1
        assert len(feat) == len(ObjectClass) + 3 + 2 + 1 + 3


class TestSceneGraph:
    def test_empty_graph(self):
        graph = SceneGraph()
        assert len(graph.nodes) == 0
        assert len(graph.edges) == 0
        assert len(graph.objects) == 0
        assert graph.bounding_box() == (0.0, 0.0, 0.0, 0.0)

    def test_simple_graph(self):
        nodes = [
            SceneNode(id=0, polyline=np.array([[0, 0], [10, 0]])),
            SceneNode(id=1, polyline=np.array([[10, 0], [20, 0]])),
        ]
        edges = [
            SceneEdge(src_id=0, dst_id=1, edge_type=EdgeType.SUCCESSOR),
        ]
        objects = [
            SceneObject(id=0, x=5.0, y=0.0),
        ]
        graph = SceneGraph(nodes=nodes, edges=edges, objects=objects)
        assert len(graph.nodes) == 2
        assert len(graph.edges) == 1
        assert len(graph.objects) == 1

    def test_get_adjacency(self):
        nodes = [
            SceneNode(id=0, polyline=np.array([[0, 0], [10, 0]])),
            SceneNode(id=1, polyline=np.array([[10, 0], [20, 0]])),
        ]
        edges = [
            SceneEdge(src_id=0, dst_id=1, edge_type=EdgeType.ADJACENT),
        ]
        graph = SceneGraph(nodes=nodes, edges=edges)
        adj = graph.get_adjacency(0)
        assert len(adj) == 1
        assert adj[0] == (1, EdgeType.ADJACENT)

    def test_bounding_box(self):
        nodes = [
            SceneNode(id=0, polyline=np.array([[-5, -5], [5, 5]])),
        ]
        objects = [
            SceneObject(id=0, x=10.0, y=10.0, length=2.0, width=2.0),
        ]
        graph = SceneGraph(nodes=nodes, objects=objects)
        bbox = graph.bounding_box()
        assert bbox[0] == -5.0
        assert bbox[1] == -5.0
        assert bbox[2] == 11.0  # 10 + 1 (half length)
        assert bbox[3] == 11.0

    def test_create_simple_warehouse(self):
        graph = SceneGraph.create_simple_warehouse(num_aisles=3, aisle_length=15.0)
        assert len(graph.nodes) >= 3
        assert len(graph.objects) > 0
        # Should have at least one robot
        robots = graph.filter_objects_by_class(ObjectClass.ROBOT)
        assert len(robots) >= 1

    def test_filter_objects_by_class(self):
        objects = [
            SceneObject(id=0, class_id=ObjectClass.HUMAN),
            SceneObject(id=1, class_id=ObjectClass.ROBOT),
            SceneObject(id=2, class_id=ObjectClass.HUMAN),
        ]
        graph = SceneGraph(objects=objects)
        humans = graph.filter_objects_by_class(ObjectClass.HUMAN)
        assert len(humans) == 2

    def test_count_objects_by_class(self):
        objects = [
            SceneObject(id=0, class_id=ObjectClass.HUMAN),
            SceneObject(id=1, class_id=ObjectClass.ROBOT),
            SceneObject(id=2, class_id=ObjectClass.HUMAN),
        ]
        graph = SceneGraph(objects=objects)
        counts = graph.count_objects_by_class()
        assert counts[ObjectClass.HUMAN] == 2
        assert counts[ObjectClass.ROBOT] == 1


class TestDeterministicOrdering:
    def test_node_ordering_consistent(self):
        nodes = [
            SceneNode(id=0, polyline=np.array([[10, 10], [20, 10]])),
            SceneNode(id=1, polyline=np.array([[0, 0], [5, 0]])),
            SceneNode(id=2, polyline=np.array([[5, 5], [10, 5]])),
        ]
        order1 = deterministic_node_order(nodes)
        order2 = deterministic_node_order(nodes)
        assert order1 == order2

    def test_node_ordering_sorts_by_bbox(self):
        nodes = [
            SceneNode(id=0, polyline=np.array([[10, 10], [20, 10]])),  # min_x=10
            SceneNode(id=1, polyline=np.array([[0, 0], [5, 0]])),  # min_x=0
            SceneNode(id=2, polyline=np.array([[5, 5], [10, 5]])),  # min_x=5
        ]
        order = deterministic_node_order(nodes)
        # Should be sorted by min_x: node 1 (x=0), node 2 (x=5), node 0 (x=10)
        assert order == [1, 2, 0]

    def test_object_ordering_consistent(self):
        objects = [
            SceneObject(id=0, x=10.0, y=10.0),
            SceneObject(id=1, x=0.0, y=0.0),
            SceneObject(id=2, x=5.0, y=5.0),
        ]
        order1 = deterministic_object_order(objects)
        order2 = deterministic_object_order(objects)
        assert order1 == order2

    def test_object_ordering_sorts_by_position(self):
        objects = [
            SceneObject(id=0, x=10.0, y=10.0),
            SceneObject(id=1, x=0.0, y=0.0),
            SceneObject(id=2, x=5.0, y=5.0),
        ]
        order = deterministic_object_order(objects)
        # Should be sorted by (x, y): obj 1 (0,0), obj 2 (5,5), obj 0 (10,10)
        assert order == [1, 2, 0]


class TestPositionalEncoding:
    def test_encoding_shape(self):
        positions = np.array([0, 1, 2, 3])
        d_model = 64
        pe = sinusoidal_positional_encoding(positions, d_model)
        assert pe.shape == (4, 64)

    def test_encoding_unique(self):
        positions = np.array([0, 1, 2])
        pe = sinusoidal_positional_encoding(positions, 32)
        # Each position should have a unique encoding
        assert not np.allclose(pe[0], pe[1])
        assert not np.allclose(pe[1], pe[2])

    def test_encoding_single_position(self):
        pe = sinusoidal_positional_encoding(np.array([5]), 16)
        assert pe.shape == (1, 16)


class TestOrderedSceneTensors:
    @pytest.fixture
    def simple_graph(self):
        nodes = [
            SceneNode(id=0, polyline=np.array([[0, 0], [10, 0]]), node_type=NodeType.CORRIDOR),
            SceneNode(id=1, polyline=np.array([[10, 0], [20, 0]]), node_type=NodeType.AISLE),
        ]
        edges = [
            SceneEdge(src_id=0, dst_id=1, edge_type=EdgeType.SUCCESSOR),
        ]
        objects = [
            SceneObject(id=0, class_id=ObjectClass.HUMAN, x=5.0, y=0.0),
            SceneObject(id=1, class_id=ObjectClass.ROBOT, x=15.0, y=0.0),
        ]
        return SceneGraph(nodes=nodes, edges=edges, objects=objects)

    def test_tensor_shapes(self, simple_graph):
        pytest.importorskip("torch")
        tensors = ordered_scene_tensors(simple_graph)
        assert "node_features" in tensors
        assert "node_positions" in tensors
        assert "object_features" in tensors
        assert "object_positions" in tensors
        assert "node_adj_matrix" in tensors
        assert "object_mask" in tensors
        assert tensors["node_features"].shape[0] == 2
        assert tensors["object_features"].shape[0] == 2
        assert tensors["node_adj_matrix"].shape[:2] == (2, 2)

    def test_empty_graph_tensors(self):
        pytest.importorskip("torch")
        graph = SceneGraph()
        tensors = ordered_scene_tensors(graph)
        assert tensors["node_features"].shape[0] == 0
        assert tensors["object_features"].shape[0] == 0

    def test_deterministic_output(self, simple_graph):
        pytest.importorskip("torch")
        import torch

        t1 = ordered_scene_tensors(simple_graph)
        t2 = ordered_scene_tensors(simple_graph)
        assert torch.allclose(t1["node_features"], t2["node_features"])
        assert torch.allclose(t1["object_features"], t2["object_features"])


class TestSceneGraphEncoder:
    @pytest.fixture
    def encoder(self):
        pytest.importorskip("torch")
        from src.scene.vector_scene.encoding import SceneGraphEncoder

        return SceneGraphEncoder(
            node_input_dim=64,
            obj_input_dim=32,
            hidden_dim=64,
            num_layers=2,
            num_heads=4,
        )

    def test_encoder_forward(self, encoder):
        pytest.importorskip("torch")
        import torch

        graph = SceneGraph.create_simple_warehouse(num_aisles=3, aisle_length=10.0)
        tensors = ordered_scene_tensors(graph, pos_dim=32)

        # Adjust input dims to match encoder expectations
        node_feat = tensors["node_features"]
        node_pos = tensors["node_positions"]
        obj_feat = tensors["object_features"]
        obj_pos = tensors["object_positions"]

        # Pad to expected dimensions
        node_input = torch.cat([node_feat, node_pos], dim=-1)
        obj_input = torch.cat([obj_feat, obj_pos], dim=-1)

        # Pad if needed
        if node_input.shape[-1] < 64:
            pad = torch.zeros(node_input.shape[0], 64 - node_input.shape[-1])
            node_input = torch.cat([node_input, pad], dim=-1)
        if obj_input.shape[-1] < 32:
            pad = torch.zeros(obj_input.shape[0], 32 - obj_input.shape[-1])
            obj_input = torch.cat([obj_input, pad], dim=-1)

        scene_tensors = {
            "node_features": node_input[:, :64],
            "node_positions": torch.zeros(node_input.shape[0], 32),
            "object_features": obj_input[:, :32],
            "object_positions": torch.zeros(obj_input.shape[0], 32),
        }

        output = encoder(scene_tensors)
        assert "node_latents" in output
        assert "object_latents" in output
        assert "scene_latent" in output
        assert output["scene_latent"].shape[-1] == 64

    def test_encoder_output_dim(self, encoder):
        assert encoder.get_output_dim() == 64
