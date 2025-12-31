"""
Scene graph data structures for vectorized scene representation.

This module provides Scenario Dreamer-style scene graph representations
generalized to robotics domains (factory, warehouse, home, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
except ImportError:
    torch = None  # type: ignore


class NodeType(Enum):
    """Types of scene nodes (analogous to lane types in driving)."""
    CORRIDOR = auto()
    ROOM = auto()
    AISLE = auto()
    DOORWAY = auto()
    STAIR = auto()
    ELEVATOR = auto()
    LOADING_DOCK = auto()
    WORKSTATION = auto()
    STORAGE_ZONE = auto()
    KITCHEN_ZONE = auto()
    BATHROOM_ZONE = auto()
    OUTDOOR = auto()
    UNKNOWN = auto()


class EdgeType(Enum):
    """Types of edges connecting scene nodes."""
    PREDECESSOR = auto()
    SUCCESSOR = auto()
    LEFT = auto()
    RIGHT = auto()
    ADJACENT = auto()
    DOOR_CONNECT = auto()
    STAIR_CONNECT = auto()
    ELEVATOR_CONNECT = auto()
    SAME_LEVEL = auto()
    LEVEL_UP = auto()
    LEVEL_DOWN = auto()


class ObjectClass(Enum):
    """Classes of objects in the scene."""
    HUMAN = auto()
    ROBOT = auto()
    PALLET = auto()
    SHELF = auto()
    SINK = auto()
    MACHINE = auto()
    TABLE = auto()
    CHAIR = auto()
    FORKLIFT = auto()
    CART = auto()
    BOX = auto()
    DOOR = auto()
    WALL = auto()
    OBSTACLE = auto()
    VEHICLE = auto()
    CONVEYOR = auto()
    UNKNOWN = auto()


# Default widths for node types (meters)
DEFAULT_NODE_WIDTHS: Dict[NodeType, float] = {
    NodeType.CORRIDOR: 2.0,
    NodeType.ROOM: 4.0,
    NodeType.AISLE: 3.0,
    NodeType.DOORWAY: 1.0,
    NodeType.STAIR: 1.5,
    NodeType.ELEVATOR: 2.0,
    NodeType.LOADING_DOCK: 5.0,
    NodeType.WORKSTATION: 2.5,
    NodeType.STORAGE_ZONE: 4.0,
    NodeType.KITCHEN_ZONE: 3.0,
    NodeType.BATHROOM_ZONE: 2.0,
    NodeType.OUTDOOR: 5.0,
    NodeType.UNKNOWN: 2.0,
}


@dataclass
class SceneNode:
    """
    A node in the scene graph, analogous to a lane segment in driving.

    Represents structural elements like corridors, rooms, aisles, doorways.
    The polyline defines the centerline of the element.

    Attributes:
        id: Unique identifier for the node
        polyline: N x 2 or N x 3 array of points defining the centerline
        node_type: Type of scene element (corridor, room, etc.)
        attributes: Additional semantic tags (e.g., "kitchen sink zone")
        width: Width of the element in meters (defaults based on type)
        height: Height/ceiling height in meters (optional)
    """
    id: int
    polyline: np.ndarray  # (N, 2) or (N, 3) float32
    node_type: NodeType = NodeType.UNKNOWN
    attributes: Dict[str, Any] = field(default_factory=dict)
    width: Optional[float] = None
    height: Optional[float] = None

    def __post_init__(self) -> None:
        self.polyline = np.asarray(self.polyline, dtype=np.float32)
        if self.polyline.ndim == 1:
            self.polyline = self.polyline.reshape(-1, 2)
        if self.width is None:
            self.width = DEFAULT_NODE_WIDTHS.get(self.node_type, 2.0)

    @property
    def bounding_box(self) -> Tuple[float, float, float, float]:
        """Return (min_x, min_y, max_x, max_y) of the polyline."""
        if len(self.polyline) == 0:
            return (0.0, 0.0, 0.0, 0.0)
        min_x = float(self.polyline[:, 0].min())
        min_y = float(self.polyline[:, 1].min())
        max_x = float(self.polyline[:, 0].max())
        max_y = float(self.polyline[:, 1].max())
        return (min_x, min_y, max_x, max_y)

    @property
    def centroid(self) -> Tuple[float, float]:
        """Return the centroid of the polyline."""
        if len(self.polyline) == 0:
            return (0.0, 0.0)
        return (float(self.polyline[:, 0].mean()), float(self.polyline[:, 1].mean()))

    @property
    def length(self) -> float:
        """Return the total length of the polyline."""
        if len(self.polyline) < 2:
            return 0.0
        diffs = np.diff(self.polyline[:, :2], axis=0)
        return float(np.sum(np.linalg.norm(diffs, axis=1)))

    def to_feature_vector(self, max_polyline_points: int = 20) -> np.ndarray:
        """
        Convert node to a fixed-size feature vector.

        Returns:
            Feature vector of shape (D_node,) containing:
            - Node type one-hot (len(NodeType))
            - Bounding box (4)
            - Width, height (2)
            - Length (1)
            - Padded polyline points (max_polyline_points * 2 or 3)
        """
        # Node type one-hot
        num_types = len(NodeType)
        type_onehot = np.zeros(num_types, dtype=np.float32)
        type_onehot[self.node_type.value - 1] = 1.0

        # Bounding box
        bbox = np.array(self.bounding_box, dtype=np.float32)

        # Width and height
        width = self.width if self.width is not None else DEFAULT_NODE_WIDTHS.get(self.node_type, 2.0)
        height = self.height if self.height is not None else 3.0
        dims = np.array([width, height], dtype=np.float32)

        # Length
        length_arr = np.array([self.length], dtype=np.float32)

        # Padded polyline (use 2D for simplicity)
        polyline_2d = self.polyline[:, :2]
        if len(polyline_2d) >= max_polyline_points:
            # Sample uniformly
            indices = np.linspace(0, len(polyline_2d) - 1, max_polyline_points, dtype=int)
            padded = polyline_2d[indices].flatten()
        else:
            # Pad with zeros
            padded = np.zeros(max_polyline_points * 2, dtype=np.float32)
            padded[: len(polyline_2d) * 2] = polyline_2d.flatten()

        return np.concatenate([type_onehot, bbox, dims, length_arr, padded])


@dataclass
class SceneEdge:
    """
    An edge connecting two scene nodes.

    Attributes:
        src_id: Source node ID
        dst_id: Destination node ID
        edge_type: Type of connection
        attributes: Additional edge properties
    """
    src_id: int
    dst_id: int
    edge_type: EdgeType = EdgeType.ADJACENT
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_feature_vector(self) -> np.ndarray:
        """Convert edge to a feature vector (one-hot edge type)."""
        num_types = len(EdgeType)
        onehot = np.zeros(num_types, dtype=np.float32)
        onehot[self.edge_type.value - 1] = 1.0
        return onehot


@dataclass
class SceneObject:
    """
    A dynamic or static object in the scene.

    Attributes:
        id: Unique identifier
        class_id: Object class (human, robot, pallet, etc.)
        x, y, z: Position in world coordinates (z=0 for 2D)
        heading: Orientation in radians
        speed: Current speed (m/s), relevant for dynamic objects
        length, width, height: Object dimensions
        attributes: Additional properties (e.g., human role, robot type)
    """
    id: int
    class_id: ObjectClass = ObjectClass.UNKNOWN
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    heading: float = 0.0
    speed: float = 0.0
    length: float = 1.0
    width: float = 1.0
    height: float = 1.0
    attributes: Dict[str, Any] = field(default_factory=dict)

    @property
    def position(self) -> Tuple[float, float, float]:
        """Return (x, y, z) position."""
        return (self.x, self.y, self.z)

    @property
    def bounding_box_2d(self) -> Tuple[float, float, float, float]:
        """Return axis-aligned 2D bounding box (min_x, min_y, max_x, max_y)."""
        half_l = self.length / 2
        half_w = self.width / 2
        return (self.x - half_l, self.y - half_w, self.x + half_l, self.y + half_w)

    def to_feature_vector(self) -> np.ndarray:
        """
        Convert object to a feature vector.

        Returns:
            Feature vector containing:
            - Class one-hot (len(ObjectClass))
            - Position (x, y, z) (3)
            - Heading (sin, cos) (2)
            - Speed (1)
            - Dimensions (length, width, height) (3)
        """
        # Class one-hot
        num_classes = len(ObjectClass)
        class_onehot = np.zeros(num_classes, dtype=np.float32)
        class_onehot[self.class_id.value - 1] = 1.0

        # Position
        pos = np.array([self.x, self.y, self.z], dtype=np.float32)

        # Heading as sin/cos
        heading_feat = np.array([np.sin(self.heading), np.cos(self.heading)], dtype=np.float32)

        # Speed
        speed_arr = np.array([self.speed], dtype=np.float32)

        # Dimensions
        dims = np.array([self.length, self.width, self.height], dtype=np.float32)

        return np.concatenate([class_onehot, pos, heading_feat, speed_arr, dims])


@dataclass
class SceneGraph:
    """
    Container for a complete scene graph.

    Attributes:
        nodes: List of scene nodes (structural elements)
        edges: List of edges connecting nodes
        objects: List of objects in the scene
        metadata: Additional scene-level metadata
    """
    nodes: List[SceneNode] = field(default_factory=list)
    edges: List[SceneEdge] = field(default_factory=list)
    objects: List[SceneObject] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_adjacency(self, node_id: int) -> List[Tuple[int, EdgeType]]:
        """
        Get all adjacent nodes and edge types for a given node.

        Args:
            node_id: The node to query

        Returns:
            List of (neighbor_id, edge_type) tuples
        """
        result = []
        for edge in self.edges:
            if edge.src_id == node_id:
                result.append((edge.dst_id, edge.edge_type))
            elif edge.dst_id == node_id:
                result.append((edge.src_id, edge.edge_type))
        return result

    def get_node(self, node_id: int) -> Optional[SceneNode]:
        """Get a node by ID."""
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None

    def get_object(self, object_id: int) -> Optional[SceneObject]:
        """Get an object by ID."""
        for obj in self.objects:
            if obj.id == object_id:
                return obj
        return None

    def bounding_box(self) -> Tuple[float, float, float, float]:
        """
        Compute the overall bounding box of the scene.

        Returns:
            (min_x, min_y, max_x, max_y) tuple
        """
        if not self.nodes and not self.objects:
            return (0.0, 0.0, 0.0, 0.0)

        min_x, min_y = float("inf"), float("inf")
        max_x, max_y = float("-inf"), float("-inf")

        for node in self.nodes:
            bbox = node.bounding_box
            min_x = min(min_x, bbox[0])
            min_y = min(min_y, bbox[1])
            max_x = max(max_x, bbox[2])
            max_y = max(max_y, bbox[3])

        for obj in self.objects:
            bbox = obj.bounding_box_2d
            min_x = min(min_x, bbox[0])
            min_y = min(min_y, bbox[1])
            max_x = max(max_x, bbox[2])
            max_y = max(max_y, bbox[3])

        if min_x == float("inf"):
            return (0.0, 0.0, 0.0, 0.0)

        return (min_x, min_y, max_x, max_y)

    def as_tensors(self, device: Optional[str] = None, max_polyline_points: int = 20) -> Dict[str, Any]:
        """
        Convert scene graph to tensor format.

        Args:
            device: Target device for tensors (e.g., "cpu", "cuda")
            max_polyline_points: Maximum polyline points per node

        Returns:
            Dictionary with:
            - "node_features": (N_nodes, D_node) tensor
            - "edge_index": (2, N_edges) tensor
            - "edge_features": (N_edges, D_edge) tensor
            - "object_features": (N_objects, D_obj) tensor
            - "node_ids": list of node IDs
            - "object_ids": list of object IDs
        """
        if torch is None:
            raise ImportError("PyTorch is required for tensor conversion")

        # Node features
        node_features = []
        node_ids = []
        for node in self.nodes:
            node_features.append(node.to_feature_vector(max_polyline_points))
            node_ids.append(node.id)

        if node_features:
            node_tensor = torch.tensor(np.stack(node_features), dtype=torch.float32)
        else:
            # Empty placeholder
            d_node = len(NodeType) + 4 + 2 + 1 + max_polyline_points * 2
            node_tensor = torch.zeros((0, d_node), dtype=torch.float32)

        # Build node ID to index mapping
        node_id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

        # Edge index and features
        edge_src = []
        edge_dst = []
        edge_features = []
        for edge in self.edges:
            if edge.src_id in node_id_to_idx and edge.dst_id in node_id_to_idx:
                edge_src.append(node_id_to_idx[edge.src_id])
                edge_dst.append(node_id_to_idx[edge.dst_id])
                edge_features.append(edge.to_feature_vector())

        if edge_src:
            edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
            edge_tensor = torch.tensor(np.stack(edge_features), dtype=torch.float32)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_tensor = torch.zeros((0, len(EdgeType)), dtype=torch.float32)

        # Object features
        object_features = []
        object_ids = []
        for obj in self.objects:
            object_features.append(obj.to_feature_vector())
            object_ids.append(obj.id)

        if object_features:
            object_tensor = torch.tensor(np.stack(object_features), dtype=torch.float32)
        else:
            d_obj = len(ObjectClass) + 3 + 2 + 1 + 3
            object_tensor = torch.zeros((0, d_obj), dtype=torch.float32)

        result = {
            "node_features": node_tensor,
            "edge_index": edge_index,
            "edge_features": edge_tensor,
            "object_features": object_tensor,
            "node_ids": node_ids,
            "object_ids": object_ids,
        }

        if device is not None:
            result["node_features"] = result["node_features"].to(device)
            result["edge_index"] = result["edge_index"].to(device)
            result["edge_features"] = result["edge_features"].to(device)
            result["object_features"] = result["object_features"].to(device)

        return result

    def filter_objects_by_class(self, class_id: ObjectClass) -> List[SceneObject]:
        """Get all objects of a specific class."""
        return [obj for obj in self.objects if obj.class_id == class_id]

    def get_dynamic_objects(self) -> List[SceneObject]:
        """Get all objects with non-zero speed (dynamic agents)."""
        return [obj for obj in self.objects if obj.speed > 0]

    def count_objects_by_class(self) -> Dict[ObjectClass, int]:
        """Count objects by class."""
        counts: Dict[ObjectClass, int] = {}
        for obj in self.objects:
            counts[obj.class_id] = counts.get(obj.class_id, 0) + 1
        return counts

    @classmethod
    def create_simple_warehouse(cls, num_aisles: int = 5, aisle_length: float = 20.0) -> "SceneGraph":
        """
        Create a simple warehouse scene graph for testing.

        Args:
            num_aisles: Number of parallel aisles
            aisle_length: Length of each aisle in meters

        Returns:
            A SceneGraph representing a simple warehouse layout
        """
        nodes = []
        edges = []
        objects = []

        aisle_spacing = 4.0
        node_id = 0

        # Create aisles
        for i in range(num_aisles):
            y_pos = i * aisle_spacing
            polyline = np.array([
                [0.0, y_pos],
                [aisle_length, y_pos],
            ], dtype=np.float32)
            nodes.append(SceneNode(
                id=node_id,
                polyline=polyline,
                node_type=NodeType.AISLE,
                attributes={"aisle_index": i},
            ))
            node_id += 1

        # Create cross aisles at ends
        cross_y_start = 0.0
        cross_y_end = (num_aisles - 1) * aisle_spacing
        for x_pos in [0.0, aisle_length]:
            polyline = np.array([
                [x_pos, cross_y_start],
                [x_pos, cross_y_end],
            ], dtype=np.float32)
            nodes.append(SceneNode(
                id=node_id,
                polyline=polyline,
                node_type=NodeType.CORRIDOR,
                attributes={"cross_aisle": True},
            ))
            node_id += 1

        # Add edges connecting aisles to cross corridors
        for i in range(num_aisles):
            edges.append(SceneEdge(src_id=i, dst_id=num_aisles, edge_type=EdgeType.ADJACENT))
            edges.append(SceneEdge(src_id=i, dst_id=num_aisles + 1, edge_type=EdgeType.ADJACENT))

        # Add some shelves
        obj_id = 0
        for i in range(num_aisles - 1):
            y_pos = (i + 0.5) * aisle_spacing
            for x_pos in [3.0, 7.0, 11.0, 15.0]:
                objects.append(SceneObject(
                    id=obj_id,
                    class_id=ObjectClass.SHELF,
                    x=x_pos,
                    y=y_pos,
                    z=0.0,
                    heading=0.0,
                    speed=0.0,
                    length=2.0,
                    width=0.8,
                    height=2.5,
                ))
                obj_id += 1

        # Add a robot and some humans
        objects.append(SceneObject(
            id=obj_id,
            class_id=ObjectClass.ROBOT,
            x=1.0,
            y=0.0,
            z=0.0,
            heading=0.0,
            speed=0.0,
            length=0.6,
            width=0.6,
            height=1.5,
            attributes={"robot_type": "mobile_manipulator"},
        ))
        obj_id += 1

        for i in range(3):
            objects.append(SceneObject(
                id=obj_id,
                class_id=ObjectClass.HUMAN,
                x=5.0 + i * 5.0,
                y=(i % num_aisles) * aisle_spacing,
                z=0.0,
                heading=np.random.uniform(0, 2 * np.pi),
                speed=1.2,  # Walking speed
                length=0.5,
                width=0.5,
                height=1.75,
                attributes={"role": "worker"},
            ))
            obj_id += 1

        return cls(
            nodes=nodes,
            edges=edges,
            objects=objects,
            metadata={"scene_type": "warehouse", "num_aisles": num_aisles},
        )
