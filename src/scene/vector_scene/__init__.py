"""
Vector scene graph representation (Scenario Dreamer-style).

This package provides a vectorized scene format that generalizes to robotics domains
(factory, warehouse, home, etc.) with:
- SceneNode: Analogous to lane segments (corridors, rooms, aisles, etc.)
- SceneEdge: Relationships between nodes (predecessor, adjacent, door connect, etc.)
- SceneObject: Dynamic and static objects (humans, robots, pallets, machines, etc.)
- SceneGraph: Container for nodes, edges, and objects with utility methods
"""

from src.scene.vector_scene.graph import (
    NodeType,
    EdgeType,
    ObjectClass,
    SceneNode,
    SceneEdge,
    SceneObject,
    SceneGraph,
)
from src.scene.vector_scene.encoding import (
    ordered_scene_tensors,
    sinusoidal_positional_encoding,
    SceneGraphEncoder,
)
from src.scene.vector_scene.autoencoder import (
    SceneGraphLatent,
    SceneGraphDecoder,
    SceneGraphVAE,
)

__all__ = [
    "NodeType",
    "EdgeType",
    "ObjectClass",
    "SceneNode",
    "SceneEdge",
    "SceneObject",
    "SceneGraph",
    "ordered_scene_tensors",
    "sinusoidal_positional_encoding",
    "SceneGraphEncoder",
    "SceneGraphLatent",
    "SceneGraphDecoder",
    "SceneGraphVAE",
]
