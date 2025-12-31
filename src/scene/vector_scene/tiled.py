"""
Tiled scene partitioning for large/infinite environments.

Implements Scenario Dreamer-style tiling with support for:
- Scene partitioning into fixed-size tiles
- Tile-based scene graph extraction
- Conditional generation of next tiles (scaffolding)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.scene.vector_scene.graph import SceneEdge, SceneGraph, SceneNode, SceneObject


@dataclass
class SceneTileSpec:
    """
    Specification for a scene tile.

    Attributes:
        origin: (x, y) world coordinates of tile origin (bottom-left)
        size: (width, height) of tile in world units
        index: Sequential index along the route/path
        level: Vertical level (for multi-floor environments)
        metadata: Additional tile metadata
    """
    origin: Tuple[float, float]
    size: Tuple[float, float]
    index: int
    level: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """Return (min_x, min_y, max_x, max_y) of the tile."""
        return (
            self.origin[0],
            self.origin[1],
            self.origin[0] + self.size[0],
            self.origin[1] + self.size[1],
        )

    def contains_point(self, x: float, y: float) -> bool:
        """Check if a point is within this tile."""
        min_x, min_y, max_x, max_y = self.bounds
        return min_x <= x < max_x and min_y <= y < max_y

    def overlaps(self, other: "SceneTileSpec") -> bool:
        """Check if this tile overlaps with another."""
        ax1, ay1, ax2, ay2 = self.bounds
        bx1, by1, bx2, by2 = other.bounds
        return not (ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1)


@dataclass
class TileLatent:
    """
    Latent representation of a tile for conditional generation.

    Attributes:
        tile_spec: The tile specification
        boundary_nodes: Node features at tile boundaries
        boundary_objects: Object features at tile boundaries
        scene_context: Global scene context embedding
        metadata: Additional metadata
    """
    tile_spec: SceneTileSpec
    boundary_nodes: np.ndarray  # (N, D) boundary node embeddings
    boundary_objects: np.ndarray  # (M, D) boundary object embeddings
    scene_context: np.ndarray  # (D,) global context
    metadata: Dict[str, Any] = field(default_factory=dict)


def partition_scene_graph(
    graph: SceneGraph,
    tile_size: float = 64.0,
    overlap: float = 0.0,
) -> List[SceneTileSpec]:
    """
    Partition a scene graph into fixed-size tiles.

    Args:
        graph: Input SceneGraph
        tile_size: Size of each tile (square) in world units
        overlap: Overlap between adjacent tiles in world units

    Returns:
        List of SceneTileSpec for each tile covering the scene
    """
    bbox = graph.bounding_box()
    min_x, min_y, max_x, max_y = bbox

    # Add small margin
    margin = 1.0
    min_x -= margin
    min_y -= margin
    max_x += margin
    max_y += margin

    # Compute tile grid
    step = tile_size - overlap
    if step <= 0:
        step = tile_size

    tiles = []
    tile_index = 0

    y = min_y
    row = 0
    while y < max_y:
        x = min_x
        col = 0
        while x < max_x:
            tile = SceneTileSpec(
                origin=(x, y),
                size=(tile_size, tile_size),
                index=tile_index,
                level=0,
                metadata={"row": row, "col": col},
            )
            tiles.append(tile)
            tile_index += 1
            x += step
            col += 1
        y += step
        row += 1

    return tiles


def partition_along_route(
    graph: SceneGraph,
    route_polyline: np.ndarray,
    tile_size: float = 64.0,
    tile_width: float = 64.0,
) -> List[SceneTileSpec]:
    """
    Partition a scene into tiles along a route (path).

    This creates tiles centered on the route, useful for linear
    environments like corridors, roads, or production lines.

    Args:
        graph: Input SceneGraph
        route_polyline: (N, 2) array of route waypoints
        tile_size: Length of each tile along the route
        tile_width: Width of tiles perpendicular to route

    Returns:
        List of SceneTileSpec along the route
    """
    if len(route_polyline) < 2:
        # Fall back to grid partitioning
        return partition_scene_graph(graph, tile_size)

    tiles = []
    tile_index = 0

    # Walk along the polyline
    accumulated_dist = 0.0
    current_tile_start = 0.0

    for i in range(len(route_polyline) - 1):
        p0 = route_polyline[i]
        p1 = route_polyline[i + 1]
        segment_length = np.linalg.norm(p1 - p0)

        if segment_length < 1e-6:
            continue

        segment_dir = (p1 - p0) / segment_length

        while accumulated_dist + segment_length >= current_tile_start + tile_size:
            # Create tile at this position
            progress = (current_tile_start + tile_size / 2 - accumulated_dist) / segment_length
            progress = np.clip(progress, 0, 1)
            tile_center = p0 + progress * (p1 - p0)

            # Compute tile origin (shifted perpendicular and back)
            perp = np.array([-segment_dir[1], segment_dir[0]])
            origin = tile_center - tile_size / 2 * segment_dir - tile_width / 2 * perp

            tile = SceneTileSpec(
                origin=(float(origin[0]), float(origin[1])),
                size=(tile_size, tile_width),
                index=tile_index,
                level=0,
                metadata={
                    "route_distance": current_tile_start,
                    "segment_index": i,
                },
            )
            tiles.append(tile)
            tile_index += 1
            current_tile_start += tile_size

        accumulated_dist += segment_length

    # Handle remaining distance
    if accumulated_dist > current_tile_start:
        # Add final tile
        p_final = route_polyline[-1]
        origin = (
            float(p_final[0] - tile_size / 2),
            float(p_final[1] - tile_width / 2),
        )
        tile = SceneTileSpec(
            origin=origin,
            size=(tile_size, tile_width),
            index=tile_index,
            level=0,
            metadata={"route_distance": current_tile_start, "final": True},
        )
        tiles.append(tile)

    return tiles


def extract_tile_subgraph(
    graph: SceneGraph,
    tile: SceneTileSpec,
    include_boundary: bool = True,
    boundary_margin: float = 2.0,
) -> SceneGraph:
    """
    Extract the subgraph contained within a tile.

    Args:
        graph: Full SceneGraph
        tile: Tile specification
        include_boundary: Include nodes/objects at tile boundary
        boundary_margin: Margin for boundary inclusion

    Returns:
        SceneGraph containing only elements within the tile
    """
    min_x, min_y, max_x, max_y = tile.bounds

    if include_boundary:
        min_x -= boundary_margin
        min_y -= boundary_margin
        max_x += boundary_margin
        max_y += boundary_margin

    # Filter nodes
    included_nodes = []
    node_id_map: Dict[int, int] = {}  # old_id -> new_id

    for node in graph.nodes:
        centroid = node.centroid
        if min_x <= centroid[0] <= max_x and min_y <= centroid[1] <= max_y:
            new_id = len(included_nodes)
            node_id_map[node.id] = new_id

            # Clip polyline to tile bounds (approximately)
            clipped_polyline = node.polyline.copy()

            included_nodes.append(SceneNode(
                id=new_id,
                polyline=clipped_polyline,
                node_type=node.node_type,
                attributes=node.attributes.copy(),
                width=node.width,
                height=node.height,
            ))

    # Filter edges (only include if both endpoints are in tile)
    included_edges = []
    for edge in graph.edges:
        if edge.src_id in node_id_map and edge.dst_id in node_id_map:
            included_edges.append(SceneEdge(
                src_id=node_id_map[edge.src_id],
                dst_id=node_id_map[edge.dst_id],
                edge_type=edge.edge_type,
                attributes=edge.attributes.copy(),
            ))

    # Filter objects
    included_objects = []
    for obj in graph.objects:
        if min_x <= obj.x <= max_x and min_y <= obj.y <= max_y:
            new_id = len(included_objects)
            included_objects.append(SceneObject(
                id=new_id,
                class_id=obj.class_id,
                x=obj.x,
                y=obj.y,
                z=obj.z,
                heading=obj.heading,
                speed=obj.speed,
                length=obj.length,
                width=obj.width,
                height=obj.height,
                attributes=obj.attributes.copy(),
            ))

    return SceneGraph(
        nodes=included_nodes,
        edges=included_edges,
        objects=included_objects,
        metadata={
            "source_tile": tile.index,
            "tile_bounds": tile.bounds,
            **graph.metadata,
        },
    )


def get_boundary_elements(
    graph: SceneGraph,
    tile: SceneTileSpec,
    margin: float = 2.0,
) -> Tuple[List[SceneNode], List[SceneObject]]:
    """
    Get nodes and objects at the tile boundary.

    These can be used for conditioning when generating adjacent tiles.

    Args:
        graph: SceneGraph (typically extracted for this tile)
        tile: Tile specification
        margin: Distance from boundary to consider

    Returns:
        Tuple of (boundary_nodes, boundary_objects)
    """
    min_x, min_y, max_x, max_y = tile.bounds

    boundary_nodes = []
    for node in graph.nodes:
        centroid = node.centroid
        x, y = centroid

        # Check if near any boundary
        near_boundary = (
            x < min_x + margin or
            x > max_x - margin or
            y < min_y + margin or
            y > max_y - margin
        )
        if near_boundary:
            boundary_nodes.append(node)

    boundary_objects = []
    for obj in graph.objects:
        near_boundary = (
            obj.x < min_x + margin or
            obj.x > max_x - margin or
            obj.y < min_y + margin or
            obj.y > max_y - margin
        )
        if near_boundary:
            boundary_objects.append(obj)

    return boundary_nodes, boundary_objects


def generate_next_tile(
    prev_tile_latent: TileLatent,
    config: Dict[str, Any],
) -> SceneGraph:
    """
    Generate the next tile conditioned on the previous tile.

    This is a stub for conditional scene generation. In production,
    this would use a trained diffusion model over scene graphs.

    Args:
        prev_tile_latent: Latent from previous tile
        config: Generation configuration (topology type, density, etc.)

    Returns:
        Generated SceneGraph for the next tile

    TODO:
        - Implement diffusion model for scene graph generation
        - Condition on boundary nodes/objects from previous tile
        - Respect topology and density constraints
    """
    # Stub implementation: create simple connecting corridor
    tile_spec = prev_tile_latent.tile_spec
    next_origin = (
        tile_spec.origin[0] + tile_spec.size[0],
        tile_spec.origin[1],
    )
    next_size = tile_spec.size

    # Create simple corridor node
    polyline = np.array([
        [next_origin[0], next_origin[1] + next_size[1] / 2],
        [next_origin[0] + next_size[0], next_origin[1] + next_size[1] / 2],
    ], dtype=np.float32)

    from src.scene.vector_scene.graph import NodeType

    nodes = [
        SceneNode(
            id=0,
            polyline=polyline,
            node_type=config.get("node_type", NodeType.CORRIDOR),
            width=config.get("corridor_width", 3.0),
        ),
    ]

    return SceneGraph(
        nodes=nodes,
        edges=[],
        objects=[],
        metadata={
            "generated": True,
            "conditioned_on_tile": tile_spec.index,
            "config": config,
        },
    )


def stitch_tiles(tiles: List[Tuple[SceneTileSpec, SceneGraph]]) -> SceneGraph:
    """
    Stitch multiple tile subgraphs into a single scene graph.

    Handles node/object ID remapping and edge reconnection.

    Args:
        tiles: List of (tile_spec, subgraph) tuples

    Returns:
        Combined SceneGraph
    """
    all_nodes = []
    all_edges = []
    all_objects = []

    node_id_offset = 0
    object_id_offset = 0

    for tile_spec, subgraph in tiles:
        # Remap node IDs
        node_id_map = {}
        for node in subgraph.nodes:
            new_id = node_id_offset + node.id
            node_id_map[node.id] = new_id
            all_nodes.append(SceneNode(
                id=new_id,
                polyline=node.polyline,
                node_type=node.node_type,
                attributes={
                    **node.attributes,
                    "source_tile": tile_spec.index,
                },
                width=node.width,
                height=node.height,
            ))
        node_id_offset += len(subgraph.nodes)

        # Remap edges
        for edge in subgraph.edges:
            all_edges.append(SceneEdge(
                src_id=node_id_map[edge.src_id],
                dst_id=node_id_map[edge.dst_id],
                edge_type=edge.edge_type,
                attributes=edge.attributes,
            ))

        # Remap object IDs
        for obj in subgraph.objects:
            all_objects.append(SceneObject(
                id=object_id_offset + obj.id,
                class_id=obj.class_id,
                x=obj.x,
                y=obj.y,
                z=obj.z,
                heading=obj.heading,
                speed=obj.speed,
                length=obj.length,
                width=obj.width,
                height=obj.height,
                attributes={
                    **obj.attributes,
                    "source_tile": tile_spec.index,
                },
            ))
        object_id_offset += len(subgraph.objects)

    return SceneGraph(
        nodes=all_nodes,
        edges=all_edges,
        objects=all_objects,
        metadata={
            "stitched": True,
            "num_tiles": len(tiles),
        },
    )
