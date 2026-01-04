"""
Deterministic ordering, positional encodings, and SceneGraph encoder.

Implements Scenario Dreamer-style ordering and factorized attention for scene graphs.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore

from src.scene.vector_scene.graph import SceneGraph, SceneNode, SceneObject


def deterministic_node_order(nodes: List[SceneNode]) -> List[int]:
    """
    Deterministically order nodes by their bounding box coordinates.

    Sorting key: (min_x, min_y, max_x, max_y) of each node's polyline.
    This ensures consistent ordering regardless of input order.

    Args:
        nodes: List of SceneNode objects

    Returns:
        List of indices into the original list, in sorted order
    """
    if not nodes:
        return []

    def sort_key(idx: int) -> Tuple[float, float, float, float]:
        bbox = nodes[idx].bounding_box
        return bbox

    indices = list(range(len(nodes)))
    indices.sort(key=sort_key)
    return indices


def deterministic_object_order(objects: List[SceneObject]) -> List[int]:
    """
    Deterministically order objects by position and class.

    Sorting key: (x, y, class_id.value, id)
    This ensures consistent ordering regardless of input order.

    Args:
        objects: List of SceneObject objects

    Returns:
        List of indices into the original list, in sorted order
    """
    if not objects:
        return []

    def sort_key(idx: int) -> Tuple[float, float, int, int]:
        obj = objects[idx]
        return (obj.x, obj.y, obj.class_id.value, obj.id)

    indices = list(range(len(objects)))
    indices.sort(key=sort_key)
    return indices


def sinusoidal_positional_encoding(
    positions: np.ndarray,
    d_model: int,
    max_len: int = 10000,
) -> np.ndarray:
    """
    Compute sinusoidal positional encodings (Transformer-style).

    Args:
        positions: Array of position indices, shape (N,)
        d_model: Dimension of the encoding
        max_len: Maximum sequence length for scaling

    Returns:
        Positional encodings of shape (N, d_model)
    """
    positions = np.asarray(positions, dtype=np.float32)
    if positions.ndim == 0:
        positions = positions.reshape(1)

    # Create position encoding matrix
    pe = np.zeros((len(positions), d_model), dtype=np.float32)

    # Compute div term for each dimension
    div_term = np.exp(np.arange(0, d_model, 2, dtype=np.float32) * (-math.log(max_len) / d_model))

    # Apply sin to even indices, cos to odd indices
    pe[:, 0::2] = np.sin(positions[:, np.newaxis] * div_term)
    pe[:, 1::2] = np.cos(positions[:, np.newaxis] * div_term)

    return pe


def ordered_scene_tensors(
    graph: SceneGraph,
    node_dim: int = 64,
    object_dim: int = 64,
    pos_dim: int = 32,
    max_polyline_points: int = 20,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Convert a SceneGraph to ordered tensors with positional encodings.

    This function:
    1. Sorts nodes deterministically by bounding box
    2. Sorts objects deterministically by position and class
    3. Computes sinusoidal positional encodings for each
    4. Builds adjacency matrix with edge type encodings

    Args:
        graph: The SceneGraph to convert
        node_dim: Feature dimension for nodes (used for padding if needed)
        object_dim: Feature dimension for objects (used for padding if needed)
        pos_dim: Dimension for positional encodings
        max_polyline_points: Max polyline points for node features
        device: Target device for tensors

    Returns:
        Dictionary with:
        - "node_features": (N_nodes, D_node) - node feature vectors
        - "node_positions": (N_nodes, D_pos) - positional encodings for nodes
        - "object_features": (N_objects, D_obj) - object feature vectors
        - "object_positions": (N_objects, D_pos) - positional encodings for objects
        - "node_adj_matrix": (N_nodes, N_nodes, D_edge_type) - adjacency with edge types
        - "object_mask": (N_objects,) - mask for valid objects (all 1s for now)
        - "node_order": list of original node indices in sorted order
        - "object_order": list of original object indices in sorted order
    """
    if torch is None:
        raise ImportError("PyTorch is required for tensor conversion")

    # Get deterministic ordering
    node_order = deterministic_node_order(graph.nodes)
    object_order = deterministic_object_order(graph.objects)

    # Build node features in sorted order
    node_features = []
    for idx in node_order:
        feat = graph.nodes[idx].to_feature_vector(max_polyline_points)
        node_features.append(feat)

    if node_features:
        node_tensor = torch.tensor(np.stack(node_features), dtype=torch.float32)
    else:
        # Empty placeholder with expected dimension
        from src.scene.vector_scene.graph import NodeType
        d_node = len(NodeType) + 4 + 2 + 1 + max_polyline_points * 2
        node_tensor = torch.zeros((0, d_node), dtype=torch.float32)

    # Node positional encodings
    n_nodes = len(node_order)
    if n_nodes > 0:
        node_positions = torch.tensor(
            sinusoidal_positional_encoding(np.arange(n_nodes), pos_dim),
            dtype=torch.float32,
        )
    else:
        node_positions = torch.zeros((0, pos_dim), dtype=torch.float32)

    # Build object features in sorted order
    object_features = []
    for idx in object_order:
        feat = graph.objects[idx].to_feature_vector()
        object_features.append(feat)

    if object_features:
        object_tensor = torch.tensor(np.stack(object_features), dtype=torch.float32)
    else:
        from src.scene.vector_scene.graph import ObjectClass
        d_obj = len(ObjectClass) + 3 + 2 + 1 + 3
        object_tensor = torch.zeros((0, d_obj), dtype=torch.float32)

    # Object positional encodings
    n_objects = len(object_order)
    if n_objects > 0:
        object_positions = torch.tensor(
            sinusoidal_positional_encoding(np.arange(n_objects), pos_dim),
            dtype=torch.float32,
        )
    else:
        object_positions = torch.zeros((0, pos_dim), dtype=torch.float32)

    # Build adjacency matrix with edge types
    # Map original node IDs to sorted indices
    original_id_to_sorted_idx = {}
    for sorted_idx, orig_idx in enumerate(node_order):
        node_id = graph.nodes[orig_idx].id
        original_id_to_sorted_idx[node_id] = sorted_idx

    from src.scene.vector_scene.graph import EdgeType
    n_edge_types = len(EdgeType)

    if n_nodes > 0:
        adj_matrix = torch.zeros((n_nodes, n_nodes, n_edge_types), dtype=torch.float32)
        for edge in graph.edges:
            src_idx = original_id_to_sorted_idx.get(edge.src_id)
            dst_idx = original_id_to_sorted_idx.get(edge.dst_id)
            if src_idx is not None and dst_idx is not None:
                edge_type_idx = edge.edge_type.value - 1
                adj_matrix[src_idx, dst_idx, edge_type_idx] = 1.0
                # Make symmetric for undirected edges
                adj_matrix[dst_idx, src_idx, edge_type_idx] = 1.0
    else:
        adj_matrix = torch.zeros((0, 0, n_edge_types), dtype=torch.float32)

    # Object mask (all valid for now)
    object_mask = torch.ones(n_objects, dtype=torch.float32)

    result = {
        "node_features": node_tensor,
        "node_positions": node_positions,
        "object_features": object_tensor,
        "object_positions": object_positions,
        "node_adj_matrix": adj_matrix,
        "object_mask": object_mask,
        "node_order": node_order,
        "object_order": object_order,
    }

    if device is not None:
        for key in ["node_features", "node_positions", "object_features",
                    "object_positions", "node_adj_matrix", "object_mask"]:
            result[key] = result[key].to(device)

    return result


class FactorizedAttentionBlock(nn.Module):
    """
    Factorized attention block for scene graph encoding.

    Supports three attention patterns:
    - L2L: Node-to-node (lane-to-lane in driving analogy)
    - L2O: Node-to-object
    - O2O: Object-to-object
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        attention_type: str = "L2L",
    ):
        """
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
            attention_type: One of "L2L", "L2O", "O2O"
        """
        super().__init__()
        self.attention_type = attention_type
        self.embed_dim = embed_dim

        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            query: Query tensor (B, N_q, D)
            key: Key tensor (B, N_k, D)
            value: Value tensor (B, N_k, D)
            attn_mask: Optional attention mask

        Returns:
            Output tensor (B, N_q, D)
        """
        # Multi-head attention with residual
        attn_out, _ = self.multihead_attn(query, key, value, attn_mask=attn_mask)
        x = self.norm1(query + attn_out)

        # Feed-forward with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x


class SceneGraphEncoder(nn.Module):
    """
    Scenario Dreamer-style encoder for scene graphs with factorized attention.

    Uses three types of attention blocks:
    - L2L: Node-to-node attention for structural relationships
    - L2O: Node-to-object attention for spatial context
    - O2O: Object-to-object attention for agent interactions
    """

    def __init__(
        self,
        node_input_dim: int,
        obj_input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
        pos_dim: int = 32,
        pool_method: str = "mean",
    ):
        """
        Args:
            node_input_dim: Input dimension for node features
            obj_input_dim: Input dimension for object features
            hidden_dim: Hidden dimension for attention blocks
            num_layers: Number of factorized attention layers
            num_heads: Number of attention heads
            dropout: Dropout probability
            pos_dim: Dimension of positional encodings
            pool_method: Method for pooling to scene-level ("mean", "max", "concat")
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pool_method = pool_method

        # Input projections
        self.node_proj = nn.Linear(node_input_dim + pos_dim, hidden_dim)
        self.obj_proj = nn.Linear(obj_input_dim + pos_dim, hidden_dim)

        # Factorized attention layers
        self.l2l_layers = nn.ModuleList([
            FactorizedAttentionBlock(hidden_dim, num_heads, dropout, "L2L")
            for _ in range(num_layers)
        ])
        self.l2o_layers = nn.ModuleList([
            FactorizedAttentionBlock(hidden_dim, num_heads, dropout, "L2O")
            for _ in range(num_layers)
        ])
        self.o2o_layers = nn.ModuleList([
            FactorizedAttentionBlock(hidden_dim, num_heads, dropout, "O2O")
            for _ in range(num_layers)
        ])

        # Output projections
        if pool_method == "concat":
            self.scene_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        else:
            self.scene_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, scene_tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Encode a scene graph.

        Args:
            scene_tensors: Output from ordered_scene_tensors(), containing:
                - "node_features": (B, N_nodes, D_node) or (N_nodes, D_node)
                - "node_positions": (B, N_nodes, D_pos) or (N_nodes, D_pos)
                - "object_features": (B, N_objects, D_obj) or (N_objects, D_obj)
                - "object_positions": (B, N_objects, D_pos) or (N_objects, D_pos)
                - "node_adj_matrix": (B, N_nodes, N_nodes, D_edge) or (N_nodes, N_nodes, D_edge)

        Returns:
            Dictionary with:
            - "node_latents": (B, N_nodes, D_latent) or (N_nodes, D_latent)
            - "object_latents": (B, N_objects, D_latent) or (N_objects, D_latent)
            - "scene_latent": (B, D_latent) or (D_latent,)
        """
        node_feat = scene_tensors["node_features"]
        node_pos = scene_tensors["node_positions"]
        obj_feat = scene_tensors["object_features"]
        obj_pos = scene_tensors["object_positions"]

        # Handle unbatched input
        unbatched = node_feat.dim() == 2
        if unbatched:
            node_feat = node_feat.unsqueeze(0)
            node_pos = node_pos.unsqueeze(0)
            obj_feat = obj_feat.unsqueeze(0)
            obj_pos = obj_pos.unsqueeze(0)

        batch_size = node_feat.size(0)
        n_nodes = node_feat.size(1)
        n_objects = obj_feat.size(1)

        # Concatenate features with positional encodings and project
        if n_nodes > 0:
            node_x = torch.cat([node_feat, node_pos], dim=-1)
            node_x = self.node_proj(node_x)
        else:
            node_x = torch.zeros((batch_size, 0, self.hidden_dim), device=node_feat.device)

        if n_objects > 0:
            obj_x = torch.cat([obj_feat, obj_pos], dim=-1)
            obj_x = self.obj_proj(obj_x)
        else:
            obj_x = torch.zeros((batch_size, 0, self.hidden_dim), device=obj_feat.device)

        # Apply factorized attention layers
        for i in range(self.num_layers):
            # L2L: Node self-attention
            if n_nodes > 0:
                node_x = self.l2l_layers[i](node_x, node_x, node_x)

            # L2O: Cross-attention from nodes to objects (nodes attend to objects)
            if n_nodes > 0 and n_objects > 0:
                node_x = self.l2o_layers[i](node_x, obj_x, obj_x)

            # O2O: Object self-attention
            if n_objects > 0:
                obj_x = self.o2o_layers[i](obj_x, obj_x, obj_x)

        # Pool to scene-level representation
        if self.pool_method == "mean":
            if n_nodes > 0:
                node_pool = node_x.mean(dim=1)
            else:
                node_pool = torch.zeros((batch_size, self.hidden_dim), device=node_feat.device)

            if n_objects > 0:
                obj_pool = obj_x.mean(dim=1)
            else:
                obj_pool = torch.zeros((batch_size, self.hidden_dim), device=obj_feat.device)

            scene_latent = self.scene_proj((node_pool + obj_pool) / 2)

        elif self.pool_method == "max":
            if n_nodes > 0:
                node_pool = node_x.max(dim=1)[0]
            else:
                node_pool = torch.zeros((batch_size, self.hidden_dim), device=node_feat.device)

            if n_objects > 0:
                obj_pool = obj_x.max(dim=1)[0]
            else:
                obj_pool = torch.zeros((batch_size, self.hidden_dim), device=obj_feat.device)

            combined = torch.max(node_pool, obj_pool)
            scene_latent = self.scene_proj(combined)

        else:  # concat
            if n_nodes > 0:
                node_pool = node_x.mean(dim=1)
            else:
                node_pool = torch.zeros((batch_size, self.hidden_dim), device=node_feat.device)

            if n_objects > 0:
                obj_pool = obj_x.mean(dim=1)
            else:
                obj_pool = torch.zeros((batch_size, self.hidden_dim), device=obj_feat.device)

            combined = torch.cat([node_pool, obj_pool], dim=-1)
            scene_latent = self.scene_proj(combined)

        # Handle unbatched output
        if unbatched:
            node_x = node_x.squeeze(0)
            obj_x = obj_x.squeeze(0)
            scene_latent = scene_latent.squeeze(0)

        return {
            "node_latents": node_x,
            "object_latents": obj_x,
            "scene_latent": scene_latent,
        }

    def get_output_dim(self) -> int:
        """Return the output dimension for latent representations."""
        return self.hidden_dim
