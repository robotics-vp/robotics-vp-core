"""
VAE scaffolding for scene graph encoding/decoding.

Provides SceneGraphVAE for learning latent representations of scene graphs.
This is scaffolding only - training logic is not implemented.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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

from src.scene.vector_scene.graph import (
    EdgeType,
    NodeType,
    ObjectClass,
    SceneEdge,
    SceneGraph,
    SceneNode,
    SceneObject,
)
from src.scene.vector_scene.encoding import SceneGraphEncoder, ordered_scene_tensors


@dataclass
class SceneGraphLatent:
    """
    Latent representation of a scene graph.

    Attributes:
        node_latents: Per-node latent vectors, shape (N_nodes, D_latent)
        object_latents: Per-object latent vectors, shape (N_objects, D_latent)
        scene_latent: Scene-level latent vector, shape (D_latent,)
        mu: Mean of the latent distribution (for VAE)
        logvar: Log-variance of the latent distribution (for VAE)
        metadata: Additional metadata (e.g., original graph info)
    """
    node_latents: np.ndarray
    object_latents: np.ndarray
    scene_latent: np.ndarray
    mu: Optional[np.ndarray] = None
    logvar: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_tensors(
        cls,
        node_latents: "torch.Tensor",
        object_latents: "torch.Tensor",
        scene_latent: "torch.Tensor",
        mu: Optional["torch.Tensor"] = None,
        logvar: Optional["torch.Tensor"] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "SceneGraphLatent":
        """Create from PyTorch tensors."""
        return cls(
            node_latents=node_latents.detach().cpu().numpy(),
            object_latents=object_latents.detach().cpu().numpy(),
            scene_latent=scene_latent.detach().cpu().numpy(),
            mu=mu.detach().cpu().numpy() if mu is not None else None,
            logvar=logvar.detach().cpu().numpy() if logvar is not None else None,
            metadata=metadata or {},
        )

    def to_tensors(self, device: Optional[str] = None) -> Dict[str, "torch.Tensor"]:
        """Convert back to PyTorch tensors."""
        if torch is None:
            raise ImportError("PyTorch is required")

        result = {
            "node_latents": torch.tensor(self.node_latents, dtype=torch.float32),
            "object_latents": torch.tensor(self.object_latents, dtype=torch.float32),
            "scene_latent": torch.tensor(self.scene_latent, dtype=torch.float32),
        }
        if self.mu is not None:
            result["mu"] = torch.tensor(self.mu, dtype=torch.float32)
        if self.logvar is not None:
            result["logvar"] = torch.tensor(self.logvar, dtype=torch.float32)

        if device is not None:
            result = {k: v.to(device) for k, v in result.items()}

        return result


class SceneGraphDecoder(nn.Module):
    """
    Decoder for reconstructing scene graphs from latent representations.

    This mirrors the encoder architecture, mapping latent vectors back to
    node features, object features, and edge predictions.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        node_output_dim: int = 64,
        obj_output_dim: int = 32,
        max_nodes: int = 50,
        max_objects: int = 100,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        """
        Args:
            latent_dim: Dimension of latent representations
            node_output_dim: Output dimension for node features
            obj_output_dim: Output dimension for object features
            max_nodes: Maximum number of nodes to decode
            max_objects: Maximum number of objects to decode
            num_layers: Number of decoder layers
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.node_output_dim = node_output_dim
        self.obj_output_dim = obj_output_dim
        self.max_nodes = max_nodes
        self.max_objects = max_objects

        # Learned queries for nodes and objects
        self.node_queries = nn.Parameter(torch.randn(max_nodes, latent_dim))
        self.object_queries = nn.Parameter(torch.randn(max_objects, latent_dim))

        # Scene latent to query conditioning
        self.scene_to_node_cond = nn.Linear(latent_dim, latent_dim)
        self.scene_to_obj_cond = nn.Linear(latent_dim, latent_dim)

        # Decoder transformer layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=latent_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.node_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.obj_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output heads
        self.node_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, node_output_dim),
        )
        self.node_type_head = nn.Linear(latent_dim, len(NodeType))
        self.node_existence_head = nn.Linear(latent_dim, 1)

        self.obj_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, obj_output_dim),
        )
        self.obj_class_head = nn.Linear(latent_dim, len(ObjectClass))
        self.obj_existence_head = nn.Linear(latent_dim, 1)

        # Edge prediction head (for node pairs)
        self.edge_head = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, len(EdgeType) + 1),  # +1 for "no edge"
        )

    def forward(
        self,
        scene_latent: torch.Tensor,
        node_latents: Optional[torch.Tensor] = None,
        object_latents: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Decode latent representation to scene graph predictions.

        Args:
            scene_latent: Scene-level latent, shape (B, D_latent) or (D_latent,)
            node_latents: Optional node latents for conditioning
            object_latents: Optional object latents for conditioning

        Returns:
            Dictionary with:
            - "node_features": (B, max_nodes, D_node)
            - "node_types": (B, max_nodes, num_node_types)
            - "node_existence": (B, max_nodes, 1)
            - "object_features": (B, max_objects, D_obj)
            - "object_classes": (B, max_objects, num_classes)
            - "object_existence": (B, max_objects, 1)
            - "edge_logits": (B, max_nodes, max_nodes, num_edge_types + 1)
        """
        unbatched = scene_latent.dim() == 1
        if unbatched:
            scene_latent = scene_latent.unsqueeze(0)
            if node_latents is not None:
                node_latents = node_latents.unsqueeze(0)
            if object_latents is not None:
                object_latents = object_latents.unsqueeze(0)

        batch_size = scene_latent.size(0)
        device = scene_latent.device

        # Condition queries with scene latent
        node_cond = self.scene_to_node_cond(scene_latent).unsqueeze(1)  # (B, 1, D)
        obj_cond = self.scene_to_obj_cond(scene_latent).unsqueeze(1)  # (B, 1, D)

        # Expand queries to batch
        node_q = self.node_queries.unsqueeze(0).expand(batch_size, -1, -1) + node_cond
        obj_q = self.object_queries.unsqueeze(0).expand(batch_size, -1, -1) + obj_cond

        # Use node/object latents as memory if provided, else use scene latent
        if node_latents is not None and node_latents.size(1) > 0:
            node_memory = node_latents
        else:
            node_memory = scene_latent.unsqueeze(1)

        if object_latents is not None and object_latents.size(1) > 0:
            obj_memory = object_latents
        else:
            obj_memory = scene_latent.unsqueeze(1)

        # Decode nodes
        node_decoded = self.node_decoder(node_q, node_memory)
        node_features = self.node_head(node_decoded)
        node_types = self.node_type_head(node_decoded)
        node_existence = self.node_existence_head(node_decoded)

        # Decode objects
        obj_decoded = self.obj_decoder(obj_q, obj_memory)
        object_features = self.obj_head(obj_decoded)
        object_classes = self.obj_class_head(obj_decoded)
        object_existence = self.obj_existence_head(obj_decoded)

        # Predict edges between all node pairs
        # Create pairwise features
        node_i = node_decoded.unsqueeze(2).expand(-1, -1, self.max_nodes, -1)
        node_j = node_decoded.unsqueeze(1).expand(-1, self.max_nodes, -1, -1)
        pair_features = torch.cat([node_i, node_j], dim=-1)
        edge_logits = self.edge_head(pair_features)

        result = {
            "node_features": node_features,
            "node_types": node_types,
            "node_existence": node_existence,
            "object_features": object_features,
            "object_classes": object_classes,
            "object_existence": object_existence,
            "edge_logits": edge_logits,
        }

        if unbatched:
            result = {k: v.squeeze(0) for k, v in result.items()}

        return result


class SceneGraphVAE(nn.Module):
    """
    Variational Autoencoder for scene graphs.

    Encodes scene graphs to a latent space and decodes back to scene predictions.
    Supports both reconstruction and generation from latent samples.
    """

    def __init__(
        self,
        node_input_dim: int,
        obj_input_dim: int,
        latent_dim: int = 128,
        hidden_dim: int = 128,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
        pos_dim: int = 32,
        max_nodes: int = 50,
        max_objects: int = 100,
    ):
        """
        Args:
            node_input_dim: Input dimension for node features
            obj_input_dim: Input dimension for object features
            latent_dim: Dimension of latent space
            hidden_dim: Hidden dimension for encoder
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            num_heads: Number of attention heads
            dropout: Dropout probability
            pos_dim: Dimension of positional encodings
            max_nodes: Maximum nodes for decoder
            max_objects: Maximum objects for decoder
        """
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = SceneGraphEncoder(
            node_input_dim=node_input_dim,
            obj_input_dim=obj_input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_encoder_layers,
            num_heads=num_heads,
            dropout=dropout,
            pos_dim=pos_dim,
        )

        # VAE bottleneck
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.decoder = SceneGraphDecoder(
            latent_dim=latent_dim,
            node_output_dim=node_input_dim,
            obj_output_dim=obj_input_dim,
            max_nodes=max_nodes,
            max_objects=max_objects,
            num_layers=num_decoder_layers,
            num_heads=num_heads,
            dropout=dropout,
        )

    def encode(self, graph: SceneGraph, device: Optional[str] = None) -> SceneGraphLatent:
        """
        Encode a scene graph to latent representation.

        Args:
            graph: Input SceneGraph
            device: Target device

        Returns:
            SceneGraphLatent with encoded representations
        """
        # Convert graph to tensors
        scene_tensors = ordered_scene_tensors(graph, pos_dim=32, device=device)

        # Run encoder
        encoded = self.encoder(scene_tensors)

        # Compute mu and logvar from scene latent
        scene_latent = encoded["scene_latent"]
        if scene_latent.dim() == 1:
            scene_latent = scene_latent.unsqueeze(0)

        mu = self.fc_mu(scene_latent)
        logvar = self.fc_logvar(scene_latent)

        # Reparameterization
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        return SceneGraphLatent.from_tensors(
            node_latents=encoded["node_latents"],
            object_latents=encoded["object_latents"],
            scene_latent=z.squeeze(0),
            mu=mu.squeeze(0),
            logvar=logvar.squeeze(0),
            metadata={
                "num_nodes": len(graph.nodes),
                "num_objects": len(graph.objects),
                "num_edges": len(graph.edges),
            },
        )

    def decode(self, latent: SceneGraphLatent) -> SceneGraph:
        """
        Decode a latent representation to a scene graph.

        Note: This returns a SceneGraph with predicted properties.
        The actual structure is discretized from continuous predictions.

        Args:
            latent: SceneGraphLatent to decode

        Returns:
            Reconstructed SceneGraph
        """
        # Convert latent to tensors
        tensors = latent.to_tensors()
        scene_latent = tensors["scene_latent"]
        node_latents = tensors.get("node_latents")
        object_latents = tensors.get("object_latents")

        # Run decoder
        decoded = self.decoder(scene_latent, node_latents, object_latents)

        # Convert predictions to SceneGraph
        return self._predictions_to_graph(decoded)

    def _predictions_to_graph(self, predictions: Dict[str, torch.Tensor]) -> SceneGraph:
        """
        Convert decoder predictions to a SceneGraph.

        Uses argmax for categorical predictions and thresholding for existence.
        """
        # Get existence masks
        node_exist = (torch.sigmoid(predictions["node_existence"]) > 0.5).squeeze(-1)
        obj_exist = (torch.sigmoid(predictions["object_existence"]) > 0.5).squeeze(-1)

        nodes = []
        for i in range(node_exist.size(0)):
            if not node_exist[i]:
                continue
            node_type_idx = predictions["node_types"][i].argmax().item()
            node_type = list(NodeType)[node_type_idx]

            # Extract basic polyline from features (placeholder)
            # In practice, this would be more sophisticated
            feat = predictions["node_features"][i].detach().cpu().numpy()
            # Use first 4 values as bbox approximation
            if len(feat) >= 4:
                min_x, min_y, max_x, max_y = feat[:4]
                polyline = np.array([
                    [min_x, min_y],
                    [max_x, max_y],
                ], dtype=np.float32)
            else:
                polyline = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)

            nodes.append(SceneNode(
                id=i,
                polyline=polyline,
                node_type=node_type,
            ))

        objects = []
        for i in range(obj_exist.size(0)):
            if not obj_exist[i]:
                continue
            class_idx = predictions["object_classes"][i].argmax().item()
            obj_class = list(ObjectClass)[class_idx]

            # Extract position from features (placeholder)
            feat = predictions["object_features"][i].detach().cpu().numpy()
            x, y, z = feat[:3] if len(feat) >= 3 else (0.0, 0.0, 0.0)

            objects.append(SceneObject(
                id=i,
                class_id=obj_class,
                x=float(x),
                y=float(y),
                z=float(z),
            ))

        # Decode edges
        edges = []
        edge_logits = predictions["edge_logits"]
        num_edge_types = len(EdgeType)
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                if i < edge_logits.size(0) and j < edge_logits.size(1):
                    logits = edge_logits[i, j]
                    pred_idx = logits.argmax().item()
                    # Last index is "no edge"
                    if pred_idx < num_edge_types:
                        edge_type = list(EdgeType)[pred_idx]
                        edges.append(SceneEdge(
                            src_id=nodes[i].id,
                            dst_id=nodes[j].id,
                            edge_type=edge_type,
                        ))

        return SceneGraph(nodes=nodes, edges=edges, objects=objects)

    def forward(
        self,
        scene_tensors: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Forward pass for training.

        Args:
            scene_tensors: Output from ordered_scene_tensors()

        Returns:
            Tuple of (decoder_outputs, mu, logvar)
        """
        # Encode
        encoded = self.encoder(scene_tensors)
        scene_latent = encoded["scene_latent"]

        if scene_latent.dim() == 1:
            scene_latent = scene_latent.unsqueeze(0)

        # VAE bottleneck
        mu = self.fc_mu(scene_latent)
        logvar = self.fc_logvar(scene_latent)

        # Reparameterization
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        # Decode
        decoded = self.decoder(
            z.squeeze(0) if z.size(0) == 1 else z,
            encoded["node_latents"],
            encoded["object_latents"],
        )

        return decoded, mu, logvar

    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        mu: torch.Tensor,
        logvar: torch.Tensor,
        beta: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute VAE loss (reconstruction + KL divergence).

        Args:
            predictions: Decoder outputs
            targets: Ground truth tensors
            mu: Latent mean
            logvar: Latent log-variance
            beta: Weight for KL divergence term

        Returns:
            Dictionary with loss components:
            - "total": Total loss
            - "reconstruction": Reconstruction loss
            - "kl": KL divergence loss
            - "node_loss": Node reconstruction loss
            - "object_loss": Object reconstruction loss
            - "edge_loss": Edge prediction loss
        """
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # Reconstruction losses (stubs - would need proper target alignment)
        node_loss = torch.tensor(0.0, device=mu.device)
        object_loss = torch.tensor(0.0, device=mu.device)
        edge_loss = torch.tensor(0.0, device=mu.device)

        if "node_features" in targets and predictions["node_features"].size(0) > 0:
            # MSE for node features
            min_nodes = min(
                predictions["node_features"].size(0),
                targets["node_features"].size(0) if targets["node_features"].dim() > 1 else 0,
            )
            if min_nodes > 0:
                node_loss = F.mse_loss(
                    predictions["node_features"][:min_nodes],
                    targets["node_features"][:min_nodes],
                )

        if "object_features" in targets and predictions["object_features"].size(0) > 0:
            min_objects = min(
                predictions["object_features"].size(0),
                targets["object_features"].size(0) if targets["object_features"].dim() > 1 else 0,
            )
            if min_objects > 0:
                object_loss = F.mse_loss(
                    predictions["object_features"][:min_objects],
                    targets["object_features"][:min_objects],
                )

        reconstruction_loss = node_loss + object_loss + edge_loss
        total_loss = reconstruction_loss + beta * kl_loss

        return {
            "total": total_loss,
            "reconstruction": reconstruction_loss,
            "kl": kl_loss,
            "node_loss": node_loss,
            "object_loss": object_loss,
            "edge_loss": edge_loss,
        }

    def sample(self, num_samples: int = 1, device: Optional[str] = None) -> List[SceneGraph]:
        """
        Sample scene graphs from the prior.

        Args:
            num_samples: Number of samples to generate
            device: Target device

        Returns:
            List of sampled SceneGraphs
        """
        if device is None:
            device = next(self.parameters()).device

        # Sample from prior
        z = torch.randn(num_samples, self.latent_dim, device=device)

        graphs = []
        for i in range(num_samples):
            decoded = self.decoder(z[i])
            graph = self._predictions_to_graph(decoded)
            graphs.append(graph)

        return graphs
