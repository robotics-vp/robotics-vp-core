"""
Configuration for Motion Hierarchy Node (MHN).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class MotionHierarchyConfig:
    """
    Configuration for MotionHierarchyNode.

    Attributes:
        d_model: Hidden dimension for node and edge MLPs.
        num_gnn_layers: Number of message passing layers (>=1).
        k_neighbors: Number of nearest neighbors per node.
        l_max: Truncation depth for Neumann-series decoder.
        lambda_residual: Weight for residual penalty.
        use_batch_norm: Whether to use batch normalization in MLPs.
        gumbel_tau: Temperature for Gumbel-Softmax.
        gumbel_hard: Whether to use straight-through one-hot sampling.
        max_nodes: Upper bound for node counts (used for sanity checks).
        device: Device string ("cpu", "cuda").
    """

    d_model: int = 128
    num_gnn_layers: int = 3
    k_neighbors: int = 8
    l_max: int = 4
    lambda_residual: float = 1.0
    use_batch_norm: bool = True
    gumbel_tau: float = 0.5
    gumbel_hard: bool = True
    max_nodes: int = 256
    device: str = "cuda"

    def __post_init__(self) -> None:
        self.d_model = int(self.d_model)
        self.num_gnn_layers = max(1, int(self.num_gnn_layers))
        self.k_neighbors = max(1, int(self.k_neighbors))
        self.l_max = max(0, int(self.l_max))
        self.lambda_residual = float(self.lambda_residual)
        self.use_batch_norm = bool(self.use_batch_norm)
        self.gumbel_tau = float(self.gumbel_tau)
        self.gumbel_hard = bool(self.gumbel_hard)
        self.max_nodes = max(1, int(self.max_nodes))
        self.device = str(self.device)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dictionary."""
        return {
            "d_model": self.d_model,
            "num_gnn_layers": self.num_gnn_layers,
            "k_neighbors": self.k_neighbors,
            "l_max": self.l_max,
            "lambda_residual": self.lambda_residual,
            "use_batch_norm": self.use_batch_norm,
            "gumbel_tau": self.gumbel_tau,
            "gumbel_hard": self.gumbel_hard,
            "max_nodes": self.max_nodes,
            "device": self.device,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MotionHierarchyConfig":
        """Create config from a dictionary."""
        return cls(
            d_model=data.get("d_model", 128),
            num_gnn_layers=data.get("num_gnn_layers", 3),
            k_neighbors=data.get("k_neighbors", 8),
            l_max=data.get("l_max", 4),
            lambda_residual=data.get("lambda_residual", 1.0),
            use_batch_norm=data.get("use_batch_norm", True),
            gumbel_tau=data.get("gumbel_tau", 0.5),
            gumbel_hard=data.get("gumbel_hard", True),
            max_nodes=data.get("max_nodes", 256),
            device=data.get("device", "cuda"),
        )
