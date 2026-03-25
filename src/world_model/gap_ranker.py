"""Learned gap ranker — replaces heuristic gap_score() with a trained MLP.

Phase 2 of the semantic WM neuralization path.

The ``LearnedGapRanker`` predicts the *expected marginal value* of filling
a coverage gap edge, trained on historical fill-outcome records:

    marginal_value = (coverage_delta × quality_score) / wall_time_s

This is strictly more informative than the heuristic
``economic_weight × econ + trust_weight × trust + readiness_weight × readiness``
because it learns which combinations of features *actually* produce
coverage improvements.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

# Feature vocabulary for one-hot encoding
EDGE_TYPES = ["requires", "realizes", "covers", "supports"]
NODE_TYPES = [
    "task", "skill", "env_primitive", "backend",
    "object_family", "risk_family", "affordance_family",
]


@dataclass
class GapFeatureVector:
    """Fixed-dim feature vector extracted from a CoverageEdge + graph context."""
    raw: np.ndarray  # shape (feature_dim,)

    @property
    def dim(self) -> int:
        return len(self.raw)


class GapFeatureExtractor:
    """Extract fixed-dim feature vector from a CoverageEdge.

    Features (total dim = 7 scalar + 4 one-hot edge_type + 7×2 one-hot node_types = 25):
        0: evidence_count (log-scaled)
        1: evidence_strength
        2: economic_priority
        3: trust_priority
        4: promotion_readiness
        5: in_degree (log-scaled)
        6: out_degree (log-scaled)
        7-10: edge_type one-hot
        11-17: source node_type one-hot
        18-24: target node_type one-hot
    """

    FEATURE_DIM = 7 + len(EDGE_TYPES) + 2 * len(NODE_TYPES)  # 25

    def __call__(
        self,
        edge: Any,
        graph: Any,
    ) -> GapFeatureVector:
        """Extract features from a single edge in context of its graph."""
        features = np.zeros(self.FEATURE_DIM, dtype=np.float32)

        # Scalar features
        features[0] = np.log1p(float(getattr(edge, "evidence_count", 0)))
        features[1] = float(getattr(edge, "evidence_strength", 0.0))
        features[2] = float(getattr(edge, "economic_priority", 0.0))
        features[3] = float(getattr(edge, "trust_priority", 0.0))
        features[4] = float(getattr(edge, "promotion_readiness", 0.0))

        # Graph topology features
        if graph is not None:
            in_deg = len(getattr(graph, "edges_to", lambda _: [])(edge.target_id))
            out_deg = len(getattr(graph, "edges_from", lambda _: [])(edge.source_id))
            features[5] = np.log1p(float(in_deg))
            features[6] = np.log1p(float(out_deg))

        # Edge type one-hot
        edge_type = str(getattr(edge, "edge_type", ""))
        offset = 7
        if edge_type in EDGE_TYPES:
            features[offset + EDGE_TYPES.index(edge_type)] = 1.0

        # Source node type one-hot
        offset = 7 + len(EDGE_TYPES)
        src_node = graph.node_by_id(edge.source_id) if graph else None
        if src_node is not None:
            src_type = getattr(src_node, "node_type", "")
            if src_type in NODE_TYPES:
                features[offset + NODE_TYPES.index(src_type)] = 1.0

        # Target node type one-hot
        offset = 7 + len(EDGE_TYPES) + len(NODE_TYPES)
        tgt_node = graph.node_by_id(edge.target_id) if graph else None
        if tgt_node is not None:
            tgt_type = getattr(tgt_node, "node_type", "")
            if tgt_type in NODE_TYPES:
                features[offset + NODE_TYPES.index(tgt_type)] = 1.0

        return GapFeatureVector(raw=features)

    def extract_batch(
        self,
        edges: Sequence[Any],
        graph: Any,
    ) -> np.ndarray:
        """Extract features for multiple edges. Returns (N, feature_dim)."""
        return np.array([self(e, graph).raw for e in edges], dtype=np.float32)

    @classmethod
    def from_outcome_record(cls, record: Any) -> GapFeatureVector:
        """Extract features from a FillOutcomeRecord's gap_features dict."""
        features = np.zeros(cls.FEATURE_DIM, dtype=np.float32)
        gap_feats = getattr(record, "gap_features", {})
        if isinstance(gap_feats, dict):
            features[2] = float(gap_feats.get("economic_priority", 0.0))
            features[3] = float(gap_feats.get("trust_priority", 0.0))
            features[4] = float(gap_feats.get("readiness", 0.0))
            features[0] = np.log1p(float(getattr(record, "pre_evidence_count", 0)))
        return GapFeatureVector(raw=features)


# ---------------------------------------------------------------------------
# Learned ranker (torch-optional)
# ---------------------------------------------------------------------------

try:
    import torch
    import torch.nn as nn

    class LearnedGapRanker(nn.Module):
        """MLP: edge_features → scalar marginal_value prediction.

        Architecture: 2-layer MLP with ReLU activation.
        Training target: ``FillOutcomeRecord.marginal_value``.
        """

        def __init__(self, input_dim: int = GapFeatureExtractor.FEATURE_DIM, hidden_dim: int = 64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Predict marginal value for each edge.

            Args:
                x: (B, input_dim) gap feature vectors

            Returns:
                values: (B, 1) predicted marginal values
            """
            return self.net(x)

        def rank_edges(
            self,
            edges: Sequence[Any],
            graph: Any,
            extractor: Optional[GapFeatureExtractor] = None,
        ) -> List[Tuple[Any, float]]:
            """Rank edges by predicted marginal value (descending).

            Returns list of (edge, predicted_value) tuples.
            """
            ext = extractor or GapFeatureExtractor()
            features = ext.extract_batch(edges, graph)
            with torch.no_grad():
                values = self.forward(torch.from_numpy(features)).squeeze(-1).numpy()
            pairs = list(zip(edges, values.tolist()))
            pairs.sort(key=lambda p: p[1], reverse=True)
            return pairs

        @classmethod
        def from_checkpoint(cls, path: str) -> "LearnedGapRanker":
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            model = cls(
                input_dim=ckpt.get("input_dim", GapFeatureExtractor.FEATURE_DIM),
                hidden_dim=ckpt.get("hidden_dim", 64),
            )
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            return model

    _TORCH_AVAILABLE = True

except ImportError:
    _TORCH_AVAILABLE = False

    class LearnedGapRanker:  # type: ignore[no-redef]
        """Stub when torch is unavailable."""
        def __init__(self, *args, **kwargs):
            raise ImportError("LearnedGapRanker requires torch")

        @classmethod
        def from_checkpoint(cls, path: str):
            raise ImportError("LearnedGapRanker requires torch")


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------

def train_gap_ranker(
    outcome_records: Sequence[Any],
    *,
    epochs: int = 50,
    lr: float = 1e-3,
    hidden_dim: int = 64,
    save_path: Optional[str] = None,
) -> Any:
    """Train a LearnedGapRanker from fill-outcome records.

    Args:
        outcome_records: list of FillOutcomeRecord
        epochs: training epochs
        lr: learning rate
        hidden_dim: MLP hidden dimension
        save_path: optional path to save checkpoint

    Returns:
        Trained LearnedGapRanker
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("Training requires torch")

    extractor = GapFeatureExtractor()

    # Build training data
    features_list = []
    targets_list = []
    for rec in outcome_records:
        fv = extractor.from_outcome_record(rec)
        features_list.append(fv.raw)
        targets_list.append(float(getattr(rec, "marginal_value", 0.0)))

    if not features_list:
        raise ValueError("No training data provided")

    X = torch.from_numpy(np.array(features_list, dtype=np.float32))
    y = torch.tensor(targets_list, dtype=torch.float32).unsqueeze(-1)

    model = LearnedGapRanker(
        input_dim=GapFeatureExtractor.FEATURE_DIM,
        hidden_dim=hidden_dim,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()

    if save_path:
        torch.save({
            "model_state_dict": model.state_dict(),
            "input_dim": GapFeatureExtractor.FEATURE_DIM,
            "hidden_dim": hidden_dim,
            "epochs": epochs,
            "n_records": len(outcome_records),
        }, save_path)

    return model


__all__ = [
    "GapFeatureExtractor",
    "GapFeatureVector",
    "LearnedGapRanker",
    "train_gap_ranker",
]
