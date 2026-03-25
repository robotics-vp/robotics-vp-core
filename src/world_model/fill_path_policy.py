"""Learned fill-path policy — replaces heuristic _decide_fill_path().

Phase 3 of the semantic WM neuralization path.

The ``LearnedFillPathPolicy`` classifies gap features into the optimal
fill method (real_sim, diffusion, synthetic_branch, blocked), trained
on historical fill-outcome records to maximize coverage improvement.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.world_model.gap_ranker import GapFeatureExtractor

FILL_METHODS = ["real_sim", "diffusion", "synthetic_branch", "blocked"]


# ---------------------------------------------------------------------------
# Torch-backed policy
# ---------------------------------------------------------------------------

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class LearnedFillPathPolicy(nn.Module):
        """Classifier: gap features → fill method + confidence.

        Architecture: 2-layer MLP with softmax over 4 methods.
        """

        def __init__(
            self,
            input_dim: int = GapFeatureExtractor.FEATURE_DIM,
            hidden_dim: int = 64,
            n_methods: int = len(FILL_METHODS),
        ):
            super().__init__()
            self.n_methods = n_methods
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            self.method_head = nn.Linear(hidden_dim, n_methods)
            self.confidence_head = nn.Sequential(
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid(),
            )

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """Forward pass.

            Args:
                x: (B, input_dim) gap feature vectors

            Returns:
                method_logits: (B, n_methods)
                confidence: (B, 1) in [0, 1]
            """
            h = self.net(x)
            return self.method_head(h), self.confidence_head(h)

        def predict(
            self,
            edge: Any,
            graph: Any,
            extractor: Optional[GapFeatureExtractor] = None,
        ) -> Tuple[str, float]:
            """Predict fill method and confidence for a single edge.

            Returns:
                (method_name, confidence)
            """
            ext = extractor or GapFeatureExtractor()
            fv = ext(edge, graph)
            x = torch.from_numpy(fv.raw).unsqueeze(0)
            with torch.no_grad():
                logits, conf = self.forward(x)
                probs = F.softmax(logits, dim=-1)
                method_idx = probs.argmax(dim=-1).item()
            return FILL_METHODS[int(method_idx)], float(conf.item())

        def predict_batch(
            self,
            edges: Sequence[Any],
            graph: Any,
            extractor: Optional[GapFeatureExtractor] = None,
        ) -> List[Tuple[str, float]]:
            """Predict fill methods for multiple edges."""
            ext = extractor or GapFeatureExtractor()
            features = ext.extract_batch(edges, graph)
            x = torch.from_numpy(features)
            with torch.no_grad():
                logits, conf = self.forward(x)
                probs = F.softmax(logits, dim=-1)
                method_indices = probs.argmax(dim=-1).tolist()
                confidences = conf.squeeze(-1).tolist()
            return [
                (FILL_METHODS[int(idx)], float(c))
                for idx, c in zip(method_indices, confidences)
            ]

        @classmethod
        def from_checkpoint(cls, path: str) -> "LearnedFillPathPolicy":
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

    class LearnedFillPathPolicy:  # type: ignore[no-redef]
        """Stub when torch is unavailable."""
        def __init__(self, *args, **kwargs):
            raise ImportError("LearnedFillPathPolicy requires torch")

        @classmethod
        def from_checkpoint(cls, path: str):
            raise ImportError("LearnedFillPathPolicy requires torch")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _best_method_for_edge(
    records: Sequence[Any],
) -> str:
    """Find the method that maximized marginal_value for a given edge."""
    by_method: Dict[str, List[float]] = {}
    for rec in records:
        method = str(getattr(rec, "fill_method", ""))
        by_method.setdefault(method, []).append(float(getattr(rec, "marginal_value", 0.0)))

    best_method = "blocked"
    best_avg = -float("inf")
    for method, values in by_method.items():
        avg = sum(values) / len(values)
        if avg > best_avg:
            best_avg = avg
            best_method = method
    return best_method


def train_fill_path_policy(
    outcome_records: Sequence[Any],
    *,
    epochs: int = 50,
    lr: float = 1e-3,
    hidden_dim: int = 64,
    save_path: Optional[str] = None,
) -> Any:
    """Train a LearnedFillPathPolicy from fill-outcome records.

    Training objective: classify to the method that historically
    maximized ``marginal_value`` for similar gap features.
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("Training requires torch")

    extractor = GapFeatureExtractor()

    # Group records by edge, find best method per edge
    by_edge: Dict[str, List[Any]] = {}
    for rec in outcome_records:
        edge_key = str(getattr(rec, "edge_key", ""))
        by_edge.setdefault(edge_key, []).append(rec)

    features_list = []
    labels_list = []
    for edge_key, recs in by_edge.items():
        best = _best_method_for_edge(recs)
        if best not in FILL_METHODS:
            continue
        label = FILL_METHODS.index(best)
        # Use the most recent record's features as representative
        fv = extractor.from_outcome_record(recs[-1])
        features_list.append(fv.raw)
        labels_list.append(label)

    if not features_list:
        raise ValueError("No training data produced")

    X = torch.from_numpy(np.array(features_list, dtype=np.float32))
    y = torch.tensor(labels_list, dtype=torch.long)

    model = LearnedFillPathPolicy(
        input_dim=GapFeatureExtractor.FEATURE_DIM,
        hidden_dim=hidden_dim,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        logits, _ = model(X)
        loss = loss_fn(logits, y)
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
    "FILL_METHODS",
    "LearnedFillPathPolicy",
    "train_fill_path_policy",
]
