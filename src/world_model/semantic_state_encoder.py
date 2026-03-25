"""Semantic state encoder — maps SemanticWorldModelState to fixed-dim embedding.

Phase 4 of the semantic WM neuralization path.

The ``SemanticStateEncoder`` bridges the deterministic world model
(variable-length objects, relations, meta-nodes) and neural consumers
that require fixed-dim conditioning vectors.

Architecture: Per-element encoders (objects, relations, meta-nodes) followed
by Pooling-by-Multihead-Attention (PMA) from the Set Transformer paper.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Feature engineering (no torch dependency)
# ---------------------------------------------------------------------------

# Vocabulary for category → index mapping
OBJECT_CATEGORIES = [
    "agent", "end_effector", "scene_region", "container",
    "fragile_object", "manipulated_object", "support_surface",
    "human_body", "unknown",
]

RELATION_TYPES = [
    "spatial_near", "part_of", "supports", "contains",
    "contacts", "grasps", "occludes", "unknown",
]

META_NODE_TYPES = [
    "risk_alert", "capability_gap", "optimization_target",
    "safety_constraint", "exploration_target", "unknown",
]


def _one_hot(value: str, vocab: List[str], fallback: str = "unknown") -> np.ndarray:
    vec = np.zeros(len(vocab), dtype=np.float32)
    idx = vocab.index(value) if value in vocab else vocab.index(fallback)
    vec[idx] = 1.0
    return vec


def encode_object(obj: Any) -> np.ndarray:
    """Encode a SemanticObjectState to fixed-dim vector.

    Features: confidence, salience, category (one-hot), n_affordances,
    n_risk_tags, n_state_tags.

    Dim: 2 scalars + len(OBJECT_CATEGORIES) + 3 = 14
    """
    category = str(getattr(obj, "category", "unknown"))
    cat_oh = _one_hot(category, OBJECT_CATEGORIES)
    scalars = np.array([
        float(getattr(obj, "confidence", 0.0)),
        float(getattr(obj, "salience", 0.0)),
        float(len(getattr(obj, "affordances", []))),
        float(len(getattr(obj, "risk_tags", []))),
        float(len(getattr(obj, "state_tags", []))),
    ], dtype=np.float32)
    return np.concatenate([scalars, cat_oh])


OBJECT_FEATURE_DIM = 5 + len(OBJECT_CATEGORIES)  # 14


def encode_relation(rel: Any) -> np.ndarray:
    """Encode a SemanticRelationState to fixed-dim vector.

    Dim: 1 scalar + len(RELATION_TYPES) = 9
    """
    rtype = str(getattr(rel, "relation_type", "unknown"))
    type_oh = _one_hot(rtype, RELATION_TYPES)
    scalars = np.array([
        float(getattr(rel, "confidence", 0.0)),
    ], dtype=np.float32)
    return np.concatenate([scalars, type_oh])


RELATION_FEATURE_DIM = 1 + len(RELATION_TYPES)  # 9


def encode_meta_node(mn: Any) -> np.ndarray:
    """Encode a SemanticMetaNode to fixed-dim vector.

    Dim: 1 scalar + len(META_NODE_TYPES) + 2 counts = 9
    """
    ntype = str(getattr(mn, "node_type", "unknown"))
    type_oh = _one_hot(ntype, META_NODE_TYPES)
    scalars = np.array([
        float(getattr(mn, "score", 0.0)),
        float(len(getattr(mn, "target_refs", []))),
        float(len(getattr(mn, "suggested_actions", []))),
    ], dtype=np.float32)
    return np.concatenate([scalars, type_oh])


META_NODE_FEATURE_DIM = 3 + len(META_NODE_TYPES)  # 9


def encode_wm_state_flat(wm_state: Any) -> np.ndarray:
    """Encode a SemanticWorldModelState as a fixed-dim vector by aggregation.

    This is the non-neural baseline: mean-pools per-element features and
    concatenates them with global scalars.

    Output dim: OBJECT_FEATURE_DIM + RELATION_FEATURE_DIM +
                META_NODE_FEATURE_DIM + n_global

    Used when torch is unavailable or as a fast fallback.
    """
    objects = getattr(wm_state, "objects", [])
    relations = getattr(wm_state, "relations", [])
    meta_nodes = getattr(wm_state, "meta_nodes", [])

    # Per-element mean pooling
    if objects:
        obj_feats = np.mean([encode_object(o) for o in objects], axis=0)
    else:
        obj_feats = np.zeros(OBJECT_FEATURE_DIM, dtype=np.float32)

    if relations:
        rel_feats = np.mean([encode_relation(r) for r in relations], axis=0)
    else:
        rel_feats = np.zeros(RELATION_FEATURE_DIM, dtype=np.float32)

    if meta_nodes:
        mn_feats = np.mean([encode_meta_node(m) for m in meta_nodes], axis=0)
    else:
        mn_feats = np.zeros(META_NODE_FEATURE_DIM, dtype=np.float32)

    # Global scalars
    cap_scores = getattr(wm_state, "capability_scores", {})
    global_scalars = np.array([
        float(len(objects)),
        float(len(relations)),
        float(len(meta_nodes)),
        float(len(getattr(wm_state, "semantic_tags", []))),
        float(np.mean(list(cap_scores.values())) if cap_scores else 0.0),
    ], dtype=np.float32)

    return np.concatenate([obj_feats, rel_feats, mn_feats, global_scalars])


FLAT_EMBED_DIM = OBJECT_FEATURE_DIM + RELATION_FEATURE_DIM + META_NODE_FEATURE_DIM + 5  # 37


# ---------------------------------------------------------------------------
# Torch-backed set encoder
# ---------------------------------------------------------------------------

try:
    import torch
    import torch.nn as nn

    class _PoolingByMultiheadAttention(nn.Module):
        """PMA from Set Transformer (Lee et al., 2019)."""

        def __init__(self, dim: int, num_heads: int = 4, num_seeds: int = 1):
            super().__init__()
            self.seed = nn.Parameter(torch.randn(1, num_seeds, dim))
            self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
            self.norm = nn.LayerNorm(dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """x: (B, N, D) → (B, num_seeds, D)."""
            b = x.size(0)
            seeds = self.seed.expand(b, -1, -1)
            out, _ = self.attn(seeds, x, x)
            return self.norm(out)

    class SemanticStateEncoder(nn.Module):
        """Set-transformer encoder: WM state → fixed-dim embedding.

        Encodes variable-length lists of objects, relations, and meta-nodes
        into a fixed-dim vector via PMA pooling + projection.
        """

        def __init__(
            self,
            embed_dim: int = 64,
            num_heads: int = 4,
            obj_hidden: int = 32,
            rel_hidden: int = 32,
            mn_hidden: int = 32,
        ):
            super().__init__()
            self.embed_dim = embed_dim

            # Per-element encoders
            self.obj_enc = nn.Sequential(
                nn.Linear(OBJECT_FEATURE_DIM, obj_hidden),
                nn.ReLU(),
                nn.Linear(obj_hidden, embed_dim),
            )
            self.rel_enc = nn.Sequential(
                nn.Linear(RELATION_FEATURE_DIM, rel_hidden),
                nn.ReLU(),
                nn.Linear(rel_hidden, embed_dim),
            )
            self.mn_enc = nn.Sequential(
                nn.Linear(META_NODE_FEATURE_DIM, mn_hidden),
                nn.ReLU(),
                nn.Linear(mn_hidden, embed_dim),
            )

            # PMA pooling per element type
            self.obj_pool = _PoolingByMultiheadAttention(embed_dim, num_heads)
            self.rel_pool = _PoolingByMultiheadAttention(embed_dim, num_heads)
            self.mn_pool = _PoolingByMultiheadAttention(embed_dim, num_heads)

            # Global scalar encoder
            self.global_enc = nn.Sequential(
                nn.Linear(5, embed_dim),
                nn.ReLU(),
            )

            # Final projection (4 pooled + 1 global → embed_dim)
            self.final_proj = nn.Sequential(
                nn.Linear(embed_dim * 4, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim),
            )

        def _encode_elements(
            self,
            features: List[np.ndarray],
            encoder: nn.Module,
            pool: _PoolingByMultiheadAttention,
        ) -> torch.Tensor:
            """Encode variable-length element list → (1, embed_dim)."""
            if not features:
                return torch.zeros(1, 1, self.embed_dim)
            x = torch.from_numpy(np.array(features, dtype=np.float32)).unsqueeze(0)
            encoded = encoder(x)  # (1, N, embed_dim)
            pooled = pool(encoded)  # (1, 1, embed_dim)
            return pooled

        def encode_state(self, wm_state: Any) -> torch.Tensor:
            """Encode a SemanticWorldModelState → (embed_dim,) tensor."""
            objects = getattr(wm_state, "objects", [])
            relations = getattr(wm_state, "relations", [])
            meta_nodes = getattr(wm_state, "meta_nodes", [])

            obj_feats = [encode_object(o) for o in objects]
            rel_feats = [encode_relation(r) for r in relations]
            mn_feats = [encode_meta_node(m) for m in meta_nodes]

            obj_pooled = self._encode_elements(obj_feats, self.obj_enc, self.obj_pool)
            rel_pooled = self._encode_elements(rel_feats, self.rel_enc, self.rel_pool)
            mn_pooled = self._encode_elements(mn_feats, self.mn_enc, self.mn_pool)

            # Global features
            cap_scores = getattr(wm_state, "capability_scores", {})
            global_feats = torch.tensor([
                float(len(objects)),
                float(len(relations)),
                float(len(meta_nodes)),
                float(len(getattr(wm_state, "semantic_tags", []))),
                float(np.mean(list(cap_scores.values())) if cap_scores else 0.0),
            ], dtype=torch.float32).unsqueeze(0)
            global_enc = self.global_enc(global_feats).unsqueeze(1)  # (1, 1, embed_dim)

            # Concatenate all pooled representations
            combined = torch.cat([
                obj_pooled.squeeze(1),
                rel_pooled.squeeze(1),
                mn_pooled.squeeze(1),
                global_enc.squeeze(1),
            ], dim=-1)  # (1, embed_dim * 4)

            embedding = self.final_proj(combined).squeeze(0)  # (embed_dim,)
            return embedding

        def encode_batch(self, wm_states: Sequence[Any]) -> torch.Tensor:
            """Encode multiple WM states → (B, embed_dim)."""
            embeddings = [self.encode_state(s).unsqueeze(0) for s in wm_states]
            return torch.cat(embeddings, dim=0)

        @classmethod
        def from_checkpoint(cls, path: str) -> "SemanticStateEncoder":
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            model = cls(embed_dim=ckpt.get("embed_dim", 64))
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            return model

    def train_semantic_encoder(
        wm_states: Sequence[Any],
        *,
        embed_dim: int = 64,
        epochs: int = 100,
        lr: float = 1e-3,
        margin: float = 1.0,
        save_path: Optional[str] = None,
    ) -> "SemanticStateEncoder":
        """Train via contrastive loss: same task_id → close, different → far.

        This is a simplified triplet-margin training loop.
        """
        if len(wm_states) < 3:
            raise ValueError("Need at least 3 WM states for contrastive training")

        model = SemanticStateEncoder(embed_dim=embed_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.TripletMarginLoss(margin=margin)

        # Group by task_id
        by_task: Dict[str, List[Any]] = {}
        for state in wm_states:
            task_id = str(getattr(state, "task_id", ""))
            by_task.setdefault(task_id, []).append(state)

        # Filter to tasks with at least 2 examples
        task_ids_with_pairs = [tid for tid, states in by_task.items() if len(states) >= 2]
        if not task_ids_with_pairs:
            raise ValueError("Need at least one task_id with 2+ states for contrastive training")

        all_task_ids = list(by_task.keys())

        model.train()
        rng = np.random.RandomState(42)
        for epoch in range(epochs):
            total_loss = 0.0
            count = 0
            for anchor_task in task_ids_with_pairs:
                states = by_task[anchor_task]
                # Pick anchor and positive
                indices = rng.choice(len(states), size=min(2, len(states)), replace=False)
                anchor = states[indices[0]]
                positive = states[indices[1] if len(indices) > 1 else 0]

                # Pick negative (different task)
                neg_tasks = [t for t in all_task_ids if t != anchor_task]
                if not neg_tasks:
                    continue
                neg_task = neg_tasks[rng.randint(len(neg_tasks))]
                neg_states = by_task[neg_task]
                negative = neg_states[rng.randint(len(neg_states))]

                a_emb = model.encode_state(anchor).unsqueeze(0)
                p_emb = model.encode_state(positive).unsqueeze(0)
                n_emb = model.encode_state(negative).unsqueeze(0)

                loss = loss_fn(a_emb, p_emb, n_emb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                count += 1

        model.eval()

        if save_path:
            torch.save({
                "model_state_dict": model.state_dict(),
                "embed_dim": embed_dim,
                "epochs": epochs,
                "n_states": len(wm_states),
            }, save_path)

        return model

    _TORCH_AVAILABLE = True

except ImportError:
    _TORCH_AVAILABLE = False

    class SemanticStateEncoder:  # type: ignore[no-redef]
        """Stub when torch is unavailable."""
        def __init__(self, *args, **kwargs):
            raise ImportError("SemanticStateEncoder requires torch")

        @classmethod
        def from_checkpoint(cls, path: str):
            raise ImportError("SemanticStateEncoder requires torch")

    def train_semantic_encoder(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("train_semantic_encoder requires torch")


__all__ = [
    "FLAT_EMBED_DIM",
    "SemanticStateEncoder",
    "encode_object",
    "encode_relation",
    "encode_meta_node",
    "encode_wm_state_flat",
    "train_semantic_encoder",
]
