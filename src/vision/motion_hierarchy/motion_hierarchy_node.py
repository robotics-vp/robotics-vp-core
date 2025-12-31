"""
Motion Hierarchy Node (MHN) for unsupervised motion structure learning.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.vision.motion_hierarchy.config import MotionHierarchyConfig


class MLP(nn.Module):
    """Simple MLP with optional batch normalization and lazy first layer."""

    def __init__(
        self,
        in_dim: Optional[int],
        hidden_dims: list[int],
        out_dim: int,
        use_batch_norm: bool = True,
    ) -> None:
        super().__init__()
        self._out_dim = int(out_dim)
        self._layers: nn.ModuleList = nn.ModuleList()

        dims: list[Optional[int]] = [in_dim] + [int(h) for h in hidden_dims] + [self._out_dim]
        for idx in range(len(dims) - 1):
            in_features = dims[idx]
            out_features = int(dims[idx + 1]) if dims[idx + 1] is not None else None
            if in_features is None:
                self._layers.append(nn.LazyLinear(out_features))  # type: ignore[arg-type]
            else:
                self._layers.append(nn.Linear(int(in_features), int(out_features)))
            if idx < len(dims) - 2:
                if use_batch_norm:
                    self._layers.append(nn.BatchNorm1d(int(out_features)))
                self._layers.append(nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])
        for layer in self._layers:
            x = layer(x)
        return x.reshape(*orig_shape, self._out_dim)


def _normalize_node_mask(mask: Optional[torch.Tensor], positions: torch.Tensor) -> Optional[torch.Tensor]:
    if mask is None:
        return None
    if mask.dim() == 4:
        mask_nodes = mask.squeeze(1).squeeze(-1)
    elif mask.dim() == 2:
        mask_nodes = mask
    else:
        raise ValueError(f"mask must have shape (B, N) or (B, 1, N, 1). Got {mask.shape}.")
    if mask_nodes.shape[0] != positions.shape[0]:
        raise ValueError("mask batch dimension must match positions batch dimension")
    return mask_nodes.to(dtype=positions.dtype, device=positions.device)


def build_knn_graph(
    positions: torch.Tensor,
    k: int,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Positions: (B, T, N, D)
    Returns neighbor_idx: (B, N, k) long

    - Use mean over time: mean_pos = positions.mean(dim=1)  # (B, N, D)
    - Compute pairwise squared distances per batch.
    - For each node i, select k nearest neighbors (including itself).
    - Honor mask by excluding invalid nodes from being chosen.
    """
    if positions.ndim != 4:
        raise ValueError(f"positions must be (B, T, N, D). Got {positions.shape}.")

    B, _, N, _ = positions.shape
    mean_pos = positions.mean(dim=1)
    diff = mean_pos.unsqueeze(2) - mean_pos.unsqueeze(1)
    dist = (diff ** 2).sum(dim=-1)

    mask_nodes = _normalize_node_mask(mask, positions)
    if mask_nodes is not None:
        valid = mask_nodes.bool().unsqueeze(1)
        dist = dist.masked_fill(~valid, float("inf"))

    idx = torch.arange(N, device=positions.device)
    dist[:, idx, idx] = 0.0

    k_eff = min(int(k), N)
    neighbor_idx = dist.topk(k_eff, largest=False).indices
    return neighbor_idx.long()


def reconstruct_deltas(
    delta_resid: torch.Tensor,
    H: torch.Tensor,
    l_max: int,
) -> torch.Tensor:
    """Reconstruct deltas via truncated Neumann series."""
    if delta_resid.ndim != 4:
        raise ValueError(f"delta_resid must be (B, T-1, N, D). Got {delta_resid.shape}.")
    if H.ndim != 3:
        raise ValueError(f"H must be (B, N, N). Got {H.shape}.")

    delta_hat = torch.zeros_like(delta_resid)
    v = delta_resid

    for level in range(int(l_max) + 1):
        delta_hat = delta_hat + v
        if level < int(l_max):
            v = torch.einsum("bij, btjd -> btid", H, v)

    return delta_hat


def compute_hierarchy_stats(
    hierarchy: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Compute simple hierarchy statistics from a (B, N, N) adjacency tensor."""
    if hierarchy.ndim != 3:
        raise ValueError(f"hierarchy must be (B, N, N). Got {hierarchy.shape}.")

    B, N, _ = hierarchy.shape
    device = hierarchy.device
    mask_nodes = mask
    if mask_nodes is not None:
        if mask_nodes.dim() == 4:
            mask_nodes = mask_nodes.squeeze(1).squeeze(-1)
        mask_nodes = mask_nodes.to(device=device)

    parent_idx = hierarchy.argmax(dim=-1)
    idx = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)
    self_parent = (parent_idx == idx).to(dtype=hierarchy.dtype)

    child_counts = torch.zeros((B, N), device=device, dtype=hierarchy.dtype)
    ones = torch.ones_like(parent_idx, dtype=hierarchy.dtype)
    child_counts.scatter_add_(1, parent_idx, ones)
    branching = child_counts - self_parent

    depths = torch.zeros((B, N), device=device, dtype=torch.long)
    current = parent_idx.clone()
    for _ in range(N):
        is_root = current == idx
        depths = depths + (~is_root)
        next_current = parent_idx.gather(1, current)
        current = torch.where(is_root, current, next_current)

    if mask_nodes is not None:
        valid = mask_nodes.bool()
        denom = valid.sum(dim=1).clamp(min=1).to(dtype=hierarchy.dtype)
        mean_depth = (depths.to(hierarchy.dtype) * valid).sum(dim=1) / denom
        mean_branch = (branching * valid).sum(dim=1) / denom
        mean_self = (self_parent * valid).sum(dim=1) / denom
    else:
        mean_depth = depths.to(hierarchy.dtype).mean(dim=1)
        mean_branch = branching.mean(dim=1)
        mean_self = self_parent.mean(dim=1)

    return {
        "mean_tree_depth": mean_depth,
        "mean_branch_factor": mean_branch,
        "mean_self_parent_prob": mean_self,
    }


class MotionHierarchyNode(nn.Module):
    """
    Motion hierarchy node that infers parent-child structure from trajectories.

    Assumes positions are in world coordinates with contiguous time steps.
    Mask indicates valid nodes (1=valid, 0=invalid).
    """

    def __init__(self, config: MotionHierarchyConfig):
        super().__init__()
        self.config = config
        self.node_mlp = MLP(
            in_dim=None,
            hidden_dims=[config.d_model],
            out_dim=config.d_model,
            use_batch_norm=config.use_batch_norm,
        )
        self.edge_mlp = MLP(
            in_dim=None,
            hidden_dims=[config.d_model, config.d_model],
            out_dim=1,
            use_batch_norm=config.use_batch_norm,
        )
        self.node_update_mlps = nn.ModuleList()
        for _ in range(self.config.num_gnn_layers - 1):
            self.node_update_mlps.append(
                MLP(
                    in_dim=self.config.d_model * 2,
                    hidden_dims=[self.config.d_model],
                    out_dim=self.config.d_model,
                    use_batch_norm=self.config.use_batch_norm,
                )
            )

    def _score_edges(
        self,
        node_feats: torch.Tensor,
        mean_pos: torch.Tensor,
        neighbor_idx: torch.Tensor,
    ) -> torch.Tensor:
        B, N, _ = node_feats.shape
        k = neighbor_idx.shape[-1]
        batch_idx = torch.arange(B, device=node_feats.device).view(B, 1, 1)
        neighbor_feats = node_feats[batch_idx, neighbor_idx]
        neighbor_pos = mean_pos[batch_idx, neighbor_idx]

        node_feats_i = node_feats.unsqueeze(2).expand(-1, -1, k, -1)
        mean_pos_i = mean_pos.unsqueeze(2).expand(-1, -1, k, -1)
        delta_pos = mean_pos_i - neighbor_pos

        edge_input = torch.cat([node_feats_i, neighbor_feats, delta_pos], dim=-1)
        edge_logits = self.edge_mlp(edge_input).squeeze(-1)
        return edge_logits

    def _ensure_self_parent(self, parent_logits: torch.Tensor) -> torch.Tensor:
        B, N, _ = parent_logits.shape
        row_has_finite = torch.isfinite(parent_logits).any(dim=-1)
        if row_has_finite.all():
            return parent_logits

        fill_logits = torch.full_like(parent_logits, float("-inf"))
        idx = torch.arange(N, device=parent_logits.device)
        fill_logits[:, idx, idx] = 0.0
        return torch.where(row_has_finite.unsqueeze(-1), parent_logits, fill_logits)

    def forward(
        self,
        positions: torch.Tensor,  # (B, T, N, D)
        mask: Optional[torch.Tensor] = None,  # (B, N) or (B, 1, N, 1)
        *,
        return_losses: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            positions: World-coordinate positions with contiguous time steps.
            mask: Optional node validity mask (1=valid, 0=invalid).
            return_losses: Whether to include reconstruction losses.

        Returns:
            deltas, delta_hat, delta_resid: (B, T-1, N, D)
            hierarchy: (B, N, N)
            parent_logits: (B, N, N)
            parent_probs: (B, N, N)
            losses: dict[str, torch.Tensor]    # only if return_losses=True
        """
        if positions.ndim != 4:
            raise ValueError(f"positions must be (B, T, N, D). Got {positions.shape}.")
        B, T, N, D = positions.shape
        if N > self.config.max_nodes:
            raise ValueError(f"N={N} exceeds max_nodes={self.config.max_nodes}.")

        deltas = positions[:, 1:, :, :] - positions[:, :-1, :, :]
        mean_pos = positions.mean(dim=1)

        neighbor_idx = build_knn_graph(positions, self.config.k_neighbors, mask=mask)

        node_feats = self.node_mlp(mean_pos)
        edge_logits = None
        for layer_idx in range(self.config.num_gnn_layers):
            edge_logits = self._score_edges(node_feats, mean_pos, neighbor_idx)
            if layer_idx < self.config.num_gnn_layers - 1:
                weights = torch.softmax(edge_logits, dim=-1).unsqueeze(-1)
                batch_idx = torch.arange(B, device=node_feats.device).view(B, 1, 1)
                neighbor_feats = node_feats[batch_idx, neighbor_idx]
                agg = (neighbor_feats * weights).sum(dim=2)
                node_feats = self.node_update_mlps[layer_idx](torch.cat([node_feats, agg], dim=-1))

        if edge_logits is None:
            raise RuntimeError("Failed to compute edge logits.")

        parent_logits = torch.full((B, N, N), float("-inf"), device=positions.device)
        parent_logits.scatter_(2, neighbor_idx, edge_logits)

        mask_nodes = _normalize_node_mask(mask, positions)
        if mask_nodes is not None:
            parent_logits = parent_logits.masked_fill(~mask_nodes.bool().unsqueeze(1), float("-inf"))
            parent_logits = parent_logits.masked_fill(~mask_nodes.bool().unsqueeze(-1), float("-inf"))

        parent_logits = self._ensure_self_parent(parent_logits)

        parent_probs = torch.softmax(parent_logits, dim=-1)
        if mask_nodes is not None:
            diag = torch.eye(N, device=positions.device, dtype=parent_probs.dtype).unsqueeze(0)
            row_mask = mask_nodes.unsqueeze(-1)
            parent_probs = parent_probs * row_mask + diag * (1.0 - row_mask)

        inherited = torch.einsum("bij, btjd -> btid", parent_probs, deltas)
        delta_resid = deltas - inherited

        hierarchy = F.gumbel_softmax(
            parent_logits,
            tau=self.config.gumbel_tau,
            hard=self.config.gumbel_hard,
            dim=-1,
        )
        if mask_nodes is not None:
            diag = torch.eye(N, device=positions.device, dtype=hierarchy.dtype).unsqueeze(0)
            row_mask = mask_nodes.unsqueeze(-1)
            hierarchy = hierarchy * row_mask + diag * (1.0 - row_mask)

        delta_hat = reconstruct_deltas(delta_resid, hierarchy, self.config.l_max)

        stats = compute_hierarchy_stats(hierarchy, mask=mask_nodes)

        output: Dict[str, torch.Tensor] = {
            "deltas": deltas,
            "delta_hat": delta_hat,
            "delta_resid": delta_resid,
            "hierarchy": hierarchy,
            "parent_logits": parent_logits,
            "parent_probs": parent_probs,
        }
        for key, value in stats.items():
            output[key] = value

        if return_losses:
            if mask_nodes is not None:
                mask_exp = mask_nodes.unsqueeze(1).unsqueeze(-1)
                denom = mask_exp.sum().clamp(min=1.0)
                recon = F.smooth_l1_loss(delta_hat * mask_exp, deltas * mask_exp, reduction="sum") / denom
                resid = (delta_resid.abs() * mask_exp).sum() / denom
            else:
                recon = F.smooth_l1_loss(delta_hat, deltas)
                resid = delta_resid.abs().mean()
            loss_resid = self.config.lambda_residual * resid
            loss_total = recon + loss_resid
            output["losses"] = {
                "total": loss_total,
                "recon": recon,
                "resid": loss_resid,
            }

        return output
