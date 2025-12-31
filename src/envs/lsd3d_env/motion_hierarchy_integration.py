"""
Integration between LSD episodes and Motion Hierarchy Node (MHN).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from src.vision.motion_hierarchy.config import MotionHierarchyConfig
from src.vision.motion_hierarchy.motion_hierarchy_node import MotionHierarchyNode


@dataclass
class LSDMotionHierarchyInput:
    positions: torch.Tensor  # (T, N, D) world-space
    mask: Optional[torch.Tensor]  # (N,)
    node_labels: List[str]
    episode_id: str


def _extract_trajectory_payload(episode: Any) -> Dict[str, Any]:
    if hasattr(episode, "trajectory_data"):
        return getattr(episode, "trajectory_data") or {}
    if isinstance(episode, dict):
        if "agent_trajectories" in episode:
            return {
                "agent_trajectories": episode.get("agent_trajectories", []),
                "agent_labels": episode.get("agent_labels"),
            }
        return episode.get("trajectory_data", {}) or {}
    return {}


def _extract_episode_id(episode: Any) -> str:
    if hasattr(episode, "episode_id"):
        return str(getattr(episode, "episode_id"))
    if isinstance(episode, dict):
        return str(episode.get("episode_id", "unknown_episode"))
    return "unknown_episode"


def build_motion_hierarchy_input_from_lsd_episode(episode: Any) -> LSDMotionHierarchyInput:
    """
    Inspect the LSD episode structure and extract:

    - Robot link poses (base, joints, end-effector) in world coordinates.
    - Other dynamic agents' positions (forklifts, humans, etc.).
    - Optionally dynamic objects / keypoints if available.

    Map them to a unified (T, N, D) tensor and node_labels list.
    """
    payload = _extract_trajectory_payload(episode)
    agent_trajs = payload.get("agent_trajectories", [])
    if not agent_trajs:
        raise ValueError("LSD episode does not contain agent_trajectories payload")

    positions_list: List[np.ndarray] = []
    node_labels: List[str] = []
    valid_lengths: List[int] = []

    for traj in agent_trajs:
        pos = np.asarray(traj.get("positions", []), dtype=np.float32)
        if pos.ndim != 2 or pos.shape[0] == 0:
            positions_list.append(pos.reshape(0, 0))
            node_labels.append(str(traj.get("label", f"agent_{traj.get('agent_id', 'unknown')}")))
            continue
        valid_lengths.append(pos.shape[0])
        positions_list.append(pos)
        node_labels.append(str(traj.get("label", f"agent_{traj.get('agent_id', 'unknown')}")))

    if not valid_lengths:
        raise ValueError("No valid trajectories found in LSD episode")

    T = min(valid_lengths)
    valid_positions = [pos for pos in positions_list if pos.ndim == 2 and pos.shape[0] > 0]
    D = valid_positions[0].shape[1]
    for pos in valid_positions:
        if pos.shape[1] != D:
            raise ValueError("Inconsistent position dimensionality across trajectories")

    stacked: List[np.ndarray] = []
    mask_nodes = []
    for pos in positions_list:
        if pos.ndim != 2 or pos.shape[0] < T:
            stacked.append(np.zeros((T, D), dtype=np.float32))
            mask_nodes.append(0.0)
        else:
            stacked.append(pos[:T])
            mask_nodes.append(1.0)

    positions = np.stack(stacked, axis=1)  # (T, N, D)

    return LSDMotionHierarchyInput(
        positions=torch.tensor(positions, dtype=torch.float32),
        mask=torch.tensor(mask_nodes, dtype=torch.float32),
        node_labels=node_labels,
        episode_id=_extract_episode_id(episode),
    )


def _compute_tree_stats_from_hierarchy(
    hierarchy: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    parent_idx = hierarchy.argmax(dim=-1)
    N = parent_idx.shape[0]
    idx = torch.arange(N, device=hierarchy.device)
    self_parent = (parent_idx == idx).to(dtype=hierarchy.dtype)

    child_counts = torch.zeros((N,), device=hierarchy.device, dtype=hierarchy.dtype)
    child_counts.scatter_add_(0, parent_idx, torch.ones_like(parent_idx, dtype=hierarchy.dtype))
    branching = child_counts - self_parent

    depth = torch.zeros((N,), device=hierarchy.device, dtype=torch.long)
    current = parent_idx.clone()
    for _ in range(N):
        is_root = current == idx
        depth = depth + (~is_root)
        current = torch.where(is_root, current, parent_idx[current])

    if mask is not None:
        valid = mask.bool()
        denom = valid.sum().clamp(min=1).float()
        mean_depth = float((depth.float() * valid).sum() / denom)
        mean_branch = float((branching * valid).sum() / denom)
        root_frac = float((self_parent * valid).sum() / denom)
    else:
        mean_depth = float(depth.float().mean())
        mean_branch = float(branching.mean())
        root_frac = float(self_parent.mean())

    return {
        "mean_tree_depth": mean_depth,
        "mean_branch_factor": mean_branch,
        "root_fraction": root_frac,
    }


def compute_motion_hierarchy_for_lsd_episode(
    episode: Any,
    model: MotionHierarchyNode,
    config: MotionHierarchyConfig,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Builds LSDMotionHierarchyInput, runs the model once, and returns:

    - hierarchy: (N, N)
    - delta_resid_stats: mean/std per node
    - simple tree stats: depth, branching factor
    """
    mh_input = build_motion_hierarchy_input_from_lsd_episode(episode)
    device = device or torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")

    positions = mh_input.positions.unsqueeze(0).to(device)
    mask = mh_input.mask.unsqueeze(0).to(device) if mh_input.mask is not None else None

    model = model.to(device)
    model.eval()
    with torch.no_grad():
        out = model(positions, mask=mask, return_losses=False)

    hierarchy = out["hierarchy"][0].detach().cpu()
    delta_resid = out["delta_resid"][0].detach().cpu()  # (T-1, N, D)

    resid_mag = torch.norm(delta_resid, dim=-1)  # (T-1, N)
    resid_mean = resid_mag.mean(dim=0)
    resid_std = resid_mag.std(dim=0)

    tree_stats = _compute_tree_stats_from_hierarchy(hierarchy, mask=mh_input.mask)

    return {
        "episode_id": mh_input.episode_id,
        "node_labels": list(mh_input.node_labels),
        "hierarchy": hierarchy,
        "delta_resid_stats": {
            "mean": resid_mean.numpy().tolist(),
            "std": resid_std.numpy().tolist(),
        },
        "tree_stats": tree_stats,
    }


def build_motion_hierarchy_input_from_nag_scene(
    nag_scene: Any,
    camera_params: Any,
    num_frames: int,
) -> LSDMotionHierarchyInput:
    """
    OPTIONAL stub (can leave unimplemented or minimally implemented):

    Idea: sample plane poses from nag_scene over time (world-space or camera-space)
    and pass them through MotionHierarchyNode just like LSD episodes.
    """
    raise NotImplementedError("NAG-to-MHN integration is not implemented yet")
