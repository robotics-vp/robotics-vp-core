"""
Differentiable renderer for NAGScene.

Renders NAG scenes by ray-plane intersection and alpha compositing.
"""

from __future__ import annotations

from typing import Collection, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    F = None  # type: ignore
    TORCH_AVAILABLE = False

from src.vision.nag.types import CameraParams, NAGNodeId


def _check_torch() -> None:
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for NAG renderer")


def ray_plane_intersection(
    ray_origins: torch.Tensor,
    ray_dirs: torch.Tensor,
    plane_origin: torch.Tensor,
    plane_normal: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute ray-plane intersections.

    Args:
        ray_origins: (..., 3) ray origins
        ray_dirs: (..., 3) normalized ray directions
        plane_origin: (3,) point on the plane
        plane_normal: (3,) plane normal

    Returns:
        t: (...,) intersection distances (inf if no intersection)
        valid: (...,) boolean mask for valid intersections
    """
    # Plane equation: dot(n, p - o) = 0
    # Ray: p = ray_o + t * ray_d
    # Solve: dot(n, ray_o + t * ray_d - plane_o) = 0
    # t = dot(n, plane_o - ray_o) / dot(n, ray_d)

    denom = torch.sum(plane_normal * ray_dirs, dim=-1)  # (...)
    numer = torch.sum(plane_normal * (plane_origin - ray_origins), dim=-1)  # (...)

    # Avoid division by zero
    valid_denom = torch.abs(denom) > 1e-8
    t = torch.where(valid_denom, numer / (denom + 1e-8), torch.tensor(float("inf"), device=ray_origins.device))

    # Valid if t > 0 (in front of camera)
    valid = valid_denom & (t > 0)

    return t, valid


def render_scene(
    scene: "NAGScene",
    camera: CameraParams,
    t: torch.Tensor,
    include_nodes: Optional[Collection[NAGNodeId]] = None,
    exclude_nodes: Optional[Collection[NAGNodeId]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Render a NAGScene from a camera viewpoint.

    Args:
        scene: NAGScene to render
        camera: Camera parameters
        t: Scalar or (B,) tensor of times
        include_nodes: If provided, only render these nodes
        exclude_nodes: If provided, exclude these nodes

    Returns:
        Dict with:
            "rgb": (3, H, W) or (B, 3, H, W) rendered RGB
            "depth": (1, H, W) or (B, 1, H, W) depth map
            "node_index": (1, H, W) or (B, 1, H, W) node indices
    """
    _check_torch()

    from src.vision.nag.scene import NAGScene

    device = t.device if isinstance(t, torch.Tensor) else torch.device("cpu")
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t, device=device)

    is_scalar = t.dim() == 0
    if is_scalar:
        t = t.unsqueeze(0)

    B = t.shape[0]
    H, W = camera.height, camera.width

    # Determine which nodes to render
    all_nodes = scene.list_nodes()
    if include_nodes is not None:
        nodes_to_render = [n for n in all_nodes if n in include_nodes]
    else:
        nodes_to_render = all_nodes

    if exclude_nodes is not None:
        nodes_to_render = [n for n in nodes_to_render if n not in exclude_nodes]

    # Get rays for frame 0 (assume static camera for simplicity)
    ray_origins_np, ray_dirs_np = camera.get_rays(t=0)
    ray_origins = torch.from_numpy(ray_origins_np.astype(np.float32)).to(device)  # (H, W, 3)
    ray_dirs = torch.from_numpy(ray_dirs_np.astype(np.float32)).to(device)  # (H, W, 3)

    # Initialize output buffers
    rgb_accum = torch.zeros(B, H, W, 3, device=device)
    alpha_accum = torch.zeros(B, H, W, device=device)
    depth_buffer = torch.full((B, H, W), float("inf"), device=device)
    node_index_buffer = torch.full((B, H, W), -1, dtype=torch.long, device=device)

    # Collect all hit info for depth sorting
    all_hits: List[Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []

    for node_idx, node_id in enumerate(nodes_to_render):
        node = scene.get_node(node_id)

        for b in range(B):
            t_val = t[b]

            # Get plane pose at time t
            pose = node.pose_at(t_val)  # (4, 4)
            plane_origin = pose[:3, 3]
            plane_normal = pose[:3, 2]  # Z-axis is normal

            # Ray-plane intersection
            hit_t, valid = ray_plane_intersection(
                ray_origins, ray_dirs, plane_origin, plane_normal
            )  # (H, W), (H, W)

            # Compute hit points
            hit_points = ray_origins + ray_dirs * hit_t.unsqueeze(-1)  # (H, W, 3)

            # Transform to plane-local coordinates
            plane_from_world = torch.inverse(pose)
            hit_points_homo = torch.cat([hit_points, torch.ones_like(hit_points[..., :1])], dim=-1)
            hit_local = torch.einsum("ij,hwj->hwi", plane_from_world, hit_points_homo)[..., :3]

            # Convert to UV (plane extent centered at origin)
            extent = torch.from_numpy(node.extent).to(device)
            u = hit_local[..., 0] / extent[0] + 0.5
            v = hit_local[..., 1] / extent[1] + 0.5

            # Check if within plane bounds
            in_bounds = (u >= 0) & (u <= 1) & (v >= 0) & (v <= 1)
            valid = valid & in_bounds

            # Stack UV coordinates
            uv = torch.stack([u, v], dim=-1)  # (H, W, 2)

            # Compute view direction in plane frame
            view_dir_world = -ray_dirs  # Direction from hit to camera
            view_dir_local = torch.einsum("ij,hwj->hwi", plane_from_world[:3, :3], view_dir_world)

            # Sample atlas for valid pixels
            rgb = torch.zeros(H, W, 3, device=device)
            alpha = torch.zeros(H, W, device=device)

            if valid.any():
                valid_uv = uv[valid]
                valid_view = view_dir_local[valid]
                valid_rgb, valid_alpha = node.sample_atlas(valid_uv, t_val, valid_view)
                rgb[valid] = valid_rgb
                alpha[valid] = valid_alpha

            all_hits.append((node_idx, hit_t.clone(), valid.clone(), rgb.clone(), alpha.clone(), torch.tensor(b)))

    # Sort hits by depth and composite front-to-back for each batch
    for b in range(B):
        batch_hits = [(idx, ht, v, rgb, a) for idx, ht, v, rgb, a, bi in all_hits if bi == b]

        if not batch_hits:
            continue

        # Stack all hits
        num_hits = len(batch_hits)
        all_depths = torch.stack([ht for _, ht, _, _, _ in batch_hits], dim=0)  # (N, H, W)
        all_valid = torch.stack([v for _, _, v, _, _ in batch_hits], dim=0)  # (N, H, W)
        all_rgb = torch.stack([rgb for _, _, _, rgb, _ in batch_hits], dim=0)  # (N, H, W, 3)
        all_alpha = torch.stack([a for _, _, _, _, a in batch_hits], dim=0)  # (N, H, W)
        all_node_idx = torch.tensor([idx for idx, _, _, _, _ in batch_hits], device=device)

        # Set invalid depths to infinity
        all_depths = torch.where(all_valid, all_depths, torch.tensor(float("inf"), device=device))

        # Sort by depth per pixel
        sorted_depths, sort_indices = torch.sort(all_depths, dim=0)  # (N, H, W)

        # Gather sorted values
        H_idx = torch.arange(H, device=device).view(1, H, 1).expand(num_hits, -1, W)
        W_idx = torch.arange(W, device=device).view(1, 1, W).expand(num_hits, H, -1)
        N_idx = sort_indices

        sorted_rgb = all_rgb[N_idx, H_idx, W_idx]  # (N, H, W, 3)
        sorted_alpha = all_alpha[N_idx, H_idx, W_idx]  # (N, H, W)
        sorted_valid = all_valid[N_idx, H_idx, W_idx]  # (N, H, W)

        # Front-to-back compositing
        T = torch.ones(H, W, device=device)  # Transmittance

        for n in range(num_hits):
            rgb_n = sorted_rgb[n]  # (H, W, 3)
            alpha_n = sorted_alpha[n]  # (H, W)
            valid_n = sorted_valid[n]  # (H, W)
            depth_n = sorted_depths[n]  # (H, W)
            node_idx_n = all_node_idx[sort_indices[n, 0, 0]]  # Approximate

            # Only contribute where valid and transmittance > 0
            mask = valid_n & (T > 1e-4)

            # Composite
            contrib = alpha_n * T
            rgb_accum[b] += contrib.unsqueeze(-1) * rgb_n * mask.unsqueeze(-1).float()
            alpha_accum[b] += contrib * mask.float()

            # Update depth (first valid hit)
            first_hit_mask = mask & (depth_buffer[b] == float("inf"))
            depth_buffer[b] = torch.where(first_hit_mask, depth_n, depth_buffer[b])

            # Update node index
            node_index_buffer[b] = torch.where(
                first_hit_mask,
                sort_indices[n].long(),
                node_index_buffer[b],
            )

            # Update transmittance
            T = T * (1 - alpha_n * mask.float())

    # Normalize RGB by accumulated alpha (avoid division by zero)
    rgb_out = rgb_accum / (alpha_accum.unsqueeze(-1) + 1e-8)
    rgb_out = torch.clamp(rgb_out, 0, 1)

    # Fill background with gray where no hits
    bg_mask = alpha_accum < 0.01
    rgb_out[bg_mask] = 0.5

    # Reshape outputs
    rgb_out = rgb_out.permute(0, 3, 1, 2)  # (B, 3, H, W)
    depth_out = depth_buffer.unsqueeze(1)  # (B, 1, H, W)
    node_index_out = node_index_buffer.unsqueeze(1)  # (B, 1, H, W)

    if is_scalar:
        rgb_out = rgb_out.squeeze(0)
        depth_out = depth_out.squeeze(0)
        node_index_out = node_index_out.squeeze(0)

    return {
        "rgb": rgb_out,
        "depth": depth_out,
        "node_index": node_index_out,
        "alpha": alpha_accum.unsqueeze(1) if not is_scalar else alpha_accum.unsqueeze(0),
    }


def render_node_mask(
    scene: "NAGScene",
    camera: CameraParams,
    t: torch.Tensor,
    node_id: NAGNodeId,
) -> torch.Tensor:
    """
    Render a binary mask for a single node.

    Args:
        scene: NAGScene
        camera: Camera parameters
        t: Time to render
        node_id: Node to render mask for

    Returns:
        (1, H, W) binary mask
    """
    result = render_scene(scene, camera, t, include_nodes=[node_id])
    return (result["alpha"] > 0.5).float()
