"""
Editor API for NAG scenes.

Provides structured editing operations for generating counterfactuals:
- Texture editing
- Pose editing
- Node duplication
- Clip rendering
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    F = None  # type: ignore
    TORCH_AVAILABLE = False

from src.vision.nag.types import CameraParams, NAGNodeId, NAGEditVector, make_node_id
from src.vision.nag.scene import NAGScene
from src.vision.nag.renderer import render_scene


def _check_torch() -> None:
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for NAG editor")


def edit_texture_from_rgba(
    scene: NAGScene,
    node_id: NAGNodeId,
    rgba_frame: torch.Tensor,
    mask: torch.Tensor,
    camera: CameraParams,
    t_ref: float,
    blend_weight: float = 1.0,
) -> NAGEditVector:
    """
    Project RGBA pixels onto a node's atlas texture.

    Args:
        scene: NAGScene to edit
        node_id: Node to edit
        rgba_frame: (4, H, W) RGBA image in image space
        mask: (1, H, W) edit mask
        camera: Camera parameters
        t_ref: Reference time for projection
        blend_weight: Blending weight for new texture (0-1)

    Returns:
        NAGEditVector describing the edit
    """
    _check_torch()

    node = scene.get_node(node_id)
    device = node.base_texture.device

    rgba_frame = rgba_frame.to(device)
    mask = mask.to(device)

    H, W = camera.height, camera.width
    t_tensor = torch.tensor(t_ref, device=device)

    # Get node pose at reference time
    pose = node.pose_at(t_tensor)  # (4, 4)
    plane_origin = pose[:3, 3]
    plane_normal = pose[:3, 2]

    # Get camera rays
    ray_origins_np, ray_dirs_np = camera.get_rays(t=0)
    ray_origins = torch.from_numpy(ray_origins_np).to(device)
    ray_dirs = torch.from_numpy(ray_dirs_np).to(device)

    # Ray-plane intersection
    from src.vision.nag.renderer import ray_plane_intersection
    hit_t, valid = ray_plane_intersection(ray_origins, ray_dirs, plane_origin, plane_normal)

    # Compute hit points
    hit_points = ray_origins + ray_dirs * hit_t.unsqueeze(-1)

    # Transform to plane-local coordinates
    plane_from_world = torch.inverse(pose)
    hit_points_homo = torch.cat([hit_points, torch.ones_like(hit_points[..., :1])], dim=-1)
    hit_local = torch.einsum("ij,hwj->hwi", plane_from_world, hit_points_homo)[..., :3]

    # Convert to UV
    extent = torch.from_numpy(node.extent).to(device)
    u = hit_local[..., 0] / extent[0] + 0.5
    v = hit_local[..., 1] / extent[1] + 0.5

    # Valid mask: in bounds and masked
    in_bounds = (u >= 0) & (u <= 1) & (v >= 0) & (v <= 1)
    edit_mask = valid & in_bounds & (mask.squeeze(0) > 0.5)

    if not edit_mask.any():
        return NAGEditVector(
            node_id=node_id,
            edit_type="texture",
            parameters={"blend_weight": blend_weight, "num_pixels": 0},
            timestamp=t_ref,
        )

    # Get atlas size
    atlas_H, atlas_W = node.atlas_size

    # For each valid pixel, update the atlas
    with torch.no_grad():
        # Get pixel coordinates in image space
        valid_y, valid_x = torch.where(edit_mask)

        # Get corresponding UV coordinates
        valid_u = u[valid_y, valid_x]
        valid_v = v[valid_y, valid_x]

        # Convert to atlas pixel coordinates
        atlas_x = (valid_u * atlas_W).long().clamp(0, atlas_W - 1)
        atlas_y = (valid_v * atlas_H).long().clamp(0, atlas_H - 1)

        # Get RGBA values from input
        rgb_values = rgba_frame[:3, valid_y, valid_x]  # (3, N)
        alpha_values = rgba_frame[3, valid_y, valid_x]  # (N,)

        # Blend into atlas
        for i in range(len(valid_x)):
            ax, ay = atlas_x[i], atlas_y[i]
            rgb = rgb_values[:, i]
            alpha = alpha_values[i] * blend_weight

            # Alpha blend
            node.base_texture.data[:, ay, ax] = (
                alpha * rgb + (1 - alpha) * node.base_texture.data[:, ay, ax]
            )
            node.base_alpha.data[0, ay, ax] = max(
                node.base_alpha.data[0, ay, ax],
                alpha,
            )

    return NAGEditVector(
        node_id=node_id,
        edit_type="texture",
        parameters={
            "blend_weight": blend_weight,
            "num_pixels": int(edit_mask.sum()),
            "t_ref": t_ref,
        },
        timestamp=t_ref,
    )


def edit_pose(
    scene: NAGScene,
    node_id: NAGNodeId,
    delta_translation: torch.Tensor,
    delta_rotation_euler: torch.Tensor,
    apply_to_all_knots: bool = True,
) -> NAGEditVector:
    """
    Apply pose offset to a node's trajectory.

    Args:
        scene: NAGScene to edit
        node_id: Node to edit
        delta_translation: (3,) translation offset
        delta_rotation_euler: (3,) Euler angle offset (roll, pitch, yaw)
        apply_to_all_knots: If True, apply to all spline knots; otherwise just the first

    Returns:
        NAGEditVector describing the edit
    """
    _check_torch()

    node = scene.get_node(node_id)
    device = node.spline_translations.device

    delta_t = delta_translation.to(device)
    delta_r = delta_rotation_euler.to(device)

    with torch.no_grad():
        if apply_to_all_knots:
            node.spline_translations.data += delta_t.unsqueeze(0)
            node.spline_euler_angles.data += delta_r.unsqueeze(0)
        else:
            node.spline_translations.data[0] += delta_t
            node.spline_euler_angles.data[0] += delta_r

    return NAGEditVector(
        node_id=node_id,
        edit_type="pose",
        parameters={
            "delta_translation": delta_translation.cpu().numpy().tolist(),
            "delta_rotation_euler": delta_rotation_euler.cpu().numpy().tolist(),
            "apply_to_all_knots": apply_to_all_knots,
        },
    )


def duplicate_node(
    scene: NAGScene,
    node_id: NAGNodeId,
    new_id: NAGNodeId,
    pose_offset: Dict[str, torch.Tensor],
) -> NAGEditVector:
    """
    Duplicate a node with a pose offset.

    Args:
        scene: NAGScene to edit
        node_id: Source node to duplicate
        new_id: ID for the new node
        pose_offset: Dict with 'translation' and/or 'euler' offsets

    Returns:
        NAGEditVector describing the edit
    """
    _check_torch()

    # Convert tensors to numpy for scene.clone_node
    offset_np = {}
    if "translation" in pose_offset:
        t = pose_offset["translation"]
        offset_np["translation"] = t.cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)
    if "euler" in pose_offset:
        e = pose_offset["euler"]
        offset_np["euler"] = e.cpu().numpy() if isinstance(e, torch.Tensor) else np.asarray(e)

    scene.clone_node(node_id, new_id, offset_np)

    return NAGEditVector(
        node_id=new_id,
        edit_type="duplicate",
        parameters={
            "source_node_id": str(node_id),
            "translation_offset": offset_np.get("translation", [0, 0, 0]).tolist() if isinstance(offset_np.get("translation"), np.ndarray) else [0, 0, 0],
            "euler_offset": offset_np.get("euler", [0, 0, 0]).tolist() if isinstance(offset_np.get("euler"), np.ndarray) else [0, 0, 0],
        },
    )


def remove_node(
    scene: NAGScene,
    node_id: NAGNodeId,
) -> NAGEditVector:
    """
    Remove a node from the scene.

    Args:
        scene: NAGScene to edit
        node_id: Node to remove

    Returns:
        NAGEditVector describing the edit
    """
    scene.remove_node(node_id)

    return NAGEditVector(
        node_id=node_id,
        edit_type="remove",
        parameters={},
    )


def render_clip(
    scene: NAGScene,
    camera: CameraParams,
    times: torch.Tensor,
) -> torch.Tensor:
    """
    Render a video clip from the scene.

    Args:
        scene: NAGScene to render
        camera: Camera parameters
        times: (T,) tensor of time values

    Returns:
        (T, 3, H, W) RGB clip
    """
    _check_torch()

    device = times.device
    T = times.shape[0]
    H, W = camera.height, camera.width

    frames = []
    for t_idx in range(T):
        t = times[t_idx]
        result = render_scene(scene, camera, t)
        frames.append(result["rgb"])

    return torch.stack(frames, dim=0)


def apply_color_shift(
    scene: NAGScene,
    node_id: NAGNodeId,
    hue_shift: float = 0.0,
    saturation_scale: float = 1.0,
    brightness_shift: float = 0.0,
) -> NAGEditVector:
    """
    Apply a color transformation to a node's texture.

    Args:
        scene: NAGScene to edit
        node_id: Node to edit
        hue_shift: Hue shift in radians (approximate via channel rotation)
        saturation_scale: Saturation multiplier
        brightness_shift: Brightness offset

    Returns:
        NAGEditVector describing the edit
    """
    _check_torch()

    node = scene.get_node(node_id)

    with torch.no_grad():
        tex = node.base_texture.data  # (3, H, W)

        # Simple brightness/contrast adjustment
        tex = tex + brightness_shift
        tex = torch.clamp(tex, 0, 1)

        # Approximate saturation adjustment
        gray = tex.mean(dim=0, keepdim=True)
        tex = gray + saturation_scale * (tex - gray)
        tex = torch.clamp(tex, 0, 1)

        # Approximate hue shift via channel rotation
        if abs(hue_shift) > 0.01:
            cos_h = np.cos(hue_shift)
            sin_h = np.sin(hue_shift)
            # Simple RGB rotation approximation
            r, g, b = tex[0], tex[1], tex[2]
            tex[0] = cos_h * r + sin_h * g
            tex[1] = -sin_h * r + cos_h * g
            tex = torch.clamp(tex, 0, 1)

        node.base_texture.data = tex

    return NAGEditVector(
        node_id=node_id,
        edit_type="color_shift",
        parameters={
            "hue_shift": hue_shift,
            "saturation_scale": saturation_scale,
            "brightness_shift": brightness_shift,
        },
    )


@dataclass
class NAGEditPolicy:
    """Policy for generating random edits.

    Attributes:
        prob_remove: Probability of removing an object
        prob_duplicate: Probability of duplicating an object
        prob_pose_shift: Probability of shifting object pose
        prob_color_shift: Probability of color adjustment
        translation_range: Range for random translation offsets
        rotation_range: Range for random rotation offsets (radians)
        brightness_range: Range for brightness shift
        saturation_range: Range for saturation scale
        hue_range: Range for hue shift (radians)
    """
    prob_remove: float = 0.1
    prob_duplicate: float = 0.2
    prob_pose_shift: float = 0.3
    prob_color_shift: float = 0.2

    # Magnitude ranges
    translation_range: Tuple[float, float] = (-1.0, 1.0)
    rotation_range: Tuple[float, float] = (-0.2, 0.2)
    brightness_range: Tuple[float, float] = (-0.1, 0.1)
    saturation_range: Tuple[float, float] = (0.8, 1.2)
    hue_range: Tuple[float, float] = (-0.1, 0.1)

    @classmethod
    def from_config(cls, config: Any) -> "NAGEditPolicy":
        """Create NAGEditPolicy from NAGEditPolicyConfig.

        Args:
            config: NAGEditPolicyConfig (or any object with matching attributes)

        Returns:
            NAGEditPolicy instance
        """
        return cls(
            prob_remove=getattr(config, "prob_remove", cls.prob_remove),
            prob_duplicate=getattr(config, "prob_duplicate", cls.prob_duplicate),
            prob_pose_shift=getattr(config, "prob_pose_shift", cls.prob_pose_shift),
            prob_color_shift=getattr(config, "prob_color_shift", cls.prob_color_shift),
            translation_range=getattr(config, "translation_range", cls.translation_range),
            rotation_range=getattr(config, "rotation_range", cls.rotation_range),
            brightness_range=getattr(config, "brightness_range", cls.brightness_range),
            saturation_range=getattr(config, "saturation_range", cls.saturation_range),
            hue_range=getattr(config, "hue_range", (-0.1, 0.1)),
        )


def apply_random_edits(
    scene: NAGScene,
    policy: NAGEditPolicy,
    rng: Optional[np.random.Generator] = None,
    max_edits: int = 3,
    seed: Optional[int] = None,
) -> List[NAGEditVector]:
    """
    Apply random edits to a scene according to a policy.

    Args:
        scene: NAGScene to edit (modified in-place)
        policy: Edit policy with probabilities and magnitudes
        rng: Random number generator. If None and seed provided, creates one from seed.
        max_edits: Maximum number of edits to apply
        seed: Optional random seed for reproducible edits. Ignored if rng is provided.

    Returns:
        List of NAGEditVectors describing applied edits

    Note:
        To get deterministic edits, either pass a seeded rng or pass seed parameter.
        Example:
            # Option 1: seeded rng
            rng = np.random.default_rng(42)
            edits = apply_random_edits(scene, policy, rng=rng)

            # Option 2: seed parameter
            edits = apply_random_edits(scene, policy, seed=42)
    """
    _check_torch()

    # Create RNG from seed if not provided
    if rng is None:
        rng = np.random.default_rng(seed)  # seed=None gives non-deterministic

    edits = []

    foreground_nodes = scene.get_foreground_nodes()
    if not foreground_nodes:
        return edits

    for _ in range(max_edits):
        if not foreground_nodes:
            break

        node_id = rng.choice(foreground_nodes)
        edit_roll = rng.random()

        if edit_roll < policy.prob_remove:
            # Remove node
            edit = remove_node(scene, node_id)
            foreground_nodes.remove(node_id)
            edits.append(edit)

        elif edit_roll < policy.prob_remove + policy.prob_duplicate:
            # Duplicate node
            new_id = make_node_id(f"{node_id}_dup_{len(edits)}")
            delta_t = rng.uniform(*policy.translation_range, size=3)
            delta_r = rng.uniform(*policy.rotation_range, size=3)

            edit = duplicate_node(
                scene, node_id, new_id,
                {
                    "translation": torch.from_numpy(delta_t.astype(np.float32)),
                    "euler": torch.from_numpy(delta_r.astype(np.float32)),
                },
            )
            foreground_nodes.append(new_id)
            edits.append(edit)

        elif edit_roll < policy.prob_remove + policy.prob_duplicate + policy.prob_pose_shift:
            # Pose shift
            delta_t = rng.uniform(*policy.translation_range, size=3)
            delta_r = rng.uniform(*policy.rotation_range, size=3)

            edit = edit_pose(
                scene, node_id,
                torch.from_numpy(delta_t.astype(np.float32)),
                torch.from_numpy(delta_r.astype(np.float32)),
            )
            edits.append(edit)

        elif edit_roll < policy.prob_remove + policy.prob_duplicate + policy.prob_pose_shift + policy.prob_color_shift:
            # Color shift with hue
            brightness = rng.uniform(*policy.brightness_range)
            saturation = rng.uniform(*policy.saturation_range)
            hue = rng.uniform(*policy.hue_range)

            edit = apply_color_shift(
                scene, node_id,
                hue_shift=hue,
                brightness_shift=brightness,
                saturation_scale=saturation,
            )
            edits.append(edit)

    return edits
