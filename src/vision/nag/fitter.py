"""
Fitter for NAGScene from RGB sequences with masks/boxes.

Fits a NAGScene to input video sequences, initializing from detected
objects and optimizing texture + pose parameters.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import Adam
    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore
    Adam = None  # type: ignore
    TORCH_AVAILABLE = False

from src.vision.nag.types import CameraParams, NAGNodeId, PlaneParams, PoseSplineParams, make_node_id
from src.vision.nag.plane_node import NAGPlaneNode, create_plane_node_from_box
from src.vision.nag.scene import NAGScene, NAGSceneConfig, create_scene_with_background
from src.vision.nag.renderer import render_scene

logger = logging.getLogger(__name__)


def _check_torch() -> None:
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for NAG fitter")


@dataclass
class FitterConfig:
    """Configuration for NAG fitting.

    Attributes:
        max_iters: Maximum optimization iterations
        lr: Learning rate for Adam optimizer
        atlas_size: (H, W) resolution for per-node atlas textures
        default_depth: Default depth estimate for objects
        background_depth: Depth for the background plane
        photometric_weight: Weight for L1 photometric loss
        mask_weight: Weight for mask consistency loss
        pose_smoothness_weight: Weight for trajectory smoothness regularization
        flow_regularization_weight: Weight for optical flow regularization (unused)
        batch_frames: Number of frames to sample per iteration
        log_interval: Iterations between log messages
        early_stop_loss: Stop if total loss drops below this threshold
        nan_check_interval: Check for NaN every N iterations
    """
    max_iters: int = 500
    lr: float = 1e-3
    atlas_size: Tuple[int, int] = (256, 256)
    default_depth: float = 5.0
    background_depth: float = 20.0

    # Loss weights
    photometric_weight: float = 1.0
    mask_weight: float = 0.5
    pose_smoothness_weight: float = 0.1
    flow_regularization_weight: float = 0.01

    # Training
    batch_frames: int = 4
    log_interval: int = 50
    early_stop_loss: float = 0.001
    nan_check_interval: int = 25

    def __post_init__(self) -> None:
        # Validate atlas_size
        if not isinstance(self.atlas_size, tuple) or len(self.atlas_size) != 2:
            raise ValueError(f"atlas_size must be (H, W) tuple, got {self.atlas_size}")
        if self.atlas_size[0] < 8 or self.atlas_size[1] < 8:
            raise ValueError(f"atlas_size too small: {self.atlas_size}")
        if self.max_iters < 1:
            raise ValueError(f"max_iters must be positive, got {self.max_iters}")
        if self.lr <= 0:
            raise ValueError(f"lr must be positive, got {self.lr}")


@dataclass
class FitStats:
    """Statistics from NAG fitting.

    Attributes:
        final_loss: Final total loss value
        final_photo_loss: Final photometric loss
        final_mask_loss: Final mask consistency loss
        num_iterations: Number of iterations completed
        converged: Whether fitting converged (early stopped or completed)
        had_nan: Whether NaN was encountered during fitting
        per_node_error: Per-node reconstruction error (if computed)
    """
    final_loss: float
    final_photo_loss: float
    final_mask_loss: float
    num_iterations: int
    converged: bool
    had_nan: bool = False
    per_node_error: Optional[Dict[str, float]] = None


def estimate_depth_from_box(
    box: np.ndarray,
    camera: CameraParams,
    expected_size: float = 1.0,
) -> float:
    """
    Estimate object depth from bounding box size.

    Assumes object has expected_size in world units and uses
    the box size to estimate depth.

    Args:
        box: (4,) xyxy bounding box
        camera: Camera parameters
        expected_size: Expected object size in world units

    Returns:
        Estimated depth
    """
    x1, y1, x2, y2 = box
    box_size = max(x2 - x1, y2 - y1)
    if box_size < 1:
        return 5.0

    # Depth = focal_length * object_size / pixel_size
    depth = camera.fy * expected_size / box_size
    return float(np.clip(depth, 1.0, 50.0))


def initialize_node_from_box_and_mask(
    node_id: NAGNodeId,
    frames: torch.Tensor,
    mask: torch.Tensor,
    box: torch.Tensor,
    camera: CameraParams,
    config: FitterConfig,
    ref_frame: int = 0,
) -> NAGPlaneNode:
    """
    Initialize a NAGPlaneNode from box and mask sequence.

    Args:
        node_id: Node identifier
        frames: (T, 3, H, W) RGB frames
        mask: (T, 1, H, W) object mask
        box: (T, 4) xyxy boxes
        camera: Camera parameters
        config: Fitter configuration
        ref_frame: Reference frame for texture initialization

    Returns:
        Initialized NAGPlaneNode
    """
    _check_torch()

    T = frames.shape[0]
    device = frames.device

    # Estimate depth from reference frame box
    ref_box = box[ref_frame].cpu().numpy()
    depth = estimate_depth_from_box(ref_box, camera)

    # Create node from reference box
    node = create_plane_node_from_box(
        node_id=node_id,
        box=ref_box,
        depth=depth,
        camera=camera,
        t_ref=ref_frame / max(T - 1, 1),
        atlas_size=config.atlas_size,
    )

    # Extract texture from reference frame
    x1, y1, x2, y2 = [int(v) for v in ref_box]
    x1, x2 = max(0, x1), min(camera.width, x2)
    y1, y2 = max(0, y1), min(camera.height, y2)

    if x2 > x1 and y2 > y1:
        ref_image = frames[ref_frame, :, y1:y2, x1:x2]  # (3, h, w)
        ref_mask_patch = mask[ref_frame, :, y1:y2, x1:x2]  # (1, h, w)
        node.initialize_from_image(ref_image, ref_mask_patch)

    # Initialize pose spline from box trajectory
    if T > 1:
        translations = []
        for t_idx in range(T):
            t_box = box[t_idx].cpu().numpy()
            cx = (t_box[0] + t_box[2]) / 2
            cy = (t_box[1] + t_box[3]) / 2

            # Backproject to 3D
            K_inv = camera.K_inv
            point_cam = K_inv @ np.array([cx, cy, 1.0])
            point_cam = point_cam * depth

            w2c = camera.world_from_cam[min(t_idx, camera.num_frames - 1)]
            point_world = (w2c[:3, :3] @ point_cam) + w2c[:3, 3]
            translations.append(point_world)

        translations = np.array(translations, dtype=np.float32)
        knot_times = np.linspace(0, 1, T, dtype=np.float32)

        with torch.no_grad():
            node.spline_translations.data = torch.from_numpy(translations).to(device)
            node.spline_knot_times = torch.from_numpy(knot_times).to(device)
            node.spline_euler_angles.data = torch.zeros(T, 3, device=device)

    return node.to(device)


def create_background_mask(
    masks: Dict[NAGNodeId, torch.Tensor],
) -> torch.Tensor:
    """
    Create background mask by subtracting all foreground masks.

    Args:
        masks: Dict of node_id -> (T, 1, H, W) masks

    Returns:
        (T, 1, H, W) background mask
    """
    if not masks:
        raise ValueError("No masks provided")

    first_mask = next(iter(masks.values()))
    T, _, H, W = first_mask.shape
    device = first_mask.device

    fg_union = torch.zeros(T, 1, H, W, device=device)
    for mask in masks.values():
        fg_union = torch.maximum(fg_union, mask)

    bg_mask = 1.0 - fg_union
    return bg_mask


def _validate_fit_inputs(
    frames: torch.Tensor,
    masks: Dict[NAGNodeId, torch.Tensor],
    boxes: Dict[NAGNodeId, torch.Tensor],
    camera: CameraParams,
) -> None:
    """Validate inputs to fit_nag_to_sequence."""
    # Validate frames
    if frames.dim() != 4:
        raise ValueError(f"frames must be 4D (T, C, H, W), got {frames.dim()}D")
    T, C, H, W = frames.shape
    if C != 3:
        raise ValueError(f"frames must have 3 channels, got {C}")
    if T < 1:
        raise ValueError(f"frames must have at least 1 frame, got {T}")

    # Validate camera matches frame size
    if camera.height != H or camera.width != W:
        raise ValueError(
            f"Camera size ({camera.height}, {camera.width}) doesn't match "
            f"frame size ({H}, {W})"
        )

    # Validate masks and boxes
    for node_id in masks.keys():
        mask = masks[node_id]
        if mask.dim() != 4 or mask.shape[0] != T:
            raise ValueError(
                f"Mask for {node_id} has wrong shape: {mask.shape}, expected (T={T}, 1, H, W)"
            )
        if mask.shape[2] != H or mask.shape[3] != W:
            raise ValueError(
                f"Mask for {node_id} has wrong spatial size: {mask.shape[2:]}, expected ({H}, {W})"
            )

        if node_id not in boxes:
            raise ValueError(f"Missing box for node {node_id}")

        box = boxes[node_id]
        if box.dim() != 2 or box.shape[0] != T or box.shape[1] != 4:
            raise ValueError(
                f"Box for {node_id} has wrong shape: {box.shape}, expected ({T}, 4)"
            )

        # Check for degenerate boxes
        for t in range(T):
            x1, y1, x2, y2 = box[t].tolist()
            if x2 <= x1 or y2 <= y1:
                logger.warning(f"Degenerate box for {node_id} at frame {t}: {box[t].tolist()}")


def fit_nag_to_sequence(
    frames: torch.Tensor,
    masks: Dict[NAGNodeId, torch.Tensor],
    boxes: Dict[NAGNodeId, torch.Tensor],
    camera: CameraParams,
    device: torch.device,
    max_iters: int = 500,
    lr: float = 1e-3,
    config: Optional[FitterConfig] = None,
    verbose: bool = True,
    return_stats: bool = False,
) -> "NAGScene | Tuple[NAGScene, FitStats]":
    """
    Fit a NAGScene to a video sequence with object masks/boxes.

    Args:
        frames: (T, 3, H, W) RGB frames in [0, 1]
        masks: Dict[node_id, (T, 1, H, W)] per-object masks
        boxes: Dict[node_id, (T, 4)] per-object xyxy boxes
        camera: Camera parameters
        device: Torch device
        max_iters: Maximum optimization iterations
        lr: Learning rate
        config: Optional detailed configuration
        verbose: Whether to log progress
        return_stats: If True, return (scene, FitStats) tuple

    Returns:
        Fitted NAGScene, or (NAGScene, FitStats) if return_stats=True

    Raises:
        ValueError: If inputs are invalid
        RuntimeError: If fitting fails catastrophically (NaN explosion)
    """
    _check_torch()

    config = config or FitterConfig(max_iters=max_iters, lr=lr)

    # Validate inputs
    _validate_fit_inputs(frames, masks, boxes, camera)

    frames = frames.to(device)
    T, C, H, W = frames.shape

    # Move masks and boxes to device
    masks = {k: v.to(device) for k, v in masks.items()}
    boxes = {k: v.to(device) for k, v in boxes.items()}

    # Create scene with background
    bg_mask = create_background_mask(masks)
    bg_image = frames[T // 2] * bg_mask[T // 2] + 0.5 * (1 - bg_mask[T // 2])
    scene = create_scene_with_background(
        bg_image,
        camera,
        NAGSceneConfig(
            atlas_size=config.atlas_size,
            background_depth=config.background_depth,
        ),
    )

    # Initialize foreground nodes
    for node_id in masks.keys():
        if node_id == scene.background_node_id:
            continue

        node = initialize_node_from_box_and_mask(
            node_id=node_id,
            frames=frames,
            mask=masks[node_id],
            box=boxes[node_id],
            camera=camera,
            config=config,
            ref_frame=T // 2,
        )
        scene.add_node(node_id, node)

    scene = scene.to(device)

    # Setup optimizer
    optimizer = Adam(scene.parameters(), lr=config.lr)

    # Training loop
    time_steps = torch.linspace(0, 1, T, device=device)

    had_nan = False
    converged = False
    final_photo_loss = 0.0
    final_mask_loss = 0.0
    actual_iters = 0

    for iteration in range(config.max_iters):
        optimizer.zero_grad()

        # Sample batch of frames
        if T <= config.batch_frames:
            batch_indices = list(range(T))
        else:
            batch_indices = np.random.choice(T, config.batch_frames, replace=False).tolist()

        total_loss = torch.tensor(0.0, device=device)
        photo_loss = torch.tensor(0.0, device=device)
        mask_loss = torch.tensor(0.0, device=device)

        for t_idx in batch_indices:
            t = time_steps[t_idx]
            target_frame = frames[t_idx]  # (3, H, W)

            # Render scene
            rendered = render_scene(scene, camera, t)
            rendered_rgb = rendered["rgb"]  # (3, H, W)
            rendered_alpha = rendered["alpha"].squeeze(0)  # (H, W)

            # Photometric loss (L1)
            photo = F.l1_loss(rendered_rgb, target_frame)
            photo_loss = photo_loss + photo

            # Mask consistency loss
            for node_id, gt_mask in masks.items():
                if node_id == scene.background_node_id:
                    continue

                gt_mask_t = gt_mask[t_idx, 0]  # (H, W)

                # Render just this node
                node_rendered = render_scene(scene, camera, t, include_nodes=[node_id])
                node_alpha = node_rendered["alpha"].squeeze(0)  # (H, W)

                # Loss: encourage node alpha to match ground truth mask
                mask_l = F.binary_cross_entropy(
                    torch.clamp(node_alpha, 1e-6, 1 - 1e-6),
                    gt_mask_t,
                )
                mask_loss = mask_loss + mask_l

        # Average over batch
        photo_loss = photo_loss / len(batch_indices)
        mask_loss = mask_loss / (len(batch_indices) * max(len(masks) - 1, 1))

        # Regularization: pose smoothness
        smooth_loss = torch.tensor(0.0, device=device)
        for node_id in scene.get_foreground_nodes():
            node = scene.get_node(node_id)
            if node.spline_translations.shape[0] > 2:
                # Second derivative penalty (acceleration)
                trans = node.spline_translations
                accel = trans[2:] - 2 * trans[1:-1] + trans[:-2]
                smooth_loss = smooth_loss + torch.mean(accel ** 2)

        # Total loss
        total_loss = (
            config.photometric_weight * photo_loss +
            config.mask_weight * mask_loss +
            config.pose_smoothness_weight * smooth_loss
        )

        # Check for NaN
        if (iteration + 1) % config.nan_check_interval == 0 or iteration == 0:
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                logger.error(f"Iter {iteration + 1}: NaN/Inf detected in loss")
                had_nan = True
                break

        total_loss.backward()
        optimizer.step()

        actual_iters = iteration + 1
        final_photo_loss = float(photo_loss.item())
        final_mask_loss = float(mask_loss.item())

        if verbose and (iteration + 1) % config.log_interval == 0:
            logger.info(
                f"Iter {iteration + 1}/{config.max_iters}: "
                f"total={total_loss.item():.4f} "
                f"photo={photo_loss.item():.4f} "
                f"mask={mask_loss.item():.4f}"
            )

        # Early stopping
        if total_loss.item() < config.early_stop_loss:
            logger.info(f"Early stopping at iter {iteration + 1}: loss={total_loss.item():.6f}")
            converged = True
            break

    final_loss_val = float(total_loss.item()) if not had_nan else float("inf")
    scene.metadata["fit_iters"] = actual_iters
    scene.metadata["final_loss"] = final_loss_val
    scene.metadata["converged"] = converged
    scene.metadata["had_nan"] = had_nan

    if return_stats:
        stats = FitStats(
            final_loss=final_loss_val,
            final_photo_loss=final_photo_loss,
            final_mask_loss=final_mask_loss,
            num_iterations=actual_iters,
            converged=converged or (actual_iters == config.max_iters),
            had_nan=had_nan,
        )
        return scene, stats

    return scene


def fit_nag_simple(
    frames: torch.Tensor,
    camera: CameraParams,
    device: torch.device,
    num_objects: int = 3,
    max_iters: int = 200,
) -> NAGScene:
    """
    Simplified NAG fitting without explicit masks/boxes.

    Creates a scene with uniform grid of planes and fits to frames.
    Useful for quick testing or when segmentation is not available.

    Args:
        frames: (T, 3, H, W) RGB frames
        camera: Camera parameters
        device: Torch device
        num_objects: Number of foreground planes to create
        max_iters: Optimization iterations

    Returns:
        Fitted NAGScene
    """
    _check_torch()

    T, C, H, W = frames.shape
    frames = frames.to(device)

    # Create background-only scene
    bg_image = frames[T // 2]
    scene = create_scene_with_background(
        bg_image,
        camera,
        NAGSceneConfig(atlas_size=(128, 128)),
    )

    # Add random foreground planes
    for i in range(num_objects):
        node_id = make_node_id(f"obj_{i}")

        # Random position in view frustum
        x = np.random.uniform(0.2, 0.8) * W
        y = np.random.uniform(0.2, 0.8) * H
        depth = np.random.uniform(3.0, 8.0)

        box = np.array([x - 50, y - 50, x + 50, y + 50], dtype=np.float32)
        node = create_plane_node_from_box(node_id, box, depth, camera)

        # Initialize with image patch
        x1, y1, x2, y2 = [int(v) for v in np.clip(box, 0, [W, H, W, H])]
        if x2 > x1 and y2 > y1:
            patch = frames[T // 2, :, y1:y2, x1:x2]
            node.initialize_from_image(patch)

        scene.add_node(node_id, node.to(device))

    scene = scene.to(device)

    # Quick optimization
    optimizer = Adam(scene.parameters(), lr=1e-3)
    time_steps = torch.linspace(0, 1, T, device=device)

    for iteration in range(max_iters):
        optimizer.zero_grad()

        t_idx = np.random.randint(T)
        t = time_steps[t_idx]

        rendered = render_scene(scene, camera, t)
        loss = F.l1_loss(rendered["rgb"], frames[t_idx])

        loss.backward()
        optimizer.step()

    return scene
