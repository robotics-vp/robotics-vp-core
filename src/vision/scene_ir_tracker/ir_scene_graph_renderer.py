"""
IR Scene Graph Renderer.

Provides occlusion-aware differentiable rendering for multiple entities.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
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

from src.vision.nag.types import CameraParams
from src.vision.scene_ir_tracker.types import SceneEntity3D

logger = logging.getLogger(__name__)


def _check_torch() -> None:
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for IRSceneGraphRenderer")


@dataclass
class IRRendererConfig:
    """Configuration for IR scene graph renderer.

    Attributes:
        device: Device for rendering.
        background_color: RGB background color.
        max_entities_per_pixel: Maximum entities contributing to a pixel.
        depth_epsilon: Epsilon for depth comparison.
        alpha_threshold: Minimum alpha for visible contribution.
        use_soft_ordering: Use soft depth ordering (differentiable).
    """

    device: str = "cuda"
    background_color: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    max_entities_per_pixel: int = 8
    depth_epsilon: float = 0.01
    alpha_threshold: float = 0.01
    use_soft_ordering: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "device": self.device,
            "background_color": self.background_color,
            "max_entities_per_pixel": self.max_entities_per_pixel,
            "depth_epsilon": self.depth_epsilon,
            "alpha_threshold": self.alpha_threshold,
            "use_soft_ordering": self.use_soft_ordering,
        }


class IRSceneGraphRenderer:
    """Occlusion-aware differentiable renderer for scene IR tracking.

    Renders multiple entities with proper depth ordering and alpha composition.
    Supports both bodies and objects with different geometry representations.
    """

    def __init__(self, config: Optional[IRRendererConfig] = None):
        """Initialize renderer.

        Args:
            config: Renderer configuration.
        """
        self.config = config or IRRendererConfig()
        self._device: Optional["torch.device"] = None

    @property
    def device(self) -> "torch.device":
        """Get torch device."""
        _check_torch()
        if self._device is None:
            if self.config.device == "cuda" and torch.cuda.is_available():
                self._device = torch.device("cuda")
            else:
                self._device = torch.device("cpu")
        return self._device

    def render_scene(
        self,
        entities: List[SceneEntity3D],
        camera: CameraParams,
        height: Optional[int] = None,
        width: Optional[int] = None,
    ) -> Tuple["torch.Tensor", Dict[str, "torch.Tensor"], "torch.Tensor"]:
        """Render scene with multiple entities.

        Args:
            entities: List of entities to render.
            camera: Camera parameters.
            height: Output height (uses camera height if None).
            width: Output width (uses camera width if None).

        Returns:
            Tuple of:
                - rgb: (3, H, W) rendered RGB image in [0, 1]
                - per_entity_masks: dict of track_id -> (H, W) soft mask
                - depth_order: (H, W) per-pixel depth values
        """
        _check_torch()

        H = height or camera.height
        W = width or camera.width

        if not entities:
            return self._render_empty(H, W)

        # Render each entity
        entity_renders = []
        for entity in entities:
            rgb, alpha, depth = self._render_entity(entity, camera, H, W)
            entity_renders.append({
                "track_id": entity.track_id,
                "rgb": rgb,
                "alpha": alpha,
                "depth": depth,
            })

        # Compose with depth ordering
        composed_rgb, masks, depth_map = self._compose_entities(entity_renders, H, W)

        return composed_rgb, masks, depth_map

    def _render_empty(
        self,
        H: int,
        W: int,
    ) -> Tuple["torch.Tensor", Dict[str, "torch.Tensor"], "torch.Tensor"]:
        """Render empty scene (background only)."""
        bg = torch.tensor(self.config.background_color, device=self.device)
        rgb = bg.view(3, 1, 1).expand(3, H, W)
        depth_map = torch.full((H, W), float("inf"), device=self.device)
        return rgb, {}, depth_map

    def _render_entity(
        self,
        entity: SceneEntity3D,
        camera: CameraParams,
        H: int,
        W: int,
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        """Render single entity.

        Returns (rgb, alpha, depth) each as tensors.
        """
        # Use provided mask if available
        if entity.mask_2d is not None:
            mask_np = entity.mask_2d.astype(np.float32)
            if mask_np.shape != (H, W):
                # Resize mask
                mask_np = np.array(
                    self._resize_mask(mask_np, H, W), dtype=np.float32
                )
            alpha = torch.from_numpy(mask_np).to(self.device)
        else:
            # Create mask from geometry projection
            alpha = self._project_geometry_to_mask(entity, camera, H, W)

        # Create RGB from geometry colors or use placeholder
        rgb = self._get_entity_color(entity, alpha, H, W)

        # Compute depth from entity position
        depth = self._compute_entity_depth(entity, camera, H, W)

        return rgb, alpha, depth

    def _resize_mask(
        self,
        mask: np.ndarray,
        H: int,
        W: int,
    ) -> np.ndarray:
        """Resize mask to target size."""
        if mask.shape == (H, W):
            return mask
        # Simple nearest neighbor resize
        h, w = mask.shape
        y_ratio = h / H
        x_ratio = w / W
        ys = (np.arange(H) * y_ratio).astype(int)
        xs = (np.arange(W) * x_ratio).astype(int)
        ys = np.clip(ys, 0, h - 1)
        xs = np.clip(xs, 0, w - 1)
        return mask[ys[:, None], xs[None, :]]

    def _project_geometry_to_mask(
        self,
        entity: SceneEntity3D,
        camera: CameraParams,
        H: int,
        W: int,
    ) -> "torch.Tensor":
        """Project entity geometry to create a soft mask."""
        # Get entity position in camera frame
        pos_world = entity.position
        pose_inv = np.linalg.inv(camera.world_from_cam[0])  # cam_from_world
        pos_cam = pose_inv[:3, :3] @ pos_world + pose_inv[:3, 3]

        if pos_cam[2] <= 0:
            return torch.zeros((H, W), device=self.device)

        # Project to image
        fx, fy = camera.fx, camera.fy
        cx, cy = camera.cx, camera.cy
        u = fx * pos_cam[0] / pos_cam[2] + cx
        v = fy * pos_cam[1] / pos_cam[2] + cy

        # Create gaussian blob mask
        radius_world = entity.scale * 0.5
        radius_px = fx * radius_world / pos_cam[2]
        radius_px = max(5, min(radius_px, 100))

        yy, xx = torch.meshgrid(
            torch.arange(H, device=self.device),
            torch.arange(W, device=self.device),
            indexing="ij",
        )
        dist_sq = (xx - u) ** 2 + (yy - v) ** 2
        alpha = torch.exp(-dist_sq / (2 * (radius_px / 2) ** 2))
        alpha = torch.clamp(alpha, 0, 1)

        return alpha

    def _get_entity_color(
        self,
        entity: SceneEntity3D,
        alpha: "torch.Tensor",
        H: int,
        W: int,
    ) -> "torch.Tensor":
        """Get RGB color for entity."""
        # Hash track_id for deterministic color
        color_hash = hash(entity.track_id) % (256**3)
        r = ((color_hash >> 16) & 0xFF) / 255.0
        g = ((color_hash >> 8) & 0xFF) / 255.0
        b = (color_hash & 0xFF) / 255.0

        # Ensure minimum brightness
        min_val = min(r, g, b)
        if min_val < 0.2:
            r, g, b = r + 0.3, g + 0.3, b + 0.3

        color = torch.tensor([r, g, b], device=self.device)
        rgb = color.view(3, 1, 1).expand(3, H, W)
        return rgb

    def _compute_entity_depth(
        self,
        entity: SceneEntity3D,
        camera: CameraParams,
        H: int,
        W: int,
    ) -> "torch.Tensor":
        """Compute depth map for entity."""
        pos_world = entity.position
        pose_inv = np.linalg.inv(camera.world_from_cam[0])
        pos_cam = pose_inv[:3, :3] @ pos_world + pose_inv[:3, 3]

        # Uniform depth for the entity (simplified)
        depth = float(max(0.1, pos_cam[2]))
        return torch.full((H, W), depth, device=self.device)

    def _compose_entities(
        self,
        entity_renders: List[Dict[str, Any]],
        H: int,
        W: int,
    ) -> Tuple["torch.Tensor", Dict[str, "torch.Tensor"], "torch.Tensor"]:
        """Compose multiple entity renders with depth ordering.

        Uses front-to-back alpha compositing.
        """
        # Sort by depth (front to back)
        entity_renders = sorted(
            entity_renders,
            key=lambda e: e["depth"].mean().item(),
        )

        # Initialize output
        bg = torch.tensor(self.config.background_color, device=self.device)
        composed_rgb = bg.view(3, 1, 1).expand(3, H, W).clone()
        accumulated_alpha = torch.zeros((H, W), device=self.device)
        depth_map = torch.full((H, W), float("inf"), device=self.device)
        masks = {}

        for er in entity_renders:
            track_id = er["track_id"]
            rgb = er["rgb"]
            alpha = er["alpha"]
            depth = er["depth"]

            # Compute contribution (remaining alpha)
            remaining = 1.0 - accumulated_alpha
            contribution = alpha * remaining

            # Blend RGB
            composed_rgb = composed_rgb * (1 - contribution.unsqueeze(0)) + rgb * contribution.unsqueeze(0)

            # Update depth (first hit)
            first_hit = (depth_map == float("inf")) & (alpha > self.config.alpha_threshold)
            depth_map = torch.where(first_hit, depth, depth_map)

            # Store mask
            masks[track_id] = contribution

            # Accumulate alpha
            accumulated_alpha = accumulated_alpha + contribution

        return composed_rgb, masks, depth_map

    def render_sequence(
        self,
        frames_entities: List[List[SceneEntity3D]],
        camera: CameraParams,
        height: Optional[int] = None,
        width: Optional[int] = None,
    ) -> Tuple["torch.Tensor", List[Dict[str, "torch.Tensor"]], "torch.Tensor"]:
        """Render sequence of frames.

        Args:
            frames_entities: List of entity lists per frame.
            camera: Camera parameters.
            height: Output height.
            width: Output width.

        Returns:
            Tuple of:
                - rgb_sequence: (T, 3, H, W)
                - masks_sequence: list of per-frame mask dicts
                - depth_sequence: (T, H, W)
        """
        rgb_frames = []
        masks_frames = []
        depth_frames = []

        for entities in frames_entities:
            rgb, masks, depth = self.render_scene(entities, camera, height, width)
            rgb_frames.append(rgb)
            masks_frames.append(masks)
            depth_frames.append(depth)

        if rgb_frames:
            rgb_seq = torch.stack(rgb_frames)
            depth_seq = torch.stack(depth_frames)
        else:
            H = height or camera.height
            W = width or camera.width
            rgb_seq = torch.zeros((0, 3, H, W), device=self.device)
            depth_seq = torch.zeros((0, H, W), device=self.device)

        return rgb_seq, masks_frames, depth_seq


def create_ir_renderer(
    config: Optional[Dict[str, Any]] = None,
) -> IRSceneGraphRenderer:
    """Factory function to create IR scene graph renderer.

    Args:
        config: Configuration dict.

    Returns:
        Configured IRSceneGraphRenderer.
    """
    if config:
        cfg = IRRendererConfig(
            device=config.get("device", "cuda"),
            background_color=tuple(config.get("background_color", (0.5, 0.5, 0.5))),
            max_entities_per_pixel=config.get("max_entities_per_pixel", 8),
            depth_epsilon=config.get("depth_epsilon", 0.01),
            alpha_threshold=config.get("alpha_threshold", 0.01),
            use_soft_ordering=config.get("use_soft_ordering", True),
        )
    else:
        cfg = IRRendererConfig()
    return IRSceneGraphRenderer(config=cfg)
