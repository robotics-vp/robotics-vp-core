"""
Concrete GaussianRenderer implementation for NAG-LSD integration.

Provides a splatting-based renderer that projects 3D Gaussians to 2D images.
This is a simplified implementation suitable for NAG overlay generation.
For production use, consider integrating with gsplat or diff-gaussian-rasterization.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Protocol, Tuple, runtime_checkable

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    F = None  # type: ignore
    TORCH_AVAILABLE = False

from src.envs.lsd3d_env.gaussian_scene import GaussianScene

logger = logging.getLogger(__name__)


def _check_torch() -> None:
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for GaussianRenderer")


@runtime_checkable
class GaussianRendererProtocol(Protocol):
    """Protocol for 3D Gaussian renderers."""

    def render(
        self,
        scene: GaussianScene,
        camera_position: np.ndarray,
        camera_look_at: np.ndarray,
        camera_up: np.ndarray,
        fov: float,
        width: int,
        height: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Render scene from camera viewpoint.

        Returns:
            Tuple of (rgb_image, depth_map) as numpy arrays
        """
        ...


@dataclass
class SplattingRendererConfig:
    """Configuration for splatting-based Gaussian renderer.

    Attributes:
        splat_size_multiplier: Multiplier for Gaussian splat size
        max_gaussians_per_pixel: Maximum Gaussians contributing to a pixel
        depth_tolerance: Tolerance for depth ordering
        background_color: RGB background color
        use_gpu: Whether to use GPU if available
    """
    splat_size_multiplier: float = 2.0
    max_gaussians_per_pixel: int = 32
    depth_tolerance: float = 0.01
    background_color: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    use_gpu: bool = True


class SplattingGaussianRenderer:
    """
    Software splatting renderer for 3D Gaussians.

    Implements a simplified splatting approach that:
    1. Projects Gaussian means to 2D
    2. Computes 2D covariance via Jacobian approximation
    3. Splats each Gaussian with alpha blending
    4. Composites front-to-back by depth

    This is not as fast as CUDA-based rasterizers but provides
    a pure-Python/PyTorch implementation for development and testing.
    """

    def __init__(self, config: Optional[SplattingRendererConfig] = None):
        self.config = config or SplattingRendererConfig()
        self._device: Optional["torch.device"] = None

    @property
    def device(self) -> "torch.device":
        if self._device is None:
            if self.config.use_gpu and TORCH_AVAILABLE and torch.cuda.is_available():
                self._device = torch.device("cuda")
            else:
                self._device = torch.device("cpu")
        return self._device

    def render(
        self,
        scene: GaussianScene,
        camera_position: np.ndarray,
        camera_look_at: np.ndarray,
        camera_up: np.ndarray,
        fov: float,
        width: int,
        height: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Render scene from camera viewpoint.

        Args:
            scene: GaussianScene to render
            camera_position: (3,) camera position
            camera_look_at: (3,) look-at point
            camera_up: (3,) up vector
            fov: Field of view in degrees
            width: Image width
            height: Image height

        Returns:
            Tuple of (rgb, depth) where:
                rgb: (H, W, 3) float32 in [0, 1]
                depth: (H, W) float32 depth values
        """
        _check_torch()

        if scene.num_gaussians == 0:
            rgb = np.full((height, width, 3), self.config.background_color, dtype=np.float32)
            depth = np.full((height, width), np.inf, dtype=np.float32)
            return rgb, depth

        # Build camera matrices
        cam_pos = np.asarray(camera_position, dtype=np.float32)
        look_at = np.asarray(camera_look_at, dtype=np.float32)
        up = np.asarray(camera_up, dtype=np.float32)

        # Camera coordinate system
        forward = look_at - cam_pos
        forward = forward / (np.linalg.norm(forward) + 1e-8)
        right = np.cross(forward, up)
        right = right / (np.linalg.norm(right) + 1e-8)
        up_corrected = np.cross(right, forward)

        # View matrix (world to camera)
        R = np.stack([right, up_corrected, -forward], axis=0)  # (3, 3)
        t = -R @ cam_pos  # (3,)

        # Intrinsics
        fy = height / (2.0 * np.tan(np.radians(fov) / 2.0))
        fx = fy
        cx, cy = width / 2.0, height / 2.0

        # Convert to torch
        device = self.device
        means = torch.from_numpy(scene.means).to(device)  # (N, 3)
        colors = torch.from_numpy(scene.colors).to(device)  # (N, 3)
        opacities = torch.from_numpy(scene.opacities).to(device)  # (N,)
        covs = torch.from_numpy(scene.covs).to(device)  # (N, 6)

        R_t = torch.from_numpy(R).to(device)
        t_t = torch.from_numpy(t).to(device)

        # Transform means to camera space
        means_cam = means @ R_t.T + t_t  # (N, 3)

        # Filter to visible (in front of camera)
        visible_mask = means_cam[:, 2] > 0.1
        if not visible_mask.any():
            rgb = np.full((height, width, 3), self.config.background_color, dtype=np.float32)
            depth = np.full((height, width), np.inf, dtype=np.float32)
            return rgb, depth

        means_cam = means_cam[visible_mask]
        colors_vis = colors[visible_mask]
        opacities_vis = opacities[visible_mask]
        covs_vis = covs[visible_mask]

        # Project to 2D
        depths = means_cam[:, 2]
        x_proj = fx * means_cam[:, 0] / depths + cx
        y_proj = fy * means_cam[:, 1] / depths + cy

        # Compute 2D covariance (simplified - using scales)
        if scene.scales is not None:
            scales_vis = torch.from_numpy(scene.scales[visible_mask.cpu().numpy()]).to(device)
            # Approximate 2D radius from 3D scale
            scale_2d = (scales_vis[:, 0] + scales_vis[:, 1]) / 2.0
            radius_px = fx * scale_2d / depths * self.config.splat_size_multiplier
        else:
            # Use covariance trace as proxy for scale
            cov_trace = covs_vis[:, 0] + covs_vis[:, 3] + covs_vis[:, 5]
            scale_approx = torch.sqrt(cov_trace / 3.0 + 1e-8)
            radius_px = fx * scale_approx / depths * self.config.splat_size_multiplier

        radius_px = torch.clamp(radius_px, 1.0, 100.0)

        # Sort by depth (front to back)
        sort_indices = torch.argsort(depths)
        depths = depths[sort_indices]
        x_proj = x_proj[sort_indices]
        y_proj = y_proj[sort_indices]
        colors_vis = colors_vis[sort_indices]
        opacities_vis = opacities_vis[sort_indices]
        radius_px = radius_px[sort_indices]

        # Initialize output buffers
        rgb_out = torch.full((height, width, 3), self.config.background_color[0],
                             device=device, dtype=torch.float32)
        rgb_out[:, :, 1] = self.config.background_color[1]
        rgb_out[:, :, 2] = self.config.background_color[2]
        depth_out = torch.full((height, width), float("inf"), device=device, dtype=torch.float32)
        alpha_accum = torch.zeros(height, width, device=device, dtype=torch.float32)

        # Splat each Gaussian (vectorized per-Gaussian)
        N = len(depths)
        for i in range(min(N, 1000)):  # Limit for performance
            px, py = x_proj[i].item(), y_proj[i].item()
            r = int(radius_px[i].item())
            d = depths[i].item()
            color = colors_vis[i]
            opacity = opacities_vis[i].item()

            # Bounding box
            x0, x1 = max(0, int(px - r)), min(width, int(px + r + 1))
            y0, y1 = max(0, int(py - r)), min(height, int(py + r + 1))

            if x1 <= x0 or y1 <= y0:
                continue

            # Create Gaussian splat
            yy, xx = torch.meshgrid(
                torch.arange(y0, y1, device=device),
                torch.arange(x0, x1, device=device),
                indexing="ij"
            )
            dist_sq = (xx - px) ** 2 + (yy - py) ** 2
            gauss = torch.exp(-dist_sq / (2 * (r / 2) ** 2 + 1e-8))
            alpha = gauss * opacity

            # Composite (front-to-back)
            remaining = 1.0 - alpha_accum[y0:y1, x0:x1]
            contrib = alpha * remaining

            rgb_out[y0:y1, x0:x1] = (
                rgb_out[y0:y1, x0:x1] * (1 - contrib.unsqueeze(-1)) +
                color.view(1, 1, 3) * contrib.unsqueeze(-1)
            )

            # Update depth (first hit)
            first_hit = (depth_out[y0:y1, x0:x1] == float("inf")) & (alpha > 0.1)
            depth_out[y0:y1, x0:x1] = torch.where(first_hit, d, depth_out[y0:y1, x0:x1])

            alpha_accum[y0:y1, x0:x1] = alpha_accum[y0:y1, x0:x1] + contrib

        # Convert to numpy
        rgb_np = rgb_out.cpu().numpy()
        depth_np = depth_out.cpu().numpy()

        return rgb_np, depth_np

    def render_sequence(
        self,
        scene: GaussianScene,
        camera_positions: np.ndarray,
        camera_look_ats: np.ndarray,
        camera_ups: np.ndarray,
        fov: float,
        width: int,
        height: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Render sequence of frames.

        Args:
            scene: GaussianScene to render
            camera_positions: (T, 3) camera positions
            camera_look_ats: (T, 3) look-at points
            camera_ups: (T, 3) up vectors
            fov: Field of view in degrees
            width: Image width
            height: Image height

        Returns:
            Tuple of (rgb_sequence, depth_sequence) where:
                rgb_sequence: (T, H, W, 3) float32
                depth_sequence: (T, H, W) float32
        """
        T = len(camera_positions)
        rgb_frames = []
        depth_frames = []

        for t in range(T):
            rgb, depth = self.render(
                scene,
                camera_positions[t],
                camera_look_ats[t],
                camera_ups[t],
                fov,
                width,
                height,
            )
            rgb_frames.append(rgb)
            depth_frames.append(depth)

        return np.stack(rgb_frames), np.stack(depth_frames)


def create_default_renderer(use_gpu: bool = True) -> SplattingGaussianRenderer:
    """Create a default Gaussian renderer."""
    return SplattingGaussianRenderer(SplattingRendererConfig(use_gpu=use_gpu))


def render_gaussian_scene_to_frames(
    scene: GaussianScene,
    camera_positions: np.ndarray,
    camera_look_ats: np.ndarray,
    camera_ups: np.ndarray,
    fov: float,
    width: int,
    height: int,
    renderer: Optional[SplattingGaussianRenderer] = None,
) -> "torch.Tensor":
    """
    Convenience function to render GaussianScene to torch frames.

    Args:
        scene: GaussianScene to render
        camera_positions: (T, 3) camera positions
        camera_look_ats: (T, 3) look-at points
        camera_ups: (T, 3) up vectors
        fov: Field of view in degrees
        width: Image width
        height: Image height
        renderer: Optional renderer instance

    Returns:
        (T, 3, H, W) torch tensor of RGB frames in [0, 1]
    """
    _check_torch()

    renderer = renderer or create_default_renderer()
    rgb_np, _ = renderer.render_sequence(
        scene, camera_positions, camera_look_ats, camera_ups, fov, width, height
    )

    # Convert to (T, 3, H, W) torch tensor
    frames = torch.from_numpy(rgb_np).permute(0, 3, 1, 2).contiguous()
    return frames
