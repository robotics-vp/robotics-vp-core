"""
NAGPlaneNode: A moving plane with a neural atlas.

Represents one object/region in the NAG decomposition as a textured plane
with time-varying pose and view-dependent appearance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore
    TORCH_AVAILABLE = False

from src.vision.nag.types import NAGNodeId, PlaneParams, PoseSplineParams


def _check_torch() -> None:
    """Raise ImportError if torch is not available."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for NAGPlaneNode")


class SmallMLP(nn.Module):
    """Small MLP for atlas queries."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        activation: str = "relu",
    ):
        super().__init__()
        _check_torch()

        layers = []
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # No activation on last layer
                if activation == "relu":
                    layers.append(nn.ReLU(inplace=True))
                elif activation == "gelu":
                    layers.append(nn.GELU())

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NAGPlaneNode(nn.Module):
    """
    A plane node with neural atlas for NAG representation.

    Combines:
    - PlaneParams: static plane geometry (extent, reference pose)
    - PoseSplineParams: time-varying pose as trainable parameters
    - Neural texture: base_texture + color_mlp residual
    - Alpha MLP: view/time-dependent opacity
    - Flow MLP: atlas-space motion

    Attributes:
        node_id: Unique identifier for this node
        plane_params: Static plane geometry
        pose_spline: Time-varying pose (parameters are trainable)
        atlas_size: Resolution of the atlas texture
        base_texture: (3, H, W) base RGB texture
        base_alpha: (1, H, W) base alpha mask
    """

    def __init__(
        self,
        node_id: NAGNodeId,
        plane_params: PlaneParams,
        pose_spline: PoseSplineParams,
        atlas_size: Tuple[int, int] = (256, 256),
        hidden_dim: int = 32,
    ):
        super().__init__()
        _check_torch()

        self.node_id = node_id
        self._plane_params = plane_params
        self._pose_spline_data = pose_spline
        self.atlas_size = atlas_size

        H, W = atlas_size

        # Trainable pose spline parameters
        self.spline_translations = nn.Parameter(
            torch.from_numpy(pose_spline.translations.copy())
        )
        self.spline_euler_angles = nn.Parameter(
            torch.from_numpy(pose_spline.euler_angles.copy())
        )
        self.register_buffer(
            "spline_knot_times",
            torch.from_numpy(pose_spline.knot_times.copy()),
        )

        # Base texture (initialized to gray, will be set from image)
        self.base_texture = nn.Parameter(
            torch.ones(3, H, W) * 0.5
        )

        # Base alpha (initialized to opaque)
        self.base_alpha = nn.Parameter(
            torch.ones(1, H, W)
        )

        # Color residual MLP: (u, v, t, view_dir) -> RGB residual
        # Input: 2 (uv) + 1 (t) + 3 (view_dir) = 6
        self.color_mlp = SmallMLP(
            in_dim=6,
            out_dim=3,
            hidden_dim=hidden_dim,
            num_layers=2,
        )

        # Alpha MLP: (u, v, t, view_dir) -> alpha
        self.alpha_mlp = SmallMLP(
            in_dim=6,
            out_dim=1,
            hidden_dim=hidden_dim,
            num_layers=2,
        )

        # Flow MLP: (u, v, t) -> (du, dv)
        self.flow_mlp = SmallMLP(
            in_dim=3,
            out_dim=2,
            hidden_dim=hidden_dim,
            num_layers=2,
        )

    @property
    def plane_params(self) -> PlaneParams:
        """Get the static plane parameters."""
        return self._plane_params

    @property
    def extent(self) -> np.ndarray:
        """Get plane extent."""
        return self._plane_params.extent

    def get_pose_spline(self) -> PoseSplineParams:
        """Get current pose spline with updated parameters."""
        return PoseSplineParams(
            knot_times=self.spline_knot_times.detach().cpu().numpy(),
            translations=self.spline_translations.detach().cpu().numpy(),
            euler_angles=self.spline_euler_angles.detach().cpu().numpy(),
        )

    def pose_at(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute 4x4 world_from_plane transform at time t.

        Args:
            t: Scalar or (B,) tensor of times

        Returns:
            (4, 4) or (B, 4, 4) transform matrices
        """
        is_scalar = t.dim() == 0
        if is_scalar:
            t = t.unsqueeze(0)

        B = t.shape[0]
        device = t.device

        # Clamp t to valid range
        t_min = self.spline_knot_times[0]
        t_max = self.spline_knot_times[-1]
        t_clamped = torch.clamp(t, t_min, t_max)

        # Find segment indices
        # Note: simple linear interpolation for now
        idx = torch.searchsorted(self.spline_knot_times, t_clamped, right=True) - 1
        idx = torch.clamp(idx, 0, len(self.spline_knot_times) - 2)

        t0 = self.spline_knot_times[idx]
        t1 = self.spline_knot_times[idx + 1]
        alpha = (t_clamped - t0) / (t1 - t0 + 1e-8)
        alpha = torch.clamp(alpha, 0, 1).unsqueeze(-1)

        # Interpolate translation and Euler angles
        trans = (1 - alpha) * self.spline_translations[idx] + alpha * self.spline_translations[idx + 1]
        euler = (1 - alpha) * self.spline_euler_angles[idx] + alpha * self.spline_euler_angles[idx + 1]

        # Build rotation matrices
        transforms = self._euler_to_matrix_batch(trans, euler)

        if is_scalar:
            return transforms[0]
        return transforms

    def _euler_to_matrix_batch(
        self,
        trans: torch.Tensor,
        euler: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert batch of translations + Euler angles to 4x4 matrices.

        Args:
            trans: (B, 3) translations
            euler: (B, 3) roll, pitch, yaw

        Returns:
            (B, 4, 4) transform matrices
        """
        B = trans.shape[0]
        device = trans.device

        roll = euler[:, 0]
        pitch = euler[:, 1]
        yaw = euler[:, 2]

        cr, sr = torch.cos(roll), torch.sin(roll)
        cp, sp = torch.cos(pitch), torch.sin(pitch)
        cy, sy = torch.cos(yaw), torch.sin(yaw)

        # Build rotation (ZYX order)
        R = torch.zeros(B, 3, 3, device=device, dtype=trans.dtype)
        R[:, 0, 0] = cy * cp
        R[:, 0, 1] = cy * sp * sr - sy * cr
        R[:, 0, 2] = cy * sp * cr + sy * sr
        R[:, 1, 0] = sy * cp
        R[:, 1, 1] = sy * sp * sr + cy * cr
        R[:, 1, 2] = sy * sp * cr - cy * sr
        R[:, 2, 0] = -sp
        R[:, 2, 1] = cp * sr
        R[:, 2, 2] = cp * cr

        # Build 4x4 transform
        T = torch.eye(4, device=device, dtype=trans.dtype).unsqueeze(0).expand(B, -1, -1).clone()
        T[:, :3, :3] = R
        T[:, :3, 3] = trans

        return T

    def sample_atlas(
        self,
        uv: torch.Tensor,
        t: torch.Tensor,
        view_dir: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample the neural atlas at given UV coordinates.

        Args:
            uv: (..., 2) UV coordinates in [0, 1]
            t: (...,) or scalar time values
            view_dir: (..., 3) view directions in plane frame

        Returns:
            rgb: (..., 3) RGB values
            alpha: (...,) alpha values
        """
        shape = uv.shape[:-1]
        device = uv.device

        # Flatten for processing
        uv_flat = uv.reshape(-1, 2)
        N = uv_flat.shape[0]

        # Broadcast t and view_dir
        if t.dim() == 0:
            t_flat = t.expand(N)
        else:
            t_flat = t.reshape(-1)
            if t_flat.shape[0] == 1:
                t_flat = t_flat.expand(N)

        if view_dir.dim() == 1:
            view_dir_flat = view_dir.unsqueeze(0).expand(N, -1)
        else:
            view_dir_flat = view_dir.reshape(-1, 3)
            if view_dir_flat.shape[0] == 1:
                view_dir_flat = view_dir_flat.expand(N, -1)

        # Compute flow to warp UV
        flow_input = torch.cat([uv_flat, t_flat.unsqueeze(-1)], dim=-1)
        flow = self.flow_mlp(flow_input)  # (N, 2)
        uv_warped = uv_flat + flow * 0.1  # Small flow magnitude

        # Clamp to valid range
        uv_warped = torch.clamp(uv_warped, 0, 1)

        # Sample base texture using grid_sample
        # Convert UV to grid coordinates [-1, 1]
        grid = uv_warped * 2 - 1  # (N, 2)
        grid = grid.view(1, 1, N, 2)  # (1, 1, N, 2) for grid_sample

        base_tex = self.base_texture.unsqueeze(0)  # (1, 3, H, W)
        base_rgb = F.grid_sample(
            base_tex, grid, mode="bilinear", padding_mode="border", align_corners=True
        )  # (1, 3, 1, N)
        base_rgb = base_rgb.squeeze(0).squeeze(1).T  # (N, 3)

        base_a = self.base_alpha.unsqueeze(0)  # (1, 1, H, W)
        base_alpha = F.grid_sample(
            base_a, grid, mode="bilinear", padding_mode="border", align_corners=True
        )  # (1, 1, 1, N)
        base_alpha = base_alpha.squeeze()  # (N,)
        if base_alpha.dim() == 0:
            base_alpha = base_alpha.unsqueeze(0)

        # Compute color residual
        mlp_input = torch.cat([uv_flat, t_flat.unsqueeze(-1), view_dir_flat], dim=-1)
        color_residual = self.color_mlp(mlp_input)  # (N, 3)
        rgb = torch.clamp(base_rgb + color_residual * 0.2, 0, 1)

        # Compute alpha
        alpha_logit = self.alpha_mlp(mlp_input).squeeze(-1)  # (N,)
        alpha = torch.sigmoid(alpha_logit) * base_alpha

        # Reshape to original
        rgb = rgb.reshape(*shape, 3)
        alpha = alpha.reshape(*shape)

        return rgb, alpha

    def initialize_from_image(
        self,
        image: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Initialize base texture from an image patch.

        Args:
            image: (3, H, W) or (H, W, 3) RGB image
            mask: (1, H, W) or (H, W) optional alpha mask
        """
        if image.shape[-1] == 3:
            image = image.permute(2, 0, 1)

        # Resize to atlas size
        H, W = self.atlas_size
        image_resized = F.interpolate(
            image.unsqueeze(0),
            size=(H, W),
            mode="bilinear",
            align_corners=True,
        ).squeeze(0)

        self.base_texture.data.copy_(image_resized)

        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)
            mask_resized = F.interpolate(
                mask.unsqueeze(0).float(),
                size=(H, W),
                mode="bilinear",
                align_corners=True,
            ).squeeze(0)
            self.base_alpha.data.copy_(mask_resized)

    def clone(self) -> "NAGPlaneNode":
        """Create a deep copy of this node."""
        new_node = NAGPlaneNode(
            node_id=self.node_id,
            plane_params=PlaneParams(
                world_from_plane=self._plane_params.world_from_plane.copy(),
                extent=self._plane_params.extent.copy(),
            ),
            pose_spline=self.get_pose_spline(),
            atlas_size=self.atlas_size,
        )

        # Copy parameters
        with torch.no_grad():
            new_node.base_texture.data.copy_(self.base_texture.data)
            new_node.base_alpha.data.copy_(self.base_alpha.data)
            new_node.spline_translations.data.copy_(self.spline_translations.data)
            new_node.spline_euler_angles.data.copy_(self.spline_euler_angles.data)

            # Copy MLP weights
            new_node.color_mlp.load_state_dict(self.color_mlp.state_dict())
            new_node.alpha_mlp.load_state_dict(self.alpha_mlp.state_dict())
            new_node.flow_mlp.load_state_dict(self.flow_mlp.state_dict())

        return new_node

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization (excluding neural network weights)."""
        return {
            "node_id": str(self.node_id),
            "plane_params": self._plane_params.to_dict(),
            "pose_spline": self.get_pose_spline().to_dict(),
            "atlas_size": list(self.atlas_size),
            "base_texture": self.base_texture.detach().cpu().numpy().tolist(),
            "base_alpha": self.base_alpha.detach().cpu().numpy().tolist(),
        }


def create_plane_node_from_box(
    node_id: NAGNodeId,
    box: np.ndarray,
    depth: float,
    camera: "CameraParams",
    t_ref: float = 0.0,
    atlas_size: Tuple[int, int] = (256, 256),
) -> NAGPlaneNode:
    """
    Create a NAGPlaneNode from a 2D bounding box.

    Args:
        node_id: Identifier for the node
        box: (4,) xyxy bounding box in image coordinates
        depth: Estimated depth of the object
        camera: Camera parameters
        t_ref: Reference time for initial pose
        atlas_size: Resolution of the atlas texture

    Returns:
        Initialized NAGPlaneNode
    """
    from src.vision.nag.types import CameraParams

    _check_torch()

    x1, y1, x2, y2 = box
    cx_img = (x1 + x2) / 2
    cy_img = (y1 + y2) / 2

    # Backproject box center to 3D
    K_inv = camera.K_inv
    point_cam = K_inv @ np.array([cx_img, cy_img, 1.0])
    point_cam = point_cam * depth

    # Transform to world
    w2c = camera.world_from_cam[0]
    point_world = (w2c[:3, :3] @ point_cam) + w2c[:3, 3]

    # Estimate plane extent from box size at depth
    box_width = x2 - x1
    box_height = y2 - y1
    extent_x = box_width * depth / camera.fx
    extent_y = box_height * depth / camera.fy

    # Create plane facing camera
    cam_pos = w2c[:3, 3]
    normal = cam_pos - point_world
    normal = normal / (np.linalg.norm(normal) + 1e-8)

    plane_params = PlaneParams.create_frontal(
        center=tuple(point_world),
        extent=(extent_x, extent_y),
        normal=tuple(normal),
    )

    # Static pose spline
    pose_spline = PoseSplineParams.create_static(
        translation=tuple(point_world),
        t_range=(0, 1),
    )

    return NAGPlaneNode(
        node_id=node_id,
        plane_params=plane_params,
        pose_spline=pose_spline,
        atlas_size=atlas_size,
    )
