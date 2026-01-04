"""
NAGScene: Container for multiple NAGPlaneNodes.

Groups plane nodes and orchestrates rendering/editing operations.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    nn = None  # type: ignore
    TORCH_AVAILABLE = False

from src.vision.nag.types import CameraParams, NAGNodeId, make_node_id
from src.vision.nag.plane_node import NAGPlaneNode


def _check_torch() -> None:
    """Raise ImportError if torch is not available."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for NAGScene")


class NAGScene(nn.Module):
    """
    Container for NAG plane nodes.

    Manages a collection of NAGPlaneNodes and provides methods for:
    - Adding/removing/cloning nodes
    - Rendering the complete scene
    - Serialization

    Attributes:
        nodes: Dictionary mapping NAGNodeId to NAGPlaneNode
        background_node_id: Optional ID of the background plane
        metadata: Additional scene metadata
    """

    def __init__(
        self,
        background_node_id: Optional[NAGNodeId] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        _check_torch()

        self._nodes: nn.ModuleDict = nn.ModuleDict()
        self.background_node_id = background_node_id
        self.metadata = metadata or {}

    @property
    def nodes(self) -> Dict[NAGNodeId, NAGPlaneNode]:
        """Get dictionary of nodes."""
        return {make_node_id(k): v for k, v in self._nodes.items()}

    def add_node(self, node_id: NAGNodeId, node: NAGPlaneNode) -> None:
        """
        Add a plane node to the scene.

        Args:
            node_id: Unique identifier for the node
            node: NAGPlaneNode to add
        """
        key = str(node_id)
        if key in self._nodes:
            raise ValueError(f"Node {node_id} already exists in scene")
        self._nodes[key] = node

    def get_node(self, node_id: NAGNodeId) -> NAGPlaneNode:
        """
        Get a plane node by ID.

        Args:
            node_id: Node identifier

        Returns:
            The NAGPlaneNode

        Raises:
            KeyError: If node not found
        """
        key = str(node_id)
        if key not in self._nodes:
            raise KeyError(f"Node {node_id} not found in scene")
        return self._nodes[key]

    def has_node(self, node_id: NAGNodeId) -> bool:
        """Check if a node exists in the scene."""
        return str(node_id) in self._nodes

    def remove_node(self, node_id: NAGNodeId) -> NAGPlaneNode:
        """
        Remove a node from the scene.

        Args:
            node_id: Node to remove

        Returns:
            The removed node
        """
        key = str(node_id)
        if key not in self._nodes:
            raise KeyError(f"Node {node_id} not found in scene")
        node = self._nodes[key]
        del self._nodes[key]
        return node

    def list_nodes(self) -> List[NAGNodeId]:
        """Get list of all node IDs."""
        return [make_node_id(k) for k in self._nodes.keys()]

    def num_nodes(self) -> int:
        """Get number of nodes in the scene."""
        return len(self._nodes)

    def clone_node(
        self,
        node_id: NAGNodeId,
        new_id: NAGNodeId,
        pose_offset: Optional[Dict[str, np.ndarray]] = None,
    ) -> NAGPlaneNode:
        """
        Deep-copy a node and add it with a new ID.

        Args:
            node_id: Source node to clone
            new_id: ID for the cloned node
            pose_offset: Optional dict with 'translation' and/or 'euler' offsets

        Returns:
            The newly created node
        """
        source = self.get_node(node_id)
        cloned = source.clone()

        # Update node ID
        cloned.node_id = new_id

        # Apply pose offset
        if pose_offset is not None:
            delta_trans = pose_offset.get("translation", np.zeros(3))
            delta_euler = pose_offset.get("euler", np.zeros(3))

            with torch.no_grad():
                cloned.spline_translations.data += torch.from_numpy(
                    delta_trans.astype(np.float32)
                ).to(cloned.spline_translations.device)
                cloned.spline_euler_angles.data += torch.from_numpy(
                    delta_euler.astype(np.float32)
                ).to(cloned.spline_euler_angles.device)

        self.add_node(new_id, cloned)
        return cloned

    def render(
        self,
        t: torch.Tensor,
        camera: CameraParams,
        include_nodes: Optional[List[NAGNodeId]] = None,
        exclude_nodes: Optional[List[NAGNodeId]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Render the scene at time t.

        This is a convenience wrapper around render_scene().

        Args:
            t: Time(s) to render
            camera: Camera parameters
            include_nodes: If provided, only render these nodes
            exclude_nodes: If provided, exclude these nodes

        Returns:
            Dict with "rgb", optionally "depth" and "node_index"
        """
        from src.vision.nag.renderer import render_scene
        return render_scene(
            scene=self,
            camera=camera,
            t=t,
            include_nodes=include_nodes,
            exclude_nodes=exclude_nodes,
        )

    def get_foreground_nodes(self) -> List[NAGNodeId]:
        """Get all non-background node IDs."""
        return [
            nid for nid in self.list_nodes()
            if nid != self.background_node_id
        ]

    def clone(self) -> "NAGScene":
        """Create a deep copy of the entire scene."""
        new_scene = NAGScene(
            background_node_id=self.background_node_id,
            metadata=copy.deepcopy(self.metadata),
        )

        for node_id, node in self.nodes.items():
            new_scene.add_node(node_id, node.clone())

        return new_scene

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "nodes": {str(nid): node.to_dict() for nid, node in self.nodes.items()},
            "background_node_id": str(self.background_node_id) if self.background_node_id else None,
            "metadata": self.metadata,
        }

    def parameters_by_node(self) -> Dict[NAGNodeId, List[torch.nn.Parameter]]:
        """Get parameters grouped by node for per-node optimization."""
        result = {}
        for node_id, node in self.nodes.items():
            result[node_id] = list(node.parameters())
        return result

    def get_bounding_box(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the world-space bounding box of all nodes.

        Returns:
            (min_corner, max_corner) as (3,) arrays
        """
        if self.num_nodes() == 0:
            return np.zeros(3), np.zeros(3)

        all_points = []
        for node in self._nodes.values():
            # Get corners of the plane
            extent = node.extent
            corners_local = np.array([
                [-extent[0]/2, -extent[1]/2, 0],
                [extent[0]/2, -extent[1]/2, 0],
                [extent[0]/2, extent[1]/2, 0],
                [-extent[0]/2, extent[1]/2, 0],
            ], dtype=np.float32)

            # Transform to world (using t=0)
            pose = node.pose_at(torch.tensor(0.0)).detach().cpu().numpy()
            corners_world = (pose[:3, :3] @ corners_local.T).T + pose[:3, 3]
            all_points.append(corners_world)

        all_points = np.concatenate(all_points, axis=0)
        return all_points.min(axis=0), all_points.max(axis=0)


@dataclass
class NAGSceneConfig:
    """Configuration for NAG scene fitting."""
    atlas_size: Tuple[int, int] = (256, 256)
    max_nodes: int = 10
    background_depth: float = 20.0
    default_object_depth: float = 5.0
    hidden_dim: int = 32


def create_empty_scene(
    config: Optional[NAGSceneConfig] = None,
) -> NAGScene:
    """Create an empty NAG scene."""
    _check_torch()
    return NAGScene(metadata={"config": config.__dict__ if config else {}})


def create_scene_with_background(
    background_image: torch.Tensor,
    camera: CameraParams,
    config: Optional[NAGSceneConfig] = None,
) -> NAGScene:
    """
    Create a NAG scene with a background plane.

    Args:
        background_image: (3, H, W) or (H, W, 3) background image
        camera: Camera parameters
        config: Scene configuration

    Returns:
        NAGScene with background node
    """
    from src.vision.nag.types import PlaneParams, PoseSplineParams

    _check_torch()

    config = config or NAGSceneConfig()

    if background_image.shape[-1] == 3:
        background_image = background_image.permute(2, 0, 1)

    H, W = camera.height, camera.width
    depth = config.background_depth

    # Create a large background plane perpendicular to view direction
    w2c = camera.world_from_cam[0]
    cam_pos = w2c[:3, 3]
    cam_forward = -w2c[:3, 2]  # Camera looks down -Z

    bg_center = cam_pos + cam_forward * depth

    # Compute extent to cover full FOV at depth
    fov_rad = np.radians(camera.fov_deg)
    extent_y = 2 * depth * np.tan(fov_rad / 2) * 1.5  # 1.5x for margin
    extent_x = extent_y * W / H

    plane_params = PlaneParams.create_frontal(
        center=tuple(bg_center),
        extent=(extent_x, extent_y),
        normal=tuple(-cam_forward),  # Face the camera
    )

    pose_spline = PoseSplineParams.create_static(
        translation=tuple(bg_center),
        t_range=(0, 1),
    )

    bg_node_id = make_node_id("background")
    bg_node = NAGPlaneNode(
        node_id=bg_node_id,
        plane_params=plane_params,
        pose_spline=pose_spline,
        atlas_size=config.atlas_size,
    )
    bg_node.initialize_from_image(background_image)

    scene = NAGScene(background_node_id=bg_node_id, metadata={"config": config.__dict__})
    scene.add_node(bg_node_id, bg_node)

    return scene
