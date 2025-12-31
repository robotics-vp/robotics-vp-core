"""
Integration between NAG and LSD Vector Scene backend.

Provides functions to:
- Build NAGScene from LSD rollout data
- Generate counterfactuals using NAG edits
- Package edited clips into datapacks

Note on rendering:
    The render_lsd_episode_frames() function is currently a stub that generates
    synthetic frames. For production use, either:
    1. Pass pre_rendered_frames directly to build_nag_scene_from_lsd_rollout()
    2. Implement a real GaussianRenderer and hook it up
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    TORCH_AVAILABLE = False

from src.vision.nag.types import (
    CameraParams,
    NAGEditVector,
    NAGNodeId,
    make_node_id,
)

logger = logging.getLogger(__name__)


def _check_torch() -> None:
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for NAG-LSD integration")


@dataclass
class NAGFromLSDConfig:
    """Configuration for building NAGScene from LSD rollout.

    Attributes:
        atlas_size: Resolution of per-node atlas textures
        max_iters: Maximum fitting iterations
        lr: Learning rate for fitting
        max_nodes: Maximum number of foreground nodes
        default_depth: Default depth for objects
        background_depth: Depth for background plane
        image_size: (H, W) resolution for rendering
        fov_deg: Field of view in degrees
        num_camera_views: Number of camera views (for multi-view fitting)
        interesting_classes: Object classes to include as nodes
        use_stub_renderer: If True, use synthetic frames (for testing)
    """
    atlas_size: Tuple[int, int] = (256, 256)
    max_iters: int = 200
    lr: float = 1e-3
    max_nodes: int = 8
    default_depth: float = 5.0
    background_depth: float = 20.0
    image_size: Tuple[int, int] = (256, 256)
    fov_deg: float = 60.0
    num_camera_views: int = 1
    interesting_classes: List[str] = field(
        default_factory=lambda: ["HUMAN", "ROBOT", "FORKLIFT", "PALLET"]
    )
    use_stub_renderer: bool = True  # Set to False when real renderer available


@dataclass
class NAGEditPolicyConfig:
    """Configuration for NAG edit policy.

    Attributes:
        num_counterfactuals: Number of counterfactual clips to generate
        prob_remove: Probability of removing an object
        prob_duplicate: Probability of duplicating an object
        prob_pose_shift: Probability of shifting object pose
        prob_color_shift: Probability of color adjustment
        translation_range: Range for random translation offsets
        rotation_range: Range for random rotation offsets (radians)
        brightness_range: Range for brightness shift
        saturation_range: Range for saturation scale
        max_edits_per_counterfactual: Maximum edits per counterfactual
    """
    num_counterfactuals: int = 3
    prob_remove: float = 0.15
    prob_duplicate: float = 0.2
    prob_pose_shift: float = 0.35
    prob_color_shift: float = 0.2
    translation_range: Tuple[float, float] = (-1.5, 1.5)
    rotation_range: Tuple[float, float] = (-0.3, 0.3)
    brightness_range: Tuple[float, float] = (-0.15, 0.15)
    saturation_range: Tuple[float, float] = (0.7, 1.3)
    max_edits_per_counterfactual: int = 3


@dataclass
class NAGDatapack:
    """Datapack structure for NAG-edited episodes.

    Attributes:
        base_episode_id: ID of the source episode
        counterfactual_id: Unique ID for this counterfactual
        frames: (T, 3, H, W) RGB frames in [0, 1]
        depth_maps: Optional (T, H, W) depth maps
        segmentation: Optional (T, H, W) segmentation masks
        nag_edit_vector: List of NAGEditVector dicts describing edits
        difficulty_features: Dict of difficulty metrics
        lsd_metadata: Additional metadata from LSD backend
    """
    base_episode_id: str
    counterfactual_id: str
    frames: np.ndarray  # (T, 3, H, W) float32 in [0, 1]
    depth_maps: Optional[np.ndarray] = None
    segmentation: Optional[np.ndarray] = None
    nag_edit_vector: List[Dict[str, Any]] = field(default_factory=list)
    difficulty_features: Dict[str, float] = field(default_factory=dict)
    lsd_metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Validate frame shape
        if self.frames.ndim != 4:
            raise ValueError(f"frames must be 4D (T, 3, H, W), got shape {self.frames.shape}")
        if self.frames.shape[1] != 3:
            raise ValueError(f"frames must have 3 channels, got {self.frames.shape[1]}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "base_episode_id": self.base_episode_id,
            "counterfactual_id": self.counterfactual_id,
            "frames_shape": list(self.frames.shape),
            "has_depth": self.depth_maps is not None,
            "has_segmentation": self.segmentation is not None,
            "nag_edit_vector": self.nag_edit_vector,
            "difficulty_features": self.difficulty_features,
            "lsd_metadata": self.lsd_metadata,
            "num_edits": len(self.nag_edit_vector),
            "edit_types": [e.get("edit_type", "unknown") for e in self.nag_edit_vector],
        }


def render_lsd_episode_frames(
    gaussian_scene: Any,
    camera_params: CameraParams,
    num_frames: int = 10,
    device: Optional["torch.device"] = None,
) -> "torch.Tensor":
    """
    Render RGB frames from a GaussianScene.

    NOTE: This is a STUB implementation that generates synthetic frames.
    For production, use pre_rendered_frames parameter in build_nag_scene_from_lsd_rollout()
    or implement a proper GaussianRenderer.

    Args:
        gaussian_scene: GaussianScene from LSD backend (may be None for stub)
        camera_params: Camera parameters
        num_frames: Number of frames to render
        device: Torch device (respects incoming tensor devices)

    Returns:
        (T, 3, H, W) tensor of RGB frames in [0, 1]
    """
    _check_torch()

    device = device or torch.device("cpu")
    H, W = camera_params.height, camera_params.width

    logger.debug(f"render_lsd_episode_frames: stub mode, generating {num_frames} synthetic frames")

    # Stub: generate frames based on Gaussian positions if available
    if gaussian_scene is not None and hasattr(gaussian_scene, "means"):
        means = gaussian_scene.means
        colors = gaussian_scene.colors

        frames = []
        for t in range(num_frames):
            # Simple projection of Gaussians to image
            frame = torch.ones(3, H, W, device=device, dtype=torch.float32) * 0.5

            # Project each Gaussian (simplified orthographic projection)
            for i in range(min(len(means), 100)):
                pos = means[i]
                col = colors[i]

                px = int((pos[0] / 20 + 0.5) * W)
                py = int((pos[1] / 20 + 0.5) * H)

                if 0 <= px < W and 0 <= py < H:
                    r = 5
                    for dx in range(-r, r + 1):
                        for dy in range(-r, r + 1):
                            if dx * dx + dy * dy <= r * r:
                                npx, npy = px + dx, py + dy
                                if 0 <= npx < W and 0 <= npy < H:
                                    frame[:, npy, npx] = torch.tensor(
                                        col, device=device, dtype=torch.float32
                                    )

            frames.append(frame)

        return torch.stack(frames, dim=0)

    # Fallback: gradient noise frames (more visually distinct than pure noise)
    frames = torch.rand(num_frames, 3, H, W, device=device, dtype=torch.float32) * 0.5 + 0.25
    return frames


def extract_object_masks_from_scene_graph(
    scene_graph: Any,
    camera_params: CameraParams,
    num_frames: int,
    config: NAGFromLSDConfig,
) -> Tuple[Dict[NAGNodeId, "torch.Tensor"], Dict[NAGNodeId, "torch.Tensor"]]:
    """
    Extract per-object masks and boxes from scene graph.

    Uses simple projection of scene objects to generate masks.
    Logs which objects are included/excluded.

    Args:
        scene_graph: SceneGraph from LSD backend
        camera_params: Camera parameters
        num_frames: Number of frames
        config: NAG configuration

    Returns:
        Tuple of (masks dict, boxes dict) where:
            - masks[node_id] is (T, 1, H, W) tensor
            - boxes[node_id] is (T, 4) tensor of xyxy boxes
    """
    _check_torch()

    H, W = camera_params.height, camera_params.width
    device = torch.device("cpu")

    masks: Dict[NAGNodeId, torch.Tensor] = {}
    boxes: Dict[NAGNodeId, torch.Tensor] = {}

    if scene_graph is None or not hasattr(scene_graph, "objects"):
        logger.debug("extract_object_masks: no scene_graph or no objects attribute")
        return masks, boxes

    # Filter to interesting objects
    interesting_objects = []
    excluded_classes: Dict[str, int] = {}

    for obj in scene_graph.objects:
        class_name = str(obj.class_id.name) if hasattr(obj.class_id, "name") else str(obj.class_id)
        if class_name in config.interesting_classes:
            interesting_objects.append(obj)
        else:
            excluded_classes[class_name] = excluded_classes.get(class_name, 0) + 1

    # Log exclusions
    if excluded_classes:
        logger.debug(f"extract_object_masks: excluded classes: {excluded_classes}")

    # Limit to max_nodes
    if len(interesting_objects) > config.max_nodes:
        logger.info(
            f"extract_object_masks: limiting from {len(interesting_objects)} to {config.max_nodes} objects"
        )
        interesting_objects = interesting_objects[: config.max_nodes]

    logger.debug(f"extract_object_masks: including {len(interesting_objects)} objects")

    for obj in interesting_objects:
        node_id = make_node_id(f"obj_{obj.id}")

        # Simple projection to image coordinates
        bbox = scene_graph.bounding_box() if hasattr(scene_graph, "bounding_box") else (0, 0, 20, 20)
        scale_x = W / (bbox[2] - bbox[0] + 1)
        scale_y = H / (bbox[3] - bbox[1] + 1)

        cx = (obj.x - bbox[0]) * scale_x
        cy = (obj.y - bbox[1]) * scale_y

        obj_w = obj.length * scale_x * 0.5
        obj_h = obj.width * scale_y * 0.5

        # Create box sequence (with simple motion)
        frame_boxes = []
        frame_masks = []

        for t in range(num_frames):
            motion_factor = t / max(num_frames - 1, 1)
            dx = obj.speed * np.cos(obj.heading) * motion_factor * scale_x
            dy = obj.speed * np.sin(obj.heading) * motion_factor * scale_y

            x1 = max(0, cx + dx - obj_w)
            y1 = max(0, cy + dy - obj_h)
            x2 = min(W, cx + dx + obj_w)
            y2 = min(H, cy + dy + obj_h)

            frame_boxes.append([x1, y1, x2, y2])

            # Create mask
            mask = torch.zeros(1, H, W, device=device, dtype=torch.float32)
            x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
            if x2i > x1i and y2i > y1i:
                mask[0, y1i:y2i, x1i:x2i] = 1.0

            frame_masks.append(mask)

        masks[node_id] = torch.stack(frame_masks, dim=0)  # (T, 1, H, W)
        boxes[node_id] = torch.tensor(frame_boxes, dtype=torch.float32, device=device)

    return masks, boxes


def build_nag_scene_from_lsd_rollout(
    backend_episode: Dict[str, Any],
    camera: CameraParams,
    config: NAGFromLSDConfig,
    device: Optional["torch.device"] = None,
    pre_rendered_frames: Optional["torch.Tensor"] = None,
) -> "NAGScene":
    """
    Build a NAGScene from an LSD vector scene rollout.

    Args:
        backend_episode: Episode data from LSD backend containing:
            - gaussian_scene: GaussianScene (optional if pre_rendered_frames provided)
            - scene_graph: SceneGraph for object detection
            - num_frames: Number of frames in episode
            - episode_id: Unique episode identifier
            - difficulty_features: Dict of difficulty metrics
        camera: Camera parameters
        config: NAG configuration
        device: Torch device (defaults to CPU)
        pre_rendered_frames: Optional (T, 3, H, W) pre-rendered frames.
            If provided, skips stub renderer and uses these directly.

    Returns:
        Fitted NAGScene

    Raises:
        RuntimeError: If NAG fitting fails critically
    """
    _check_torch()

    from src.vision.nag.fitter import fit_nag_to_sequence, FitterConfig
    from src.vision.nag.scene import NAGScene, NAGSceneConfig, create_scene_with_background

    device = device or torch.device("cpu")

    # Extract components from episode
    gaussian_scene = backend_episode.get("gaussian_scene")
    scene_graph = backend_episode.get("scene_graph")
    num_frames = backend_episode.get("num_frames", 10)
    episode_id = backend_episode.get("episode_id", "unknown")

    logger.debug(f"build_nag_scene: episode={episode_id}, num_frames={num_frames}")

    # Get frames - either pre-rendered or from stub
    if pre_rendered_frames is not None:
        frames = pre_rendered_frames.to(device)
        if frames.shape[0] != num_frames:
            logger.warning(
                f"pre_rendered_frames has {frames.shape[0]} frames, expected {num_frames}"
            )
            num_frames = frames.shape[0]
    else:
        if not config.use_stub_renderer and gaussian_scene is None:
            logger.warning(
                "build_nag_scene: use_stub_renderer=False but no gaussian_scene; using stub anyway"
            )
        frames = render_lsd_episode_frames(
            gaussian_scene, camera, num_frames=num_frames, device=device
        )

    # Validate frame shape
    assert frames.dim() == 4, f"frames must be 4D, got {frames.dim()}D"
    assert frames.shape[1] == 3, f"frames must have 3 channels, got {frames.shape[1]}"

    # Extract object masks and boxes
    masks, boxes = extract_object_masks_from_scene_graph(
        scene_graph, camera, num_frames, config
    )

    if not masks:
        # No objects detected - return background-only scene
        logger.info(f"build_nag_scene: no foreground objects, creating background-only scene")
        bg_image = frames[num_frames // 2]
        scene = create_scene_with_background(
            bg_image, camera, NAGSceneConfig(atlas_size=config.atlas_size)
        )
        scene.metadata["lsd_episode_id"] = episode_id
        scene.metadata["source"] = "lsd_vector_scene"
        scene.metadata["foreground_nodes"] = 0
        return scene

    # Fit NAG scene
    fitter_config = FitterConfig(
        max_iters=config.max_iters,
        lr=config.lr,
        atlas_size=config.atlas_size,
        default_depth=config.default_depth,
        background_depth=config.background_depth,
    )

    try:
        scene = fit_nag_to_sequence(
            frames=frames,
            masks=masks,
            boxes=boxes,
            camera=camera,
            device=device,
            config=fitter_config,
            verbose=False,
        )
    except Exception as e:
        logger.error(f"build_nag_scene: fitting failed: {e}")
        # Fall back to background-only scene
        bg_image = frames[num_frames // 2]
        scene = create_scene_with_background(
            bg_image, camera, NAGSceneConfig(atlas_size=config.atlas_size)
        )
        scene.metadata["fit_error"] = str(e)

    scene.metadata["lsd_episode_id"] = episode_id
    scene.metadata["source"] = "lsd_vector_scene"
    scene.metadata["foreground_nodes"] = len(masks)

    return scene


def generate_nag_counterfactuals_for_lsd_episode(
    backend_episode: Dict[str, Any],
    camera: CameraParams,
    nag_config: NAGFromLSDConfig,
    edit_config: NAGEditPolicyConfig,
    device: Optional["torch.device"] = None,
    pre_rendered_frames: Optional["torch.Tensor"] = None,
) -> List[NAGDatapack]:
    """
    Generate NAG counterfactuals for an LSD episode.

    Args:
        backend_episode: Episode data from LSD backend
        camera: Camera parameters
        nag_config: NAG scene configuration
        edit_config: Edit policy configuration
        device: Torch device
        pre_rendered_frames: Optional pre-rendered frames (bypasses stub renderer)

    Returns:
        List of NAGDatapack with edited clips. Each contains:
            - frames: (T, 3, H, W) rendered counterfactual
            - nag_edit_vector: List of NAGEditVector dicts
            - difficulty_features: Updated difficulty metrics
            - lsd_metadata: Source episode metadata
    """
    _check_torch()

    from src.vision.nag.editor import NAGEditPolicy, apply_random_edits, render_clip

    device = device or torch.device("cpu")
    rng = np.random.default_rng()

    base_episode_id = backend_episode.get("episode_id", str(uuid.uuid4())[:8])
    num_frames = backend_episode.get("num_frames", 10)

    logger.debug(
        f"generate_nag_counterfactuals: episode={base_episode_id}, "
        f"num_counterfactuals={edit_config.num_counterfactuals}"
    )

    # Build base NAG scene
    try:
        base_scene = build_nag_scene_from_lsd_rollout(
            backend_episode, camera, nag_config, device, pre_rendered_frames
        )
    except Exception as e:
        logger.error(f"generate_nag_counterfactuals: failed to build base scene: {e}")
        return []

    times = torch.linspace(0, 1, num_frames, device=device)

    # Create edit policy
    policy = NAGEditPolicy(
        prob_remove=edit_config.prob_remove,
        prob_duplicate=edit_config.prob_duplicate,
        prob_pose_shift=edit_config.prob_pose_shift,
        prob_color_shift=edit_config.prob_color_shift,
        translation_range=edit_config.translation_range,
        rotation_range=edit_config.rotation_range,
        brightness_range=edit_config.brightness_range,
        saturation_range=edit_config.saturation_range,
    )

    datapacks: List[NAGDatapack] = []

    for cf_idx in range(edit_config.num_counterfactuals):
        try:
            # Clone scene for editing
            edited_scene = base_scene.clone()
            edited_scene = edited_scene.to(device)

            # Apply random edits
            edits: List[NAGEditVector] = apply_random_edits(
                edited_scene, policy, rng, max_edits=edit_config.max_edits_per_counterfactual
            )

            # Render edited clip
            edited_frames = render_clip(edited_scene, camera, times)
            edited_frames_np = edited_frames.detach().cpu().numpy().astype(np.float32)

            # Validate output shape
            assert edited_frames_np.shape == (num_frames, 3, camera.height, camera.width), (
                f"Unexpected frame shape: {edited_frames_np.shape}"
            )

            # Create datapack
            counterfactual_id = f"{base_episode_id}_cf{cf_idx}"

            # Compute edit-adjusted difficulty
            difficulty = backend_episode.get("difficulty_features", {}).copy()

            num_removed = sum(1 for e in edits if e.edit_type == "remove")
            num_duplicated = sum(1 for e in edits if e.edit_type == "duplicate")
            num_pose_shifts = sum(1 for e in edits if e.edit_type == "pose")
            num_color_shifts = sum(1 for e in edits if e.edit_type == "color_shift")

            difficulty["nag_objects_removed"] = float(num_removed)
            difficulty["nag_objects_added"] = float(num_duplicated)
            difficulty["nag_pose_perturbations"] = float(num_pose_shifts)
            difficulty["nag_color_shifts"] = float(num_color_shifts)
            difficulty["nag_total_edits"] = float(len(edits))

            datapack = NAGDatapack(
                base_episode_id=base_episode_id,
                counterfactual_id=counterfactual_id,
                frames=edited_frames_np,
                nag_edit_vector=[e.to_dict() for e in edits],
                difficulty_features=difficulty,
                lsd_metadata={
                    "scene_id": backend_episode.get("scene_id", "unknown"),
                    "scene_graph_config": backend_episode.get("scene_graph_config", {}),
                    "base_mpl": backend_episode.get("mpl_metrics", {}).get("mpl_units_per_hour", 0),
                },
            )

            datapacks.append(datapack)

        except Exception as e:
            logger.warning(f"generate_nag_counterfactuals: counterfactual {cf_idx} failed: {e}")
            continue

    logger.debug(f"generate_nag_counterfactuals: generated {len(datapacks)} datapacks")
    return datapacks


def create_camera_from_lsd_config(
    config: NAGFromLSDConfig,
    scene_bbox: Optional[Tuple[float, float, float, float]] = None,
) -> CameraParams:
    """
    Create camera parameters suitable for rendering LSD scenes.

    Args:
        config: NAG configuration
        scene_bbox: Optional (min_x, min_y, max_x, max_y) bounding box

    Returns:
        CameraParams for rendering
    """
    H, W = config.image_size
    fov = config.fov_deg

    # Default camera position looking down at scene
    if scene_bbox is not None:
        cx = (scene_bbox[0] + scene_bbox[2]) / 2
        cy = (scene_bbox[1] + scene_bbox[3]) / 2
        scene_size = max(scene_bbox[2] - scene_bbox[0], scene_bbox[3] - scene_bbox[1])
        cam_height = scene_size * 1.5
    else:
        cx, cy = 10.0, 10.0
        cam_height = 15.0

    # Camera looking down at scene center
    cam_pos = np.array([cx, cy, cam_height], dtype=np.float32)
    look_at = np.array([cx, cy, 0], dtype=np.float32)
    up = np.array([0, 1, 0], dtype=np.float32)

    return CameraParams.from_camera_rig(
        positions=cam_pos[np.newaxis, :],
        look_at=look_at[np.newaxis, :],
        up=up[np.newaxis, :],
        fov=fov,
        width=W,
        height=H,
    )
