"""
Scene IR Tracker.

Main tracker class orchestrating SAM3D adapters, IR refinement, and Kalman tracking.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    TORCH_AVAILABLE = False

from src.vision.nag.types import CameraParams
from src.vision.scene_ir_tracker.config import SceneIRTrackerConfig
from src.vision.scene_ir_tracker.ir_refiner import IRRefiner, IRRefinementResult
from src.vision.scene_ir_tracker.ir_scene_graph_renderer import IRSceneGraphRenderer
from src.vision.scene_ir_tracker.kalman_track_manager import KalmanTrackManager
from src.vision.scene_ir_tracker.sam3d_body_adapter import (
    SAM3DBodyAdapter,
    SAM3DBodyConfig,
    BodyPrediction,
)
from src.vision.scene_ir_tracker.sam3d_objects_adapter import (
    SAM3DObjectsAdapter,
    SAM3DObjectsConfig,
    ObjectPrediction,
)
from src.vision.scene_ir_tracker.types import (
    SceneEntity3D,
    SceneTracks,
    SceneTrackerMetrics,
)

logger = logging.getLogger(__name__)


def _check_torch() -> None:
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for SceneIRTracker")


class SceneIRTracker:
    """Main Scene IR Tracker orchestrating all components.

    Combines SAM3D-Body and SAM3D-Objects for initial reconstruction,
    IR refinement for parameter optimization, and Kalman tracking for
    stable track IDs across frames.
    """

    def __init__(self, config: Optional[SceneIRTrackerConfig] = None):
        """Initialize tracker.

        Args:
            config: Tracker configuration.
        """
        self.config = config or SceneIRTrackerConfig()
        self._setup_components()

    def _setup_components(self) -> None:
        """Initialize sub-components."""
        # SAM3D adapters
        objects_cfg = SAM3DObjectsConfig.from_dict(self.config.sam3d_objects_config)
        objects_cfg.device = self.config.device
        self.objects_adapter = SAM3DObjectsAdapter(
            config=objects_cfg,
            use_stub=self.config.use_stub_adapters,
        )

        body_cfg = SAM3DBodyConfig.from_dict(self.config.sam3d_body_config)
        body_cfg.device = self.config.device
        self.body_adapter = SAM3DBodyAdapter(
            config=body_cfg,
            use_stub=self.config.use_stub_adapters,
        )

        # Renderer
        self.renderer = IRSceneGraphRenderer()

        # Refiner
        self.refiner = IRRefiner(
            config=self.config.ir_refiner_config,
            renderer=self.renderer,
            device=self.config.device,
        )

        # Tracker
        self.track_manager = KalmanTrackManager(
            config=self.config.tracking_config,
        )

    def process_episode(
        self,
        frames: List[np.ndarray],
        instance_masks: List[Dict[str, np.ndarray]],
        camera: CameraParams,
        class_labels: Optional[List[Dict[str, str]]] = None,
        keypoints: Optional[List[Dict[str, np.ndarray]]] = None,
        point_maps: Optional[List[np.ndarray]] = None,
    ) -> SceneTracks:
        """Process complete episode.

        Args:
            frames: List of (H, W, 3) RGB frames in [0, 255] uint8.
            instance_masks: Per-frame dict of instance_id -> (H, W) boolean mask.
            camera: Camera parameters.
            class_labels: Optional per-frame dict of instance_id -> class name.
            keypoints: Optional per-frame dict of instance_id -> (J, 3) keypoints.
            point_maps: Optional per-frame (H, W, 3) point maps.

        Returns:
            SceneTracks with all frame entities and metrics.
        """
        logger.info(f"Processing episode with {len(frames)} frames")

        self.track_manager.reset()
        all_frame_entities: List[List[SceneEntity3D]] = []
        ir_losses: List[float] = []
        converged_count = 0
        total_occlusion = 0.0
        total_entities = 0

        for t, frame in enumerate(frames):
            # Normalize frame
            if frame.dtype == np.uint8:
                frame_float = frame.astype(np.float32) / 255.0
            else:
                frame_float = frame.astype(np.float32)

            # Get masks for this frame
            frame_masks = instance_masks[t] if t < len(instance_masks) else {}
            frame_labels = class_labels[t] if class_labels and t < len(class_labels) else {}
            frame_kpts = keypoints[t] if keypoints and t < len(keypoints) else {}
            point_map = point_maps[t] if point_maps and t < len(point_maps) else None

            # Run SAM3D reconstruction
            entities = self._reconstruct_frame(
                frame_float,
                frame_masks,
                frame_labels,
                frame_kpts,
                point_map,
            )

            # Run IR refinement
            if entities and TORCH_AVAILABLE:
                refined_entities, refine_result = self._refine_frame(
                    entities,
                    frame_float,
                    frame_masks,
                    camera,
                )
                ir_losses.append(refine_result.final_loss)
                if refine_result.converged:
                    converged_count += 1
            else:
                refined_entities = entities
                ir_losses.append(0.0)

            # Run tracking
            tracked_entities = self.track_manager.update(refined_entities)

            # Collect stats
            for e in tracked_entities:
                total_occlusion += e.occlusion_score
                total_entities += 1

            all_frame_entities.append(tracked_entities)

        # Build SceneTracks
        tracks_dict = self.track_manager.get_tracks_dict()

        metrics = SceneTrackerMetrics(
            ir_loss_per_frame=ir_losses,
            id_switch_count=self.track_manager.id_switch_count,
            occlusion_rate=total_occlusion / max(1, total_entities),
            mean_ir_loss=float(np.mean(ir_losses)) if ir_losses else 0.0,
            converged_count=converged_count,
            total_frames=len(frames),
            total_tracks=len(tracks_dict),
            track_lengths=[len(hist) for hist in tracks_dict.values()],
        )

        return SceneTracks(
            frames=all_frame_entities,
            tracks=tracks_dict,
            metrics=metrics,
            config_used=self.config.to_dict(),
        )

    def _reconstruct_frame(
        self,
        frame: np.ndarray,
        masks: Dict[str, np.ndarray],
        class_labels: Dict[str, str],
        keypoints: Dict[str, np.ndarray],
        point_map: Optional[np.ndarray],
    ) -> List[SceneEntity3D]:
        """Reconstruct entities for a single frame."""
        entities = []

        for instance_id in sorted(masks.keys(), key=lambda key: str(key)):
            mask = masks[instance_id]
            class_name = class_labels.get(instance_id, "unknown")
            kpts = keypoints.get(instance_id)

            # Determine if body or object
            is_body = class_name.lower() in ("person", "human", "body") or kpts is not None

            if is_body:
                entity = self._reconstruct_body(
                    frame, mask, kpts, instance_id,
                )
            else:
                entity = self._reconstruct_object(
                    frame, mask, class_name, point_map, instance_id,
                )

            if entity is not None:
                entities.append(entity)

        return entities

    def _reconstruct_body(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        keypoints: Optional[np.ndarray],
        instance_id: str,
    ) -> Optional[SceneEntity3D]:
        """Reconstruct body using SAM3D-Body."""
        try:
            pred = self.body_adapter.infer(
                rgb=frame,
                person_mask=mask,
                keypoints=keypoints,
                body_id=instance_id,
            )

            return SceneEntity3D(
                entity_type="body",
                track_id=instance_id,  # Will be updated by tracker
                pose=pred.get_pose_matrix(),
                scale=1.0,
                class_name="person",
                mask_2d=mask,
                z_shape=pred.shape_latent,
                z_tex=pred.pose_latent,  # Use pose as "texture" for bodies
                joints_3d=pred.joints_3d,
                visibility=1.0,
                occlusion_score=0.0,
            )
        except Exception as e:
            logger.warning(f"Body reconstruction failed for {instance_id}: {e}")
            return None

    def _reconstruct_object(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        class_name: str,
        point_map: Optional[np.ndarray],
        instance_id: str,
    ) -> Optional[SceneEntity3D]:
        """Reconstruct object using SAM3D-Objects."""
        try:
            predictions = self.objects_adapter.infer(
                rgb=frame,
                instance_masks=[mask],
                point_map=point_map,
                class_names=[class_name],
            )

            if not predictions:
                return None

            pred = predictions[0]

            # Build pose from layout
            pose = np.eye(4, dtype=np.float32)
            pose[:3, 3] = np.array(pred.layout["position"], dtype=np.float32)

            return SceneEntity3D(
                entity_type="object",
                track_id=instance_id,
                pose=pose,
                scale=float(pred.layout["scale"]),
                class_name=class_name,
                mask_2d=mask,
                geometry_handle=pred.geometry,
                z_shape=pred.shape_latent,
                z_tex=pred.appearance_latent,
                visibility=1.0,
                occlusion_score=0.0,
            )
        except Exception as e:
            logger.warning(f"Object reconstruction failed for {instance_id}: {e}")
            return None

    def _refine_frame(
        self,
        entities: List[SceneEntity3D],
        frame: np.ndarray,
        masks: Dict[str, np.ndarray],
        camera: CameraParams,
    ) -> tuple:
        """Refine entities for a single frame."""
        _check_torch()

        H, W = frame.shape[:2]

        # Convert frame to tensor
        target_rgb = torch.from_numpy(frame).permute(2, 0, 1).float()
        if target_rgb.device != self.refiner.device:
            target_rgb = target_rgb.to(self.refiner.device)

        # Convert masks to tensor dict
        target_masks = {}
        for entity in entities:
            if entity.mask_2d is not None:
                mask_t = torch.from_numpy(entity.mask_2d.astype(np.float32))
                target_masks[entity.track_id] = mask_t.to(self.refiner.device)

        return self.refiner.refine(entities, target_rgb, target_masks, camera)

    def process_frame(
        self,
        frame: np.ndarray,
        masks: Dict[str, np.ndarray],
        camera: CameraParams,
        class_labels: Optional[Dict[str, str]] = None,
        keypoints: Optional[Dict[str, np.ndarray]] = None,
        point_map: Optional[np.ndarray] = None,
    ) -> List[SceneEntity3D]:
        """Process single frame (for online use).

        Args:
            frame: (H, W, 3) RGB frame.
            masks: Dict of instance_id -> (H, W) mask.
            camera: Camera parameters.
            class_labels: Optional dict of instance_id -> class name.
            keypoints: Optional dict of instance_id -> keypoints.
            point_map: Optional (H, W, 3) point map.

        Returns:
            List of tracked entities with stable IDs.
        """
        if frame.dtype == np.uint8:
            frame = frame.astype(np.float32) / 255.0

        class_labels = class_labels or {}
        keypoints = keypoints or {}

        # Reconstruct
        entities = self._reconstruct_frame(
            frame, masks, class_labels, keypoints, point_map,
        )

        # Refine
        if entities and TORCH_AVAILABLE:
            refined, _ = self._refine_frame(entities, frame, masks, camera)
        else:
            refined = entities

        # Track
        tracked = self.track_manager.update(refined)

        return tracked


def create_scene_ir_tracker(
    config: Optional[Dict[str, Any]] = None,
) -> SceneIRTracker:
    """Factory function to create Scene IR Tracker.

    Args:
        config: Configuration dict.

    Returns:
        Configured SceneIRTracker.
    """
    if config:
        cfg = SceneIRTrackerConfig.from_dict(config)
    else:
        cfg = SceneIRTrackerConfig()
    return SceneIRTracker(config=cfg)
