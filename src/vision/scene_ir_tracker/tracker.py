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
from src.vision.scene_ir_tracker.ir_refiner import IRRefiner
from src.vision.scene_ir_tracker.ir_scene_graph_renderer import IRSceneGraphRenderer
from src.vision.scene_ir_tracker.kalman_track_manager import KalmanTrackManager
from src.vision.scene_ir_tracker.sam3d_body_adapter import (
    SAM3DBodyAdapter,
    SAM3DBodyConfig,
)
from src.vision.scene_ir_tracker.sam3d_objects_adapter import (
    SAM3DObjectsAdapter,
    SAM3DObjectsConfig,
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
        self.objects_adapter: Optional[SAM3DObjectsAdapter]
        self.body_adapter: Optional[SAM3DBodyAdapter]
        if self.config.zero_inference_passthrough:
            self.objects_adapter = None
            self.body_adapter = None
        else:
            objects_cfg = SAM3DObjectsConfig.from_dict(self.config.sam3d_objects_config)
            objects_cfg.device = self.config.device
            self.objects_adapter = SAM3DObjectsAdapter(
                config=objects_cfg,
                use_stub=self.config.use_stub_adapters,
                allow_fallbacks=self.config.allow_fallbacks,
            )

            body_cfg = SAM3DBodyConfig.from_dict(self.config.sam3d_body_config)
            body_cfg.device = self.config.device
            self.body_adapter = SAM3DBodyAdapter(
                config=body_cfg,
                use_stub=self.config.use_stub_adapters,
                allow_fallbacks=self.config.allow_fallbacks,
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
        object_refs: Optional[List[Dict[str, str]]] = None,
        keypoints: Optional[List[Dict[str, np.ndarray]]] = None,
        point_maps: Optional[List[np.ndarray]] = None,
        depth_frames: Optional[List[np.ndarray]] = None,
    ) -> SceneTracks:
        """Process complete episode.

        Args:
            frames: List of (H, W, 3) RGB frames in [0, 255] uint8.
            instance_masks: Per-frame dict of instance_id -> (H, W) boolean mask.
            camera: Camera parameters.
            class_labels: Optional per-frame dict of instance_id -> class name.
            object_refs: Optional per-frame dict of instance_id -> upstream object ref.
            keypoints: Optional per-frame dict of instance_id -> (J, 3) keypoints.
            point_maps: Optional per-frame (H, W, 3) point maps in camera frame.
            depth_frames: Optional per-frame depth maps in meters.

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
            frame_object_refs = object_refs[t] if object_refs and t < len(object_refs) else {}
            frame_kpts = keypoints[t] if keypoints and t < len(keypoints) else {}
            point_map = point_maps[t] if point_maps and t < len(point_maps) else None
            depth_frame = depth_frames[t] if depth_frames and t < len(depth_frames) else None
            if point_map is None and depth_frame is not None:
                point_map = self._point_map_from_depth(depth_frame, camera, t)

            # Run SAM3D reconstruction
            entities = self._reconstruct_frame(
                frame_float,
                frame_masks,
                frame_labels,
                frame_object_refs,
                frame_kpts,
                point_map,
                camera=camera,
                frame_index=t,
                depth_frame=depth_frame,
            )

            # Run IR refinement
            if entities and TORCH_AVAILABLE and not self.config.zero_inference_passthrough:
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
            config_used={**self.config.to_dict(), "output_frame": "world"},
        )

    def _reconstruct_frame(
        self,
        frame: np.ndarray,
        masks: Dict[str, np.ndarray],
        class_labels: Dict[str, str],
        object_refs: Dict[str, str],
        keypoints: Dict[str, np.ndarray],
        point_map: Optional[np.ndarray],
        *,
        camera: Optional[CameraParams] = None,
        frame_index: int = 0,
        depth_frame: Optional[np.ndarray] = None,
    ) -> List[SceneEntity3D]:
        """Reconstruct entities for a single frame."""
        entities = []

        for instance_id in sorted(masks.keys(), key=lambda key: str(key)):
            mask = masks[instance_id]
            class_name = class_labels.get(instance_id, "unknown")
            object_ref = object_refs.get(instance_id, "")
            kpts = keypoints.get(instance_id)

            # Determine if body or object
            is_body = class_name.lower() in ("person", "human", "body") or kpts is not None

            if is_body:
                entity = self._reconstruct_body(
                    frame,
                    mask,
                    kpts,
                    instance_id,
                    object_ref,
                    point_map=point_map,
                    camera=camera,
                    frame_index=frame_index,
                    depth_frame=depth_frame,
                )
            else:
                entity = self._reconstruct_object(
                    frame,
                    mask,
                    class_name,
                    point_map,
                    instance_id,
                    object_ref,
                    camera=camera,
                    frame_index=frame_index,
                    depth_frame=depth_frame,
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
        object_ref: str = "",
        *,
        point_map: Optional[np.ndarray] = None,
        camera: Optional[CameraParams] = None,
        frame_index: int = 0,
        depth_frame: Optional[np.ndarray] = None,
    ) -> Optional[SceneEntity3D]:
        """Reconstruct body using SAM3D-Body."""
        if self.config.zero_inference_passthrough:
            return self._reconstruct_body_passthrough(
                mask,
                keypoints=keypoints,
                instance_id=instance_id,
                object_ref=object_ref,
                point_map=point_map,
                camera=camera,
                frame_index=frame_index,
                depth_frame=depth_frame,
            )

        if self.body_adapter is None:
            return None
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
                source_instance_id=instance_id,
                source_object_id=object_ref or None,
                label_source="explicit_segmentation_map" if object_ref else "",
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
        object_ref: str = "",
        *,
        camera: Optional[CameraParams] = None,
        frame_index: int = 0,
        depth_frame: Optional[np.ndarray] = None,
    ) -> Optional[SceneEntity3D]:
        """Reconstruct object using SAM3D-Objects."""
        if self.config.zero_inference_passthrough:
            return self._reconstruct_object_passthrough(
                mask,
                class_name=class_name,
                instance_id=instance_id,
                object_ref=object_ref,
                point_map=point_map,
                camera=camera,
                frame_index=frame_index,
                depth_frame=depth_frame,
            )

        if self.objects_adapter is None:
            return None
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
                source_instance_id=instance_id,
                source_object_id=object_ref or None,
                label_source="explicit_segmentation_map" if object_ref else "",
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
        object_refs: Optional[Dict[str, str]] = None,
        keypoints: Optional[Dict[str, np.ndarray]] = None,
        point_map: Optional[np.ndarray] = None,
        depth_frame: Optional[np.ndarray] = None,
    ) -> List[SceneEntity3D]:
        """Process single frame (for online use).

        Args:
            frame: (H, W, 3) RGB frame.
            masks: Dict of instance_id -> (H, W) mask.
            camera: Camera parameters.
            class_labels: Optional dict of instance_id -> class name.
            object_refs: Optional dict of instance_id -> upstream object ref.
            keypoints: Optional dict of instance_id -> keypoints.
            point_map: Optional (H, W, 3) point map in camera frame.
            depth_frame: Optional (H, W) depth map in meters.

        Returns:
            List of tracked entities with stable IDs.
        """
        if frame.dtype == np.uint8:
            frame = frame.astype(np.float32) / 255.0

        class_labels = class_labels or {}
        object_refs = object_refs or {}
        keypoints = keypoints or {}
        if point_map is None and depth_frame is not None:
            point_map = self._point_map_from_depth(depth_frame, camera, 0)

        # Reconstruct
        entities = self._reconstruct_frame(
            frame,
            masks,
            class_labels,
            object_refs,
            keypoints,
            point_map,
            camera=camera,
            frame_index=0,
            depth_frame=depth_frame,
        )

        # Refine
        if entities and TORCH_AVAILABLE and not self.config.zero_inference_passthrough:
            refined, _ = self._refine_frame(entities, frame, masks, camera)
        else:
            refined = entities

        # Track
        tracked = self.track_manager.update(refined)

        return tracked

    def adapter_status(self) -> Dict[str, Any]:
        if self.config.zero_inference_passthrough:
            return {
                "object_adapter_mode": "zero_inference_passthrough",
                "body_adapter_mode": "zero_inference_passthrough",
                "object_adapter_real": False,
                "body_adapter_real": False,
                "object_adapter_passthrough": True,
                "body_adapter_passthrough": True,
                "no_inference_backend": True,
                "deterministic_backend": True,
                "overall_mode": "passthrough",
            }
        object_mode = str(getattr(self.objects_adapter, "backend_mode", "unknown"))
        body_mode = str(getattr(self.body_adapter, "backend_mode", "unknown"))
        return {
            "object_adapter_mode": object_mode,
            "body_adapter_mode": body_mode,
            "object_adapter_real": bool(object_mode == "real"),
            "body_adapter_real": bool(body_mode == "real"),
            "object_adapter_passthrough": False,
            "body_adapter_passthrough": False,
            "no_inference_backend": False,
            "deterministic_backend": False,
            "overall_mode": "real" if object_mode == "real" and body_mode == "real" else "degraded",
        }

    def _point_map_from_depth(
        self,
        depth_frame: np.ndarray,
        camera: CameraParams,
        frame_index: int,
    ) -> np.ndarray:
        depth = np.asarray(depth_frame, dtype=np.float32)
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        if depth.shape[:2] != (int(camera.height), int(camera.width)):
            raise ValueError(
                f"Depth frame shape {depth.shape[:2]} does not match camera resolution "
                f"{(int(camera.height), int(camera.width))}"
            )
        u, v = np.meshgrid(np.arange(camera.width, dtype=np.float32), np.arange(camera.height, dtype=np.float32))
        z = np.where(np.isfinite(depth) & (depth > 1e-6), depth, 0.0)
        x = ((u - float(camera.cx)) / max(float(camera.fx), 1e-6)) * z
        y = ((v - float(camera.cy)) / max(float(camera.fy), 1e-6)) * z
        point_map = np.stack([x, y, z], axis=-1).astype(np.float32)
        if frame_index < 0:
            frame_index = 0
        return point_map

    def _world_points_from_mask(
        self,
        mask: np.ndarray,
        *,
        point_map: Optional[np.ndarray],
        camera: Optional[CameraParams],
        frame_index: int,
    ) -> np.ndarray:
        if camera is None or point_map is None or not np.any(mask):
            return np.zeros((0, 3), dtype=np.float32)
        points_cam = np.asarray(point_map[mask], dtype=np.float32)
        if points_cam.ndim != 2 or points_cam.shape[-1] != 3 or points_cam.size == 0:
            return np.zeros((0, 3), dtype=np.float32)
        valid = np.isfinite(points_cam).all(axis=1) & (points_cam[:, 2] > 1e-6)
        if not np.any(valid):
            return np.zeros((0, 3), dtype=np.float32)
        points_cam = points_cam[valid]
        # CameraParams stores cameras looking down -Z in camera space.
        points_cam = points_cam.copy()
        points_cam[:, 2] *= -1.0
        world_from_cam = np.asarray(
            camera.world_from_cam[min(max(frame_index, 0), camera.world_from_cam.shape[0] - 1)],
            dtype=np.float32,
        )
        ones = np.ones((points_cam.shape[0], 1), dtype=np.float32)
        points_cam_h = np.concatenate([points_cam, ones], axis=1)
        return (world_from_cam @ points_cam_h.T).T[:, :3].astype(np.float32)

    def _default_world_center_from_mask(
        self,
        mask: np.ndarray,
        camera: Optional[CameraParams],
        frame_index: int,
    ) -> np.ndarray:
        if camera is None or not np.any(mask):
            return np.zeros(3, dtype=np.float32)
        ys, xs = np.where(mask)
        u = float(np.mean(xs))
        v = float(np.mean(ys))
        t = min(max(frame_index, 0), camera.num_frames - 1)
        cam_pos = camera.get_position(t)
        x_cam = (u - float(camera.cx)) / max(float(camera.fx), 1e-6)
        y_cam = (v - float(camera.cy)) / max(float(camera.fy), 1e-6)
        ray_cam = np.array([x_cam, y_cam, 1.0], dtype=np.float32)
        ray_cam /= max(np.linalg.norm(ray_cam), 1e-6)
        world_from_cam = np.asarray(camera.world_from_cam[t], dtype=np.float32)
        ray_world = world_from_cam[:3, :3] @ ray_cam
        ray_world /= max(np.linalg.norm(ray_world), 1e-6)
        default_depth = 1.5
        return (cam_pos + ray_world * default_depth).astype(np.float32)

    def _default_scale_from_mask(
        self,
        mask: np.ndarray,
        camera: Optional[CameraParams],
    ) -> float:
        if camera is None or not np.any(mask):
            return 0.05
        area_px = float(np.count_nonzero(mask))
        linear_px = max(np.sqrt(area_px), 1.0)
        return float(max((linear_px / max(float(camera.fx), 1.0)) * 1.5, 0.02))

    def _estimate_world_pose_from_mask(
        self,
        mask: np.ndarray,
        *,
        point_map: Optional[np.ndarray],
        camera: Optional[CameraParams],
        frame_index: int,
    ) -> tuple[np.ndarray, float, int]:
        world_points = self._world_points_from_mask(
            mask,
            point_map=point_map,
            camera=camera,
            frame_index=frame_index,
        )
        pose = np.eye(4, dtype=np.float32)
        if world_points.size == 0:
            pose[:3, 3] = self._default_world_center_from_mask(mask, camera, frame_index)
            return pose, self._default_scale_from_mask(mask, camera), 0

        center = np.mean(world_points, axis=0).astype(np.float32)
        extent = np.ptp(world_points, axis=0) if world_points.shape[0] > 1 else np.zeros(3, dtype=np.float32)
        fallback_scale = self._default_scale_from_mask(mask, camera)
        scale = float(max(np.linalg.norm(extent), fallback_scale, 0.02))
        pose[:3, 3] = center
        return pose, scale, int(world_points.shape[0])

    def _reconstruct_object_passthrough(
        self,
        mask: np.ndarray,
        *,
        class_name: str,
        instance_id: str,
        object_ref: str,
        point_map: Optional[np.ndarray],
        camera: Optional[CameraParams],
        frame_index: int,
        depth_frame: Optional[np.ndarray],
    ) -> Optional[SceneEntity3D]:
        if point_map is None and depth_frame is not None and camera is not None:
            point_map = self._point_map_from_depth(depth_frame, camera, frame_index)
        if not np.any(mask):
            return None
        pose, scale, point_count = self._estimate_world_pose_from_mask(
            mask,
            point_map=point_map,
            camera=camera,
            frame_index=frame_index,
        )
        ys, xs = np.where(mask)
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(xs) else [0, 0, 0, 0]
        label_source = "explicit_segmentation_map" if object_ref else "segmentation_mask_passthrough"
        return SceneEntity3D(
            entity_type="object",
            track_id=instance_id,
            pose=pose,
            scale=scale,
            class_name=class_name,
            mask_2d=mask,
            geometry_handle={
                "backend": "zero_inference_passthrough",
                "point_count": int(point_count),
                "bbox_xyxy": bbox,
            },
            visibility=float(min(1.0, max(np.count_nonzero(mask) / max(mask.size, 1), 0.01))),
            occlusion_score=0.0,
            source_instance_id=instance_id,
            source_object_id=object_ref or None,
            label_source=label_source,
        )

    def _reconstruct_body_passthrough(
        self,
        mask: np.ndarray,
        *,
        keypoints: Optional[np.ndarray],
        instance_id: str,
        object_ref: str,
        point_map: Optional[np.ndarray],
        camera: Optional[CameraParams],
        frame_index: int,
        depth_frame: Optional[np.ndarray],
    ) -> Optional[SceneEntity3D]:
        if point_map is None and depth_frame is not None and camera is not None:
            point_map = self._point_map_from_depth(depth_frame, camera, frame_index)
        if not np.any(mask):
            return None
        pose, scale, point_count = self._estimate_world_pose_from_mask(
            mask,
            point_map=point_map,
            camera=camera,
            frame_index=frame_index,
        )
        pelvis = pose[:3, 3].copy()
        joints_3d = {"pelvis": pelvis}
        if keypoints is not None and np.asarray(keypoints).ndim >= 2 and np.asarray(keypoints).shape[0] > 0:
            joints_3d["head"] = pelvis + np.array([0.0, 0.0, max(scale * 0.5, 0.1)], dtype=np.float32)
        label_source = "explicit_segmentation_map" if object_ref else "segmentation_mask_passthrough"
        return SceneEntity3D(
            entity_type="body",
            track_id=instance_id,
            pose=pose,
            scale=scale,
            class_name="person",
            mask_2d=mask,
            geometry_handle={
                "backend": "zero_inference_passthrough",
                "point_count": int(point_count),
            },
            joints_3d=joints_3d,
            visibility=float(min(1.0, max(np.count_nonzero(mask) / max(mask.size, 1), 0.01))),
            occlusion_score=0.0,
            source_instance_id=instance_id,
            source_object_id=object_ref or None,
            label_source=label_source,
        )


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
