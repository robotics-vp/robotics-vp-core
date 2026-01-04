"""Map-First Pseudo-Supervision Node."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from src.vision.map_first_supervision.artifacts import (
    MapFirstArtifact,
    MapFirstSummary,
    compute_map_first_summary,
)
from src.vision.map_first_supervision.config import MapFirstSupervisionConfig
from src.vision.map_first_supervision.densify import (
    densify_depth_targets,
    densify_world_points,
)
from src.vision.map_first_supervision.geometry_provider import (
    GeometryProvider,
    PrimitiveGeometryProvider,
)
from src.vision.map_first_supervision.inconsistency import compute_inconsistency
from src.vision.map_first_supervision.semantics import (
    SemanticStabilizer,
    parse_vla_semantic_evidence,
)
from src.vision.map_first_supervision.static_map import VoxelHashMap
from src.vision.scene_ir_tracker.serialization import deserialize_scene_tracks_v1
from src.vision.nag.types import CameraParams


@dataclass
class MapFirstRunOutput:
    artifact_path: str
    summary: MapFirstSummary


def _get_attr(obj: Any, key: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _build_camera_params(episode_assets: Any, num_frames: int) -> Optional[CameraParams]:
    if episode_assets is None:
        return None
    cam_params = _get_attr(episode_assets, "camera_params")
    if isinstance(cam_params, CameraParams):
        return cam_params
    if isinstance(cam_params, dict):
        if "world_from_cam" in cam_params:
            cam_params = cam_params.copy()
            world_from_cam = np.asarray(cam_params["world_from_cam"], dtype=np.float32)
            cam_params["world_from_cam"] = world_from_cam
            return CameraParams(**cam_params)

    intrinsics = _get_attr(episode_assets, "camera_intrinsics")
    extrinsics = _get_attr(episode_assets, "camera_extrinsics") or _get_attr(episode_assets, "camera_pose")
    if intrinsics is None or extrinsics is None:
        return None

    intrinsics = dict(intrinsics)
    resolution = intrinsics.get("resolution") or intrinsics.get("image_size")
    if resolution is None:
        return None
    width = int(resolution[0])
    height = int(resolution[1])

    fx = intrinsics.get("fx")
    fy = intrinsics.get("fy")
    cx = intrinsics.get("cx")
    cy = intrinsics.get("cy")
    if fx is None or fy is None:
        fov_deg = intrinsics.get("fov_deg", 90.0)
        fy = 0.5 * height / np.tan(np.deg2rad(fov_deg) / 2.0)
        fx = fy
    if cx is None:
        cx = width / 2.0
    if cy is None:
        cy = height / 2.0

    world_from_cam = None
    if isinstance(extrinsics, dict) and "world_from_cam" in extrinsics:
        world_from_cam = np.asarray(extrinsics["world_from_cam"], dtype=np.float32)
    elif isinstance(extrinsics, dict):
        translation = np.asarray(extrinsics.get("translation", [0.0, 0.0, 0.0]), dtype=np.float32)
        rotation_rpy = np.asarray(extrinsics.get("rotation_rpy", [0.0, 0.0, 0.0]), dtype=np.float32)
        roll, pitch, yaw = rotation_rpy
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cyw, syw = np.cos(yaw), np.sin(yaw)
        Rz = np.array([[cyw, -syw, 0.0], [syw, cyw, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float32)
        Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float32)
        R = Rz @ Ry @ Rx
        world_from_cam = np.eye(4, dtype=np.float32)
        world_from_cam[:3, :3] = R
        world_from_cam[:3, 3] = translation
    else:
        world_from_cam = np.asarray(extrinsics, dtype=np.float32)

    if world_from_cam.ndim == 2:
        world_from_cam = world_from_cam[np.newaxis, ...]
    if world_from_cam.shape[0] == 1 and num_frames > 1:
        world_from_cam = np.repeat(world_from_cam, num_frames, axis=0)

    return CameraParams(
        fx=float(fx),
        fy=float(fy),
        cx=float(cx),
        cy=float(cy),
        height=height,
        width=width,
        world_from_cam=world_from_cam,
    )


def _ensure_scene_tracks_lite(scene_tracks: Any) -> Any:
    if hasattr(scene_tracks, "poses_R") and hasattr(scene_tracks, "poses_t"):
        return scene_tracks
    if isinstance(scene_tracks, dict):
        return deserialize_scene_tracks_v1(scene_tracks)
    raise ValueError("Unsupported scene_tracks input; expected SceneTracksLite or dict")


def _is_visible(visibility: float, occlusion: float) -> bool:
    return (visibility * (1.0 - occlusion)) >= 0.2


def _extract_semantic_inputs(
    optional_semantics: Any,
    scene_tracks: Any,
) -> Tuple[
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[Dict[str, np.ndarray]],
    Optional[Any],
]:
    if optional_semantics is None:
        return None, None, None, None
    if isinstance(optional_semantics, np.ndarray):
        return optional_semantics, None, None, None
    if isinstance(optional_semantics, dict):
        evidence = parse_vla_semantic_evidence(optional_semantics, scene_track_ids=getattr(scene_tracks, "track_ids", None))
        if evidence is not None and evidence.class_probs is not None:
            return evidence.class_probs, evidence.confidence, optional_semantics.get("point_labels"), evidence
        return (
            optional_semantics.get("entity_class_probs"),
            optional_semantics.get("entity_confidence"),
            optional_semantics.get("point_labels"),
            None,
        )
    evidence = parse_vla_semantic_evidence(optional_semantics, scene_track_ids=getattr(scene_tracks, "track_ids", None))
    if evidence is not None and evidence.class_probs is not None:
        return evidence.class_probs, evidence.confidence, None, evidence
    return None, None, None, None


def _accumulate_static_map(
    scene_tracks: Any,
    geometry_provider: GeometryProvider,
    map_store: VoxelHashMap,
    points_per_entity: int,
    allow_mask: Optional[np.ndarray] = None,
    semantics_stabilizer: Optional[SemanticStabilizer] = None,
    entity_class_probs: Optional[np.ndarray] = None,
    entity_confidence: Optional[np.ndarray] = None,
) -> None:
    visibility = getattr(scene_tracks, "visibility", None)
    occlusion = getattr(scene_tracks, "occlusion", None)
    if visibility is None:
        visibility = np.ones((scene_tracks.poses_t.shape[0], scene_tracks.poses_t.shape[1]), dtype=np.float32)
    if occlusion is None:
        occlusion = np.zeros((scene_tracks.poses_t.shape[0], scene_tracks.poses_t.shape[1]), dtype=np.float32)

    T, K = scene_tracks.poses_t.shape[:2]
    for t in range(T):
        for k in range(K):
            if not _is_visible(float(visibility[t, k]), float(occlusion[t, k])):
                continue
            if allow_mask is not None and not bool(allow_mask[t, k]):
                continue
            points = geometry_provider.sample_points_world(t, k, points_per_entity)
            map_store.update(points)
            if semantics_stabilizer is not None and entity_class_probs is not None:
                class_probs = entity_class_probs[t, k]
                if np.any(class_probs):
                    confidence = None
                    if entity_confidence is not None:
                        confidence = float(entity_confidence[t, k])
                    semantics_stabilizer.update_from_entity_probs(points, class_probs, confidence=confidence)


def _update_map_with_point_labels(
    semantics_stabilizer: Optional[SemanticStabilizer],
    point_labels: Optional[Dict[str, np.ndarray]],
) -> None:
    if semantics_stabilizer is None or point_labels is None:
        return
    points = point_labels.get("points_world")
    labels = point_labels.get("labels")
    if points is None or labels is None:
        return
    points = np.asarray(points, dtype=np.float32)
    labels = np.asarray(labels)
    if points.ndim == 3 and labels.ndim == 2:
        for t in range(points.shape[0]):
            semantics_stabilizer.update_from_point_labels(points[t], labels[t])
    elif points.ndim == 2 and labels.ndim == 1:
        semantics_stabilizer.update_from_point_labels(points, labels)


def _compute_boxes3d(scene_tracks: Any, geometry_provider: GeometryProvider) -> np.ndarray:
    poses_R = scene_tracks.poses_R
    poses_t = scene_tracks.poses_t
    T, K = poses_t.shape[:2]
    boxes = np.zeros((T, K, 7), dtype=np.float32)
    scales = getattr(scene_tracks, "scales", None)

    for t in range(T):
        for k in range(K):
            R = poses_R[t, k]
            yaw = float(np.arctan2(R[1, 0], R[0, 0]))
            if hasattr(geometry_provider, "box_dims"):
                dims = geometry_provider.box_dims(t, k)
            elif scales is not None:
                scale = float(scales[t, k])
                dims = np.array([scale, scale, scale], dtype=np.float32)
            else:
                dims = np.array([1.0, 1.0, 1.0], dtype=np.float32)
            boxes[t, k] = np.array(
                [
                    poses_t[t, k, 0],
                    poses_t[t, k, 1],
                    poses_t[t, k, 2],
                    dims[0],
                    dims[1],
                    dims[2],
                    yaw,
                ],
                dtype=np.float32,
            )
    return boxes


def _compute_semantics_stable(
    scene_tracks: Any,
    geometry_provider: GeometryProvider,
    stabilizer: Optional[SemanticStabilizer],
    points_per_entity: int,
) -> Optional[np.ndarray]:
    if stabilizer is None:
        return None
    T, K = scene_tracks.poses_t.shape[:2]
    C = stabilizer.map_store.semantics_num_classes
    semantics_stable = np.zeros((T, K, C), dtype=np.float32)
    for t in range(T):
        for k in range(K):
            points = geometry_provider.sample_points_world(t, k, points_per_entity)
            semantics_stable[t, k] = stabilizer.aggregate_entity_probs(points)
    return semantics_stable


def _compute_semantic_stability(semantics_stable: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if semantics_stable is None or semantics_stable.size == 0:
        return None
    return np.max(semantics_stable, axis=-1).astype(np.float32)


class MapFirstPseudoSupervisionNode:
    """Map-First pseudo-supervision builder downstream of SceneTracks_v1."""

    def __init__(
        self,
        config: Optional[MapFirstSupervisionConfig] = None,
        geometry_provider: Optional[GeometryProvider] = None,
        points_per_entity: int = 64,
    ) -> None:
        self.config = config or MapFirstSupervisionConfig()
        self.geometry_provider = geometry_provider
        self.points_per_entity = int(points_per_entity)

    def run(
        self,
        scene_tracks: Any,
        episode_assets: Any = None,
        optional_semantics: Any = None,
        output_path: Optional[str] = None,
    ) -> MapFirstRunOutput:
        scene_tracks_lite = _ensure_scene_tracks_lite(scene_tracks)
        geometry_provider = self.geometry_provider or PrimitiveGeometryProvider(scene_tracks_lite)

        entity_class_probs, entity_confidence, point_labels, vla_evidence = _extract_semantic_inputs(
            optional_semantics,
            scene_tracks_lite,
        )
        semantics_enabled = self.config.semantics_enabled and (entity_class_probs is not None or point_labels is not None)
        semantics_num_classes = self.config.semantics_num_classes
        if semantics_enabled and entity_class_probs is not None:
            semantics_num_classes = int(entity_class_probs.shape[-1])
        if semantics_enabled and semantics_num_classes <= 0:
            raise ValueError("semantics_num_classes must be > 0 when semantics are enabled")

        map_store = VoxelHashMap(
            voxel_size=self.config.voxel_size,
            max_points_per_voxel=self.config.map_max_points_per_voxel,
            semantics_num_classes=semantics_num_classes if semantics_enabled else 0,
        )
        stabilizer = SemanticStabilizer(map_store) if semantics_enabled else None

        _accumulate_static_map(
            scene_tracks_lite,
            geometry_provider,
            map_store,
            self.points_per_entity,
            allow_mask=None,
            semantics_stabilizer=stabilizer,
            entity_class_probs=entity_class_probs,
            entity_confidence=entity_confidence,
        )
        _update_map_with_point_labels(stabilizer, point_labels)

        inconsistency = compute_inconsistency(
            scene_tracks_lite,
            map_store,
            geometry_provider,
            self.config,
            points_per_entity=self.points_per_entity,
        )

        if self.config.static_update_policy == "visible_and_low_residual":
            allow_mask = (~inconsistency.dynamic_mask).astype(bool)
            map_store = VoxelHashMap(
                voxel_size=self.config.voxel_size,
                max_points_per_voxel=self.config.map_max_points_per_voxel,
                semantics_num_classes=semantics_num_classes if semantics_enabled else 0,
            )
            stabilizer = SemanticStabilizer(map_store) if semantics_enabled else None
            _accumulate_static_map(
                scene_tracks_lite,
                geometry_provider,
                map_store,
                self.points_per_entity,
                allow_mask=allow_mask,
                semantics_stabilizer=stabilizer,
                entity_class_probs=entity_class_probs,
                entity_confidence=entity_confidence,
            )
            _update_map_with_point_labels(stabilizer, point_labels)
            inconsistency = compute_inconsistency(
                scene_tracks_lite,
                map_store,
                geometry_provider,
                self.config,
                points_per_entity=self.points_per_entity,
            )

        camera_params = _build_camera_params(episode_assets, scene_tracks_lite.num_frames)
        densify_out = None
        if self.config.densify_enabled:
            if camera_params is not None and self.config.densify_mode == "depth_map":
                densify_out = densify_depth_targets(
                    map_store,
                    camera_params,
                    scene_tracks_lite.num_frames,
                    occlusion_culling=self.config.occlusion_culling,
                )
            else:
                densify_out = densify_world_points(map_store, scene_tracks_lite.num_frames)

        semantics_stable = _compute_semantics_stable(
            scene_tracks_lite,
            geometry_provider,
            stabilizer,
            self.points_per_entity,
        )
        semantics_stability = _compute_semantic_stability(semantics_stable)

        boxes3d = _compute_boxes3d(scene_tracks_lite, geometry_provider)
        occlusion = getattr(scene_tracks_lite, "occlusion", None)
        if occlusion is None or occlusion.size == 0:
            occlusion = np.zeros_like(inconsistency.dynamic_evidence, dtype=np.float32)
        vla_class_probs = None
        vla_confidence = None
        vla_embed = None
        vla_provenance_json = None
        if vla_evidence is not None:
            vla_class_probs = vla_evidence.class_probs
            vla_confidence = vla_evidence.confidence
            vla_embed = vla_evidence.embed
            if vla_evidence.provenance is not None:
                vla_provenance_json = json.dumps(vla_evidence.provenance)

        artifact = MapFirstArtifact(
            dynamic_evidence=inconsistency.dynamic_evidence,
            dynamic_mask=inconsistency.dynamic_mask,
            residual_mean=inconsistency.residual_mean,
            boxes3d=boxes3d,
            confidence=inconsistency.confidence,
            densify_depth=densify_out.depth if densify_out else None,
            densify_mask=densify_out.mask if densify_out else None,
            densify_world_points=densify_out.world_points if densify_out else None,
            densify_world_mask=densify_out.world_mask if densify_out else None,
            semantics_stable=semantics_stable,
            semantics_stability=semantics_stability,
            static_map_centroids=map_store.voxel_centroids(),
            static_map_counts=map_store.voxel_counts(),
            static_map_semantics=map_store.voxel_semantics(),
            evidence_occlusion=occlusion.astype(np.float32),
            vla_class_probs=vla_class_probs,
            vla_confidence=vla_confidence,
            vla_embed=vla_embed,
            vla_provenance_json=vla_provenance_json,
        )

        summary = compute_map_first_summary(
            residual_mean=inconsistency.residual_mean,
            dynamic_mask=inconsistency.dynamic_mask,
            coverage=inconsistency.coverage,
            visibility_weight=inconsistency.visibility_weight,
            confidence=inconsistency.confidence,
            densify_mask=densify_out.mask if densify_out else None,
            semantic_stability=semantics_stability,
        )

        if isinstance(episode_assets, dict):
            episode_assets["map_first_summary"] = summary.to_dict()
            episode_assets["map_first_quality_score"] = summary.map_first_quality_score

        if output_path is None:
            output_path = "map_first_supervision_v1.npz"

        np.savez_compressed(output_path, **artifact.to_npz(summary=summary, export_float16=self.config.export_float16))

        return MapFirstRunOutput(artifact_path=output_path, summary=summary)
