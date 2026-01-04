"""
Coordinate Frame Transforms for Scene IR Tracker.

Handles world/camera frame conversions and validation.
"""
from __future__ import annotations

import logging
from typing import Literal, Optional

import numpy as np

from src.vision.scene_ir_tracker.types import SceneTracks, SceneEntity3D

logger = logging.getLogger(__name__)


# Frame type literal
FrameType = Literal["world", "camera"]


class FrameMismatchError(RuntimeError):
    """Raised when coordinate frames don't match expectations."""
    pass


def validate_frame_consistency(
    scene_tracks: SceneTracks,
    expected_frame: FrameType,
    frame_field: Optional[str] = None,
) -> None:
    """Validate that scene tracks are in the expected coordinate frame.
    
    Args:
        scene_tracks: Tracks to validate.
        expected_frame: Expected frame ("world" or "camera").
        frame_field: Optional field name for frame info (default checks config).
    
    Raises:
        FrameMismatchError: If frame doesn't match and no transform available.
    """
    # Check if frame is specified in config
    actual_frame = None
    
    if scene_tracks.config_used:
        actual_frame = scene_tracks.config_used.get("output_frame")
    
    if frame_field and hasattr(scene_tracks, frame_field):
        actual_frame = getattr(scene_tracks, frame_field)
    
    if actual_frame is None:
        # No frame specified - warn but allow
        logger.warning(
            f"No coordinate frame specified in scene tracks. "
            f"Assuming {expected_frame} frame."
        )
        return
    
    if actual_frame != expected_frame:
        raise FrameMismatchError(
            f"Scene tracks are in {actual_frame} frame but {expected_frame} frame expected. "
            f"Use transform_scene_tracks_to_world() with camera extrinsics to convert."
        )


def transform_pose_camera_to_world(
    pose_camera: np.ndarray,
    camera_extrinsics: np.ndarray,
) -> np.ndarray:
    """Transform a 4x4 pose from camera frame to world frame.
    
    Args:
        pose_camera: (4, 4) pose in camera frame.
        camera_extrinsics: (4, 4) world_from_camera transform.
    
    Returns:
        (4, 4) pose in world frame.
    """
    pose_camera = np.asarray(pose_camera, dtype=np.float32)
    camera_extrinsics = np.asarray(camera_extrinsics, dtype=np.float32)
    
    assert pose_camera.shape == (4, 4), f"Expected (4,4) pose, got {pose_camera.shape}"
    assert camera_extrinsics.shape == (4, 4), f"Expected (4,4) extrinsics, got {camera_extrinsics.shape}"
    
    return camera_extrinsics @ pose_camera


def transform_entity_to_world(
    entity: SceneEntity3D,
    camera_extrinsics: np.ndarray,
) -> SceneEntity3D:
    """Transform entity pose from camera to world frame.
    
    Args:
        entity: Entity in camera frame.
        camera_extrinsics: (4, 4) world_from_camera transform.
    
    Returns:
        New entity with world-frame pose.
    """
    world_pose = transform_pose_camera_to_world(entity.pose, camera_extrinsics)
    
    # Transform joints if present
    world_joints = None
    if entity.joints_3d:
        world_joints = {}
        for joint_name, pos_camera in entity.joints_3d.items():
            pos_homog = np.append(pos_camera, 1.0)
            pos_world = camera_extrinsics @ pos_homog
            world_joints[joint_name] = pos_world[:3].astype(np.float32)
    
    return SceneEntity3D(
        entity_type=entity.entity_type,
        track_id=entity.track_id,
        pose=world_pose,
        scale=entity.scale,
        class_name=entity.class_name,
        mask_2d=entity.mask_2d,
        mask_logits=entity.mask_logits,
        geometry_handle=entity.geometry_handle,
        z_shape=entity.z_shape,
        z_tex=entity.z_tex,
        z_shape_ema=entity.z_shape_ema,
        z_tex_ema=entity.z_tex_ema,
        visibility=entity.visibility,
        occlusion_score=entity.occlusion_score,
        ir_loss=entity.ir_loss,
        joints_3d=world_joints,
    )


def transform_scene_tracks_to_world(
    scene_tracks: SceneTracks,
    camera_extrinsics: np.ndarray,
) -> SceneTracks:
    """Transform all entities in scene tracks from camera to world frame.
    
    Args:
        scene_tracks: Tracks in camera frame.
        camera_extrinsics: (4, 4) world_from_camera transform.
    
    Returns:
        New SceneTracks with world-frame poses.
    """
    camera_extrinsics = np.asarray(camera_extrinsics, dtype=np.float32)
    
    new_frames = []
    for frame_entities in scene_tracks.frames:
        new_entities = [
            transform_entity_to_world(entity, camera_extrinsics)
            for entity in frame_entities
        ]
        new_frames.append(new_entities)
    
    new_tracks = {}
    for track_id, track_history in scene_tracks.tracks.items():
        new_tracks[track_id] = [
            transform_entity_to_world(entity, camera_extrinsics)
            for entity in track_history
        ]
    
    # Update config to indicate world frame
    new_config = dict(scene_tracks.config_used or {})
    new_config["output_frame"] = "world"
    
    return SceneTracks(
        frames=new_frames,
        tracks=new_tracks,
        metrics=scene_tracks.metrics,
        config_used=new_config,
    )


def ensure_world_frame(
    scene_tracks: SceneTracks,
    camera_extrinsics: Optional[np.ndarray] = None,
) -> SceneTracks:
    """Ensure scene tracks are in world frame, transforming if needed.
    
    Args:
        scene_tracks: Input scene tracks.
        camera_extrinsics: (4, 4) world_from_camera transform, required if in camera frame.
    
    Returns:
        Scene tracks in world frame.
    
    Raises:
        FrameMismatchError: If in camera frame but no extrinsics provided.
    """
    # Check current frame
    current_frame = None
    if scene_tracks.config_used:
        current_frame = scene_tracks.config_used.get("output_frame")
    
    if current_frame == "world":
        return scene_tracks
    
    if current_frame == "camera":
        if camera_extrinsics is None:
            raise FrameMismatchError(
                "Scene tracks are in camera frame but no camera_extrinsics provided. "
                "Cannot transform to world frame without extrinsics. "
                "Either provide camera extrinsics or ensure upstream produces world-frame output."
            )
        return transform_scene_tracks_to_world(scene_tracks, camera_extrinsics)
    
    # Frame not specified - assume world
    logger.debug("No frame specified in scene tracks, assuming world frame")
    return scene_tracks
