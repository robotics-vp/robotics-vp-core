"""CPU-local geometry helpers for sim/synth/physics provider lanes."""

from .camera_geometry import (
    camera_intrinsics_from_fov,
    camera_intrinsics_from_mapping,
    camera_round_trip_error,
    compose_transforms,
    invert_transform,
    project_points,
    transform_from_mapping,
    unproject_depth,
)

__all__ = [
    "camera_intrinsics_from_fov",
    "camera_intrinsics_from_mapping",
    "camera_round_trip_error",
    "compose_transforms",
    "invert_transform",
    "project_points",
    "transform_from_mapping",
    "unproject_depth",
]
