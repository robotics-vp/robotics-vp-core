import numpy as np
import pytest

from src.world_model.sim_synth_physics.utils.camera_geometry import (
    camera_intrinsics_from_fov,
    compose_transforms,
    invert_transform,
    project_points,
    unproject_depth,
)


def test_camera_intrinsics_from_fov_builds_centered_pinhole_matrix() -> None:
    intrinsics = camera_intrinsics_from_fov(640, 480, 90.0)

    assert intrinsics.shape == (3, 3)
    assert intrinsics[0, 0] == pytest.approx(320.0)
    assert intrinsics[1, 1] == pytest.approx(320.0)
    assert intrinsics[0, 2] == pytest.approx(319.5)
    assert intrinsics[1, 2] == pytest.approx(239.5)


def test_projection_round_trip_with_extrinsics() -> None:
    intrinsics = camera_intrinsics_from_fov(
        3,
        3,
        90.0,
        principal_point=(1.0, 1.0),
    )
    camera_to_world = np.eye(4, dtype=np.float64)
    camera_to_world[:3, 3] = np.asarray([1.0, -2.0, 0.5])
    world_to_camera = invert_transform(camera_to_world)
    depth = np.ones((3, 3), dtype=np.float64)

    world_points = unproject_depth(depth, intrinsics, camera_to_world=camera_to_world)
    projected = project_points(
        world_points.reshape(-1, 3),
        intrinsics,
        world_to_camera=world_to_camera,
    )

    expected_pixels = np.asarray(
        [[x, y] for y in range(3) for x in range(3)],
        dtype=np.float64,
    )
    assert projected[:, :2] == pytest.approx(expected_pixels)
    assert projected[:, 2] == pytest.approx(np.ones(9))


def test_compose_transforms_matches_matrix_chain() -> None:
    first = np.eye(4, dtype=np.float64)
    first[:3, 3] = np.asarray([1.0, 0.0, 0.0])
    second = np.eye(4, dtype=np.float64)
    second[:3, 3] = np.asarray([0.0, 2.0, 0.0])

    composed = compose_transforms(first, second)

    assert composed[:3, 3] == pytest.approx(np.asarray([1.0, 2.0, 0.0]))
