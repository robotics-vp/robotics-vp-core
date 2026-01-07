from __future__ import annotations

import importlib.util
import math

import pytest


def _mujoco_available() -> bool:
    return importlib.util.find_spec("mujoco") is not None


@pytest.mark.skipif(not _mujoco_available(), reason="mujoco not installed")
def test_workcell_mujoco_contact_metrics() -> None:
    from src.envs.workcell_env.physics.mujoco_adapter import MujocoPhysicsAdapter
    from src.envs.workcell_env.scene.scene_spec import FixtureSpec, PartSpec, WorkcellSceneSpec

    scene_spec = WorkcellSceneSpec(
        workcell_id="mujoco_contact_test",
        fixtures=[
            FixtureSpec(
                id="fixture_0",
                position=(0.0, 0.0, 0.06),
                orientation=(1.0, 0.0, 0.0, 0.0),
                fixture_type="vise",
            )
        ],
        parts=[
            PartSpec(
                id="part_0",
                position=(0.0, 0.0, 0.08),
                orientation=(1.0, 0.0, 0.0, 0.0),
                part_type="peg",
                dimensions_mm=(40.0, 40.0, 40.0),
            )
        ],
        spatial_bounds=(1.0, 1.0, 1.0),
    )

    adapter = MujocoPhysicsAdapter(spatial_bounds=scene_spec.spatial_bounds)
    adapter.reset(scene_spec, seed=123)
    for _ in range(10):
        adapter.step(0.01)

    state = adapter.get_state()
    assert state.get("collision_count", 0) >= 0
    assert math.isfinite(state.get("contact_force_N", 0.0))
    assert adapter.check_collision("fixture_0", "part_0")


@pytest.mark.skipif(not _mujoco_available(), reason="mujoco not installed")
def test_workcell_mujoco_determinism_hashes_match() -> None:
    from src.envs.workcell_env.physics.mujoco_adapter import MujocoPhysicsAdapter
    from src.envs.workcell_env.scene.scene_spec import PartSpec, WorkcellSceneSpec
    from src.envs.workcell_env.utils.determinism import hash_state

    scene_spec = WorkcellSceneSpec(
        workcell_id="mujoco_determinism",
        parts=[
            PartSpec(
                id="part_0",
                position=(0.1, 0.0, 0.2),
                orientation=(1.0, 0.0, 0.0, 0.0),
                part_type="bolt",
                dimensions_mm=(30.0, 30.0, 30.0),
            )
        ],
        spatial_bounds=(1.0, 1.0, 1.0),
    )

    adapter_a = MujocoPhysicsAdapter(spatial_bounds=scene_spec.spatial_bounds)
    adapter_b = MujocoPhysicsAdapter(spatial_bounds=scene_spec.spatial_bounds)
    adapter_a.reset(scene_spec, seed=42)
    adapter_b.reset(scene_spec, seed=42)

    hashes_a = []
    hashes_b = []
    for _ in range(5):
        adapter_a.step(0.01)
        adapter_b.step(0.01)
        hashes_a.append(hash_state(adapter_a.get_state()))
        hashes_b.append(hash_state(adapter_b.get_state()))

    assert hashes_a == hashes_b

