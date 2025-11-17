#!/usr/bin/env python3
"""
Backend parity smoke: ensure PyBullet and Isaac stubs expose expected API.
"""
from src.envs.drawer_vase_physics_env import DrawerVasePhysicsEnv, DrawerVaseConfig, summarize_drawer_vase_episode
from src.envs.physics.backend_factory import make_backend


def check_backend(backend):
    assert hasattr(backend, "reset")
    assert hasattr(backend, "step")
    assert hasattr(backend, "get_episode_info")
    assert hasattr(backend, "engine_type")
    print("Methods present for", backend.engine_type)
    return True


def main():
    env = DrawerVasePhysicsEnv(DrawerVaseConfig(), obs_mode="state", render_mode=None)
    pyb = make_backend("pybullet", env=env, env_name="drawer_vase", summarize_fn=summarize_drawer_vase_episode)
    check_backend(pyb)
    try:
        isaac = make_backend("isaac", env_name="drawer_vase")
        check_backend(isaac)
        isaac.reset()
    except NotImplementedError:
        print("Isaac backend stubbed correctly (NotImplementedError as expected).")


if __name__ == "__main__":
    main()
