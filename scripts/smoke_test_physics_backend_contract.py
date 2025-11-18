#!/usr/bin/env python3
"""
Smoke test for physics backend contract (pybullet + isaac stub).
"""
import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from src.physics.backends.factory import make_backend
from src.physics.backends.isaac_stub_backend import IsaacStubBackend


def main():
    # PyBullet backend
    backend = make_backend("pybullet", {"econ_preset": "toy"})
    obs = backend.reset(seed=0)
    assert isinstance(obs, dict)
    obs2, reward, done, info = backend.step([0.5, 0.5])
    assert isinstance(obs2, dict)
    assert isinstance(info, dict)
    assert isinstance(reward, float)
    state_summary = backend.get_state_summary()
    assert isinstance(state_summary, dict)
    frame = backend.build_vision_frame(task_id="task", episode_id="ep", timestep=0)
    assert frame.backend == "pybullet"
    assert frame.state_digest
    print(f"[smoke_test_physics_backend_contract] pybullet step reward={reward} done={done}")

    # Isaac stub backend should raise on control but expose vision frame
    stub = IsaacStubBackend()
    try:
        stub.reset()
    except NotImplementedError:
        pass
    else:
        raise AssertionError("Isaac stub must raise NotImplementedError")
    frame_stub = stub.build_vision_frame(task_id="task", episode_id="ep", timestep=0)
    assert frame_stub.backend in ("isaac_stub", "isaac")
    assert frame_stub.state_digest

    # Factory should plumb isaac alias to stub
    alias_backend = make_backend("isaac", {"econ_preset": "toy"})
    assert alias_backend.backend_name in ("isaac_stub", "isaac")

    print("[smoke_test_physics_backend_contract] All tests passed.")


if __name__ == "__main__":
    main()
