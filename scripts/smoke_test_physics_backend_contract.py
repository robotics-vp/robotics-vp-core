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

    print(f"[smoke_test_physics_backend_contract] pybullet step reward={reward} done={done}")

    # Isaac stub backend should raise
    stub = IsaacStubBackend()
    try:
        stub.reset()
    except NotImplementedError:
        pass
    else:
        raise AssertionError("Isaac stub must raise NotImplementedError")

    print("[smoke_test_physics_backend_contract] All tests passed.")


if __name__ == "__main__":
    main()
