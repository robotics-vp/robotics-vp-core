"""Tests for defensive MuJoCo renderer cleanup."""
from __future__ import annotations

from src.envs.workcell_env.physics.mujoco_adapter import _safe_close_renderer


class _RendererWithoutMjrContext:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_safe_close_renderer_handles_missing_mjr_context() -> None:
    renderer = _RendererWithoutMjrContext()
    assert not hasattr(renderer, "_mjr_context")

    _safe_close_renderer(renderer)

    assert renderer.closed is True
    assert hasattr(renderer, "_mjr_context")

