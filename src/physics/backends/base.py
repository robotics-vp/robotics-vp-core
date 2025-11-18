"""
Physics backend interface for future multi-engine support.
"""
from typing import Any, Dict, Tuple, Optional


class PhysicsBackend:
    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """Reset environment; returns initial observation summary."""
        raise NotImplementedError

    def step(self, action: Any) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Returns:
          obs_summary: Dict (JSON-safe)
          reward: float
          done: bool
          info: Dict (may contain backend-specific fields)
        """
        raise NotImplementedError

    def get_state_summary(self) -> Dict[str, Any]:
        """Return a compact JSON-safe snapshot of state suitable for logging."""
        raise NotImplementedError

    @property
    def backend_name(self) -> str:
        return self.__class__.__name__
