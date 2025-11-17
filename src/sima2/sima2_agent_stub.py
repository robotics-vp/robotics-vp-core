"""
SIMA-2 agent stub (placeholder only).
"""
from typing import Dict, Any


class Sima2AgentStub:
    """
    Returns synthetic semantic and skill rollouts with confidence.
    """

    def run_episode(self, observation: Any = None, instruction: str = "") -> Dict[str, Any]:
        return {
            "semantic_rollout": ["observe", "plan", "act"],
            "skill_rollout": [0, 1, 2],
            "confidence": 0.5,
        }
