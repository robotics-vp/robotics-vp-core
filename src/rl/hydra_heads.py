"""
Hydra-style shared trunk + skill-specific heads.

Deterministic head selection keyed by ConditionVector.skill_mode with optional
default fallback. Reward/econ math untouched: heads only emit action/value
distributions.
"""
from typing import Dict, Optional

import torch
import torch.nn as nn

from src.observation.condition_vector import ConditionVector


def _call_module(module: nn.Module, *args):
    """Invoke module, tolerating absence of condition arguments."""
    try:
        return module(*args)
    except TypeError:
        # Gracefully retry without the last arg (assumed condition)
        if len(args) > 1:
            return module(*args[:-1])
        raise


class HydraActor(nn.Module):
    """
    Shared trunk + multiple skill heads.

    Head selection is deterministic based on condition.skill_mode; optional
    default fallback preserves behavior when tags are missing.
    """

    def __init__(
        self,
        trunk: nn.Module,
        heads: Dict[str, nn.Module],
        default_skill_mode: Optional[str] = None,
        strict: bool = False,
    ) -> None:
        super().__init__()
        self.trunk = trunk
        self.heads = nn.ModuleDict(heads)
        self.default_skill_mode = default_skill_mode
        self.strict = strict

    def forward(self, obs, condition: Optional[ConditionVector] = None):
        head_key = self._resolve_head_key(condition)
        if head_key not in self.heads:
            if self.strict:
                raise KeyError(f"Missing Hydra head for skill_mode={head_key}")
            head_key = self.default_skill_mode or (sorted(self.heads.keys()) or [None])[0]
        head = self.heads.get(head_key)
        if head is None:
            raise KeyError("HydraActor has no registered heads.")

        trunk_features = _call_module(self.trunk, obs, condition)
        return _call_module(head, trunk_features, condition)

    def _resolve_head_key(self, condition: Optional[ConditionVector]) -> str:
        if condition and getattr(condition, "skill_mode", None):
            return str(condition.skill_mode)
        if self.default_skill_mode:
            return self.default_skill_mode
        return sorted(self.heads.keys())[0]


class HydraCritic(nn.Module):
    """
    Shared trunk + skill-specific value heads.
    """

    def __init__(
        self,
        trunk: nn.Module,
        heads: Dict[str, nn.Module],
        default_skill_mode: Optional[str] = None,
        strict: bool = False,
    ) -> None:
        super().__init__()
        self.trunk = trunk
        self.heads = nn.ModuleDict(heads)
        self.default_skill_mode = default_skill_mode
        self.strict = strict

    def forward(self, obs, condition: Optional[ConditionVector] = None):
        head_key = self._resolve_head_key(condition)
        if head_key not in self.heads:
            if self.strict:
                raise KeyError(f"Missing Hydra critic head for skill_mode={head_key}")
            head_key = self.default_skill_mode or (sorted(self.heads.keys()) or [None])[0]
        head = self.heads.get(head_key)
        if head is None:
            raise KeyError("HydraCritic has no registered heads.")

        trunk_features = _call_module(self.trunk, obs, condition)
        return _call_module(head, trunk_features, condition)

    def _resolve_head_key(self, condition: Optional[ConditionVector]) -> str:
        if condition and getattr(condition, "skill_mode", None):
            return str(condition.skill_mode)
        if self.default_skill_mode:
            return self.default_skill_mode
        return sorted(self.heads.keys())[0]


class HydraPolicy(nn.Module):
    """
    Lightweight wrapper for a fixed head selection (e.g., registry returns a partial).
    """

    def __init__(self, trunk: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self.trunk = trunk
        self.head = head

    def forward(self, obs, condition: Optional[ConditionVector] = None):
        trunk_features = _call_module(self.trunk, obs, condition)
        return _call_module(self.head, trunk_features, condition)
