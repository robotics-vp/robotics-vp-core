"""Embodiment module: contacts, affordances, and econ-aware diagnostics."""
from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from src.embodiment.config import EmbodimentConfig
from src.embodiment.artifacts import (
    EMBODIMENT_PROFILE_PREFIX,
    AFFORDANCE_GRAPH_PREFIX,
    SKILL_SEGMENTS_PREFIX,
    EmbodimentSummary,
    EmbodimentProfileArtifact,
    AffordanceGraphArtifact,
    SkillSegmentsArtifact,
)

if TYPE_CHECKING:
    from src.embodiment.core import EmbodimentInputs, EmbodimentResult, compute_embodiment
    from src.embodiment.datapack_adapter import embodiment_profile_from_summary

__all__ = [
    "EmbodimentConfig",
    "EmbodimentInputs",
    "EmbodimentResult",
    "compute_embodiment",
    "embodiment_profile_from_summary",
    "EMBODIMENT_PROFILE_PREFIX",
    "AFFORDANCE_GRAPH_PREFIX",
    "SKILL_SEGMENTS_PREFIX",
    "EmbodimentSummary",
    "EmbodimentProfileArtifact",
    "AffordanceGraphArtifact",
    "SkillSegmentsArtifact",
]

_LAZY_ATTRS = {
    "EmbodimentInputs": "src.embodiment.core",
    "EmbodimentResult": "src.embodiment.core",
    "compute_embodiment": "src.embodiment.core",
    "embodiment_profile_from_summary": "src.embodiment.datapack_adapter",
}


def __getattr__(name: str):
    if name in _LAZY_ATTRS:
        module = importlib.import_module(_LAZY_ATTRS[name])
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
