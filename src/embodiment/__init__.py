"""Embodiment module: contacts, affordances, and econ-aware diagnostics."""
from src.embodiment.config import EmbodimentConfig
from src.embodiment.core import EmbodimentInputs, EmbodimentResult, compute_embodiment
from src.embodiment.datapack_adapter import embodiment_profile_from_summary
from src.embodiment.artifacts import (
    EMBODIMENT_PROFILE_PREFIX,
    AFFORDANCE_GRAPH_PREFIX,
    SKILL_SEGMENTS_PREFIX,
    EmbodimentSummary,
    EmbodimentProfileArtifact,
    AffordanceGraphArtifact,
    SkillSegmentsArtifact,
)

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
