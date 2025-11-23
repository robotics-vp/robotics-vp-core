"""
Observation adapter package.

Holds canonical observation models and adapter utilities for RL/policy stacks.
"""

from src.observation.models import (
    Observation,
    VisionSlice,
    SemanticSlice,
    EconSlice,
    RecapSlice,
    ControlSlice,
)
from src.observation.condition_vector import ConditionVector
from src.observation.condition_vector_builder import ConditionVectorBuilder
from src.observation.adapter import ObservationAdapter

__all__ = [
    "Observation",
    "VisionSlice",
    "SemanticSlice",
    "EconSlice",
    "RecapSlice",
    "ControlSlice",
    "ConditionVector",
    "ConditionVectorBuilder",
    "ObservationAdapter",
]
