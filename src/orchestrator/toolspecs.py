from dataclasses import dataclass
from typing import Dict, Any, Optional, Literal, List

ToolName = Literal[
    "SET_ENERGY_PROFILE",     # choose energy profile or continuous knobs
    "SET_OBJECTIVE_PRESET",   # throughput / energy_saver / safety_first / balanced
    "SET_BACKEND",            # pybullet / isaac / ue5
    "SET_DATA_MIX",           # weights over real / synthetic / hybrid
    "QUERY_DATAPACKS",        # filter datapacks for context
    "QUERY_ENERGY_SURFACE",   # query EnergyResponseModel for candidate profiles
]


@dataclass
class ToolCall:
    name: ToolName
    args: Dict[str, Any]


@dataclass
class ToolObservation:
    name: ToolName
    result: Dict[str, Any]


@dataclass
class OrchestrationStep:
    instruction: str
    objective_vector: List[float]  # weights on MPL / error / energy / safety / novelty
    backend_id: str
    env_name: str
    tool_call: ToolCall
    observation: Optional[ToolObservation] = None
