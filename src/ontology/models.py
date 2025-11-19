"""
Ontology spine models (Phase A).

Lightweight dataclasses for tasks, robots, datapacks, episodes, econ vectors,
and per-step events. JSON-serializable and deterministic.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass
class Task:
    task_id: str
    name: str
    description: Optional[str] = None
    environment_id: str = ""
    human_mpl_units_per_hour: float = 0.0
    human_wage_per_hour: float = 0.0
    default_energy_cost_per_wh: float = 0.0
    metadata: Dict = field(default_factory=dict)


@dataclass
class Robot:
    robot_id: str
    name: str
    hardware_profile: Dict = field(default_factory=dict)
    energy_cost_per_wh: Optional[float] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class Datapack:
    datapack_id: str
    source_type: str        # "human_video" | "synthetic_video" | "physics" | ...
    task_id: str
    modality: str           # "video" | "state" | "mixed" | ...
    storage_uri: str        # pointer into filesystem / bucket
    novelty_score: float = 0.0
    quality_score: float = 0.0
    tags: Dict = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)
    sima2_backend_id: Optional[str] = None
    sima2_model_version: Optional[str] = None
    sima2_task_spec: Dict[str, Any] = field(default_factory=dict)
    auditor_rating: Optional[str] = None
    auditor_score: Optional[float] = None
    auditor_predicted_econ: Optional[Dict[str, float]] = None
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Episode:
    episode_id: str
    task_id: str
    robot_id: str
    datapack_id: Optional[str] = None
    started_at: datetime = field(default_factory=datetime.utcnow)
    ended_at: Optional[datetime] = None
    status: str = "running"  # "running" | "success" | "failure"
    metadata: Dict = field(default_factory=dict)
    vision_config: Dict = field(default_factory=dict)
    vision_conditions: Dict = field(default_factory=dict)
    sima2_backend_id: Optional[str] = None
    sima2_model_version: Optional[str] = None
    sima2_task_spec: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EconVector:
    episode_id: str
    mpl_units_per_hour: float
    wage_parity: float
    energy_cost: float
    damage_cost: float
    novelty_delta: float
    reward_scalar_sum: float
    mobility_penalty: float = 0.0
    precision_bonus: float = 0.0
    stability_risk_score: float = 0.0
    components: Dict[str, float] = field(default_factory=dict)
    source_domain: str = "pybullet"  # "pybullet", "isaac", "real_lab"
    calibration_version: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class EpisodeEvent:
    episode_id: str
    timestep: int
    event_type: str          # "step" | "collision" | "reward" | "success" | ...
    timestamp: datetime
    reward_scalar: float
    reward_components: Dict[str, float] = field(default_factory=dict)
    state_summary: Dict = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)
