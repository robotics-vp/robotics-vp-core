"""
DataPack Schema: Two-Bucket Taxonomy for Data Valuation.

Provides dataclasses for:
- ConditionProfile: Environment/economic conditions
- AttributionProfile: Impact attribution (ΔMPL, Δerror, ΔJ, trust, w_econ, λ)
- SimaAnnotation: SIMA-2 co-agent language annotations
- DataPackMeta: Complete datapack metadata (positive/negative bucket)

All schemas are engine-agnostic (PyBullet, Isaac Gym, UE5).
"""

import json
import uuid
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Literal
from datetime import datetime


@dataclass
class ConditionProfile:
    """
    Environmental and economic conditions under which data was collected.

    Engine-aware for multi-world support (PyBullet, Isaac Gym, UE5).
    """
    # Task and engine
    task_name: str = "drawer_vase"
    engine_type: Literal["pybullet", "isaac", "ue5"] = "pybullet"
    world_id: str = "pyb_drawer_v1"

    # Environment configuration (drawer_vase specific, but extensible)
    vase_offset: tuple = (0.0, 0.0, 0.0)  # (x, y, z) offset from default
    drawer_friction: float = 0.3
    lighting_profile: str = "normal"  # "normal", "low_light", "high_contrast"
    occlusion_level: float = 0.0  # 0 = no occlusion, 1 = full occlusion

    # Economic regime
    econ_preset: str = "drawer_vase"
    price_per_unit: float = 5.0
    vase_break_cost: float = 50.0
    energy_price_kWh: float = 0.12

    # Objective vector at the time
    objective_vector: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    # [α_mpl, α_error, α_energy, α_safety, ...]

    # Additional tags
    tags: Dict[str, Any] = field(default_factory=dict)
    # e.g., {"fragile": True, "multi_object": False, "has_occlusion": False}

    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        # Convert tuple to list for JSON
        d['vase_offset'] = list(self.vase_offset)
        return d

    @classmethod
    def from_dict(cls, d):
        """Create from dictionary."""
        # Convert list back to tuple
        if 'vase_offset' in d and isinstance(d['vase_offset'], list):
            d['vase_offset'] = tuple(d['vase_offset'])
        return cls(**d)


@dataclass
class AttributionProfile:
    """
    Impact attribution for the datapack.

    Tracks how this data affected key metrics: MPL, error rate, EP, J.
    Includes gating signals (trust, w_econ, λ) and world model metadata.
    """
    # Core impact deltas (vs baseline)
    delta_mpl: float = 0.0  # Change in marginal product of labor
    delta_error: float = 0.0  # Change in error rate
    delta_ep: float = 0.0  # Change in energy productivity (MPL/Wh)
    delta_wage_parity: float = 0.0  # Change in wage parity
    delta_J: float = 0.0  # Change in meta-objective J

    # Gating signals from Phase B
    trust_score: float = 0.0  # From trust_net
    w_econ: float = 0.0  # From J-trained lattice
    lambda_budget: float = 0.0  # λ synthetic budget used

    # World model metadata
    world_model_horizon: int = 0
    world_model_trust_over_horizon: List[float] = field(default_factory=list)

    # Data source type (for world model integration)
    source_type: Literal["real", "synthetic", "hybrid"] = "real"
    wm_model_id: Optional[str] = None
    wm_horizon_used: Optional[int] = None
    wm_branch_depth: Optional[int] = None
    wm_trust_over_horizon: Optional[List[float]] = None

    # Visual latent statistics (for z_V)
    zv_stats: Optional[Dict[str, float]] = None
    # e.g., {"mean": 0.1, "std": 0.05, "norm": 1.2}

    # Marginal value-of-data estimate
    mvd_score: Optional[float] = None  # Estimated marginal value

    # Final sampling weight used
    econ_weight_final: Optional[float] = None  # trust * w_econ (or scaled)

    # Training run tracking
    used_in_training_runs: List[str] = field(default_factory=list)

    # World model role
    wm_role: Optional[Literal["wm_train", "wm_eval", "wm_synth_source", "wm_synth_target"]] = None

    # Optional per-skill contributions
    skill_contribs: Dict[int, Dict[str, float]] = field(default_factory=dict)
    # e.g., {4: {"delta_mpl": 3.2, "delta_error": -0.05}}

    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        """Create from dictionary."""
        return cls(**d)

    def compute_quality_score(self):
        """Compute overall quality score from trust and w_econ."""
        return self.trust_score * self.w_econ


@dataclass
class SimaAnnotation:
    """
    SIMA-2 co-agent language annotations.

    Captures natural language instruction, step-level narrations,
    and derived skill plans for training VLA.
    """
    instruction: str = ""  # High-level natural language instruction
    step_narrations: List[str] = field(default_factory=list)  # Per-step narrations
    sima_agent_id: str = "sima_v1"  # Which SIMA model/config
    source_world: str = "pyb_drawer_v1"  # World where trajectory was collected

    # Optional: derived skill plan inferred from SIMA trajectory
    derived_skill_plan: List[int] = field(default_factory=list)

    # Metadata
    narration_count: int = 0
    average_narration_length: float = 0.0

    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        """Create from dictionary."""
        return cls(**d)

    def compute_stats(self):
        """Compute narration statistics."""
        self.narration_count = len(self.step_narrations)
        if self.step_narrations:
            self.average_narration_length = sum(
                len(n) for n in self.step_narrations
            ) / len(self.step_narrations)


@dataclass
class DataPackMeta:
    """
    Complete datapack metadata for the two-bucket taxonomy.

    Bucket types:
    - positive: Data that improved J/MPL/error/EP
    - negative: Data that worsened J/MPL/error/EP (with counterfactual plan)
    """
    # Identification
    pack_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_name: str = "drawer_vase"
    bucket: Literal["positive", "negative"] = "positive"

    # Semantic tags for filtering
    semantic_tags: List[str] = field(default_factory=list)
    # e.g., ["fragile glassware", "multi-object", "low-light", "top drawer"]

    # Core profiles
    condition: ConditionProfile = field(default_factory=ConditionProfile)
    attribution: AttributionProfile = field(default_factory=AttributionProfile)

    # Skill trace (time-ordered skill usage)
    skill_trace: List[Dict[str, Any]] = field(default_factory=list)
    # Each element: {
    #   "t": int,  # timestep
    #   "skill_id": int,
    #   "params": Dict,  # skill parameters
    #   "duration": int,  # steps for this skill
    #   "local_metrics": Dict[str, float]  # local ΔMPL, Δerror, etc.
    # }

    # SIMA annotation (optional)
    sima_annotation: Optional[SimaAnnotation] = None

    # For negative datapacks: counterfactual plan
    counterfactual_plan: Optional[Dict[str, Any]] = None
    # e.g., {
    #   "skills": [0, 1, 2, 3, 4, 5],
    #   "waypoints": [[x, y, z], ...],
    #   "source": "hrl_teacher"
    # }
    counterfactual_source: Optional[str] = None
    # "scripted_teacher", "hrl_teacher", "vla_planner", "sima_teacher"

    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    episode_id: Optional[str] = None
    episode_index: Optional[int] = None

    # Raw data references (for reconstruction)
    raw_data_path: Optional[str] = None  # Path to underlying npz/episode data

    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        d = {
            'pack_id': self.pack_id,
            'task_name': self.task_name,
            'bucket': self.bucket,
            'semantic_tags': self.semantic_tags,
            'condition': self.condition.to_dict(),
            'attribution': self.attribution.to_dict(),
            'skill_trace': self.skill_trace,
            'sima_annotation': self.sima_annotation.to_dict() if self.sima_annotation else None,
            'counterfactual_plan': self.counterfactual_plan,
            'counterfactual_source': self.counterfactual_source,
            'created_at': self.created_at,
            'episode_id': self.episode_id,
            'episode_index': self.episode_index,
            'raw_data_path': self.raw_data_path,
        }
        return d

    @classmethod
    def from_dict(cls, d):
        """Create from dictionary."""
        condition = ConditionProfile.from_dict(d['condition'])
        attribution = AttributionProfile.from_dict(d['attribution'])

        sima_annotation = None
        if d.get('sima_annotation'):
            sima_annotation = SimaAnnotation.from_dict(d['sima_annotation'])

        return cls(
            pack_id=d['pack_id'],
            task_name=d['task_name'],
            bucket=d['bucket'],
            semantic_tags=d['semantic_tags'],
            condition=condition,
            attribution=attribution,
            skill_trace=d['skill_trace'],
            sima_annotation=sima_annotation,
            counterfactual_plan=d.get('counterfactual_plan'),
            counterfactual_source=d.get('counterfactual_source'),
            created_at=d.get('created_at', datetime.now().isoformat()),
            episode_id=d.get('episode_id'),
            episode_index=d.get('episode_index'),
            raw_data_path=d.get('raw_data_path'),
        )

    def to_json(self):
        """Convert to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str):
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def get_skill_ids(self):
        """Extract skill ID sequence from skill trace."""
        return [entry['skill_id'] for entry in self.skill_trace]

    def has_skill(self, skill_id):
        """Check if a specific skill was used."""
        return skill_id in self.get_skill_ids()

    def get_total_duration(self):
        """Get total duration of skill trace."""
        return sum(entry.get('duration', 0) for entry in self.skill_trace)

    def matches_condition_filters(self, filters):
        """
        Check if this datapack matches given condition filters.

        Args:
            filters: Dict of filter conditions

        Returns:
            bool: Whether all filters match
        """
        for key, value in filters.items():
            # Check in condition tags
            if key in self.condition.tags:
                if self.condition.tags[key] != value:
                    return False
            # Check semantic tags
            elif isinstance(value, bool) and value:
                if key not in self.semantic_tags:
                    return False
            # Check numeric thresholds
            elif key.endswith('_min'):
                attr_name = key[:-4]
                if hasattr(self.condition, attr_name):
                    if getattr(self.condition, attr_name) < value:
                        return False
            elif key.endswith('_max'):
                attr_name = key[:-4]
                if hasattr(self.condition, attr_name):
                    if getattr(self.condition, attr_name) > value:
                        return False
            # Direct attribute check
            elif hasattr(self.condition, key):
                if getattr(self.condition, key) != value:
                    return False

        return True

    def summary(self):
        """Get human-readable summary."""
        return (
            f"DataPack[{self.pack_id[:8]}] "
            f"{self.bucket.upper()} | "
            f"task={self.task_name} | "
            f"ΔJ={self.attribution.delta_J:.4f} | "
            f"trust={self.attribution.trust_score:.4f} | "
            f"skills={len(self.skill_trace)} | "
            f"tags={self.semantic_tags}"
        )


def create_positive_datapack(
    task_name,
    condition,
    attribution,
    skill_trace,
    semantic_tags=None,
    sima_annotation=None,
    episode_id=None
):
    """
    Factory function to create a positive datapack.

    Args:
        task_name: Task identifier
        condition: ConditionProfile
        attribution: AttributionProfile (with positive ΔJ)
        skill_trace: List of skill trace entries
        semantic_tags: Optional list of tags
        sima_annotation: Optional SimaAnnotation
        episode_id: Optional episode identifier

    Returns:
        DataPackMeta: Positive bucket datapack
    """
    return DataPackMeta(
        task_name=task_name,
        bucket="positive",
        condition=condition,
        attribution=attribution,
        skill_trace=skill_trace,
        semantic_tags=semantic_tags or [],
        sima_annotation=sima_annotation,
        episode_id=episode_id
    )


def create_negative_datapack(
    task_name,
    condition,
    attribution,
    skill_trace,
    counterfactual_plan,
    counterfactual_source,
    semantic_tags=None,
    sima_annotation=None,
    episode_id=None
):
    """
    Factory function to create a negative datapack with counterfactual.

    Args:
        task_name: Task identifier
        condition: ConditionProfile
        attribution: AttributionProfile (with negative ΔJ)
        skill_trace: List of skill trace entries (what went wrong)
        counterfactual_plan: Dict with correct plan
        counterfactual_source: Source of counterfactual
        semantic_tags: Optional list of tags
        sima_annotation: Optional SimaAnnotation
        episode_id: Optional episode identifier

    Returns:
        DataPackMeta: Negative bucket datapack with counterfactual
    """
    return DataPackMeta(
        task_name=task_name,
        bucket="negative",
        condition=condition,
        attribution=attribution,
        skill_trace=skill_trace,
        counterfactual_plan=counterfactual_plan,
        counterfactual_source=counterfactual_source,
        semantic_tags=semantic_tags or [],
        sima_annotation=sima_annotation,
        episode_id=episode_id
    )
