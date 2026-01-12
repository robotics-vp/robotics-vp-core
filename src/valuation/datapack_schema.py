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

from src.utils.json_safe import to_json_safe
from src.valuation.guidance_profile import GuidanceProfile

# Unified schema version - aligns with existing 2.0-energy format
DATAPACK_SCHEMA_VERSION = "2.0-energy"
DATAPACK_SCHEMA_VERSION_PORTABLE = "2.1-portable"
DATAPACK_SCHEMA_VERSION_REPR = "2.2-repr"


@dataclass
class EnergyProfile:
    """
    Energy breakdown for 2.0-energy schema compatibility.

    Captures per-limb, per-skill, per-joint, and per-effector energy usage.
    """
    total_Wh: float = 0.0
    Wh_per_unit: float = 0.0
    Wh_per_hour: float = 0.0

    # Hierarchical energy breakdown (2.0-energy fields)
    energy_per_limb: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # e.g., {"shoulder": {"Wh": 0.5, "fraction": 0.3}, ...}

    energy_per_skill: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # e.g., {"grasp": {"Wh": 0.2, "fraction": 0.15}, ...}

    energy_per_joint: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # e.g., {"joint_0": {"Wh": 0.1, "torque_integral": 5.0}, ...}

    energy_per_effector: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # e.g., {"gripper": {"Wh": 0.05, "activation_time": 10.0}, ...}

    coordination_metrics: Dict[str, float] = field(default_factory=dict)
    # e.g., {"mean_active_joints": 2.5, "peak_power": 10.0, ...}

    # Legacy fields for compatibility
    limb_energy_Wh: Dict[str, float] = field(default_factory=dict)
    skill_energy_Wh: Dict[str, float] = field(default_factory=dict)

    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        """Create from dictionary."""
        return cls(**d)

    @classmethod
    def from_episode_metrics(cls, metrics: Dict[str, Any]):
        """Create from EpisodeInfoSummary metrics dict."""
        return cls(
            total_Wh=metrics.get("energy_Wh", 0.0),
            Wh_per_unit=metrics.get("energy_Wh_per_unit", 0.0),
            Wh_per_hour=metrics.get("energy_Wh_per_hour", 0.0),
            energy_per_limb=metrics.get("energy_per_limb", {}),
            energy_per_skill=metrics.get("energy_per_skill", {}),
            energy_per_joint=metrics.get("energy_per_joint", {}),
            energy_per_effector=metrics.get("energy_per_effector", {}),
            coordination_metrics=metrics.get("coordination_metrics", {}),
            limb_energy_Wh=metrics.get("limb_energy_Wh", {}),
            skill_energy_Wh=metrics.get("skill_energy_Wh", {}),
        )


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

    # Optional attribution breakdowns (all default to 0.0 for backward compatibility)
    delta_mpl_model: float = 0.0
    delta_mpl_data: float = 0.0
    delta_mpl_energy: float = 0.0

    # Economic spread/rebate metrics (no profit)
    rebate_pct: float = 0.0
    attributable_spread_capture: float = 0.0
    data_premium: float = 0.0

    # Data tier (0=redundant, 1=context-novel, 2=causal-novel/frontier)
    tier: int = 1

    # Additional fields for compatibility
    env_name: Optional[str] = None
    engine_type: Optional[str] = None

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

    # Optional: VLA crossover references
    vla_references: Optional[Dict[str, Any]] = None

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
class ProcessRewardProfile:
    """
    Process reward metrics for the datapack.

    Captures PBRS potential, fusion confidence, and diagnostic signals
    from the process reward module. Used for:
    - Episode filtering/gating by confidence
    - Sampling weight computation
    - Data quality analysis
    """
    # Core PBRS metrics
    phi_star_mean: float = 0.0  # Mean fused potential
    phi_star_final: float = 0.0  # Final potential value
    phi_star_delta: float = 0.0  # Progress: final - initial

    # Confidence metrics
    conf_mean: float = 0.5  # Mean fusion confidence
    conf_p10: float = 0.5  # 10th percentile (conservative estimate)
    conf_min: float = 0.5  # Minimum confidence

    # Shaped reward
    r_shape_sum: float = 0.0  # Sum of shaped rewards

    # Fusion diagnostics
    disagreement_mean: float = 0.0  # Mean perspective disagreement
    disagreement_max: float = 0.0  # Max disagreement
    entropy_mean: float = 0.0  # Mean fusion entropy
    entropy_max: float = 0.0  # Max entropy

    # Configuration snapshot
    phi_B_disabled: bool = False  # Was backward perspective disabled?
    orchestrator_override: Optional[Dict[str, Any]] = None  # FusionOverride used

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ProcessRewardProfile":
        """Create from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_episode_output(cls, result: Any) -> "ProcessRewardProfile":
        """Create from ProcessRewardEpisodeOutput.

        Args:
            result: ProcessRewardEpisodeOutput from process_reward_episode()

        Returns:
            ProcessRewardProfile with extracted metrics.
        """
        import numpy as np

        phi_star = result.phi_star
        conf = result.conf

        return cls(
            phi_star_mean=float(np.mean(phi_star)),
            phi_star_final=float(phi_star[-1]) if len(phi_star) > 0 else 0.0,
            phi_star_delta=float(phi_star[-1] - phi_star[0]) if len(phi_star) > 0 else 0.0,
            conf_mean=float(np.mean(conf)),
            conf_p10=float(np.percentile(conf, 10)) if len(conf) > 0 else 0.5,
            conf_min=float(np.min(conf)) if len(conf) > 0 else 0.5,
            r_shape_sum=float(np.sum(result.r_shape)),
            disagreement_mean=float(np.mean(result.diagnostics.disagreement)),
            disagreement_max=float(np.max(result.diagnostics.disagreement)),
            entropy_mean=float(np.mean(result.diagnostics.entropy)),
            entropy_max=float(np.max(result.diagnostics.entropy)),
            phi_B_disabled=result.metadata.get("phi_B_disabled", False),
            orchestrator_override=None,  # Set separately if needed
        )

    def has_data(self) -> bool:
        """Heuristic check for non-default process reward metrics."""
        metrics = (
            self.phi_star_mean,
            self.phi_star_final,
            self.phi_star_delta,
            self.r_shape_sum,
            self.disagreement_mean,
            self.entropy_mean,
        )
        if any(abs(val) > 1e-6 for val in metrics):
            return True
        conf_defaults = (self.conf_mean, self.conf_p10, self.conf_min)
        return any(abs(val - 0.5) > 1e-6 for val in conf_defaults)

    def quality_score(self, include_disagreement: bool = True, delta_cap: float = 1.0) -> float:
        """Compute quality score for sampling.

        Formula: conf * (1 + delta_pos) * disagree_factor

        Properties:
        - Bounded in [0, ~2] (conf in [0,1], delta clamped, disagreement factor capped)
        - Monotonic increasing in conf_mean
        - Monotonic increasing in positive phi_star_delta (clamped)
        - Monotonic decreasing in disagreement_mean
        - Negative delta doesn't penalize (avoids double-penalizing stuck episodes)

        Args:
            include_disagreement: If True, penalize high disagreement.
            delta_cap: Max positive delta contribution (clamped).

        Returns:
            Quality score in [0, ~2].
        """
        delta_pos = max(0.0, self.phi_star_delta)
        delta_pos = min(delta_pos, max(0.0, delta_cap))
        base = self.conf_mean * (1.0 + delta_pos)

        if include_disagreement:
            # Disagreement factor: 1 at disagreement=0, 0 at disagreement>=~3.33
            disagree_factor = 1.0 - 0.3 * self.disagreement_mean
            disagree_factor = max(0.0, min(1.0, disagree_factor))
            base *= disagree_factor

        return max(0.0, base)

    def is_reliable(self, conf_threshold: float = 0.3) -> bool:
        """Check if episode is reliable based on confidence."""
        return self.conf_p10 >= conf_threshold

    def is_stagnant(
        self,
        conf_min: float = 0.4,
        delta_max: float = 0.05,
        duration_min: int = 10,
        num_frames: int = 0,
    ) -> bool:
        """Detect stagnation: high confidence but no progress.

        Stagnation = well-observed episode that's stuck. These are valuable
        for curriculum: they indicate hard cases that need targeted intervention.

        Args:
            conf_min: Minimum confidence to consider "well-observed".
            delta_max: Maximum delta to consider "no progress".
            duration_min: Minimum frames to consider "long enough".
            num_frames: Actual number of frames (if known).

        Returns:
            True if episode shows stagnation pattern.
        """
        is_confident = self.conf_p10 >= conf_min
        is_stuck = abs(self.phi_star_delta) <= delta_max
        is_long = num_frames >= duration_min if num_frames > 0 else True
        return is_confident and is_stuck and is_long


@dataclass
class EmbodimentProfileSummary:
    """
    Embodiment metrics and artifact references for the datapack.

    Captures contact/affordance diagnostics, drift signals, and pointers
    to embodiment artifacts produced downstream of SceneTracks/MHN/SemFusion.
    """
    w_embodiment: float = 1.0
    embodiment_quality_score: float = 1.0
    trust_override_candidate: bool = False
    physically_impossible_contacts: int = 0
    contact_coverage_pct: float = 0.0
    semantic_confidence_mean: float = 0.0
    drift_score: float = 0.0

    # Artifact pointers
    embodiment_profile_npz: Optional[str] = None
    affordance_graph_npz: Optional[str] = None
    skill_segments_npz: Optional[str] = None
    cost_breakdown_json: Optional[str] = None
    value_attribution_json: Optional[str] = None
    drift_report_json: Optional[str] = None
    calibration_targets_json: Optional[str] = None
    summary_jsonl: Optional[str] = None

    # Compact summaries for quick filters
    cost_summary: Optional[Dict[str, Any]] = None
    value_summary: Optional[Dict[str, Any]] = None
    diagnostics: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EmbodimentProfileSummary":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ObjectiveProfile:
    """
    Objective and economic profile for the datapack.

    Captures:
    - Which objective_vector was declared for the run
    - Economic context (wage, energy price, market, customer segment)
    - EconProfileNet deltas applied (if any)
    - Effective EconParams used

    This is the bridge to programmable objectives and DL econ hyperparameters.
    """
    # Objective vector (what the system was optimizing for)
    objective_vector: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0, 0.0])
    # [w_mpl, w_error, w_energy, w_safety, w_novelty]

    # Economic context
    wage_human: float = 18.0
    energy_price_kWh: float = 0.12
    market_region: str = "US"
    task_family: str = "dishwashing"
    customer_segment: str = "balanced"
    baseline_mpl_human: float = 60.0
    baseline_error_human: float = 0.05

    # Environment context (from EconProfileContext)
    env_name: str = "drawer_vase"
    engine_type: str = "pybullet"
    task_type: str = "fragility"

    # EconProfileNet deltas (if applied)
    econ_profile_deltas: Optional[List[float]] = None
    # [Δbase_rate, Δdamage_cost, Δcare_cost, Δenergy_Wh_per_attempt, Δmax_steps_scale]

    # Effective EconParams used (after deltas applied)
    econ_params_effective: Optional[Dict[str, float]] = None
    # {"base_rate": ..., "damage_cost": ..., "care_cost": ..., "energy_Wh_per_attempt": ..., "max_steps": ...}

    # Reward weights (if EconObjectiveNet was used)
    reward_weights: Optional[List[float]] = None
    # [alpha_mpl, alpha_ep, alpha_error, alpha_energy, alpha_safety]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ObjectiveProfile":
        """Create from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def has_econ_profile_deltas(self) -> bool:
        """Check if EconProfileNet deltas were applied."""
        return self.econ_profile_deltas is not None

    def summary(self) -> str:
        """Get human-readable summary."""
        deltas_str = "none" if not self.econ_profile_deltas else f"{len(self.econ_profile_deltas)} deltas"
        return (
            f"ObjectiveProfile[{self.task_family}/{self.env_name}/{self.engine_type}] "
            f"obj={self.objective_vector[:4]} | "
            f"segment={self.customer_segment} | "
            f"deltas={deltas_str}"
        )


@dataclass
class DataPackMeta:
    """
    Complete datapack metadata for the two-bucket taxonomy.

    Unified with 2.0-energy schema for compatibility with Phase B/C scripts.

    Bucket types:
    - positive: Data that improved J/MPL/error/EP
    - negative: Data that worsened J/MPL/error/EP (with counterfactual plan)
    """
    # Schema version (unified with existing 2.0-energy format)
    schema_version: str = DATAPACK_SCHEMA_VERSION

    # Identification
    pack_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_name: str = "drawer_vase"
    env_type: str = "drawer_vase"  # "dishwashing", "drawer_vase", "bricklaying", etc.
    brick_id: Optional[str] = None  # For tracing specific data units
    bucket: Literal["positive", "negative"] = "positive"

    # Semantic tags for filtering
    semantic_tags: List[str] = field(default_factory=list)
    # e.g., ["fragile glassware", "multi-object", "low-light", "top drawer"]

    # Optional econ/semantic advisory tags
    econ_semantic_tags: Optional[List[str]] = None
    # Advisory-only semantic quality score in [0, 1]
    semantic_quality: Optional[float] = None

    # Energy driver tags (from energy_tags.py)
    energy_driver_tags: List[str] = field(default_factory=list)
    # e.g., ["energy_driver:long_reach", "energy_driver:fragility_cautious", ...]

    # Core profiles
    condition: ConditionProfile = field(default_factory=ConditionProfile)
    attribution: AttributionProfile = field(default_factory=AttributionProfile)

    # Energy profile (2.0-energy fields)
    energy: EnergyProfile = field(default_factory=EnergyProfile)

    # Agent profile (policy info)
    agent_profile: Dict[str, Any] = field(default_factory=dict)
    # e.g., {"policy": "scripted", "model_version": "v1", ...}

    # Skill trace (time-ordered skill usage)
    skill_trace: List[Dict[str, Any]] = field(default_factory=list)
    # Each element: {
    #   "t": int,  # timestep
    #   "skill_id": int,
    #   "params": Dict,  # skill parameters
    #   "duration": int,  # steps for this skill
    #   "local_metrics": Dict[str, float]  # local ΔMPL, Δerror, etc.
    # }

    # Episode metrics (raw from EpisodeInfoSummary)
    episode_metrics: Dict[str, Any] = field(default_factory=dict)

    # SIMA annotation (optional)
    sima_annotation: Optional[SimaAnnotation] = None

    # VLA plan annotation (optional)
    vla_plan: Optional[Dict[str, Any]] = None
    # e.g., {"instruction": "...", "skill_sequence": [...], "confidence": [...]}

    # Objective profile (optional) - captures DL econ hyperparameters
    objective_profile: Optional[ObjectiveProfile] = None

    # Process reward profile (optional) - PBRS metrics and confidence
    process_reward_profile: Optional[ProcessRewardProfile] = None

    # Embodiment profile (optional) - contacts/affordances + economics hooks
    embodiment_profile: Optional[EmbodimentProfileSummary] = None

    # Optional orchestration guidance
    guidance_profile: Optional["GuidanceProfile"] = None

    # Optional VLA action summary
    vla_action_summary: Optional[Dict[str, Any]] = None

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

    # Portable artifacts (optional) - enable curated eval without raw rehydration
    scene_tracks_v1: Optional[Dict[str, Any]] = None
    rgb_features_v1: Optional[Dict[str, Any]] = None
    slice_labels_v1: Optional[Dict[str, Any]] = None

    # Representation token outputs (optional) - enable token-only eval
    repr_tokens: Optional[Dict[str, Dict[str, Any]]] = None
    # Keys: repr name (e.g., "vision_rgb", "geometry_bev")
    # Value: {"version": str, "dim": int, "features": List[float], "metadata": Dict}

    # Vision backbone embedding (optional) - for novelty/regime analysis
    episode_embedding: Optional[List[float]] = None
    # e.g., [0.1, -0.2, 0.3, ...] - pooled embedding from VisionBackbone.encode_sequence()

    # Epiplexity / prequential-MDL metrics (optional, advisory)
    epiplexity: Optional[Dict[str, Any]] = None
    epiplexity_summary: Optional[Dict[str, Any]] = None

    # Homeostatic control signals (optional) - for closed-loop controller
    signal_bundle: Optional[Dict[str, Any]] = None
    # Contains serialized SignalBundle with epiplexity, stability, alignment metrics

    # Graph small-world metrics (optional) - computed from geometry_bev
    graph_summary_v1: Optional[Dict[str, Any]] = None
    # Contains serialized GraphSummaryV1 with sigma, nav_success_rate, etc.

    # Econ tensor (optional) - canonical coordinate chart for econ metrics
    econ_tensor_v1: Optional[Dict[str, Any]] = None
    # Contains serialized EconTensorV1 with basis_sha, x, and provenance

    # Regal annotations (optional) - P1 typed regal metadata for training disposition
    regal_annotations: Optional[Dict[str, Any]] = None
    # Contains serialized RegalAnnotationsV1 with violation_tags, training_disposition, etc.

    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        d = {
            'schema_version': self.schema_version,
            'pack_id': self.pack_id,
            'task_name': self.task_name,
            'env_type': self.env_type,
            'brick_id': self.brick_id,
            'bucket': self.bucket,
            'semantic_tags': self.semantic_tags,
            'econ_semantic_tags': self.econ_semantic_tags,
            'semantic_quality': self.semantic_quality,
            'energy_driver_tags': self.energy_driver_tags,
            'condition': self.condition.to_dict(),
            'attribution': self.attribution.to_dict(),
            'energy': self.energy.to_dict(),
            'agent_profile': to_json_safe(self.agent_profile),
            'skill_trace': to_json_safe(self.skill_trace),
            'episode_metrics': to_json_safe(self.episode_metrics),
            'sima_annotation': self.sima_annotation.to_dict() if self.sima_annotation else None,
            'vla_plan': to_json_safe(self.vla_plan),
            'objective_profile': self.objective_profile.to_dict() if self.objective_profile else None,
            'process_reward_profile': self.process_reward_profile.to_dict() if self.process_reward_profile else None,
            'embodiment_profile': self.embodiment_profile.to_dict() if self.embodiment_profile else None,
            'counterfactual_plan': to_json_safe(self.counterfactual_plan),
            'counterfactual_source': self.counterfactual_source,
            'created_at': self.created_at,
            'episode_id': self.episode_id,
            'episode_index': self.episode_index,
            'raw_data_path': self.raw_data_path,
            'scene_tracks_v1': to_json_safe(self.scene_tracks_v1),
            'rgb_features_v1': to_json_safe(self.rgb_features_v1),
            'slice_labels_v1': to_json_safe(self.slice_labels_v1),
            'repr_tokens': to_json_safe(self.repr_tokens),
            'guidance_profile': self.guidance_profile.to_dict() if self.guidance_profile else None,
            'vla_action_summary': to_json_safe(self.vla_action_summary),
            'episode_embedding': to_json_safe(self.episode_embedding),
            'epiplexity': to_json_safe(self.epiplexity),
            'epiplexity_summary': to_json_safe(self.epiplexity_summary),
            'signal_bundle': to_json_safe(self.signal_bundle),
            'graph_summary_v1': to_json_safe(self.graph_summary_v1),
            'econ_tensor_v1': to_json_safe(self.econ_tensor_v1),
            'regal_annotations': to_json_safe(self.regal_annotations),
        }
        return to_json_safe(d)

    @classmethod
    def from_dict(cls, d):
        """Create from dictionary."""
        condition = ConditionProfile.from_dict(d.get('condition', {}))
        attribution = AttributionProfile.from_dict(d.get('attribution', {}))

        # Handle energy profile (new in unified schema)
        energy = EnergyProfile()
        if 'energy' in d:
            energy = EnergyProfile.from_dict(d['energy'])

        sima_annotation = None
        if d.get('sima_annotation'):
            sima_annotation = SimaAnnotation.from_dict(d['sima_annotation'])

        objective_profile = None
        if d.get('objective_profile'):
            objective_profile = ObjectiveProfile.from_dict(d['objective_profile'])

        process_reward_profile = None
        if d.get('process_reward_profile'):
            process_reward_profile = ProcessRewardProfile.from_dict(d['process_reward_profile'])

        embodiment_profile = None
        if d.get('embodiment_profile'):
            embodiment_profile = EmbodimentProfileSummary.from_dict(d['embodiment_profile'])

        return cls(
            schema_version=d.get('schema_version', DATAPACK_SCHEMA_VERSION),
            pack_id=d.get('pack_id', str(uuid.uuid4())),
            task_name=d.get('task_name', 'drawer_vase'),
            env_type=d.get('env_type', d.get('task_name', 'drawer_vase')),
            brick_id=d.get('brick_id'),
            bucket=d.get('bucket', 'positive'),
            semantic_tags=d.get('semantic_tags', []),
            econ_semantic_tags=d.get('econ_semantic_tags'),
            semantic_quality=d.get('semantic_quality'),
            energy_driver_tags=d.get('energy_driver_tags', []),
            condition=condition,
            attribution=attribution,
            energy=energy,
            agent_profile=d.get('agent_profile', {}),
            skill_trace=d.get('skill_trace', []),
            episode_metrics=d.get('episode_metrics', {}),
            sima_annotation=sima_annotation,
            vla_plan=d.get('vla_plan'),
            objective_profile=objective_profile,
            process_reward_profile=process_reward_profile,
            embodiment_profile=embodiment_profile,
            counterfactual_plan=d.get('counterfactual_plan'),
            counterfactual_source=d.get('counterfactual_source'),
            created_at=d.get('created_at', datetime.now().isoformat()),
            episode_id=d.get('episode_id'),
            episode_index=d.get('episode_index'),
            raw_data_path=d.get('raw_data_path'),
            scene_tracks_v1=d.get('scene_tracks_v1'),
            rgb_features_v1=d.get('rgb_features_v1'),
            slice_labels_v1=d.get('slice_labels_v1'),
            repr_tokens=d.get('repr_tokens'),
            guidance_profile=GuidanceProfile.from_dict(d['guidance_profile']) if d.get('guidance_profile') else None,
            vla_action_summary=d.get('vla_action_summary'),
            episode_embedding=d.get('episode_embedding'),
            epiplexity=d.get('epiplexity'),
            epiplexity_summary=d.get('epiplexity_summary'),
            signal_bundle=d.get('signal_bundle'),
            graph_summary_v1=d.get('graph_summary_v1'),
            econ_tensor_v1=d.get('econ_tensor_v1'),
            regal_annotations=d.get('regal_annotations'),
        )

    @classmethod
    def from_legacy_energy_dict(cls, legacy_dict: Dict[str, Any]):
        """
        Create DataPackMeta from legacy 2.0-energy dict format.

        This is the format produced by build_datapack_from_episode() in datapacks.py.
        """
        # Extract episode metrics
        episode_metrics = legacy_dict.get('episode_metrics', {})

        # Extract condition profile from legacy format
        legacy_condition = legacy_dict.get('condition_profile', {})
        condition = ConditionProfile(
            task_name=legacy_dict.get('env_type', 'drawer_vase'),
            engine_type=legacy_condition.get('engine_type', 'pybullet'),
            world_id=legacy_condition.get('world_id', 'pyb_drawer_v1'),
            econ_preset=legacy_dict.get('econ_params', {}).get('preset', 'drawer_vase'),
            price_per_unit=legacy_dict.get('econ_params', {}).get('price_per_unit', 5.0),
            energy_price_kWh=legacy_dict.get('econ_params', {}).get('energy_price_kWh', 0.12),
            tags=legacy_condition,
        )

        # Extract attribution from legacy format
        legacy_attribution = legacy_dict.get('attribution', {})
        attribution = AttributionProfile(
            delta_mpl=legacy_attribution.get('delta_mpl', 0.0),
            delta_error=legacy_attribution.get('delta_error', 0.0),
            delta_ep=legacy_attribution.get('delta_ep', 0.0),
            trust_score=legacy_attribution.get('trust', 0.0),
            w_econ=legacy_attribution.get('econ_weight', 0.0) or 0.0,
        )

        # Extract energy profile from legacy format
        legacy_energy = legacy_dict.get('energy', {})
        energy = EnergyProfile(
            total_Wh=legacy_energy.get('total_Wh', 0.0),
            Wh_per_unit=legacy_energy.get('Wh_per_unit', 0.0),
            Wh_per_hour=legacy_energy.get('Wh_per_hour', 0.0),
            energy_per_limb=legacy_energy.get('energy_per_limb', {}),
            energy_per_skill=legacy_energy.get('energy_per_skill', {}),
            energy_per_joint=legacy_energy.get('energy_per_joint', {}),
            energy_per_effector=legacy_energy.get('energy_per_effector', {}),
            coordination_metrics=legacy_energy.get('coordination_metrics', {}),
            limb_energy_Wh=legacy_energy.get('limb_energy_Wh', {}),
            skill_energy_Wh=legacy_energy.get('skill_energy_Wh', {}),
        )

        # Determine bucket based on delta_J or defaults
        delta_j = legacy_attribution.get('delta_J', legacy_attribution.get('delta_mpl', 0.0))
        bucket = "positive" if delta_j >= 0 else "negative"

        return cls(
            schema_version=legacy_dict.get('schema_version', DATAPACK_SCHEMA_VERSION),
            pack_id=legacy_dict.get('brick_id', str(uuid.uuid4())),
            task_name=legacy_dict.get('env_type', 'drawer_vase'),
            env_type=legacy_dict.get('env_type', 'drawer_vase'),
            brick_id=legacy_dict.get('brick_id'),
            bucket=bucket,
            semantic_tags=legacy_dict.get('tags', []),
            econ_semantic_tags=legacy_dict.get('econ_semantic_tags'),
            semantic_quality=legacy_dict.get('semantic_quality'),
            energy_driver_tags=legacy_dict.get('semantic_energy_drivers', []),
            condition=condition,
            attribution=attribution,
            energy=energy,
            agent_profile=legacy_dict.get('agent_profile', {}),
            skill_trace=[],
            episode_metrics=episode_metrics,
            sima_annotation=None,
            vla_plan=None,
            counterfactual_plan=None,
            counterfactual_source=None,
            created_at=datetime.now().isoformat(),
            episode_id=legacy_dict.get('brick_id'),
            episode_index=None,
            raw_data_path=None,
        )

    def to_legacy_energy_dict(self) -> Dict[str, Any]:
        """
        Convert DataPackMeta back to legacy 2.0-energy dict format.

        For backwards compatibility with existing scripts.
        """
        return {
            'schema_version': self.schema_version,
            'env_type': self.env_type,
            'brick_id': self.brick_id or self.pack_id,
            'episode_metrics': self.episode_metrics,
            'econ_params': {
                'price_per_unit': self.condition.price_per_unit,
                'damage_cost': self.condition.vase_break_cost,
                'energy_Wh_per_attempt': self.condition.energy_price_kWh,
                'time_step_s': 0.01,
                'max_steps': 1000,
                'preset': self.condition.econ_preset,
            },
            'condition_profile': self.condition.tags,
            'agent_profile': self.agent_profile,
            'tags': self.semantic_tags,
            'attribution': {
                'delta_mpl': self.attribution.delta_mpl,
                'delta_error': self.attribution.delta_error,
                'delta_ep': self.attribution.delta_ep,
                'novelty': None,
                'trust': self.attribution.trust_score,
                'econ_weight': self.attribution.w_econ,
            },
            'energy': self.energy.to_dict(),
            'semantic_energy_drivers': self.energy_driver_tags,
        }

    def to_json(self):
        """Convert to JSON string."""
        def _convert_numpy(obj):
            """Recursively convert numpy types to Python natives."""
            import numpy as np
            if isinstance(obj, np.generic):
                return obj.item()
            if isinstance(obj, dict):
                return {k: _convert_numpy(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_convert_numpy(v) for v in obj]
            if isinstance(obj, tuple):
                return tuple(_convert_numpy(v) for v in obj)
            return obj

        return json.dumps(_convert_numpy(self.to_dict()))

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
