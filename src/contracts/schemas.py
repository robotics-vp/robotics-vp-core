"""Strict Pydantic schemas for closed-loop cybernetic system.

All schemas reject unknown keys by default and use strict validation.
These are the single source of truth for data contracts.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, ConfigDict


# =============================================================================
# Schema Version Constants
# =============================================================================

SCHEMA_VERSION_PLAN = "v1"
SCHEMA_VERSION_EPISODE = "v1"
SCHEMA_VERSION_LEDGER = "v1"
SCHEMA_VERSION_PROBE = "v1"


# =============================================================================
# SemanticUpdatePlanV1 - Plan for task/datapack selection changes
# =============================================================================

class PlanOpType(str, Enum):
    """Types of plan operations."""
    SET_WEIGHT = "set_weight"
    DISABLE = "disable"
    ENABLE = "enable"


class TaskGraphOp(BaseModel):
    """Single operation on the task graph."""
    model_config = ConfigDict(extra="forbid")

    op: PlanOpType
    task_family: str
    weight: Optional[float] = None  # Required for SET_WEIGHT


class DatapackSelectionConfig(BaseModel):
    """Configuration for datapack/slice selection."""
    model_config = ConfigDict(extra="forbid")

    allowlist: Optional[List[str]] = None  # datapack_ids to include
    denylist: Optional[List[str]] = None  # datapack_ids to exclude
    quotas: Optional[Dict[str, int]] = None  # per-slice quotas
    curated_params: Optional[Dict[str, Any]] = None  # curated slice params


class SemanticUpdatePlanV1(BaseModel):
    """Semantic update plan for hot-reload actuation.

    This is the contract between the controller and the sampler.
    """
    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION_PLAN
    plan_id: str
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    source_commit: Optional[str] = None

    # Task graph changes
    task_graph_changes: List[TaskGraphOp] = Field(default_factory=list)

    # Datapack selection
    datapack_selection: Optional[DatapackSelectionConfig] = None
    
    # Rationale
    rationales: List[str] = Field(default_factory=list)
    
    # Provenance
    plan_sha: Optional[str] = None

    # Reward weights (pass-through only, no behavior changes)
    reward_weights: Optional[Dict[str, float]] = None

    notes: Optional[str] = None

    def sha256(self) -> str:
        """Compute SHA-256 of the plan for provenance."""
        from src.utils.config_digest import sha256_json
        return sha256_json(self.model_dump(mode="json"))


# =============================================================================
# EpisodeInfoSummaryV1 - Per-episode summary for audit
# =============================================================================

class DeterminismConfig(BaseModel):
    """Determinism configuration for reproducibility."""
    model_config = ConfigDict(extra="forbid")

    seed: int
    backend_id: Optional[str] = None
    strict: bool = False


class EpisodeInfoSummaryV1(BaseModel):
    """Per-episode summary for deterministic audit eval.

    Contains only metrics that exist; null otherwise.
    """
    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION_EPISODE
    episode_id: str
    episode_index: int

    # Task info
    task_name: str
    task_family: Optional[str] = None

    # Outcome
    success: bool
    termination_reason: Optional[str] = None

    # Scalars (only if they exist)
    error: Optional[float] = None
    total_return: Optional[float] = None
    energy_Wh: Optional[float] = None
    mpl_proxy: Optional[float] = None

    # Determinism
    determinism: DeterminismConfig

    # Datapack context
    datapack_id: Optional[str] = None
    slice_id: Optional[str] = None

    # Policy reference
    policy_id: Optional[str] = None
    checkpoint_ref: Optional[str] = None

    # Timestamps
    ts_start: Optional[str] = None
    ts_end: Optional[str] = None


class AuditAggregateV1(BaseModel):
    """Aggregated metrics from audit eval suite."""
    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION_EPISODE
    audit_suite_id: str
    seed: int
    num_episodes: int

    # Aggregate metrics
    success_rate: float
    mean_error: Optional[float] = None
    mean_return: Optional[float] = None
    mean_energy_Wh: Optional[float] = None
    mean_mpl_proxy: Optional[float] = None

    # Per-task breakdown
    per_task: Optional[Dict[str, Dict[str, float]]] = None

    # Hashes
    episodes_sha: str
    config_sha: str
    
    # Regal context provenance (P2: what regal context was this audit run under)
    regal_context_sha: Optional[str] = None


# =============================================================================
# ProbeEpiReportV1 - Delta-epiplexity-per-flop discrimination report
# =============================================================================

class ProbeConfigV1(BaseModel):
    """Configuration for probe epiplexity harness."""
    model_config = ConfigDict(extra="forbid")

    probe_variant: str  # "linear", "mlp", "transformer"
    probe_steps: int
    batch_size: int
    seeds: List[int]
    input_dim: int
    hidden_dim: Optional[int] = None  # For MLP
    
    def sha256(self) -> str:
        from src.utils.config_digest import sha256_json
        return sha256_json(self.model_dump(mode="json"))


class ProbeEpiReportV1(BaseModel):
    """Report from probe epiplexity harness.
    
    Contains delta-epiplexity-per-flop with stability and transfer gates.
    """
    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION_PROBE
    report_id: str
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())

    # Probe config
    probe_config: ProbeConfigV1

    # Scores
    baseline_score: float
    after_score: float
    delta: float
    
    # FLOPs
    flops_estimate: float
    delta_epi_per_flop: float

    # Stability gate
    per_seed_deltas: List[float]
    sign_consistency: float  # Fraction with same sign as mean
    stability_pass: bool

    # Transfer gate (OOD)
    ood_baseline_score: Optional[float] = None
    ood_after_score: Optional[float] = None
    ood_delta: Optional[float] = None
    transfer_pass: bool = False

    # Coverage
    num_samples_id: int = 0
    num_samples_ood: int = 0

    # Hashes
    probe_config_sha: str = ""
    report_sha: str = ""

    def compute_hashes(self) -> None:
        """Compute and set hash fields."""
        from src.utils.config_digest import sha256_json
        self.probe_config_sha = self.probe_config.sha256()
        # Exclude report_sha from self-hash
        data = self.model_dump(mode="json")
        data["report_sha"] = ""
        self.report_sha = sha256_json(data)


# =============================================================================
# Plan Policy & Gain Schedule
# =============================================================================

class PlanGainScheduleV1(BaseModel):
    """Gain schedule determining weight changes based on gate outcomes."""
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "v1"
    
    # Multipliers
    conservative_multiplier: float = 1.05
    full_multiplier: float = 1.2
    
    # Safety clamps
    max_abs_weight_change: Optional[float] = 0.1
    min_weight_clamp: Optional[float] = 0.05
    max_weight_clamp: Optional[float] = 5.0
    
    # Optional logic
    per_task_family_overrides: Optional[Dict[str, float]] = None
    cooldown_steps: Optional[int] = None

    def sha256(self) -> str:
        from src.utils.config_digest import sha256_json
        return sha256_json(self.model_dump(mode="json"))


class PlanPolicyConfigV1(BaseModel):
    """Configuration for homeostatic plan generation policy."""
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "v1"
    
    # Gain schedule
    gain_schedule: PlanGainScheduleV1
    
    # Gate thresholds (from PlanFromSignalsConfig)
    epiplexity_low_threshold: float = 0.2
    epiplexity_high_threshold: float = 0.8
    stability_threshold: float = 0.7
    coverage_threshold: float = 0.3
    
    delta_epi_per_flop_threshold: float = 0.0
    stability_gate_threshold: float = 0.7
    transfer_gate_threshold: float = 0.0
    min_raw_delta: float = 0.01
    max_transfer_failures: int = 3
    
    hysteresis_margin: float = 0.05
    default_weights: Dict[str, float]
    
    # Cooldowns
    min_apply_interval_steps: int = 0
    max_changes_per_window: int = 100
    
    # Graph gates (optional)
    graph_gates: Optional["GraphGatesV1"] = None

    # Regal gates (optional, Stage-6 meta-regal nodes)
    regal_gates: Optional["RegalGatesV1"] = None

    def sha256(self) -> str:
        from src.utils.config_digest import sha256_json
        return sha256_json(self.model_dump(mode="json"))


# =============================================================================
# Graph Small-World Metrics
# =============================================================================

class GraphSpecV1(BaseModel):
    """Configuration for small-world graph construction."""
    model_config = ConfigDict(extra="forbid")

    spec_id: str = "default_graph_spec_v1"
    local_connectivity: Literal[4, 8] = 4  # 4-neighbor or 8-neighbor lattice
    knn_k: int = 6  # k for kNN shortcut edges
    min_lattice_hops_for_shortcut: int = 4  # Min grid distance for shortcut
    n_sources: int = 32  # Sampled BFS sources for path length estimation
    n_queries: int = 64  # Navigability queries
    max_hops: int = 50  # Max hops for greedy navigation
    seed: int = 42  # Random seed for sampling

    # Score-based shortcut selection (principled and adaptive)
    # Score formula: score(i,j) = cos_sim(ei,ej) * log(1 + lattice_dist(i,j))
    shortcut_score_mode: Literal["cos_sim_logdist"] = "cos_sim_logdist"

    # Selection mode:
    #   "threshold": keep edges with score >= shortcut_score_threshold
    #   "top_m_per_node": keep top M scoring candidates per node
    #   "target_nav_gain": add shortcuts until nav_gain >= target (Goldilocks sparse)
    shortcut_select_mode: Literal["threshold", "top_m_per_node", "target_nav_gain"] = "top_m_per_node"

    # For threshold mode: minimum score to keep a shortcut
    shortcut_score_threshold: Optional[float] = None

    # For top_m_per_node mode: max shortcuts per node (replaces shortcut_budget_per_node as primary)
    shortcut_top_m_per_node: int = 2

    # For target_nav_gain mode: stop adding shortcuts when nav_gain >= target
    target_nav_gain: Optional[float] = None  # e.g., 0.15 means stop at 15% nav improvement
    target_nav_gain_step: int = 1  # Shortcuts to add per iteration when searching for target

    # Quality filters
    mutual_knn_only: bool = True  # Keep shortcut only if mutual (i in knn(j) AND j in knn(i))
    shortcut_budget_per_node: Optional[int] = 2  # Legacy alias for shortcut_top_m_per_node

    # Safety ceiling only - NOT the primary limiter, just a global cap
    # max_shortcut_fraction * total_edges is the absolute maximum shortcuts allowed
    max_shortcut_fraction: float = 0.25

    # Baseline type for sigma calculation
    # "ER_expected_degree": Erdos-Renyi with same expected degree (fast, approximate)
    # "configuration_model": Degree-matched stub-matching shuffle (more accurate sigma)
    baseline_type: Literal["ER_expected_degree", "configuration_model"] = "ER_expected_degree"

    def sha256(self) -> str:
        from src.utils.config_digest import sha256_json
        return sha256_json(self.model_dump(mode="json"))


class GraphSummaryV1(BaseModel):
    """Per-episode graph metrics summary."""
    model_config = ConfigDict(extra="forbid")

    graph_spec_id: str
    graph_spec_sha: str
    summary_sha: str
    node_mode: Literal["grid", "tokens", "pooled"]

    # Graph structure
    node_count: int
    mean_degree: float
    local_edge_count: int
    shortcut_edge_count: int

    # Small-world metrics
    clustering_coefficient: float  # C
    avg_path_length: float  # L (sampled)
    c_rand: float  # Random baseline clustering
    l_rand: float  # Random baseline path length
    sigma: float  # Small-world index: (C/C_rand)/(L/L_rand)
    baseline_type: str = "ER_expected_degree"  # Baseline graph type for σ

    # Shortcut selection mode tracking
    shortcut_score_mode: str = "cos_sim_logdist"  # Score formula used
    shortcut_select_mode: str = "top_m_per_node"  # Selection strategy used
    shortcut_score_threshold_used: Optional[float] = None  # For threshold mode

    # Shortcut stats
    shortcut_fraction: float
    shortcut_lattice_hop_mean: float
    shortcut_lattice_hop_p50: float
    shortcut_lattice_hop_p90: float

    # Shortcut quality scores (cos_sim * log(1+lattice_dist))
    shortcut_score_mean: float = 0.0
    shortcut_score_min: float = 0.0
    shortcut_score_max: float = 0.0
    shortcut_score_p50: float = 0.0  # Median score
    shortcut_score_p90: float = 0.0  # 90th percentile score

    # Bounded navigability (full graph with shortcuts)
    nav_success_rate: float
    nav_mean_hops: float
    nav_stretch: float  # greedy_hops / shortest_path
    nav_visited_nodes_mean: float = 0.0  # Average nodes visited per query (bounded compute)

    # Lattice-only navigability baseline
    nav_success_lattice: float = 0.0  # Success rate on lattice-only
    nav_gain: float = 0.0  # nav_success_rate - nav_success_lattice (wormhole benefit)

    # Metadata
    compute_time_ms: float


class GraphGatesV1(BaseModel):
    """Thresholds for graph-based safety gates."""
    model_config = ConfigDict(extra="forbid")

    sigma_min: float = 0.8  # Force action if sigma drops below
    nav_success_min: float = 0.5  # Force action if nav success drops below
    patience: int = 3  # Allow N consecutive violations before action
    penalty_mode: Literal["clamp", "noop"] = "clamp"


# =============================================================================
# TrajectoryAuditV1 - Per-episode action/state/reward decomposition
# =============================================================================

class TrajectoryAuditV1(BaseModel):
    """Per-episode trajectory audit for meta-regal grounding.

    Provides the geometric/spatiotemporal substrate for:
    - Spec violations via event/state constraints
    - Coherence via perturbation sensitivity
    - Reward hacking detection via reward-work correlation
    """
    model_config = ConfigDict(extra="forbid")

    episode_id: str
    num_steps: int

    # Action/state summaries (full logs too heavy, use stats)
    action_mean: Optional[List[float]] = None  # Mean action per dimension
    action_std: Optional[List[float]] = None   # Std action per dimension
    state_bounds: Optional[Dict[str, List[float]]] = None  # {state_name: [min, max]}

    # Reward decomposition
    total_return: float = 0.0
    reward_components: Optional[Dict[str, float]] = None  # {component_name: total}

    # Events (for spec violation detection)
    events: List[str] = Field(default_factory=list)  # e.g., ["collision", "drop", "timeout"]
    event_counts: Optional[Dict[str, int]] = None

    # Physics anomalies (for coherence/exploit detection)
    penetration_max: Optional[float] = None
    velocity_spike_count: int = 0
    contact_anomaly_count: int = 0

    # References to geometry stack
    scene_tracks_sha: Optional[str] = None
    bev_summary_sha: Optional[str] = None

    def sha256(self) -> str:
        from src.utils.config_digest import sha256_json
        return sha256_json(self.model_dump(mode="json"))


# =============================================================================
# Regal Phases and Context (Stage-6 Temporal Formalism)
# =============================================================================


class RegalPhaseV1(str, Enum):
    """Temporal phases for regal evaluation.

    Formalizes when regals run in the cybernetic loop:
    - PRE_PLAN: Before plan generation (rare, for pre-conditions)
    - POST_PLAN_PRE_APPLY: After plan generated, before application (gating)
    - POST_APPLY_PRE_TRAIN: After plan applied, before training window
    - DURING_TRAIN: Streaming checks during training (future)
    - POST_TRAIN_PRE_AUDIT: After training, before audit eval
    - POST_AUDIT: After audit, for final ledger/manifest recording
    """
    PRE_PLAN = "pre_plan"
    POST_PLAN_PRE_APPLY = "post_plan_pre_apply"
    POST_APPLY_PRE_TRAIN = "post_apply_pre_train"
    DURING_TRAIN = "during_train"
    POST_TRAIN_PRE_AUDIT = "post_train_pre_audit"
    POST_AUDIT = "post_audit"


class RegalContextV1(BaseModel):
    """Strict typed context for regal evaluation substrate.

    Replaces ad-hoc Dict[str, Any] context with typed, hashed, stable schema.
    Extra keys are rejected (extra='forbid') to enforce schema discipline.
    """
    model_config = ConfigDict(extra="forbid")

    # Core identifiers
    run_id: str
    step: Optional[int] = None

    # Policy provenance
    policy_before: Optional[str] = None
    policy_after: Optional[str] = None

    # Plan provenance
    plan_sha: Optional[str] = None

    # Weight provenance
    baseline_weights_sha: Optional[str] = None
    final_weights_sha: Optional[str] = None

    # Audit provenance
    audit_suite_id: Optional[str] = None
    audit_suite_sha: Optional[str] = None
    audit_seed: Optional[int] = None

    # Exposure/substrate provenance
    exposure_manifest_sha: Optional[str] = None
    probe_report_sha: Optional[str] = None
    graph_summary_sha: Optional[str] = None
    trajectory_audit_sha: Optional[str] = None

    # Econ provenance
    econ_basis_sha: Optional[str] = None
    econ_tensor_sha: Optional[str] = None

    # Escape hatch for future extensions (still typed)
    notes: Optional[Dict[str, Any]] = None

    def sha256(self) -> str:
        """Compute deterministic SHA-256 of context."""
        from src.utils.config_digest import sha256_json
        return sha256_json(self.model_dump(mode="json"))


# =============================================================================
# Meta-Regal Nodes (Stage-6 Deterministic Audit Gates)
# =============================================================================

class RegalGatesV1(BaseModel):
    """Opt-in configuration for meta-regal gates (Stage-6 deterministic nodes).

    Regal nodes are deterministic audit evaluators that run at each planning cycle.
    They provide semantic checks beyond numeric thresholds (spec conformance,
    world coherence, reward integrity).

    Ordering: Guardians (spec/coherence/integrity) → Econ/Datapacks (allocation)
    """
    model_config = ConfigDict(extra="forbid")

    # Which regal nodes to enable (e.g., ["spec_guardian", "world_coherence", "reward_integrity"])
    enabled_regal_ids: List[str] = Field(default_factory=list)

    # Gate thresholds (min acceptable scores, 0.0-1.0)
    spec_consistency_min: float = 0.5  # Min spec adherence score
    coherence_min: float = 0.5         # Min world coherence score
    hack_prob_max: float = 0.3         # Max acceptable reward hacking probability

    # Patience: consecutive failures before triggering penalty
    patience: int = 3

    # Penalty mode when gates fail after patience exhausted
    penalty_mode: Literal["clamp", "noop", "warn"] = "warn"

    # Seed for deterministic evaluation
    determinism_seed: int = 42

    # Per-task-family threshold overrides
    per_task_family_overrides: Optional[Dict[str, Dict[str, float]]] = None

    # Trajectory audit anomaly thresholds (configurable, included in SHA for provenance)
    # These trigger WorldCoherenceRegal failure
    velocity_spike_threshold: int = 5      # >= this count triggers failure
    penetration_max_threshold: float = 0.01  # > this distance triggers failure
    contact_anomaly_threshold: int = 3     # >= this count triggers failure
    # These trigger RewardIntegrityRegal flags
    extreme_reward_component_threshold: float = 10.0  # abs > this triggers flag
    high_total_return_threshold: float = 10.0  # > this triggers flag

    def sha256(self) -> str:
        from src.utils.config_digest import sha256_json
        return sha256_json(self.model_dump(mode="json"))


class RegalReportV1(BaseModel):
    """Deterministic report from a single regal node evaluation.

    Each regal node produces a hashable report that can be audited
    and reproduced given the same inputs and seed.
    """
    model_config = ConfigDict(extra="forbid")

    regal_id: str  # e.g., "spec_guardian", "world_coherence", "reward_integrity"
    phase: RegalPhaseV1 = RegalPhaseV1.POST_PLAN_PRE_APPLY  # Temporal phase
    regal_version: str = "v1"
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())

    # Evaluation inputs (for reproducibility)
    inputs_sha: str  # SHA of all inputs fed to this regal
    determinism_seed: int

    # Verdict
    passed: bool
    confidence: float = 1.0  # 0.0-1.0, how confident the regal is
    rationale: str = ""  # Human-readable explanation

    # Structured output (spec_guardian)
    spec_consistency_score: float = 1.0  # 0.0-1.0
    spec_violations: List[str] = Field(default_factory=list)

    # Structured output (world_coherence)
    coherence_score: float = 1.0  # 0.0-1.0
    coherence_tags: List[str] = Field(default_factory=list)  # e.g., ["sim_exploit", "physics_violation"]

    # Structured output (reward_integrity)
    hack_probability: float = 0.0  # 0.0-1.0, estimated probability of reward hacking
    integrity_flags: List[str] = Field(default_factory=list)  # e.g., ["oscillation", "reward_spike"]

    # Forced action (if this regal triggered a penalty)
    forced_action: Optional[Literal["noop", "clamp"]] = None

    # Detailed findings (optional, regal-specific)
    findings: Optional[Dict[str, Any]] = None

    # Report hash (computed after creation)
    report_sha: str = ""

    def compute_sha(self) -> None:
        """Compute and set the report_sha field."""
        from src.utils.config_digest import sha256_json
        data = self.model_dump(mode="json")
        data["report_sha"] = ""
        self.report_sha = sha256_json(data)


class LedgerRegalV1(BaseModel):
    """Regal evaluation results for ledger entry.

    Aggregates all regal reports from a planning cycle for
    inclusion in the value ledger.
    """
    model_config = ConfigDict(extra="forbid")

    regal_config_sha: str  # SHA of RegalGatesV1 used
    reports: List[RegalReportV1] = Field(default_factory=list)
    all_passed: bool = True
    combined_inputs_sha: str = ""  # SHA of all inputs across all regals

    # Summary stats for quick reference
    spec_consistency_min_observed: float = 1.0
    coherence_min_observed: float = 1.0
    hack_probability_max_observed: float = 0.0

    # Whether regal forced NOOP or clamp
    forced_noop: bool = False
    forced_clamp: bool = False


class LedgerRegalSummaryV1(BaseModel):
    """Compact regal summary for value ledger records.

    Provides a typed, space-efficient summary of regal outcomes
    without embedding full RegalReportV1 objects.
    """
    model_config = ConfigDict(extra="forbid")

    passed: bool
    phase: RegalPhaseV1
    confidence: float = 1.0

    # Key scalar metrics (regal-specific)
    spec_consistency_score: Optional[float] = None
    coherence_score: Optional[float] = None
    hack_probability: Optional[float] = None

    # Tags for quick filtering
    tags: List[str] = Field(default_factory=list)

    # Link to full report
    report_sha: str


class RegalAnnotationsV1(BaseModel):
    """Typed regal annotations for datapack metadata.

    Attached to datapacks/episodes to record regal outcomes
    and inform downstream curriculum/training decisions.
    """
    model_config = ConfigDict(extra="forbid")

    # Violation tags from all regals
    violation_tags: List[str] = Field(default_factory=list)

    # Training disposition
    training_disposition: Literal["allow", "eval_only", "quarantine"] = "allow"

    # Correction recipe (advisory, for similar patterns)
    correction_recipe: Optional[Dict[str, Any]] = None

    # Phase-keyed report SHAs
    phase_report_shas: Dict[str, str] = Field(default_factory=dict)

    # Key findings summary
    physics_anomaly_detected: bool = False
    hack_pattern_detected: bool = False
    constraint_violations: List[str] = Field(default_factory=list)

    # Provenance
    regal_config_sha: str = ""


class KnobDeltaV1(BaseModel):
    """Advisory knob delta emitted by regals.

    Regals cannot modify PlanGainScheduleV1 directly. Instead they emit
    advisory deltas that the knob policy layer applies subject to constraints.
    """
    model_config = ConfigDict(extra="forbid")

    source_regal: str  # e.g., "econ_data", "world_coherence"
    phase: RegalPhaseV1

    # Advisory preference (not a command)
    prefer_conservative_multiplier: bool = False  # Use conservative instead of full
    multiplier_reduction_factor: Optional[float] = None  # e.g., 0.8 for 20% reduction (advisory)

    # Task-specific advisories
    task_family_advisories: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    # e.g., {"manipulation": {"weight_cap": 0.7, "cooldown_steps": 5}}

    rationale: str = ""


# =============================================================================
# Knob Calibration (D4 - Learned hyperparameters with heuristic fallback)
# =============================================================================

class RegimeFeaturesV1(BaseModel):
    """Input features for knob calibration model.

    Captures the current "regime" state for knob adaptation.
    """
    model_config = ConfigDict(extra="forbid")

    # Audit performance
    audit_delta_success: Optional[float] = None
    audit_delta_error: Optional[float] = None
    audit_success_rate: Optional[float] = None

    # Exposure
    exposure_count: int = 0
    datapack_count: int = 0

    # Probe discriminator
    probe_delta_epi_per_flop: Optional[float] = None
    probe_stability_pass: Optional[bool] = None
    probe_transfer_pass: Optional[bool] = None

    # Graph metrics
    graph_sigma: Optional[float] = None
    graph_nav_success: Optional[float] = None
    graph_shortcut_fraction: Optional[float] = None

    # Regal summary (from previous cycle)
    regal_spec_score: Optional[float] = None
    regal_coherence_score: Optional[float] = None
    regal_hack_prob: Optional[float] = None

    # Objective profile / task context
    objective_profile: Optional[str] = None  # e.g., "exploration", "exploitation", "validation"
    task_family_weights: Optional[Dict[str, float]] = None

    def sha256(self) -> str:
        from src.utils.config_digest import sha256_json
        return sha256_json(self.model_dump(mode="json"))


class KnobPolicyV1(BaseModel):
    """Output from knob calibration - overrides to PlanPolicyConfig.

    Learned or heuristic knob adjustments, always bounded by hard constraints.
    """
    model_config = ConfigDict(extra="forbid")

    policy_source: Literal["learned", "heuristic_fallback"] = "heuristic_fallback"
    model_sha: Optional[str] = None  # SHA of learned model (if used)
    regime_features_sha: Optional[str] = None  # SHA of input features

    # Gain schedule overrides (clamped by hard limits)
    gain_multiplier_override: Optional[float] = None
    conservative_multiplier_override: Optional[float] = None

    # Threshold overrides
    threshold_overrides: Optional[Dict[str, float]] = None  # e.g., {"spec_consistency_min": 0.6}

    # Patience override
    patience_override: Optional[int] = None

    # Per-task-family weight biases
    task_family_biases: Optional[Dict[str, float]] = None

    # Hard constraint validation
    clamped: bool = False  # True if any output was clamped by hard limits
    clamp_reasons: List[str] = Field(default_factory=list)

    def sha256(self) -> str:
        from src.utils.config_digest import sha256_json
        return sha256_json(self.model_dump(mode="json"))


# =============================================================================
# EconTensor - Canonical coordinate chart for economic metrics
# =============================================================================

class EconBasisSpecV1(BaseModel):
    """Basis specification for econ tensor coordinate system.

    Defines the canonical axis ordering, units, and normalization for econ tensors.
    Basis is immutable and registered - treat as an API.
    """
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "v1"
    basis_id: str  # e.g., "econ_basis_v1"

    # Canonical axis ordering (immutable once registered)
    axes: List[str]  # e.g., ["mpl_units_per_hour", "wage_parity", ...]

    # Units per axis (for documentation/validation)
    units: Dict[str, str] = Field(default_factory=dict)  # axis -> unit string

    # Normalization scales (for learned models)
    scales: Dict[str, float] = Field(default_factory=dict)  # axis -> scale (default 1.0)

    # How to handle missing axes
    missing_policy: Literal["zero_fill", "mask"] = "zero_fill"

    def sha256(self) -> str:
        """Compute stable SHA of basis spec."""
        from src.utils.config_digest import sha256_json
        return sha256_json(self.model_dump(mode="json"))


class EconTensorV1(BaseModel):
    """Economic metrics in canonical tensor form.

    The tensor is ordered by the basis axes and includes provenance.
    This is the invariant interface for econ data across the system.
    """
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "v1"

    # Basis reference
    basis_id: str
    basis_sha: str  # SHA of EconBasisSpecV1

    # Tensor values (ordered by basis.axes)
    x: List[float]

    # Optional mask for missing values (if missing_policy="mask")
    mask: Optional[List[bool]] = None

    # Source of the tensor values
    source: Literal["episode_metrics", "econ_vector", "datapack", "synthetic"] = "episode_metrics"

    # Linked regime features (if computed)
    regime_features_sha: Optional[str] = None

    # Optional debug stats
    stats: Optional[Dict[str, float]] = None  # e.g., {"norm": 1.23, "min": 0.0, "max": 2.5}

    def sha256(self) -> str:
        """Compute stable SHA of tensor."""
        from src.utils.config_digest import sha256_json
        return sha256_json(self.model_dump(mode="json"))


class LedgerEconV1(BaseModel):
    """Econ tensor provenance for ledger records."""
    model_config = ConfigDict(extra="forbid")

    basis_sha: str
    econ_tensor_sha: str
    econ_tensor_summary: Optional[Dict[str, float]] = None  # e.g., {"norm": 1.5, "mpl": 0.8}


# =============================================================================
# ValueLedgerRecordV1 - Single ledger entry for realized value
# =============================================================================

class LedgerWindowV1(BaseModel):
    """Training window specification."""
    model_config = ConfigDict(extra="forbid")

    step_start: int
    step_end: int
    ts_start: str
    ts_end: str


class LedgerExposureV1(BaseModel):
    """Exposure record for datapacks/slices used during window."""
    model_config = ConfigDict(extra="forbid")

    datapack_ids: List[str]
    slice_ids: Optional[List[str]] = None
    exposure_manifest_sha: str


class LedgerPolicyV1(BaseModel):
    """Policy checkpoint references."""
    model_config = ConfigDict(extra="forbid")

    policy_before: str
    policy_after: str


class LedgerAuditV1(BaseModel):
    """Audit suite references and results."""
    model_config = ConfigDict(extra="forbid")

    audit_suite_id: str
    audit_seed: int
    audit_config_sha: str
    audit_results_before_sha: str
    audit_results_after_sha: str


class LedgerDeltasV1(BaseModel):
    """Realized deltas from audit before/after."""
    model_config = ConfigDict(extra="forbid")

    delta_success: Optional[float] = None
    delta_error: Optional[float] = None
    delta_energy_Wh: Optional[float] = None
    delta_mpl_proxy: Optional[float] = None


class LedgerProbeV1(BaseModel):
    """Probe harness results in ledger (optional)."""
    model_config = ConfigDict(extra="forbid")

    probe_config_sha: str
    probe_report_sha: str
    delta_epi_per_flop: float
    stability_pass: bool
    transfer_pass: bool


class LedgerPlanPolicyV1(BaseModel):
    """Plan policy application details."""
    model_config = ConfigDict(extra="forbid")

    policy_config_sha: str
    gain_schedule_sha: str
    gain_schedule_source: str = "default"
    applied_multiplier: Optional[float] = None
    clamped: bool = False

    # Normalization & Audit
    pre_weights_sha: Optional[str] = None
    post_weights_sha: Optional[str] = None
    renormalized: bool = False
    clamp_reasons: List[str] = Field(default_factory=list)

    transfer_failure_count: int = 0

    # D4 Knob calibration provenance
    knob_policy: Optional["KnobPolicyV1"] = None


class LedgerGraphV1(BaseModel):
    """Graph small-world metrics for ledger entry."""
    model_config = ConfigDict(extra="forbid")

    graph_spec_sha: str
    graph_summary_sha: Optional[str] = None
    sigma: float
    nav_success_rate: float
    shortcut_fraction: float


class ValueLedgerRecordV1(BaseModel):
    """Single record in the realized value ledger.

    Append-only, immutable record linking:
    - Plan exposure
    - Training window
    - Policy checkpoints
    - Audit results
    - Realized deltas
    - Plan policy
    """
    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION_LEDGER
    record_id: str
    run_id: str
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())

    # Plan reference
    plan_id: str
    plan_sha: str

    # Exposure, window, policy, audit
    exposure: LedgerExposureV1
    window: LedgerWindowV1
    policy: LedgerPolicyV1
    audit: LedgerAuditV1

    # Deltas
    deltas: LedgerDeltasV1

    # Probe (optional)
    probe: Optional[LedgerProbeV1] = None
    
    # Plan Policy (optional)
    plan_policy: Optional[LedgerPlanPolicyV1] = None
    
    # Graph metrics (optional)
    graph: Optional[LedgerGraphV1] = None

    # Regal evaluation (optional, Stage-6 meta-regal)
    regal: Optional[LedgerRegalV1] = None

    # Knob policy (optional, D4 learned/heuristic calibration)
    knob_policy: Optional["KnobPolicyV1"] = None

    # Econ tensor provenance (canonical coordinate chart)
    econ: Optional[LedgerEconV1] = None

    notes: Optional[str] = None

    # Regal provenance status (P0: never-silent-failure)
    # True if regal was missing or incomplete during this run
    regal_degraded: bool = False
    # True if regal gates passed (allow deployment)
    allow_deploy: bool = True
    # True if plan was actually applied (False if halted before apply)
    plan_applied: bool = True
    # Regal summary (per-regal pass/fail for quick lookup)
    regal_summary: Optional[Dict[str, "LedgerRegalSummaryV1"]] = None


# =============================================================================
# Run Manifest - Provenance for deterministic runs
# =============================================================================

class RunManifestV1(BaseModel):
    """Provenance manifest for a closed-loop run."""
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "v1"
    run_id: str
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())

    # Source control
    source_commit: Optional[str] = None

    # Plan
    plan_path: Optional[str] = None
    plan_sha: str

    # Audit
    audit_suite_id: str
    audit_seed: int
    audit_config_sha: str

    # Datapacks
    datapack_manifest_sha: str

    # Determinism
    seeds: Dict[str, int]
    determinism_config: Optional[Dict[str, Any]] = None

    # Probe (optional)
    probe_config_sha: Optional[str] = None
    probe_report_sha: Optional[str] = None

    # Plan Policy (optional)
    plan_policy_config_sha: Optional[str] = None
    
    # Weight Provenance
    baseline_weights_sha: Optional[str] = None
    final_weights_sha: Optional[str] = None
    plan_applied_events_sha: Optional[str] = None
    
    # Graph provenance (optional)
    graph_spec_sha: Optional[str] = None
    graph_summary_sha: Optional[str] = None

    # Regal provenance (optional, Stage-6 meta-regal)
    regal_config_sha: Optional[str] = None
    regal_report_sha: Optional[str] = None
    regal_inputs_sha: Optional[str] = None
    regal_context_sha: Optional[str] = None  # SHA of RegalContextV1

    # Knob calibration provenance (D4)
    knob_model_sha: Optional[str] = None  # SHA of learned model (if used)
    knob_policy_sha: Optional[str] = None  # SHA of KnobPolicyV1 output
    knob_policy_used: Optional[Literal["learned", "heuristic_fallback"]] = None

    # Trajectory audit provenance (for meta-regal grounding)
    trajectory_audit_sha: Optional[str] = None  # Aggregated SHA of trajectory audits

    # Econ tensor provenance (canonical coordinate chart)
    econ_basis_sha: Optional[str] = None  # SHA of EconBasisSpecV1
    econ_tensor_sha: Optional[str] = None  # SHA of EconTensorV1 (aggregated if multiple)

    # Schema versions used
    schema_versions: Dict[str, str] = Field(default_factory=lambda: {
        "plan": SCHEMA_VERSION_PLAN,
        "episode": SCHEMA_VERSION_EPISODE,
        "ledger": SCHEMA_VERSION_LEDGER,
        "probe": SCHEMA_VERSION_PROBE,
    })


# =============================================================================
# Overrides - Applied by PlanApplier
# =============================================================================

class TaskSamplerOverrides(BaseModel):
    """Overrides for task sampling weights."""
    model_config = ConfigDict(extra="forbid")

    weights: Dict[str, float] = Field(default_factory=dict)  # task_family -> weight
    disabled: List[str] = Field(default_factory=list)  # disabled task families


class DatapackSelectionOverrides(BaseModel):
    """Overrides for datapack selection."""
    model_config = ConfigDict(extra="forbid")

    allowlist: Optional[List[str]] = None
    denylist: Optional[List[str]] = None
    quotas: Optional[Dict[str, int]] = None


__all__ = [
    # Constants
    "SCHEMA_VERSION_PLAN",
    "SCHEMA_VERSION_EPISODE",
    "SCHEMA_VERSION_LEDGER",
    "SCHEMA_VERSION_PROBE",
    # Plan
    "PlanOpType",
    "TaskGraphOp",
    "DatapackSelectionConfig",
    "SemanticUpdatePlanV1",
    # Episode
    "DeterminismConfig",
    "EpisodeInfoSummaryV1",
    "AuditAggregateV1",
    # Probe
    "ProbeConfigV1",
    "ProbeEpiReportV1",
    # Policy
    "PlanGainScheduleV1",
    "PlanPolicyConfigV1",
    # Graph
    "GraphSpecV1",
    "GraphSummaryV1",
    "GraphGatesV1",
    # Trajectory Audit (meta-regal grounding)
    "TrajectoryAuditV1",
    # Regal (Stage-6 meta-regal)
    "RegalPhaseV1",
    "RegalContextV1",
    "RegalGatesV1",
    "RegalReportV1",
    "LedgerRegalV1",
    "LedgerRegalSummaryV1",
    "RegalAnnotationsV1",
    "KnobDeltaV1",
    # Knob Calibration (D4)
    "RegimeFeaturesV1",
    "KnobPolicyV1",
    # Econ Tensor (coordinate chart)
    "EconBasisSpecV1",
    "EconTensorV1",
    "LedgerEconV1",
    # Ledger
    "LedgerWindowV1",
    "LedgerExposureV1",
    "LedgerPolicyV1",
    "LedgerAuditV1",
    "LedgerDeltasV1",
    "LedgerProbeV1",
    "LedgerPlanPolicyV1",
    "LedgerGraphV1",
    "ValueLedgerRecordV1",
    # Manifest
    "RunManifestV1",
    # Overrides
    "TaskSamplerOverrides",
    "DatapackSelectionOverrides",
]

