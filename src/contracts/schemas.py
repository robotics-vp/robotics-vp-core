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

    # Selection mode: "threshold" keeps edges >= threshold, "top_m_per_node" keeps top M per node
    shortcut_select_mode: Literal["threshold", "top_m_per_node"] = "top_m_per_node"

    # For threshold mode: minimum score to keep a shortcut
    shortcut_score_threshold: Optional[float] = None

    # For top_m_per_node mode: max shortcuts per node (replaces shortcut_budget_per_node as primary)
    shortcut_top_m_per_node: int = 2

    # Quality filters
    mutual_knn_only: bool = True  # Keep shortcut only if mutual (i in knn(j) AND j in knn(i))
    shortcut_budget_per_node: Optional[int] = 2  # Legacy alias for shortcut_top_m_per_node

    # Safety ceiling only - NOT the primary limiter, just a global cap
    # max_shortcut_fraction * total_edges is the absolute maximum shortcuts allowed
    max_shortcut_fraction: float = 0.25

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

    notes: Optional[str] = None


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
    # Ledger
    "LedgerWindowV1",
    "LedgerExposureV1",
    "LedgerPolicyV1",
    "LedgerAuditV1",
    "LedgerDeltasV1",
    "LedgerProbeV1",
    "LedgerPlanPolicyV1",
    "ValueLedgerRecordV1",
    # Manifest
    "RunManifestV1",
    # Overrides
    "TaskSamplerOverrides",
    "DatapackSelectionOverrides",
]

