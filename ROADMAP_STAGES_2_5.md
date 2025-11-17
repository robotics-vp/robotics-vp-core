# Roadmap: Stages 2-5 Implementation

**Status**: Post-Stage 1 completion. All smokes passing, schema-validated, additive-only architecture.

**Constraints**:
- Advisory-only, additive-only
- No Phase B or reward math modifications
- No changes to existing RL training loops
- All new modules must have smoke tests

---

## Current Architecture Summary

### ✅ **Stage 1: Complete**
- **Video → Diffusion → VLA → DataPack** pipeline functional
- DataPackMeta schema with econ/semantic tags
- RL episode descriptor normalization
- Stage 1 → RL conversion validated
- Econ-semantic analysis infrastructure
- Meta-transformer scaffolding (synthetic data)
- SIMA-2 adapter stubs
- Orchestrator supervision dataset builder (unpopulated)
- All contract-level smokes passing

### 🏗️ **Infrastructure in Place**
- **Ontology**: ObjectSpec, AffordanceSpec, EnvironmentOntology
- **Task Graph**: TaskNode, TaskGraph with hierarchical decomposition
- **SemanticOrchestrator**: Consumes econ/datapack signals, mutates task graph & ontology
- **MetaTransformer**: Cross-attention VLA+DINO scaffold (not trained)
- **SIMA-2 Bridge**: Context adapters and agent stubs
- **Econ/Datapack Engines**: EconomicController, DatapackEngine
- **Diffusion Stubs**: Real video diffusion with semantic tags
- **VLA Planners**: Transformer-based skill plan generation

---

## 🎯 Stage 2: SIMA-2 Semantic Co-Learning + Ontology Integration

### **Goal**
Connect SIMA-2's semantic primitives to ontology updates and task graph refinement, creating a feedback loop where semantic discoveries update the ontology and influence datapack semantic tags.

### **Components to Build**

#### 2.1 SIMA-2 Semantic Primitive Extraction
**File**: `src/sima2/semantic_primitive_extractor.py`

```python
@dataclass
class SemanticPrimitive:
    """Discovered semantic primitive from SIMA-2 rollout"""
    primitive_id: str
    primitive_type: str  # "action", "object", "relation", "constraint"
    name: str
    description: str
    confidence: float
    evidence: List[Any]  # Supporting observations
    ontology_mapping: Optional[str] = None  # Maps to ontology concept

class SemanticPrimitiveExtractor:
    """Extract semantic primitives from SIMA-2 agent rollouts"""

    def extract_primitives(
        self,
        sima_rollout: Dict[str, Any],
        context: Dict[str, Any]
    ) -> List[SemanticPrimitive]:
        """
        Parse SIMA-2 semantic rollout and extract primitives.

        Returns list of discovered semantic primitives that can be
        mapped to ontology concepts.
        """
        pass

    def map_to_ontology(
        self,
        primitive: SemanticPrimitive,
        ontology: EnvironmentOntology
    ) -> Optional[str]:
        """
        Find closest ontology concept for a semantic primitive.

        Returns ontology object_id or affordance_type if match found.
        """
        pass
```

**Smoke Test**: `scripts/smoke_test_sima2_semantic_extraction.py`

---

#### 2.2 Ontology Update Engine
**File**: `src/orchestrator/ontology_updater.py`

```python
@dataclass
class OntologyUpdate:
    """Proposed update to ontology based on semantic discovery"""
    update_type: str  # "add_object", "add_affordance", "update_confidence", "add_constraint"
    target_id: str
    changes: Dict[str, Any]
    rationale: str
    confidence: float
    source: str  # "sima2", "vla", "diffusion", "econ_signals"

class OntologyUpdateEngine:
    """
    Manages ontology updates from semantic discoveries.

    Consumes:
    - SIMA-2 semantic primitives
    - VLA affordance predictions
    - Diffusion semantic tags
    - Econ/datapack signals (via SemanticOrchestrator)

    Produces:
    - Ontology mutations
    - Updated affordance confidences
    - New object/constraint discoveries
    """

    def propose_update_from_primitive(
        self,
        primitive: SemanticPrimitive,
        ontology: EnvironmentOntology
    ) -> Optional[OntologyUpdate]:
        """Propose ontology update from semantic primitive"""
        pass

    def apply_update(
        self,
        update: OntologyUpdate,
        ontology: EnvironmentOntology
    ) -> bool:
        """Apply validated update to ontology"""
        pass

    def reconcile_conflicts(
        self,
        updates: List[OntologyUpdate]
    ) -> List[OntologyUpdate]:
        """Resolve conflicting updates (e.g., different confidence scores)"""
        pass
```

**Smoke Test**: `scripts/smoke_test_ontology_updater.py`

---

#### 2.3 Task Graph Semantic Refinement
**File**: `src/orchestrator/task_graph_refiner.py`

```python
class TaskGraphRefiner:
    """
    Refines task graph based on semantic discoveries and econ signals.

    Works WITH SemanticOrchestrator:
    - SemanticOrchestrator sets priorities based on econ/datapack urgencies
    - TaskGraphRefiner adds/removes/merges nodes based on semantic discoveries
    """

    def refine_from_primitives(
        self,
        task_graph: TaskGraph,
        primitives: List[SemanticPrimitive],
        ontology: EnvironmentOntology
    ) -> List[TaskNode]:
        """
        Propose task graph refinements based on discovered primitives.

        May suggest:
        - New checkpoint nodes (e.g., "verify fragile object safe")
        - Skill decomposition (split complex skill into sub-skills)
        - Task merging (combine redundant tasks)
        """
        pass

    def insert_safety_checkpoints(
        self,
        task_graph: TaskGraph,
        fragile_objects: List[ObjectSpec]
    ) -> List[str]:
        """Insert checkpoint nodes near fragile object interactions"""
        pass

    def reorder_for_efficiency(
        self,
        task_graph: TaskGraph,
        energy_costs: Dict[str, float]
    ) -> TaskGraph:
        """Reorder tasks to minimize energy (if econ signals warrant)"""
        pass
```

**Smoke Test**: `scripts/smoke_test_task_graph_refiner.py`

---

#### 2.4 Semantic Tag Propagation to DataPacks
**File**: `src/valuation/semantic_tag_propagator.py`

```python
class SemanticTagPropagator:
    """
    Propagates semantic discoveries back to datapack tags.

    Flow:
    1. SIMA-2/VLA/diffusion discover new semantics
    2. Ontology updated
    3. Existing datapacks get retroactive semantic tags
    4. Future datapacks inherit updated semantic vocabulary
    """

    def update_datapack_tags(
        self,
        datapack: DataPackMeta,
        ontology: EnvironmentOntology,
        primitives: List[SemanticPrimitive]
    ) -> List[str]:
        """
        Add semantic tags to existing datapack based on ontology updates.

        Returns new tags added.
        """
        pass

    def compute_semantic_quality_score(
        self,
        datapack: DataPackMeta,
        ontology: EnvironmentOntology
    ) -> float:
        """
        Compute semantic_quality based on ontology alignment.

        Higher score = better alignment with current ontology.
        """
        pass
```

**Smoke Test**: `scripts/smoke_test_semantic_tag_propagation.py`

---

#### 2.5 Integrated Stage 2 Pipeline
**File**: `scripts/run_stage2_semantic_colearning.py`

```python
def run_stage2_pipeline(
    sima_rollouts: List[Dict[str, Any]],
    ontology: EnvironmentOntology,
    task_graph: TaskGraph,
    existing_datapacks: List[DataPackMeta],
    output_dir: str
) -> Dict[str, Any]:
    """
    Stage 2: SIMA-2 semantic co-learning pipeline.

    Steps:
    1. Extract semantic primitives from SIMA-2 rollouts
    2. Propose ontology updates
    3. Refine task graph
    4. Propagate semantic tags to datapacks
    5. Export updated ontology + task graph + datapack tags

    Returns:
        stats: Pipeline statistics
    """
    pass
```

**Smoke Test**: `scripts/smoke_test_stage2_pipeline.py`

---

### **Stage 2 Success Metrics**
- [ ] SIMA-2 semantic primitives extracted and mapped to ontology
- [ ] Ontology updated with new affordances/objects/constraints
- [ ] Task graph refined with safety checkpoints and efficiency reorderings
- [ ] Existing datapacks enriched with retroactive semantic tags
- [ ] All Stage 2 smoke tests passing
- [ ] No modifications to Phase B or reward math

---

## 🎯 Stage 3: RL Episode Sampling from Stage 1 Datapacks

### **Goal**
Build production-ready RL sampling pipeline that uses Stage 1 datapacks + meta-transformer outputs to generate training episodes with correct semantic/econ conditioning.

### **Components to Build**

#### 3.1 DataPack-Driven RL Sampler
**File**: `src/training/datapack_rl_sampler.py`

```python
@dataclass
class RLSampleSpec:
    """Specification for an RL training sample"""
    datapack_id: str
    episode_id: str
    init_condition: ConditionProfile
    objective_vector: List[float]
    guidance: GuidanceProfile
    meta_transformer_state: Optional[np.ndarray] = None
    semantic_context: Dict[str, Any] = field(default_factory=dict)

class DataPackRLSampler:
    """
    Samples RL training episodes from Stage 1 datapacks.

    Uses:
    - DataPackMeta (from Stage 1)
    - MetaTransformer outputs (advisory)
    - Task graph + ontology (for semantic context)
    - Econ/datapack signals (for urgency-based sampling)
    """

    def __init__(
        self,
        datapacks: List[DataPackMeta],
        meta_transformer: MetaTransformer,
        task_graph: TaskGraph,
        ontology: EnvironmentOntology,
        econ_controller: EconomicController,
        datapack_engine: DatapackEngine
    ):
        pass

    def sample_batch(
        self,
        batch_size: int,
        sampling_strategy: str = "balanced"  # "balanced", "frontier_focused", "urgency_driven"
    ) -> List[RLSampleSpec]:
        """
        Sample batch of RL episodes from datapacks.

        Sampling strategies:
        - balanced: Uniform sampling across tiers
        - frontier_focused: Oversample tier-2 (causal-novel)
        - urgency_driven: Sample based on econ urgency signals
        """
        pass

    def apply_meta_transformer(
        self,
        sample_spec: RLSampleSpec
    ) -> RLSampleSpec:
        """
        Run meta-transformer on sample to get advisory state.

        Does NOT modify reward path - only provides context.
        """
        pass

    def construct_episode_descriptor(
        self,
        sample_spec: RLSampleSpec
    ) -> Dict[str, Any]:
        """
        Convert RLSampleSpec to normalized episode descriptor.

        Compatible with existing RL training loop.
        """
        pass
```

**Smoke Test**: Already exists - `scripts/smoke_test_stage1_to_rl_sampling.py`
**Enhancement**: Update to test all sampling strategies.

---

#### 3.2 Curriculum Learning Scheduler
**File**: `src/training/datapack_curriculum.py`

```python
@dataclass
class CurriculumPhase:
    """Phase in curriculum learning schedule"""
    phase_name: str
    start_episode: int
    end_episode: int
    tier_weights: Dict[int, float]  # Tier -> sampling weight
    objective_preset: str
    exploration_rate: float
    semantic_focus: List[str]  # Which semantic tags to emphasize

class DataPackCurriculum:
    """
    Manages curriculum learning schedule for datapack sampling.

    Phases:
    1. Warmup: Easy tasks, tier-0/1 only
    2. Skill building: Balanced tier sampling
    3. Frontier exploration: Oversample tier-2
    4. Economic fine-tuning: Urgency-driven sampling
    """

    def __init__(self, phases: List[CurriculumPhase]):
        pass

    def get_current_phase(self, episode: int) -> CurriculumPhase:
        """Get curriculum phase for given episode"""
        pass

    def get_sampling_weights(self, episode: int) -> Dict[int, float]:
        """Get tier sampling weights for current phase"""
        pass

    def should_advance_phase(
        self,
        episode: int,
        performance_metrics: Dict[str, float]
    ) -> bool:
        """Check if ready to advance to next curriculum phase"""
        pass
```

**Smoke Test**: `scripts/smoke_test_datapack_curriculum.py`

---

#### 3.3 Integrated Stage 3 Training Script
**File**: `scripts/run_stage3_rl_training.py`

```python
def run_stage3_training(
    datapacks: List[DataPackMeta],
    curriculum: DataPackCurriculum,
    num_episodes: int,
    output_dir: str
) -> Dict[str, Any]:
    """
    Stage 3: RL training using datapack-driven sampling.

    Steps:
    1. Initialize sampler with datapacks
    2. For each curriculum phase:
       a. Sample episodes according to tier weights
       b. Run RL training (existing loop, no modifications)
       c. Log semantic/econ metrics
       d. Check phase advancement criteria
    3. Export training logs and final policy

    IMPORTANT: Does NOT modify RL training loop or reward math.
    Only changes HOW episodes are sampled.
    """
    pass
```

**Smoke Test**: `scripts/smoke_test_stage3_training.py`

---

### **Stage 3 Success Metrics**
- [ ] RL episodes sampled from Stage 1 datapacks
- [ ] All sampling strategies (balanced, frontier, urgency) working
- [ ] Curriculum learning phases transition correctly
- [ ] Meta-transformer provides advisory context (not used in reward)
- [ ] Semantic/econ tags logged for every episode
- [ ] Existing RL training loop unchanged
- [ ] All Stage 3 smoke tests passing

---

## 🎯 Stage 4: Diffusion on Optimal Simulations

### **Goal**
Use RL-trained policies to generate "optimal" simulation rollouts, then train diffusion models on these rollouts to generate high-quality synthetic training data.

### **Components to Build**

#### 4.1 Optimal Simulation Rollout Generator
**File**: `src/diffusion/optimal_sim_generator.py`

```python
@dataclass
class OptimalRollout:
    """Rollout from optimal (RL-trained) policy"""
    episode_id: str
    task_type: str
    policy_checkpoint: str
    frames: List[np.ndarray]  # Visual observations
    states: List[np.ndarray]  # Robot states
    actions: List[np.ndarray]  # Actions taken
    rewards: List[float]
    semantic_tags: List[str]
    econ_metrics: Dict[str, float]
    success: bool

class OptimalSimGenerator:
    """
    Generates optimal simulation rollouts using trained RL policies.

    Flow:
    1. Load trained policy from Stage 3
    2. Run policy in simulation with various initial conditions
    3. Record high-quality rollouts (success rate, MPL, low error)
    4. Export rollouts for diffusion training
    """

    def generate_rollouts(
        self,
        policy_checkpoint: str,
        datapacks: List[DataPackMeta],
        num_rollouts: int,
        quality_threshold: float = 0.8
    ) -> List[OptimalRollout]:
        """
        Generate optimal rollouts by running trained policy.

        Only keeps rollouts that meet quality threshold.
        """
        pass

    def augment_with_semantic_tags(
        self,
        rollout: OptimalRollout,
        ontology: EnvironmentOntology,
        task_graph: TaskGraph
    ) -> List[str]:
        """Add semantic tags based on observed behavior"""
        pass
```

**Smoke Test**: `scripts/smoke_test_optimal_sim_generator.py`

---

#### 4.2 Diffusion Training on Optimal Sims
**File**: `src/diffusion/optimal_sim_diffusion_trainer.py`

```python
class OptimalSimDiffusionTrainer:
    """
    Trains diffusion model on optimal simulation rollouts.

    Produces:
    - High-quality synthetic data for policy distillation
    - Data augmentation for under-represented scenarios
    - Counterfactual simulations ("what if vase was closer?")
    """

    def prepare_training_data(
        self,
        rollouts: List[OptimalRollout],
        semantic_conditioning: bool = True
    ) -> Dataset:
        """Convert optimal rollouts to diffusion training format"""
        pass

    def train(
        self,
        training_data: Dataset,
        num_epochs: int,
        conditioning_mode: str = "semantic_tags"
    ) -> DiffusionModel:
        """
        Train diffusion model (stub - no real training).

        Returns trained diffusion model stub.
        """
        pass

    def generate_synthetic_rollouts(
        self,
        diffusion_model: DiffusionModel,
        conditioning: Dict[str, Any],
        num_samples: int
    ) -> List[OptimalRollout]:
        """
        Generate synthetic rollouts from trained diffusion model.

        Conditioning can include:
        - Semantic tags ("fragile", "high_speed")
        - Objective preset ("safety", "throughput")
        - Econ constraints (MPL target, error budget)
        """
        pass
```

**Smoke Test**: `scripts/smoke_test_optimal_sim_diffusion.py`

---

#### 4.3 Synthetic Data → DataPack Pipeline
**File**: `src/diffusion/synthetic_to_datapack.py`

```python
class SyntheticDataPackBuilder:
    """
    Converts synthetic diffusion rollouts to Stage 1 datapacks.

    Enables recursive improvement:
    1. Stage 3 trains policy
    2. Stage 4 generates optimal sims
    3. Diffusion creates variations
    4. Convert to datapacks
    5. Feed back to Stage 3
    """

    def build_datapack_from_synthetic(
        self,
        synthetic_rollout: OptimalRollout,
        ontology: EnvironmentOntology,
        task_graph: TaskGraph
    ) -> DataPackMeta:
        """
        Build DataPackMeta from synthetic rollout.

        Similar to Stage 1 pipeline but uses RL-optimal data.
        """
        pass
```

**Smoke Test**: `scripts/smoke_test_synthetic_to_datapack.py`

---

#### 4.4 Integrated Stage 4 Pipeline
**File**: `scripts/run_stage4_diffusion_optimal.py`

```python
def run_stage4_pipeline(
    trained_policy: str,
    datapacks: List[DataPackMeta],
    ontology: EnvironmentOntology,
    task_graph: TaskGraph,
    num_rollouts: int,
    num_synthetic: int,
    output_dir: str
) -> Dict[str, Any]:
    """
    Stage 4: Diffusion on optimal simulations.

    Steps:
    1. Generate optimal rollouts from trained policy
    2. Train diffusion model on optimal rollouts
    3. Generate synthetic variations
    4. Convert synthetic data to datapacks
    5. Export for Stage 3 recursive improvement
    """
    pass
```

**Smoke Test**: `scripts/smoke_test_stage4_pipeline.py`

---

### **Stage 4 Success Metrics**
- [ ] Optimal simulation rollouts generated from trained RL policy
- [ ] Diffusion model stub trained on optimal sims
- [ ] Synthetic rollouts generated with semantic conditioning
- [ ] Synthetic data converted to Stage 1 datapacks
- [ ] Recursive improvement loop functional (Stage 3 → 4 → 3)
- [ ] All Stage 4 smoke tests passing

---

## 🎯 Stage 5: Real Robot Logs → Datapacks → Econ Loop

### **Goal**
Close the loop: real robot deployments generate logs, logs become datapacks, datapacks feed back to training, economic metrics drive pricing and data valuation.

### **Components to Build**

#### 5.1 Real Robot Log Ingestion
**File**: `src/robot/real_log_ingestor.py`

```python
@dataclass
class RealRobotLog:
    """Log from real robot deployment"""
    deployment_id: str
    timestamp: float
    task_type: str
    customer_id: str
    frames: List[np.ndarray]  # Camera feeds
    depth: Optional[List[np.ndarray]]
    robot_states: List[np.ndarray]
    actions: List[np.ndarray]
    events: List[Dict[str, Any]]  # Errors, collisions, etc.
    outcome: Dict[str, Any]  # Success, MPL, error rate, energy used
    econ_context: Dict[str, Any]  # Wage, pricing, customer segment

class RealLogIngestor:
    """
    Ingests real robot deployment logs.

    Sources:
    - Production robot fleets
    - Field testing deployments
    - Customer pilot programs
    """

    def ingest_log(
        self,
        log_path: str,
        validate: bool = True
    ) -> RealRobotLog:
        """Load and validate real robot log"""
        pass

    def extract_econ_metrics(
        self,
        log: RealRobotLog
    ) -> Dict[str, float]:
        """
        Extract realized economic metrics from deployment.

        Returns:
        - realized_mpl: Actual marginal product
        - realized_error_rate: Actual error rate
        - realized_energy_Wh: Actual energy consumption
        - realized_wage_parity: ŵᵣ/wₕ
        """
        pass
```

**Smoke Test**: `scripts/smoke_test_real_log_ingestion.py`

---

#### 5.2 Real Log → DataPack Converter
**File**: `src/robot/real_log_to_datapack.py`

```python
class RealLogDataPackBuilder:
    """
    Converts real robot logs to Stage 1 datapacks.

    Key difference from synthetic:
    - Uses ACTUAL econ metrics (not estimated)
    - Higher trust_score (real data)
    - Enables ex-post true-up (compare predicted vs realized)
    """

    def build_datapack_from_real_log(
        self,
        log: RealRobotLog,
        ontology: EnvironmentOntology,
        task_graph: TaskGraph,
        vla_analysis: Optional[Dict[str, Any]] = None
    ) -> DataPackMeta:
        """
        Build DataPackMeta from real deployment log.

        Uses realized metrics for attribution:
        - delta_mpl = actual MPL improvement
        - delta_error = actual error reduction
        - trust_score = 1.0 (ground truth)
        """
        pass

    def compute_novelty_from_real(
        self,
        log: RealRobotLog,
        existing_datapacks: List[DataPackMeta]
    ) -> float:
        """
        Compute novelty score by comparing to existing training data.

        High novelty = new scenario not well-covered in training.
        """
        pass
```

**Smoke Test**: `scripts/smoke_test_real_log_to_datapack.py`

---

#### 5.3 Ex-Post Economic Reconciliation
**File**: `src/valuation/ex_post_reconciliation.py`

```python
@dataclass
class EconReconciliation:
    """Ex-post reconciliation of predicted vs realized metrics"""
    datapack_id: str
    customer_id: str

    # Predicted (ex-ante)
    predicted_mpl: float
    predicted_error: float
    predicted_energy_Wh: float
    premium_quoted: float

    # Realized (ex-post)
    realized_mpl: float
    realized_error: float
    realized_energy_Wh: float

    # True-up
    prediction_error_mpl: float
    prediction_error_error: float
    prediction_error_energy: float
    trueup_amount: float  # Refund (+) or charge (-)

class ExPostReconciliator:
    """
    Reconciles predicted vs realized economic metrics.

    Flow:
    1. Training datapack has predicted delta_mpl, tier, premium
    2. Real deployment measures actual performance
    3. Compare predicted vs realized
    4. True-up pricing (refund if over-charged, bill if under-charged)
    5. Update prediction model (improve future estimates)
    """

    def reconcile(
        self,
        training_datapack: DataPackMeta,
        real_log: RealRobotLog
    ) -> EconReconciliation:
        """Compute ex-post reconciliation"""
        pass

    def update_prediction_model(
        self,
        reconciliations: List[EconReconciliation]
    ) -> None:
        """
        Use reconciliation data to improve future predictions.

        Adjusts:
        - Novelty → MPL gain mapping
        - Tier thresholds
        - Confidence discounts (κ)
        """
        pass
```

**Smoke Test**: `scripts/smoke_test_ex_post_reconciliation.py`

---

#### 5.4 Data Sharing Incentive Engine
**File**: `src/valuation/data_sharing_incentives.py`

```python
@dataclass
class DataSharingContract:
    """Contract terms for data sharing"""
    customer_id: str
    sharing_tier: str  # "full_share", "partial_share", "no_share"
    discount_rate: float  # Discount if data is shared
    premium_rate: float  # Premium if data withheld
    horizon_hours: int
    contract_start: float
    contract_end: float

class DataSharingIncentiveEngine:
    """
    Manages economic incentives for data sharing.

    Principle: Share data → cheaper pricing; don't share → pay premium

    Uses:
    - Novelty scores
    - Expected MPL gains
    - Customer segments
    - Market dynamics
    """

    def quote_pricing(
        self,
        customer_id: str,
        task_type: str,
        sharing_status: str,
        datapack_meta: DataPackMeta
    ) -> Dict[str, float]:
        """
        Quote pricing based on sharing status.

        Returns:
        - base_price: Standard price
        - discount: If sharing data
        - premium: If withholding data
        - final_price: Actual charge
        """
        pass

    def simulate_market_equilibrium(
        self,
        num_customers: int,
        sharing_rates: List[float]
    ) -> Dict[str, Any]:
        """
        Simulate market equilibrium under different data sharing scenarios.

        Shows:
        - Total data collected vs withheld
        - Average pricing by tier
        - MPL improvement rates
        - Customer surplus
        """
        pass
```

**Smoke Test**: `scripts/smoke_test_data_sharing_incentives.py`

---

#### 5.5 Integrated Stage 5 Pipeline
**File**: `scripts/run_stage5_real_robot_loop.py`

```python
def run_stage5_pipeline(
    real_logs_dir: str,
    training_datapacks: List[DataPackMeta],
    ontology: EnvironmentOntology,
    task_graph: TaskGraph,
    customer_contracts: List[DataSharingContract],
    output_dir: str
) -> Dict[str, Any]:
    """
    Stage 5: Real robot logs → datapacks → econ loop.

    Steps:
    1. Ingest real robot deployment logs
    2. Extract realized econ metrics
    3. Convert logs to datapacks (if customer shared data)
    4. Reconcile predicted vs realized (ex-post true-up)
    5. Update pricing model
    6. Feed real datapacks back to Stage 3 training
    7. Export reconciliation reports and updated pricing
    """
    pass
```

**Smoke Test**: `scripts/smoke_test_stage5_pipeline.py`

---

### **Stage 5 Success Metrics**
- [ ] Real robot logs ingested and validated
- [ ] Realized econ metrics extracted from deployments
- [ ] Real logs converted to high-trust datapacks
- [ ] Ex-post reconciliation computed (predicted vs realized)
- [ ] Pricing model updated based on reconciliation
- [ ] Data sharing incentives functional
- [ ] Real datapacks fed back to Stage 3 training
- [ ] All Stage 5 smoke tests passing
- [ ] Full feedback loop: Real → DataPack → Training → Real

---

## 🧬 Meta-Transformer Training Curriculum (Real Supervision)

### **Goal**
Transition meta-transformer from synthetic data to real supervision from Stages 1-5.

### **Supervision Sources**

#### 1. **Stage 1 Supervision**: Diffusion + VLA Alignment
**File**: `src/orchestrator/meta_transformer_stage1_data.py`

```python
class Stage1SupervisionBuilder:
    """
    Build meta-transformer training data from Stage 1 pipeline.

    Supervision signal:
    - Input: Diffusion semantic tags + VLA skill plans
    - Target: Objective preset that maximizes datapack trust_score
    """

    def build_supervision_dataset(
        self,
        stage1_datapacks: List[DataPackMeta]
    ) -> List[MetaTransformerTrainingExample]:
        """
        Convert Stage 1 datapacks to meta-transformer supervision.

        Each example:
        - Input: (semantic_tags, vla_skills, econ_context)
        - Target: (objective_preset, expected_delta_mpl, tier)
        """
        pass
```

---

#### 2. **Stage 3 Supervision**: RL Performance Prediction
**File**: `src/orchestrator/meta_transformer_stage3_data.py`

```python
class Stage3SupervisionBuilder:
    """
    Build meta-transformer training data from Stage 3 RL training.

    Supervision signal:
    - Input: DataPack semantic/econ features
    - Target: Actual RL performance (reward, MPL, error rate)
    """

    def build_supervision_dataset(
        self,
        rl_training_logs: List[Dict[str, Any]]
    ) -> List[MetaTransformerTrainingExample]:
        """
        Convert RL training logs to meta-transformer supervision.

        Each example:
        - Input: (datapack features, curriculum phase)
        - Target: (achieved reward, MPL improvement, error rate)
        """
        pass
```

---

#### 3. **Stage 5 Supervision**: Real Deployment Outcomes
**File**: `src/orchestrator/meta_transformer_stage5_data.py`

```python
class Stage5SupervisionBuilder:
    """
    Build meta-transformer training data from real deployments.

    Supervision signal (STRONGEST):
    - Input: Predicted econ metrics
    - Target: Realized econ metrics from deployment

    This is ground truth - use for fine-tuning.
    """

    def build_supervision_dataset(
        self,
        reconciliations: List[EconReconciliation]
    ) -> List[MetaTransformerTrainingExample]:
        """
        Convert ex-post reconciliations to meta-transformer supervision.

        Each example:
        - Input: (predicted_mpl, predicted_error, semantic_context)
        - Target: (realized_mpl, realized_error, prediction_error)
        """
        pass
```

---

#### 4. **Integrated Meta-Transformer Training**
**File**: `scripts/train_meta_transformer_real_supervision.py`

```python
def train_meta_transformer_with_real_data(
    stage1_datapacks: List[DataPackMeta],
    stage3_logs: List[Dict[str, Any]],
    stage5_reconciliations: List[EconReconciliation],
    num_epochs: int,
    output_dir: str
) -> Dict[str, Any]:
    """
    Train meta-transformer with real supervision from all stages.

    Training schedule:
    1. Warmup: Stage 1 data (diffusion + VLA alignment)
    2. Skill building: Stage 3 data (RL performance)
    3. Fine-tuning: Stage 5 data (real deployments)

    Loss components:
    - Objective preset classification
    - Delta MPL regression
    - Error rate regression
    - Energy prediction

    Does NOT touch reward math - purely advisory.
    """
    pass
```

**Smoke Test**: `scripts/smoke_test_meta_transformer_real_training.py`

---

## 🔗 SemanticOrchestrator + Econ/Datapack Integration

### **Enhanced Integration**
**File**: `src/orchestrator/semantic_orchestrator_v2.py`

```python
class SemanticOrchestratorV2(SemanticOrchestrator):
    """
    Extended SemanticOrchestrator with Stage 2-5 integration.

    New capabilities:
    - Consumes SIMA-2 semantic primitives
    - Coordinates ontology updates from multiple sources
    - Prioritizes based on real deployment data
    - Adjusts task graph based on actual failure modes
    """

    def integrate_sima2_primitives(
        self,
        primitives: List[SemanticPrimitive],
        econ_signals: EconSignals
    ) -> SemanticUpdatePlan:
        """
        Build update plan incorporating SIMA-2 discoveries.

        Prioritizes primitives that address econ urgencies.
        """
        pass

    def update_from_real_deployment(
        self,
        real_log: RealRobotLog,
        reconciliation: EconReconciliation
    ) -> SemanticUpdatePlan:
        """
        Update semantics based on real deployment outcomes.

        Highest priority updates - ground truth from field.
        """
        pass

    def propose_training_focus(
        self,
        current_datapacks: List[DataPackMeta],
        target_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Propose training focus based on gap analysis.

        Returns:
        - Recommended datapack tiers to sample
        - Semantic tags to emphasize
        - Curriculum phase suggestions
        """
        pass
```

**Smoke Test**: `scripts/smoke_test_semantic_orchestrator_v2.py`

---

## 📋 Recommended Implementation Order for Codex

### **Phase 1: Stage 2 Foundation** (Codex implements first)
1. **SemanticPrimitiveExtractor** (`src/sima2/semantic_primitive_extractor.py`)
   - Extract primitives from SIMA-2 rollouts
   - Map to ontology concepts
   - Smoke test

2. **OntologyUpdateEngine** (`src/orchestrator/ontology_updater.py`)
   - Propose updates from primitives
   - Apply updates to ontology
   - Conflict resolution
   - Smoke test

3. **TaskGraphRefiner** (`src/orchestrator/task_graph_refiner.py`)
   - Refine task graph from primitives
   - Insert safety checkpoints
   - Reorder for efficiency
   - Smoke test

4. **SemanticTagPropagator** (`src/valuation/semantic_tag_propagator.py`)
   - Update datapack tags from ontology
   - Compute semantic quality scores
   - Smoke test

5. **Stage 2 Integrated Pipeline** (`scripts/run_stage2_semantic_colearning.py`)
   - End-to-end pipeline
   - Smoke test

---

### **Phase 2: Stage 3 Sampling** (Codex implements second)
1. **DataPackRLSampler** (`src/training/datapack_rl_sampler.py`)
   - Sampling strategies (balanced, frontier, urgency)
   - Meta-transformer integration
   - Episode descriptor construction
   - Enhancement to existing smoke test

2. **DataPackCurriculum** (`src/training/datapack_curriculum.py`)
   - Curriculum phase definitions
   - Phase advancement logic
   - Smoke test

3. **Stage 3 Training Script** (`scripts/run_stage3_rl_training.py`)
   - RL training with datapack sampling
   - Curriculum integration
   - Smoke test

---

### **Phase 3: Stage 4 & 5 Preparation** (Codex implements third)
1. **OptimalSimGenerator** (`src/diffusion/optimal_sim_generator.py`)
   - Generate optimal rollouts from trained policy
   - Semantic tagging
   - Smoke test

2. **RealLogIngestor** (`src/robot/real_log_ingestor.py`)
   - Ingest real robot logs
   - Extract econ metrics
   - Smoke test

3. **RealLogDataPackBuilder** (`src/robot/real_log_to_datapack.py`)
   - Convert real logs to datapacks
   - Novelty computation
   - Smoke test

4. **ExPostReconciliator** (`src/valuation/ex_post_reconciliation.py`)
   - Predicted vs realized reconciliation
   - Prediction model updates
   - Smoke test

---

### **Phase 4: Meta-Transformer Real Training** (Codex implements fourth)
1. **Stage1SupervisionBuilder** (`src/orchestrator/meta_transformer_stage1_data.py`)
2. **Stage3SupervisionBuilder** (`src/orchestrator/meta_transformer_stage3_data.py`)
3. **Stage5SupervisionBuilder** (`src/orchestrator/meta_transformer_stage5_data.py`)
4. **Real Supervision Training** (`scripts/train_meta_transformer_real_supervision.py`)
5. **Smoke test for each**

---

### **Phase 5: Integration & Testing** (Codex implements fifth)
1. **SemanticOrchestratorV2** with full integration
2. **Stage 4 complete pipeline** (`scripts/run_stage4_diffusion_optimal.py`)
3. **Stage 5 complete pipeline** (`scripts/run_stage5_real_robot_loop.py`)
4. **End-to-end integration test**: Stage 1 → 2 → 3 → 4 → 5 → 1 (recursive)
5. **Data sharing incentives** (`src/valuation/data_sharing_incentives.py`)
6. **Market equilibrium simulator**

---

## 🎯 Success Criteria (Overall)

### **All Stages Functional**
- [ ] Stage 1: Video → Diffusion → VLA → DataPack ✅ (Complete)
- [ ] Stage 2: SIMA-2 semantic co-learning + ontology updates
- [ ] Stage 3: RL training with datapack-driven sampling
- [ ] Stage 4: Diffusion on optimal sims
- [ ] Stage 5: Real robot logs → datapacks → econ loop

### **Semantic Consistency**
- [ ] Ontology updated from SIMA-2, VLA, diffusion, real logs
- [ ] Task graph refined based on semantic discoveries
- [ ] Semantic tags propagated across all datapacks
- [ ] Cross-module vocabulary alignment (VLA ↔ SIMA ↔ diffusion ↔ RL)

### **Economic Loop Closed**
- [ ] Ex-post reconciliation functional
- [ ] Prediction model improves over time
- [ ] Data sharing incentives aligned with MPL gains
- [ ] Market equilibrium simulations show pricing works

### **Meta-Transformer Trained**
- [ ] Real supervision from Stages 1, 3, 5
- [ ] Advisory outputs improve datapack selection
- [ ] Does NOT modify reward math (advisory only)

### **All Smoke Tests Passing**
- [ ] Stage 1 smokes ✅
- [ ] Stage 2 smokes (5 new tests)
- [ ] Stage 3 smokes (3 new tests)
- [ ] Stage 4 smokes (4 new tests)
- [ ] Stage 5 smokes (5 new tests)
- [ ] Meta-transformer training smokes (4 new tests)
- [ ] Integration smoke (1 end-to-end test)

### **No Breaking Changes**
- [ ] Phase B math untouched
- [ ] Reward builder untouched
- [ ] RL training loops unchanged (only sampling modified)
- [ ] All existing smokes still passing
- [ ] Additive-only, advisory-only architecture maintained

---

## 📝 Documentation Requirements

For each stage, Codex should create:

1. **Module docstrings** explaining purpose, inputs, outputs
2. **Smoke test** demonstrating core functionality
3. **Integration notes** in this roadmap (update as implemented)
4. **Contract validation** ensuring schema compatibility
5. **Example usage** script in `scripts/examples/`

---

## 🚀 Next Actions for Codex

**Immediate priorities** (implement in order):

1. ✅ Review this roadmap for completeness and coherence
2. 🔨 Implement **SemanticPrimitiveExtractor** (Stage 2.1)
3. 🔨 Implement **OntologyUpdateEngine** (Stage 2.2)
4. 🔨 Implement **TaskGraphRefiner** (Stage 2.3)
5. 🔨 Implement **SemanticTagPropagator** (Stage 2.4)
6. 🔨 Implement **Stage 2 Integrated Pipeline** (Stage 2.5)
7. ✅ Run all Stage 2 smoke tests
8. 📋 Report back: Stage 2 complete, ready for Stage 3

**After Stage 2 completion**, pause and ask user:
- "Stage 2 complete. Proceed with Stage 3 (RL sampling), or adjust priorities?"

---

## 🎓 Key Architectural Principles

1. **Advisory-only**: Meta-transformer, semantic orchestrator provide suggestions, not commands
2. **Additive-only**: No modifications to existing Phase B, reward, or RL logic
3. **Contract-driven**: All inter-stage communication via validated schemas
4. **Smoke-tested**: Every module has a smoke test before integration
5. **Econ-grounded**: Semantics serve economics, not vice versa
6. **Feedback loops**: Each stage feeds insights back to previous stages
7. **Real-world first**: Prioritize real deployment data over synthetic

---

## 📊 Tracking Progress

**Current state**: ✅ Stage 1 complete, all smokes passing

**Next milestone**: 🔨 Stage 2 complete (semantic co-learning + ontology)

**Future milestones**:
- 🔜 Stage 3: RL sampling from datapacks
- 🔜 Stage 4: Diffusion on optimal sims
- 🔜 Stage 5: Real robot loop closure
- 🔜 Meta-transformer real training
- 🔜 End-to-end integration test

---

**End of Roadmap**
