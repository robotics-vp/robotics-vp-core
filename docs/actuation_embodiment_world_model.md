# Embodiment / Actuation World Model

## Purpose

The Embodiment / Actuation WM is a canonical adjacent world model in this
stack's multi-WM topology. Its mission:

> Turn task intent + local world state + embodiment constraints into
> body-aware, capability-aware, contact-aware control state and action
> proposals for real robot embodiments.

This WM makes the robot ready to actually control a body. Its primary
design center is bipedal whole-body humanoid control for a Unitree G1/R1-class
robot. Stable-base mobile manipulation is the safety fallback / degraded-mode
posture, and fixed-base tabletop manipulation is a restricted curriculum and
regression profile.

It is **not** a global planner. It is the canonical body/capability/contact/
control WM for real embodiments operating under physics, safety, energy,
and deployment constraints.

---

## Embodiment Target Hierarchy

The Embodiment / Actuation WM should not treat every robot body profile as an
equal target. Its posture hierarchy is:

| Posture | Role inside this WM | Authority boundary |
| --- | --- | --- |
| `bipedal_whole_body` | Primary target for body state, action proposals, dynamics, retargeting, safety, and resource prediction | Required for G1/R1-class promotion |
| `stable_base_mobile_manipulator` | Safety fallback / degraded-mode posture for recovery, conservative manipulation, operator handoff, and partial task continuity | Can veto/defer/recover; cannot replace bipedal readiness |
| `fixed_base_tabletop` | Curriculum, smoke tests, regression, and narrow manipulation skill islands | Pretraining/regression only |

Every Embodiment WM state object, action proposal, sim receipt, replay row,
neural scaffold, and benchmark should preserve this posture. Missing posture
metadata should be treated as unknown/fixed-base evidence, not as bipedal
whole-body evidence.

The current Phase 3.5 refit artifact is
`docs/economic_world_model/phase35_humanoid_capacity_env_refit.md`. It records
the local capacity bands, observation/action schema deltas, env taxonomy,
Unitree sim integration target, and benchmark taxonomy that this WM should use
when returning from Phase-6 local transport closure.

As of 2026-05-24, that refit is also code-backed by
`src/world_model/humanoid_readiness/phase35.py` and
`scripts/economic_world_model/prepare_phase35_humanoid_capacity_env_refit.py`.
The current local artifact reports `local_structural_refit_complete=true`, but
this remains scaffold evidence only: no Unitree sim runtime, hardware execution,
training, promotion, live policy control, or reward-math mutation is claimed.

The local bipedal chassis layer is
`src/world_model/embodiment_actuation/bipedal_chassis.py`, materialized by
`scripts/economic_world_model/prepare_phase35_bipedal_chassis_scaffold.py`.
It adds a 29-DoF chassis profile, limb coordinate frames, planning joint-limit
envelopes, whole-body observation/action schemas, support-state slots, and
balance receipts. Those envelopes are not hardware-calibrated safety limits;
they are canonical local surfaces for future URDF/sim/hardware evidence.

The no-GPU/no-hardware readiness layer is
`src/world_model/embodiment_actuation/bipedal_readiness.py`, materialized by
`scripts/economic_world_model/audit_phase35_bipedal_readiness.py`. It adds
asset intake contracts, local asset parse receipts, kinematic consistency
reports, joint-vector validation receipts, balance-geometry reports, and
whole-body replay row slots. It can parse a supplied local URDF/MJCF/SRDF-style
asset enough to test joint-name alignment, but that is still contract evidence,
not calibrated Unitree runtime, hardware, training, or promotion evidence.

The Phase 4 downstream-controller layer is
`src/world_model/humanoid_readiness/downstream_controller.py`, materialized by
`scripts/economic_world_model/prepare_phase4_downstream_controller_scaffold.py`.
It consumes the bipedal chassis/readiness artifacts and emits dry-run controller
bridge targets, modes, proposals, low-level command frames, safety receipts,
dispatch-denied invocations, and replay-ready controller receipts. It is shaped
for Unitree ROS2 / SDK2 and G1Pilot-style fallback integration, but it does not
publish ROS2/DDS messages, write Unitree commands, invoke G1Pilot, or claim
live actuator authority.

The Phase 4 Unitree/G1 bring-up readiness layer is
`src/world_model/humanoid_readiness/unitree_bringup_readiness.py`, materialized
by `scripts/economic_world_model/prepare_phase4_unitree_bringup_readiness.py`.
It inventories local Unitree/G1 OSS roots, parses available G1 assets for
canonical 29-joint subset alignment, emits stream and command conformance
contracts, runs a local-only timing probe, records safety preflight gates, and
creates operator e-stop/recovery runbooks plus a sim/hardware evidence ledger.
It is a pre-purchase readiness pack only: no Unitree ROS2 / SDK2 build or write,
G1Pilot invocation, sim launch, hardware execution, safety certification,
training, reward mutation, or promotion is claimed.

The Phase 4 Unitree/G1 local harness layer is
`src/world_model/humanoid_readiness/unitree_local_harness.py`, materialized by
`scripts/economic_world_model/prepare_phase4_unitree_local_harnesses.py`. It
executes synthetic low-state, IMU, wireless/e-stop, and contact trace
import/export; parses Unitree ROS2 command and stream message definitions;
validates no-publish `LowCmd` and sport-request payload shapes; runs local
timing/watchdog loops; drives a local safety/recovery state machine; and emits
Unitree ROS2 / MuJoCo / G1Pilot preflight receipts. It remains local harness
evidence only: no live streams, ROS2/DDS publish, SDK2 write, G1Pilot
invocation, MuJoCo launch, hardware execution, safety certification, training,
reward mutation, or promotion is claimed.

---

## How Our WM Topology Differs from the Dominant Framing

Many public "world model" discussions treat the term as synonymous with a
single predictive model — an action-conditioned latent predictor, an
inverse-dynamics network, a joint world-action model, or a latent video
predictor. These are useful model families, and we borrow from them. But
they are not our ontology.

In this stack:

- **World models are adjacent canonical modules in a typed topology.**
  Perception, Embodiment, Sim/Synth/Physics, Economic, and Meta-Governance
  are separate WMs communicating via typed, replayable state surfaces.
- **No mother-latent.** We do not collapse perception, control, physics,
  economics, and governance into one uninterpretable vector embedding.
  Typed contracts (BeliefState, ObjectiveTensor, ConstraintSet, EconTensor)
  are the primary defense against that failure mode.
- **Imported architectures enter as bounded subsystem seams**, not as
  replacement ontologies. A predictive model like V-JEPA 2 or TD-MPC2
  enters as a promotion-gated, receipt-emitting, rollback-safe neural seam
  inside a specific WM — it does not become the WM.
- **Economics, governance, and constraints remain first-class and legible.**
  Every WM-boundary crossing preserves typed observability, economic
  attribution, and governance auditability.

This matters for the Embodiment / Actuation WM specifically because the
public literature often conflates "world model for control" with "one latent
dynamics model that replaces explicit body/contact/capability state." We
explicitly reject that conflation. The Embodiment WM is a typed subsystem
that *contains* local dynamics models as bounded seams, but *is not
equivalent to* any single predictive model.

---

## Relation to the Multi-WM Stack

```
Perception / Grounding WM
    ↓ scene graph, affordance bridge, object tracks, evidence routing
Embodiment / Actuation WM    ← this document
    ↓ capability state, action proposals, cost vectors, drift reports
Sim / Synth / Physics WM
    ↓ sim agendas, branch evaluations, synthetic evidence
Economic WM
    ↓ allocation envelopes, shaping fields, resource budgets
Meta-Regal-Node Superposition / Control WM
```

The Embodiment WM:

- **Consumes** Perception WM scene graph, embodiment bridge, provider truth
- **Consumes** Sim/Synth WM branch evaluations and physics forecasts
- **Consumes** Economic WM resource envelopes and deployment constraints
- **Produces** body-aware control state for downstream actuator backends
- **Produces** typed receipts for economic valuation, governance audit, and
  replay/training export
- **Produces** calibration/drift signals that feed back to Sim/Synth and
  Perception WMs

---

## Core Subsystems

The Embodiment / Actuation WM contains six named subsystems. Each is a
functional contributor to the robostack, not an abstract schema.

### 1. Capability / Embodiment State Surface

**What it is**: The canonical typed entry surface for body-aware control.
Every downstream subsystem within this WM reads from this surface.

**What it carries**:

| Field | Type | Description |
|-------|------|-------------|
| `embodiment_id` | str | Robot family + serial (e.g. `g1_unit_001`) |
| `embodiment_class` | str | `bipedal_whole_body`, `stable_base_mobile_manipulator`, `fixed_base_tabletop` (`tabletop_arm`, `mobile_manipulator`, and `humanoid` remain legacy aliases only) |
| `actuator_config` | ActuatorConfigState | Joint count, limits, torque profiles, gear ratios |
| `joint_state` | JointStateVector | Current positions, velocities, torques, temperatures |
| `contact_state` | ContactStateVector | Per-effector contact normals, forces, slip estimates |
| `tool_state` | ToolStateDescriptor | Active tool, tool-change state, tool wear estimate |
| `safety_envelope` | SafetyEnvelopeState | Joint limits, velocity limits, force limits, collision zones |
| `backend_tags` | list[str] | `isaac`, `pybullet`, `holosoma`, `real_hw`, etc. |
| `physics_profile_hash` | str | Hash of the physics config used for this embodiment |
| `compute_placement` | str | `on_device`, `companion`, `cloud` |
| `battery_fraction` | float | Current battery level (relevant for G1 energy budgeting) |

**Why it matters**: Without explicit body state, downstream control is blind
to joint limits, tool availability, contact truth, and safety constraints.
This is what turns generic task proposals into body-aware action feasibility.

**Relation to existing artifacts**: This subsystem generalizes and subsumes
`EmbodimentProfile_v1.npz`, which currently stores contacts, confidence, and
impossible-contact flags. `EmbodimentProfile_v1` is the embryonic typed output
of this surface.

### 2. Contact / Affordance Graph Builder

**What it is**: The Embodiment WM's "local world" — the actionable graph of
what can be grasped, inserted, placed, fastened, obstructed, or otherwise
physically engaged by the current embodiment configuration.

**How it works**:

- Consumes Perception WM scene graph (object tracks, edges, affordance hints)
- Consumes Perception WM embodiment bridge (per-object affordance scores,
  body-object pairwise scores, action feasibility summary)
- Consumes workcell ontology (fixture positions, tool availability, part specs)
- Consumes current Capability/Embodiment state (joint state, tool, safety)
- Produces a locally actionable contact/affordance graph:
  - per-object: graspable? insertable? placeable? fastenable? obstructed?
    misaligned? risky? within reach? within force budget?
  - per-pair: body-object engagement feasibility, approach vector quality,
    collision risk, tool compatibility
  - per-fixture: fixture state, clamp state, insertion alignment

**What it does NOT do**: It does not plan full task sequences. It builds the
local contact/affordance truth that the skill/action proposal head and the
local dynamics model consume.

**Relation to existing artifacts**: This generalizes `AffordanceGraph_v1.npz`,
which currently stores contact/affordance edges with confidence. The graph
builder is the functional compiler that produces `AffordanceGraph_v1`
content plus richer body-aware feasibility scoring.

### 3. Local Contact Dynamics Model

**What it is**: A short-horizon, embodiment-specific predictive subsystem.
It predicts what happens in the near future when the robot executes a
candidate action in the current contact/affordance state.

**What it predicts** (1-10 steps ahead, embodiment-specific):

- Contact transitions: will contact engage/disengage? Will slip occur?
- Force transients: expected force profile during insertion/engagement
- Insertion/engagement success probability
- Jam/wedge risk under current approach vector
- Recovery feasibility if current action fails
- Energy cost of the predicted trajectory

**Architecture pattern**: Inspired by action-conditioned latent dynamics
models (TD-MPC2 style bounded short-horizon planning, V-JEPA 2 style latent
prediction) but scoped to the local contact/embodiment regime. This is
explicitly NOT the global planning WM — it is a bounded predictive seam
inside the Embodiment WM, operating over the local contact graph state.

**Promotion posture**: `disabled|auto|required`. At `heuristic_fallback`,
uses analytic contact models (spring-damper, Coulomb friction). At
`promoted`, uses the learned local dynamics model. At `disabled`, passes
through without dynamics prediction.

**Relation to existing artifacts**: No direct predecessor artifact. This is
a new subsystem that should emit `LocalDynamicsForecast` typed outputs
consumable by the action proposal head, the drift evaluator, and the
economic cost attribution pipeline.

### 4. Inverse-Dynamics / Retargeting Lane

**What it is**: Recovers candidate actions / skill traces from state
transitions, demonstrations, teleoperation logs, video, or future real robot
recordings. This is the **primary home for scalable imitation-learning
ingestion** in the Embodiment / Actuation WM.

**What it does**:

- Given (state_t, state_{t+1}), recovers the action that could produce that
  transition for the current embodiment configuration
- Retargets demonstrations from one embodiment to another (e.g. teleop with
  one arm → G1 bimanual mapping)
- Bootstraps action priors from video or human demonstrations
- Produces replay-ready traces for datapack/training construction

**Scalable imitation-learning pipeline functions**:

- **Demonstration ingestion**: Teleop traces, video demonstrations, sim rollouts,
  real robot recordings all enter through typed provider contracts
- **Cross-embodiment retargeting**: Demonstrations from source embodiment
  (teleop arm, human video) mapped to target embodiment (G1, tabletop arm)
  with kinematic feasibility filtering
- **Action recovery from state transitions**: Given observation pairs, recover
  plausible action sequences that could produce the transition
- **Dataset quality assessment**: Quality scoring, provenance tracking, and
  replay/training readiness validation
- **Chunked policy/skill prior formation**: Packaging recovered action
  sequences into trainable skill priors for downstream heads

**What it does NOT do**: It is not the final controller. It produces
candidate actions and retargeted traces that the skill/action proposal head
evaluates and the training pipeline consumes.

**Architecture patterns and model families**:

| Function | Model family | Justification |
|----------|--------------|---------------|
| State-to-action recovery | Inverse-dynamics heads (MLP/Transformer) | Direct mapping from state deltas |
| Cross-embodiment retargeting | Retargeting networks with embodiment embeddings | Generalize across kinematic chains |
| Video-to-action extraction | LeRobot-style interfaces, OpenVLA (as provider) | Standardized policy/data contracts |
| Demonstration quality scoring | Learned quality predictors | Filter low-quality demonstrations |

These architectures enter as bounded subsystem seams, not as WM replacements.

**Typed artifacts emitted**:

- `DemonstrationIngestReceipt`: source type, source embodiment, frame count,
  quality estimate, ingestion timestamp, provenance refs
- `RetargetingTraceBundle`: source/target embodiment, recovered action chunks,
  retargeting quality, kinematic feasibility, failure point identification
- `ActionRecoveryReceipt`: state pairs processed, actions recovered, model
  used, confidence distribution
- `DatapackQualityReceipt`: dataset quality scores, provenance chain,
  replay/training readiness, calibration status

**Relation to existing artifacts**: This is the functional compiler behind
`SkillSegments_v1.npz`, which currently stores interaction primitive
segmentation. The inverse lane produces the raw material that
`SkillSegments_v1` captures.

### 5. Joint Skill / Action Proposal Head

**What it is**: Proposes short skill chunks or action chunks jointly with
expected state evolution. This is the Embodiment WM's primary output to the
downstream control loop and the **downstream consumer of imitation-learning
priors** from the Inverse-Dynamics / Retargeting Lane.

**What it proposes**:

- Short action chunks (4-16 steps) with expected state trajectories
- Skill primitives from the workcell task catalog (PICK, PLACE, INSERT,
  FASTEN, etc.) instantiated with concrete embodiment parameters
- Multi-modal proposals when multiple approaches are feasible (e.g. two
  valid grasp poses)
- Action-chunk confidence and expected cost vector per proposal
- **Imitation-derived action priors** from the upstream retargeting lane

**Imitation-learning integration**:

The action proposal head is where imitation-derived priors become actionable
for real control. The relationship to the upstream inverse-dynamics /
retargeting lane:

1. Inverse-dynamics lane produces `RetargetingTraceBundle` with recovered
   action chunks
2. This head consumes those bundles as trainable skill/action priors
3. Training uses imitation loss plus downstream success/safety signals
4. Promoted heads blend scripted fallbacks with learned imitation priors

**Architecture patterns and model families**:

| Function | Model family | Justification |
|----------|--------------|---------------|
| Action chunk proposal | ACT-style action chunking transformers | Multi-step chunk prediction |
| Multimodal proposal | Diffusion policy heads | Multiple valid action modes |
| Imitation prior injection | Prior-conditioned policy heads | Blend scripted and learned |
| Skill primitive instantiation | Task-conditioned policy networks | Workcell task catalog |

These architectures enter as bounded subsystem seams, not as the whole stack.

**Typed artifacts emitted**:

- `ImitationPriorSnapshot`: chunk horizon, action distribution summary,
  embodiment binding, training corpus refs, promotion stage
- `ActionProposalBundle`: proposals with imitation-derived confidence scores

**Promotion posture**: `disabled|auto|required`. At `heuristic_fallback`,
uses scripted skill primitives. At `promoted`, uses learned action chunk
proposers. The promotion gate is benchmark-gated on downstream task success
rate and safety compliance.

**Imitation-to-promotion ladder**:

1. **Scripted fallback**: Hand-coded skill primitives remain the authority
2. **Imitation prior shadow**: Imitation-derived proposals logged, not executed
3. **Imitation prior advisory**: Imitation proposals inform scripted selection
4. **Benchmark-gated promotion**: Imitation head takes primary authority after
   passing task success, safety, and embodiment-feasibility gates
5. **Production recurrent**: Imitation-derived head is primary with scripted
   fallback for edge cases

**Relation to existing artifacts**: This produces the action proposals that
the motor backends (`src/motor_backend/`) execute. The motor backend
interface (`train_policy`, `evaluate_policy`, `deploy_policy_handle`) is the
downstream consumer.

### 6. Drift / Calibration / Cost Evaluator

**What it is**: Continuously estimates the gap between expected and actual
embodiment behavior, and the costs incurred by that gap.

**What it evaluates**:

| Signal | Description |
|--------|-------------|
| Sim/backend mismatch | Predicted vs actual contact forces, joint responses |
| Calibration drift | Joint encoder drift, tool-tip calibration degradation |
| Contact drift | Expected vs actual contact topology changes |
| Capability degradation | Increasing torque demand, thermal throttling, wear |
| Energy cost | Wh per segment, energy efficiency relative to plan |
| Time cost | Actual vs planned execution time per skill segment |
| Risk cost | Safety margin erosion, near-miss frequency |
| Recalibration trigger | Whether current drift exceeds recalibration threshold |
| Policy demotion trigger | Whether current mismatch exceeds demotion threshold |

**Relation to existing artifacts**: This is the functional evaluator behind
`EmbodimentCostBreakdown_v1.json` (Wh/time/risk per segment),
`EmbodimentValueAttribution_v1.json` (ΔMPL/Δerror/ΔEP attribution by
segment), `EmbodimentDriftReport_v1.json` (contact/constraint/sim-backend
drift), and `CalibrationTargets_v1.json` (advisory physics knob deltas).

These existing artifacts are not random side outputs. They are the embryonic
typed outputs / sidecars / diagnostics of the broader Embodiment / Actuation
WM's drift/cost evaluation subsystem.

---

## Mapping to Existing Embodiment Artifacts

The repo already emits a family of embodiment artifacts from
`docs/embodiment_module.md`. These are not random side artifacts; they are
embryonic typed outputs of the Embodiment / Actuation WM subsystems:

| Artifact | Producing Subsystem | Role |
|----------|---------------------|------|
| `EmbodimentProfile_v1.npz` | Capability / Embodiment State Surface | Canonical body state snapshot |
| `AffordanceGraph_v1.npz` | Contact / Affordance Graph Builder | Local actionable graph |
| `SkillSegments_v1.npz` | Inverse-Dynamics / Retargeting Lane | Interaction primitive traces |
| `EmbodimentCostBreakdown_v1.json` | Drift / Calibration / Cost Evaluator | Segment-level cost vectors |
| `EmbodimentValueAttribution_v1.json` | Drift / Calibration / Cost Evaluator | Economic attribution |
| `EmbodimentDriftReport_v1.json` | Drift / Calibration / Cost Evaluator | Mismatch diagnostics |
| `CalibrationTargets_v1.json` | Drift / Calibration / Cost Evaluator | Recalibration advisories |

As the Embodiment / Actuation WM matures, these artifacts should:

1. Become typed dataclass outputs (frozen, serializable, versioned) like the
   Perception WM state objects
2. Carry explicit provenance linking them to the producing subsystem and
   promotion stage
3. Be consumable by downstream replay, training, economic valuation, and
   governance audit paths
4. Emit receipts alongside state (following the evidence-fusion-seam pattern:
   bounded seam → promotion-gated → receipt-emitting → rollback-safe)

---

## Proposed Typed Interfaces

These are doc-level contract proposals. They are not implemented yet but
describe the canonical typed surfaces this WM should eventually own.

### EmbodimentState

```
EmbodimentState:
    state_id: str
    embodiment_id: str
    embodiment_class: str           # bipedal_whole_body | stable_base_mobile_manipulator | fixed_base_tabletop
    actuator_config: ActuatorConfigState
    joint_state: JointStateVector   # positions, velocities, torques, temperatures
    contact_state: ContactStateVector
    tool_state: ToolStateDescriptor
    safety_envelope: SafetyEnvelopeState
    backend_tags: list[str]
    physics_profile_hash: str
    compute_placement: str          # on_device | companion | cloud
    battery_fraction: float
    metadata: dict
```

**Downstream consumers**: Contact/Affordance Graph Builder, Local Dynamics
Model, Action Proposal Head, Drift Evaluator, Economic WM (energy/compute
allocation).

### ContactAffordanceGraph

```
ContactAffordanceGraph:
    graph_id: str
    source_scene_graph_id: str
    source_embodiment_state_id: str
    per_object_affordance: dict[str, AffordanceAssessment]
        # graspable, insertable, placeable, fastenable, obstructed,
        # within_reach, within_force_budget, risk_level
    body_object_pairs: list[BodyObjectPair]
        # body_part_id, object_id, engagement_feasibility,
        # approach_quality, collision_risk, tool_compatibility
    fixture_states: list[FixtureContactState]
    actionable_object_count: int
    metadata: dict
```

**Downstream consumers**: Action Proposal Head, Local Dynamics Model,
Sim/Synth WM (branch evaluation conditioning), Economic WM (task-feasibility
pricing).

**Relation to ObjectiveTensor / ConstraintSet**: The ContactAffordanceGraph
does not replace ConstraintSet. ConstraintSet carries task-level hard/soft
constraints; the ContactAffordanceGraph carries embodiment-local contact
feasibility within those constraints.

### LocalDynamicsQuery / LocalDynamicsForecast

```
LocalDynamicsQuery:
    query_id: str
    current_contact_state: ContactStateVector
    candidate_action_chunk: ActionChunk
    embodiment_state_id: str
    horizon_steps: int              # 1-10

LocalDynamicsForecast:
    forecast_id: str
    query_id: str
    predicted_contact_sequence: list[ContactStateVector]
    slip_risk: float
    jam_risk: float
    insertion_success_probability: float
    force_profile: list[float]
    energy_cost_estimate: float
    recovery_feasibility: float
    promotion_stage: str            # heuristic_fallback | promoted
    confidence: float
    metadata: dict
```

**Downstream consumers**: Action Proposal Head (scoring), Drift Evaluator
(predicted vs actual comparison), Sim/Synth WM (real-vs-sim dynamics gap).

### InverseRetargetTrace

```
InverseRetargetTrace:
    trace_id: str
    source_embodiment: str
    target_embodiment: str
    source_trajectory: list[JointStateVector]
    recovered_actions: list[ActionChunk]
    retarget_quality: float
    kinematic_feasibility: float
    source_type: str                # teleop | video | demonstration | replay
    metadata: dict
```

**Downstream consumers**: Training pipeline (datapack construction), Replay
buffer, Action Proposal Head (action priors), Economic WM (data valuation).

### ActionProposalBundle

```
ActionProposalBundle:
    bundle_id: str
    proposals: list[ActionProposal]
        # action_chunk, expected_trajectory, confidence, cost_vector,
        # skill_type, promotion_stage
    selected_index: int             # -1 if no selection yet
    selection_method: str           # heuristic | learned | governance_constrained
    contact_graph_id: str
    embodiment_state_id: str
    metadata: dict
```

**Downstream consumers**: Motor backends (execution), Sim/Synth WM (rollout
scoring), Economic WM (cost/benefit evaluation), Governance nodes
(safety/constraint checking).

**Relation to ObjectiveTensor**: ActionProposalBundle carries per-proposal
cost vectors that should be consumable by ObjectiveCompiler for
scalarization under the active ObjectiveProfile. No premature scalarization
inside the Embodiment WM.

### EmbodimentDriftSummary

```
EmbodimentDriftSummary:
    summary_id: str
    sim_backend_mismatch: float
    calibration_drift: float
    contact_drift: float
    capability_degradation: float
    recalibration_recommended: bool
    policy_demotion_recommended: bool
    drift_sources: list[DriftSource]
    metadata: dict
```

**Downstream consumers**: Sim/Synth WM (sim fidelity adjustment), Economic
WM (cost attribution), Governance nodes (safety assessment), Training
pipeline (curriculum adjustment).

**Relation to existing artifacts**: This is the typed successor to
`EmbodimentDriftReport_v1.json`.

### CalibrationTargetSet

```
CalibrationTargetSet:
    target_set_id: str
    targets: list[CalibrationTarget]
        # parameter_name, current_value, recommended_value,
        # drift_magnitude, priority, source_evidence
    physics_profile_hash: str
    embodiment_id: str
    metadata: dict
```

**Relation to existing artifacts**: Typed successor to
`CalibrationTargets_v1.json`.

### EmbodimentCostVector

```
EmbodimentCostVector:
    vector_id: str
    episode_id: str
    segment_id: str
    energy_wh: float
    time_s: float
    risk_score: float
    wear_estimate: float
    compute_cost: float
    tool_change_count: int
    safety_margin_used: float
    metadata: dict
```

**Downstream consumers**: Economic WM (direct cost ingestion), EconTensor
construction, PricingSentinel (deployment cost attribution), Value Ledger.

**Relation to EconTensor**: EmbodimentCostVector is a lower-WM input to
EconTensor construction. It should not bypass the Economic WM to directly
influence reward; economics sits above lower-WM cost surfaces as the
allocative authority.

---

## Sim / Synth / Physics Transfer Boundary

The most important transfer example is kinematic remapping, but the boundary is
broader than that. This interface is where simulated branch assumptions meet
deployment embodiment truth.

### Ownership split

The Sim / Synth / Physics WM should own:

- embodiment-facing simulation assumptions and the backend/fidelity regime used
  to produce a branch
- morphology/backend mismatch state on the simulation side
- transfer/calibration receipts and rollout-conditioned adaptation candidates
- slow-loop deployment metadata, transfer summaries, and promotion evidence

The Embodiment / Actuation WM should own:

- kinematic remapping
- retargeting
- capability filtering
- realized post-transfer mismatch
- deployment-side action-feasibility degradation
- control-rate / latency / actuator response divergence
- deployment-side drift handling
- local recovery / degradation posture
- local embodiment adaptation

This keeps remap and deployment adaptation body-local while letting Sim /
Synth / Physics remain the owner of the simulation-side assumptions and
evidence.

UE-backed simulation assumptions should enter this boundary the same way any
other sim-side provider family does:

- as scene / render / sensor / timing assumptions emitted by Sim / Synth /
  Physics
- as transfer-risk and calibration evidence
- not as ownership over body truth or controller truth

### Broader remap classes

| Remap / transfer class | Primary owner | Role |
|---|---|---|
| Morphology remap | Inverse-Dynamics / Retargeting Lane + Capability State Surface | Map source-body trajectories and policies into target-body feasible traces |
| Actuator / control-space remap | Inverse-Dynamics / Retargeting Lane + Joint Skill / Action Proposal Head | Convert policy/action representations into embodiment-native control chunks |
| Sensor / render / domain remap | Sim / Synth / Physics WM primary | Translate scene/render/domain assumptions into transfer-risk and branch-conditioning evidence |
| Timing / latency / control-rate remap | Embodiment WM fast/runtime side with Sim assumptions as inputs | Reconcile simulated control cadence, controller latency, and actuator response with deployment execution realities |
| Contact / friction / dynamics calibration remap | Shared: Sim proposes, Embodiment validates | Push system-ID/calibration candidates from sim and compare them against realized traces |
| Environment / scene abstraction remap | Sim / Synth / Physics WM to Embodiment contact/affordance consumers | Convert scene/layout abstractions into body-local actionable structure |
| Capability-envelope remap | Capability State Surface + Drift / Calibration / Cost Evaluator | Narrow simulated feasibility claims to the currently valid deployment envelope |

### Bounded learned seams

Sim / Synth / Physics-side learned seams should stay bounded and advisory-to-
promotable:

- backend mismatch estimator
- transfer-success predictor
- morphology-conditioned rollout scorer
- surrogate-physics / inverse-design scorer
- calibration parameter proposer

Embodiment-side learned seams should stay body-local and execution-facing:

- retargeting / remap seam
- action-space remap or inverse-dynamics seam
- local capability adaptation seam
- calibration / drift evaluator
- deployment degradation predictor

None of these seams should become a replacement ontology or a cross-WM
mother-latent.

### Timescales and bridge surfaces

The boundary should operate across the same fast / mid / slow loop hierarchy as
the rest of the Embodiment WM:

- **Fast exchange**: compact execution constraints, active capability envelope,
  safety posture, and latency class from Embodiment; compact adaptation
  candidates from Sim. These must not block the inner control loop.
- **Mid-loop exchange**: `SimulationOutcomeReceipt`,
  `PhysicsAdaptationReceipt`, `SimRealGapReceipt`, and
  `BackendMismatchReceipt` from Sim; `EmbodimentDriftSummary`,
  `CalibrationTargetSet`, `DeploymentTransferDriftReceipt`,
  `ActionFeasibilityDegradationReceipt`, `ControllerLatencyMismatchReceipt`,
  and realized mismatch traces from Embodiment.
- **Slow-loop exchange**: stable remap tables, transfer summaries, backend
  quality trends, promotion evidence, deployment-side drift baselines, and
  `EmbodimentTransferOutcomeReceipt` summaries.

These are the surfaces the future WM-transport layer should later consume as
typed bridge objects. Transport should not become the first owner of remapping,
drift handling, or transfer truth.

### UE-backed simulation inputs

When UE5 / Unreal is the upstream provider family, the Embodiment side should
be able to consume:

- UE-backed sensor and synchronization profiles
- simulated latency and control-rate assumptions
- visual-domain and digital-twin transfer summaries
- middleware-connected runtime assumptions when they materially affect
  deployment timing or degraded-mode posture

Those remain inputs into Embodiment-local adaptation. Unreal does not become
the owner of body truth, action feasibility, or control truth.

### Candidate typed transfer receipts

The embodiment side of this boundary should later emit explicit deployment-side
transfer receipts such as:

- `DeploymentTransferDriftReceipt` — realized post-transfer mismatch, local
  drift sources, and comparison against the sim-side assumptions
- `ActionFeasibilityDegradationReceipt` — degradation in reachable, stable, or
  safe action space after deployment
- `EmbodimentTransferOutcomeReceipt` — bounded summary of transfer outcome,
  recovery posture, and whether local adaptation preserved useful prior
- `ControllerLatencyMismatchReceipt` — observed control-rate / latency /
  actuator-response divergence relative to the simulated cadence
- `SensorTimingMismatchReceipt` — divergence between simulated sensor timing /
  synchronization assumptions and deployment reality
- `SimulationToEmbodimentTransferReceipt` — compact receipt linking the
  sim-side transfer assumptions actually consumed by Embodiment to the
  deployment-side adaptation posture
- `EmbodimentLatencyDivergenceReceipt` — embodiment-local timing and control
  divergence summary when the deployed loop no longer matches the simulated
  cadence or middleware assumptions

These receipts should feed replay/training export, Sim / Synth / Physics
calibration feedback, and later Economic WM consumption. They should not make
the Economic WM the first owner of transfer mechanics.

---

## What We Borrow from External Architectures

Imported architectures enter this stack as bounded, promotable, typed,
receipt-emitting subsystem seams inside the Embodiment / Actuation WM. They
do not replace the multi-WM topology.

### V-JEPA 2 / I-JEPA

**What we borrow**: Latent predictive modeling, representation-first
prediction, no obligation to decode pixels at test time, semantic predictive
priors for contact-state forecasting.

**Where it enters**: Local Contact Dynamics Model (latent short-horizon
prediction of contact state evolution). Also dual-homed in Perception WM
for temporal grounding.

**Promotion posture**: Provider-backed, behind typed contract, benchmark-gated.

### LeRobot / ACT

**What we borrow**: Practical action chunking interfaces, standardized
policy/data contracts, training/eval ergonomics for robot policies, inverse
dynamics for action recovery from demonstrations, scalable imitation-learning
dataflow patterns.

**Where it enters**: Inverse-Dynamics / Retargeting Lane (action recovery,
demonstration ingestion, cross-embodiment retargeting), Joint Skill / Action
Proposal Head (action chunk format, imitation-derived priors).

**Promotion posture**: Pattern adoption for interfaces and data format. Neural
seams from ACT-style models behind promotion gates. See "Scalable
Imitation-Learning Pipelines" section for the full promotion ladder.

### UMI / Retargeting Patterns

**What we borrow**: Universal manipulation interface patterns for
cross-embodiment demonstration collection and retargeting, hand-object
trajectory transfer, embodiment-agnostic demonstration formats.

**Where it enters**: Inverse-Dynamics / Retargeting Lane (cross-embodiment
retargeting, kinematic feasibility filtering, embodiment-agnostic trace
formats).

**Promotion posture**: Provider-backed retargeting with typed contracts. Quality
scoring determines which retargeted traces enter the training pipeline.

### Diffusion Policy

**What we borrow**: Action-chunk proposal structure, multimodal short-horizon
control priors, conditional generation for action distribution modeling.

**Where it enters**: Joint Skill / Action Proposal Head (multimodal proposal
generation when multiple approaches are feasible).

**Promotion posture**: Neural seam behind `disabled|auto|required` with
benchmark gating on task success rate.

### Isaac Lab

**What we borrow**: Scalable embodiment-aware task environments, contact-rich
simulation surfaces, sensor/backend/task abstraction patterns, articulated
agent configuration discipline.

**Where it enters**: Capability / Embodiment State Surface (embodiment config
patterns), Contact / Affordance Graph Builder (contact-rich sim surfaces),
motor backend integration (`IsaacBackend`).

**What we explicitly do NOT import**: Isaac Lab's environment abstraction does
not become the master environment ontology. Our WM boundaries remain
separate.

### GR00T / VIRAL / DoorMan

**What we borrow**: Concrete sim-to-real training plant discipline for
humanoid loco-manipulation: privileged teacher to deployable student
training, typed experiment/config surfaces, camera observation and delay
profiles, domain randomization, dataset-reset curricula, eval/checkpoint/export
gates, and G1-facing robot/action-space configuration hygiene.

**Where it enters**:

- Capability / Embodiment State Surface: G1-style joint/action-space,
  primitive-action, sensor, camera, and deployment-profile discipline.
- Contact / Affordance Graph Builder: sim-trained contact/affordance teacher
  traces as evidence for reachability, grasp/hold feasibility, obstruction,
  slip, and stage transition labels.
- Local Contact Dynamics Model: privileged teacher rollouts, randomized
  physics/contact settings, and delayed/degraded observation tests as bounded
  training/eval slices.
- Inverse-Dynamics / Retargeting Lane: reset-from-dataset and
  demonstration-curriculum profiles for teleop, sim, video, and later real
  robot traces.
- Joint Skill / Action Proposal Head: deployable student/action-proposal
  heads distilled from teacher traces, gated by deployment observation
  profiles and benchmark evidence.
- Drift / Calibration / Cost Evaluator: sim-real gap, backend mismatch,
  observation-delay sensitivity, randomization provenance, export artifact
  refs, and runtime/economic cost receipts.

**Promotion posture**: GR00T-style teacher/student lanes are future Phase 1.x
and Phase 3 surfaces. They should be typed, receipt-emitting,
benchmark-gated, and subordinate to the six Embodiment subsystems. Export
artifacts such as ONNX are deployment candidates only after eval/export gates
pass.

**What we explicitly do NOT import**: GR00T task primitives, PPO, DAgger,
ResNet, ONNX, or Isaac Lab as the primitive ontology of this WM. They are
examples of training/config/export discipline, not replacements for
CapabilityState, ContactAffordanceGraph, LocalDynamicsForecast,
RetargetingTraceBundle, ActionProposalBundle, or EmbodimentDriftSummary.

### TD-MPC2

**What we borrow**: Bounded short-horizon latent dynamics + planning for
continuous control, model-predictive control within a latent space, world
model as local planning substrate.

**Where it enters**: Local Contact Dynamics Model (bounded short-horizon
planning over contact state). This is a subsystem seam, not the whole stack.

**What we explicitly do NOT import**: TD-MPC2's implicit assumption that the
world model IS the planner. In our topology, the Embodiment WM contains a
local dynamics model that aids planning; it does not collapse dynamics,
planning, control, economics, and governance into one model.

### What We Explicitly Do NOT Import

From any external architecture:

- Any ontology that collapses economics/governance/perception/actuation into
  one uninterpretable model
- Any assumption that video generation or full pixel decoding should sit on
  the real-time actuation critical path
- Any replacement of typed contracts with opaque hidden state
- Any architecture that sidelines constraint/governance/economic legibility
- Any "world model" framing that equates the entire stack with one predictive
  network

---

## Scalable Imitation-Learning Pipelines

Imitation learning should not be treated as an ambient training tactic outside
the WM. It belongs explicitly in the Embodiment / Actuation WM as a real
subsystem concern.

### Ownership Placement

The primary home for imitation learning is:

1. **Inverse-Dynamics / Retargeting Lane** (Subsystem 4): where demonstrations,
   teleop traces, video-derived traces, and action-recovery traces are
   normalized into embodiment-native training material
2. **Joint Skill / Action Proposal Head** (Subsystem 5): where imitation-derived
   priors become actionable chunk proposals for real control

The inverse/retargeting lane is where demonstrations are ingested, retargeted
across embodiments, and packaged into replay-ready traces. The action proposal
head is where those imitation-derived priors become actionable for control.

### Dataflow Through the WM

```
Demonstrations (teleop, video, sim replay, real robot)
    ↓ typed provider contracts
Inverse-Dynamics / Retargeting Lane
    ↓ DemonstrationIngestReceipt, RetargetingTraceBundle, ActionRecoveryReceipt
    ↓ DatapackQualityReceipt → Economic WM (data valuation)
Joint Skill / Action Proposal Head
    ↓ ImitationPriorSnapshot, ActionProposalBundle
Motor Backends (execution)
    ↓ execution traces
Drift / Calibration / Cost Evaluator
    ↓ demonstration-to-deployment gap estimation
Training Pipeline (corpus construction)
```

### Typed Artifacts and Receipts

| Artifact | Emitting Subsystem | Downstream Consumers |
|----------|-------------------|---------------------|
| `DemonstrationIngestReceipt` | Inverse-Dynamics Lane | Training pipeline, Economic WM |
| `RetargetingTraceBundle` | Inverse-Dynamics Lane | Action Proposal Head, Replay buffer |
| `ActionRecoveryReceipt` | Inverse-Dynamics Lane | Training pipeline |
| `DatapackQualityReceipt` | Inverse-Dynamics Lane | Economic WM (data valuation) |
| `ImitationPriorSnapshot` | Action Proposal Head | Training pipeline, promotion gates |
| `ImitationDriftReceipt` | Drift Evaluator | Policy demotion, recalibration |

### Model Family Candidates by Function

| Function | Model Family | Justification |
|----------|--------------|---------------|
| Action recovery from state transitions | Inverse-dynamics heads (MLP/Transformer) | Direct state-delta to action mapping |
| Cross-embodiment retargeting | Retargeting networks with embodiment embeddings | Generalize across kinematic chains |
| Chunked policy/skill prior | ACT-style action chunking transformers | Multi-step action chunk prediction |
| Multimodal action proposal | Diffusion policy heads | Multiple valid action modes in contact-rich tasks |
| Video-to-action extraction | LeRobot-style interfaces | Standardized policy/data contracts |
| Demonstration quality scoring | Learned quality predictors | Filter low-quality demonstrations |

These enter as bounded subsystem seams, not as the Embodiment WM ontology. They
must remain typed, receipt-emitting, benchmark-gated, and subordinate to the
WM's canonical body/contact/capability/control state.

### Hyperparameter Governance by the WM

Imitation-learning hyperparameters should **not** be treated as globally
free-floating. They should be shaped by the Embodiment / Actuation WM's own
burdens and constraints.

**Shaping constraints**:

| Hyperparameter | Shaping Constraint |
|----------------|-------------------|
| Action-chunk length / horizon | Embodiment DoF, contact richness, task family |
| Proposal multiplicity | Contact ambiguity, grasp multiplicity, safety envelope |
| Retargeting tolerance / alignment thresholds | Source-target kinematic similarity, safety margins |
| Inverse-dynamics model capacity | State dimensionality, action space complexity |
| Imitation-derived proposal head capacity | Embodiment complexity, skill catalog size |
| Promotion thresholds (scripted → learned) | Benchmark gates, safety compliance, downstream success |
| Dataset quality thresholds | Deployment realism, embodiment match quality |

**What the WM governs**:

- Chunk length and action horizon selection
- Proposal multiplicity limits
- Retargeting tolerance and alignment thresholds
- Capacity of inverse/retargeting models
- Capacity of imitation-derived proposal heads
- Promotion thresholds from scripted/action-prior fallback into learned chunk
  proposers
- Dataset quality filtering thresholds

**What stays shaped by adjacent WMs**:

- Economic WM: data valuation weights, corpus prioritization, training budget
- Perception WM: video/visual feature quality requirements
- Sim/Synth WM: sim-to-real gap thresholds, synthetic demonstration validity

### Promotion Ladder for Imitation-Derived Heads

1. **Scripted fallback**: Hand-coded skill primitives remain the authority.
   Imitation models may exist but have no runtime influence.

2. **Imitation prior shadow**: Imitation-derived proposals are logged but not
   executed. Used for offline comparison and benchmark development.

3. **Imitation prior advisory**: Imitation proposals inform selection among
   scripted options. Scripted primitives still execute; imitation priors
   influence ranking/selection.

4. **Benchmark-gated promotion**: Imitation head takes primary authority after
   passing:
   - Task success rate gates
   - Safety compliance gates
   - Embodiment-feasibility gates
   - Sim-to-real transfer quality gates

5. **Production recurrent**: Imitation-derived head is primary with scripted
   fallback for edge cases and safety-critical recovery.

### What Imitation Learning Does NOT Replace

Imitation-learning seams do not replace:

- The WM's canonical body/contact/capability/control state ownership
- The typed interface surfaces (EmbodimentState, ContactAffordanceGraph, etc.)
- The Local Contact Dynamics Model (short-horizon prediction remains separate
  from imitation priors)
- The Drift / Calibration / Cost Evaluator (imitation heads are evaluated by
  this subsystem, not replacements for it)
- Economic attribution and governance audit paths
- Safety envelope enforcement at the fast loop level

Imitation learning enters as a **training methodology and prior source** for
the action proposal head. It does not become the Embodiment WM ontology.

---

## Hierarchy and Timescales

The Embodiment / Actuation WM operates across three nested control
timescales. Imported predictive models fit into this hierarchy without
displacing it.

### Fast Inner Loop (1-10ms, per-step)

- Proprio / joint state update
- Contact detection and force feedback
- Low-level torque/position control
- Safety limit enforcement
- Local impedance/compliance regulation

**What runs here**: Actuator backends (Isaac, PyBullet, real hardware),
safety envelope monitoring. No learned seam operates at this timescale
initially — it is physics and control engineering.

### Mid-Level Loop (10-200ms, per-chunk)

- Action chunk execution and monitoring
- Local contact dynamics prediction (short-horizon forecasting)
- Skill primitive execution (PICK, PLACE, INSERT, FASTEN)
- Contact/affordance graph update
- Drift detection within execution

**What runs here**: The Local Contact Dynamics Model, the Action Proposal
Head, and real-time contact/affordance graph updates. This is where learned
seams like diffusion policy chunks or TD-MPC2-style planning operate.

### Slow Supervisory Loop (200ms-seconds, per-task-segment)

- Skill selection and sequencing
- Simulation branch evaluation
- Economic cost/benefit assessment
- Calibration drift evaluation
- Policy promotion/demotion decisions
- Replay/training export decisions

**What runs here**: The Drift/Calibration/Cost Evaluator, Economic WM
integration, governance audit, and training pipeline feedback. This loop
consumes the results of the faster loops and shapes the overall strategy.

**Key principle**: Faster loops do not wait for slower loops. Slower loops
shape the parameters (which skill, which action prior, which calibration) that
faster loops execute. This is the same multi-timescale discipline as the
Economic WM's fast/meso/slow variable split.

---

## Why This Matters for Real Robot Readiness

The Embodiment / Actuation WM is the substrate that gets the stack closer to
controlling an actual robot body. Without it, the stack can compile rich
semantic scene state and evaluate simulation branches, but cannot translate
those into body-aware, constraint-respecting, contact-aware actuator commands.

### Concrete readiness targets

| Capability | WM Subsystem | Workcell Task |
|------------|-------------|---------------|
| Grasp planning under contact uncertainty | Contact/Affordance Graph + Local Dynamics | Bin picking (TASK-L002) |
| Tight-tolerance insertion | Local Dynamics + Action Proposal | Peg-in-hole (TASK-A001) |
| Fastener installation with torque control | Action Proposal + Drift Evaluator | Fastener installation (TASK-A002) |
| Kitting with multi-object sequencing | Contact/Affordance Graph + Action Proposal | Kitting (TASK-L001) |
| Tool change and recalibration | Capability State + Drift Evaluator | Tool change (TASK-M001) |
| Sim-to-real transfer monitoring | Drift Evaluator | All tasks |
| Energy-aware skill selection | Cost Evaluator + Economic WM | All tasks (G1 battery-constrained) |
| Backend mismatch detection | Drift Evaluator + Sim/Synth WM | All tasks |

### G1 / Unitree / humanoid integration path

The Embodiment / Actuation WM is designed around bipedal whole-body Unitree
G1/R1-class control as the primary standard. It must accommodate:

- 29 DoF bimanual humanoid (G1)
- floating-base whole-body state, support/contact phase, and balance context
- locomotion plus manipulation and bimanual/dexterous coordination
- stable-base mobile-manipulator fallback for conservative recovery, operator
  handoff, and degraded-mode task continuity
- egocentric camera feeds (consumed via Perception WM embodiment bridge)
- on-device vs companion compute placement decisions
- battery-constrained operation (battery_fraction as first-class state)
- real-time safety envelope enforcement

The WM does not require the G1 hardware to be present today. The typed
interfaces are designed so that:

1. Current tabletop-arm and workcell sim backends exercise the same WM
   subsystems only as `fixed_base_tabletop` curriculum/regression lanes
2. Stable-base mobile-manipulator surfaces can be used as explicit safety
   fallback/degraded-mode lanes without satisfying bipedal promotion gates
3. G1-specific embodiment profiles and safety envelopes can be added as
   configuration, not architecture changes
4. Calibration drift and backend mismatch detection generalize across
   embodiments

Neural scaffolding should therefore target:

- whole-body state encoders over floating base, limbs, hands, IMU, proprioception,
  contact, support phase, and body-relative scene state
- contact/support/balance predictors with fall/slip risk and support-margin
  receipts
- loco-manipulation action proposal heads that couple gait, posture, arms, hands,
  and task constraints
- inverse-dynamics / retargeting lanes that produce Unitree-native action/state
  rows from teleop, sim, video-derived, or demonstration traces
- learned fallback selectors that emit veto/defer/recovery/operator-handoff
  receipts rather than silently redefining the primary bipedal target
- latency/watchdog/resource predictors tied to Phase 4A and 4E control and
  communication contracts

---

## Phase Sequencing

This document is Phase 2-compatible doctrine sharpening. It clarifies the
doctrinal target for the Embodiment / Actuation WM so that:

1. Current Phase 2 Perception WM work can validate that its outputs
   (especially the embodiment bridge) are shaped correctly for later
   Embodiment WM consumption
2. The transition from Phase 2 to Phase 3 has a concrete, legible target
   rather than a vague "embodiment work"
3. Any narrow Phase 2 shadow-boundary validation (e.g., proving that
   Perception outputs can translate into typed affordance/contact bundles)
   has explicit doctrinal grounding

This does **not** mean we are executing Phase 3 now. The current
implementation center remains Phase 2 Perception / Grounding WM. This
document is spec-first, not implement-first.

2026-05-19 update: Phase 3 spec prep is now split out into
`docs/economic_world_model/phase3_embodiment_actuation_spec_prep.md`. That note
defines the proposed first canonical state contracts, receipt family, shadow
compiler posture, and acceptance gates to use once the Phase 1.x closure
assessment is accepted.

2026-05-20 update: the first implementation pass is live in
`src/world_model/embodiment_actuation/`, covering Phase 3.1 through 3.3:
canonical state contracts, receipts, provider/runtime contracts, promotion
posture, a shadow compiler, and first shadow downstream consumers. It remains
advisory: no runtime authority, no provider/GPU claims, and no GR00T/Isaac
ontology import.

2026-05-20 follow-up: Phase 3.4 scaffolding is now live. G1 morphology and
OSS/local evidence are typed in `morphology.py`; bounded neural seams are in
`neural_seams.py`; seam training rows and non-promotional manifests are in
`training_corpus.py`; and `scripts/smoke_test_embodiment_phase34.py` proves
CPU-local forward passes. This is learned-path readiness, not a GPU training or
policy-promotion claim.

When the implementation priority shifts to the Embodiment / Actuation WM,
the first tranches should follow the repo's established pattern:

1. Typed state contracts (frozen dataclasses)
2. Shadow compiler that produces canonical state from existing inputs
3. First downstream consumers wired
4. Receipt emission from compilation
5. First bounded neural seams behind promotion posture
6. Provider contracts for external backends

---

## Cross-WM Resource Surfaces

Following the pattern established in Phase 2 Perception, the Embodiment /
Actuation WM should independently carry its version of the typed lower-WM
resource surfaces:

- **Provider surface**: which motor backends are available, their truth class
  (real hardware, sim, synthetic stub), latency characteristics
- **Compute surface**: on-device vs companion placement, inference headroom
  for learned control seams
- **Energy surface**: battery fraction, energy cost per skill segment,
  thermal headroom
- **Safety surface**: current safety envelope utilization, margin remaining,
  near-miss count

These are lower-WM owned surfaces. The Economic WM allocates across them but
does not originate them.

---

## Related Doctrine: Bio/Neuro Inspirations

Organizational principles that land primarily on this WM are specified in
`docs/economic_world_model/doctrine_bio_neuro_architecture_inspirations.md`.
That note owns candidates, typed surfaces, and boundaries—do not restate it
here.

## Autoencoder / Codebook Use Inside Embodiment

Autoencoder-family modules (VAE, VQ-VAE, contractive/denoising bottlenecks)
may appear as **bounded seams** inside this WM—for example posture/contact
or action-chunk **manifold compression**, **skill/synergy codebooks**,
**interoceptive** bottlenecks, or compact latents for **retargeting** traces.

They **support** the existing plan: inverse-dynamics heads, retargeting
networks, ACT-style chunking, diffusion proposal heads, task-conditioned
policy networks. They **do not** replace those lanes or the six-subsystem
decomposition. Same rule as other imports: promotion-gated, receipt-emitting,
**subsystem** scope—not a replacement ontology. Bridge-layer classification
and stack-wide posture: `docs/economic_world_model/neuralization_bridge_doctrine.md`
§ Autoencoder / Codebook Posture Across the Stack.

## Anti-Patterns

Do not:

- Collapse the Embodiment WM into one end-to-end learned controller with no
  typed internal structure
- Let external policy architectures replace the typed contract surfaces
- Push economic reward directly into low-level motor control (economics
  allocates resources and shapes objectives; it does not output torques)
- Treat the Embodiment WM as "just motor backends" — it owns body state,
  contact truth, affordance assessment, dynamics prediction, and cost
  attribution, not just command execution
- Let the Contact/Affordance Graph become a static data structure — it must
  update within the mid-level loop as contact state changes
- Skip calibration/drift evaluation — this is what makes sim-to-real
  transfer honest rather than assumed
- Import external architectures as top-level ontology — they enter as
  bounded seams, not as architectural replacement
