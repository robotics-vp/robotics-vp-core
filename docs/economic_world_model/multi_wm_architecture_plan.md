# Multi-WM Architecture Plan

## Purpose

This document defines the next architectural expansion beyond the current semantic-world-model and economic-world-model readiness work.

It is intentionally a plan document, not an implementation pass.

The central question is:

- which world models should exist as canonical state owners
- in what order they should be built
- how they should communicate without collapsing into one giant latent blob
- what should be built next now

## Executive Conclusion

Yes, the proposed topology makes architectural sense, with one important constraint:

- each lower world model must own a canonical, typed, replayable state surface
- the economic world model should sit above those canonical state surfaces
- the meta-node superposition/control world model should sit above the economic world model
- the cross-WM transport layer should be middleware between adjacent world models, not a premature mother-latent that dissolves boundaries

The recommended canonical stack is:

1. Perception / grounding WM
2. Embodiment / actuation WM
3. Sim / synth / physics WM
4. Economic WM over those lower WMs
5. Meta-node superposition / control WM above the economic WM

The recommended next WM to build is:

- **Sim / synth / physics WM**

That is the highest-leverage next step because the current production flywheel gap is less about semantic representation itself and more about what the stack actually decides to simulate, diffuse, synthesize, admit, and feed back into training.

## Cross-Phase Neuralization Rule

Every new WM or future enabling phase should launch with bounded learned seams from its first implementation tranche.

That means:

- heuristics are allowed only as explicit priors and fallback logic
- those priors must sit behind typed `disabled|auto|required` helper/runtime contracts
- helper status, promotion stage, and decision traces must be emitted as canonical receipts
- epiplexity-based inferential learnability should be carried as canonical typed metadata once a WM starts affecting replay, admission, simulation, diffusion, or training selection
- no new WM should be allowed to land as a heuristic-only island that will later require a cleanup-style “heuristic purge”

This rule applies to:

- sim / synth / physics WM
- perception / grounding WM
- embodiment / actuation WM
- economic WM consolidation
- cross-WM transport bridges
- local meta-node neuralization
- the later meta-node superposition / control WM
- future deployment-enabler phases anywhere learned routing, calibration, or policy support appears

This matters especially for the downstream WMs. Perception, embodiment, and sim/synth/physics should not only emit state; they should also emit bounded inferential learnability classes about which scenes, branches, physics regimes, or control contexts are actually improving the stack under compute and governance constraints. The economic WM should eventually consume those typed inferential receipts directly rather than reconstructing learnability from scattered sidecars later.

## Complete Subsystem Rule

Each WM should be built as a complete subsystem target, not as a partial architecture exercise.

For this repo, "complete subsystem" means:

- canonical typed state contracts exist
- bounded learned seams exist anywhere the subsystem needs learned routing, calibration, scoring, or control support
- those seams have real runtime-package contracts and `disabled|auto|required` loading posture
- the live production loops already call through the subsystem boundary rather than bypassing it with parallel helper logic
- receipts, replay exports, benchmark gates, and training-manifest artifacts already exist so later training and promotion do not require bespoke joins
- the remaining blockers are stated honestly as:
  - dataset density
  - GPU budget / training time
  - grounded hardware or sim assets
  - calibration truth
  - benchmark evidence
  - external provider maturity

The target posture is:

- damn-near production-ready in subsystem structure and runtime wiring
- explicitly not pretending the subsystem is production-ready when the missing pieces are still real data, GPUs, calibration, or Unitree-class hardware/sim prerequisites

For the G1/R1-facing roadmap, each subsystem should be pushed until the main remaining bottlenecks are honest Unitree-class readiness inputs rather than missing neural scaffolds, missing runtime loops, or missing package contracts.

## Phase Exit Rule

Do not move to the next named phase just because the next phase is conceptually attractive.

The correct sequencing rule is:

- keep executing the current phase while any named explicit gap is still addressable by:
  - owned runtime wiring
  - typed contracts or receipts
  - helper/runtime-package integration
  - adapter implementation
  - canonical-state consolidation
  - replay/training/reporting integration
- only advance when the honest main blockers for the current phase are primarily:
  - data or corpus density
  - GPU budget or training time
  - unavailable external assets or providers
  - calibration truth
  - benchmark evidence
  - target-hardware or target-sim access

In other words:

- do not leave an implementable ownership/adaptation/runtime gap behind just because a later WM is more interesting
- do move on once the current phase is structurally real and the main remainder is genuinely externalized into data/GPU/asset/benchmark constraints

This rule applies to Phase 1 and every later WM or deployment-enabler phase in this document.

## Architectural Position

### What already exists

The repo already has important canonical-state substrate pieces:

- `RuntimePacket` in `src/runtime/packets.py`
- `EvidenceBus` and `BeliefState` in `src/evidence/bus.py` and `src/evidence/belief_state.py`
- `SemanticWorldModelState` in `src/world_model/semantic_world_model.py`
- semantic selection, orchestration, meta-transformer, queue, sampler, and coverage helper lanes that now emit honest runtime receipts

That means the repo is no longer starting from zero. The missing step is to extend this pattern to additional WMs rather than leaving sim/physics/vision/actuation functionality scattered across scripts, stubs, and helper surfaces.

### What should not happen

Do not:

- build a giant shared latent that replaces typed contracts
- make external OSS models native truth owners
- deeply neuralize the economic WM before lower WMs emit their own canonical state
- treat the meta-node WM as the next immediate task

### What should happen

Do:

- keep frozen OSS models as typed state providers where they are already strong
- make each WM cybernetic and neuralized around canonical state, receipts, and governance
- let the economic WM become the allocator/governor over lower WMs once those lower surfaces are real
- add the transport layer only after at least two adjacent WMs emit stable canonical surfaces

### Advisory posture inside the WM topology

The multi-WM topology only stays honest if internal WM-to-WM communication is not treated as another advisory blob layer.

The right doctrine is:

- external OSS providers can remain advisory or pluggable
- preview/report tools can remain advisory
- internal typed state, readiness classes, helper traces, and selection receipts should graduate into canonical metadata, preconditions, work orders, or bounded authority as they begin to affect runtime or training

This matters because later:

- the sim / synth / physics WM
- the perception / grounding WM
- the embodiment / actuation WM
- the economic WM
- the transport bridges between them

must communicate through typed state and receipts rather than culturally "optional" advisories.

## Two Ontology Layers

Do not collapse ontology into one vague layer.

The architecture needs two distinct ontology roles:

1. operational / module-level ontology inside the stack
2. WM-transport ontology for adjacent-WM interoperability

Neither of these should become a giant symbolic mother-WM.

### 1. Operational / module-level ontology

This is the existing in-stack ontology substrate:

- canonical typed operational state for entities, tasks, datapacks, events, provenance, governance hooks, and module/runtime state
- the cybernetic operational digital-twin layer that modules actually read from and write to
- not just static persistence or a passive registry

This layer should become more neuralized over time, but remain operationally grounded:

- learned embeddings over object, action, event, and datapack types
- uncertainty-aware assertions rather than flat boolean claims only
- richer temporal and event structure over operational trajectories
- trainable module-to-ontology and ontology-to-module adaptors where that improves fidelity

Its RL / training role should be:

- improving module-to-ontology and ontology-to-module encoding / decoding fidelity
- improving event and state prediction plus temporal consistency
- improving uncertainty calibration and provenance quality
- improving governance satisfaction and policy-compliance quality
- using completed-loop postmortems, reconstruction quality, calibration quality, and operational yield as the main training signals

Important boundary:

- this is not permission to rewrite the repo’s frozen core reward math directly right now
- reward for this layer should come from completed-loop quality, reconstruction, calibration, provenance, and operational yield, not from ontology taking over the core reward path

### 2. WM-transport ontology

This is distinct from the operational ontology.

It is the typed semantic and governance contract for cross-WM interoperability between adjacent world models.

Its job is to define:

- which WM-to-WM mappings are semantically valid
- which uncertainty, provenance, and governance fields must survive translation
- which translated outputs remain actionable for the downstream WM

This layer should work with, not replace, the differentiable bridge:

- ontology = typed semantic and governance contract
- isomorphic tensor / transport bridge = compiled differentiable realization that respects that contract

Do not replace the transport tensor with ontology text or symbolic graph rewriting.
Do not replace the ontology contract with an unconstrained tensor either.
They are complementary layers.

Its RL / training role should be:

- improving WM-to-ontology-to-WM translation quality
- preserving topology, causal structure, dependency structure, and downstream actionability across WMs
- increasing successful synchronization across full loops
- decomposing gains into bridge-only, downstream-WM-only, joint, and interaction effects
- using completed-loop and postmortem reward, counterfactual improvement, governance satisfaction, and downstream economic yield as training signals for the adaptor / bridge layer

### Current honest status

Today the repo mostly has:

- operational ontology substrate and plumbing
- event/provenance/governance hooks
- module-facing typed state scaffolding

Today the repo does **not** yet have:

- a fully neuralized operational ontology layer
- a full WM-transport ontology implementation
- a full ontology-mediated adjacent-WM transport runtime

The sequencing remains:

1. lower WMs first
2. economic WM consolidation over lower canonical state
3. ontology-mediated WM transport between adjacent WMs

That sequencing should not be inverted by introducing a premature ontology mother-layer.

## Topology

```mermaid
flowchart TD
    A["Raw Sensors / Sim Frames / Rollout Artifacts"] --> B["Perception / Grounding WM"]
    A --> C["Embodiment / Actuation WM"]
    A --> D["Sim / Synth / Physics WM"]
    B --> E["Economic WM"]
    C --> E
    D --> E
    E --> F["Meta-Node Superposition / Control WM"]
    B <-->|"typed transport bridge"| C
    C <-->|"typed transport bridge"| D
    D <-->|"typed transport bridge"| E
    E <-->|"typed transport bridge"| F
```

Interpretation:

- perception owns canonical scene and grounding state
- embodiment owns canonical body, capability, and control state
- sim/synth/physics owns canonical synthetic branch, physics fidelity, and simulation agenda state
- economic owns pricing, allocation, queueing, value, and governance over the lower states
- meta-node superposition owns cross-WM policy over objectives, Pareto tradeoffs, and governance meta-choice

## Sequencing Rule

The right sequencing is:

1. finish enough lower-WM canonical state so the economic WM has real things to consume
2. consolidate the economic WM over those lower-WM surfaces
3. add the cross-WM transport bridges
4. only then build the meta-node superposition/control WM

So the answer to “should the economic WM be fully stood up first, or should downstream WMs be stood up first?” is:

- the economic WM should continue to be hardened now
- but it should **not** be considered fully neuralized or complete before at least the sim/synth/physics WM and one of perception or embodiment emit canonical state

## Humanoid Target Implications

The target hardware matters architecturally.

If the intended long-term target is Unitree G1/R1-class readiness, then several current repo assumptions must be treated as provisional rather than production-shaped:

- current environments are mostly fixed-base and tabletop-centric
- current embodiment assumptions are still arm/gripper-oriented
- current policy and adapter widths were not chosen under a 21+ DoF humanoid requirement
- current safety and observation assumptions are not yet real-time, proprioceptive, or whole-body enough

### Model-capacity implication

Not every model in the stack needs to become large.

The right rule is:

- lower WMs that must represent high-dimensional body state, contact, locomotion, dexterous manipulation, and spatial perception should be allowed to scale materially
- the economic WM can remain relatively compact if it consumes rich canonical lower-WM state instead of trying to internalize all embodiment/perception complexity itself
- the later meta-node WM should also stay governance-sized rather than trying to become the place where raw whole-body complexity lives

That means the stack needs an explicit future **model-capacity audit**, not just more modules.

The concrete follow-on checklist for this is now captured in:

- `docs/economic_world_model/humanoid_target_readiness.md`

Subsystems that likely need capacity reconsideration for G1/R1-class readiness:

- embodiment / actuation WM encoders and action-state models
- low-level and mid-level policy heads that currently assume small action spaces
- sim / synth / physics WM components that must model contact-rich whole-body behavior
- perception / grounding WM components that must fuse egocentric vision, depth, proprioception, and spatial state
- transport bridges that must preserve richer topology across body, scene, and physics state

Subsystems that do **not** necessarily need to become large if the architecture is correct:

- economic WM policy/value layers
- governance and meta-choice helper layers
- some orchestration/control-plane helpers that operate over typed summaries rather than raw robot state

### Environment implication

The current environment portfolio should not be treated as sufficient for humanoid readiness.

Current envs such as:

- `workcell`
- `dishwashing`
- `drawer_vase`

are still useful, but they are best understood as:

- manipulation skill islands
- control-plane and replay substrate testbeds
- partial pretraining domains

They are **not** yet valid proxies for a G1/R1-class deployment regime.

For humanoid-target readiness, the environment set eventually needs to cover:

- whole-body reaching and balance while manipulating
- locomotion-plus-manipulation transitions
- bimanual coordination
- dexterous hand contact
- human-proximate safety constraints
- mobile navigation and scene traversal
- recovery from pushes, slips, and contact disturbances
- spatially extended tasks rather than fixed workcell-only episodes

This should also include an explicit future simulation lane for:

- Unitree G1/R1-class robot simulation integration

That means the repo should eventually carry a named sim-env integration path for Unitree-class embodiments rather than assuming current workcell/tabletop envs can be stretched into that role.

### Contract implication

Preparing for G1/R1-class hardware also changes what the canonical contracts must carry:

- richer proprioception
- IMU and force/torque integration
- whole-body kinematic state
- contact state and support polygon / balance context
- latency and control-frequency metadata
- hardware safety envelope refs
- mobile spatial state

This means the later embodiment, sensor-fusion, safety, and SLAM phases are not optional polish. They are part of what “hardware-readiness” means.

### Compute and deployment implication

For G1/R1-class readiness, the stack also needs an explicit compute-placement plan.

A realistic deployment split is:

- fast reflex and servo control on robot or in the robot-adjacent low-level controller
- heavier perception, 3D grounding, and some WM inference on a companion GPU or high-end onboard compute
- economic/governance layers operating at slower rates on the same companion stack or a nearby controller

That means the architecture must eventually make explicit:

- what runs onboard versus offboard
- latency and bandwidth budgets between those layers
- what happens when the companion perception stack lags or drops
- how ROS2 / DDS / Unitree SDK2 or equivalent middleware is bridged into canonical WM state
- how battery, thermal, and compute-pressure receipts feed back into planning and economics

### Asset and calibration implication

Humanoid-target readiness also requires a named robot-asset and calibration discipline.

The stack will eventually need canonical handling for:

- URDF / Xacro / SRDF and related robot-description assets
- joint naming and action-index contracts
- sensor extrinsics and intrinsics
- hand / end-effector definitions
- self-collision models
- controller gain and hardware-calibration sidecars

Without that, the lower WMs can become structurally correct while still failing on actual robot identity and calibration truth.

### Benchmark implication

The benchmark story also changes once the target is G1/R1-class hardware.

Current manipulation/workcell benchmarks remain useful, but humanoid-target promotion will need additional benchmark classes such as:

- standing and balance stability
- locomotion plus manipulation success
- push / slip / stumble recovery
- self-collision and joint-limit compliance
- foot contact and support-phase consistency
- dexterous hand task completion
- human-proximate safety behavior
- sensor-dropout and degraded-perception robustness
- latency / watchdog / companion-link degradation behavior

This means the repo eventually needs a humanoid-target benchmark gate, not just stronger versions of the current workcell gates.

## Recommended WM Set

### 1. Perception / Grounding WM

Purpose:

- convert raw camera/depth/segmentation/tracking outputs into canonical world state
- own object, geometry, grounding, uncertainty, and scene persistence

Why needed:

- the repo currently has strong ingredients, but still has split ownership across SceneTracks, semantic fusion, map-first, rollout labeling, and vision stubs
- real production needs canonical state, not “vision sidecars plus semantic consumers”

Current anchors:

- `src/vision/scene_ir_tracker/io/scene_tracks_runner.py`
- `src/vision/reconstruction/four_d_reconstruction.py`
- `src/orchestrator/semantic_fusion.py`
- `src/world_model/semantic_world_model.py`
- `src/vla/rollout_labeler.py`

Current gaps:

- `src/vision/backbone_stub.py` is still a literal placeholder
- `src/vla/semantic_vla.py` is still a placeholder analyzer
- SceneTracks real grounding is host-blocked by SAM3D/GPU
- representation ownership is split instead of WM-owned

Disposition:

- this should become an upgraded WM that supersedes and encapsulates the current vision/grounding path rather than living beside it forever

### 2. Embodiment / Actuation WM

Purpose:

- own canonical body state, action semantics, capability envelopes, latency, and physical feasibility
- bridge from task/governance intent to embodiment-feasible action plans

Why needed:

- the repo has embodiment scaffolding, but not a canonical actuation WM
- current action/control assumptions are still mostly fixed-base manipulation assumptions
- future Unitree R1/G1-style control requires body-aware state, not just policy outputs

Current anchors:

- `src/embodiment/core.py`
- `src/embodiment/registry.py`
- `src/runtime/observation_adapter_v2.py`
- `src/runtime/action_adapter_v2.py`
- `src/motor_backend/*`

Current gaps:

- no whole-body embodiment state
- no URDF-driven body contract
- no humanoid locomotion or dexterous hand interface
- no real-time reflex/control separation yet

### 3. Sim / Synth / Physics WM

Purpose:

- own the canonical state for:
  - what should be simulated
  - what should be diffused/generated
  - what synthetic branches should be created
  - which physics backend/fidelity/randomization regime should be used
  - what receipts determine whether branches are valuable, admissible, and training-worthy

Why needed:

- this is currently the thickest flywheel gap
- current logic is improved but still distributed across orchestrator, diffusion, synthetic branch collection, backend shims, and physics stubs

Current anchors:

- `src/orchestrator/semantic_simulation.py`
- `src/orchestrator/diffusion_requests.py`
- `src/evidence/gen2sim_validity.py`
- `scripts/collect_local_synthetic_branches.py`
- `src/envs/physics/*`
- `src/motor_backend/holosoma_backend.py`

Current gaps:

- `src/envs/physics/isaac_backend.py` is still a stub
- parts of the LSD / NAG / GGDS path remain stubby
- agenda ownership is still orchestrator-heavy instead of WM-owned
- diffusion, gen2sim, and synthetic branch generation are not yet owned by one canonical state service

### 4. Economic WM

Purpose:

- value, allocate, schedule, gate, and learn over lower-WM states and receipts

Why it remains central:

- this is the repo’s core identity
- semantic WM is already functioning as one subset feeding toward this layer
- the recent work has made many control-plane helpers real and bounded

What still changes later:

- the economic WM should consume lower canonical WMs more directly instead of leaning on summary proxies
- its future neuralization should condition on lower-WM receipts, not just heuristic planning fields

### 5. Meta-Node Superposition / Control WM

Purpose:

- learn cross-WM governance and Pareto-objective policy over the entire stack
- run counterfactuals over meta-node choices, not only over local task actions

Important clarification:

- there are already local meta-node objects inside the semantic WM and adjacent helper lanes
- this future WM does not replace those local objects
- instead, it becomes the higher-order control layer that conditions, calibrates, and learns over them

Current local meta-node maturity:

- local meta-nodes are real routing/control objects now
- but they are still mostly named bounded-control surfaces with learned layers around them
- they are not yet fully learned geometric/cybernetic objects in their own right

Implication:

- the stack should not jump from today’s bounded local meta-node surfaces directly to an overarching superposition WM
- there needs to be a local meta-node neuralization and robustness tranche first
- that tranche should mature the local meta-node objects themselves before a mother-WM tries to learn over them

Current status:

- not ready to build next
- should remain a later phase after lower WMs and transport layers are real

## Why Sim / Synth / Physics WM Is Next

The current repo is already much farther along on semantic control-plane honesty than it is on self-improvement loop ownership.

The most valuable unresolved question is no longer “can the stack describe semantics?” It is:

- what exact simulation jobs get launched
- what diffusion jobs get requested
- what synthetic branches get created
- how those branches are physics-conditioned
- how outcome receipts come back into replay, training, and economic valuation

This is why the next WM should be sim/synth/physics before a standalone vision WM build-out.

That does **not** mean vision is unimportant. It means:

- the immediate flywheel bottleneck is agenda compilation and synthetic branch governance
- the later perception WM should plug into the sim/synth/physics WM as a provider of grounded state and uncertainty

## Phase Plan

### Phase 0 - Current Baseline and Preconditions

Status:

- largely in progress / partially landed

Objective:

- preserve the current economic-WM readiness work as the substrate for the multi-WM buildout

Already present:

- runtime packet substrate
- evidence bus and belief state
- semantic world-model packet
- event spine / governance / counterfactual / value-target sidecars
- honest helper promotion paths around selection, orchestration, queueing, sampler, fill routing, gen2sim admission, and semantic runtime scorers

Phase gate to start Phase 1:

- keep the current semantic/economic control-plane receipts as the source of truth
- do not reopen frozen Phase B math

### Phase 1 - Sim / Synth / Physics WM

This is the phase to focus on next. It is the only phase described here at near-implementation detail.

### Phase 1 objective

Create a canonical sim/synth/physics WM that consolidates simulation agenda compilation, diffusion conditioning, gen2sim admission context, backend/fidelity selection, and synthetic branch receipts into one governed state surface.

Complete-subsystem interpretation for Phase 1:

- the subsystem should own live simulation planning, diffusion planning, backend/fidelity routing, branch planning, and receipt emission in the real runtime loop
- it should already contain the runtime-package lanes for the learned seams it needs
- do not advance to Phase 2 while Phase 1 still has implementable ownership, adapter, receipt, or training-feedback gaps inside the sim/synth/physics boundary
- after that, the remaining blockers should be honest ones such as:
  - lack of real Unitree-class sim adapters
  - lack of grounded whole-body datasets
  - lack of GPU-backed large-scale predictive training
  - lack of calibration and benchmark receipt density

### Phase 1 ownership boundary

This WM should own:

- simulation agenda state
- branch-generation state
- diffusion-conditioning state
- backend and physics fidelity state
- synthetic admission and outcome receipts
- branch-to-training feedback state

This WM should not own:

- raw perception model execution
- whole-body actuation control
- pricing or economic settlement itself
- final meta-node governance policy

### Phase 1 current repo surfaces to absorb

| Current surface | Current problem | Future role under sim/synth/physics WM |
| --- | --- | --- |
| `src/orchestrator/semantic_simulation.py` | mixes selection, agenda, execution setup, and artifact emission | becomes a client of the WM runtime/compiler |
| `src/orchestrator/diffusion_requests.py` | diffusion request logic is adjacent to but not owned by a WM | becomes a diffusion-plan adapter fed by WM state |
| `src/evidence/gen2sim_validity.py` | admission logic exists but is not the sole owner of synth state | becomes one submodule inside WM admission/receipt logic |
| `scripts/collect_local_synthetic_branches.py` | branch generation is script-owned | becomes a WM branch-generation adapter / worker |
| `src/envs/physics/isaac_backend.py` | stub backend with no canonical contract | becomes one backend adapter behind WM backend routing |
| `src/motor_backend/holosoma_backend.py` | backend-specific bridge is outside WM ownership | becomes an execution adapter used by the WM |
| `src/envs/lsd3d_env/ggds.py` and NAG/LSD surfaces | stubby optimization path | becomes an explicit branch renderer/generator provider, not an owner |

### Phase 1 proposed module structure

Recommended additive package:

```text
src/world_model/sim_synth_physics/
  __init__.py
  state.py
  agenda.py
  compiler.py
  backend_router.py
  physics_contracts.py
  diffusion_contracts.py
  synthetic_branches.py
  gen2sim_admission.py
  receipts.py
  calibration.py
  runtime.py
  dataset.py
  training.py
  promotion.py
  adapters/
    semantic_inputs.py
    economic_inputs.py
    embodiment_inputs.py
    backend_pybullet.py
    backend_holosoma.py
    backend_isaac.py
```

Recommended scripts:

```text
scripts/compile_sim_synth_physics_plan.py
scripts/run_sim_synth_physics_loop.py
scripts/train_sim_synth_physics_planner.py
scripts/eval_sim_synth_physics_planner.py
```

### Phase 1 canonical state objects

Recommended typed objects:

- `SimSynthPhysicsWorldState`
  - top-level canonical state for one run/episode/planning window
- `SimulationAgenda`
  - ranked set of simulation jobs to execute
- `SimulationJobSpec`
  - one job with env family, backend, seed, objective preset, coverage targets, and expected receipts
- `PhysicsContextState`
  - fidelity, backend, timestep, randomization, calibration, and safety-relevant physics settings
- `DiffusionConditioningState`
  - governed diffusion request structure, not just a prompt string
- `SyntheticBranchPlan`
  - branch family, gap targets, rendering/generation mode, admission preconditions
- `Gen2SimAdmissionState`
  - explicit branch admissibility context and helper traces
- `SimulationOutcomeReceipt`
  - outcome refs, replay refs, event/debt/governance/value refs, and readiness summary
- `PhysicsCalibrationReceipt`
  - domain-randomization/system-ID summary and backend-quality flags

### Phase 1 input contracts

The WM should consume:

- `BeliefState`
- `SemanticWorldModelState`
- `RuntimePacket`
- coverage-loop outputs
- benchmark-gating signals
- gen2sim validity/admission receipts
- embodiment constraints when available
- economic urgency and value-target summaries

The WM should emit:

- simulation agenda
- diffusion conditioning plan
- synthetic branch plan
- backend/fidelity choice receipt
- branch outcome receipts
- replay-ready artifact refs
- training-feedback refs

### Phase 1 runtime flow

Recommended flow:

1. ingest semantic, economic, and embodiment context through typed adapters
2. compile a `SimSynthPhysicsWorldState`
3. rank simulation jobs and branch jobs inside one agenda
4. choose backend, fidelity, and domain-randomization regime
5. emit a `DiffusionConditioningState` for any render/generation branch
6. execute backend adapters
7. emit `SimulationOutcomeReceipt` and related sidecars
8. feed receipts into replay, benchmark gating, and training datasets

### Phase 1 neuralization plan

Do not start with a giant generative simulator model.

Do start with bounded learned helper seams in Phase 1A / 1B itself rather than adding them later as a cleanup:

- agenda ranking
- backend/fidelity selection
- synthetic branch planning
- synthetic branch admission
- branch-value / branch-yield prediction
- physics-calibration confidence prediction

For this phase, heuristics are only acceptable as explicit priors with fallback semantics and receipt traces.

Only after the receipt density is real should the WM expand into:

- learned transition models
- learned synthetic branch proposal models
- v-JEPA-2-style predictive state modules for video-conditioned future estimation

### Where v-JEPA 2 fits

v-JEPA-2-style predictive modeling should be treated as a component inside the future sim/synth/perception stack, not as a separate top-level WM.

Recommended placement:

- later as a predictive latent module that conditions on governed video state, scene tracks, embodiment context, and action plans
- downstream of canonical state
- upstream of render/synthetic branch proposal ranking

### Phase 1 training targets

The first honest training targets for this WM should come from:

- branch evaluations
- value-target packs
- counterfactual evals
- governance traces
- replay outcome receipts
- coverage improvement deltas
- branch yield into later trainer datasets

### Phase 1 acceptance criteria

Phase 1 should count as landed only when:

- simulation agenda ownership is WM-owned rather than spread across orchestrator helpers
- diffusion conditioning is derived from WM state instead of flat prompt assembly alone
- synthetic branch plans are typed objects, not script-local metadata
- backend/fidelity selection is emitted as a first-class receipt
- replay/training consume WM receipts without bespoke joins
- Isaac remains an explicit fallback until a real adapter exists, but it is no longer hidden behind a generic backend name
- the target posture is real Isaac Sim / Isaac Gym / Unitree-class backend functionality behind typed backend routing, not a permanent pybullet-only fallback loop
- the learned seams needed by the subsystem already have runtime-package loading and live-loop integration rather than existing only as loose code hooks
- the remaining gap list is primarily real-data / GPU / adapter / benchmark work, not missing subsystem wiring

### Phase 1 explicit gaps

Named gaps that should remain explicit in this phase:

- real Isaac Sim / Isaac Gym backend implementation with typed adapter ownership
- Unitree-class humanoid sim-env integration behind a typed backend contract
- richer Holosoma integration contract
- domain randomization and system identification policy
- NAG / LSD / GGDS productionization
- real GPU-backed grounded video state for perception-conditioned sim

### Phase 1 OSS dependency posture

Use OSS as providers, not truth owners:

- MuJoCo / dm_control / PyBullet / Isaac for physics execution
- Holosoma where already integrated
- later v-JEPA-2-style predictive modules for future estimation

Target posture for this phase:

- real Isaac Sim / Isaac Gym / Unitree-class functionality should eventually sit behind the WM's typed backend adapters
- explicit fallback to PyBullet is acceptable only while the repo is still missing those adapters or the target assets
- the fallback posture should stay honest in receipts and benchmark reporting until the adapter gap is truly closed

The WM should own:

- canonical state
- routing
- receipts
- promotion logic
- governance hooks

### Phase 2 - Perception / Grounding WM

Objective:

- turn the current vision, scene-tracks, map-first, teacher-runtime, and grounding path into one canonical perception/grounding WM

Why after Phase 1:

- sim/synth/physics is the tighter current flywheel bottleneck
- this phase should absorb and supersede the current split vision ownership rather than creating yet another parallel vision lane

Phase outcome:

- one canonical scene/grounding state surface feeding semantic, embodiment, and sim WMs

Neuralization rule from tranche 1:

- perception WM should expose bounded learned seams for grounding confidence, sensor-fusion calibration, view selection, and evidence routing immediately
- fallback heuristics may remain, but only as explicit priors with helper traces and promotion posture

Complete-subsystem rule:

- perception WM should be pushed until the main missing pieces are real grounded 3D data, real SAM3D/GPU hosts, camera calibration truth, and Unitree-class sensor corpora, not missing runtime-package or production-loop wiring
- do not advance to Phase 3 while the perception WM still has implementable canonical-state, adapter, helper-package, or replay-wiring gaps

Key named gaps:

- `src/vision/backbone_stub.py`
- `src/vla/semantic_vla.py`
- partial SceneTracks truth still gated by SAM3D host availability
- recap and vision feature builders still contain stub/fallback roots

OSS dependency map:

- segmentation: SAM2
- open-vocab grounding: GroundingDINO
- visual features: DINOv2 or SigLIP
- depth: Depth Anything V2 or UniDepth
- 3D grounding: SAM3D, ConceptGraphs, ScanNet-style baselines

Preconditions:

- Phase 1 receipts must be able to consume richer grounded scene state
- real GPU/SAM3D host needed for promotion-grade grounding

### Phase 3 - Embodiment / Actuation WM

Objective:

- create canonical body/action state for fixed-base and future humanoid/mobile embodiments

Why needed:

- the repo still lacks a real body model for Unitree-class systems
- the current embodiment layer is useful but advisory and manipulation-centric

Phase outcome:

- action feasibility, embodiment capability, latency, and control envelopes become first-class typed state

Neuralization rule from tranche 1:

- embodiment WM should expose bounded learned seams for capability estimation, action-feasibility scoring, latency-envelope prediction, and backend/robot adapter selection immediately
- fallback heuristics may remain only as explicit priors with helper traces and promotion posture

Complete-subsystem rule:

- embodiment WM should be pushed until the main missing pieces are real robot-description assets, whole-body control datasets, Unitree-class sim assets, safety calibration, and hardware benchmark receipts, not missing runtime-package or live-loop integration
- do not advance past Phase 3 while embodiment still has implementable body-state, adapter, capability-model, or control-envelope gaps

OSS dependency map:

- whole-body and IK: MuJoCo, Pinocchio, Pink, Drake
- locomotion: legged_gym, walk-these-ways, Unitree RL Gym
- dexterous manipulation: DexGraspNet, IsaacGymEnvs dexterous tasks, HORA
- robot description: Unitree URDFs

Preconditions:

- stable action/observation schema refs
- capability profiles beyond fixed-base workcell assumptions
- initial robot descriptions for target embodiments

### Phase 3.5 - Humanoid Target Capacity and Environment Refit

Objective:

- explicitly audit model capacity and redesign environment assumptions for G1/R1-class readiness

Why this phase must exist:

- a 21+ DoF humanoid target changes what “enough model” means in embodiment, perception, simulation, and transport layers
- current envs were not designed as humanoid-readiness benchmarks
- without an explicit refit phase, the repo could accumulate elegant middleware around the wrong training domains

What this phase should deliver:

- a model-capacity review across lower WMs and submodule models
- an explicit list of modules that can stay compact versus modules that must scale
- revised humanoid-facing observation/action/schema requirements
- an environment roadmap that reclassifies current workcell/tabletop envs as partial domains rather than full humanoid proxies
- a named plan for integrating Unitree G1/R1 simulation environments into the sim/synth/physics stack through typed backend adapters rather than ad hoc env forks
- named future envs or env families for:
  - locomotion + manipulation
  - balance-constrained reaching
  - bimanual manipulation
  - dexterous hand tasks
  - mobile navigation + task execution
  - contact disturbance and recovery
- do not advance beyond this refit until the remaining uncertainty is genuinely about assets, datasets, and benchmark evidence rather than carrying forward wrong environment assumptions

Minimum outputs:

- `humanoid_target_readiness.md`-style architecture note or equivalent roadmap artifact
- capacity budgets or scaling bands for key lower-WM modules
- canonical schema deltas needed for proprioception, IMU, force/torque, safety, and spatial state
- a concrete Unitree sim-env integration target:
  - backend choice
  - robot description source
  - observation/action contract deltas
  - receipt and replay compatibility plan
- a first humanoid benchmark taxonomy covering:
  - balance
  - locomotion-manipulation
  - recovery
  - dexterous manipulation
  - degraded-sensing robustness

Preconditions:

- initial embodiment WM contract draft
- initial perception WM contract draft
- identified target hardware assumptions for Unitree-class robots

Reference artifact:

- `docs/economic_world_model/humanoid_target_readiness.md`

### Phase 4 - Deployment Enabler Phases

These are not optional side quests. They are named future phases that must exist before serious embodied deployment.

Cross-phase rule here too:

- if an enabler phase introduces learned routing, calibration, safety scoring, operator-handoff selection, or degradation handling, that seam should ship as a bounded helper/runtime contract from the first landing
- do not strand those decisions in untyped helper code and plan to “neuralize later”

### Phase 4A - Real-Time Control Loop Separation

Objective:

- separate servo/reflex timescale control from economic/governance timescale control

Reason:

- humanoid or mobile robots need 200-1000 Hz reflex control
- the economic/orchestrator layer should stay slow and deliberative

OSS dependency map:

- `ros2_control`
- MuJoCo
- Unitree SDK2

Preconditions:

- Embodiment / actuation WM with typed action semantics
- target robot control interface

### Phase 4B - Sensor Fusion Shim

Objective:

- build the plumbing from Unitree SDK2 or equivalent raw streams into OSS perception outputs and then into canonical state

Reason:

- somebody has to turn raw cameras, IMU, joint states, and depth into the typed state the WMs consume

OSS dependency map:

- `robot_localization`
- GTSAM
- Kalibr
- Unitree SDK2

Preconditions:

- Perception WM canonical contract
- observation schema with timestamp and proprio support
- access to live sensor streams

### Phase 4C - Physical Safety Layer

Objective:

- add physical safety below the governance/economic layer

Reason:

- economic safety signals are not enough for real robots
- joint limits, self-collision, e-stop, and reflex policies require a separate layer

OSS dependency map:

- `ros2_control`
- `safe-control-gym`
- collision checking from MuJoCo / Drake / Pinocchio stacks

Preconditions:

- embodiment state and robot description
- real-time control loop separation

### Phase 4D - Spatial State / SLAM Integration

Objective:

- support mobile spatial reasoning and navigation state

Reason:

- current repo assumptions are mostly fixed-workcell
- a Unitree-class system needs mapping, localization, and navigation

OSS dependency map:

- ORB-SLAM3
- RTAB-Map
- Nav2

Preconditions:

- perception WM canonical state
- sensor fusion shim
- mobile embodiment target

### Phase 4E - Companion Compute and Communication Middleware

Objective:

- formalize the robot/companion compute split and the middleware that moves typed state between them

Reason:

- G1/R1-class deployments are unlikely to run every perception, grounding, and WM component in one undifferentiated process
- communication latency, packet loss, QoS, and degraded-link behavior become real runtime concerns

OSS dependency map:

- ROS2 / DDS
- Unitree SDK2
- ZeroMQ or equivalent auxiliary transport where needed

Preconditions:

- embodiment and perception contracts must already exist
- timing metadata must be part of canonical state
- deployment targets must be explicit about onboard vs companion compute

### Phase 4F - Operator / Teleop / Recovery Fallback Layer

Objective:

- provide explicit human-override, teleoperation, and recovery-mode contracts below the economic/governance layer

Reason:

- serious humanoid readiness requires a bounded operator-recovery path for bring-up, failure handling, and safety-critical override
- this should become part of the canonical event/governance/replay story rather than an informal manual escape hatch

Preconditions:

- real-time control split
- physical safety layer
- communication middleware

### Phase 5 - Economic WM Consolidation

Objective:

- make the economic WM consume lower-WM canonical state directly rather than through scattered summaries and heuristic planning fields

This phase is where the current economic-WM work graduates from “strong middleware and bounded helpers” to “real federated world-model governance.”

Key changes in this phase:

- consume `SimSynthPhysicsWorldState`, perception state, and embodiment state directly
- train on lower-WM receipts and cross-WM counterfactuals
- condition meta-transformer and higher planners on lower-WM canonical contracts instead of only derived summary vectors

Neuralization rule from tranche 1:

- economic-WM consolidation should keep bounded learned allocators, critics, and governance helpers wired from the first lower-WM-native runtime pass
- lower-WM state consumption should not regress into summary-only heuristics while waiting for a later purge

Complete-subsystem rule:

- economic-WM consolidation should be pushed until the main blockers are receipt density, deployment economics, and lower-WM maturity, not missing federated runtime loops or missing learned-package infrastructure

Preconditions:

- at least two lower WMs emitting stable canonical state
- replay/evidence/value provenance must preserve WM identity explicitly

### Phase 6 - Cross-WM Isomorphic Transport Layer

Objective:

- create learned typed transport bridges between adjacent WMs

This should be middleware, not a mother-WM.

### WM-transport ontology vs differentiable bridge

This phase should explicitly contain two cooperating pieces:

- a WM-transport ontology that defines the typed semantic, uncertainty, provenance, and governance contract for valid adjacent-WM mappings
- an isomorphic tensor / transport bridge that is the fast differentiable realization of that contract

The ontology is the contract layer.
The tensor bridge is the compiled differentiable bridge.
Do not collapse one into the other.

### Transport layer role

Each bridge should:

- translate one WM’s canonical state into the vocabulary of the adjacent WM
- preserve topology where topology matters
- preserve causal structure where causal structure matters
- preserve semantic/actionability constraints defined by the WM-transport ontology
- carry uncertainty and provenance explicitly
- learn from completed loops and postmortem receipts

### Recommended boundary rule

Bridges should operate over typed objects, not raw hidden states.

Examples:

- `BeliefState` -> perception WM state
- perception WM state -> `SemanticWorldModelState`
- `SemanticWorldModelState` -> `SimSynthPhysicsWorldState`
- lower WMs -> economic WM state
- economic WM state -> meta-node control WM state

### Recommended module posture

Recommended additive package:

```text
src/world_model/transport/
  __init__.py
  bridge_contracts.py
  topology_metrics.py
  uncertainty.py
  roundtrip.py
  training.py
  runtime.py
```

### Training rule

Train bridges by freezing adjacent WMs first.

Neuralization rule from tranche 1:

- each bridge should launch with bounded learned translation/calibration seams and round-trip receipts immediately
- heuristic adapters may remain as the prior, but not as the only runtime story

Complete-subsystem rule:

- transport bridges should be pushed until the main blockers are cross-WM corpus density and topology/latency evaluation, not missing bridge runtime contracts or missing live consumers

Core decomposed evaluation:

- bridge-only improvement
- downstream-WM-only improvement
- joint improvement
- interaction term

Recommended training signals:

- WM -> ontology -> WM translation quality
- round-trip reconstruction
- topology preservation
- causal / dependency preservation
- uncertainty calibration
- downstream economic yield
- governance satisfaction
- postmortem counterfactual improvement

Preconditions:

- at least two adjacent WMs emitting stable canonical state
- replay must preserve WM identity and provenance
- freeze-one-side training scaffold must exist

### Phase 6.5 - Local Meta-Node Neuralization and Robustness

Objective:

- upgrade the existing local meta-node surfaces from named bounded-control objects into more genuinely learned, stateful, cybernetic objects

Why this phase must exist:

- the repo already has local meta-node routing state in the semantic WM and adjacent orchestration layers
- those objects are useful and real, but they are still closer to bounded executor/control surfaces than to learned dynamical objects
- an overarching meta-node superposition WM would be premature if its “atoms” are still mostly hand-named shells with learned wrappers around them

What should improve in this phase:

- local meta-node state should become canonical and replayable rather than mostly summary-like
- local meta-node behavior should train on its own receipts, counterfactuals, and governance outcomes
- local meta-node embeddings should become more geometric/topological and less just named scalar buckets
- meta-node success should be measured independently from downstream loop success where possible
- meta-node interaction effects should be logged, not only final routed actions

Neuralization rule from tranche 1:

- each local meta-node should expose bounded learned activation / intervention seams and explicit promotion posture as soon as it becomes canonical state
- do not reintroduce a “named shell first, neuralization later” pattern here

Complete-subsystem rule:

- local meta-node work should be pushed until the main blockers are counterfactual corpus density and robustness evaluation, not missing package/runtime contracts or missing replay hooks

Concrete targets for this tranche:

- canonical `MetaNodeState`-style packet surfaces inside the lower WMs
- richer meta-node trajectory and intervention receipts
- counterfactual training targets for:
  - when a meta-node should activate
  - how strongly it should activate
  - which targets it should act on
  - when it should defer or veto
- robustness metrics over:
  - stability under replay shift
  - governance satisfaction
  - calibration
  - interaction consistency with neighboring meta-nodes

Exit criteria:

- local meta-nodes are no longer just bounded named routing shells with learned helpers around them
- they have their own honest training/runtime/promotion story
- the later superposition WM can treat them as mature lower-level objects instead of pseudo-symbolic placeholders

### Phase 7 - Meta-Node Superposition / Control WM

Objective:

- learn the policy over cross-WM governance, objective tradeoffs, and meta-node Pareto control

This is the final mother-layer, not the next layer.

What it should learn over:

- economic WM receipts
- lower-WM readiness and uncertainty
- counterfactual governance outcomes
- meta-node action histories
- cross-WM transport quality

What it should not do:

- directly replace lower WMs
- become a giant hidden-state monolith
- erase local WM contracts or governance traces

Neuralization rule from tranche 1:

- the mother-layer should also launch with bounded learned control seams, typed helper traces, and explicit promotion posture
- it should not begin life as a heuristic governor that later needs a purge pass

Complete-subsystem rule:

- the mother-layer should only be considered blocked when the remaining issue is lower-WM maturity, cross-WM corpus density, or real governance benchmark evidence, not absent runtime packaging or absent live-loop wiring

Preconditions:

- lower WMs are robust and honest
- economic WM is consuming canonical lower-WM state
- transport bridges are working between adjacent WMs
- local meta-nodes have already passed their own neuralization/robustness tranche and emit canonical state plus trainable receipts
- meta-node actions and governance satisfaction are already logged as trainable receipts

## OSS Dependency Map

| Function | Preferred OSS posture | Role in this architecture |
| --- | --- | --- |
| Segmentation / masks | SAM2 | frozen provider into perception WM |
| Open-vocab grounding | GroundingDINO | frozen provider into perception WM |
| Depth | Depth Anything V2 / UniDepth | frozen provider into perception WM |
| Visual features | DINOv2 or SigLIP | frozen provider into perception WM and recap/selector conditioning |
| 3D grounding | SAM3D / ConceptGraphs | frozen provider into perception WM |
| Physics | MuJoCo / PyBullet / Isaac / Holosoma | execution providers behind sim/synth/physics WM |
| VLA | OpenVLA / Octo / RT-family OSS | external proposal provider, not canonical truth owner |
| Whole-body dynamics / IK | Pinocchio / Pink / Drake | provider into embodiment WM |
| Whole-body / locomotion RL | legged_gym / walk-these-ways / Unitree RL Gym | provider into embodiment WM |
| Safety / low-level control | ros2_control / safe-control-gym | provider below governance layer |
| Sensor fusion | robot_localization / GTSAM / Kalibr | provider into sensor-fusion shim |
| SLAM / nav | ORB-SLAM3 / RTAB-Map / Nav2 | provider into spatial-state phase |

The stack’s differentiator should remain:

- typed canonical state
- governance
- allocation
- valuation
- promotion discipline

not rebuilding every OSS foundation model in-house.

## Stranded Modules and Explicit Future Absorptions

The following existing surfaces should be treated as absorption targets, not permanent architecture:

- `src/vla/semantic_vla.py`
- `src/vision/backbone_stub.py`
- `src/vla/recap_dataset_builder.py`
- `src/envs/physics/isaac_backend.py`
- `src/envs/lsd3d_env/ggds.py`
- `src/vision/spatial_rnn_adapter.py`
- `src/vla/backbones/dummy_backbone.py`

Disposition rule:

- keep them explicit until replaced
- do not let them silently count as canonical state owners

## Decision Summary

The repo should not jump straight from the current semantic/economic stack to a meta-node mother-WM.

The correct next macro-sequence is:

1. build the sim/synth/physics WM
2. build the perception/grounding WM
3. build the embodiment/actuation WM
4. audit model capacity and refit environment assumptions for humanoid-target readiness
5. land the real-time, sensor-fusion, safety, and SLAM enabler phases
6. consolidate the economic WM over those lower WMs
7. add cross-WM typed transport bridges
8. neuralize and harden the local meta-node objects themselves
9. build the meta-node superposition/control WM

## Immediate Follow-On Recommendation

For the next implementation tranche after this plan:

- start **Phase 1A and Phase 1B** only
- define the `sim_synth_physics` package structure and state contracts
- move agenda ownership out of scattered orchestrator surfaces and into that canonical WM boundary
- wire bounded learned seams for backend/fidelity and branch planning into that boundary immediately so the new WM does not start life as heuristic-only
- treat the subsystem target as "push until data, GPUs, adapters, and Unitree-class assets are the real bottleneck"

That is the cleanest path to make the stack a real self-improvement machine rather than only a better-instrumented control plane.
