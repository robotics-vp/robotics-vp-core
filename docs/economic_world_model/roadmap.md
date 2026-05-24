# Economic World Model Multi-Week Roadmap

## Planning Rules

- Favor additive infrastructure over refactors.
- Keep VLA and foundation-model paths external, pluggable, and sidecar/advisory.
- Narrow the advisory doctrine for internal surfaces: external providers and preview layers may remain advisory, but internal typed quality/readiness/learnability artifacts should graduate into canonical metadata, preconditions, work orders, or bounded authority once they affect runtime or training.
- Preserve objective integrity. No premature scalarization upstream of explicit contract compile.
- Preserve the stable Phase B checkpoint and legacy contractive dynamics math as a baseline, but additive governed video-state work in `src/world_model/` is now in scope.
- Treat Phase B as a two-layer program: frozen baseline for rollback/comparison, additive successor scaffolding for governed video-state and later learned top-layer work.
- Treat frozen Phase B math as the current canonical rollback anchor, not as philosophically immutable forever; any future replacement must come only through benchmark-gated successor evidence, not casual edits.
- Sequence work so each stage leaves behind reusable docs, schemas, tests, and automation hooks.
- Treat video-world-model work as a subset of economic-world-model readiness: geometry/evidence/governance first, rendering and training second.
- Do not overfit the stack to one paper architecture. Keep predictor, planner, context, and rollout configuration modular.
- Treat sim-to-online stabilization lessons from individual papers as future
  training-loop doctrine under the current WM topology, not as topology drivers
  or phase-ordering overrides.
- Treat a tranche as incomplete until it is wired into at least one already-executed path. For the current subset that means the Stage-1 video loop, rollout labeling, shadow runtime, replay ingest, or another live path must emit the new artifacts.
- For any new WM or enabling subsystem, ship bounded learned seams and `disabled|auto|required` promotion posture from the first landing; keep heuristics only as explicit priors/fallbacks with receipts, not as temporary hidden owners that will later require a heuristic-purge rewrite.
- For any new WM or enabling subsystem, bias against literal stub defaults: use real-or-unavailable provider contracts, make `stub` explicit-only for smoke/scaffolding, and record planning-only fallback truth when weights/GPU/assets are the actual blocker.
- Treat each WM as a complete subsystem target: push canonical state, learned-package lanes, replay/receipt wiring, and production-loop integration until the honest remaining blockers are data, GPU budget, calibration assets, Unitree-class sim/hardware assets, or benchmark evidence.
- For multi-WM work, explicitly rerun the deterministic-prior / heuristic audit inside each WM boundary; do not assume the earlier global heuristic purge already finished that job for every later WM module.
- Treat V-JEPA 2 as an external/provider lane that belongs in both the Phase-1 sim/synth/physics WM and the later Phase-2 perception/grounding WM; prefer upstream `facebookresearch/vjepa2` integration where it beats local reimplementation, but keep it behind typed provider/runtime contracts and receipts.
- Treat on-device inferential compute capacity and concrete battery state as lower-WM canonical resource contracts first; let the economic WM allocate them later and the meta-node layers govern them only after those lower contracts are real.
- For the next cross-WM expansion beyond the current semantic/economic readiness work, see `docs/economic_world_model/multi_wm_architecture_plan.md`.

## Mereotopological Datapack Rule

Treat every datapack as a structured composite object, not a flat bag of trajectories or simple source buckets like "real / sim / video." This requires preserving source lineage, transformation lineage, WM-touch history, and temporal insertion order. Preserve two distinct decompositions:
1. **Material provenance composition**: what percentage of the datapack materially came from real data, sim, video priors, synthetic branches, or transformed artifacts.
2. **Functional contribution composition**: what percentage of the datapack’s useful effect came from grounding, semantic compression, robustness expansion, counterfactual exploration, or embodiment relevance.

Evaluate datapacks under the currently active objective/economic manifold. Do not overfit the stack to arbitrary source categories; the question is not "how much sim?" or "how much real?" in isolation, but what each part contributed to the loop under current constraints.

Epiplexity should be used as a major input for this reasoning: datapacks should be judged partly by the structured usable information they contain under bounded compute, bounded embodiment, bounded time, and bounded deployment constraints. Epiplexity is not just an auxiliary ranking trick, but one of the main ways the stack estimates structured informational yield, marginal learnability, compression of actionable structure, and whether a candidate branch genuinely increases usable order versus just adding volume or noise.

This is a standing doctrine from Phase 1 onward, integrating directly into the logic of lower-WM state ownership, bounded seams, and later economic ingestion.

## Ontology Layering Rule

Keep two ontology layers explicit throughout roadmap work.

Operational / module-level ontology:

- this is the in-stack ontology substrate for entities, tasks, datapacks, events, provenance, governance hooks, and module/runtime state
- it should become more neuralized over time through learned embeddings, uncertainty-aware assertions, temporal/event structure, and trainable module-to-ontology adaptors
- its training role is operational fidelity: better encoding/decoding, event/state prediction, temporal consistency, uncertainty calibration, provenance quality, and governance satisfaction
- its reward should come from completed-loop postmortems, reconstruction quality, calibration quality, and operational yield, not from taking over the frozen core reward math right now

WM-transport ontology:

- this is distinct from the operational ontology and should appear only when adjacent WMs are real
- it is the typed semantic/governance contract for WM-to-WM interoperability
- the isomorphic tensor / transport bridge is the fast differentiable realization of that contract, not a replacement for it
- the transporter is received through per-WM functionality-specific transformer layers; it must not decode straight into every WM as if all WMs shared one vocabulary or authority surface
- its training role is improving WM-to-ontology-to-WM translation quality, source-exporter quality, target-receiver actionability, topology/causal structure/actionability preservation, synchronized-loop success, and decomposing bridge-only vs receiver-only vs downstream-only vs joint gains
- exporter/bridge/receiver training should be staged and measured separately: WM-local exporter/receiver pretraining, bridge-only topology alignment, round-trip/receiver-actionability training, then downstream shadow shaping
- direct task-reward RL should not train transport as a policy; RL-style signals should enter through Economic WM allocator/governance receipts, constraints, sample weights, and downstream labels
- its reward should come from completed-loop/postmortem outcomes, counterfactual improvement, governance satisfaction, and downstream economic yield for the adaptor/bridge layer

Current honest state:

- today the repo mostly has operational ontology substrate/plumbing
- it does not yet have a fully neural ontology layer
- it now has a local Phase-6.0-6.4 WM-transport scaffold: contracts, per-WM exporters/receivers, rows, topology/round-trip/uncertainty receipts, neural manifest, loss ledger, non-training trainer scaffold, advisory runtime proposals/receipts, decomposed eval reports, and shadow outcome join slots
- the Phase-6 local closure audit reports `missing_local_runtime_contracts=[]`; the remaining blockers are corpus density, GPU bridge/receiver training, latency/topology benchmarks, provider/hardware transport evidence, promotion-grade downstream evidence, and live transport authority
- keep the current sequencing: lower WMs first, then economic-WM consolidation, then ontology-mediated WM transport promotion only after evidence exists

## Open-Vocabulary Segmentation / Concept-Tracking Provider Rule

- the stack should reserve explicit provider/runtime contracts for open-vocabulary segmentation and concept-conditioned video tracking
- SAM 3 / 3.1 is the preferred initial provider lane for this capability
- these capabilities should not remain implicit inside ad hoc labeling code, weak placeholder analyzers, or opaque teacher-sidecars
- concept-conditioned segmentation and tracking should become typed, replayable, provider-truthed artifacts
- those artifacts should be consumable by:
  - Perception / Grounding WM
  - Sim / Synth / Physics WM branch evaluation
  - rollout labeling / annotation
  - semantic evidence
  - later embodiment-facing object memory and humanoid egocentric perception
- the doctrine should remain consistent with the repo’s anti-stub posture:
  - prefer **real-or-unavailable**
  - use explicit `stub` posture only for smoke/scaffolding
  - record provider truth honestly
  - do not let weak placeholders silently masquerade as semantic capability

SAM 3 / 3.1 should be treated as part of **provider bring-up / OSS backlog discipline**. A named provider bring-up item must cover:
- image predictor bring-up
- video predictor / tracking bring-up
- memory / multiplex mode evaluation
- weights / checkpoint access
- runtime packaging
- GPU/runtime host requirements
- provider truth and unavailable-mode semantics
- benchmark / smoke expectations
- real vs planning-only posture on hosts where the environment is not yet available

This must be treated like a real external-provider lane that needs to be burned down through bring-up work, not just a “future good idea.”

## Semantic Analysis Successor Posture

- `SemanticVLA` should either become a real provider-backed semantic-analysis layer or be explicitly demoted to scaffolding-only status.
- The likely successor to `SemanticVLA` is **not** one monolithic semantic-analysis model. It is a composed semantic layer built from:
  1. Perception-WM canonical object/track state
  2. teacher/runtime semantic proposals
  3. affordance / role inference over canonical objects
  4. primitive/action segmentation
  5. semantic-evidence fusion / annotation crosswalk
- This successor should be treated as a **provider-backed and fusion-backed semantic stack**, not a one-model replacement fantasy.

The successor must be a **real neural semantic subsystem**, not a thin wrapper or lightly neural tagger:
- It sits **downstream of Perception / Grounding WM canonical state** and **upstream of Sim/Synth/Physics WM, Embodiment/Actuation WM, annotation/semantic-evidence surfaces, and Economic WM consumption**.
- It is a **load-bearing semantic state layer** that must be given enough structured neural capacity to bind objects/tracks to action, affordance, and task structure.
- Do not starve it into a tiny helper head, but keep its capacity **topologically distributed and WM-shaped** rather than collapsing it into one monolithic model. Neural capacity and hyperparameter posture should be governed by the relevant WM that owns or consumes the semantics:
  - **Perception/Grounding WM** shapes the object-token, relational, and temporal layers.
  - **Embodiment/Actuation WM** shapes the affordance- and action-relevance layers for bodily control.
  - **Sim/Synth/Physics WM** shapes the branch-comparison and object-preservation layers for synthetic evaluation.

The repo must track a parallel backlog / provider bring-up item stating the stack should either:
- identify a real semantic proposal provider or family of providers for this role, or
- keep `SemanticVLA` explicitly scaffolding-only until that provider stack is named and brought up honestly.
- unsettled provider choice is acceptable.
- unowned placeholder status is not acceptable.

## Habitat-Derived Adoption Track

The Habitat pass is not exhausted across the stack. The following summarizes
what has been absorbed and what remains open.

### Perception / Grounding WM (Phase 2) — mostly absorbed

The right Habitat-level lessons are already in the branch:

- clean separation of provider/dataset/task/resource surfaces
- explicit sensor/provider truth rather than one opaque env blob
- loop-facing compilation rather than pure schema
- deployment/headroom surfaces as lower-WM typed state

No further Habitat copying is needed for Perception; the remaining work is
real benchmark-evidence production, runtime materialization of provider-backed
token paths, and provider-specific calibration artifact generation.

UE5 / Unreal outputs should later be treated as strong Phase 2 inputs rather
than as a replacement ontology: photoreal scene renders, synthetic sensor
streams, render-time labels, and deployment-matched digital-twin captures can
become useful Perception-WM training and evaluation corpora once the Sim /
Synth provider/runtime lanes and receipts are real enough to produce them
honestly.

### Sim / Synth / Physics WM (Phase 1.x) — biggest remaining opportunity

This is where the most Habitat-derived learning still sits. It should be
named as an explicit reopenable Phase 1 adoption track, not forgotten.

> **GPU-Era Decomposition Revisit**: Phase 1 was declared structurally closed
> on 2026-04-02 with zero Category A items. The current implementation
> establishes runtime truth, adapter ladders, and typed contracts/receipts.
> When this WM is reopened during the GPU/runtime era, it should be held to
> the same **internal subsystem decomposition rigor** now exhibited by the
> Economic WM doctrine and the Embodiment / Actuation WM plan. This is not a
> repudiation of Phase 1 closure but an **elevation of the standard**. The
> 10-subsystem decomposition target and the 9-point WM Section Readiness
> Standard are documented in `multi_wm_architecture_plan.md`. The current
> Phase 1 work established runtime truth and structural honesty; the later
> Phase 1.x revisit will establish richer internal subsystem legibility,
> neural seam placement, interface rigor, and data-engine contribution
> clarity.

#### Design-pattern adoption (no code dependency, implementable now)

These require no Habitat code import. They are contract/architecture patterns:

- Habitat-style simulator/task separation for backend execution discipline
  (source: `habitat.core.env` + `habitat.core.embodied_task.EmbodiedTask`
  — clean sim→task→measurement→episode protocol)
  → **implementation target**: runtime adapter (`SimulatorBackend` /
  `TaskDefinition` contract pair in Sim/Synth WM)
- **landed locally on 2026-05-18**: explicit
  `SimulatorBackendContractState` / `TaskDefinitionContractState` pair plus
  first live `TaskMeasurementReceipt`
- Articulated embodiment + sensor config schema borrowing
  (source: `habitat.articulated_agents` + `habitat_sim.physics` URDF/SDF)
  → **implementation target**: provider contract (`EmbodimentConfig` in
  Embodiment WM, URDF-based)
- Measure/Measurement registry patterns for sim branch evaluation
  (source: `habitat.core.embodied_task.Measure` — UUID-keyed, dependency-
  ordered, reset/update protocol; directly adoptable for our
  `TaskMeasurementSurface` receipt family)
  → **implementation target**: measurement/receipt family
  (`TaskMeasurementSurface` in Sim/Synth WM)
- Semantic scene hierarchy / object-region-scene decomposition
  (source: `habitat_sim.scene` + `habitat_sim.metadata`)
  → **implementation target**: utility module (`SceneGraph` types, already
  partially in Perception WM `state.py`)
- SensorSuite composition pattern for provider registry
  (source: `habitat.core.simulator.SensorSuite` — registry-composed sensor
  suite with per-step observation updates)
  → **implementation target**: provider contract (`PerceptionProviderRegistry`
  pattern, already partially landed)

#### Real code / provider adoption candidates (requires evaluation)

These may involve selective code borrowing or provider integration:

- Camera geometry / view-warp / calibration utilities for sim-real consistency
  (source: Habitat-Lab view-transform-warp tutorial — concrete K-matrix
  construction, extrinsic transform chains, depth unprojection/projection.
  CPU-only math, no GPU dependency. Implementable now.)
  → **implementation target**: utility module
  (`camera_geometry.py` in `src/world_model/sim_synth_physics/utils/`)
- **landed locally on 2026-05-18**: CPU-only pinhole intrinsics, transform
  composition / inversion, depth unprojection, and point projection helpers
- Vectorized runtime/eval patterns for batch sim execution
  (source: Habitat-Lab `VectorEnv`, evaluate patterns)
  → **implementation target**: runtime adapter
  (`VectorizedSimRunner` in Sim/Synth WM)
- **landed locally on 2026-05-18**: an honest `sequential_batch` local facade;
  provider-season parallelism remains future work rather than a current claim
- Interactive play / benchmark harness patterns
  (source: Habitat-Lab interactive play script + `habitat_baselines` eval)
  → **implementation target**: benchmark harness
  (`scripts/` benchmark/play harness)
- Differentiable physics provider
  (candidate: JaxSim — JAX-native, URDF/SDF, reduced-coordinate dynamics,
  CPU/GPU/TPU. Reference: `ami-iit/jaxsim`.)
  → **implementation target**: provider contract
  (`DifferentiablePhysicsProvider` in Sim/Synth WM)

Each requires a scoping evaluation: what to borrow, what to adapt, what to
ignore. Do not bulk-import.

#### Integrated Sim / Synth / Physics provider and boundary pass

This reopenable Phase 1.x lane should be treated as one coherent Sim / Synth /
Physics update, not as separate Newton / Unreal / Habitat / surrogate waves.

Also preserve a narrow SIM1 rule:

- borrow lane-specific tactics from SIM1 only as concrete provider/runtime
  tactics under our WM doctrine
- do not let SIM1's deformable-manipulation data engine become the ontology or
  architecture template for the Sim / Synth / Physics WM
- the useful tactics are:
  - real runnable provider lanes with explicit bring-up and smoke discipline
  - `generate -> smooth -> replay -> filter` branch materialization
  - explicit reject filtering and typed reject receipts
  - replay-validity / task-consistency checks for branch-validity and
    sim-real-gap evaluation
  - optional render/materialization as a downstream lane, not the WM's
    sovereign center

**Near-term doctrine (docs/contracts first)**:

- make provider-family placement explicit in the Sim / Synth / Physics WM
  owning doc:
  - Newton → Subsystems 1/6/8/9 as physics-fidelity +
    differentiable-calibration provider family
  - UnrealRoboticsLab → Subsystems 3/7 mediated by 1 as paired
    backend+render/materialization provider
  - WinDiNet-like lanes → Subsystem 8 with Subsystems 4/5/10 consumers as
    surrogate-physics / inverse-design sublane
  - Habitat-style scene priors, asset composition, and layout generation →
    Subsystems 2/3/4 rather than a separate ontology
- reserve common typed surfaces across this family:
  `TaskMeasurementSurface`, `SceneHierarchyState`,
  `DifferentiablePhysicsProviderState`, `SurrogatePhysicsProviderState`,
  `SimRealGapReceipt`, `BackendMismatchReceipt`,
  `SurrogatePhysicsReceipt`, `SurrogateCalibrationReceipt`
- **landed locally on 2026-05-18**: the first CPU-local Phase 1.x tranche now
  implements that shared surface / receipt family in the live
  `sim_synth_physics` compiler/runtime path; differentiable and surrogate lanes
  remain explicitly `contract_reserved` unless future evidence says otherwise
- **landed locally on 2026-05-18**: scene hierarchy is now consumed by
  branch/materialization planning, and task/transfer/surrogate receipts are
  harvested into backend-selector and branch-planner training rows
- **landed locally on 2026-05-18**: branch-validity / reject-filter receipts
  now emit through runtime artifacts and become training-visible in
  backend-selector aggregates plus branch-planner per-branch metadata
- **landed locally on 2026-05-18**: geometry-backed sensor-alignment receipts
  now validate camera intrinsics/extrinsics metadata via CPU-local projection
  round trips and expose alignment posture to runtime and training rows
- **landed locally on 2026-05-18**: replay-validity / task-consistency
  receipts now filter branch outcomes for training using outcome, task,
  transfer, branch-admission, and sensor-alignment evidence
- **landed locally on 2026-05-19**: runtime receipt manifests now consolidate
  emitted receipt families, artifact paths, missing-required checks, optional
  provider/runtime absences, and training-row linkage into one harvestable
  audit surface
- **landed locally on 2026-05-19**: runtime receipt-manifest validation now
  compares manifest receipt-family counts against harvested bundle contents and
  exposes mismatch diagnostics to training rows
- **landed locally on 2026-05-19**: training-admissibility gates now label
  backend-selector and branch-planner rows as positive training, negative
  supervision, or diagnostic-only based on manifest validation, target source,
  branch validity, and replay validity
- **landed locally on 2026-05-19**: trainer entrypoints now enforce the
  admissibility boundary for current positive-only helper losses: positive and
  explicit legacy dataset rows train, while negative-supervision and
  diagnostic-only rows are counted and excluded until negative losses exist
- **landed locally on 2026-05-19**: excluded negative-supervision and
  diagnostic rows now persist as trainer sidecar JSONL artifacts and Regal
  registered artifacts, preserving them for later reject/utility heads
- **landed locally on 2026-05-19**: backend-selector and branch-planner helpers
  now train bounded reject-probability heads from negative-supervision sidecars;
  rejected learned payloads remain trace-visible but do not override heuristic
  runtime decisions
- **landed locally on 2026-05-19**: trainer runtime packages now include a
  `phase1x_training_gate_v1` promotion/precondition surface, requiring selected
  rows to match admissibility summaries, diagnostic rows to stay out of
  promotion, runtime receipt manifests to validate cleanly, and
  negative-supervision sidecars to be backed by reject-head training
- **landed locally on 2026-05-19**: compiled Sim / Synth / Physics world states
  now carry `phase1x_subsystem_index_v1`, mapping the 10 Phase 1.x subsystems
  to typed state surfaces, receipt families, learned/reserved seams, promotion
  gates, provider families, runtime artifact refs, and honest external blockers
- **landed locally on 2026-05-19**: backend-selector and branch-planner
  trainer rows now preserve that subsystem index ID, coverage summary,
  subsystem IDs, ownership rule, structural status, and honest blocker class so
  subsystem legibility survives into training and promotion artifacts
- **landed locally on 2026-05-19**: Holosoma local smoke now has an explicit
  preflight mode, auto-selected local policy ref, ONNX deploy/action smoke, and
  a reproducible no-pip path-shim bootstrap; the current local proof loads
  `actor_obs [1, 100]` and emits finite `action [1, 29]`, while WM runtime
  routing remains shadow/fallback unless `ROBOTICS_VP_ENABLE_HOLOSOMA_RUNTIME=1`
  is set and simulated episode evidence remains provider/GPU debt
- **closure assessment recorded on 2026-05-19**:
  `docs/economic_world_model/phase1x_closure_assessment.md` classifies the
  remaining Phase 1.x findings as Category B external provider / GPU / asset /
  calibration / benchmark / native-runtime blockers, with Category A = `0` and
  unresolved Category C = `0`. Treat this as locally structural closure-ready
  pending owner / Claude review, not as provider readiness or promotion
  evidence.
- keep the constitutional rule: providers may span multiple subsystems, but
  never own WM truth

**Later Phase 1.x evaluation / bring-up**:

- evaluate Newton as a solver-backed fidelity and differentiable calibration
  lane alongside MuJoCo-style backends
- evaluate UnrealRoboticsLab first as a MuJoCo-paired high-fidelity
  materialization provider, while keeping the contract pairable toward
  Isaac-family physics later
- evaluate Isaac Lab, Isaac Sim, Isaac Gym, Unitree Isaac Gym, and Holosoma
  against the same provider contracts instead of bespoke one-off wiring
- keep Habitat-style room/scene/layout learnings inside the provider/back-end
  pattern rather than creating a Habitat-native environment layer

**Later calibration / transfer / deployment follow-on**:

- keep WinDiNet-like lanes bounded to surrogate preview, branch scoring,
  inverse-design proposal, and later surrogate-vs-backend calibration
- route sim-real gap, backend mismatch, and calibration receipts into the
  Sim↔Embodiment transfer boundary rather than into transport-specific doctrine
- Phase 4A and 4E should later make timing, control-rate, degraded-mode, and
  communication consequences real
- Phase 6 should later learn over those typed transfer/calibration surfaces,
  not invent them from scratch

#### UE5 / Unreal provider-family posture

See `docs/economic_world_model/doctrine_unreal_ue5_provider_posture.md`.

UE5 / Unreal should be treated as a major Sim / Synth / Physics provider
family for:

- scene materialization
- branch rendering
- digital-twin generation and ingest
- synthetic-data generation
- sensor simulation
- randomization and PCG-driven variation

This remains doctrine under the current topology, not a topology rewrite and
not a reason to divert current branch priorities into "build Unreal now."

**Adoption staging rule**:

1. docs and contracts first
2. provider/runtime lanes second
3. branch/materialization/sensor receipts third
4. headless/cloud recurring generation later
5. deployment twin / HIL / teleop / industrial usage last

**Near-term Phase 1.x doctrine**:

- reserve UE5 / Unreal as a provider family spanning scene/materialization,
  render, sensors, digital twins, randomization, and middleware-connected
  runtime
- keep hybrid backend posture explicit: UE for realism, rendering, sensors,
  digital twins, and large-scene variation; MuJoCo / Bullet / Newton /
  AGX-like lanes where they are the more honest contact/control choice
- keep the anti-overfit rule explicit: UE5 is not the stack ontology, not the
  controller owner, and not the owner of scene truth

**Later Phase 1.x evaluation / bring-up**:

- UE5 headless branch rendering and synthetic-data generation
- UE-backed digital-twin ingest for deployment-matched regression scenes
- UE sensor-simulation lanes and timing/noise/synchronization receipts
- UE PCG / randomization lanes for clutter, occlusion, weather, and layout
  variation
- UE hybrid-backend evaluation paired with MuJoCo-style or Newton-style lanes

**Later follow-on**:

- Phase 4A / 4B / 4E reserve UE middleware, sensor-simulation, and companion
  bridge lanes as concrete enablers once those phases open
- Phase 8 weekly GPU operations can later include headless UE generation,
  validation, and industrial-twin runs once the corresponding provider/runtime
  lanes and receipts are real

#### Future sim-to-online stabilization doctrine

See `docs/economic_world_model/doctrine_sim_to_online_stabilization.md`.

This is a future training/execution doctrine under the current Ixion topology,
not a topology driver and not a reason to recenter the repo on SAC or on
real-robot finetuning.

**What should be reserved now at the doc/contract level**:

- replay-mixture doctrine that distinguishes retained simulation data, retained
  prior real data, and new online data rather than flattening them into one
  buffer story
- typed warm-start buffer policies for cases where prior data is not retained
- transfer-instability receipts rather than vague "transfer worked / failed"
- checkpoint-completeness requirements for resume and deployment restart
- actor/critic asymmetry as a future training-manifest field, not a repo-wide
  algorithm decree
- asynchronous episodic real-hardware update discipline keyed to receipts,
  windows, and replay export

**Where it belongs later**:

- Sim / Synth / Physics owns simulation-side transfer assumptions, replay
  provenance, transfer-risk summaries, and training-worthiness under transfer
  instability
- Embodiment / Actuation owns deployment-side realized drift, action-feasibility
  degradation, remap posture, and recovery posture
- Economic WM later consumes transfer cost/stability/yield evidence, but does
  not become the owner of transfer mechanics

#### GR00T / VIRAL / DoorMan borrowing track

See
`docs/economic_world_model/doctrine_groot_visualsim2real_borrowings.md`.

GR00T-VisualSim2Real should be treated as an adoption track for
training/eval/config/promotion discipline, not topology. Its useful lessons
are composable experiment specs, privileged teacher to deployable student
seams, domain-randomization provenance, dataset-reset curricula,
eval/checkpoint/export gates, callback/measurement emitters, and run-ledger
discipline.

This does not move the repo away from Phase 2. Phase 2 Perception /
Grounding remains the active implementation center. Phase 2 may borrow
deployable observation discipline now: camera observation bundles,
egocentric sensor profiles, extrinsics-randomization receipts,
observation-delay or degraded-observation surfaces, and visual augmentation
provenance. The current Phase 2 ordering remains intact: embodiment-facing
usefulness first, provider truth and receipt emission second, cheap bounded
prototype-train proof-of-life only where useful, and no promotion claims
until benchmark/GPU evidence exists.

GR00T is most relevant to the later Phase 1.x Sim / Synth / Physics return
and Phase 3 Embodiment prep. After Phase 2, the roadmap returns to
Sim/Synth/Physics Phase 1.x because additional provider-family,
transfer-boundary, runtime/materialization, and run-manifest obligations were
added after Phase 1 structural closure. That later return should use GR00T
patterns to make randomized-sim, teacher/student training, dataset-reset,
eval/export, and Sim-to-Embodiment transfer receipts legible under Ixion's
existing WM topology.

#### GPU/runtime-blocked adoption items (requires hardware/assets)

These are real but blocked by external resources:

- real Isaac Sim + Habitat-style scene loading with Unitree URDF assets
- real GPU-backed vectorized sim with Habitat-style batch rendering
- real sensor-suite config with Isaac camera sensors and egocentric views
- real benchmark harnesses with GPU-backed physics evaluation

These should sit in the Phase 1 external backlog alongside existing
Isaac/Holosoma items.

#### Anti-overfit rule

Borrow patterns, borrow contract ideas, borrow utilities, maybe borrow
selective code. **Do not** inherit Habitat's ontology or make Habitat the
master environment abstraction. This repo owns its WM boundaries.

### Embodiment / Actuation WM (Phase 3) — future preparatory

> Full specification: `docs/actuation_embodiment_world_model.md`

The Embodiment / Actuation WM is now explicitly specified as a six-subsystem
canonical WM with typed interfaces, external-architecture borrowing logic,
and a concrete timescale hierarchy. The existing embryonic artifacts
(EmbodimentProfile_v1 through CalibrationTargets_v1) are identified as early
typed outputs of those subsystems. When Phase 3 begins, the first tranches
should follow the repo's established pattern: typed state contracts → shadow
compiler → downstream consumers → receipt emission → bounded neural seams.

As of 2026-05-18, Phase 2 is structurally closure-ready on the audited internal
surfaces. As of 2026-05-19, the local Phase 1.x Sim / Synth / Physics return leg
is also recorded as structurally closure-ready in
`docs/economic_world_model/phase1x_closure_assessment.md`: remaining blockers
are external provider / GPU / asset / calibration / benchmark /
native-runtime evidence, not known internal structure. Phase 3 should therefore
begin with spec/canonical-state preparation after owner / Claude acceptance,
not with provider bring-up or hardware claims. The first prep artifact is
`docs/economic_world_model/phase3_embodiment_actuation_spec_prep.md`.

GR00T / Isaac / VisualSim2Real borrowing status is summarized in
`docs/economic_world_model/groot_inspired_functionality_status.md`: these are
pattern sources for teacher/student, deploy-shaped observations, sim-to-real
receipts, randomization/reset curricula, and promotion gates, not replacement
ontologies.

Useful contract ideas from Habitat include articulated-agent config discipline,
sensor schema, and action-space normalization. Additionally, external
architecture borrowing from V-JEPA 2 (local dynamics), LeRobot/ACT (action
chunking, inverse dynamics), Diffusion Policy (multimodal proposals), Isaac
Lab (embodiment-aware sim), and TD-MPC2 (bounded latent planning) should
enter as bounded, promotion-gated, receipt-emitting seams inside the
Embodiment WM — not as replacement ontologies. See
`docs/actuation_embodiment_world_model.md` for the borrowing discipline.

The same Phase 3 prep should now explicitly include the Sim↔Embodiment transfer
boundary: Sim / Synth / Physics owns simulation assumptions, backend-mismatch
state, and transfer/calibration receipts; Embodiment owns remap/retarget,
capability filtering, and deployment-side adaptation. Later transport work
should consume those typed surfaces rather than becoming their first owner.

**Landed locally on 2026-05-20**: Phase 3.1-3.3 now has an additive native
package at `src/world_model/embodiment_actuation/`. The package defines the
canonical state family, full initial receipt family, provider/runtime
contracts, promotion posture, a shadow compiler, and first shadow consumers for
Sim/Synth transfer, Perception feedback, Runtime adapter validation, and
Economic receipt ingest. This is structural/shadow runtime work only:
`authority_level` remains `none`, provider/GPU/hardware claims remain external,
and no GR00T/Isaac ontology is imported.

**Landed locally on 2026-05-20**: Phase 3.4 now has CPU-runnable neural
scaffolding and training-row materialization. `morphology.py` turns Unitree G1
public/local repo evidence into typed morphology profiles and evidence
receipts; `neural_seams.py` provides bounded local-dynamics, inverse-retarget,
action-proposal, and drift/calibration modules; `training_corpus.py` emits
seam rows and non-promotional manifests; and
`scripts/smoke_test_embodiment_phase34.py` proves finite local forward passes.
Promotion remains blocked on GPU/provider/benchmark plus latency/watchdog and
hardware drift evidence.

**Landed locally on 2026-05-20**: Phase 3 sidecars are now emitted by the normal
local embodiment runner. Each processed episode can now carry canonical
Embodiment / Actuation state, receipt, consumer, morphology, Phase 3.4 training
row, training-manifest, and neural-architecture-manifest refs into metadata and
datapack surfaces. `neural_architectures.py` also adds CPU-forward JEPA-style,
ACT-style, Diffusion Policy-style, and topology-contrastive architecture
scaffolds. These are training-ready shapes and contracts only: promotion remains
blocked on GPU training, provider/runtime evaluation, benchmark evidence,
latency/watchdog evidence, and demotion-path tests.

Phase 3 local closure is recorded in
`docs/economic_world_model/phase3_closure_assessment.md`: under the current
no-GPU/provider constraint, the remaining Phase 3 blockers are external
evidence gates rather than known local substrate debt.

That embodiment-side transfer truth should later include realized post-transfer
drift, action-feasibility degradation, control-rate / latency mismatch, and
local recovery posture as typed replay/exportable receipts rather than as
ambient deployment notes.

UE-backed simulation assumptions and sensor/timing profiles should later help
precondition humanoid/mobile readiness here, but they should remain inputs into
Embodiment-local truth rather than replacements for remap, action-feasibility,
latency-divergence, or degradation ownership.

### Economic WM and Transport / Meta-Node — not a Habitat concern

Almost nothing should be copied from Habitat for these layers. The Economic
WM consumes lower-WM receipts/substrates; the transport/meta-node layers
are not Habitat's problem domain.

### Cross-WM resource surfaces

Resource surfaces (provider/dataset/task/deployment-resource) should not be
treated as a Perception-only pattern. They are a universal lower-WM pattern
that later feeds:

- Sim / Synth / Physics backend/runtime/fidelity choice
- Embodiment action-feasibility / latency posture
- Economic allocation and governance

Each lower WM should independently own its version of these typed surfaces,
with receipts, following the same pattern established in Phase 2 but with
WM-specific semantics.

## Anti-Heuristic-Without-Neuralization Rule

Structural preparation (receipts, promotion gates, provider contracts,
dimensional markers) is **necessary but not sufficient**.

Bounded neural seams should begin existing as real codepaths as soon as the
substrate is honest enough. The branch should not drift into a comfort zone
where heuristics get cleaner, receipts get richer, and neuralization keeps
being postponed as "later." That would violate the core anti-heuristic
posture: heuristic fusion, heuristic bridge scoring, and heuristic graph
construction are **transitional priors**, not acceptable resting places.

Concretely:

- Heuristic fusion, heuristic bridge scoring, and heuristic scene graph
  construction are acceptable only as transitional priors behind typed
  `disabled|auto|required` promotion posture.
- Once receipt emission and promotion-gate wiring are landed for a subsystem,
  the **next** step should be to implement the first bounded neural seam for
  that subsystem, not to indefinitely refine the heuristic path.
- Early bounded neural seams should exist even if initially:
  - tiny learned modules (100K-500K params)
  - partially trained or heuristic-initialized
  - disabled/auto by default
  - benchmark-gated before promotion
  - provider-blocked in some environments
- The key neural seams that should begin earliest:
  - evidence fusion (set transformer / perceiver over provider tokens)
  - annotation bridge (projection heads for labeling quality)
  - provider calibration/projection heads (DINOv2→d=128, SAM mask→token)
- This rule applies to every lower WM, not just Perception.

This does not mean "implement GPU-scale models immediately." It means the
compiler/runtime path should increasingly prepare for and then execute
immediate bounded neural substitution. Heuristic paths are the prior;
learned paths are the target.

The correct test: after each structural tranche lands, ask "is the next step
more structure, or is it time for a real neural seam?" If the substrate is
honest (receipts emitting, promotion gates wired, provider truth compiled),
then the answer should be "neural seam" — even a tiny one.

## Embodiment-Facing Subsystem Usefulness Rule

A lower WM is not "real enough" merely because it compiles and feeds one or
two narrow consumers. The Perception / Grounding WM in particular should not
be judged as a real subsystem until it is demonstrably useful for
**embodiment-facing action relevance**.

This means:

- the next important proof of Perception subsystem usefulness is
  embodiment-facing affordance / action-relevance shadow consumption
- this is where Perception stops being descriptive (scene state, labels,
  semantic tags) and starts becoming obviously relevant to actual G1-operable
  loop behavior
- an embodiment-facing consumer validates that the canonical scene graph and
  bridge outputs carry enough structured information for bodily feasibility,
  grasp planning, and action-space filtering

Without embodiment-facing consumption, the Perception WM risks becoming a
well-instrumented semantic shell: structurally complete, receipt-emitting,
but not actually useful for robot control.

The same principle applies later to other WMs:

- Sim / Synth / Physics is not real until its outputs demonstrably affect
  training data selection and replay
- Embodiment / Actuation is not real until its outputs demonstrably affect
  motor control and safety gating
- Economic WM is not real until its outputs demonstrably affect lower-WM
  allocation and resource routing

## Future Economic WM Posture

The future upstream Economic WM (not the current shadow economic control
plane) should be framed as:

> **a neuralizable, scalable, typed allocator-governor — the canonical world
> model of productive flow, constraint, dissipation, and allocative
> opportunity across the robotics stack.**

It is **not**: a scalar reward head, a dashboard, a thin weight-picker, a PnL
tracker, or a mother-latent.

**Staging guard:** everything below is **future doctrine** for when the
Economic WM is built. It is not a call to divert current Phase 2 work.
The sequencing remains: lower WMs first → typed receipt/state surfaces →
Economic WM neuralization phases later.

It is also **not the sole sovereign governor** of the stack. The stack's telos
is governed robot control under multiple non-collapsible realities (physics,
safety, anti-reward-hacking, deployment truth, embodiment limits, coordination
integrity). The Economic WM is a first-class allocative contributor within a
broader superposed governance field — the meta-regal-node WM composes it with
other domain-governance nodes under regime-sensitive Pareto, veto, and
admissibility logic. See `doctrine_meta_regal_node_wm.md`.

### Multi-timescale design

The Economic WM must be explicitly multi-timescale:

- **Fast**: local routing, queue dispatch, sim-branch selection, local shaping
- **Meso**: task-family allocation, episode budget routing, exploration quotas
- **Slow / near-adiabatic**: objective structure, topology-level priors,
  global constraint manifolds, deployment-trust invariants

Slow variables must not swing violently with local noise.

### Upward vs downward transport

Upward transport (lower WMs → Economic WM): receipt transport, abstraction,
bottleneck aggregation, macro-state estimation.

Downward transport (Economic WM → lower WMs): allocative fields, shaping
fields, budget envelopes, admissible Pareto slices, governance guidance.

These are **asymmetric objects**, not the same tensor reversed. Upward carries
raw receipts; downward carries compiled allocative fields with regime context.
Phase 6 should preserve that asymmetry with per-WM exporter and receiver
transformers around the isomorphic transport bridge rather than one universal
decoder.

### Internal decomposition

1. **Economic State Estimator**: consumes lower-WM receipts → `EconomicState`
   + `EconomicRegime` + `BottleneckMap` + `SlowManifoldProjection`.
   Architecture: switching SSMs / regime-aware sequence models (confirmed:
   DS3M for long-range regime detection, RED-SDS for explicit-duration regime
   persistence). Emits slow manifold projection enforcing adiabatic
   separation: fast receipts projected before affecting macro state.
2. **Economic Dynamics Model**: forecasts state evolution under candidate
   allocations, **conditioned on slow manifold state** (regime, constraint
   manifold, macro-pressure vector). Architecture: regime-switching rollouts,
   typed receipt forecasts, differentiable-physics coupling where appropriate.
   No-thrashing rule: fast dynamics don't feed back into slow manifold
   without explicit gating.
3. **Economic Allocator / Compiler**: converts state + forecasts into
   structured allocative fields. The Pareto allocator should be
   **distributional, regime-aware, and execution-aware** (confirmed:
   DPMORL distributional Pareto for frontier slices, risk budgeting via
   augmented Lagrangian for `ShadowPriceField` per resource constraint).
   Emits `ParetoFrontierSlice` objects with tail-risk metadata.
4. **Economic Governance / Reciprocity Layer**: reciprocal coupling to lower
   WMs. Bottom-up: receipts. Top-down: shaping fields, budget envelopes,
   admissible operating regions, `PersistenceAnnotation` hysteresis.

### Quant-inspired algorithmic imports

Borrow as **algorithmic patterns**, not worldview:

- coherent risk measures (CVaR-style tail sensitivity)
- distributional Pareto policies (preserve uncertainty over return vectors)
- regime switching (latent operating-mode inference, persistent regimes)
- risk budgeting / shadow-price-like signals for scarce resources
- stress testing / scenario-conditioned allocation
- execution-cost awareness (friction in `DissipationField`, not penalty hack)

### Staged neuralization

1. typed non-neural scaffolding (receipts, regimes, surfaces)
2. neural state estimation (switching SSMs over receipt streams)
3. neural dynamics / counterfactual forecasting
4. neural Pareto allocator (distributional frontier engine)
5. local shaping compilers
6. meta-node integration (only after lower layers stable)

Neuralization follows typed ontology and transport design, not precedes it.

### Optional discrete allocation solver sublane (future, post Stage B/C/D)

Once Stage B–D neuralization structures are real (neural state estimation,
dynamics/forecasting, distributional Pareto allocator), evaluate an **optional
finite-set receding-horizon allocation solver lane** inside the Economic
Allocator / Compiler for discrete combinatorial routing problems
(compute-budget routing, sim-budget dispatch, replay-slice selection, queue
relief under finite action sets). Motivated by QUBO / Ising-formulated MPPI
(arXiv:2512.15533). This lane is downstream of estimated state + forecast +
frontier/risk structure. It does not replace the primary estimator, dynamics,
or distributional Pareto allocator doctrine. Treat probabilistic-computing /
Ising hardware as a future optional backend, not a current dependency. See
`doctrine_economic_wm_future_architecture.md` § Discrete Receding-Horizon
Allocation Solver for full placement and limitations.

Full doctrine: `docs/economic_world_model/doctrine_economic_wm_future_architecture.md`

## Future Meta-Regal-Node Superposition WM Posture

Above the Economic WM sits the meta-regal-node superposition / control WM.
Its job is to compose multiple domain-governance nodes under regime-sensitive
logic.

### Why the Economic WM is not sovereign

The stack's telos is not "optimize economics." It is governed robot control
under multiple non-collapsible realities. If the Economic WM becomes the sole
governor, physical/safety/deployment reality gets treated as subordinate
constraints to an economic worldview. That is dangerous for a control stack.

### Three governance levels

1. **Subsystem/local WM**: perception, embodiment, sim/synth. Local truths.
2. **Domain governance**: economic allocation, anti-reward-hacking,
   plausibility/geometry, deployment truth, safety, data value, coordination.
3. **Meta-governance**: the WM that composes the governance nodes themselves.

### Two kinds of Pareto

- **Intra-domain** (Economic WM): throughput vs energy vs wear vs compute.
- **Inter-domain** (meta-regal-node): economics vs anti-reward-hacking vs
  plausibility vs safety. More fundamental: governs whether intra-domain
  optimization can be trusted.

### Governance pluralism principle

The architecture preserves pluralism at the governance layer while allowing
strong specialization below. No single domain ontology can silently redefine
the others. The composition is regime-sensitive, confidence-aware, and typed.

### Staging

The meta-regal-node WM is built last: after lower WMs, Economic WM, and
transport bridges are mature. Individual domain-governance nodes must be
neuralized before the meta-layer can learn to compose them.

Full doctrine: `docs/economic_world_model/doctrine_meta_regal_node_wm.md`

## Phase 2 Provider / Dataset / Resource Surface Rule

Phase 2 should explicitly separate, under WM-owned typed state:

- `DatasetSurfaceState`
- `ProviderSurfaceState`
- `TaskMeasurementSurface`
- `DeploymentResourceSurface`

With deployment/resource detail named early:

- `ComputeEnvelopeState`
- `InferenceCapacityState`
- `BatteryState`
- `ThermalState`

And typed receipts:

- `ProviderAvailabilityReceipt`
- `InferenceHeadroomReceipt`
- `DeploymentResourceReceipt`

These are lower-WM surfaces first. They should inform Perception runtime truth
now, later Sim / Synth / Physics backend/fidelity/materialization truth,
later Embodiment latency/action-feasibility truth, and only after that become
allocatable Economic-WM objects.

Current branch status:

- the typed Phase 2 surface family is now live in `src/world_model/perception_grounding/`
- the first functional compiler path is landed
- the first shadow consumers are landed in:
  - `src/world_model/sim_synth_physics/adapters/semantic_inputs.py`
  - `src/vla/rollout_labeler.py`
- external data adapter pathway is now real (`lerobot_perception_adapter.py`)
- seam training infrastructure is landed (losses, data loaders, trainer, benchmarks)
- **Landed 2026-04-04**: embodiment-facing shadow consumer (`embodiment_shadow_consumer.py`), full 8-receipt-type emission in compiler, 7-subsystem internal decomposition codified
- **Landed 2026-04-08**: bounded annotation-export successor lane with `AnnotationBridgeProjectionSeam`, annotation-export evaluation, trainer wiring, and provisional-evidence gating
- **Landed 2026-04-08**: persisted benchmark-evidence artifact contract with provenance-bearing annotation export and stricter promotion gating for graph transformer / annotation bridge / provider-adapter seams
- **Landed 2026-05-11**: routine persisted annotation-export benchmark-evidence emitter (`benchmark_evidence_emitter.py` plus `scripts/emit_perception_annotation_benchmark_evidence.py`) for scene-graph transformer and annotation-bridge evidence artifacts
- **Landed 2026-05-11**: receipt-backed runtime provider-token selection for benchmark object tokens: successful `vision_backbone_projection` and `vjepa_temporal_alignment` provider invocations can feed annotation/export evidence; skipped or failed invocations remain heuristic/provisional instead of claiming provider-backed truth
- **Landed 2026-05-11**: provider-adapter benchmark-evidence emitter (`scripts/emit_perception_provider_adapter_benchmark_evidence.py`) for `vision_backbone_projection`, `sam_calibration`, `depth_metric_calibration`, and `vjepa_temporal_alignment`; it aggregates `ProviderInvocationReceipt` artifacts, links optional training manifests / metric reports, and keeps receipt-only evidence provisional by default
- **Landed 2026-05-11**: local CPU perception proof-of-life artifacts now cover both `EvidenceFusionSeam` (`scripts/smoke_test_perception_seam_training.py`) and `VJEPATemporalAlignmentSeam` (`scripts/smoke_test_vjepa_temporal_seam.py`); both scripts emit persistent checkpoints, metric reports, provisional benchmark evidence, training runtime manifests, and loss-decrease proof; both support synthetic, DROID-shaped mock LeRobot adapter paths, and local LeRobot-like JSON/JSONL row bundles; this is provisional plumbing evidence only, not promotion evidence
- **Landed 2026-05-18**: `vision_backbone_projection` now has a first-class local training-data and benchmark lane (`VisionBackboneProjectionSample` / `Batch` / `Dataset`, synthetic sample generator, loader factory, and `VisionBackboneProjectionBenchmark`) so the first promotion-chain seam no longer depends on ad hoc batch plumbing before future DINOv2/SigLIP provider runs
- **Landed 2026-05-18**: local CPU proof-of-life artifacts now cover `VisionBackboneProjectionSeam` as well (`scripts/smoke_test_vision_backbone_projection_seam.py`), emitting the same checkpoint / metric-report / provisional benchmark-evidence / manifest / receipt family while keeping promotion explicitly held
- **Landed 2026-05-18**: LeRobot adapter parity for the first promotion-chain seam; `vision_backbone_projection` now accepts the same `mock_lerobot_droid` and `local_lerobot_rows` intake grammar as the evidence-fusion and temporal proof lanes, with explicit `camera_slot_proxy` labels rather than false object-ID claims
- **Landed 2026-05-18**: live compiler emission for the WM-native semantic bridge family; `SemanticBridgeReceipt` is now emitted for `sim_synth`, `embodiment`, `annotation`, and `economic` bridges and returned by `compile_perception_grounding_with_receipts(...)`
- 3 shadow consumers now wired: SimSynth, Annotation/VLA, Embodiment
- full audited receipt family live: ProviderAvailability, EvidenceFusion, ProviderInvocation, SemanticBridge, GroundingCalibration, InferenceHeadroom, DeploymentResource, TemporalGrounding, PerceptionContribution
- with GPU/provider bring-up intentionally deferred, the local Phase 2 pocket is
  now exhausted enough to leave cleanly:
  1. **Tiny real-data proof, only if cheap later** — use the existing local
     row-bundle intake path only when a real export is already available; do not
     turn dataset acquisition into a side quest
  2. **Provider-specific non-provisional metric reports, later** — produce these
     only once real provider executions exist
  3. **Hold off on pretending promotion is near** — structural closure does not
     imply promotion credibility; dependency-ordered benchmark evidence is still
     required (`vision_backbone_projection` → `scene_graph_transformer` →
     `annotation_bridge_projection` → provider calibrators)
- current implementation center after this final local pocket:
  **Phase 1.x Sim / Synth / Physics return leg**

**Caution**: Do not let the adapter layer become another comfort zone. It exists to serve seam training, downstream usefulness, and promotion honesty—not to become its own mini-project.

This borrows the useful layering pattern from Habitat-style stacks:

1. dataset/world inventory
2. provider/runtime
3. task/measurement
4. deployment/resource posture

But keeps canonical WM ownership, typed receipts, and economics-aware resource
state native to this repo instead of flattening everything into one env object.

## WM Section Decomposition Standard

A future WM section is not considered "ready" if it remains only a box-and-arrow
description. The 9-point WM Section Readiness Standard in
`multi_wm_architecture_plan.md` establishes the minimum for all WM roadmap
sections:

1. canonical mission / ownership
2. internal subsystem decomposition
3. typed state / receipt / interface surfaces
4. neural structure candidates by subsystem
5. hyperparameter / promotion / governance shaping
6. topological placement
7. timescale hierarchy
8. robostack / G1 contribution
9. phase sequencing honesty

The Economic WM doctrine and Embodiment / Actuation WM plan exemplify this
standard. The Sim / Synth / Physics WM is now explicitly marked for a
Phase 1.x decomposition revisit during the GPU/runtime era, applying this
same standard. The Perception / Grounding WM Phase 2 work is increasingly
approaching this standard through its provider surfaces, seam training
infrastructure, and typed receipts.

## Program Calendar

Assumed dates for this roadmap:

- March 27, 2026 through August 31, 2026: plumbing-first execution window
- September 1, 2026: first serious multi-WM training runs start
- July 2027: pre-purchase readiness window for a Unitree G1 program step
- September 30, 2027: target for sustainably autonomous G1 operation with recurring data collection and bounded self-improvement
- after Phase 7 matures: Phase 8 production-loop runtime, weekly GPU operations, backlog exhaustion, then latency/inference hardening

This means the roadmap should be read in two major phases.

Phase A: March 27, 2026 to August 31, 2026

- finish the structural plumbing from the current multi-WM plan
- make lower-WM state ownership, receipts, runtime-package seams, provider truth, and replay/training exports real
- reserve the economic-WM ingestion contracts and later transport insertion points now so September training does not trigger another architecture rewrite
- do not spend this window pretending benchmark or corpus gaps are architecture work if the missing piece is actually data, GPUs, assets, or calibration

Must be true by August 31, 2026:

- the embodiment target hierarchy is explicit across docs and scaffolds: bipedal whole-body humanoid control is the primary standard, stable-base mobile manipulation is the safety fallback/degraded-mode posture, and fixed-base tabletop remains only curriculum/regression evidence
- sim / synth / physics WM plumbing is structurally real
- for Phase 1 backend closure, the repo-root host scan must emit explicit local usable-profile / install / preflight truth for Isaac/Unitree and Holosoma instead of leaving Category B runtime reality implicit across many artifacts
- and that same blocked truth must survive launch, work-order, and trainer-facing exports instead of being softened after the scan/runtime-binding layer
- once public local Unitree repos/assets are on disk, Phase 1 should derive the remaining honest non-GPU asset truth from them before calling the rest external; on the current branch that reduces the explicit Unitree asset blockers to whole-body latency and watchdog contracts rather than generic robot-description/joint-map/joint-limit gaps
- perception / grounding WM plumbing is structurally real
- embodiment / actuation WM plumbing is structurally real enough to start training and later Unitree integration without another contract purge, including posture tags for bipedal whole-body, stable-base fallback, and fixed-base curriculum artifacts
- economic-WM ingestion over lower-WM receipts is structurally real
- WM-transport seams are reserved at the contract level even if transport training itself still comes later

Phase B: September 1, 2026 to September 30, 2027

- shift effort toward training, provider bring-up, calibration, benchmark accumulation, and Unitree-specific integration
- keep architecture churn low; new structure should only land when it closes a proven training or deployment blocker
- use the lower-WM receipts to train helper packages, predictive lanes, and later economic-WM consolidation honestly

Recommended sub-phases after training starts:

- September 1, 2026 through December 31, 2026: first lower-WM training season, receipt accumulation, provider bring-up, and replay/corpus expansion
- when real loop runs start, treat sim-to-online stabilization as a first-class
  benchmark/evidence problem: replay-mixture policy, warm-start policy,
  checkpoint completeness, update schedule, and transfer stability should be
  tracked explicitly rather than tuned informally
- January 1, 2027 through March 31, 2027: benchmark and calibration season, especially for perception temporal state, whole-body sim execution, backend truth, and promotion gates
- April 1, 2027 through June 30, 2027: pre-purchase hardening for Unitree G1 readiness, including safety-adjacent middleware, embodiment contracts, whole-body replay, and hardware-facing adapter discipline
- when those later deployment-enabler phases open, UE middleware and sensor
  simulation should be treated as concrete Phase 4A / 4B / 4E candidate lanes:
  real-time control-loop separation still owns the rate split, the sensor-fusion
  shim still owns fusion consequences, and companion middleware still owns the
  transport consequences
- July 2027: purchase/integration window where the honest blockers should mostly be hardware, data, calibration, GPUs, and benchmark evidence rather than missing canonical plumbing
- July 1, 2027 through August 31, 2027: turn first-hardware bring-up into a recurring robot loop with replay capture, degraded-mode handling, operator/recovery traces, and training-export discipline
- By September 30, 2027: the G1 control loop should be sustainably autonomous enough to keep running, collecting data, and improving without recurrent architecture churn

Phase C: Production-loop runtime after the major WM layers are mature

- operationalize the full stack as a recurring weekly GPU / Runpod program
- interleave:
  - external dataset aggregation
  - loop runs
  - corpus/receipt export
  - training
  - fine-tuning
  - benchmarking
  - promotion / redeployment
- keep burning down the explicit run/training/provider backlogs until there are no important external or internal model lanes sitting unused
- only after that backlog burn-down becomes routine should latency, inference throughput, and deployment cost become the dominant optimization target
- later Phase 8 execution may include headless UE-based synthetic generation,
  sensor-validation runs, digital-twin regression runs, and industrial/workcell
  twin validation loops once those lanes have honest provider/runtime receipts

Weekly operating model from September 1, 2026 onward:

- treat the post-September program as a weekly A100 cycle, not an ad hoc training queue
- schedule work sub-module by sub-module inside each WM
- within each weekly cycle, prefer:
  - loop runs
  - corpus/receipt export
  - training runs
  - fine-tuning
- do not skip directly to fine-tuning when the corresponding loop-run or provider-truth lane is still weak
- do not spread one weekly A100 budget across too many WMs at once; finish a concrete sub-module tranche, record receipts and gates, then move to the next sub-module
- after hardware arrives, keep the same cadence on robot-origin loops: run, capture receipts, export corpus, train/fine-tune, redeploy bounded changes, then repeat
- for real-hardware adaptation, prefer asynchronous episodic training windows
  with explicit transfer receipts over pretending that per-step synchronous
  optimization is the governing doctrine

Mechanics-first advancement rule:

- do not call a WM "ready" because it can emit logs, summaries, or canonical-looking state in isolation
- do not call fixed-base tabletop or stable-base fallback evidence humanoid-ready unless the artifact explicitly names its transfer boundary and the bipedal whole-body gaps that remain
- a WM only counts as structurally real when it owns a bounded closed loop with real ingress, real execution or honest execution gating, replay/training exports, and all relevant downstream consumers for the future hardware-ready loop wired
- neuralization remains part of scalable mechanics rather than a separate luxury layer; learned control, prediction, adaptation, and routing should be made load-bearing as soon as the surrounding subsystem can carry them honestly
- keep the scalable mechanics substrate ahead of non-load-bearing learned claims; if a phase is still missing executors, adapters, safety gates, replay exports, or live downstream consumers, that phase is still structurally incomplete even if training code already exists
- do not let a higher WM treat a lower WM as canonical until the lower WM has crossed bounded runtime authority and is affecting the relevant downstream loop rather than merely being logged

Env/sim and neural scaffold sequencing rule:

- Phase 3.5 should refit env and sim layout around posture-tagged families: `bipedal_whole_body_*` as the primary readiness lane, `stable_base_mobile_manipulator_*` as fallback/degraded-mode safety lane, and `fixed_base_tabletop_*` as curriculum/regression lane
- The local Phase 3.5 return artifact is `phase35_humanoid_capacity_env_refit.md`: it records G1/R1-class planning capacity bands, onboard/companion/battery assumptions, humanoid observation/action/schema deltas, posture-tagged env taxonomy, Unitree sim integration target, and benchmark taxonomy without claiming sim, hardware, training, or promotion evidence
- As of 2026-05-24, Phase 3.5 also has a typed local scaffold in `src/world_model/humanoid_readiness/phase35.py` plus `scripts/economic_world_model/prepare_phase35_humanoid_capacity_env_refit.py`; the current artifact run emits `capacity_band_count=5`, `schema_delta_count=10`, `env_taxonomy_count=3`, `benchmark_target_count=7`, and `local_structural_refit_complete=true`, with all training/provider/hardware/live/promotion gates denied
- Phase 3.5 now includes the canonical bipedal chassis layer in `src/world_model/embodiment_actuation/bipedal_chassis.py` plus `scripts/economic_world_model/prepare_phase35_bipedal_chassis_scaffold.py`; the current local artifact emits `controlled_joint_count=29`, `frame_count=22`, `joint_limit_envelope_count=29`, `support_state_count=3`, `balance_receipt_count=3`, `whole_body_observation_schema_present=true`, and `whole_body_action_schema_present=true`, while keeping `hardware_calibrated_limits=false`, `ready_for_unitree_runtime=false`, and `promotion_eligible=false`
- Phase 3.5 now includes a no-GPU/no-hardware bipedal readiness audit in `src/world_model/embodiment_actuation/bipedal_readiness.py` plus `scripts/economic_world_model/audit_phase35_bipedal_readiness.py`; the current local artifact emits `local_asset_ingestion_contract_present=true`, `asset_parse_receipt_count=1`, `kinematic_validators_present=true`, `joint_vector_validation_receipt_count=2`, `balance_geometry_report_count=3`, `whole_body_replay_row_count=3`, and `phase35_no_gpu_no_hardware_prepared=true`, while keeping `real_asset_parsed=false` in the default no-asset run and keeping Unitree runtime, training, live policy control, reward mutation, and promotion denied
- Phase 4 now includes a dry-run downstream-controller scaffold in `src/world_model/humanoid_readiness/downstream_controller.py` plus `scripts/economic_world_model/prepare_phase4_downstream_controller_scaffold.py`; it emits Unitree ROS2 / SDK2-shaped bridge targets, G1Pilot-style fallback bridge targets, controller modes, proposals, command frames, safety receipts, invocations, and controller receipts while keeping ROS2 publish, Unitree SDK2 writes, G1Pilot invocation, hardware dispatch, live control, reward mutation, training, and promotion denied
- lower-WM neural scaffolds should target bipedal whole-body complexity first where they represent body/contact/control directly: whole-body state encoders, support/contact/balance predictors, loco-manipulation action heads, inverse-dynamics/retargeting lanes, fallback selectors, and latency/watchdog/resource predictors
- stable-base fallback classifiers may become learned seams, but they should emit veto/defer/recovery/operator-handoff receipts rather than silently redefining the primary standard

Compute and battery sequencing rule:

- Phase 3 should emit canonical body-adjacent compute / battery / thermal / placement state and receipts
- Phase 3.5 should audit whether those contracts and the submodule capacities behind them are realistic for G1/R1-class onboard and companion deployment
- The local Phase 4 sweep artifact is `phase4_local_deployment_enabler_sweep.md`: it limits current work to non-hardware 4A control-loop separation contracts, 4E companion-compute/comms contracts, 4F operator/teleop/recovery contracts, and 4B/4C/4D schema/runbook/interface stubs while deferring full closure to live streams, control interfaces, and hardware/sim runtime evidence
- As of 2026-05-24, Phase 4 also has a typed local scaffold in `src/world_model/humanoid_readiness/phase4.py` plus `scripts/economic_world_model/prepare_phase4_deployment_enabler_sweep.py`; the current artifact run emits `contract_surface_count=15`, `stub_surface_count=3`, `local_non_hardware_scaffold_complete=true`, and `ready_for_phase65_local_meta_nodes=true`, with full deployment closure still blocked on live streams, control interfaces, timing/jitter traces, companion middleware, operator/recovery traces, and hardware or honest sim runtime evidence
- Phase 4A and 4E should make the runtime consequences real:
  - control-rate changes
  - offload decisions
  - communication QoS
  - degraded-mode behavior
- Phase 5 should turn those lower-WM resource receipts into allocatable economic budget objects
- only later should transport and meta-node layers learn over those allocations as higher-order governance objects
- The local Phase 6.5 scaffold note is `phase65_local_meta_node_neuralization.md`: it names `MetaNodeState`, trajectory/intervention receipts, counterfactual targets, robustness reports, and denied-by-default promotion gates before any Phase 7 control-WM authority exists
- As of 2026-05-24, Phase 6.5 has typed local scaffolds in `src/world_model/humanoid_readiness/phase65.py` and `src/world_model/humanoid_readiness/closure.py` plus `scripts/economic_world_model/prepare_phase65_meta_node_neuralization.py` and `scripts/economic_world_model/audit_phase35_4_65_local_closure.py`; the current artifact run emits `node_state_count=5`, `counterfactual_target_count=5`, `robustness_report_count=5`, `promotion_gate_count=5`, `local_meta_node_scaffold_complete=true`, and `ready_for_phase7_scaffold=true`, while `phase7_authority_granted=false`

Suggested RL staging for those resources:

1. lower-WM prediction and calibration of compute availability, battery depletion, thermal posture, latency impact, and action feasibility
2. bounded helper allocation under local constraints:
   - backend choice
   - fidelity choice
   - simulation / diffusion spend
   - defer versus execute
3. economic-WM cross-resource tradeoffs
4. later meta-node Pareto policy over economic allocation receipts

Suggested weekly WM order for the first training season:

1. Sim / synth / physics WM sub-modules
2. Perception / grounding WM sub-modules
3. Embodiment / actuation WM sub-modules
4. Economic-WM ingestion and consolidation over lower-WM outputs
5. Local meta-node neuralization and later meta-node superposition / control lanes over the stabilized lower-WM and economic-WM outputs
6. Only later, ontology-mediated WM transport where adjacent WMs are already stable enough to justify the bridge budget
7. After the layered stack is mature enough, transition into Phase C production-loop runtime discipline with weekly GPU operations and backlog exhaustion

## Workstream Summary

| Week / stage | Goal | Classification | Primary targets |
| --- | --- | --- | --- |
| Week 0 | Establish roadmap docs, nightly audit substrate, skill, automation, and the first packet/embodiment scaffolds | docs-only + scaffolding-only | `docs/economic_world_model/*`, `codex_skills/economic-world-model-roadmap/`, `scripts/economic_world_model/*`, `src/runtime/packets.py`, `src/embodiment/registry.py` |
| Week 1 | Emit canonical runtime packets as additive sidecars in shadow runtime and replay | additive_wiring | `src/shadow_runtime/control_plane.py`, `src/replay/ingest.py`, packet tests |
| Week 2 | Normalize embodiment, action, and observation contracts | scaffolding-only + additive_wiring | `src/runtime/action_adapter_v2.py`, `src/runtime/observation_adapter_v2.py`, `src/embodiment/registry.py`, `src/inference/demo_policy.py` |
| Week 3 | Create the temporal event spine and governance trace spec/code path | docs-only + scaffolding-only | `src/runtime/event_spine.py`, `src/governance/trace.py`, replay/ontology sidecars |
| Week 4 | Build the evidence bus and belief-state layer, plus teacher trace sidecars | scaffolding-only + additive_wiring | `src/evidence/*`, `src/orchestrator/semantic_fusion_runner.py`, `src/vla/rollout_labeler.py` |
| Week 4.5 | Reopen the world-model package with governed video-state and geometry-first hypothesis scaffolding | additive_wiring + scaffolding-only | `src/world_model/governed_video_world_model.py`, `scripts/run_stage1_pipeline.py`, `src/diffusion/video_diffusion_runtime.py`, `src/vision/scene_ir_tracker/io/scene_tracks_runner.py` |
| Week 5 | Add dense economic supervision and counterfactual evaluation traces | scaffolding-only + additive_wiring | `src/economics/counterfactual_eval.py`, `src/economics/value_targets.py`, `src/economics/value_ledger.py`, datapack schema sidecars |
| Week 6 | Train and evaluate a packet-native learned control-plane scaffold | additive_wiring + behavior-changing behind flags | `src/control_plane/*`, `src/phase_h/*`, `src/orchestrator/*`, training harnesses |
| Week 6.5 | Ground real-video plumbing with 4D reconstruction and explicit teacher runtime contracts | scaffolding-only + additive_wiring | `src/vision/reconstruction/four_d_reconstruction.py`, `src/vla/teacher_runtime.py`, `scripts/run_stage1_pipeline.py`, `src/vision/scene_ir_tracker/io/scene_tracks_runner.py` |
| Week 6.75 | Emit governed video supervision bundles and economic targets from candidate futures | scaffolding-only + additive_wiring | `src/world_model/governed_video_supervision.py`, `src/economics/counterfactual_eval.py`, `src/economics/value_targets.py`, `src/governance/trace.py` |
| Week 7+ | Add dataset bridges and opt-in integration passes | docs-only + additive_wiring | `src/dataset_bridges/*`, `src/valuation/portable_datapacks.py`, replay export/import |

## Week 0 - Current Pass

- Goal: Create a practical roadmap, a repeatable nightly audit loop, real local/cloud Codex actuation paths, and the first low-risk middleware scaffolds.
- Rationale: Without these, the repo cannot progress nightly in a disciplined way and future work will keep re-litigating architecture.
- Target modules/files:
  - `docs/economic_world_model/architecture_gap_analysis.md`
  - `docs/economic_world_model/roadmap.md`
  - `docs/economic_world_model/progress_log.md`
  - `docs/economic_world_model/nightly_audit.md`
  - `docs/economic_world_model/codex_skill.md`
  - `docs/economic_world_model/implementation_notes.md`
  - `docs/economic_world_model/AUTOMATION_SPEC.md`
  - `codex_skills/economic-world-model-roadmap/SKILL.md`
  - `scripts/economic_world_model/nightly_audit.py`
  - `scripts/economic_world_model/update_status_issue.py`
  - `scripts/economic_world_model/run_nightly_codex_task.sh`
  - `.github/workflows/economic-world-model-nightly.yml`
  - `src/runtime/packets.py`
  - `src/embodiment/registry.py`
- Deliverables:
  - Grounded gap analysis mapped to the seven preconditions.
  - Multi-week staged implementation plan.
  - Repo-local skill and nightly audit docs.
  - GitHub Actions nightly audit/update path plus actual Codex execution path when credentials are present.
  - First `RuntimePacket` / `ContractPacket` scaffold.
  - First `EmbodimentRegistry` / `CapabilityProfile` scaffold.
- Acceptance tests / verification commands:
  - `./scripts/agent/verify.sh`
  - `python3 -m compileall src scripts/economic_world_model -q`
  - `python3 -m pytest -q tests/test_runtime_packets.py tests/embodiment/test_registry.py`
- Dependencies / blockers:
  - Local actual execution requires Codex CLI plus `CODEX_API_KEY` or `OPENAI_API_KEY`.
  - GitHub/cloud actual execution requires `CODEX_API_KEY` secret.
  - App automation still requires manual UI creation.
- Do not touch:
  - the stable baseline checkpoint or legacy baseline world-model math
  - `checkpoints/stable_world_model.pt`
  - `src/controllers/synthetic_weight_controller.py` core logic
  - Trust-net, `w_econ`, and lambda controller equations
- Classification: `docs-only` and `scaffolding-only`

## Week 1 - Packet Sidecar Emission

- Goal: Emit `RuntimePacket` sidecars anywhere the repo already emits objective/econ/constraint artifacts.
- Rationale: The canonical packet only matters once live paths produce it.
- Target modules/files:
  - `src/shadow_runtime/control_plane.py`
  - `src/replay/ingest.py`
  - `src/objectives/runtime_builder.py`
  - `src/replay/schema.py`
  - `tests/test_shadow_econ_runner.py`
  - `tests/test_replay_schema.py`
  - `tests/test_runtime_packets.py`
- Deliverables:
  - Packet emission in the shadow economic control plane as an additive artifact.
  - Packet sidecars in replay ingest for episode and window records.
  - No reward-path or deployment behavior changes.
- Acceptance tests / verification commands:
  - `python3 -m compileall src -q`
  - `python3 -m pytest -q tests/test_runtime_packets.py tests/test_shadow_econ_runner.py tests/test_replay_schema.py`
- Dependencies / blockers:
  - Requires Week 0 packet scaffold.
  - Existing artifact writers must stay backward compatible.
- Do not touch:
  - Baseline SAC/PPO actor logic
  - Frozen Phase B math
- Classification: `additive_wiring`

## Week 2 - Embodiment and Adapter Normalization

- Goal: Normalize embodiment metadata plus canonical observation/action schema refs.
- Rationale: The future top layer should route against a capability model, not hardcoded backend assumptions.
- Target modules/files:
  - `src/runtime/action_adapter_v2.py`
  - `src/runtime/observation_adapter_v2.py`
  - `src/embodiment/registry.py`
  - `src/inference/demo_policy.py`
  - `src/motor_backend/*`
  - `src/ingestion/*`
  - `tests/test_runtime_packets.py`
  - `tests/embodiment/test_registry.py`
- Deliverables:
  - `ActionAdapterV2` and `ObservationAdapterV2` schema contracts with timing semantics and provenance.
  - Embodiment registry entries for current workcell/shadow/demo backends.
  - Translator refs rather than hardwired backend assumptions.
- Acceptance tests / verification commands:
  - `python3 -m compileall src -q`
  - `python3 -m pytest -q tests/embodiment/test_registry.py tests/test_runtime_packets.py tests/test_wrapper_golden_integration.py`
- Dependencies / blockers:
  - Requires Week 0 registry scaffold.
  - May need lightweight fixture data for non-workcell backends.
- Do not touch:
  - Existing reward math
  - Phase B controllers
- Classification: `scaffolding-only` plus small `additive_wiring`

## Week 3 - Event Spine and Governance Trace

- Goal: Add dense temporal event logging for decisions, vetoes, replans, price ticks, and realized outcomes.
- Rationale: A future world model cannot be trained honestly from episode summaries alone.
- Target modules/files:
  - `src/runtime/event_spine.py`
  - `src/governance/trace.py`
  - `src/ontology/store.py`
  - `src/logging/episode_logger.py`
  - `src/replay/schema.py`
  - `tests/test_replay_schema.py`
  - `tests/test_value_ledger.py`
- Deliverables:
  - Spec and initial code path for append-only event rows.
  - `GovernanceTrace` sidecar schema for veto reasons and rule provenance.
  - Mapping from event rows to replay and ledger refs.
- Acceptance tests / verification commands:
  - `python3 -m compileall src -q`
  - `python3 -m pytest -q tests/test_replay_schema.py tests/test_value_ledger.py tests/test_regal_gates.py`
- Dependencies / blockers:
  - Easier after Week 1 packet sidecars exist.
  - No live fleet bus yet, so initial path remains shadow/replay-first.
- Do not touch:
  - Frozen world model and valuation math
- Classification: `docs-only` first, then `scaffolding-only`

## Week 4 - Evidence Bus and TeacherTrace Sidecars

- Goal: Lift specialist outputs into a first-class evidence publication layer.
- Rationale: Current evidence artifacts exist, but they are still component-local and not router-ready.
- Target modules/files:
  - `src/evidence/bus.py`
  - `src/evidence/belief_state.py`
  - `src/evidence/teacher_trace.py`
  - `src/orchestrator/semantic_fusion_runner.py`
  - `src/vla/rollout_labeler.py`
  - `src/vision/map_first_supervision/*`
  - `tests/test_semantic_fusion_mvp.py`
  - `tests/test_vla_semantic_evidence.py`
- Deliverables:
  - Common evidence publication envelope with confidence, disagreement, and validity windows.
  - External VLA/FM traces stored as teacher sidecars, not native truth.
  - Belief-state snapshots suitable for later router/planner training.
- Acceptance tests / verification commands:
  - `python3 -m compileall src -q`
  - `python3 -m pytest -q tests/test_semantic_fusion_mvp.py tests/test_vla_semantic_evidence.py tests/test_semantic_fusion_orchestrator_smoke.py`
- Dependencies / blockers:
  - Easier after Week 3 event/gov schemas exist.
  - Real calibration data still absent; keep confidence logic heuristic and explicit.
- Do not touch:
  - Existing semantic fusion behavior unless behind flags
- Classification: `scaffolding-only` plus small `additive_wiring`

## Week 4.5 - Governed Video-World-Model Preconditions

- Goal: Move the repo from diffusion-first video placeholders toward an evidence-first, geometry-first video-state loop.
- Rationale: A future video world model should be supervised by the same contract/evidence/governance substrate as the broader economic world model. Rendering should stay downstream of state and plausibility.
- Target modules/files:
  - `src/world_model/governed_video_world_model.py`
  - `src/diffusion/video_diffusion_runtime.py`
  - `src/diffusion/real_video_diffusion_stub.py`
  - `scripts/run_stage1_pipeline.py`
  - `src/orchestrator/semantic_fusion_runner.py`
  - `src/vision/scene_ir_tracker/io/scene_tracks_runner.py`
  - `src/vla/openvla_controller.py`
  - `src/vla/semantic_evidence.py`
- Deliverables:
  - `GovernedVideoWorldModel` service that consumes belief-state/evidence-first inputs and proposes candidate futures before any rendering step.
  - Stage-1 sidecars for `EvidenceBus`, `BeliefState`, governed video state, and ranked hypotheses.
  - Diffusion runtime/provider contract that plans governed hypotheses first and records real-or-unavailable provider truth instead of silently normalizing stub behavior.
  - Geometry/plausibility gating in the live Stage-1 loop so governed hypotheses are judged by `RegalGenPlausibilityNode` before datapacks are admitted.
  - Configurable SceneTracks stub usage rather than hardwired `use_stub_adapters=True`.
  - Clear separation between frozen stable checkpoint baseline and new advisory video-state scaffolding.
- Acceptance tests / verification commands:
  - `python3 -m compileall src scripts/run_stage1_pipeline.py tests -q`
  - `python3 -m pytest -q tests/test_evidence_bus.py tests/test_runtime_adapters_v2.py tests/test_governed_video_world_model.py tests/test_rollout_labeler.py tests/test_semantic_fusion_orchestrator_smoke.py tests/test_stage1_pipeline_governed.py`
  - `python3 scripts/run_stage1_pipeline.py --num-videos 1 --proposals-per-video 1 --output-dir /tmp/stage1_governed_smoke`
- Dependencies / blockers:
  - Real 4D reconstruction, real SceneTracks adapters, and non-stub teacher models still remain future work.
  - Real video diffusion materialization and GGDS/LDM bring-up now sit in `scripts/FOUNDATION_MODEL_BRINGUP_BACKLOG.json`; those are no longer hidden behind stub-default posture.
  - Non-training GPU runtime/materialization smokes now sit in `scripts/NON_TRAINING_GPU_RUN_BACKLOG.json`, and the Phase-1 sim/synth WM now emits runtime bundles and launch specs so runtime bring-up can be burned down separately from trainer migration or fine-tuning work.
  - The Isaac/Unitree lane now also expects WM-owned executable-adapter requests over those launch specs, so deployment mode, robot variant, asset/calibration posture, and output expectations remain typed loop artifacts instead of living only in shell commands.
  - The next maturity rung after that request is now explicit too: the WM carries a typed executable-adapter consumer so it can name whether a request is being handed to a local bridge or only an external launch consumer, without overstating actual hardware/runtime execution.
  - The next rung after the consumer is now explicit as well: the WM emits typed adapter-execution mediation and an adapter receipt, so request, consumer, execution mediation, launch, and harvested outcome remain distinct runtime truths rather than collapsing into one launch-status bit.
  - The next rung after execution mediation is now explicit too: the WM emits a typed adapter-realization surface, so it can say whether the lane is concretely realized through a local backend-factory handoff or only through an external launch delegate before any actual runtime success claim.
  - The next rung after realization is now explicit too: the WM emits a typed local backend-factory invocation/result surface, so explicit local adapter materialization is no longer hidden inside a direct backend-factory jump.
  - The next closure rung after local materialization is now explicit too: the WM carries backend-specific deployment contracts and upstream runtime packs, so runtime profile / policy-bank / asset-pack / telemetry-pack readiness is canonical WM state rather than an implicit reading of repo roots.
  - The next closure rung after runtime packs is now explicit too: the WM emits backend-specific runtime bindings that select mode-relevant policy, motion, retargeting, launch, and target surfaces from those packs before executable-adapter requests are built.
  - Local Holosoma concrete execution now uses that runtime-binding layer to clear irrelevant external-pack blockers when the selected local mode is already satisfied by explicit policy refs or motion datapacks, so local train/eval paths are no longer falsely blocked by missing repo-root or launch-surface state.
  - Local concrete runtime execution is now judged by explicit runtime-output harvest rather than only launch receipts: Isaac/Unitree and Holosoma concrete runtime paths emit outcome receipts with `harvest_mode=local_runtime_execution`, and those receipts preserve policy / dataset / metrics surface readiness instead of collapsing back into launch-shaped status.
  - The local Isaac/Unitree bridge path also now filters out stale external-pack blockers when a real local runtime bridge, policy ref, SDK root, and asset root are already present, so Phase 1 no longer overstates “blocked” status on a concretely executable local lane.
  - Upstream runtime roots/checkpoints/assets are now also represented with more concrete evidence:
    - runtime profiles carry candidate counts plus repo git metadata when locally available
    - policy contracts carry primary checkpoint / deploy-config / runtime-report refs rather than only candidate lists
    - Isaac runtime packs now distinguish declared asset refs from locally verified asset refs, so “manifest says it exists” is no longer the only asset readiness signal
    - Holosoma runtime packs now preserve which motion sources actually exist locally versus which are only named
  - Runtime bindings now also carry selected-surface host-preflight truth, so the branch can distinguish contract-ready lanes from locally verified lanes without inventing a new runtime rung:
    - Isaac selected launch/policy/target/asset refs now emit explicit verification status
    - Holosoma selected policy/retargeting refs and selected existing motion sources now emit explicit verification status
    - launch/work-order/training surfaces preserve that host-preflight truth instead of flattening it back into pack-level readiness
  - Shadow execution now consumes those selected runtime-binding surfaces directly when building Isaac shadow env-configs and Holosoma shadow work orders, so Phase 1 no longer leaves the shadow lane as a context-shaped bypass around the deeper runtime ladder.
  - Runtime layouts and upstream packs now also need profile-level install/preflight evidence, and the binding layer must resolve that evidence against the actually selected profile rather than the pack’s preferred profile alone; otherwise local motion-train or local bridge lanes inherit false repo/install blockers from the wrong profile.
  - Runtime targets themselves now also need selected-target install-shape verification, and the binding/work-order/training path must preserve verified-vs-partial selected targets so an empty SDK, asset, motion, or retargeting root does not look ready just because the path exists.
  - The runtime target/layout/policy path can now also autodiscover common local upstream repo roots and fall back to those roots when looking for policy/checkpoint banks, so a host with real local clones does not need every env var pre-wired before Phase 1 can consume that evidence honestly.
  - Policy-root and preferred-profile selection now also need to consume that stronger truth: an explicit-but-empty policy root must not outrank a discovered checkpoint-bearing root, and install-blocked profiles must not count as deployable just because the repo root exists.
  - The runtime-layout contract itself now also needs to expose `usable_profiles`, and downstream bundle/bridge/work-order/training surfaces need to preserve it; otherwise the stronger truth gets rebuilt or blurred downstream even after the selection logic is fixed.
  - Concrete checkpoint / deploy-config / runtime-report selection also needs to prefer the best verified local artifact over earlier missing candidates, and that chosen-ref source/candidate-evidence truth needs to survive into work-order and training surfaces; otherwise seemingly concrete primary refs still hide candidate-ordering ambiguity.
  - Harvested runtime outputs also need to be validated against those selected runtime refs, and that validation truth needs to survive into work-order and training surfaces; otherwise “runtime outputs harvested” still quietly mixes correct execution with wrong-artifact harvests.
  - That validation truth also needs to change downstream completion posture and trainer target-source selection; otherwise mismatch remains “known but ignored.”
  - Tier 3.4 inferential scoring and Tier 3.5 humanoid-target randomization/calibration should also be directly re-audited and then removed from the unresolved bucket, so the remaining closure debate stays focused on actual external runtime/install/GPU blockers rather than stale unclassified surfaces.
  - Those runtime bindings now survive into runtime bundles, launch specs, work orders, loop summaries, and trainer-facing corpus rows, so downstream consumers no longer have to infer “why this lane is still blocked” from pack metadata or launch status alone.
  - Branch-planner traces now also carry explicit control truth through planning and trainer exports, so a learned branch payload can remain visible without pretending it actually controlled branch generation when the heuristic path retained authority.
  - Holosoma now follows the same request -> consumer -> execution -> realization ladder as Isaac/Unitree, so one backend is not allowed to remain a structurally looser special case while the other gets typed runtime truth.
  - No training loop lands here; the service is advisory and structural.
- Do not touch:
  - `checkpoints/world_model_stable_canonical.pt`
  - Trust-net, `w_econ`, or lambda controller math
  - Reward-path scalarization rules
- Classification: `additive_wiring` plus `scaffolding-only`

## Week 5 - Dense Economic Supervision

- Goal: Create trainable local targets for counterfactual economic decisions.
- Rationale: Price ticks and ledgers exist, but the repo still lacks explicit local supervision for adapt/data-route decisions.
- Target modules/files:
  - `src/economics/counterfactual_eval.py`
  - `src/economics/value_targets.py`
  - `src/economics/value_ledger.py`
  - `src/valuation/datapack_schema.py`
  - `tests/test_value_ledger.py`
  - new counterfactual tests
- Deliverables:
  - `CounterfactualEval` traces for adapt vs no-op / collect-data vs no-op / route A vs B.
  - Dense supervision target sidecars tied to packets and event rows.
  - Governance-aware value targets that can supervise successor video-state loops without touching the stable Phase B baseline.
  - No change to frozen valuation or reward equations.
- Acceptance tests / verification commands:
  - `python3 -m compileall src -q`
  - `python3 -m pytest -q tests/test_value_ledger.py tests/test_objective_econ_functor_consistency.py tests/test_regal_uses_econ_tensor.py`
- Dependencies / blockers:
  - Better after Week 1 through Week 4 provide packet, event, and evidence joins.
- Do not touch:
  - `src/valuation/trust_net.py`
  - `src/valuation/w_econ_lattice.py`
  - lambda controller math
- Classification: `scaffolding-only` plus targeted `additive_wiring`

## Week 6 - Learned Control-Plane Scaffolds

- Goal: Train and evaluate a higher-level router/planner/critic before any sovereign learned world model.
- Rationale: This is the right place to learn which specialist to invoke, when to adapt, and when to collect more data.
- Target modules/files:
  - `src/control_plane/router.py`
  - `src/control_plane/planner.py`
  - `src/control_plane/critic.py`
  - `src/phase_h/*`
  - `src/orchestrator/*`
  - training harnesses and eval scripts
- Deliverables:
  - Packet-native router inputs.
  - Evaluation against `CounterfactualEval`, `BeliefState`, and governance traces.
  - Strictly flag-gated behavior changes.
- Acceptance tests / verification commands:
  - `python3 -m compileall src -q`
  - targeted pytest for new control-plane modules
  - relevant shadow smoke tests
- Dependencies / blockers:
  - Requires Weeks 1 through 5 to produce truthful packets, traces, and local targets.
- Do not touch:
  - Baseline online RL unless the new control-plane path is opt-in
- Classification: `behavior-changing` only behind explicit flags

## Week 6.5 - Real Video Grounding and Teacher Runtime Hardening

- Goal: Replace fallback-heavy real-video plumbing with camera-grounded reconstruction sidecars and explicit teacher runtime contracts.
- Rationale: Production-grade video-world-model preconditions require honest camera geometry, explicit adapter failure semantics, and replayable teacher envelopes before any serious learned predictor can be trusted.
- Target modules/files:
  - `src/vision/reconstruction/four_d_reconstruction.py`
  - `src/vla/teacher_runtime.py`
  - `src/vision/scene_ir_tracker/io/scene_tracks_runner.py`
  - `src/vla/openvla_controller.py`
  - `src/ingestion/x_humanoid_adapter.py`
  - `scripts/run_stage1_pipeline.py`
- Deliverables:
  - D4RT-style reconstruction sidecars with camera calibration, confidence windows, and provenance.
  - Teacher runtime envelopes with explicit fallback metadata, latency, and adapter contract refs.
  - Real-video stage wiring that prefers grounded reconstruction and non-stub adapters before heuristic fallback.
  - Stage-1 outputs that carry reconstruction refs, calibration refs, and teacher-runtime refs alongside governed video-state artifacts, and rollout-labeling outputs that always emit teacher contract/action sidecars even when the teacher is disabled or unavailable.
  - Runner and ingestion paths where stub usage is explicit, inspectable, and never silently treated as production truth.
- Concrete implementation order:
  - Wire `src/vision/reconstruction/four_d_reconstruction.py` into `scripts/run_stage1_pipeline.py` so each processed video emits a reconstruction sidecar.
  - Thread reconstruction refs and calibration metadata through `src/vision/scene_ir_tracker/io/scene_tracks_runner.py` and `src/ingestion/x_humanoid_adapter.py`.
  - Route OpenVLA and similar teacher outputs through `src/vla/teacher_runtime.py` first, then publish them into semantic evidence and teacher traces.
  - Keep fallback mode explicit in every packet, evidence record, and replayable sidecar.
- Acceptance tests / verification commands:
  - `python3 -m compileall src scripts/run_stage1_pipeline.py tests -q`
  - `python3 -m pytest -q tests/test_scene_tracks_runner.py tests/test_openvla_controller.py tests/test_stage1_pipeline_governed.py`
- Dependencies / blockers:
  - Requires Week 2 and Week 4 contracts plus Week 4.5 governed video-state scaffolding.
  - Real external models remain optional; fallback behavior must stay explicit.
- Do not touch:
  - Stable baseline checkpoint math
  - Trust-net, `w_econ`, or lambda controller math
- Classification: `scaffolding-only` plus `additive_wiring`

## Week 6.75 - Governed Video Supervision and Economic Targets

- Goal: Turn governed video hypotheses into replayable supervision artifacts for downstream economic-world-model training.
- Rationale: Branches are only useful if they emit auditable value, governance, and counterfactual traces tied back to packets and replay.
- Target modules/files:
  - `src/world_model/governed_video_supervision.py`
  - `src/economics/counterfactual_eval.py`
  - `src/economics/value_targets.py`
  - `src/governance/trace.py`
  - `src/runtime/event_spine.py`
- Deliverables:
  - Governed video supervision bundles that attach branch evaluations, governance traces, value targets, and value-ledger receipts to the video loop.
  - Counterfactual economic evaluations derived from candidate futures rather than only post hoc episode summaries.
  - Replayable receipts that join candidate futures back to `RuntimePacket`, `BeliefState`, `EvidenceBus`, and downstream datapack context.
  - Evaluation artifacts that keep geometry plausibility, regality, and economic value distinct instead of collapsing them into one scalar too early.
  - Live-loop datapack linkage so accepted datapacks carry counterfactual/value-target refs and blocked branches still retain governance/counterfactual artifacts for replay.
- Concrete implementation order:
  - Wire `src/world_model/governed_video_supervision.py` into the governed Stage-1 loop immediately after hypothesis ranking.
  - Emit `CounterfactualEval`, `ValueTargetPack`, and governance-trace sidecars for each accepted or vetoed branch.
  - Thread those refs into replay/datapack metadata so later training jobs can consume them without bespoke join logic.
  - Verify that blocked branches still emit auditable supervision artifacts rather than disappearing from the trace.
- Acceptance tests / verification commands:
  - `python3 -m compileall src tests -q`
  - `python3 -m pytest -q tests/test_value_ledger.py tests/test_runtime_packets.py tests/test_governed_video_world_model.py`
- Dependencies / blockers:
  - Requires Week 5 supervision seams and Week 6.5 grounded video plumbing.
- Do not touch:
  - Stable baseline checkpoint math
  - Reward-path scalarization rules
- Classification: `scaffolding-only` plus `additive_wiring`

## Week 7+ - Dataset Bridges and Integration Passes

- Goal: Export/import standard dataset bridges without flattening the repo's richer internal schema.
- Rationale: Standardized interchange matters, but internal economics/governance detail must remain first-class.
- Target modules/files:
  - `src/dataset_bridges/rlds_bridge.py`
  - `src/dataset_bridges/lerobot_bridge.py`
  - `src/valuation/portable_datapacks.py`
  - replay export/import glue
- Deliverables:
  - RLDS and LeRobot bridges as lossy public adapters.
  - Internal sidecars for objective/econ/governance/evidence preserved alongside bridge exports.
- Acceptance tests / verification commands:
  - `python3 -m compileall src -q`
  - new bridge tests
- Dependencies / blockers:
  - Most useful after packets, event spine, and teacher traces exist.
- Do not touch:
  - Internal schema richness. Bridges are adapters, not replacements.
- Classification: `docs-only` plus `additive_wiring`

## Explicitly Deferred

- Training a multimodal economic world model before the packet/event/evidence/governance substrate exists.
- Training a JEPA-style or DreamGen-style video model before real adapters, belief-state traces, and geometry-grounded supervision are in place, even though V-JEPA 2 is already part of the planned Phase-1 sim/synth/physics and Phase-2 perception/grounding backlog.
- Collapsing the stack into a monolithic model.
- Making external VLA/FM traces native truth.
- Rewriting the stable Phase B baseline checkpoint instead of layering additive successor modules beside it.

## Training Backlog Placement

- Learned video-state modeling belongs in the training backlog, not the immediate middleware pass.
- The first admissible training target is a governed, action-conditioned latent predictor over fused video, scene-track, geometry, embodiment, and economic context rather than a raw-pixel-only generator.
- Training should begin only after real-video grounding, teacher-runtime hardening, and governed supervision bundles are present.
- When those prerequisites are real, track V-JEPA 2 explicitly in two fine-tuning lanes rather than one vague future bucket:
  - sim / synth / physics predictive-state bring-up
  - perception / grounding temporal-state bring-up
- Prefer upstream `facebookresearch/vjepa2` bring-up and wrapper contracts over local reimplementation when the goal is faster honest subsystem progress.

## Active Autonomous Priority Order

The current local stage is the **Post-Phase-3 / Pre-Economic-WM Integration Stage**; see `docs/economic_world_model/pre_economic_wm_integration_stage.md`.

The next autonomous passes should consume the video-world-model subset in this order:

1. Deepen real-video grounding with non-stub SceneTracks adapters, richer calibration metadata, and stronger reconstruction joins in the live Stage-1 path. **Landed locally on 2026-05-20** for `reconstruction_grounding_report_v1` eligibility sidecars and camera-calibrated Stage-1 benchmark gates; broader ingestion/runner calibration joins remain next.
2. Keep teacher-runtime hardening live by pushing explicit contract/action fallback semantics through every remaining real-video or ingestion boundary, not just rollout labeling. **Landed locally on 2026-05-20** for Stage-1 teacher contract/action/trace sidecars, including unavailable fallback sidecars.
3. Export governed supervision artifacts cleanly into replay and dataset-bridge paths so Stage-1 outputs are directly trainable later. **Landed locally on 2026-05-20** for governed-video admission replay import plus RLDS/LeRobot bridge export preserving internal sidecar refs, benchmark gates, and future-training signals.
4. Test and smoke coverage that proves the new refs and sidecars persist end to end in live loops. **Landed locally on 2026-05-20** via `scripts/economic_world_model/sweep_stage1_bridge_readiness.py`, covering five manifest shapes through Stage-1, replay, RLDS, and LeRobot.
5. Refresh the training backlog and prepare the Economic WM entry preflight before considering `train_governed_video_world_model.py`. **Landed locally on 2026-05-21** via `scripts/economic_world_model/economic_wm_entry_preflight.py`; Economic WM scaffold entry is allowed, while training remains GPU/provider/evidence blocked.
6. Build the first native Economic WM scaffold artifacts from the entry preflight. **Landed locally on 2026-05-21** via `src/world_model/economic_world_model/scaffold.py` and `scripts/economic_world_model/build_economic_wm_scaffold.py`; outputs remain scaffold-only with `reward_math_mutation=false` and `promotion_eligible=false`.
7. Materialize local Economic WM replay/training rows from the scaffold and Stage-1 proposal admissions. **Landed locally on 2026-05-21** via `src/world_model/economic_world_model/training_rows.py` and `scripts/economic_world_model/materialize_economic_wm_training_rows.py`; rows preserve benchmark/shadow truth and remain `ready_for_training=false`.
8. Run a shadow-only Economic WM allocation eval over local rows. **Landed locally on 2026-05-21** via `src/world_model/economic_world_model/allocation_eval.py` and `scripts/economic_world_model/evaluate_economic_wm_shadow_allocations.py`; the current recommendation is to prepare teacher/provider evidence contracts, while GPU training remains denied.
9. Prepare Economic WM teacher/provider evidence contracts from the shadow allocation recommendation. **Landed locally on 2026-05-21** via `src/world_model/economic_world_model/evidence_contracts.py` and `scripts/economic_world_model/prepare_economic_wm_teacher_provider_contracts.py`; the contract pack names non-stub teacher, provider truth, GPU runtime, and promotion benchmark evidence requirements while keeping provider bring-up and training blocked.
10. Compile the evidence contracts into manifest-shaped provider/GPU runbook templates. **Landed locally on 2026-05-21** via `src/world_model/economic_world_model/provider_runbook.py` and `scripts/economic_world_model/compile_economic_wm_provider_runbook.py`; the runbook makes future RunPod/provider windows ledger-ready while keeping `launch_allowed=false`, `provider_bringup_ready=false`, `gpu_training_ready=false`, and `promotion_eligible=false`.
11. Validate provider/GPU runbook templates before they can be treated as stored planning artifacts. **Landed locally on 2026-05-21** via `src/world_model/economic_world_model/provider_runbook_validation.py` and `scripts/economic_world_model/validate_economic_wm_provider_runbook.py`; validation requires pending manifest stubs, empty runtime fields, guard commands for external/provider/GPU templates, and `safe_for_launch=false`.
12. Prove Economic WM rows consume canonical lower-WM state refs rather than summary-only leftovers. **Landed locally on 2026-05-21** via `src/world_model/economic_world_model/lower_wm_consumption.py` and `scripts/economic_world_model/prepare_economic_wm_lower_wm_consumption_preflight.py`; then upgraded the producer path so `scripts/run_stage1_pipeline.py` emits native Perception / Grounding, Sim / Synth / Physics, and Embodiment / Actuation state refs and `src/world_model/economic_world_model/training_rows.py` preserves them. Current fresh rows pass the preflight with `--no-compile-missing-refs`, `direct_reference_count=15`, and `compiled_reference_count=0`, while `ready_for_training=false` and `promotion_eligible=false` remain enforced.
13. Add the Economic WM neural architecture manifest for estimator, dynamics, allocator, governance, datapack-composition, and bounded discrete-allocation surfaces. **Landed locally on 2026-05-21** via `src/world_model/economic_world_model/neural_architecture_manifest.py` and `scripts/economic_world_model/build_economic_wm_neural_architecture_manifest.py`; the manifest records six future learned components, five GPU-training-required components, one local solver scaffold lane, explicit inputs/outputs/losses/gates, and no promotion or reward-math mutation.
14. Define Phase-5 resource and compute surfaces before GPU/provider execution. **Landed locally on 2026-05-21** via `src/world_model/economic_world_model/resource_surfaces.py` and `scripts/economic_world_model/prepare_economic_wm_resource_surfaces.py`; this adds capacity, latency, thermal, battery, companion-compute, degraded-mode, and queue-telemetry receipt schemas plus Economic WM ingestion slots while keeping live control, training, and promotion denied.
15. Deepen Phase-5 local prep beyond Stage-1 rows. **Landed locally on 2026-05-21** via `src/world_model/economic_world_model/phase5_local_prep.py` and `scripts/economic_world_model/prepare_economic_wm_phase5_local_prep.py`; this emits datapack-composition rows, counterfactual/value-target join rows, and temporal-window rows over canonical lower-WM refs and resource surfaces.
16. Add the non-training Economic WM trainer scaffold. **Landed locally on 2026-05-21** via `scripts/train_economic_world_model_v0.py`; it emits dataset contracts, component configs, loss definitions, CPU smoke forwards, and a denied-promotion trainer manifest with `training_executed=false` and `weights_written=false`.
17. Add shadow execution work-order loops. **Landed locally on 2026-05-21** via `src/world_model/economic_world_model/shadow_execution.py` and `scripts/economic_world_model/run_economic_wm_shadow_execution.py`; Economic WM recommendations now become advisory shadow work orders and outcome-comparison slots without controlling reward math or live policy.
18. Materialize typed Economic WM supervision records from Phase-5 counterfactual/value refs. **Landed locally on 2026-05-21** via `src/world_model/economic_world_model/supervision_substrate.py` and `scripts/economic_world_model/prepare_economic_wm_supervision_substrate.py`; current local artifacts load 5 counterfactual evals, 5 value-target packs, and 5 value-ledger receipts while keeping `ready_for_training=false` and `promotion_eligible=false`.
19. Close the local shadow outcome loop structurally. **Landed locally on 2026-05-21** via `src/world_model/economic_world_model/shadow_outcomes.py` and `scripts/economic_world_model/run_economic_wm_shadow_outcome_loop.py`; current artifacts emit 3 local structural outcome receipts and 3 joined comparisons with no hardware, provider, live-policy, training, or promotion claim.
20. Sweep lower-WM maturity behind Economic WM consumption. **Landed locally on 2026-05-21** via `src/world_model/economic_world_model/lower_wm_maturity_sweep.py` and `scripts/economic_world_model/sweep_economic_wm_lower_wm_maturity.py`; current artifacts show 15/15 canonical refs structurally ready for Phase-6 contracts and 0/15 production-ready refs, making the remaining runtime/provider/hardware maturity gap explicit.

Nightly or autonomous execution should not skip directly to training or raw model experimentation while live grounding and live supervision wiring remain incomplete.
