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
- its training role is improving WM-to-ontology-to-WM translation quality, preserving topology/causal structure/actionability, increasing synchronized loop success, and decomposing bridge-only vs downstream-only vs joint gains
- its reward should come from completed-loop/postmortem outcomes, counterfactual improvement, governance satisfaction, and downstream economic yield for the adaptor/bridge layer

Current honest state:

- today the repo mostly has operational ontology substrate/plumbing
- it does not yet have a fully neural ontology layer
- it does not yet have a full WM-transport ontology implementation
- keep the current sequencing: lower WMs first, then economic WM consolidation, then ontology-mediated WM transport

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

- sim / synth / physics WM plumbing is structurally real
- perception / grounding WM plumbing is structurally real
- embodiment / actuation WM plumbing is structurally real enough to start training and later Unitree integration without another contract purge
- economic-WM ingestion over lower-WM receipts is structurally real
- WM-transport seams are reserved at the contract level even if transport training itself still comes later

Phase B: September 1, 2026 to September 30, 2027

- shift effort toward training, provider bring-up, calibration, benchmark accumulation, and Unitree-specific integration
- keep architecture churn low; new structure should only land when it closes a proven training or deployment blocker
- use the lower-WM receipts to train helper packages, predictive lanes, and later economic-WM consolidation honestly

Recommended sub-phases after training starts:

- September 1, 2026 through December 31, 2026: first lower-WM training season, receipt accumulation, provider bring-up, and replay/corpus expansion
- January 1, 2027 through March 31, 2027: benchmark and calibration season, especially for perception temporal state, whole-body sim execution, backend truth, and promotion gates
- April 1, 2027 through June 30, 2027: pre-purchase hardening for Unitree G1 readiness, including safety-adjacent middleware, embodiment contracts, whole-body replay, and hardware-facing adapter discipline
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

Mechanics-first advancement rule:

- do not call a WM "ready" because it can emit logs, summaries, or canonical-looking state in isolation
- a WM only counts as structurally real when it owns a bounded closed loop with real ingress, real execution or honest execution gating, replay/training exports, and all relevant downstream consumers for the future hardware-ready loop wired
- neuralization remains part of scalable mechanics rather than a separate luxury layer; learned control, prediction, adaptation, and routing should be made load-bearing as soon as the surrounding subsystem can carry them honestly
- keep the scalable mechanics substrate ahead of non-load-bearing learned claims; if a phase is still missing executors, adapters, safety gates, replay exports, or live downstream consumers, that phase is still structurally incomplete even if training code already exists
- do not let a higher WM treat a lower WM as canonical until the lower WM has crossed bounded runtime authority and is affecting the relevant downstream loop rather than merely being logged

Compute and battery sequencing rule:

- Phase 3 should emit canonical body-adjacent compute / battery / thermal / placement state and receipts
- Phase 3.5 should audit whether those contracts and the submodule capacities behind them are realistic for G1/R1-class onboard and companion deployment
- Phase 4A and 4E should make the runtime consequences real:
  - control-rate changes
  - offload decisions
  - communication QoS
  - degraded-mode behavior
- Phase 5 should turn those lower-WM resource receipts into allocatable economic budget objects
- only later should transport and meta-node layers learn over those allocations as higher-order governance objects

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

The next autonomous passes should consume the video-world-model subset in this order:

1. Deepen real-video grounding with non-stub SceneTracks adapters, richer calibration metadata, and stronger reconstruction joins in the live Stage-1 path.
2. Keep teacher-runtime hardening live by pushing explicit contract/action fallback semantics through every remaining real-video or ingestion boundary, not just rollout labeling.
3. Export governed supervision artifacts cleanly into replay and dataset-bridge paths so Stage-1 outputs are directly trainable later.
4. Test and smoke coverage that proves the new refs and sidecars persist end to end in live loops.
5. Only after those land cleanly, refresh the training backlog and consider `train_governed_video_world_model.py`.

Nightly or autonomous execution should not skip directly to training or raw model experimentation while live grounding and live supervision wiring remain incomplete.
