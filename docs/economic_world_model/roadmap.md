# Economic World Model Multi-Week Roadmap

## Planning Rules

- Favor additive infrastructure over refactors.
- Keep VLA and foundation-model paths external, pluggable, and sidecar/advisory.
- Preserve objective integrity. No premature scalarization upstream of explicit contract compile.
- Do not touch frozen Phase B math or `src/world_model/`.
- Sequence work so each stage leaves behind reusable docs, schemas, tests, and automation hooks.

## Workstream Summary

| Week / stage | Goal | Classification | Primary targets |
| --- | --- | --- | --- |
| Week 0 | Establish roadmap docs, nightly audit substrate, skill, automation, and the first packet/embodiment scaffolds | docs-only + scaffolding-only | `docs/economic_world_model/*`, `codex_skills/economic-world-model-roadmap/`, `scripts/economic_world_model/*`, `src/runtime/packets.py`, `src/embodiment/registry.py` |
| Week 1 | Emit canonical runtime packets as additive sidecars in shadow runtime and replay | additive_wiring | `src/shadow_runtime/control_plane.py`, `src/replay/ingest.py`, packet tests |
| Week 2 | Normalize embodiment, action, and observation contracts | scaffolding-only + additive_wiring | `src/runtime/action_adapter_v2.py`, `src/runtime/observation_adapter_v2.py`, `src/embodiment/registry.py`, `src/inference/demo_policy.py` |
| Week 3 | Create the temporal event spine and governance trace spec/code path | docs-only + scaffolding-only | `src/runtime/event_spine.py`, `src/governance/trace.py`, replay/ontology sidecars |
| Week 4 | Build the evidence bus and belief-state layer, plus teacher trace sidecars | scaffolding-only + additive_wiring | `src/evidence/*`, `src/orchestrator/semantic_fusion_runner.py`, `src/vla/rollout_labeler.py` |
| Week 5 | Add dense economic supervision and counterfactual evaluation traces | scaffolding-only + additive_wiring | `src/economics/counterfactual_eval.py`, `src/economics/value_targets.py`, `src/economics/value_ledger.py`, datapack schema sidecars |
| Week 6 | Train and evaluate a packet-native learned control-plane scaffold | additive_wiring + behavior-changing behind flags | `src/control_plane/*`, `src/phase_h/*`, `src/orchestrator/*`, training harnesses |
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
  - `src/world_model/`
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
- Collapsing the stack into a monolithic model.
- Making external VLA/FM traces native truth.
- Rewriting frozen Phase B math.
