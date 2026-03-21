# Economic World Model Progress Log

## 2026-03-07

- Changed: created the architecture gap analysis, staged roadmap, nightly audit runbook, Codex skill docs, automation spec, repo-local skill, audit/update scripts, scheduled workflow, `RuntimePacket` scaffolding, and `EmbodimentRegistry` scaffolding.
- Verification: `./scripts/agent/verify.sh`, `python3 -m compileall src scripts/economic_world_model -q`, `python3 -m pytest -q tests/test_runtime_packets.py tests/embodiment/test_registry.py tests/test_objective_runtime_builder.py tests/test_constraint_set.py tests/test_pricing_sentinel.py tests/test_value_ledger.py`, `bash -n scripts/economic_world_model/run_nightly_codex_task.sh`, and the audit script all passed.
- Blocked: no live robot telemetry, app automations still require manual UI creation even though the app-first prompt/spec is ready, and GitHub/cloud Codex execution still requires a configured `CODEX_API_KEY` secret. The current audit sees no API key in the local environment.
- Next recommended task: wire `RuntimePacket` sidecars into `src/shadow_runtime/control_plane.py` and `src/replay/ingest.py` without changing default runtime behavior.

## 2026-03-08

- Changed: wired additive `RuntimePacket` sidecar emission into `src/shadow_runtime/control_plane.py`, persisted run-level packet sidecars under `runtime_packets.json`, and threaded packet refs/IDs into replay episode/step/window ingest metadata and provenance in `src/replay/ingest.py`.
- Verification: `python3 -m pytest -q tests/test_runtime_packets.py tests/test_shadow_econ_runner.py tests/test_replay_schema.py tests/test_replay_dataset.py`, plus `python3 -m compileall src -q`.
- Blocked: packet schemas are still shadow-workcell-derived and not yet backed by a generalized observation/action adapter layer; older shadow runs without `runtime_packets.json` still ingest in compatibility mode with no packet refs.
- Next recommended task: add an additive `EventSpine` / `DecisionLedger` sidecar for per-window decisions, vetoes, and pricing/adaptation events, then thread its refs into replay metadata beside the new packet refs.

- Changed: added additive `EventSpine` and `DecisionLedger` sidecars under `event_spine.json` and `decision_ledger.json`, emitted stable event/decision IDs tied to runtime packet IDs, contract IDs, artifact refs, and actor/critic/advisor provenance, and threaded those refs into replay episode/step/window `metadata` and `provenance` without changing replay record shapes.
- Verification: `python3 -m pytest -q tests/test_event_spine.py tests/test_shadow_econ_runner.py tests/test_replay_schema.py tests/test_replay_dataset.py tests/test_receipt_ingest.py`, plus `python3 -m compileall src -q`.
- Blocked: receipt label refs are currently empty placeholders because receipt labels are still attached downstream, and current event producers are shadow-only rather than shared with `sim_rollout` or training-run producers.
- Next recommended task: consume `event_spine.json` and `decision_ledger.json` in promotion reporting and multi-run stage movement so promotion holds, vetoes, pricing suppression, and collect-more-data decisions stop being inferred indirectly from summary fields.

## 2026-03-09

- Changed: added `ActionAdapterV2` and `ObservationAdapterV2`, broadened runtime packet builders to accept schema-producing adapters, and reopened `src/world_model/` with `GovernedVideoWorldModel` while keeping the stable checkpoint baseline intact.
- Changed: added `EvidenceBus`, `BeliefState`, and `TeacherTrace` scaffolding; wired teacher traces into `src/vla/rollout_labeler.py`; and wired semantic fusion to emit `*_evidence_bus_v1.json` and `*_belief_state_v1.json` sidecars.
- Changed: upgraded the Stage-1 video path to support manifest-backed video references, deterministic semantic extraction, governed video-state sidecars/hypotheses, and hypothesis-conditioned diffusion rendering; also made SceneTracks stub adapters configurable in the runner API.
- Changed: aligned repo-level docs and planning artifacts around the new Phase B posture: stable baseline frozen, `src/world_model/` reopened additively for governed successor modules, real-video grounding and governed supervision added as next roadmap stages, and learned video-state training moved into the training backlog as a deferred subset of economic-world-model readiness.
- Changed: tightened the roadmap and automation docs so autonomous execution now explicitly prioritizes Week 6.5 reconstruction/teacher-runtime work and Week 6.75 governed supervision before any learned video-state training pass.
- Verification: `python3 -m compileall src scripts/run_stage1_pipeline.py tests -q`, `python3 -m pytest -q tests/test_evidence_bus.py tests/test_runtime_adapters_v2.py tests/test_governed_video_world_model.py tests/test_rollout_labeler.py tests/test_semantic_fusion_orchestrator_smoke.py tests/test_stage1_pipeline_governed.py`, `python3 -m pytest -q tests/test_runtime_packets.py tests/test_vla_semantic_evidence.py tests/test_semantic_fusion_mvp.py tests/test_diffusion_prompt_includes_constraints.py`, and `python3 scripts/run_stage1_pipeline.py --num-videos 1 --proposals-per-video 1 --output-dir /tmp/stage1_governed_smoke` all passed.
- Blocked: the new video-state service is still heuristic/advisory, SceneTracks still defaults to stub adapters unless configured otherwise, and OpenVLA remains soft-fail instead of production-enforced.
- Next recommended task: add a D4RT-style reconstruction sidecar plus real SceneTracks/OpenVLA adapter plumbing so the governed video-state service stops depending on fallback evidence for real footage.

- Changed: wired Week 6.5 and Week 6.75 artifacts into live paths rather than leaving them as standalone helpers. `scripts/run_stage1_pipeline.py` now emits reconstruction sidecars, runtime packets, branch evaluations, event-spine sidecars, decision-ledger sidecars, governance traces, counterfactual evals, value-target packs, and value-ledger receipts for each governed video episode.
- Changed: tightened `src/vla/rollout_labeler.py` plus `src/vla/teacher_runtime.py` so rollout labeling now emits teacher contract and teacher action-envelope sidecars even when OpenVLA is disabled, missing, or failing; fallback state is now explicit and replayable.
- Changed: expanded focused coverage with reconstruction, teacher-runtime, and governed-supervision tests and strengthened Stage-1 / rollout-labeler assertions around live-loop artifact emission.
- Verification: `python3 -m compileall src scripts/run_stage1_pipeline.py tests -q` and `python3 -m pytest -q tests/test_rollout_labeler.py tests/test_stage1_pipeline_governed.py tests/test_four_d_reconstruction.py tests/test_teacher_runtime.py tests/test_governed_video_supervision.py` passed.
- Blocked: the live Stage-1 path still lacks real SceneTracks adapters, richer calibration sources, and non-stub teacher execution from real video frames; current grounding remains truthful-but-advisory rather than production-final.
- Next recommended task: push the same live-loop discipline into real-video ingestion boundaries, especially SceneTracks calibration joins and remaining teacher-runtime consumers, before any learned predictor training.

## 2026-03-19

- Changed: fixed `scripts/economic_world_model/nightly_audit.py` so progress-log freshness uses the most recent dated heading instead of the first heading, removing a stale false-positive drift signal against `scripts/TRAINING_MIGRATION_BACKLOG.json`.
- Changed: replaced the hardcoded EventSpine pending flag with real completion detection via additive code/doc checks (`src/runtime/event_spine.py`, `src/governance/trace.py`, and roadmap/gap-analysis phrase checks), so the nightly next-task selector no longer recommends already-landed work.
- Changed: updated audit compile verification to use `PYTHONPYCACHEPREFIX=/tmp/pycache` so sandboxed/local runs do not fail on unwritable default Python cache paths.
- Changed: added regression tests in `tests/test_economic_world_model_nightly_audit.py` for latest-date parsing, EventSpine pending detection, and audit-only fallback selection.
- Verification: `PYTHONPYCACHEPREFIX=/tmp/pycache python3 -m compileall src scripts/economic_world_model -q`, `python3 -m pytest -q tests/test_economic_world_model_nightly_audit.py tests/test_runtime_packets.py tests/embodiment/test_registry.py tests/test_objective_runtime_builder.py tests/test_constraint_set.py tests/test_pricing_sentinel.py tests/test_value_ledger.py`, and `python3 scripts/economic_world_model/nightly_audit.py --output-json artifacts/economic_world_model/nightly_audit_summary.json --output-markdown artifacts/economic_world_model/nightly_audit_summary.md` (result: `Status: ok`, drift signals: none).
- Blocked: `codex_api_key_present` remains `no`, so GitHub/cloud Codex execution is still credential-gated even though local CLI/app paths are ready.
- Next recommended task: prioritize a Week 6.5 additive grounding pass that wires richer SceneTracks calibration joins and remaining teacher-runtime consumers into real-video ingestion boundaries, then add focused smoke/test coverage before any learned predictor training.

## 2026-03-21

- Changed: added additive dataset-bridge scaffolding at `src/dataset_bridges/rlds_bridge.py` and `src/dataset_bridges/lerobot_bridge.py` (plus package exports) to provide lossy RLDS/LeRobot adapters while preserving references to internal objective/econ/governance/runtime sidecars in metadata.
- Changed: added focused bridge coverage in `tests/test_dataset_bridges.py` to lock down replay-step conversion semantics and sidecar-reference preservation.
- Changed: extended `scripts/economic_world_model/nightly_audit.py` with Week 7+ detection (`_dataset_bridge_scaffold_pending`) and a new `dataset_bridge_scaffold` task candidate so nightly selection no longer reports `audit_only` when dataset bridges are missing.
- Changed: expanded `tests/test_economic_world_model_nightly_audit.py` with explicit coverage for the new dataset-bridge task selection path.
- Verification: `PYTHONPYCACHEPREFIX=/tmp/pycache python3 -m compileall src scripts/economic_world_model -q`, `python3 -m pytest -q tests/test_dataset_bridges.py tests/test_economic_world_model_nightly_audit.py tests/test_runtime_packets.py tests/embodiment/test_registry.py tests/test_objective_runtime_builder.py tests/test_constraint_set.py tests/test_pricing_sentinel.py tests/test_value_ledger.py`, and `python3 scripts/economic_world_model/nightly_audit.py --output-json artifacts/economic_world_model/nightly_audit_summary.json --output-markdown artifacts/economic_world_model/nightly_audit_summary.md`.
- Blocked: `codex_api_key_present` remains `no` in this environment, so GitHub/cloud Codex execution is still credentials-gated.
- Next recommended task: deepen Week 7+ replay export/import glue so Stage-1 governed supervision artifacts can be emitted through dataset-bridge bundles without bespoke joins.
