# Economic World Model Implementation Notes

## 2026-03-07

- Introduced `RuntimePacket` and `ContractPacket` scaffolding in `src/runtime/packets.py` so objective/econ/constraint/evidence contracts can converge without touching default runtime behavior.
- Introduced `EmbodimentRegistry` and `CapabilityProfile` scaffolding in `src/embodiment/registry.py` so robot identity can be normalized before adapter rewrites begin.
- Added a repo-local nightly audit path plus real Codex execution wrappers for local CLI and GitHub/cloud runners.
- Re-centered the docs around Codex app automation plus the repo-local skill as the preferred autonomous path, with CLI and GitHub/cloud as fallbacks.
- Verified the new substrate with agent-ergonomics checks, compileall, targeted pytest, audit generation, and nightly runner shell syntax.
- Kept all new code additive and outside the stable frozen Phase B baseline zones.

## 2026-03-08

- Added `runtime_packet_sidecar_payload(...)` in `src/runtime/packets.py` so runtime packets can be emitted as a deterministic run-level sidecar without forcing replay schema changes.
- Wired `run_shadow_control_plane(...)` to emit `runtime_packets.json`, derive shadow-workcell observation/action schema refs from the existing episode log, and attach packet context into the episode artifact bundle.
- Wired `ingest_shadow_run(...)` to load `runtime_packets.json` when present and thread packet IDs, contract IDs, and sidecar refs into replay metadata/provenance while remaining backward-compatible with older runs that do not have packet sidecars.
- Extended targeted tests to cover sidecar payload serialization, shadow artifact emission, replay metadata round-tripping, and replay-dataset ingestion of packet refs.
- Kept the change additive: no replay dataclass shape changes, no stable Phase B baseline math changes, and no broad adapter refactor.

- Added `src/runtime/event_spine.py` with `RuntimeEvent`, `DecisionLedgerEntry`, and deterministic EventSpine / DecisionLedger sidecar payload builders so ordered runtime events and governance/economic decisions can be persisted without touching the replay schema.
- Wired `src/shadow_runtime/control_plane.py` to emit `event_spine.json` and `decision_ledger.json` with stable event kinds including `queue_reweight`, `pricing_tick_published`, `pricing_tick_suppressed`, `regal_warn`, `regal_veto`, `adaptation_admitted`, `adaptation_denied`, `collect_more_data`, `datapack_credit_assigned`, `promotion_hold`, and `promotion_recommend_promote` when applicable.
- Threaded stable `event_refs` and `decision_refs` through replay episode/step/window `metadata`, with sidecar file refs stored in `provenance`, so downstream consumers can join against the sidecars without requiring replay dataclass changes.
- Bound each emitted event and decision to the new runtime packet layer via `runtime_packet_id`, `contract_id`, objective/econ/pricing/regal artifact refs, and actor/critic/advisor provenance; receipt label refs are present as empty placeholders for future downstream linkage.
- Verified the new layer with targeted sidecar round-trip tests, shadow runner artifact tests, replay schema/dataset tests, receipt-ingest coverage, and `python3 -m compileall src -q`.

## 2026-03-09

- Added `src/runtime/action_adapter_v2.py` and `src/runtime/observation_adapter_v2.py`, plus packet-builder support for schema-producing adapter objects, so runtime contracts can carry explicit timing/provenance instead of relying on ad hoc `SchemaRef` construction at every call site.
- Added `src/evidence/bus.py`, `src/evidence/belief_state.py`, and `src/evidence/teacher_trace.py` to create a common evidence publication layer with validity, disagreement, artifact refs, and advisory teacher-trace semantics.
- Wired `src/vla/rollout_labeler.py` to persist `teacher_trace_v1.json` sidecars and upgraded `src/vla/semantic_evidence.py` so VLA semantic evidence carries governed provenance including teacher-trace refs and fallback mode.
- Wired `src/orchestrator/semantic_fusion_runner.py` to emit `*_evidence_bus_v1.json` and `*_belief_state_v1.json` beside semantic-fusion artifacts, so semantic evidence is no longer trapped in component-local files.
- Reopened `src/world_model/` with `src/world_model/governed_video_world_model.py`, which builds belief-state-driven video-state snapshots and ranked geometry-first hypotheses without touching the stable Phase B checkpoint.
- Upgraded `scripts/run_stage1_pipeline.py` to support manifest-backed real video references, deterministic semantic extraction, governed video-state sidecars/hypotheses, and hypothesis-conditioned diffusion proposals; the script is now directly runnable via `python3 scripts/run_stage1_pipeline.py ...` without `ModuleNotFoundError`.
- Made `src/vision/scene_ir_tracker/io/scene_tracks_runner.py` expose `use_stub_adapters` as a runner option instead of hardwiring it, and exposed richer OpenVLA fallback provenance in `src/vla/openvla_controller.py`.
- Updated the roadmap, automation spec, repo guidance, and training backlog to treat Phase B as a frozen stable baseline plus an additive successor track. Learned video-state modeling is now documented as a backlog item gated on real-video grounding, teacher-runtime hardening, and governed supervision bundles.
- Tightened the roadmap and nightly-selection rules so autonomous passes know to consume Week 6.5 and Week 6.75 in order instead of skipping ahead to learned video-state training.
- Verified the tranche with compileall, focused pytest coverage around evidence/runtime/video-state integration, and a direct Stage-1 CLI smoke run.

- Wired `src/vision/reconstruction/four_d_reconstruction.py` into the live Stage-1 loop so every governed video episode now emits a reconstruction sidecar with calibration completeness, frame windows, geometry refs, and evidence joins.
- Wired `src/world_model/governed_video_supervision.py`, `src/economics/counterfactual_eval.py`, `src/economics/value_targets.py`, and `src/governance/trace.py` into the live Stage-1 loop so candidate futures now emit runtime packets, branch evaluations, event spine rows, decision-ledger rows, governance traces, counterfactual evals, value-target packs, and value-ledger receipts.
- Tightened `src/vla/teacher_runtime.py` plus `src/vla/rollout_labeler.py` so the live rollout-labeling path emits explicit teacher adapter contracts and teacher action envelopes even when OpenVLA is disabled or unavailable; fallback is now replayable instead of implicit.
- Expanded targeted coverage with `tests/test_four_d_reconstruction.py`, `tests/test_teacher_runtime.py`, and `tests/test_governed_video_supervision.py`, and extended Stage-1 / rollout-labeler tests to assert the new live-loop sidecars exist.

## 2026-03-19

- Updated `scripts/economic_world_model/nightly_audit.py` to fix stale roadmap drift evaluation:
  - `_progress_latest_date()` now returns the last dated `## YYYY-MM-DD` heading in `docs/economic_world_model/progress_log.md`, not the first.
  - Added `_event_spine_spec_pending()` and `_contains_phrase(...)` so EventSpine/GovernanceTrace recommendation state is derived from additive code/doc presence rather than a hardcoded `pending=True`.
  - Updated the audit compile command to `PYTHONPYCACHEPREFIX=/tmp/pycache python3 -m compileall src scripts/economic_world_model -q` to avoid sandbox cache-permission failures.
- Added `tests/test_economic_world_model_nightly_audit.py` covering:
  - most-recent progress date extraction
  - EventSpine spec pending=false when code/docs are present
  - fallback to `audit_only` when all candidate tasks are complete
- Regenerated audit artifacts with the updated logic:
  - `artifacts/economic_world_model/nightly_audit_summary.json`
  - `artifacts/economic_world_model/nightly_audit_summary.md`
  - current result: `status=ok`, `roadmap_drift.signals=[]`, `next_task.id=audit_only`.

## 2026-03-21

- Added `src/dataset_bridges/rlds_bridge.py` and `src/dataset_bridges/lerobot_bridge.py` as additive, lossy adapters from canonical replay records to standard interchange formats.
- Kept internal schema richness explicit by preserving objective/econ/pricing/ledger/event/decision/runtime/governance references in bridge metadata rather than flattening them away.
- Added `src/dataset_bridges/__init__.py` exports so bridge adapters can be imported as a package-level surface.
- Added `tests/test_dataset_bridges.py` to validate ordering, terminal-step flags, and sidecar-ref preservation for both RLDS and LeRobot adapter outputs.
- Extended `scripts/economic_world_model/nightly_audit.py` with `_dataset_bridge_scaffold_pending()` and a corresponding `dataset_bridge_scaffold` candidate so roadmap selection includes Week 7+ bridge scaffolding status.
- Extended `tests/test_economic_world_model_nightly_audit.py` with coverage that asserts dataset-bridge candidate selection when the new scaffold is pending.

## 2026-03-22

- Added `src/dataset_bridges/sidecar_refs.py` with `extract_sidecar_refs(...)` to centralize replay sidecar extraction for bridge exports.
- The extractor keeps bridge exports additive and forward-compatible by harvesting references from replay record fields and `metadata`/`provenance` keys that end in `*_ref`, `*_refs`, `*_id`, or `*_ids`.
- Switched `src/dataset_bridges/rlds_bridge.py` and `src/dataset_bridges/lerobot_bridge.py` to use the shared extractor instead of hardcoded per-key sidecar mappings, reducing future maintenance when new governed-supervision refs are introduced.
- Extended `tests/test_dataset_bridges.py` so RLDS/LeRobot bridge outputs assert preservation of representative Week 6.75/7+ sidecar refs (`counterfactual_eval_ref`, `value_target_refs`, `belief_state_ref`, `teacher_trace_ref`, and `governed_supervision_refs`).
- Added `scripts/economic_world_model/publish_codex_change.sh` to publish automation commits to `origin/main` when the local change is a safe fast-forward, while falling back to a timestamped `codex/ewm-nightly-*` branch when direct main pushes are rejected.
- Updated `scripts/economic_world_model/run_nightly_codex_task.sh` so the generated Codex task now requires publication via the helper and reports either the published ref or the exact push blocker before the run is considered complete.
- Updated `docs/economic_world_model/AUTOMATION_SPEC.md`, `codex_skills/economic-world-model-roadmap/SKILL.md`, and the live app automation prompt to treat unpublished local commits as incomplete automation output.
- Added `src/economics/inferential_reward.py` as a shared successor-layer compiler for `InferentialSignalYield` and `InferentialRewardBreakdown`, keeping signal-yield math additive and outside frozen Phase B reward/dynamics code.
- Extended `InferentialTrainingCandidate` and `InferentialTrainingGate` to carry frontier gain, epiplexity, transfer, governance, and optional signal-yield overrides, then compile a canonical inferential reward breakdown before making budget decisions.
- Wired advisory consumers to use the compiled signal-yield path:
  - `src/orchestrator/shadow_advisory.py` now computes signal yield from replay frontier gain plus any available epiplexity fields.
  - `src/rl/econ_regal_sampling.py` now admits signal yield as a bounded replay-priority input.
  - `src/rl/episode_sampling.py` and `src/policies/sampler_weights.py` now emit/consume `signal_yield_score` and `inferential_replay_weight`, including a new `inferential_yield` weighting strategy.
  - `src/orchestrator/queue_selection.py` now preserves inferential reward evidence in queue metadata.
- Refactored the epiplexity core so tracker cache entries are baseline-independent absolute runs with estimator provenance and `flops_estimate`, while baseline-relative `delta_epi_vs_baseline` is derived only when consumers compare a candidate against a baseline.
- Promoted `RequentialEstimator` from a zero-return stub into an online evaluate-then-update estimator, so the second estimator path now produces nontrivial learnability scores instead of placeholder zeros.
- Added canonical epiplexity overlay helpers and automatic repo merging:
  - `src/epiplexity/metadata.py` now writes/loads `epiplexity_overlays.jsonl`, manages default selectors, and lets consumers recover the best available repr/budget even when `_default` is absent.
  - `src/valuation/datapack_repo.py` now auto-merges epiplexity overlays during `load_all(...)` and invalidates cached task loads when the overlay sidecar changes.
- Wired `scripts/run_epiplexity_curated_slices.py` to persist canonical overlays in both full and token-only modes, so portable fallback runs now emit the same summary shape consumed by samplers and replay/inferential advisory code.
- Corrected downstream consumers that had been reading the wrong epiplexity slot:
  - `src/orchestrator/datapack_engine.py` now uses `epi_repr_id` or the datapack default selector rather than incorrectly reading the baseline repr’s delta.
  - `src/orchestrator/homeostatic_plan_writer.py` and `src/representation/homeostasis.py` now understand canonical nested epiplexity summaries instead of only legacy `mean_variance`/`variance` placeholders.
  - `src/evaluation/probe_harness.py` now reports real baseline/after means rather than recycling the delta into those fields.
- Kept the change additive: no edits to the stable Phase B checkpoint, no legacy world-model math rewrite, and no baseline reward-path mutation.
- Added `docs/economic_world_model/ewm-nightly.automation.toml` as a checked-in mirror of the live Codex app automation config, omitting only local timestamp fields so the active prompt/schedule/environment are versioned with the repo.
- Updated `docs/economic_world_model/AUTOMATION_SPEC.md` to point at the checked-in automation snapshot as the Git-tracked source of truth for the live app automation state.
