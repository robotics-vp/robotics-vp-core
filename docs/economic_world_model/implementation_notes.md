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
