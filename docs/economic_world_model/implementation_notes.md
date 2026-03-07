# Economic World Model Implementation Notes

## 2026-03-07

- Introduced `RuntimePacket` and `ContractPacket` scaffolding in `src/runtime/packets.py` so objective/econ/constraint/evidence contracts can converge without touching default runtime behavior.
- Introduced `EmbodimentRegistry` and `CapabilityProfile` scaffolding in `src/embodiment/registry.py` so robot identity can be normalized before adapter rewrites begin.
- Added a repo-local nightly audit path plus real Codex execution wrappers for local CLI and GitHub/cloud runners.
- Re-centered the docs around Codex app automation plus the repo-local skill as the preferred autonomous path, with CLI and GitHub/cloud as fallbacks.
- Verified the new substrate with agent-ergonomics checks, compileall, targeted pytest, audit generation, and nightly runner shell syntax.
- Kept all new code additive and outside frozen Phase B zones.

## 2026-03-08

- Added `runtime_packet_sidecar_payload(...)` in `src/runtime/packets.py` so runtime packets can be emitted as a deterministic run-level sidecar without forcing replay schema changes.
- Wired `run_shadow_control_plane(...)` to emit `runtime_packets.json`, derive shadow-workcell observation/action schema refs from the existing episode log, and attach packet context into the episode artifact bundle.
- Wired `ingest_shadow_run(...)` to load `runtime_packets.json` when present and thread packet IDs, contract IDs, and sidecar refs into replay metadata/provenance while remaining backward-compatible with older runs that do not have packet sidecars.
- Extended targeted tests to cover sidecar payload serialization, shadow artifact emission, replay metadata round-tripping, and replay-dataset ingestion of packet refs.
- Kept the change additive: no replay dataclass shape changes, no frozen Phase B math changes, and no broad adapter refactor.

- Added `src/runtime/event_spine.py` with `RuntimeEvent`, `DecisionLedgerEntry`, and deterministic EventSpine / DecisionLedger sidecar payload builders so ordered runtime events and governance/economic decisions can be persisted without touching the replay schema.
- Wired `src/shadow_runtime/control_plane.py` to emit `event_spine.json` and `decision_ledger.json` with stable event kinds including `queue_reweight`, `pricing_tick_published`, `pricing_tick_suppressed`, `regal_warn`, `regal_veto`, `adaptation_admitted`, `adaptation_denied`, `collect_more_data`, `datapack_credit_assigned`, `promotion_hold`, and `promotion_recommend_promote` when applicable.
- Threaded stable `event_refs` and `decision_refs` through replay episode/step/window `metadata`, with sidecar file refs stored in `provenance`, so downstream consumers can join against the sidecars without requiring replay dataclass changes.
- Bound each emitted event and decision to the new runtime packet layer via `runtime_packet_id`, `contract_id`, objective/econ/pricing/regal artifact refs, and actor/critic/advisor provenance; receipt label refs are present as empty placeholders for future downstream linkage.
- Verified the new layer with targeted sidecar round-trip tests, shadow runner artifact tests, replay schema/dataset tests, receipt-ingest coverage, and `python3 -m compileall src -q`.
