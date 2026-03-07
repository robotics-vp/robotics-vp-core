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
