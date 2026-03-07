from src.runtime.event_spine import (
    DecisionLedgerEntry,
    RuntimeEvent,
    decision_ledger_sidecar_payload,
    event_spine_sidecar_payload,
)


def test_event_spine_and_decision_ledger_round_trip():
    event = RuntimeEvent.from_components(
        run_id="run_event_spine",
        episode_id="ep_001",
        timestamp="2026-03-08T00:00:00+00:00",
        event_kind="pricing_tick_published",
        sequence_idx=3,
        scope={"scope_kind": "window", "window_id": "window_0", "start_step": 0, "end_step": 2},
        runtime_packet_id="runtime_packet_1",
        contract_id="contract.shadow.kitting.v1",
        artifact_refs={"runtime_packets": "runtime_packets.json"},
        provenance={"actor": {"component": "pricing_sentinel"}},
        metadata={"tick_id": "tick_1"},
    )
    decision = DecisionLedgerEntry.from_components(
        run_id="run_event_spine",
        episode_id="ep_001",
        timestamp="2026-03-08T00:00:00+00:00",
        decision_kind="pricing_tick_published",
        outcome="publish",
        sequence_idx=2,
        scope={"scope_kind": "window", "window_id": "window_0", "start_step": 0, "end_step": 2},
        reasons=["publish"],
        source_event_ids=[event.event_id],
        runtime_packet_id="runtime_packet_1",
        contract_id="contract.shadow.kitting.v1",
        artifact_refs={"runtime_packets": "runtime_packets.json"},
        provenance={"actor": {"component": "pricing_sentinel"}},
    )

    event_payload = event_spine_sidecar_payload(run_id="run_event_spine", events=[event])
    decision_payload = decision_ledger_sidecar_payload(run_id="run_event_spine", decisions=[decision])

    assert event_payload["event_count"] == 1
    assert decision_payload["decision_count"] == 1
    assert RuntimeEvent.from_dict(event_payload["events"][0]).to_dict() == event.to_dict()
    assert DecisionLedgerEntry.from_dict(decision_payload["decisions"][0]).to_dict() == decision.to_dict()
