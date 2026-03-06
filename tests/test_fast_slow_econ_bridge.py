import pytest

from src.orchestrator.fast_slow_econ_bridge import (
    ConstraintBound,
    ConstraintShadow,
    EconTensorSample,
    FastSlowEconBridge,
    OntologyMask,
    TransientLedger,
)


def test_geometric_shadow_projects_action_to_bounds():
    shadow = ConstraintShadow(
        version=1,
        bounds={
            "vx": ConstraintBound(min_value=-0.5, max_value=0.5),
            "grip": ConstraintBound(min_value=0.0, max_value=1.0),
        },
    )

    action = {"vx": 1.2, "grip": -2.0, "unused": 4.0}
    projected = shadow.project(action)

    assert projected["vx"] == 0.5
    assert projected["grip"] == 0.0
    assert projected["unused"] == 4.0


def test_invalid_constraint_bound_raises_value_error():
    with pytest.raises(ValueError):
        ConstraintBound(min_value=1.0, max_value=0.5)


def test_transient_ledger_prepare_ack_and_reconciliation():
    ledger = TransientLedger(capacity=8)
    for tick in range(5):
        ledger.append(
            sample=EconTensorSample(
                tick_id=tick,
                energy_delta=0.1,
                error_delta=0.01,
                time_delta_ms=1.0,
            )
        )

    prepared = ledger.prepare_settlement(max_batch=3)
    assert prepared is not None
    assert prepared.sample_count == 3
    assert prepared.start_tick == 0
    assert prepared.end_tick == 2
    assert prepared.totals["energy_delta"] == pytest.approx(0.3)

    # Not acked yet, so pending still includes all samples.
    assert len(ledger.pending()) == 5

    ledger.ack_settlement(prepared.end_tick)
    assert len(ledger.pending()) == 2

    assert ledger.is_reconciled(l2_latest_tick=2, max_tick_drift=4)
    assert not ledger.is_reconciled(l2_latest_tick=0, max_tick_drift=1)


def test_bridge_combines_shadow_mask_settlement_and_deploy_gate():
    bridge = FastSlowEconBridge(mask=OntologyMask())
    zone_shadow = ConstraintShadow(
        version=7,
        bounds={"joint_0": ConstraintBound(min_value=-1.0, max_value=1.0)},
    )
    bridge.update_shadow(zone_shadow, zone_id="zone_a")

    projected = bridge.project_action({"joint_0": 3.0}, zone_id="zone_a")
    assert projected["joint_0"] == 1.0

    for tick in range(4):
        bridge.ingest_tick(
            tick_id=tick, energy_delta=0.2, error_delta=0.0, time_delta_ms=1.0
        )

    prepared = bridge.settle_to_l2(max_batch=2, acknowledge=False)
    assert prepared is not None
    assert prepared.end_tick == 1
    assert len(bridge.ledger.pending()) == 4

    acknowledged = bridge.settle_to_l2(max_batch=2, acknowledge=True)
    assert acknowledged is not None
    assert acknowledged.end_tick == 1
    assert len(bridge.ledger.pending()) == 2

    decision = bridge.deploy_gate(l2_latest_tick=3, max_tick_drift=2, zone_id="zone_a")
    assert decision.allow_deploy is True
    assert decision.active_shadow_version == 7

    blocked = bridge.deploy_gate(l2_latest_tick=0, max_tick_drift=1, zone_id="zone_a")
    assert blocked.allow_deploy is False
    assert blocked.reason == "l1_l2_drift_exceeded"
