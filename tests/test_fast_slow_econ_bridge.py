from src.orchestrator.fast_slow_econ_bridge import (
    ConstraintBound,
    ConstraintShadow,
    FastSlowEconBridge,
    OntologyMask,
    TransientLedger,
    EconTensorSample,
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


def test_transient_ledger_settlement_and_reconciliation():
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

    settled = ledger.settle(max_batch=3)
    assert settled is not None
    assert settled.sample_count == 3
    assert settled.start_tick == 0
    assert settled.end_tick == 2
    assert settled.totals["energy_delta"] == 0.30000000000000004

    assert ledger.is_reconciled(l2_latest_tick=2, max_tick_drift=4)
    assert not ledger.is_reconciled(l2_latest_tick=0, max_tick_drift=1)


def test_bridge_combines_shadow_mask_and_deploy_gate():
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

    decision = bridge.deploy_gate(l2_latest_tick=3, max_tick_drift=2, zone_id="zone_a")
    assert decision.allow_deploy is True
    assert decision.active_shadow_version == 7

    blocked = bridge.deploy_gate(l2_latest_tick=0, max_tick_drift=1, zone_id="zone_a")
    assert blocked.allow_deploy is False
    assert blocked.reason == "l1_l2_drift_exceeded"
