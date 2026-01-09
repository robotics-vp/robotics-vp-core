import pytest
from unittest.mock import Mock, ANY
from typing import Dict, Optional

from src.contracts.schemas import (
    PlanPolicyConfigV1,
    PlanGainScheduleV1,
    TaskGraphOp,
    PlanOpType,
)
from src.representation.homeostasis import (
    ActionPlan,
    ActionType,
    SignalBundle,
    ControlSignal,
    SignalType,
)
from src.orchestrator.homeostatic_plan_writer import (
    map_action_to_plan_ops,
    build_plan_from_signals,
    GateStatus,
    EconPlanPolicyProvider,
    RewardIntegrityGuard,
)

# Constants
DEFAULT_WEIGHTS = {"task_a": 0.5, "task_b": 0.5}

def make_config(
    full_mult: float = 2.0,
    cons_mult: float = 1.1,
    max_change: Optional[float] = None,
    min_clamp: Optional[float] = None,
    max_clamp: Optional[float] = None,
) -> PlanPolicyConfigV1:
    return PlanPolicyConfigV1(
        gain_schedule=PlanGainScheduleV1(
            full_multiplier=full_mult,
            conservative_multiplier=cons_mult,
            max_abs_weight_change=max_change,
            min_weight_clamp=min_clamp,
            max_weight_clamp=max_clamp,
        ),
        default_weights=DEFAULT_WEIGHTS,
        epiplexity_low_threshold=0.2, # defaults
    )

def test_map_actions_normal_increase():
    """Test normal weight increase with transfer pass."""
    config = make_config(full_mult=1.5)
    gate_status = GateStatus(transfer_pass=True, stability_pass=True)
    plan = ActionPlan(actions=[ActionType.INCREASE_DATA], priority=1)
    
    ops, mult, clamped, meta = map_action_to_plan_ops(plan, config, gate_status)
    
    assert mult == 1.5
    assert not clamped
    assert len(ops) == 2
    # 0.5 * 1.5 = 0.75. Sum=1.5. Norm->0.5.
    # Wait, symmetric increase on symmetric weights = no relative change.
    # Weights should remain 0.5 after normalization.
    weights = {op.task_family: op.weight for op in ops}
    assert weights["task_a"] == 0.5
    assert weights["task_b"] == 0.5
    assert meta["renormalized"] is True # 1.5 != 1.0

def test_map_actions_conservative_increase():
    """Test conservative increase when transfer fails."""
    config = make_config(cons_mult=1.1)
    gate_status = GateStatus(transfer_pass=False, stability_pass=True)
    plan = ActionPlan(actions=[ActionType.INCREASE_DATA], priority=1)
    
    ops, mult, clamped, _ = map_action_to_plan_ops(plan, config, gate_status)
    
    assert mult == 1.1
    # Symmetric increase -> normalized back to 0.5
    weights = {op.task_family: op.weight for op in ops}
    assert weights["task_a"] == pytest.approx(0.5)

def test_clamping_max_change():
    """Test max absolute weight change clamping."""
    # Attempting to double weight: 0.5 -> 1.0 (delta 0.5)
    # Clamp max change to 0.2
    config = make_config(full_mult=2.0, max_change=0.2)
    gate_status = GateStatus(transfer_pass=True)
    plan = ActionPlan(actions=[ActionType.INCREASE_DATA], priority=1)
    
    # 0.5 -> 0.7 (clamped). Sum=1.4. Norm -> 0.7/1.4 = 0.5.
    # Still symmetric.
    
    ops, mult, clamped, meta = map_action_to_plan_ops(plan, config, gate_status)
    
    assert mult == 2.0
    assert clamped
    assert "max_abs_weight_change" in meta["clamp_reasons"]
    
    weights = {op.task_family: op.weight for op in ops}
    assert weights["task_a"] == pytest.approx(0.5)

def test_clamping_min_max_bounds():
    """Test absolute weight bounds."""
    # Attempting to increase 0.5 * 10 = 5.0. Cap at 2.0.
    config = make_config(full_mult=10.0, max_clamp=2.0)
    gate_status = GateStatus(transfer_pass=True)
    plan = ActionPlan(actions=[ActionType.INCREASE_DATA], priority=1)
    
    ops, _, clamped, meta = map_action_to_plan_ops(plan, config, gate_status)
    # 0.5->5.0 capped to 2.0. Sum 4.0. Norm -> 0.5.
    weights = {op.task_family: op.weight for op in ops}
    assert weights["task_a"] == 0.5
    assert "max_weight_clamp" in meta["clamp_reasons"]

    # Attempting to decrease 0.5 / 10 = 0.05. Floor at 0.1.
    config = make_config(full_mult=10.0, min_clamp=0.1)
    # decrease factor is 1/mult = 0.1
    plan = ActionPlan(actions=[ActionType.DECREASE_DATA], priority=1)
    
    ops, _, clamped, meta = map_action_to_plan_ops(plan, config, gate_status)
    # 0.5->0.05 capped to 0.1. Sum 0.2. Norm -> 0.5.
    weights = {op.task_family: op.weight for op in ops}
    assert weights["task_a"] == 0.5
    assert "min_weight_clamp" in meta["clamp_reasons"]

def test_forced_noop():
    """Test forced NOOP by stability gate."""
    config = make_config()
    gate_status = GateStatus(
        stability_pass=False, 
        forced_noop=True, 
        reason="Unstable"
    )
    plan = ActionPlan(actions=[ActionType.INCREASE_DATA], priority=1)
    
    ops, mult, clamped, _ = map_action_to_plan_ops(plan, config, gate_status)
    
    assert len(ops) == 2
    weights = {op.task_family: op.weight for op in ops}
    assert weights["task_a"] == 0.5
    assert mult == 1.0

class MockEconPolicy(EconPlanPolicyProvider):
    def get_gain_schedule(self, signal_bundle: SignalBundle) -> Optional[PlanGainScheduleV1]:
        return PlanGainScheduleV1(full_multiplier=5.0) 

def test_econ_policy_hook():
    """Test integration of EconPlanPolicyProvider hook."""
    signals = SignalBundle(signals=[
        ControlSignal(SignalType.EPIPLEXITY, value=1.0)
    ])
    config = make_config(full_mult=1.5)
    
    class MockEconPolicyAggressive(EconPlanPolicyProvider):
        def get_gain_schedule(self, signal_bundle: SignalBundle) -> Optional[PlanGainScheduleV1]:
             return PlanGainScheduleV1(
                 full_multiplier=5.0, 
                 conservative_multiplier=5.0
             )
    
    provider = MockEconPolicyAggressive()

    plan, status = build_plan_from_signals(
        signals, config, econ_policy_provider=provider
    )
    
    assert status.ledger_policy is not None
    assert status.ledger_policy.applied_multiplier == 5.0
    assert status.ledger_policy.gain_schedule_source == "econ_override"

class MockIntegrityGuard(RewardIntegrityGuard):
    def adjust_gain_schedule(self, schedule: PlanGainScheduleV1) -> PlanGainScheduleV1:
        new_full = min(schedule.full_multiplier, 2.0)
        return schedule.model_copy(update={"full_multiplier": new_full})

def test_integrity_guard_hook():
    """Test RewardIntegrityGuard hook clamping aggressive policy."""
    signals = SignalBundle(signals=[])
    config = make_config(full_mult=10.0) 
    
    signals.signals.append(ControlSignal(
        SignalType.DELTA_EPI_PER_FLOP, 
        value=1.0, 
        metadata={"transfer_pass": True, "stability_pass": True}
    ))
    
    guard = MockIntegrityGuard()
    
    plan, status = build_plan_from_signals(
        signals, config, integrity_guard=guard
    )
    
    assert status.ledger_policy.applied_multiplier == 2.0

def test_normalization_asymmetry():
    """Test that weights are normalized correctly when changes are asymmetric."""
    # Induce asymmetry via clamping
    config = make_config(full_mult=2.0, max_change=0.1)
    
    current_weights = {"task_a": 0.2, "task_b": 0.8}
    gate_status = GateStatus(transfer_pass=True)
    plan = ActionPlan(actions=[ActionType.INCREASE_DATA], priority=1)
    
    # task_a: 0.2 -> 0.4 (delta 0.2) -> clamped 0.3
    # task_b: 0.8 -> 1.6 (delta 0.8) -> clamped 0.9
    # Sum: 1.2
    # Norm: a=0.3/1.2=0.25, b=0.9/1.2=0.75
    
    ops, mult, clamped, meta = map_action_to_plan_ops(
        plan, config, gate_status, current_weights
    )
    
    assert meta["renormalized"] is True
    weights = {op.task_family: op.weight for op in ops}
    assert weights["task_a"] == pytest.approx(0.25)
    assert sum(weights.values()) == pytest.approx(1.0)

def test_cooldown_logic():
    """Test cooldown enforcement."""
    config = make_config()
    config.min_apply_interval_steps = 10
    
    signals = SignalBundle(signals=[])
    
    # Cooldown active
    _, gate = build_plan_from_signals(
        signals, config, steps_since_last_change=5
    )
    assert gate.forced_noop
    assert "Cooldown" in gate.reason
    
    # Cooldown passed
    _, gate = build_plan_from_signals(
        signals, config, steps_since_last_change=15
    )
    # Check that it's NOT cooldown reasons
    if gate.forced_noop:
        assert "Cooldown" not in gate.reason
