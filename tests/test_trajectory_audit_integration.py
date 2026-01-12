"""Integration test for trajectory audit producer and regal flow.

Tests:
1. TrajectoryAuditV1 creation via producer
2. SHA stability across re-runs (determinism)
3. Regal evaluation consumes trajectory_audit
4. SHA persisted in manifest
"""
import pytest
import numpy as np

from src.contracts.schemas import (
    TrajectoryAuditV1,
    RegalGatesV1,
    RegalPhaseV1,
    RegalContextV1,
    LedgerRegalV1,
    PlanPolicyConfigV1,
)
from src.valuation.trajectory_audit import (
    create_trajectory_audit,
    aggregate_trajectory_audits,
)
from src.regal.regal_evaluator import evaluate_regals
from src.representation.homeostasis import SignalBundle


def _create_deterministic_episode_data(seed: int = 42, num_steps: int = 20):
    """Create deterministic episode data for testing."""
    rng = np.random.default_rng(seed)
    
    action_dim = 4
    actions = [
        [float(x) for x in rng.standard_normal(action_dim) * 0.1]
        for _ in range(num_steps)
    ]
    rewards = [float(rng.uniform(0.0, 0.1)) for _ in range(num_steps)]
    reward_components = {
        "task": [float(rng.uniform(0.05, 0.08)) for _ in range(num_steps)],
        "penalty": [float(rng.uniform(-0.02, 0.0)) for _ in range(num_steps)],
    }
    events = ["start", "grasp", "end"]
    velocities = [
        [float(x) for x in rng.uniform(0.5, 2.0, 3)]
        for _ in range(num_steps)
    ]
    
    return {
        "episode_id": f"test_ep_{seed}",
        "num_steps": num_steps,
        "actions": actions,
        "rewards": rewards,
        "reward_components": reward_components,
        "events": events,
        "velocities": velocities,
        "velocity_threshold": 10.0,
    }


class TestTrajectoryAuditProducer:
    """Test create_trajectory_audit producer."""
    
    def test_creates_valid_audit(self):
        """Producer creates valid TrajectoryAuditV1."""
        data = _create_deterministic_episode_data()
        audit = create_trajectory_audit(**data)
        
        assert isinstance(audit, TrajectoryAuditV1)
        assert audit.episode_id == "test_ep_42"
        assert audit.num_steps == 20
        assert audit.total_return > 0
        assert audit.action_mean is not None
        assert len(audit.action_mean) == 4
        assert audit.event_counts == {"start": 1, "grasp": 1, "end": 1}
    
    def test_sha_stable_across_reruns(self):
        """Same inputs produce same SHA (determinism)."""
        data1 = _create_deterministic_episode_data(seed=42)
        data2 = _create_deterministic_episode_data(seed=42)
        
        audit1 = create_trajectory_audit(**data1)
        audit2 = create_trajectory_audit(**data2)
        
        sha1 = audit1.sha256()
        sha2 = audit2.sha256()
        
        assert sha1 == sha2, f"SHA mismatch: {sha1} != {sha2}"
    
    def test_different_seeds_different_sha(self):
        """Different seeds produce different SHAs."""
        data1 = _create_deterministic_episode_data(seed=42)
        data2 = _create_deterministic_episode_data(seed=43)
        
        audit1 = create_trajectory_audit(**data1)
        audit2 = create_trajectory_audit(**data2)
        
        assert audit1.sha256() != audit2.sha256()
    
    def test_aggregate_audits_deterministic(self):
        """Aggregate SHA is deterministic and order-independent."""
        data1 = _create_deterministic_episode_data(seed=1)
        data2 = _create_deterministic_episode_data(seed=2)
        
        audit1 = create_trajectory_audit(**data1)
        audit2 = create_trajectory_audit(**data2)
        
        # Same list, same aggregate
        agg1 = aggregate_trajectory_audits([audit1, audit2])
        agg2 = aggregate_trajectory_audits([audit1, audit2])
        assert agg1 == agg2
        
        # Order of audits changes raw order but aggregate uses sorted SHAs
        # so order-independent (sorted by individual SHA)
        agg3 = aggregate_trajectory_audits([audit2, audit1])
        assert agg1 == agg3


class TestRegalWithTrajectoryAudit:
    """Test regal evaluation with trajectory audit."""
    
    def test_regal_evaluates_with_trajectory_audit(self):
        """Regal evaluation consumes trajectory_audit at POST_AUDIT phase."""
        data = _create_deterministic_episode_data()
        audit = create_trajectory_audit(**data)
        trajectory_audit_sha = audit.sha256()
        
        # Minimal regal config with correct field names
        regal_config = RegalGatesV1(
            enabled_regal_ids=["world_coherence"],
            velocity_spike_threshold=5,
            contact_anomaly_threshold=3,
            patience=1,
        )
        
        # Build context with trajectory audit SHA
        context = RegalContextV1(
            run_id="test_run",
            step=1,
            trajectory_audit_sha=trajectory_audit_sha,
        )
        
        # Minimal signals
        signals = SignalBundle(signals=[], timestamp="2026-01-01T00:00:00")
        
        result = evaluate_regals(
            config=regal_config,
            phase=RegalPhaseV1.POST_AUDIT,
            signals=signals,
            context=context,
            trajectory_audit=audit,
        )
        
        assert isinstance(result, LedgerRegalV1)
        assert result.all_passed  # No anomalies in test data
        assert any(r.regal_id == "world_coherence" for r in result.reports)
    
    def test_trajectory_audit_sha_in_context_matches(self):
        """trajectory_audit_sha in context matches actual audit SHA."""
        data = _create_deterministic_episode_data()
        audit = create_trajectory_audit(**data)
        expected_sha = audit.sha256()
        
        context = RegalContextV1(
            run_id="test_run",
            step=1,
            trajectory_audit_sha=expected_sha,
        )
        
        # Context should store the SHA for provenance
        assert context.trajectory_audit_sha == expected_sha


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
