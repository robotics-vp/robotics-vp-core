"""
Smoke tests for Phase H Advisory Integration.

Verifies:
- Determinism
- Advisory boundaries (±20% routing, [0.8, 1.2] multipliers)
- Weight deltas within bounds
- Feature fields present only when flag enabled
"""
import tempfile
import json
from pathlib import Path

from src.phase_h.advisory_integration import (
    PhaseHAdvisory,
    apply_sampler_advisory,
    apply_orchestrator_advisory,
    build_phase_h_condition_fields,
    load_phase_h_advisory,
    MIN_MULTIPLIER,
    MAX_MULTIPLIER,
    MAX_ROUTING_DELTA,
)
from src.phase_h.models import Skill, ExplorationBudget, SkillReturns
from src.observation.condition_vector_builder import ConditionVectorBuilder
from src.orchestrator.semantic_orchestrator_v2 import OrchestratorAdvisory


def test_advisory_multipliers_bounded():
    """Test that skill multipliers are bounded to [0.8, 1.2]."""
    # Create test skills with extreme ROI
    skills = {
        "high_roi": Skill(
            skill_id="high_roi",
            display_name="High ROI Skill",
            description="Test",
            mpl_baseline=10.0,
            mpl_current=15.0,
            mpl_target=20.0,
            training_cost_usd=100.0,
            data_cost_per_episode=1.0,
            success_rate=0.95,
            failure_rate=0.05,
            recovery_rate=0.9,
            fragility_score=0.95,
            ood_exposure=0.3,
            novelty_tier_avg=1.5,
            training_episodes=1000,
            last_updated="2025-01-01T00:00:00Z",
            status="training",
        ),
        "low_roi": Skill(
            skill_id="low_roi",
            display_name="Low ROI Skill",
            description="Test",
            mpl_baseline=10.0,
            mpl_current=10.5,
            mpl_target=20.0,
            training_cost_usd=500.0,
            data_cost_per_episode=1.0,
            success_rate=0.6,
            failure_rate=0.4,
            recovery_rate=0.5,
            fragility_score=0.6,
            ood_exposure=0.1,
            novelty_tier_avg=0.5,
            training_episodes=100,
            last_updated="2025-01-01T00:00:00Z",
            status="exploration",
        ),
    }

    budgets = {
        "high_roi": ExplorationBudget("high_roi", 1000.0, 200.0, 800.0, 0.4, 0.5, 0.1, 800),
        "low_roi": ExplorationBudget("low_roi", 500.0, 400.0, 100.0, 0.6, 0.3, 0.1, 100),
    }

    returns = [
        SkillReturns("high_roi", 5.0, 50.0, 100.0, 5.0, 10.0, 0.1, 0.2, 5, 150.0),
        SkillReturns("low_roi", 0.5, 5.0, 10.0, 1.0, 1.0, 0.0, 0.05, 1, -30.0),
    ]

    advisory = PhaseHAdvisory(skills, budgets, returns)

    # Check multipliers are bounded
    for skill_id, mult in advisory.skill_multipliers.items():
        assert MIN_MULTIPLIER <= mult <= MAX_MULTIPLIER, (
            f"Multiplier for {skill_id} out of bounds: {mult}"
        )

    print("✓ Advisory multipliers bounded to [0.8, 1.2]")


def test_sampler_advisory_bounded():
    """Test that sampler advisory changes are bounded to ±20%."""
    # Create base weights
    base_weights = {
        "ep1": 1.0,
        "ep2": 0.5,
        "ep3": 2.0,
    }

    # Create advisory
    skills = {
        "test": Skill(
            "test", "Test", "Test", 10.0, 12.0, 20.0, 100.0, 1.0,
            0.8, 0.2, 0.7, 0.8, 0.2, 1.0, 100, "2025-01-01T00:00:00Z", "training"
        ),
    }
    budgets = {"test": ExplorationBudget("test", 1000.0, 200.0, 800.0, 0.4, 0.5, 0.1, 800)}
    returns = [SkillReturns("test", 2.0, 20.0, 50.0, 2.0, 5.0, 0.05, 0.1, 2, 25.0)]

    advisory = PhaseHAdvisory(skills, budgets, returns)

    # Apply advisory
    adjusted_weights = apply_sampler_advisory(base_weights, advisory, enable_phase_h_advisories=True)

    # Check all changes are within ±20%
    for ep_key, base_weight in base_weights.items():
        adjusted = adjusted_weights[ep_key]
        delta_pct = abs(adjusted - base_weight) / base_weight
        assert delta_pct <= MAX_ROUTING_DELTA, (
            f"Weight change for {ep_key} exceeds 20%: {delta_pct:.2%}"
        )

    print("✓ Sampler advisory changes bounded to ±20%")


def test_sampler_advisory_disabled_by_default():
    """Test that sampler advisory is disabled when flag=False."""
    base_weights = {"ep1": 1.0, "ep2": 0.5}

    skills = {
        "test": Skill(
            "test", "Test", "Test", 10.0, 12.0, 20.0, 100.0, 1.0,
            0.8, 0.2, 0.7, 0.8, 0.2, 1.0, 100, "2025-01-01T00:00:00Z", "training"
        ),
    }
    budgets = {"test": ExplorationBudget("test", 1000.0, 200.0, 800.0, 0.4, 0.5, 0.1, 800)}
    returns = [SkillReturns("test", 2.0, 20.0, 50.0, 2.0, 5.0, 0.05, 0.1, 2, 25.0)]

    advisory = PhaseHAdvisory(skills, budgets, returns)

    # Apply with flag disabled
    adjusted_weights = apply_sampler_advisory(base_weights, advisory, enable_phase_h_advisories=False)

    # Should be unchanged
    assert adjusted_weights == base_weights

    print("✓ Sampler advisory disabled by default (flag=False)")


def test_orchestrator_advisory_bounded():
    """Test that orchestrator advisory changes are bounded to ±20%."""
    # Create base orchestrator advisory
    base_advisory = OrchestratorAdvisory(
        task_id="test_task",
        focus_objective_presets=["balanced"],
        sampler_strategy_overrides={"balanced": 0.5},
        datapack_priority_tags=["test"],
        safety_emphasis=0.3,
        metadata={},
    )

    # Create Phase H advisory with high safety emphasis
    skills = {
        "unsafe": Skill(
            "unsafe", "Unsafe Skill", "Test", 10.0, 11.0, 20.0, 100.0, 1.0,
            0.5, 0.5, 0.4, 0.5, 0.3, 1.0, 100, "2025-01-01T00:00:00Z", "exploration"
        ),
    }
    budgets = {"unsafe": ExplorationBudget("unsafe", 1000.0, 100.0, 900.0, 0.6, 0.3, 0.1, 900)}
    returns = [SkillReturns("unsafe", 1.0, 10.0, 20.0, 1.0, 2.0, 0.0, 0.05, 1, -10.0)]

    advisory = PhaseHAdvisory(skills, budgets, returns)

    # Apply advisory
    original_safety = base_advisory.safety_emphasis
    adjusted_advisory = apply_orchestrator_advisory(base_advisory, advisory, enable_phase_h_advisories=True)

    # Check safety emphasis change is bounded
    delta = abs(adjusted_advisory.safety_emphasis - original_safety)
    assert delta <= MAX_ROUTING_DELTA, f"Safety emphasis delta exceeds 20%: {delta:.2%}"

    print("✓ Orchestrator advisory changes bounded to ±20%")


def test_orchestrator_advisory_disabled_by_default():
    """Test that orchestrator advisory is disabled when flag=False."""
    base_advisory = OrchestratorAdvisory(
        task_id="test_task",
        focus_objective_presets=["balanced"],
        sampler_strategy_overrides={"balanced": 0.5},
        datapack_priority_tags=["test"],
        safety_emphasis=0.3,
        metadata={},
    )

    skills = {
        "test": Skill(
            "test", "Test", "Test", 10.0, 12.0, 20.0, 100.0, 1.0,
            0.8, 0.2, 0.7, 0.8, 0.2, 1.0, 100, "2025-01-01T00:00:00Z", "training"
        ),
    }
    budgets = {"test": ExplorationBudget("test", 1000.0, 200.0, 800.0, 0.4, 0.5, 0.1, 800)}
    returns = [SkillReturns("test", 2.0, 20.0, 50.0, 2.0, 5.0, 0.05, 0.1, 2, 25.0)]

    advisory = PhaseHAdvisory(skills, budgets, returns)

    original_safety = base_advisory.safety_emphasis
    adjusted_advisory = apply_orchestrator_advisory(base_advisory, advisory, enable_phase_h_advisories=False)

    # Should be unchanged
    assert adjusted_advisory.safety_emphasis == original_safety
    assert "phase_h_routing" not in adjusted_advisory.metadata

    print("✓ Orchestrator advisory disabled by default (flag=False)")


def test_condition_fields_only_when_enabled():
    """Test that ConditionVector fields are only present when flag enabled."""
    skills = {
        "test": Skill(
            "test", "Test", "Test", 10.0, 12.0, 20.0, 100.0, 1.0,
            0.8, 0.2, 0.7, 0.8, 0.2, 1.0, 100, "2025-01-01T00:00:00Z", "training"
        ),
    }
    budgets = {"test": ExplorationBudget("test", 1000.0, 200.0, 800.0, 0.4, 0.5, 0.1, 800)}
    returns = [SkillReturns("test", 2.0, 20.0, 50.0, 2.0, 5.0, 0.05, 0.1, 2, 25.0)]

    advisory = PhaseHAdvisory(skills, budgets, returns)

    # Disabled: should return empty dict
    fields_disabled = build_phase_h_condition_fields(advisory, "test", enable_phase_h_advisories=False)
    assert fields_disabled == {}

    # Enabled: should return fields
    fields_enabled = build_phase_h_condition_fields(advisory, "test", enable_phase_h_advisories=True)
    assert "exploration_uplift" in fields_enabled
    assert "skill_roi_estimate" in fields_enabled
    assert isinstance(fields_enabled["exploration_uplift"], float)
    assert isinstance(fields_enabled["skill_roi_estimate"], float)

    print("✓ ConditionVector fields only present when flag enabled")


def test_determinism():
    """Test that advisory computation is deterministic."""
    skills = {
        "test": Skill(
            "test", "Test", "Test", 10.0, 12.0, 20.0, 100.0, 1.0,
            0.8, 0.2, 0.7, 0.8, 0.2, 1.0, 100, "2025-01-01T00:00:00Z", "training"
        ),
    }
    budgets = {"test": ExplorationBudget("test", 1000.0, 200.0, 800.0, 0.4, 0.5, 0.1, 800)}
    returns = [SkillReturns("test", 2.0, 20.0, 50.0, 2.0, 5.0, 0.05, 0.1, 2, 25.0)]

    # Run twice
    advisory1 = PhaseHAdvisory(skills, budgets, returns)
    advisory2 = PhaseHAdvisory(skills, budgets, returns)

    # Should produce identical results
    assert advisory1.skill_multipliers == advisory2.skill_multipliers
    assert advisory1.skill_quality_signals == advisory2.skill_quality_signals
    assert advisory1.exploration_priorities == advisory2.exploration_priorities
    assert advisory1.routing_advisories == advisory2.routing_advisories

    print("✓ Advisory computation is deterministic")


def test_json_safe_export():
    """Test that advisory can be exported to JSON."""
    skills = {
        "test": Skill(
            "test", "Test", "Test", 10.0, 12.0, 20.0, 100.0, 1.0,
            0.8, 0.2, 0.7, 0.8, 0.2, 1.0, 100, "2025-01-01T00:00:00Z", "training"
        ),
    }
    budgets = {"test": ExplorationBudget("test", 1000.0, 200.0, 800.0, 0.4, 0.5, 0.1, 800)}
    returns = [SkillReturns("test", 2.0, 20.0, 50.0, 2.0, 5.0, 0.05, 0.1, 2, 25.0)]

    advisory = PhaseHAdvisory(skills, budgets, returns)

    # Export to dict
    advisory_dict = advisory.to_dict()

    # Should be JSON-serializable
    try:
        json_str = json.dumps(advisory_dict)
        assert len(json_str) > 0
    except Exception as e:
        assert False, f"Failed to serialize advisory to JSON: {e}"

    print("✓ Advisory is JSON-safe")


def test_load_phase_h_advisory():
    """Test loading Phase H advisory from ontology."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ontology_root = Path(tmpdir)
        phase_h_dir = ontology_root / "phase_h"
        phase_h_dir.mkdir(parents=True)

        # Write test artifacts
        market_state = {
            "timestamp": "2025-01-01T00:00:00Z",
            "total_budget_usd": 1000.0,
            "allocated_usd": 500.0,
            "skills": {
                "test": {
                    "skill_id": "test",
                    "display_name": "Test Skill",
                    "description": "Test",
                    "mpl_baseline": 10.0,
                    "mpl_current": 12.0,
                    "mpl_target": 20.0,
                    "training_cost_usd": 100.0,
                    "data_cost_per_episode": 1.0,
                    "success_rate": 0.8,
                    "failure_rate": 0.2,
                    "recovery_rate": 0.7,
                    "fragility_score": 0.8,
                    "ood_exposure": 0.2,
                    "novelty_tier_avg": 1.0,
                    "training_episodes": 100,
                    "last_updated": "2025-01-01T00:00:00Z",
                    "status": "training",
                }
            }
        }

        budget_state = {
            "total_budget_usd": 1000.0,
            "reallocation_period_episodes": 1000,
            "budgets_by_skill": {
                "test": {
                    "skill_id": "test",
                    "budget_usd": 500.0,
                    "spent_usd": 100.0,
                    "remaining_usd": 400.0,
                    "data_collection_pct": 0.4,
                    "compute_training_pct": 0.5,
                    "human_supervision_pct": 0.1,
                    "max_episodes": 400,
                }
            }
        }

        returns_state = {
            "returns": [
                {
                    "skill_id": "test",
                    "delta_mpl": 2.0,
                    "delta_mpl_pct": 20.0,
                    "delta_energy_wh": 50.0,
                    "delta_time_sec": 2.0,
                    "delta_damage": 5.0,
                    "delta_success_rate": 0.05,
                    "delta_novelty_coverage": 0.1,
                    "unique_failure_modes_discovered": 2,
                    "roi_pct": 25.0,
                }
            ]
        }

        with open(phase_h_dir / "skill_market_state.json", "w") as f:
            json.dump(market_state, f)
        with open(phase_h_dir / "exploration_budget.json", "w") as f:
            json.dump(budget_state, f)
        with open(phase_h_dir / "skill_returns.json", "w") as f:
            json.dump(returns_state, f)

        # Load advisory
        advisory = load_phase_h_advisory(ontology_root)

        assert advisory is not None
        assert len(advisory.skills) == 1
        assert "test" in advisory.skills
        assert len(advisory.budgets) == 1
        assert len(advisory.returns) == 1

        print("✓ Phase H advisory loads from ontology")


def test_expired_advisory_ttl():
    """Test that expired advisories are ignored by ConditionVectorBuilder."""
    skills = {
        "ttl_skill": Skill(
            "ttl_skill",
            "TTL Skill",
            "Test",
            10.0,
            11.0,
            15.0,
            100.0,
            1.0,
            0.7,
            0.3,
            0.6,
            0.7,
            0.3,
            1.0,
            120,
            "2025-01-01T00:00:00Z",
            "training",
        )
    }
    budgets = {"ttl_skill": ExplorationBudget("ttl_skill", 500.0, 100.0, 400.0, 0.5, 0.4, 0.1, 400)}
    returns = [SkillReturns("ttl_skill", 1.0, 10.0, 5.0, 1.0, 1.0, 0.0, 0.1, 1, 10.0)]
    advisory = PhaseHAdvisory(skills, budgets, returns, expiration_timestamp=0.0)  # Expired epoch

    builder = ConditionVectorBuilder()
    cv = builder.build(
        episode_config={"task_id": "task", "env_id": "env", "backend_id": "backend"},
        econ_state={"target_mpl": 60.0, "current_wage_parity": 1.0, "energy_budget_wh": 100.0},
        curriculum_phase="warmup",
        sima2_trust=0.5,
        datapack_metadata={"tags": ["frontier"]},
        episode_step=1,
        episode_metadata={"skill_id": "ttl_skill"},
        enable_phase_h_advisories=True,
        phase_h_advisory=advisory,
    )

    assert cv.exploration_uplift is None
    assert cv.skill_roi_estimate is None
    print("✓ Expired advisories do not influence ConditionVector")


def run_all_tests():
    """Run all smoke tests."""
    print("\n=== Phase H Advisory Integration Smoke Tests ===\n")

    test_advisory_multipliers_bounded()
    test_sampler_advisory_bounded()
    test_sampler_advisory_disabled_by_default()
    test_orchestrator_advisory_bounded()
    test_orchestrator_advisory_disabled_by_default()
    test_condition_fields_only_when_enabled()
    test_determinism()
    test_json_safe_export()
    test_expired_advisory_ttl()
    test_load_phase_h_advisory()

    print("\n=== All Tests Passed ===\n")


if __name__ == "__main__":
    run_all_tests()
