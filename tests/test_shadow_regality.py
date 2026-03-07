from src.regality import MetaRegalController
from src.regality.shadow_nodes import (
    ObjectiveIntegrityRegal,
    PricingTruthRegal,
    ShadowRegalContext,
    ShadowRegalStatus,
)


def _context(*, scalarized_upstream: bool = False, net_above_gross: bool = False, credit_without_gain: bool = False) -> ShadowRegalContext:
    pricing_tick = {
        "task_hour_price_tick": 40.0,
        "constraint_adjustment": -5.0,
        "uncertainty_adjustment": -3.0,
        "data_share_credit": 0.2 if not credit_without_gain else 5.0,
        "net_customer_rate": 31.0 if not net_above_gross else 45.0,
        "confidence": 0.7,
        "metadata": {"uncertainty": 0.2},
        "mode": "episode",
    }
    return ShadowRegalContext(
        run_id="shadow_run",
        episode_id="ep_shadow",
        source_domain="synthetic",
        objective_tensor={
            "schema": {"axes": ["throughput", "error", "safety", "energy"]},
            "context": {
                "task_id": "shadow_kitting",
                "episode_id": "ep_shadow",
                "env_id": "workcell_simple",
                "world_id": "workcell_assembly_bench_simple",
                "robot_id": "shadow_sim_arm_v1",
                "source_domain": "synthetic",
                "seed": 42,
                "run_id": "shadow_run",
                "timestamp": "2026-01-01T00:00:00+00:00",
                "schema_version_hash": "abc123",
            },
            "provenance": {"builder": "objective_runtime_builder_v1"},
        },
        objective_profile={
            "profile": {
                "profile_id": "balanced_contract",
                "weights": {"throughput": 1.0, "error": 1.0, "safety": 1.0, "energy": 1.0},
            }
        },
        compile_artifact={
            "objective_profile_id": "balanced_contract",
            "scalarization_boundary": "contract_boundary",
            "scalarized_upstream": scalarized_upstream,
            "scalar_reward": 1.8,
        },
        constraint_set={"hard_bounds": {"collision_rate": {"max": 0.1}}},
        constraint_flags=[{"severity": "soft", "axis": "constraint_error_rate", "flag": "above_max"}],
        econ_tensor={"axes": {"value_earned": 3.0, "marginal_frontier_gain": 0.4 if not credit_without_gain else 0.0}},
        pricing_ticks=[pricing_tick],
        datapack_credit_update={
            "data_share_credit": pricing_tick["data_share_credit"],
            "marginal_frontier_gain": 0.4 if not credit_without_gain else 0.0,
            "quality_score": 0.9,
        },
        episode_metrics={
            "throughput_units_per_hour": 12.0,
            "error_rate": 0.1,
            "safety_score": 0.85,
            "duration_s": 720.0,
        },
        provenance={"trace_hash": "abc"},
    )


def test_shadow_regality_pass_case():
    context = _context()
    controller = MetaRegalController()
    decision = controller.evaluate(context)

    assert decision.overall_status in {ShadowRegalStatus.PASS, ShadowRegalStatus.WARN}
    assert decision.deploy_recommendation in {"allow_shadow", "require_review"}


def test_objective_integrity_regal_fails_on_early_scalarization():
    decision = ObjectiveIntegrityRegal().evaluate(_context(scalarized_upstream=True))
    assert decision.status == ShadowRegalStatus.FAIL


def test_pricing_truth_regal_fails_on_optimistic_net_rate():
    decision = PricingTruthRegal().evaluate(_context(net_above_gross=True))
    assert decision.status == ShadowRegalStatus.FAIL


def test_meta_regal_denies_shadow_on_integrity_or_pricing_failures():
    controller = MetaRegalController()
    decision = controller.evaluate(_context(scalarized_upstream=True, net_above_gross=True, credit_without_gain=True))
    assert decision.deploy_recommendation == "deny_shadow"
    assert decision.pricing_recommendation == "suppress"
