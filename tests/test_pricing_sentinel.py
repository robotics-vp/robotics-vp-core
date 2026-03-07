from src.economics.econ_tensor import EconTensor, EconTensorSchema
from src.economics.pricing_sentinel import PricingPolicy, PricingSentinel, PricingTickInput


def _econ_tensor() -> EconTensor:
    return EconTensor(
        values=[10.0, 12.0, 0.8, 1.0, 0.2],
        schema=EconTensorSchema(),
        context={"episode_id": "ep_shadow"},
    )


def test_pricing_sentinel_is_deterministic():
    sentinel = PricingSentinel(PricingPolicy())
    tick_input = PricingTickInput(
        run_id="shadow_run",
        episode_id="ep_shadow",
        objective_profile_id="balanced_contract",
        source_domain="synthetic",
        timestamp="2026-01-01T00:00:00+00:00",
        mode="episode",
        econ_tensor=_econ_tensor(),
        uncertainty=0.2,
        trust_score=0.9,
    )
    tick1 = sentinel.emit_tick(tick_input)
    tick2 = sentinel.emit_tick(tick_input)

    assert tick1.to_dict() == tick2.to_dict()


def test_pricing_sentinel_monotonic_sanity():
    sentinel = PricingSentinel(PricingPolicy())
    clean = sentinel.emit_tick(
        PricingTickInput(
            run_id="shadow_run",
            episode_id="ep_shadow",
            objective_profile_id="balanced_contract",
            source_domain="synthetic",
            timestamp="2026-01-01T00:00:00+00:00",
            mode="episode",
            econ_tensor=_econ_tensor(),
            uncertainty=0.1,
            trust_score=0.9,
        )
    )
    uncertain = sentinel.emit_tick(
        PricingTickInput(
            run_id="shadow_run",
            episode_id="ep_shadow",
            objective_profile_id="balanced_contract",
            source_domain="synthetic",
            timestamp="2026-01-01T00:00:00+00:00",
            mode="episode",
            econ_tensor=_econ_tensor(),
            uncertainty=0.6,
            trust_score=0.9,
        )
    )
    constrained = sentinel.emit_tick(
        PricingTickInput(
            run_id="shadow_run",
            episode_id="ep_shadow",
            objective_profile_id="balanced_contract",
            source_domain="synthetic",
            timestamp="2026-01-01T00:00:00+00:00",
            mode="episode",
            econ_tensor=_econ_tensor(),
            uncertainty=0.1,
            trust_score=0.9,
            constraint_flags=[{"severity": "hard", "axis": "collision_rate", "flag": "above_max"}],
        )
    )

    assert uncertain.net_customer_rate < clean.net_customer_rate
    assert uncertain.confidence < clean.confidence
    assert constrained.net_customer_rate < clean.net_customer_rate
