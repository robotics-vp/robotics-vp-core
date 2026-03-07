from src.constraints.constraint_set import ConstraintSet
from src.economics.functor import ObjectiveEconFunctor
from src.economics.pricing_sentinel import PricingSentinel, PricingTickInput
from src.economics.value_ledger import ValueLedger
from src.objectives.runtime_builder import ObjectiveRuntimeBuilder
from src.objectives.tensor import objective_tensor_from_axes


def test_value_ledger_receipt_is_stable_and_roundtrips(tmp_path):
    ledger_path = tmp_path / "value_ledger.jsonl"
    ledger = ValueLedger(ledger_path)
    builder = ObjectiveRuntimeBuilder()
    tensor = objective_tensor_from_axes(
        {"throughput": 12.0, "error": 0.1, "safety": 0.9, "energy": 0.6},
        schema=builder.schema,
        context={"episode_id": "ep_shadow", "run_id": "shadow_run", "timestamp": "2026-01-01T00:00:00+00:00"},
    )
    econ_tensor = ObjectiveEconFunctor(base_price_per_unit=3.0).map(tensor, constraint_flags=[], uncertainty=0.2)
    pricing_tick = PricingSentinel().emit_tick(
        PricingTickInput(
            run_id="shadow_run",
            episode_id="ep_shadow",
            objective_profile_id="balanced_contract",
            source_domain="synthetic",
            timestamp="2026-01-01T00:00:00+00:00",
            mode="episode",
            econ_tensor=econ_tensor,
            uncertainty=0.2,
            trust_score=0.9,
        )
    )
    constraint_summary = ConstraintSet.from_runtime(
        hard_constraints={"collision_rate": {"max": 0.1}},
    ).summary({"collision_rate": 0.0})

    receipt1 = ledger.build_receipt(
        event_type="episode_shadow_receipt",
        run_id="shadow_run",
        episode_id="ep_shadow",
        objective_profile_id="balanced_contract",
        objective_tensor=tensor,
        econ_tensor=econ_tensor,
        pricing_tick=pricing_tick,
        constraint_set=constraint_summary,
        regal_decision_summary={"overall_status": "pass"},
        datapack_id="dp_shadow",
        source_domain="synthetic",
        timestamp="2026-01-01T00:00:00+00:00",
    )
    receipt2 = ledger.build_receipt(
        event_type="episode_shadow_receipt",
        run_id="shadow_run",
        episode_id="ep_shadow",
        objective_profile_id="balanced_contract",
        objective_tensor=tensor,
        econ_tensor=econ_tensor,
        pricing_tick=pricing_tick,
        constraint_set=constraint_summary,
        regal_decision_summary={"overall_status": "pass"},
        datapack_id="dp_shadow",
        source_domain="synthetic",
        timestamp="2026-01-01T00:00:00+00:00",
    )

    assert receipt1.ledger_event_id == receipt2.ledger_event_id
    assert receipt1.receipt_hash == receipt2.receipt_hash

    ledger.append(receipt1)
    restored = ledger.load()
    assert len(restored) == 1
    assert restored[0].ledger_event_id == receipt1.ledger_event_id
