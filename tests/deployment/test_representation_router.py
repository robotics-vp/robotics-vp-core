from src.deployment.representation_router import route_representation_source
from src.deployment.task_economics import (
    DecisionClass,
    EvidenceSource,
    EvidenceSourceState,
    TaskEconomics,
)


def _state(
    source: EvidenceSource,
    *,
    available: bool = True,
    sufficiency: float = 0.8,
    time_cost: float = 0.1,
    battery_cost: float = 0.1,
    compute_cost: float = 0.1,
    failure_risk: float = 0.1,
) -> EvidenceSourceState:
    return EvidenceSourceState(
        source=source,
        available=available,
        source_sufficiency=sufficiency,
        time_cost=time_cost,
        battery_cost=battery_cost,
        compute_cost=compute_cost,
        failure_risk=failure_risk,
    )


def test_high_value_uncertain_failure_cost_prefers_real_or_human_evidence() -> None:
    task = TaskEconomics(
        task_id="fragile_high_value_pick",
        task_value=0.95,
        uncertainty=0.9,
        failure_cost=0.95,
        time_cost=0.2,
        battery_cost=0.2,
        compute_cost=0.2,
        evidence_sufficiency=0.7,
    )

    decision = route_representation_source(
        task,
        [
            _state(EvidenceSource.PRIOR_REPLAY, sufficiency=0.82),
            _state(EvidenceSource.SIMULATION, sufficiency=0.82),
            _state(EvidenceSource.REAL_OBSERVATION, sufficiency=0.82),
            _state(EvidenceSource.HUMAN_OPERATOR_INPUT, sufficiency=0.82),
        ],
    )

    assert decision.selected_source in {
        EvidenceSource.REAL_OBSERVATION,
        EvidenceSource.HUMAN_OPERATOR_INPUT,
    }
    if decision.selected_source == EvidenceSource.HUMAN_OPERATOR_INPUT:
        assert decision.decision_class == DecisionClass.REQUIRE_HUMAN_REVIEW
    else:
        assert decision.decision_class == DecisionClass.USE
    assert decision.receipt["training_run"] is False
    assert decision.receipt["gpu_execution"] is False
    assert decision.receipt["hardware_execution"] is False


def test_generated_video_rejected_when_resource_cost_dominates() -> None:
    task = TaskEconomics(
        task_id="tight_battery_branch_check",
        task_value=0.6,
        uncertainty=0.7,
        failure_cost=0.4,
        time_cost=0.95,
        battery_cost=0.95,
        compute_cost=0.95,
        evidence_sufficiency=0.6,
    )

    decision = route_representation_source(
        task,
        [
            _state(
                EvidenceSource.GENERATED_VIDEO,
                sufficiency=0.95,
                time_cost=0.95,
                battery_cost=0.95,
                compute_cost=0.95,
                failure_risk=0.3,
            ),
            _state(EvidenceSource.GEOMETRY, sufficiency=0.7, time_cost=0.02),
        ],
    )

    assert decision.selected_source == EvidenceSource.GEOMETRY
    generated_video_rejection = next(
        row
        for row in decision.rejected_sources
        if row.source == EvidenceSource.GENERATED_VIDEO
    )
    assert "economic_cost_dominates_evidence_value" in generated_video_rejection.reasons


def test_generated_video_rejected_when_sufficiency_is_too_low() -> None:
    task = TaskEconomics(
        task_id="sufficiency_floor",
        task_value=0.5,
        uncertainty=0.5,
        failure_cost=0.5,
        time_cost=0.2,
        battery_cost=0.2,
        compute_cost=0.2,
        evidence_sufficiency=0.75,
    )

    decision = route_representation_source(
        task,
        [
            _state(EvidenceSource.GENERATED_VIDEO, sufficiency=0.3),
            _state(EvidenceSource.PRIOR_REPLAY, sufficiency=0.8),
        ],
    )

    assert decision.selected_source == EvidenceSource.PRIOR_REPLAY
    generated_video_rejection = next(
        row
        for row in decision.rejected_sources
        if row.source == EvidenceSource.GENERATED_VIDEO
    )
    assert any(
        reason.startswith("source_sufficiency_below_required")
        for reason in generated_video_rejection.reasons
    )


def test_prior_replay_wins_low_uncertainty_low_value_when_sufficient() -> None:
    task = TaskEconomics(
        task_id="known_low_value_repetition",
        task_value=0.15,
        uncertainty=0.12,
        failure_cost=0.1,
        time_cost=0.4,
        battery_cost=0.4,
        compute_cost=0.4,
        evidence_sufficiency=0.5,
    )

    decision = route_representation_source(
        task,
        [
            _state(EvidenceSource.REAL_OBSERVATION, sufficiency=0.76),
            _state(EvidenceSource.SIMULATION, sufficiency=0.76),
            _state(EvidenceSource.PRIOR_REPLAY, sufficiency=0.76),
        ],
    )

    assert decision.selected_source == EvidenceSource.PRIOR_REPLAY
    assert decision.decision_class == DecisionClass.USE


def test_geometry_or_simulation_can_win_when_cheap_and_sufficient() -> None:
    task = TaskEconomics(
        task_id="cheap_structure_probe",
        task_value=0.5,
        uncertainty=0.55,
        failure_cost=0.25,
        time_cost=0.5,
        battery_cost=0.5,
        compute_cost=0.5,
        evidence_sufficiency=0.6,
    )

    decision = route_representation_source(
        task,
        [
            _state(EvidenceSource.PRIOR_REPLAY, sufficiency=0.55),
            _state(EvidenceSource.GEOMETRY, sufficiency=0.76, time_cost=0.01),
            _state(EvidenceSource.SIMULATION, sufficiency=0.78, time_cost=0.02),
        ],
    )

    assert decision.selected_source in {
        EvidenceSource.GEOMETRY,
        EvidenceSource.SIMULATION,
    }
    assert decision.decision_class == DecisionClass.USE


def test_unavailable_when_no_source_is_available_or_sufficient() -> None:
    task = TaskEconomics(
        task_id="blocked_context",
        task_value=0.7,
        uncertainty=0.7,
        failure_cost=0.7,
        time_cost=0.2,
        battery_cost=0.2,
        compute_cost=0.2,
        evidence_sufficiency=0.8,
    )

    decision = route_representation_source(
        task,
        [
            _state(EvidenceSource.REAL_OBSERVATION, available=False, sufficiency=0.9),
            _state(EvidenceSource.HUMAN_OPERATOR_INPUT, available=False, sufficiency=0.9),
            _state(EvidenceSource.GEOMETRY, available=True, sufficiency=0.4),
        ],
    )

    assert decision.selected_source == EvidenceSource.UNAVAILABLE
    assert decision.decision_class == DecisionClass.UNAVAILABLE
    assert decision.blocker_summary["no_sufficient_source_available"] is True


def test_tie_breaking_is_deterministic() -> None:
    task = TaskEconomics(
        task_id="neutral_exact_tie",
        task_value=0.5,
        uncertainty=0.5,
        failure_cost=0.5,
        time_cost=0.0,
        battery_cost=0.0,
        compute_cost=0.0,
        evidence_sufficiency=0.5,
    )
    sources = [
        _state(
            EvidenceSource.SIMULATION,
            sufficiency=0.7,
            time_cost=0.0,
            battery_cost=0.0,
            compute_cost=0.0,
            failure_risk=0.0,
        ),
        _state(
            EvidenceSource.GEOMETRY,
            sufficiency=0.7,
            time_cost=0.0,
            battery_cost=0.0,
            compute_cost=0.0,
            failure_risk=0.0,
        ),
    ]

    first = route_representation_source(task, sources)
    second = route_representation_source(task, list(reversed(sources)))

    assert first.selected_source == EvidenceSource.GEOMETRY
    assert second.selected_source == EvidenceSource.GEOMETRY
    assert first.receipt_sha == second.receipt_sha
    simulation_rejection = next(
        row for row in first.rejected_sources if row.source == EvidenceSource.SIMULATION
    )
    assert "tie_lost_to_deterministic_order" in simulation_rejection.reasons


def test_input_and_receipt_sha_stable_and_materially_sensitive() -> None:
    task = TaskEconomics(
        task_id="sha_task",
        task_value=0.45,
        uncertainty=0.4,
        failure_cost=0.35,
        time_cost=0.2,
        battery_cost=0.2,
        compute_cost=0.2,
        evidence_sufficiency=0.55,
    )
    sources = [_state(EvidenceSource.PRIOR_REPLAY, sufficiency=0.8)]

    first = route_representation_source(task, sources)
    second = route_representation_source(task, sources)
    changed = route_representation_source(
        TaskEconomics(
            task_id="sha_task",
            task_value=0.45,
            uncertainty=0.4,
            failure_cost=0.35,
            time_cost=0.2,
            battery_cost=0.2,
            compute_cost=0.3,
            evidence_sufficiency=0.55,
        ),
        sources,
    )

    assert first.input_sha == second.input_sha
    assert first.receipt_sha == second.receipt_sha
    assert first.input_sha != changed.input_sha
    assert first.receipt_sha != changed.receipt_sha
