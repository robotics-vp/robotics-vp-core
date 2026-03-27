import json

import pytest

from src.contracts.schemas import PlanGainScheduleV1, PlanPolicyConfigV1, RegimeFeaturesV1
from src.orchestrator.homeostatic_plan_writer import build_plan_from_signals
from src.regal.knob_model_runtime import resolve_knob_model
from src.regal.knob_model_training import (
    generate_synthetic_knob_training_rows,
    train_knob_calibration_model,
)
from src.representation.homeostasis import ControlSignal, SignalBundle, SignalType


pytest.importorskip("torch")


def _write_package(tmp_path, *, benchmark_ready: bool = False) -> str:
    rows = generate_synthetic_knob_training_rows(12, seed=7)
    checkpoint_path = tmp_path / "knob_model.pt"
    train_knob_calibration_model(
        rows,
        epochs=2,
        hidden_dim=16,
        save_path=str(checkpoint_path),
    )
    package_path = tmp_path / "knob_model_package.json"
    package_path.write_text(
        json.dumps(
            {
                "package_id": "knob_model_test_pkg",
                "checkpoint_path": checkpoint_path.name,
                "benchmark_gate": {"ready": benchmark_ready},
                "execution_preconditions": {"ready": True},
                "promotion_stage": "shadow_candidate",
                "inference_contract": {
                    "helper_blend_policy": {
                        "shadow_candidate_helper_weight": 0.2,
                        "promoted_helper_weight": 0.55,
                    }
                },
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return str(package_path)


def test_resolve_knob_model_loads_runtime_package(tmp_path) -> None:
    package_path = _write_package(tmp_path, benchmark_ready=False)
    row = generate_synthetic_knob_training_rows(1, seed=9)[0]

    model = resolve_knob_model(use_learned=True, model_path=package_path)
    policy = model.predict(
        RegimeFeaturesV1(**row.regime_features),
        PlanPolicyConfigV1(**row.base_config),
    )

    assert policy.policy_source == "learned"
    assert policy.model_sha == "knob_model_test_pkg"
    assert policy.promotion_stage == "shadow_candidate"
    assert policy.trace is not None
    assert policy.trace["helper_weight"] == pytest.approx(0.2)
    assert policy.trace["benchmark_gate_ready"] is False


def test_resolve_knob_model_required_enforces_benchmark_gate(tmp_path) -> None:
    package_path = _write_package(tmp_path, benchmark_ready=False)

    with pytest.raises(ValueError, match="benchmark-gated"):
        resolve_knob_model(
            use_learned=True,
            model_path=package_path,
            required=True,
        )


def test_build_plan_from_signals_records_knob_runtime_details(tmp_path) -> None:
    package_path = _write_package(tmp_path, benchmark_ready=False)
    knob_model = resolve_knob_model(use_learned=True, model_path=package_path)
    signals = SignalBundle(
        signals=[
            ControlSignal(
                SignalType.DELTA_EPI_PER_FLOP,
                value=1e-6,
                metadata={
                    "transfer_pass": True,
                    "stability_pass": True,
                    "raw_delta": 0.02,
                    "flops_estimate": 1000.0,
                },
            )
        ]
    )
    config = PlanPolicyConfigV1(
        gain_schedule=PlanGainScheduleV1(
            full_multiplier=1.5,
            conservative_multiplier=1.1,
            cooldown_steps=3,
        ),
        default_weights={"manipulation": 0.5, "navigation": 0.5},
    )

    _, gate_status = build_plan_from_signals(
        signals,
        config,
        knob_model=knob_model,
    )

    assert gate_status.knob_policy_used == "learned"
    assert gate_status.knob_policy is not None
    assert gate_status.knob_regime_features is not None
    assert gate_status.knob_base_config is not None
    assert gate_status.knob_policy.trace is not None
