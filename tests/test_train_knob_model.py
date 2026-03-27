import json
from argparse import Namespace

import pytest

from scripts.train_knob_model import _run_training
from src.training.regal_training_runner import TrainingRunConfig, run_training_with_regality


pytest.importorskip("torch")


def _write_receipt(path) -> str:
    receipt_path = path / "knob_policy_receipt.json"
    receipt_path.write_text(
        json.dumps(
            {
                "schema_version": "knob_policy_receipt_v1",
                "receipt_id": "smoke_run:homeostatic_v1",
                "target_source": "runtime_receipt",
                "promotion_stage": "shadow_candidate",
                "regime_features": {
                    "audit_delta_success": -0.1,
                    "audit_delta_error": 0.05,
                    "audit_success_rate": 0.6,
                    "exposure_count": 12,
                    "datapack_count": 4,
                    "probe_delta_epi_per_flop": 1e-6,
                    "probe_stability_pass": True,
                    "probe_transfer_pass": False,
                    "regal_spec_score": 0.8,
                    "regal_coherence_score": 0.75,
                    "regal_hack_prob": 0.05,
                    "objective_profile": "balanced",
                    "task_family_weights": {
                        "manipulation": 0.5,
                        "navigation": 0.5,
                    },
                },
                "base_config": {
                    "gain_schedule": {
                        "full_multiplier": 1.5,
                        "conservative_multiplier": 1.1,
                        "cooldown_steps": 3,
                    },
                    "default_weights": {
                        "manipulation": 0.5,
                        "navigation": 0.5,
                    },
                },
                "knob_policy": {
                    "policy_source": "heuristic_fallback",
                    "promotion_stage": "shadow_candidate",
                    "gain_multiplier_override": 1.2,
                    "conservative_multiplier_override": 1.0,
                    "patience_override": 4,
                    "clamped": False,
                    "clamp_reasons": [],
                },
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return str(receipt_path)


def test_train_knob_model_emits_runtime_package(tmp_path) -> None:
    receipt_path = _write_receipt(tmp_path)
    output_dir = tmp_path / "knob_training"
    args = Namespace(
        dataset_json=None,
        receipt_path=receipt_path,
        synthetic_samples=2,
        epochs=2,
        lr=1e-3,
        hidden_dim=16,
        save_dir=str(output_dir),
        run_name="knob_model_test",
        seed=7,
        skip_regal_runner=False,
    )

    holder = {}

    def _wrapped(runner) -> None:
        holder["payload"] = _run_training(args, runner)

    run_training_with_regality(
        training_fn=_wrapped,
        config=TrainingRunConfig(
            output_dir=str(output_dir),
            seed=7,
            num_episodes=3,
            training_steps=2,
            fail_on_verify_error=False,
        ),
        plan_sha="plan_sha",
        plan_id="knob_model_test",
    )

    manifest = json.loads((output_dir / "training_runtime_manifest.json").read_text(encoding="utf-8"))
    package = json.loads((output_dir / "knob_model_package.json").read_text(encoding="utf-8"))
    dataset_summary = json.loads(
        (output_dir / "knob_model_dataset_summary.json").read_text(encoding="utf-8")
    )

    assert manifest["training_kind"] == "knob_model"
    assert manifest["artifact_paths"]["knob_model_runtime_package"].endswith("knob_model_package.json")
    assert manifest["artifact_paths"]["knob_model_dataset"].endswith("knob_model_dataset.json")
    assert package["promotion_stage"] == "shadow_candidate"
    assert package["inference_contract"]["runtime_receipt_contract"] == "knob_policy_receipt_v1"
    assert dataset_summary["target_source_counts"]["runtime_receipt"] == 1
    assert holder["payload"]["benchmark_gate_ready"] is False
