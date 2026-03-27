from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

from scripts.train_sampler_policy import _run_training, parse_args
from src.rl.episode_sampling import DataPackRLSampler
from src.training.regal_training_runner import TrainingRunConfig, run_training_with_regality
from src.utils.config_digest import sha256_json


def _descriptor(pack_id: str, frontier_gain: float) -> dict:
    return {
        "pack_id": pack_id,
        "env_name": "sampler_env",
        "task_type": "sampler_task",
        "engine_type": "synthetic",
        "backend": "synthetic",
        "objective_vector": [1.0, 1.0, 1.0, 1.0, 0.0],
        "tier": 1,
        "trust_score": 0.7 + 0.1 * frontier_gain,
        "sampling_weight": 1.0 + frontier_gain,
        "delta_mpl": frontier_gain,
        "delta_J": frontier_gain * 0.75,
        "episode_length": 18,
        "unified_quality_weight": 1.0,
        "unified_quality_eligible": True,
        "enrichment": {
            "novelty_tags": [{"novelty_score": min(1.0, frontier_gain), "expected_mpl_gain": frontier_gain * 3.0}],
            "supervision_hints": {"priority_level": "high", "suggested_weight_multiplier": 1.0},
            "coherence_score": 0.75,
        },
    }


def _receipt_path(tmp_path: Path) -> Path:
    sampler = DataPackRLSampler(
        existing_descriptors=[
            _descriptor("ep_a", 1.0),
            _descriptor("ep_b", 0.4),
            _descriptor("ep_c", 0.1),
        ],
        default_strategy="balanced",
    )
    sampler.sample_batch(batch_size=3, seed=2)
    receipt = dict(sampler.last_sampler_policy_artifact or {})
    receipt["strategy_targets"]["frontier_prioritized"] = 1.0
    receipt["strategy_targets"]["balanced"] = 0.0
    path = tmp_path / "sampler_policy_receipt.json"
    path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return path


def test_train_sampler_policy_emits_runtime_package(tmp_path: Path) -> None:
    receipt_path = _receipt_path(tmp_path)
    args = parse_args(
        [
            "--receipt-json",
            str(receipt_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--skip-regal-runner",
            "--epochs",
            "2",
        ]
    )

    result = _run_training(args, runner=None)
    package = json.loads(Path(result["runtime_package"]).read_text(encoding="utf-8"))

    assert package["promotion_stage"] == "shadow_candidate"
    assert package["inference_contract"]["target_contract"] == "sampler_policy_v1"


def test_regality_wrapper_registers_sampler_policy_artifacts(tmp_path: Path) -> None:
    receipt_path = _receipt_path(tmp_path)
    output_dir = tmp_path / "runner"

    def _wrapped(runner):
        args = parse_args(
            [
                "--receipt-json",
                str(receipt_path),
                "--output-dir",
                str(output_dir),
                "--epochs",
                "2",
            ]
        )
        _run_training(args, runner)

    run_training_with_regality(
        training_fn=_wrapped,
        config=TrainingRunConfig(
            output_dir=str(output_dir),
            seed=7,
            num_episodes=1,
            training_steps=2,
            fail_on_verify_error=False,
        ),
        plan_sha=sha256_json({"plan": "sampler_policy_test"}),
        plan_id="sampler_policy_test",
    )

    manifest = json.loads((output_dir / "training_runtime_manifest.json").read_text(encoding="utf-8"))
    assert manifest["training_kind"] == "sampler_policy"
    assert manifest["artifact_paths"]["sampler_policy_runtime_package"].endswith("sampler_policy_package.json")
