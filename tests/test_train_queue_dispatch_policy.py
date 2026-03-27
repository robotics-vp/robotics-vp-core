from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

from scripts.train_queue_dispatch_policy import _run_training, parse_args
from src.training.regal_training_runner import TrainingRunConfig, run_training_with_regality
from src.utils.config_digest import sha256_json


def _payload_path(tmp_path: Path) -> Path:
    payload = {
        "queue_name": "shadow_advisory_queue",
        "entries": [
            {
                "episode_id": "ep_a",
                "priority_score": 0.9,
                "replay_action": "upweight",
                "tags": ["frontier_candidate"],
                "metadata": {
                    "promotion_stage": "advisory",
                    "influence_source": "heuristic",
                    "deploy_recommendation": "allow_shadow",
                    "pricing_recommendation": "publish",
                    "datapack_recommendation": "keep",
                    "evidence": {
                        "receipt_feedback": {
                            "deployment_outcome": {
                                "task_success": True,
                                "objective_satisfied": True,
                                "realized_value": 1.4,
                                "pricing_accepted": True,
                            }
                        }
                    },
                },
            }
        ],
    }
    path = tmp_path / "live_queue_selection.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def test_train_queue_dispatch_policy_emits_runtime_package(tmp_path: Path) -> None:
    queue_path = _payload_path(tmp_path)
    args = parse_args(
        [
            "--queue-json",
            str(queue_path),
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
    assert package["inference_contract"]["target_contract"] == "queue_dispatch_policy_v1"


def test_regality_wrapper_registers_queue_dispatch_policy_artifacts(tmp_path: Path) -> None:
    queue_path = _payload_path(tmp_path)
    output_dir = tmp_path / "runner"

    def _wrapped(runner):
        args = parse_args(
            [
                "--queue-json",
                str(queue_path),
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
        plan_sha=sha256_json({"plan": "queue_dispatch_policy_test"}),
        plan_id="queue_dispatch_policy_test",
    )

    manifest = json.loads((output_dir / "training_runtime_manifest.json").read_text(encoding="utf-8"))
    assert manifest["training_kind"] == "queue_dispatch_policy"
    assert manifest["artifact_paths"]["queue_dispatch_policy_runtime_package"].endswith(
        "queue_dispatch_policy_package.json"
    )
