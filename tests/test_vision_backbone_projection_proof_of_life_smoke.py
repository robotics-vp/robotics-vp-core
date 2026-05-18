from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.perception_proof_of_life_utils import make_mock_lerobot_episode
from src.dataset_bridges.lerobot_bridge import lerobot_rows_from_replay


REPO_ROOT = Path(__file__).resolve().parent.parent


def _to_jsonable(value):
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _write_mock_lerobot_rows_bundle(
    path: Path,
    *,
    episode_count: int,
    num_steps: int,
) -> None:
    rows = []
    for episode_idx in range(episode_count):
        episode, steps = make_mock_lerobot_episode(
            episode_idx=episode_idx,
            num_steps=num_steps,
            seed=42,
            camera_format="droid",
        )
        rows.extend(lerobot_rows_from_replay(episode, steps))
    path.write_text(
        "\n".join(json.dumps(_to_jsonable(row)) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_vision_backbone_projection_proof_of_life_emits_typed_artifacts(tmp_path) -> None:
    subprocess.run(
        [
            sys.executable,
            "scripts/smoke_test_vision_backbone_projection_seam.py",
            "--steps",
            "30",
            "--artifact-dir",
            str(tmp_path),
            "--require-loss-decrease",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    summary = json.loads(
        (tmp_path / "vision_backbone_projection_seam_proof_of_life.json").read_text(
            encoding="utf-8"
        )
    )
    metric_report = json.loads(
        (tmp_path / "vision_backbone_projection_metric_report.json").read_text(
            encoding="utf-8"
        )
    )
    benchmark_evidence = json.loads(
        (tmp_path / "vision_backbone_projection_benchmark_evidence.json").read_text(
            encoding="utf-8"
        )
    )
    manifest = json.loads(
        (tmp_path / "training_runtime_manifest.json").read_text(encoding="utf-8")
    )

    assert summary["schema_version"] == "perception_seam_proof_of_life_v2"
    assert summary["seam_type"] == "vision_backbone_projection"
    assert summary["promotion_eligible"] is False
    assert summary["promotion_claim"] == "explicitly_held"
    assert summary["data_source"] == "synthetic"
    assert summary["loss_proof"]["loss_decreased"] is True
    assert Path(summary["artifact_paths"]["checkpoint"]).exists()

    assert metric_report["schema_version"] == "perception_seam_metric_report_v1"
    assert metric_report["seam_type"] == "vision_backbone_projection"
    assert metric_report["evidence_source_provisional"] is True
    assert metric_report["promotion_eligible"] is False
    assert metric_report["metrics"]["data_source"] == "synthetic"
    assert metric_report["metrics"]["loss_decreased"] is True

    assert benchmark_evidence["schema_version"] == "perception_benchmark_evidence_v1"
    assert benchmark_evidence["subsystem_key"] == "vision_backbone_projection"
    assert benchmark_evidence["benchmark_evidence_present"] is True
    assert benchmark_evidence["evidence_source_provisional"] is True
    assert benchmark_evidence["promotion_eligible"] is False
    assert (
        benchmark_evidence["metadata"]["promotion_claim"]
        == "not_implied_by_local_proof_of_life"
    )

    assert manifest["schema_version"] == "training_runtime_manifest_v1"
    assert (
        manifest["training_kind"]
        == "perception_vision_backbone_projection_proof_of_life"
    )
    assert manifest["status"] == "completed"
    assert manifest["metadata"]["epistemic_status"] == "proof_of_life"
    assert manifest["metadata"]["mock_data"] is False
    assert (
        manifest["replay_dataset_summary"]["dataset_kind"]
        == "synthetic_vision_backbone_projection"
    )
    assert manifest["promotion_policy_snapshot"]["promotion_eligible"] is False
    assert manifest["inferential_learnability_summary"]["loss_decreased"] is True
    assert Path(manifest["promotion_evidence_path"]).exists()


def test_vision_backbone_projection_proof_of_life_accepts_local_lerobot_rows(
    tmp_path,
) -> None:
    rows_path = tmp_path / "mock_lerobot_rows.jsonl"
    _write_mock_lerobot_rows_bundle(rows_path, episode_count=4, num_steps=24)

    subprocess.run(
        [
            sys.executable,
            "scripts/smoke_test_vision_backbone_projection_seam.py",
            "--steps",
            "30",
            "--artifact-dir",
            str(tmp_path / "artifacts"),
            "--data-source",
            "local_lerobot_rows",
            "--lerobot-rows-path",
            str(rows_path),
            "--max-episodes",
            "4",
            "--max-steps-per-episode",
            "24",
            "--require-loss-decrease",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    summary = json.loads(
        (
            tmp_path
            / "artifacts"
            / "vision_backbone_projection_seam_proof_of_life.json"
        ).read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (tmp_path / "artifacts" / "training_runtime_manifest.json").read_text(
            encoding="utf-8"
        )
    )

    assert summary["data_source"] == "local_lerobot_rows"
    assert (
        summary["data"]["dataset_kind"]
        == "local_lerobot_rows_vision_backbone_projection"
    )
    assert summary["loss_proof"]["loss_decreased"] is True
    assert summary["data"]["lerobot_rows_path"] == str(rows_path.resolve())

    assert manifest["metadata"]["data_source"] == "local_lerobot_rows"
    assert manifest["metadata"]["external_dataset_required"] is True
    assert manifest["metadata"]["lerobot_rows_path"] == str(rows_path.resolve())
    assert manifest["replay_dataset_summary"]["episode_count"] == 4
