#!/usr/bin/env python3
"""Run a local Stage-1 replay/export readiness sweep over manifest variants."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

if __package__ is None or __package__ == "":
    REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

from scripts.export_governed_video_stage1_bridges import (  # noqa: E402
    export_governed_video_stage1_bridges,
)
from scripts.run_stage1_pipeline import run_stage1_pipeline  # noqa: E402

SWEEP_VERSION = "stage1_bridge_readiness_sweep_v1"


def _scene_tracks_payload() -> Dict[str, Any]:
    return {
        "track_ids": ["drawer_track", "vase_track"],
        "entity_types": [0, 0],
        "class_ids": [0, 1],
        "class_names": ["drawer", "vase"],
        "poses_R": [
            [
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            ],
            [
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            ],
        ],
        "poses_t": [
            [[0.0, 0.0, 0.0], [0.2, 0.1, 0.0]],
            [[0.05, 0.0, 0.0], [0.2, 0.1, 0.0]],
        ],
        "scales": [
            [[1.0, 1.0, 1.0], [0.8, 0.8, 0.8]],
            [[1.0, 1.0, 1.0], [0.8, 0.8, 0.8]],
        ],
        "visibility": [[1.0, 1.0], [1.0, 1.0]],
        "occlusion": [[0.0, 0.0], [0.0, 0.0]],
        "ir_loss": [[0.0, 0.0], [0.0, 0.0]],
        "converged": [[1.0, 1.0], [1.0, 1.0]],
    }


def _sensor_bundle(camera: str = "front") -> Dict[str, Any]:
    return {
        "cameras": [camera],
        "intrinsics": {camera: f"intrinsics://{camera}"},
        "extrinsics": {camera: f"extrinsics://{camera}"},
        "depth_unit": "meters",
    }


def build_manifest_variants() -> list[Dict[str, Any]]:
    """Return manifest variants covering the current pre-Economic-WM truth gates."""

    base = {
        "timestamp": 1_700_000_001.0,
        "task_type": "drawer_vase",
        "instruction": "Open the drawer without hitting the vase.",
        "demonstrator": "human_expert",
        "source_type": "video_manifest",
    }
    return [
        {
            **base,
            "episode_id": "calibrated_inline_real",
            "video_path": "/tmp/calibrated_inline_real.mp4",
            "metadata": {
                "duration_s": 10.0,
                "success": True,
                "num_frames": 4,
                "scene_tracks_backend": "real",
                "vision_backbone_selected": "real",
                "teacher_runtime_backend_selected": "unavailable",
                "sensor_bundle": _sensor_bundle("front"),
                "scene_tracks_v1": _scene_tracks_payload(),
            },
        },
        {
            **base,
            "episode_id": "top_level_calibration_real",
            "video_path": "/tmp/top_level_calibration_real.mp4",
            "scene_tracks_v1": _scene_tracks_payload(),
            "camera": "wrist",
            "intrinsics_ref": "intrinsics://wrist",
            "extrinsics_ref": "extrinsics://wrist",
            "metadata": {
                "duration_s": 9.0,
                "success": True,
                "num_frames": 4,
                "scene_tracks_backend": "real",
                "vision_backbone_selected": "real",
                "teacher_runtime_backend_selected": "unavailable",
            },
        },
        {
            **base,
            "episode_id": "missing_calibration_real",
            "video_path": "/tmp/missing_calibration_real.mp4",
            "metadata": {
                "duration_s": 8.0,
                "success": True,
                "num_frames": 4,
                "scene_tracks_backend": "real",
                "vision_backbone_selected": "real",
                "teacher_runtime_backend_selected": "unavailable",
                "scene_tracks_v1": _scene_tracks_payload(),
            },
        },
        {
            **base,
            "episode_id": "unknown_artifact_calibrated",
            "video_path": "/tmp/unknown_artifact_calibrated.mp4",
            "metadata": {
                "duration_s": 7.0,
                "success": True,
                "num_frames": 4,
                "vision_backbone_selected": "real",
                "teacher_runtime_backend_selected": "unavailable",
                "sensor_bundle": _sensor_bundle("front"),
                "scene_tracks_path": "/tmp/unknown_scene_tracks_v1.npz",
            },
        },
        {
            **base,
            "episode_id": "passthrough_calibrated",
            "video_path": "/tmp/passthrough_calibrated.mp4",
            "metadata": {
                "duration_s": 6.0,
                "success": True,
                "num_frames": 4,
                "scene_tracks_backend": "passthrough",
                "vision_backbone_selected": "real",
                "teacher_runtime_backend_selected": "unavailable",
                "sensor_bundle": _sensor_bundle("front"),
                "scene_tracks_v1": _scene_tracks_payload(),
            },
        },
    ]


_EXPECTED = {
    "calibrated_inline_real": {
        "benchmark_ready": True,
        "calibration_class": "camera_calibrated",
        "grounding_class": "real_scene_tracks_joined",
        "reconstruction_training_eligible": True,
        "recommended_mode": "stage1_datapack",
    },
    "top_level_calibration_real": {
        "benchmark_ready": True,
        "calibration_class": "camera_calibrated",
        "grounding_class": "real_scene_tracks_joined",
        "reconstruction_training_eligible": True,
        "recommended_mode": "stage1_datapack",
    },
    "missing_calibration_real": {
        "benchmark_ready": False,
        "calibration_class": "camera_missing",
        "grounding_class": "real_scene_tracks_joined",
        "reconstruction_training_eligible": False,
        "recommended_mode": "shadow_stage1_datapack",
        "blocking_precondition": "blocked::camera_calibration_missing",
    },
    "unknown_artifact_calibrated": {
        "benchmark_ready": False,
        "calibration_class": "camera_calibrated",
        "grounding_class": "scene_tracks_ref_unqualified",
        "reconstruction_training_eligible": False,
        "recommended_mode": "shadow_stage1_datapack",
    },
    "passthrough_calibrated": {
        "benchmark_ready": False,
        "calibration_class": "camera_calibrated",
        "grounding_class": "scene_tracks_ref_unqualified",
        "reconstruction_training_eligible": False,
        "recommended_mode": "shadow_stage1_datapack",
        "blocking_precondition": "blocked::scene_tracks_passthrough_selected",
    },
}


def _json_rows(path: Path) -> list[Dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _bridge_rows(path: Path) -> list[Dict[str, Any]]:
    return _json_rows(path)


def _first_bridge_step_by_video(
    rlds_rows: Iterable[Mapping[str, Any]],
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for episode in rlds_rows:
        episode_id = str(episode.get("episode_id", ""))
        video_id = episode_id.split(":", 1)[0]
        steps = list(episode.get("steps", []) or [])
        if video_id and steps:
            result[video_id] = steps[0]
    return result


def _first_lerobot_row_by_video(
    lerobot_rows: Iterable[Mapping[str, Any]],
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for row in lerobot_rows:
        episode_id = str(row.get("episode_id", ""))
        video_id = episode_id.split(":", 1)[0]
        if video_id and video_id not in result:
            result[video_id] = row
    return result


def run_stage1_bridge_readiness_sweep(
    *,
    output_dir: str | Path,
    proposals_per_video: int = 1,
    clean: bool = True,
    quiet: bool = True,
) -> Dict[str, Any]:
    output_root = Path(output_dir)
    if clean and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    manifest_path = output_root / "stage1_manifest_variants.json"
    stage1_dir = output_root / "stage1"
    bridge_dir = output_root / "bridge_export"
    variants = build_manifest_variants()
    manifest_path.write_text(
        json.dumps({"videos": variants}, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    def runner() -> Dict[str, Any]:
        return run_stage1_pipeline(
            num_videos=len(variants),
            proposals_per_video=proposals_per_video,
            output_dir=str(stage1_dir),
            video_manifest=str(manifest_path),
        )

    if quiet:
        with contextlib.redirect_stdout(io.StringIO()):
            stats = runner()
            bridge_manifest = export_governed_video_stage1_bridges(
                admission_log_path=stats["proposal_admission_log"],
                output_dir=bridge_dir,
                run_id="stage1_bridge_readiness_sweep",
            )
    else:
        stats = runner()
        bridge_manifest = export_governed_video_stage1_bridges(
            admission_log_path=stats["proposal_admission_log"],
            output_dir=bridge_dir,
            run_id="stage1_bridge_readiness_sweep",
        )

    admission_rows = _json_rows(Path(stats["proposal_admission_log"]))
    rlds_rows = _bridge_rows(Path(bridge_manifest["rlds_episodes_path"]))
    lerobot_rows = _bridge_rows(Path(bridge_manifest["lerobot_rows_path"]))
    rlds_by_video = _first_bridge_step_by_video(rlds_rows)
    lerobot_by_video = _first_lerobot_row_by_video(lerobot_rows)

    scenario_reports: list[Dict[str, Any]] = []
    failures: list[str] = []
    for row in admission_rows:
        video_id = str(row.get("video_id", ""))
        expected = dict(_EXPECTED.get(video_id, {}))
        benchmark_gate = dict(row.get("benchmark_gate", {}) or {})
        work_order = dict(row.get("execution_work_order", {}) or {})
        future_signals = dict(row.get("future_training_signals", {}) or {})
        report = _load_json(row["reconstruction_grounding_report_path"])
        rlds_meta = dict(rlds_by_video.get(video_id, {}).get("metadata", {}) or {})
        lerobot_meta = dict(
            lerobot_by_video.get(video_id, {}).get("metadata", {}) or {}
        )
        observed = {
            "benchmark_ready": bool(benchmark_gate.get("ready", False)),
            "benchmark_blocking_preconditions": list(
                benchmark_gate.get("blocking_preconditions", []) or []
            ),
            "recommended_mode": str(work_order.get("recommended_mode", "")),
            "calibration_class": str(report.get("calibration_class", "")),
            "grounding_class": str(report.get("grounding_class", "")),
            "reconstruction_training_eligible": bool(
                future_signals.get("reconstruction_training_eligible", False)
            ),
            "rlds_benchmark_ready": bool(
                dict(rlds_meta.get("benchmark_gate", {}) or {}).get("ready", False)
            ),
            "rlds_reconstruction_training_eligible": bool(
                dict(rlds_meta.get("future_training_signals", {}) or {}).get(
                    "reconstruction_training_eligible", False
                )
            ),
            "lerobot_benchmark_ready": bool(
                dict(lerobot_meta.get("benchmark_gate", {}) or {}).get("ready", False)
            ),
            "lerobot_reconstruction_training_eligible": bool(
                dict(lerobot_meta.get("future_training_signals", {}) or {}).get(
                    "reconstruction_training_eligible", False
                )
            ),
        }
        scenario_failures: list[str] = []
        for key in (
            "benchmark_ready",
            "calibration_class",
            "grounding_class",
            "reconstruction_training_eligible",
            "recommended_mode",
        ):
            if key in expected and observed.get(key) != expected[key]:
                scenario_failures.append(
                    f"{video_id}:{key}: expected {expected[key]!r}, got {observed.get(key)!r}"
                )
        blocker = expected.get("blocking_precondition")
        raw_observed_blockers = observed.get("benchmark_blocking_preconditions", [])
        observed_blockers = (
            list(raw_observed_blockers)
            if isinstance(raw_observed_blockers, (list, tuple, set))
            else []
        )
        if blocker and blocker not in observed_blockers:
            scenario_failures.append(f"{video_id}: missing blocker {blocker}")
        if observed["rlds_benchmark_ready"] != observed["benchmark_ready"]:
            scenario_failures.append(f"{video_id}: RLDS benchmark gate drift")
        if (
            observed["rlds_reconstruction_training_eligible"]
            != observed["reconstruction_training_eligible"]
        ):
            scenario_failures.append(f"{video_id}: RLDS training signal drift")
        if observed["lerobot_benchmark_ready"] != observed["benchmark_ready"]:
            scenario_failures.append(f"{video_id}: LeRobot benchmark gate drift")
        if (
            observed["lerobot_reconstruction_training_eligible"]
            != observed["reconstruction_training_eligible"]
        ):
            scenario_failures.append(f"{video_id}: LeRobot training signal drift")

        failures.extend(scenario_failures)
        scenario_reports.append(
            {
                "scenario_id": video_id,
                "expected": expected,
                "observed": observed,
                "passed": not scenario_failures,
                "failures": scenario_failures,
            }
        )

    scenario_ids = {str(row.get("video_id", "")) for row in admission_rows}
    missing = sorted(set(_EXPECTED) - scenario_ids)
    for scenario_id in missing:
        failures.append(f"missing scenario admission row: {scenario_id}")

    summary = {
        "version": SWEEP_VERSION,
        "status": "ok" if not failures else "failed",
        "scenario_count": len(variants),
        "admission_count": len(admission_rows),
        "rlds_episode_count": len(rlds_rows),
        "lerobot_row_count": len(lerobot_rows),
        "benchmark_ready_count": sum(
            1
            for row in admission_rows
            if dict(row.get("benchmark_gate", {}) or {}).get("ready")
        ),
        "shadow_only_count": sum(
            1
            for row in admission_rows
            if not dict(row.get("benchmark_gate", {}) or {}).get("ready")
            and not row.get("blocked", False)
        ),
        "stage1_stats": stats,
        "bridge_manifest": bridge_manifest,
        "manifest_path": str(manifest_path),
        "scenario_reports": scenario_reports,
        "failures": failures,
        "promotion_eligible": False,
        "boundary": "local structural replay/export sweep; no GPU training, provider bring-up, or promotion claim",
    }
    report_path = output_root / "stage1_bridge_readiness_report.json"
    report_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    summary["report_path"] = str(report_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run local Stage-1 bridge readiness sweep over manifest variants."
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/economic_world_model/stage1_bridge_readiness_sweep",
        help="Output directory for manifests, Stage-1 artifacts, bridge rows, and report.",
    )
    parser.add_argument("--proposals-per-video", type=int, default=1)
    parser.add_argument(
        "--no-clean", action="store_true", help="Do not delete output dir first"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Show Stage-1 pipeline logs"
    )
    args = parser.parse_args()
    summary = run_stage1_bridge_readiness_sweep(
        output_dir=args.output_dir,
        proposals_per_video=args.proposals_per_video,
        clean=not args.no_clean,
        quiet=not args.verbose,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
