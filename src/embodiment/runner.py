"""Embodiment runner for rollout bundles."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from src.embodiment.artifacts import (
    EMBODIMENT_PROFILE_PREFIX,
    AFFORDANCE_GRAPH_PREFIX,
    SKILL_SEGMENTS_PREFIX,
)
from src.embodiment.config import EmbodimentConfig
from src.embodiment.core import EmbodimentInputs, compute_embodiment
from src.motor_backend.rollout_capture import EpisodeRollout, RolloutBundle
from src.vision.motion_hierarchy.metrics import compute_motion_hierarchy_summary_from_stats

logger = logging.getLogger(__name__)


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    return dict(np.load(path, allow_pickle=False))


def _load_trajectory_payload(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        data = np.load(path, allow_pickle=True)
    except Exception:
        return None
    if "trajectory" not in data:
        return None
    payload = data["trajectory"]
    if hasattr(payload, "item") and payload.shape == ():
        payload = payload.item()
    return payload if isinstance(payload, dict) else None


def _find_semantic_fusion_path(trajectory_path: Path) -> Optional[Path]:
    candidate = trajectory_path.with_name(f"{trajectory_path.stem}_semantic_fusion_v1.npz")
    if candidate.exists():
        return candidate
    return None


def _find_scene_tracks_payload(
    episode_dir: Path,
    trajectory_payload: Optional[Dict[str, Any]],
) -> Optional[Dict[str, np.ndarray]]:
    if trajectory_payload:
        scene_tracks = trajectory_payload.get("scene_tracks_v1") or trajectory_payload.get("scene_tracks")
        if isinstance(scene_tracks, dict):
            return scene_tracks
        scene_tracks_path = trajectory_payload.get("scene_tracks_path") or trajectory_payload.get("scene_tracks_npz")
        if scene_tracks_path:
            try:
                return _load_npz(Path(scene_tracks_path))
            except Exception:
                return None
    meta_path = episode_dir / "metadata.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            meta = {}
        scene_tracks_path = meta.get("scene_tracks_path")
        if scene_tracks_path:
            try:
                return _load_npz(Path(scene_tracks_path))
            except Exception:
                return None
    return None


def _extract_action_stream(payload: Optional[Dict[str, Any]]) -> Optional[list[Any]]:
    if not payload:
        return None
    if "actions" in payload:
        return list(payload.get("actions") or [])
    trajectory = payload.get("trajectory")
    if isinstance(trajectory, list):
        actions = []
        for step in trajectory:
            if isinstance(step, dict) and "action" in step:
                actions.append(step.get("action"))
        return actions or None
    return None


def _extract_joint_state(payload: Optional[Dict[str, Any]]) -> Optional[list[Any]]:
    if not payload:
        return None
    if "states" in payload:
        return list(payload.get("states") or [])
    return None


def _extract_failure_events(payload: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not payload:
        return None
    info_history = payload.get("info_history") if isinstance(payload.get("info_history"), list) else []
    resets = 0
    clamps = 0
    taxonomy = []
    for entry in info_history:
        if not isinstance(entry, dict):
            continue
        if entry.get("reset"):
            resets += 1
        if entry.get("safety_clamp"):
            clamps += 1
        if "failure_code" in entry:
            taxonomy.append(entry.get("failure_code"))
    return {
        "resets": resets,
        "safety_clamps": clamps,
        "failure_codes": taxonomy,
    }


def _extract_mhn_summary(payload: Optional[Dict[str, Any]]) -> Optional[Any]:
    if not payload:
        return None
    mh = payload.get("motion_hierarchy")
    if not isinstance(mh, dict):
        return None
    tree_stats = mh.get("tree_stats") or {}
    delta_resid = mh.get("delta_resid_stats") or {}
    try:
        return compute_motion_hierarchy_summary_from_stats(
            mean_tree_depth=float(tree_stats.get("mean_tree_depth", 0.0)),
            mean_branch_factor=float(tree_stats.get("mean_branch_factor", 0.0)),
            residual_mean=_scalar_from_payload(delta_resid.get("mean", 0.0)) if isinstance(delta_resid, dict) else 0.0,
            residual_std=_scalar_from_payload(delta_resid.get("std", 0.0)) if isinstance(delta_resid, dict) else 0.0,
        )
    except Exception:
        return None


def _scalar_from_payload(value: Any) -> float:
    try:
        if isinstance(value, (list, tuple, np.ndarray)):
            arr = np.asarray(value, dtype=np.float32)
            return float(np.mean(arr)) if arr.size else 0.0
        return float(value)
    except Exception:
        return 0.0


def _extract_backend_tags(episode: EpisodeRollout) -> Dict[str, Any]:
    tags: Dict[str, Any] = {}
    meta = episode.metadata
    if getattr(meta, "env_params", None):
        tags.update(dict(meta.env_params))
    if meta.robot_family:
        tags["robot_family"] = meta.robot_family
    if meta.task_id:
        tags["task_id"] = meta.task_id
    return tags


def _infer_econ_attribution(metrics: Dict[str, Any], task_constraints: Optional[Dict[str, Any]]) -> Dict[str, float]:
    baseline_mpl = None
    baseline_error = None
    baseline_ep = None
    if task_constraints:
        baseline_mpl = task_constraints.get("baseline_mpl")
        baseline_error = task_constraints.get("baseline_error")
        baseline_ep = task_constraints.get("baseline_ep")

    mpl = metrics.get("mpl_units_per_hour") or metrics.get("mpl_episode") or 0.0
    error = metrics.get("error_rate") or metrics.get("error_rate_episode") or 0.0
    energy = metrics.get("energy_wh") or metrics.get("energy_Wh") or 0.0
    ep = 0.0
    if energy:
        try:
            ep = float(mpl) / float(energy)
        except Exception:
            ep = 0.0

    delta_mpl = float(mpl) - float(baseline_mpl or 0.0)
    delta_error = float(error) - float(baseline_error or 0.0)
    delta_ep = float(ep) - float(baseline_ep or 0.0)
    return {
        "delta_mpl": delta_mpl,
        "delta_error": delta_error,
        "delta_ep": delta_ep,
    }


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _write_summary_jsonl(summary_path: Path, rows: list[Dict[str, Any]]) -> None:
    if not rows:
        return
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("a") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _update_episode_metadata(
    episode_dir: Path,
    metrics: Dict[str, Any],
    artifact_paths: Dict[str, str],
    summary: Dict[str, Any],
) -> None:
    meta_path = episode_dir / "metadata.json"
    if not meta_path.exists():
        return
    try:
        payload = json.loads(meta_path.read_text())
    except Exception:
        return
    existing_metrics = payload.get("metrics", {})
    if not isinstance(existing_metrics, dict):
        existing_metrics = {}
    existing_metrics.update(metrics)
    payload["metrics"] = existing_metrics
    payload.update(artifact_paths)
    payload["embodiment_summary"] = summary
    meta_path.write_text(json.dumps(payload, indent=2))


def run_embodiment_for_rollouts(
    rollout_bundle: RolloutBundle,
    output_dir: Optional[Path] = None,
    summary_path: Optional[Path] = None,
    config: Optional[EmbodimentConfig] = None,
    task_constraints: Optional[Dict[str, Any]] = None,
) -> list[Dict[str, Any]]:
    """Run embodiment extraction for each rollout episode."""
    summaries: list[Dict[str, Any]] = []
    cfg = config or EmbodimentConfig()

    for episode in rollout_bundle.episodes:
        episode_dir = episode.trajectory_path.parent
        trajectory_payload = _load_trajectory_payload(episode.trajectory_path)
        scene_tracks_payload = _find_scene_tracks_payload(episode_dir, trajectory_payload)
        if scene_tracks_payload is None:
            logger.warning("Embodiment skipped: scene_tracks missing for %s", episode.metadata.episode_id)
            continue

        semantic_fusion_path = _find_semantic_fusion_path(episode.trajectory_path)
        semantic_fusion = _load_npz(semantic_fusion_path) if semantic_fusion_path else None

        inputs = EmbodimentInputs(
            scene_tracks=scene_tracks_payload,
            semantic_fusion=semantic_fusion,
            mhn_summary=_extract_mhn_summary(trajectory_payload),
            process_reward=trajectory_payload.get("process_reward") if trajectory_payload else None,
            action_stream=_extract_action_stream(trajectory_payload),
            joint_state=_extract_joint_state(trajectory_payload),
            task_constraints=task_constraints,
            backend_tags=_extract_backend_tags(episode),
            failure_events=_extract_failure_events(trajectory_payload),
            episode_metrics=episode.metrics,
            econ_attribution=_infer_econ_attribution(episode.metrics, task_constraints),
        )

        result = compute_embodiment(inputs, config=cfg)

        out_dir = output_dir or episode_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        episode_id = episode.metadata.episode_id or "episode"

        profile_path = out_dir / f"{episode_id}_embodiment_profile_v1.npz"
        affordance_path = out_dir / f"{episode_id}_affordance_graph_v1.npz"
        segments_path = out_dir / f"{episode_id}_skill_segments_v1.npz"
        cost_path = out_dir / f"{episode_id}_embodiment_cost_breakdown_v1.json"
        value_path = out_dir / f"{episode_id}_embodiment_value_attribution_v1.json"
        drift_path = out_dir / f"{episode_id}_embodiment_drift_report_v1.json"
        calibration_path = out_dir / f"{episode_id}_calibration_targets_v1.json"

        np.savez_compressed(profile_path, **result.profile.to_npz(summary=result.summary))
        np.savez_compressed(affordance_path, **result.affordance_graph.to_npz())
        np.savez_compressed(segments_path, **result.skill_segments.to_npz())
        _write_json(cost_path, result.cost_breakdown)
        _write_json(value_path, result.value_attribution)
        _write_json(drift_path, result.drift_report)
        _write_json(calibration_path, result.calibration_targets)

        summary = dict(result.summary.to_dict())
        summary.update(
            {
                "episode_id": episode_id,
                "embodiment_profile_path": str(profile_path),
                "affordance_graph_path": str(affordance_path),
                "skill_segments_path": str(segments_path),
                "cost_breakdown_path": str(cost_path),
                "value_attribution_path": str(value_path),
                "drift_report_path": str(drift_path),
                "calibration_targets_path": str(calibration_path),
                "embodiment_prefix": EMBODIMENT_PROFILE_PREFIX,
                "affordance_prefix": AFFORDANCE_GRAPH_PREFIX,
                "skill_segments_prefix": SKILL_SEGMENTS_PREFIX,
            }
        )

        metrics = {
            "w_embodiment": result.w_embodiment,
            "embodiment_quality_score": result.summary.embodiment_quality_score,
            "embodiment_drift_score": result.summary.drift_score,
            "embodiment_physically_impossible_contacts": result.summary.physically_impossible_contacts,
            "embodiment_trust_override_candidate": result.trust_override_candidate,
        }

        artifact_paths = {
            "embodiment_profile_path": str(profile_path),
            "affordance_graph_path": str(affordance_path),
            "skill_segments_path": str(segments_path),
            "embodiment_cost_breakdown_path": str(cost_path),
            "embodiment_value_attribution_path": str(value_path),
            "embodiment_drift_report_path": str(drift_path),
            "embodiment_calibration_targets_path": str(calibration_path),
            "semantic_fusion_path": str(semantic_fusion_path) if semantic_fusion_path else None,
        }

        _update_episode_metadata(episode_dir, metrics, artifact_paths, summary)
        summaries.append(summary)

    if summary_path is not None:
        _write_summary_jsonl(summary_path, summaries)

    return summaries
