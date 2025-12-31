from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np


@dataclass
class EpisodeMetadata:
    episode_id: str
    task_id: str
    robot_family: str | None
    seed: int | None
    env_params: Mapping[str, Any]


@dataclass
class EpisodeRollout:
    metadata: EpisodeMetadata
    trajectory_path: Path
    rgb_video_path: Path | None = None
    depth_video_path: Path | None = None
    metrics: Mapping[str, float] = field(default_factory=dict)


@dataclass
class RolloutBundle:
    scenario_id: str
    episodes: list[EpisodeRollout] = field(default_factory=list)


def start_rollout_capture(scenario_id: str, base_dir: Path) -> None:
    """Prepare directories / state for storing rollouts for a given scenario."""
    (Path(base_dir) / scenario_id).mkdir(parents=True, exist_ok=True)


def record_episode_rollout(
    scenario_id: str,
    episode_idx: int,
    metadata: EpisodeMetadata,
    trajectory_data: Any,
    rgb_frames: Any | None,
    depth_frames: Any | None,
    metrics: Mapping[str, float],
    base_dir: Path,
) -> EpisodeRollout:
    episode_dir = Path(base_dir) / scenario_id / f"episode_{episode_idx:03d}"
    episode_dir.mkdir(parents=True, exist_ok=True)

    trajectory_path = episode_dir / "trajectory.npz"
    _write_trajectory(trajectory_path, trajectory_data)

    rgb_path = _write_video_stub(episode_dir / "rgb.mp4", rgb_frames)
    depth_path = _write_video_stub(episode_dir / "depth.mp4", depth_frames)

    rollout = EpisodeRollout(
        metadata=metadata,
        trajectory_path=trajectory_path,
        rgb_video_path=rgb_path,
        depth_video_path=depth_path,
        metrics=dict(metrics),
    )
    _write_metadata(episode_dir / "metadata.json", rollout)
    return rollout


def finalize_rollout_bundle(scenario_id: str, base_dir: Path) -> RolloutBundle:
    scenario_dir = Path(base_dir) / scenario_id
    episodes: list[EpisodeRollout] = []
    if scenario_dir.exists():
        for episode_dir in sorted(scenario_dir.glob("episode_*")):
            if not episode_dir.is_dir():
                continue
            meta_path = episode_dir / "metadata.json"
            if not meta_path.exists():
                continue
            payload = json.loads(meta_path.read_text())
            metadata = EpisodeMetadata(**payload["metadata"])
            episodes.append(
                EpisodeRollout(
                    metadata=metadata,
                    trajectory_path=Path(payload["trajectory_path"]),
                    rgb_video_path=_optional_path(payload.get("rgb_video_path")),
                    depth_video_path=_optional_path(payload.get("depth_video_path")),
                    metrics=payload.get("metrics", {}),
                )
            )
    return RolloutBundle(scenario_id=scenario_id, episodes=episodes)


def _write_trajectory(path: Path, trajectory_data: Any) -> None:
    if isinstance(trajectory_data, (str, Path)):
        src = Path(trajectory_data)
        if src.exists():
            shutil.copyfile(src, path)
            return
    try:
        np.savez_compressed(path, trajectory=trajectory_data)
    except Exception:
        np.savez_compressed(path, trajectory=np.array([]))


def _write_video_stub(path: Path, frames: Any | None) -> Path | None:
    if frames is None:
        return None
    try:
        import imageio.v3 as iio

        iio.imwrite(path, frames, fps=30)
        return path
    except Exception:
        fallback = path.with_suffix(".npz")
        try:
            np.savez_compressed(fallback, frames=np.asarray(frames))
            return fallback
        except Exception:
            return None


def _write_metadata(path: Path, rollout: EpisodeRollout) -> None:
    payload = {
        "metadata": asdict(rollout.metadata),
        "trajectory_path": str(rollout.trajectory_path),
        "rgb_video_path": str(rollout.rgb_video_path) if rollout.rgb_video_path else None,
        "depth_video_path": str(rollout.depth_video_path) if rollout.depth_video_path else None,
        "metrics": dict(rollout.metrics),
    }
    path.write_text(json.dumps(payload, indent=2))


def _optional_path(value: str | None) -> Path | None:
    if value:
        return Path(value)
    return None
