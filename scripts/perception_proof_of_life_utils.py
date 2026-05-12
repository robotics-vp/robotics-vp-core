#!/usr/bin/env python3
"""Shared local proof-of-life helpers for Phase 2 Perception smoke scripts."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any
from typing import Mapping

import torch

from src.dataset_bridges.lerobot_bridge import replay_episode_from_lerobot
from src.replay.schema import ReplayEpisodeRecord, ReplayStepRecord


def make_mock_lerobot_step(
    *,
    episode_id: str,
    step_idx: int,
    seed: int,
    final_step_idx: int,
    camera_format: str = "droid",
) -> ReplayStepRecord:
    """Build a deterministic mock LeRobot-format replay step."""
    generator = torch.Generator().manual_seed(seed + step_idx)
    obs: dict[str, object] = {}
    if camera_format == "droid":
        camera_keys = [
            "exterior_image_1_left",
            "exterior_image_2_left",
            "wrist_image_left",
        ]
        image_shape = (180, 320, 3)
    elif camera_format == "bridge":
        camera_keys = [f"image_{idx}" for idx in range(4)]
        image_shape = (256, 256, 3)
    else:
        raise ValueError(f"unsupported mock LeRobot camera_format={camera_format!r}")

    for camera_key in camera_keys:
        obs[f"images.{camera_key}"] = torch.randint(
            0,
            255,
            image_shape,
            dtype=torch.uint8,
            generator=generator,
        )
    obs["state"] = torch.rand(7, generator=generator)
    return ReplayStepRecord(
        run_id="phase2_local_mock_lerobot",
        episode_id=episode_id,
        step_idx=step_idx,
        obs=obs,
        obs_vector=[],
        action={"joint_positions": torch.rand(7, generator=generator).tolist()},
        action_vector=torch.rand(7, generator=generator).tolist(),
        reward=1.0 if step_idx == final_step_idx else 0.0,
        reward_decomposition={},
        done=step_idx == final_step_idx,
        task_id="mock_pick_and_place",
        env_id="mock_tabletop",
        condition_vector={},
        condition_vector_values=[],
        skill_mode="autonomous",
        objective_tensor_summary={},
        objective_tensor_ref=None,
        econ_tensor_summary={},
        econ_tensor_ref=None,
        constraint_flags=[],
        pricing_tick_ref=None,
        ledger_event_ref=None,
        source_domain=f"mock_lerobot_{camera_format}",
        seed=seed,
        timestamp=f"2026-05-11T00:00:{step_idx:02d}+00:00",
        metadata={"mock_camera_format": camera_format},
        provenance={"mock_data": True, "not_external_dataset": True},
    )


def make_mock_lerobot_episode(
    *,
    episode_idx: int,
    num_steps: int,
    seed: int,
    camera_format: str = "droid",
) -> tuple[ReplayEpisodeRecord, list[ReplayStepRecord]]:
    """Build a deterministic mock LeRobot-format episode plus steps."""
    episode_id = f"mock_{camera_format}_{episode_idx:03d}"
    episode_seed = seed + episode_idx * 1000
    final_step_idx = max(0, num_steps - 1)
    steps = [
        make_mock_lerobot_step(
            episode_id=episode_id,
            step_idx=step_idx,
            seed=episode_seed,
            final_step_idx=final_step_idx,
            camera_format=camera_format,
        )
        for step_idx in range(num_steps)
    ]
    episode = ReplayEpisodeRecord(
        run_id="phase2_local_mock_lerobot",
        episode_id=episode_id,
        task_id="mock_pick_and_place",
        env_id="mock_tabletop",
        source_domain=f"mock_lerobot_{camera_format}",
        seed=episode_seed,
        status="completed",
        started_at="2026-05-11T00:00:00+00:00",
        ended_at=f"2026-05-11T00:00:{final_step_idx:02d}+00:00",
        total_steps=num_steps,
        total_reward=sum(step.reward for step in steps),
        skill_mode="autonomous",
        condition_vector={},
        condition_vector_values=[],
        objective_tensor_summary={},
        objective_tensor_ref=None,
        econ_tensor_summary={},
        econ_tensor_ref=None,
        pricing_summary={},
        pricing_tick_refs=[],
        constraint_flags=[],
        regal_summary={},
        datapack_summary={},
        ledger_event_ids=[],
        metadata={"mock_camera_format": camera_format},
        provenance={"mock_data": True, "not_external_dataset": True},
    )
    return episode, steps


def _load_lerobot_rows(path: str | Path) -> list[dict[str, Any]]:
    rows_path = Path(path).resolve()
    if not rows_path.exists():
        raise FileNotFoundError(f"LeRobot rows path not found: {rows_path}")

    if rows_path.suffix == ".jsonl":
        return [
            json.loads(line)
            for line in rows_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    payload = json.loads(rows_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [dict(row) for row in payload]
    if isinstance(payload, Mapping):
        rows = payload.get("rows")
        if isinstance(rows, list):
            return [dict(row) for row in rows]
    raise ValueError(
        "LeRobot rows payload must be JSONL rows, a JSON list, or a JSON object "
        "with a top-level 'rows' list"
    )


def load_lerobot_episodes_from_path(
    path: str | Path,
    *,
    max_episodes: int | None = None,
    max_steps_per_episode: int | None = None,
) -> list[tuple[ReplayEpisodeRecord, list[ReplayStepRecord]]]:
    """Load grouped LeRobot-like rows from a local JSON/JSONL bundle."""
    grouped_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _load_lerobot_rows(path):
        grouped_rows[str(row.get("episode_id", ""))].append(row)

    episodes: list[tuple[ReplayEpisodeRecord, list[ReplayStepRecord]]] = []
    for episode_id in sorted(grouped_rows):
        episode_rows = sorted(
            grouped_rows[episode_id],
            key=lambda row: int(row.get("frame_index", 0)),
        )
        if max_steps_per_episode is not None:
            episode_rows = episode_rows[:max_steps_per_episode]
        if not episode_rows:
            continue
        episodes.append(replay_episode_from_lerobot(episode_rows))
        if max_episodes is not None and len(episodes) >= max_episodes:
            break
    if not episodes:
        raise ValueError(f"No episodes found in LeRobot rows bundle: {path}")
    return episodes


__all__ = [
    "load_lerobot_episodes_from_path",
    "make_mock_lerobot_episode",
    "make_mock_lerobot_step",
]
