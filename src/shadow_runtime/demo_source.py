"""Deterministic workcell-backed source adapter for the shadow control plane."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence

from src.envs.workcell_env.config import WorkcellEnvConfig
from src.envs.workcell_env.env import WorkcellEnv
from src.envs.workcell_env.rewards.reward_breakdown import compute_workcell_reward_breakdown
from src.envs.workcell_env.tasks.kitting import KittingTask
from src.objectives.runtime_builder import (
    ObjectiveRuntimeRecord,
    ObjectiveRuntimeWindow,
    SourceDomain,
)
from src.utils.config_digest import sha256_json


@dataclass(frozen=True)
class ShadowEpisodeTrace:
    """Single deterministic workcell episode trace and summaries."""

    episode_id: str
    datapack_id: str
    run_id: str
    task_id: str
    env_id: str
    world_id: str
    robot_id: str
    source_domain: str
    started_at: str
    ended_at: str
    status: str
    runtime_record: ObjectiveRuntimeRecord
    baseline_summary: Dict[str, Any]
    constraint_observations: Dict[str, Any]
    step_traces: List[Dict[str, Any]]
    episode_log: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "datapack_id": self.datapack_id,
            "run_id": self.run_id,
            "task_id": self.task_id,
            "env_id": self.env_id,
            "world_id": self.world_id,
            "robot_id": self.robot_id,
            "source_domain": self.source_domain,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "status": self.status,
            "runtime_record": self.runtime_record.to_dict(),
            "baseline_summary": dict(self.baseline_summary),
            "constraint_observations": dict(self.constraint_observations),
            "step_traces": list(self.step_traces),
            "episode_log": dict(self.episode_log),
        }


def generate_workcell_shadow_batch(
    *,
    run_id: str,
    seed: int,
    episodes: int = 2,
    timestamp_base: Optional[str] = None,
    source_domain: SourceDomain | str = SourceDomain.SYNTHETIC,
) -> list[ShadowEpisodeTrace]:
    """Generate a stable batch of workcell traces with no physical robot required."""

    base_dt = (
        datetime.fromisoformat(timestamp_base.replace("Z", "+00:00"))
        if timestamp_base
        else datetime(2026, 1, 1, tzinfo=timezone.utc)
    )
    traces: list[ShadowEpisodeTrace] = []
    for episode_index in range(episodes):
        episode_id = f"ep_{run_id}_{episode_index:03d}"
        datapack_id = f"dp_{run_id}_{episode_index:03d}"
        episode_seed = seed + episode_index
        started_at = base_dt + timedelta(minutes=episode_index * 20)
        trace = _run_single_episode(
            run_id=run_id,
            episode_id=episode_id,
            datapack_id=datapack_id,
            episode_index=episode_index,
            episode_seed=episode_seed,
            started_at=started_at,
            source_domain=str(source_domain),
        )
        traces.append(trace)
    return traces


def _run_single_episode(
    *,
    run_id: str,
    episode_id: str,
    datapack_id: str,
    episode_index: int,
    episode_seed: int,
    started_at: datetime,
    source_domain: str,
) -> ShadowEpisodeTrace:
    task = KittingTask(num_items=3, order_matters=True, dense_reward=0.2, completion_bonus=1.0)
    config = WorkcellEnvConfig(
        topology_type="ASSEMBLY_BENCH",
        num_stations=1,
        num_fixtures=1,
        num_bins=2,
        num_parts=6,
        part_types=("bolt", "plate"),
        max_steps=6,
        time_step_s=240.0,
        physics_mode="SIMPLE",
        tolerance_mm=2.0,
        occlusion_level=0.1 + 0.05 * episode_index,
        tool_changes_required=0,
    )
    env = WorkcellEnv(
        config=config,
        task=task,
        task_id="shadow_kitting",
        robot_family="shadow_sim_arm_v1",
        seed=episode_seed,
    )
    env.reset(seed=episode_seed, episode_id=episode_id, task_id="shadow_kitting", robot_family="shadow_sim_arm_v1")

    step_templates = _episode_script(episode_index)
    total_reward = 0.0
    total_energy_wh = 0.0
    step_traces: List[Dict[str, Any]] = []
    collisions = 0
    violation_steps = 0
    final_correct_count = 0
    last_info: Dict[str, Any] = {}

    for step_index, task_state in enumerate(step_templates):
        action = {
            "action_type": "PLACE",
            "target": f"slot_{step_index}",
            "task_state": dict(task_state),
        }
        obs, reward, terminated, truncated, info = env.step(action)
        correct_count = int((info.get("task", {}) or {}).get("correct_count", 0))
        final_correct_count = max(final_correct_count, correct_count)
        step_energy = _step_energy_wh(step_index, task_state)
        total_energy_wh += step_energy
        collisions += int(task_state.get("collision_count", 0))
        violation_steps += 1 if float(task_state.get("constraint_error", 0.0)) > 0.0 else 0
        reward_breakdown = compute_workcell_reward_breakdown(
            success=bool(info.get("success", False)),
            progress=correct_count / 3.0,
            time_cost=(step_index + 1) * config.time_step_s / 60.0,
            error_count=int(task_state.get("constraint_violations", 0)),
            collision_count=int(task_state.get("collision_count", 0)),
            items_picked=correct_count,
            items_placed=correct_count,
            items_total=3,
            energy_wh=step_energy,
        )
        reward_breakdown_dict = _reward_breakdown_to_dict(reward_breakdown)
        total_reward += float(reward)
        step_traces.append(
            {
                "step": step_index,
                "task_state": dict(task_state),
                "reward": float(reward),
                "reward_breakdown": reward_breakdown_dict,
                "task_info": dict(info.get("task", {}) or {}),
                "success": bool(info.get("success", False)),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "observation_keys": sorted(obs.keys()),
            }
        )
        last_info = info
        if terminated or truncated:
            break

    env.close()

    steps = len(step_traces)
    duration_s = steps * config.time_step_s
    duration_hours = max(duration_s / 3600.0, 1e-6)
    throughput_units_per_hour = final_correct_count / duration_hours
    error_rate = sum(int(step["task_state"].get("constraint_violations", 0)) for step in step_traces) / max(steps, 1)
    collision_rate = collisions / max(steps, 1)
    constraint_error_rate = violation_steps / max(steps, 1)
    energy_wh_per_unit = total_energy_wh / max(final_correct_count, 1)
    safety_score = max(
        0.0,
        min(1.0, 1.0 - 0.55 * error_rate - 0.25 * collision_rate - 0.20 * constraint_error_rate),
    )
    success = bool(last_info.get("success", False))
    uncertainty = max(0.05, min(0.95, 0.12 + 0.35 * constraint_error_rate + (0.10 if not success else 0.0)))
    trust_score = max(0.10, min(1.0, 1.0 - uncertainty - 0.10 * collision_rate))
    quality_score = max(
        0.0,
        min(1.0, 0.40 * float(success) + 0.35 * safety_score + 0.25 * (1.0 - min(1.0, energy_wh_per_unit / 5.0))),
    )
    semantic_tags = ["kitting", "shadow_demo"]
    fragile = episode_index % 2 == 1
    if fragile:
        semantic_tags.append("fragile")

    episode_metrics = {
        "episode_id": episode_id,
        "steps": steps,
        "duration_s": duration_s,
        "time_step_s": config.time_step_s,
        "items_completed": final_correct_count,
        "throughput_units_per_hour": throughput_units_per_hour,
        "error_rate": error_rate,
        "collision_rate": collision_rate,
        "constraint_error_rate": constraint_error_rate,
        "energy_wh": total_energy_wh,
        "energy_wh_per_unit": energy_wh_per_unit,
        "safety_score": safety_score,
        "quality_score": quality_score,
        "uncertainty": uncertainty,
        "trust_score": trust_score,
        "reward_total": total_reward,
        "reward_mean": total_reward / max(steps, 1),
        "success": success,
        "contact_force_n": 2.5 + 2.0 * collision_rate + (2.0 if fragile else 0.0),
    }
    reward_components = {
        "scalar_reward": total_reward,
        "mpl_component": throughput_units_per_hour / 24.0,
        "energy_penalty": energy_wh_per_unit / 8.0,
        "delta_errors": error_rate,
        "safety_bonus": safety_score,
    }
    telemetry = {
        "uncertainty": uncertainty,
        "trust_score": trust_score,
        "map_first_quality_score": max(0.55, 0.86 - 0.10 * episode_index),
        "semantic_fusion_confidence_mean": max(0.50, 0.82 - 0.08 * episode_index),
        "semantic_disagreement_vla_vs_map": min(0.40, 0.08 + 0.08 * episode_index),
        "vla_confidence": max(0.45, 0.79 - 0.10 * episode_index),
        "semantic_tags": semantic_tags,
        "fragile": fragile,
        "safety_critical": fragile,
        "max_joint_velocity": 0.9,
        "max_gripper_force": 0.8 if fragile else 1.0,
        "source_digest": sha256_json(step_templates),
    }
    windows = _build_windows(
        step_traces=step_traces,
        time_step_s=config.time_step_s,
        total_items=3,
    )
    runtime_record = ObjectiveRuntimeRecord(
        task_id="shadow_kitting",
        episode_id=episode_id,
        env_id="workcell_simple",
        world_id="workcell_assembly_bench_simple",
        robot_id="shadow_sim_arm_v1",
        source_domain=source_domain,
        seed=episode_seed,
        run_id=run_id,
        timestamp=started_at.isoformat(),
        episode_metrics=episode_metrics,
        reward_components=reward_components,
        telemetry=telemetry,
        windows=windows,
        policy_checkpoint="shadow_scripted_policy@v1",
        context={"datapack_id": datapack_id},
        provenance={
            "adapter": "workcell_shadow_source_v1",
            "episode_script_hash": sha256_json(step_templates),
        },
    )
    constraint_observations = {
        "throughput": throughput_units_per_hour,
        "error": error_rate,
        "safety": safety_score,
        "energy": energy_wh_per_unit,
        "collision_rate": collision_rate,
        "constraint_error_rate": constraint_error_rate,
        "contact_force_n": episode_metrics["contact_force_n"],
        "respect_fragility": success if fragile else True,
    }
    baseline_summary = {
        "episode_id": episode_id,
        "reward_total": total_reward,
        "success": success,
        "steps": steps,
        "throughput_units_per_hour": throughput_units_per_hour,
        "error_rate": error_rate,
        "energy_wh_per_unit": energy_wh_per_unit,
        "quality_score": quality_score,
    }
    ended_at = started_at + timedelta(seconds=duration_s)
    episode_log = env.get_episode_log(metrics=episode_metrics).to_dict()
    return ShadowEpisodeTrace(
        episode_id=episode_id,
        datapack_id=datapack_id,
        run_id=run_id,
        task_id="shadow_kitting",
        env_id="workcell_simple",
        world_id="workcell_assembly_bench_simple",
        robot_id="shadow_sim_arm_v1",
        source_domain=source_domain,
        started_at=started_at.isoformat(),
        ended_at=ended_at.isoformat(),
        status="success" if success else "failure",
        runtime_record=runtime_record,
        baseline_summary=baseline_summary,
        constraint_observations=constraint_observations,
        step_traces=step_traces,
        episode_log=episode_log,
    )


def _build_windows(
    *,
    step_traces: Sequence[Mapping[str, Any]],
    time_step_s: float,
    total_items: int,
) -> list[ObjectiveRuntimeWindow]:
    windows: list[ObjectiveRuntimeWindow] = []
    prior_correct = 0
    for window_start in range(0, len(step_traces), 2):
        window_steps = list(step_traces[window_start:window_start + 2])
        if not window_steps:
            continue
        window_end = window_steps[-1]["step"]
        last_correct = int(window_steps[-1]["task_info"].get("correct_count", 0))
        completed_delta = max(0, last_correct - prior_correct)
        prior_correct = last_correct
        duration_s = len(window_steps) * time_step_s
        throughput = completed_delta / max(duration_s / 3600.0, 1e-6)
        error_rate = sum(int(step["task_state"].get("constraint_violations", 0)) for step in window_steps) / len(window_steps)
        collision_rate = sum(int(step["task_state"].get("collision_count", 0)) for step in window_steps) / len(window_steps)
        constraint_error_rate = sum(1 for step in window_steps if float(step["task_state"].get("constraint_error", 0.0)) > 0.0) / len(window_steps)
        energy_wh = sum(_step_energy_wh(int(step["step"]), step["task_state"]) for step in window_steps)
        energy_wh_per_unit = energy_wh / max(completed_delta, 1)
        safety_score = max(
            0.0,
            min(1.0, 1.0 - 0.55 * error_rate - 0.25 * collision_rate - 0.20 * constraint_error_rate),
        )
        uncertainty = max(0.05, min(0.95, 0.10 + 0.35 * constraint_error_rate))
        trust_score = max(0.10, min(1.0, 1.0 - uncertainty - 0.10 * collision_rate))
        windows.append(
            ObjectiveRuntimeWindow(
                window_id=f"window_{window_start:03d}_{window_end:03d}",
                start_step=window_start,
                end_step=window_end,
                metrics={
                    "steps": len(window_steps),
                    "duration_s": duration_s,
                    "time_step_s": time_step_s,
                    "items_completed": completed_delta,
                    "throughput_units_per_hour": throughput,
                    "error_rate": error_rate,
                    "collision_rate": collision_rate,
                    "constraint_error_rate": constraint_error_rate,
                    "energy_wh": energy_wh,
                    "energy_wh_per_unit": energy_wh_per_unit,
                    "safety_score": safety_score,
                },
                reward_components={
                    "scalar_reward": sum(float(step["reward"]) for step in window_steps),
                    "mpl_component": throughput / 24.0,
                    "energy_penalty": energy_wh_per_unit / 8.0,
                    "delta_errors": error_rate,
                    "safety_bonus": safety_score,
                },
                telemetry={
                    "uncertainty": uncertainty,
                    "trust_score": trust_score,
                },
                metadata={"window_size": len(window_steps), "total_items": total_items},
            )
        )
    return windows


def _episode_script(episode_index: int) -> list[Dict[str, Any]]:
    if episode_index % 2 == 0:
        return [
            {
                "placements": {"item_0": 0},
                "placement_order": ["item_0"],
                "collision_count": 0,
                "constraint_error": 0.0,
                "constraint_violations": 0,
                "respect_fragility": True,
            },
            {
                "placements": {"item_0": 0, "item_1": 1},
                "placement_order": ["item_0", "item_1"],
                "collision_count": 0,
                "constraint_error": 0.0,
                "constraint_violations": 0,
                "respect_fragility": True,
            },
            {
                "placements": {"item_0": 0, "item_1": 1, "item_2": 2},
                "placement_order": ["item_0", "item_1", "item_2"],
                "collision_count": 0,
                "constraint_error": 0.0,
                "constraint_violations": 0,
                "respect_fragility": True,
            },
        ]
    return [
        {
            "placements": {"item_0": 0},
            "placement_order": ["item_0"],
            "collision_count": 1,
            "constraint_error": 0.06,
            "constraint_violations": 1,
            "respect_fragility": True,
        },
        {
            "placements": {"item_0": 0, "item_1": 2},
            "placement_order": ["item_0", "item_1"],
            "collision_count": 1,
            "constraint_error": 0.08,
            "constraint_violations": 1,
            "respect_fragility": False,
        },
        {
            "placements": {"item_0": 0, "item_1": 1},
            "placement_order": ["item_0", "item_1"],
            "collision_count": 0,
            "constraint_error": 0.03,
            "constraint_violations": 1,
            "respect_fragility": True,
        },
        {
            "placements": {"item_0": 0, "item_1": 1, "item_2": 2},
            "placement_order": ["item_0", "item_1", "item_2"],
            "collision_count": 0,
            "constraint_error": 0.0,
            "constraint_violations": 0,
            "respect_fragility": True,
        },
    ]


def _step_energy_wh(step_index: int, task_state: Mapping[str, Any]) -> float:
    return 0.45 + 0.08 * step_index + 0.15 * float(task_state.get("collision_count", 0)) + 0.10 * float(task_state.get("constraint_violations", 0))


def _reward_breakdown_to_dict(reward_breakdown: Any) -> Dict[str, Any]:
    if hasattr(reward_breakdown, "model_dump"):
        return reward_breakdown.model_dump(mode="json")
    if hasattr(reward_breakdown, "dict"):
        return reward_breakdown.dict()
    return dict(reward_breakdown)
