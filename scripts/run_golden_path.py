#!/usr/bin/env python3
"""Run a deterministic end-to-end ObjectiveTensor golden path demo.

This script produces a compact artifact bundle proving the contract flow:

ObjectiveTensor -> ObjectiveCompiler -> scalar reward
ObjectiveTensor -> ObjectiveEconFunctor -> runtime EconTensor
Episode traces -> TrajectoryAudit -> Regal governance reports

Usage:
    python scripts/run_golden_path.py --env workcell --episodes 10 --seed 0 --emit artifacts/golden_path
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.contracts.schemas import (
    PlanGainScheduleV1,
    PlanOpType,
    PlanPolicyConfigV1,
    RegalContextV1,
    RegalGatesV1,
    RegalPhaseV1,
    SemanticUpdatePlanV1,
    TaskGraphOp,
)
from src.economics.econ_basis_registry import get_default_basis
from src.economics.econ_tensor import EconTensor, econ_to_tensor
from src.economics.functor import ObjectiveEconFunctor
from src.objectives.compiler import ObjectiveCompiler
from src.objectives.profile import ObjectiveProfile
from src.objectives.tensor import ObjectiveTensor, objective_tensor_from_axes
from src.regal.regal_evaluator import evaluate_regals
from src.utils.config_digest import sha256_json
from src.valuation.trajectory_audit import create_trajectory_audit


def _build_objective_profile() -> ObjectiveProfile:
    return ObjectiveProfile(
        profile_id="golden_path_v1",
        scalarizer="constrained",
        weights={
            "throughput": 1.2,
            "error": 1.1,
            "safety": 1.0,
            "energy": 0.6,
        },
        constraints={
            "throughput": {"min": 0.40},
            "error": {"max": 0.35},
            "safety": {"min": 0.45},
            "energy": {"max": 0.75},
        },
        penalty_weight=5.0,
        metadata={"purpose": "golden_path"},
    )


def _build_regal_config(seed: int) -> RegalGatesV1:
    return RegalGatesV1(
        enabled_regal_ids=[
            "spec_guardian",
            "world_coherence",
            "reward_integrity",
            "econ_data",
        ],
        penalty_mode="warn",
        determinism_seed=seed,
    )


def _build_policy_config(env: str, regal_config: RegalGatesV1) -> PlanPolicyConfigV1:
    return PlanPolicyConfigV1(
        gain_schedule=PlanGainScheduleV1(
            conservative_multiplier=1.05,
            full_multiplier=1.2,
            max_abs_weight_change=0.2,
            min_weight_clamp=0.1,
            max_weight_clamp=2.0,
        ),
        default_weights={env: 1.0},
        regal_gates=regal_config,
    )


def _build_plan(env: str) -> SemanticUpdatePlanV1:
    return SemanticUpdatePlanV1(
        plan_id=f"golden_{env}_plan_v1",
        source_commit="golden_path",
        task_graph_changes=[
            TaskGraphOp(op=PlanOpType.SET_WEIGHT, task_family=env, weight=1.0),
        ],
        notes="Deterministic golden path plan.",
    )


def _clip(value: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, value)))


def _generate_objective_axes(
    rng: np.random.Generator,
    episode_index: int,
    total_episodes: int,
) -> Dict[str, float]:
    progress = float(episode_index) / max(total_episodes - 1, 1)
    throughput = _clip(0.35 + 0.55 * progress + float(rng.normal(0.0, 0.05)), 0.05, 1.50)
    error = _clip(0.42 - 0.22 * progress + float(rng.normal(0.0, 0.04)), 0.0, 1.0)
    safety = _clip(0.50 + 0.30 * progress - 0.35 * error + float(rng.normal(0.0, 0.03)), 0.0, 1.2)
    energy = _clip(0.62 - 0.18 * progress + 0.15 * error + float(rng.normal(0.0, 0.04)), 0.05, 1.2)
    uncertainty = _clip(0.30 - 0.18 * progress + float(rng.normal(0.0, 0.02)), 0.0, 1.0)
    return {
        "throughput": throughput,
        "error": error,
        "safety": safety,
        "energy": energy,
        "uncertainty": uncertainty,
    }


def _build_objective_tensor(
    env: str,
    run_id: str,
    episode_index: int,
    axes: Mapping[str, float],
) -> ObjectiveTensor:
    base_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
    deterministic_time = (base_time + timedelta(seconds=episode_index)).isoformat()
    return objective_tensor_from_axes(
        {
            "throughput": float(axes["throughput"]),
            "error": float(axes["error"]),
            "safety": float(axes["safety"]),
            "energy": float(axes["energy"]),
        },
        context={
            "env": env,
            "run_id": run_id,
            "episode_index": episode_index,
            "timestamp": deterministic_time,
        },
        provenance={
            "source": "golden_path",
            "generated_step": episode_index,
        },
    )


def _runtime_econ_to_dict(tensor: EconTensor) -> Dict[str, float]:
    return {axis: float(tensor.values[i]) for i, axis in enumerate(tensor.schema.axes)}


def _canonical_econ_dict(
    objective_axes: Mapping[str, float],
    scalar_reward: float,
    runtime_econ: Mapping[str, float],
) -> Dict[str, float]:
    success_rate = _clip(
        objective_axes["safety"]
        - 0.2 * objective_axes["error"]
        + 0.15 * objective_axes["throughput"],
        0.0,
        1.0,
    )
    return {
        "mpl_units_per_hour": float(runtime_econ["value_earned"] * 20.0),
        "wage_parity": _clip(1.0 + 0.1 * runtime_econ["marginal_frontier_gain"], 0.0, 2.0),
        "energy_cost": float(objective_axes["energy"] * 8.0),
        "damage_cost": float(objective_axes["error"] * 4.0),
        "novelty_delta": float(
            runtime_econ["marginal_frontier_gain"] - runtime_econ["uncertainty_discount"]
        ),
        "reward_scalar_sum": float(scalar_reward),
        "mobility_penalty": float(runtime_econ["constraint_penalty"]),
        "throughput": float(objective_axes["throughput"] * 50.0),
        "error_rate": float(objective_axes["error"]),
        "success_rate": float(success_rate),
    }


def _build_trajectory_audit(
    rng: np.random.Generator,
    *,
    episode_id: str,
    scalar_reward: float,
    objective_axes: Mapping[str, float],
    constraint_flags: Sequence[Mapping[str, Any]],
    anomaly: bool,
) -> Any:
    num_steps = 32
    actions = rng.normal(0.0, 0.15, size=(num_steps, 7)).tolist()
    rewards = (scalar_reward / num_steps + rng.normal(0.0, 0.01, size=num_steps)).tolist()
    reward_components = {
        "task_reward": [float(max(0.0, objective_axes["throughput"]) / num_steps)] * num_steps,
        "error_penalty": [float(-objective_axes["error"] / num_steps)] * num_steps,
        "energy_penalty": [float(-objective_axes["energy"] / num_steps)] * num_steps,
    }

    events = ["step"] * num_steps
    # Golden-path baseline should only emit violation events when an anomaly is explicitly injected.
    if anomaly and constraint_flags:
        events.append("constraint_violation")
    if anomaly:
        events.extend(["physics_violation", "velocity_violation"])

    penetrations: List[float] = []
    velocities: List[List[float]] = []
    for idx in range(num_steps):
        if anomaly and idx % 4 == 0:
            velocities.append([7.0, 7.0, 7.0])
            penetrations.append(0.02)
        else:
            velocities.append(rng.uniform(0.2, 2.0, size=3).tolist())
            penetrations.append(float(rng.uniform(0.0, 0.005)))

    audit = create_trajectory_audit(
        episode_id=episode_id,
        num_steps=num_steps,
        actions=actions,
        rewards=[float(r) for r in rewards],
        reward_components=reward_components,
        events=events,
        penetrations=penetrations,
        velocities=velocities,
    )
    if anomaly:
        payload = audit.model_dump(mode="json")
        payload["contact_anomaly_count"] = 4
        return type(audit)(**payload)
    return audit


def _summarize_regal(reports: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    summary: List[Dict[str, Any]] = []
    for report in reports:
        summary.append(
            {
                "regal_id": report["regal_id"],
                "phase": report["phase"],
                "passed": bool(report["passed"]),
                "confidence": float(report.get("confidence", 0.0)),
                "rationale": report.get("rationale", ""),
                "spec_violations": list(report.get("spec_violations", []) or []),
                "coherence_tags": list(report.get("coherence_tags", []) or []),
                "integrity_flags": list(report.get("integrity_flags", []) or []),
                "hack_probability": float(report.get("hack_probability", 0.0)),
                "findings": dict(report.get("findings") or {}),
            }
        )
    return summary


def _extract_threshold_and_actual(value: Any) -> tuple[Any, Any]:
    if not isinstance(value, Mapping):
        return None, value
    threshold = None
    actual = None
    for key in ("threshold", "limit", "max", "min", "expected"):
        if key in value:
            threshold = value[key]
            break
    for key in ("actual", "value", "observed", "count"):
        if key in value:
            actual = value[key]
            break
    return threshold, actual if actual is not None else value


def _extract_step_index(value: Any) -> int | None:
    if isinstance(value, Mapping):
        for key in ("step", "step_index", "timestep"):
            candidate = value.get(key)
            if isinstance(candidate, int):
                return candidate
    if isinstance(value, str):
        match = re.search(r"(?:step|step_index|timestep)\s*[:=]\s*(-?\d+)", value)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                return None
    return None


def _build_governance_explain(
    governance_rows: Sequence[Mapping[str, Any]],
    *,
    total_episodes: int,
) -> Dict[str, Any]:
    failed_reports: List[Mapping[str, Any]] = []
    failed_episode_ids: set[str] = set()
    failed_rule_rows: MutableMapping[str, Dict[str, Any]] = {}
    per_regal_failures: MutableMapping[str, int] = {}

    for episode in governance_rows:
        episode_id = str(episode.get("episode_id", "unknown"))
        episode_index = int(episode.get("episode_index", -1))
        for report in episode.get("reports", []):
            if bool(report.get("passed", True)):
                continue

            failed_reports.append(report)
            failed_episode_ids.add(episode_id)
            regal_id = str(report.get("regal_id", "unknown"))
            per_regal_failures[regal_id] = int(per_regal_failures.get(regal_id, 0) + 1)
            phase = str(report.get("phase", "unknown"))

            def _record_rule(
                rule_id: str,
                source: str,
                threshold: Any,
                offending_value: Any,
                step_index: int | None,
            ) -> None:
                row = failed_rule_rows.get(rule_id)
                if row is None:
                    failed_rule_rows[rule_id] = {
                        "rule_id": rule_id,
                        "source": source,
                        "regal_id": regal_id,
                        "phase": phase,
                        "count": 1,
                        "first_episode_index": episode_index,
                        "first_episode_id": episode_id,
                        "first_step_index": step_index,
                        "threshold": threshold,
                        "offending_value": offending_value,
                    }
                    return
                row["count"] = int(row["count"]) + 1

            for violation in report.get("spec_violations", []) or []:
                violation_text = str(violation)
                _record_rule(
                    rule_id=f"{regal_id}.spec_violation.{violation_text}",
                    source="spec_violations",
                    threshold=None,
                    offending_value=violation_text,
                    step_index=_extract_step_index(violation_text),
                )

            for flag in report.get("integrity_flags", []) or []:
                flag_text = str(flag)
                _record_rule(
                    rule_id=f"{regal_id}.integrity_flag.{flag_text}",
                    source="integrity_flags",
                    threshold=None,
                    offending_value=flag_text,
                    step_index=_extract_step_index(flag_text),
                )

            findings = report.get("findings") or {}
            if isinstance(findings, Mapping):
                for key, value in findings.items():
                    if (
                        key == "violations"
                        and isinstance(value, Sequence)
                        and not isinstance(value, (str, bytes))
                    ):
                        for violation in value:
                            violation_text = str(violation)
                            _record_rule(
                                rule_id=f"{regal_id}.finding.violations.{violation_text}",
                                source="findings.violations",
                                threshold=None,
                                offending_value=violation_text,
                                step_index=_extract_step_index(violation_text),
                            )
                        continue
                    threshold, actual = _extract_threshold_and_actual(value)
                    _record_rule(
                        rule_id=f"{regal_id}.finding.{str(key)}",
                        source=f"findings.{str(key)}",
                        threshold=threshold,
                        offending_value=actual,
                        step_index=_extract_step_index(actual),
                    )

            if (
                not report.get("spec_violations")
                and not report.get("integrity_flags")
                and not report.get("findings")
            ):
                _record_rule(
                    rule_id=f"{regal_id}.report_failed",
                    source="report",
                    threshold=None,
                    offending_value=report.get("rationale", ""),
                    step_index=_extract_step_index(report.get("rationale", "")),
                )

    top_failing_rules = sorted(
        failed_rule_rows.values(),
        key=lambda row: (-int(row["count"]), str(row["rule_id"])),
    )
    top_failing_regals = [
        {"regal_id": regal_id, "count": count}
        for regal_id, count in sorted(
            per_regal_failures.items(),
            key=lambda item: (-int(item[1]), str(item[0])),
        )
    ]
    failed_episode_count = len(failed_episode_ids)
    fail_rate = float(failed_episode_count / max(total_episodes, 1))
    return {
        "total_episodes": total_episodes,
        "failed_episode_count": failed_episode_count,
        "failed_episode_rate": fail_rate,
        "failed_report_count": len(failed_reports),
        "top_failing_regals": top_failing_regals,
        "top_failing_rules": top_failing_rules,
    }


def _print_governance_explain(explain: Mapping[str, Any], explain_path: Path) -> None:
    failed_episode_count = int(explain.get("failed_episode_count", 0))
    total_episodes = int(explain.get("total_episodes", 0))
    failed_report_count = int(explain.get("failed_report_count", 0))
    print(f"Governance explain: {explain_path}")
    print(
        f"Governance failures: episodes={failed_episode_count}/{total_episodes}, reports={failed_report_count}"
    )
    top_rules = explain.get("top_failing_rules") or []
    if not isinstance(top_rules, Sequence) or not top_rules:
        return
    print("Top failing rules:")
    for row in list(top_rules)[:5]:
        if not isinstance(row, Mapping):
            continue
        threshold = row.get("threshold")
        threshold_fragment = f", threshold={threshold}" if threshold is not None else ""
        step_index = row.get("first_step_index")
        step_fragment = f", first_step={step_index}" if step_index is not None else ""
        print(
            "  - "
            f"{row.get('rule_id')} (count={row.get('count')}, "
            f"first_episode={row.get('first_episode_index')}{step_fragment}, "
            f"source={row.get('source')}{threshold_fragment})"
        )


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")


def _plot_objectives(episodes: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    xs = [int(item["episode_index"]) for item in episodes]
    throughput = [float(item["objective_axes"]["throughput"]) for item in episodes]
    error = [float(item["objective_axes"]["error"]) for item in episodes]
    safety = [float(item["objective_axes"]["safety"]) for item in episodes]
    energy = [float(item["objective_axes"]["energy"]) for item in episodes]
    scalar_reward = [float(item["scalar_reward"]) for item in episodes]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(xs, throughput, label="throughput", linewidth=2.0)
    axes[0].plot(xs, error, label="error", linewidth=2.0)
    axes[0].plot(xs, safety, label="safety", linewidth=2.0)
    axes[0].plot(xs, energy, label="energy", linewidth=2.0)
    axes[0].set_title("ObjectiveTensor Axes")
    axes[0].set_xlabel("Episode")
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="best")

    axes[1].plot(xs, scalar_reward, label="scalar_reward", color="#1f77b4", linewidth=2.2)
    axes[1].set_title("Compiler Scalar Reward")
    axes[1].set_xlabel("Episode")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="best")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_econ_and_governance(episodes: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    xs = [int(item["episode_index"]) for item in episodes]
    value_earned = [float(item["runtime_econ"]["value_earned"]) for item in episodes]
    penalty = [float(item["runtime_econ"]["constraint_penalty"]) for item in episodes]
    discount = [float(item["runtime_econ"]["uncertainty_discount"]) for item in episodes]
    regal_failures = [
        int(sum(1 for report in item["governance"]["reports"] if not report["passed"]))
        for item in episodes
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(xs, value_earned, label="value_earned", linewidth=2.0)
    axes[0].plot(xs, penalty, label="constraint_penalty", linewidth=2.0)
    axes[0].plot(xs, discount, label="uncertainty_discount", linewidth=2.0)
    axes[0].set_title("ObjectiveEconFunctor Outputs")
    axes[0].set_xlabel("Episode")
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="best")

    axes[1].bar(xs, regal_failures, color="#d62728", width=0.8)
    axes[1].set_title("Governance Failures Per Episode")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Failed Regals")
    axes[1].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def run_golden_path(
    *,
    env: str,
    episodes: int,
    seed: int,
    output_dir: Path,
    regal_anomaly_episode: int,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    run_id = sha256_json({"env": env, "episodes": episodes, "seed": seed})[:8]
    timestamp = datetime.now(timezone.utc).isoformat()

    objective_profile = _build_objective_profile()
    compiler = ObjectiveCompiler(objective_profile)
    functor = ObjectiveEconFunctor(base_price_per_unit=2.0)

    regal_config = _build_regal_config(seed=seed)
    policy_config = _build_policy_config(env=env, regal_config=regal_config)
    plan = _build_plan(env=env)
    basis = get_default_basis()

    episode_rows: List[Dict[str, Any]] = []
    objective_logs: List[Dict[str, Any]] = []
    scalar_rows: List[Dict[str, Any]] = []
    econ_rows: List[Dict[str, Any]] = []
    econ_delta_rows: List[Dict[str, Any]] = []
    governance_rows: List[Dict[str, Any]] = []

    prev_runtime_econ: Dict[str, float] | None = None

    for episode_index in range(episodes):
        episode_id = f"{env}_ep_{episode_index:03d}"
        objective_axes = _generate_objective_axes(rng, episode_index, episodes)
        uncertainty = float(objective_axes["uncertainty"])
        tensor = _build_objective_tensor(
            env=env,
            run_id=run_id,
            episode_index=episode_index,
            axes=objective_axes,
        )

        scalar_reward = float(compiler.scalarize(tensor))
        constraint_flags = compiler.constraint_flags(tensor)

        runtime_econ_tensor = functor.map(
            tensor,
            constraint_flags=constraint_flags,
            uncertainty=uncertainty,
            context={"episode_id": episode_id, "run_id": run_id},
        )
        runtime_econ = _runtime_econ_to_dict(runtime_econ_tensor)

        if prev_runtime_econ is None:
            runtime_econ_delta = {axis: 0.0 for axis in runtime_econ}
        else:
            runtime_econ_delta = {
                axis: float(runtime_econ[axis] - prev_runtime_econ.get(axis, 0.0))
                for axis in runtime_econ
            }
        prev_runtime_econ = dict(runtime_econ)

        anomaly = regal_anomaly_episode >= 0 and episode_index == regal_anomaly_episode
        trajectory_audit = _build_trajectory_audit(
            rng,
            episode_id=episode_id,
            scalar_reward=scalar_reward,
            objective_axes=objective_axes,
            constraint_flags=constraint_flags,
            anomaly=anomaly,
        )
        canonical_econ = _canonical_econ_dict(objective_axes, scalar_reward, runtime_econ)
        econ_tensor_v1 = econ_to_tensor(canonical_econ, basis=basis.spec, source="episode_metrics")

        regal_context = RegalContextV1(
            run_id=run_id,
            step=episode_index,
            plan_sha=plan.sha256(),
            trajectory_audit_sha=trajectory_audit.sha256(),
            econ_basis_sha=econ_tensor_v1.basis_sha,
            econ_tensor_sha=econ_tensor_v1.sha256(),
            notes={
                "episode_id": episode_id,
                "constraint_flag_count": len(constraint_flags),
            },
        )
        governance = evaluate_regals(
            config=regal_config,
            phase=RegalPhaseV1.POST_AUDIT,
            plan=plan,
            policy_config=policy_config,
            context=regal_context,
            trajectory_audit=trajectory_audit,
            econ_tensor=econ_tensor_v1,
        )
        governance_dump = governance.model_dump(mode="json")
        governance_dump["reports"] = _summarize_regal(governance_dump.get("reports", []))

        objective_log = {
            "episode_id": episode_id,
            "episode_index": episode_index,
            "objective_tensor": tensor.to_dict(),
        }
        scalar_row = {
            "episode_id": episode_id,
            "episode_index": episode_index,
            "scalar_reward": scalar_reward,
            "constraint_flags": constraint_flags,
        }
        econ_row = {
            "episode_id": episode_id,
            "episode_index": episode_index,
            "runtime_econ": runtime_econ,
            "canonical_econ_tensor": econ_tensor_v1.model_dump(mode="json"),
        }
        econ_delta_row = {
            "episode_id": episode_id,
            "episode_index": episode_index,
            "runtime_econ_delta": runtime_econ_delta,
        }
        governance_row = {
            "episode_id": episode_id,
            "episode_index": episode_index,
            "all_passed": bool(governance.all_passed),
            "reports": governance_dump["reports"],
        }

        row = {
            "episode_id": episode_id,
            "episode_index": episode_index,
            "objective_axes": {
                key: float(value) for key, value in objective_axes.items() if key != "uncertainty"
            },
            "uncertainty": uncertainty,
            "scalar_reward": scalar_reward,
            "constraint_flags": constraint_flags,
            "runtime_econ": runtime_econ,
            "runtime_econ_delta": runtime_econ_delta,
            "governance": governance_row,
        }

        episode_rows.append(row)
        objective_logs.append(objective_log)
        scalar_rows.append(scalar_row)
        econ_rows.append(econ_row)
        econ_delta_rows.append(econ_delta_row)
        governance_rows.append(governance_row)

    objective_path = output_dir / "objective_tensors.jsonl"
    scalar_path = output_dir / "scalar_rewards.json"
    econ_path = output_dir / "econ_tensors.json"
    econ_delta_path = output_dir / "econ_deltas.json"
    governance_path = output_dir / "governance_report.json"
    episodes_path = output_dir / "episodes.json"

    _write_jsonl(objective_path, objective_logs)
    _write_json(scalar_path, scalar_rows)
    _write_json(econ_path, econ_rows)
    _write_json(econ_delta_path, econ_delta_rows)
    _write_json(governance_path, governance_rows)
    _write_json(episodes_path, episode_rows)

    objective_plot_path = output_dir / "plots" / "objective_scalar.png"
    econ_plot_path = output_dir / "plots" / "econ_governance.png"
    _plot_objectives(episode_rows, objective_plot_path)
    _plot_econ_and_governance(episode_rows, econ_plot_path)

    all_governance_passed = all(bool(row["all_passed"]) for row in governance_rows)
    per_regal_failures: MutableMapping[str, int] = {}
    for episode in governance_rows:
        for report in episode["reports"]:
            if not report["passed"]:
                regal_id = str(report["regal_id"])
                per_regal_failures[regal_id] = int(per_regal_failures.get(regal_id, 0) + 1)

    summary = {
        "run_id": run_id,
        "timestamp_utc": timestamp,
        "env": env,
        "episodes": episodes,
        "seed": seed,
        "all_governance_passed": all_governance_passed,
        "mean_scalar_reward": float(np.mean([row["scalar_reward"] for row in scalar_rows])),
        "mean_constraint_flags": float(
            np.mean([len(row["constraint_flags"]) for row in scalar_rows])
        ),
        "per_regal_failures": dict(per_regal_failures),
        "objective_profile": objective_profile.to_dict(),
        "regal_config": regal_config.model_dump(mode="json"),
        "plan": plan.model_dump(mode="json"),
    }
    summary["summary_sha"] = sha256_json(summary)
    governance_explain = _build_governance_explain(
        governance_rows,
        total_episodes=episodes,
    )
    governance_explain_path = output_dir / "governance_explain.json"
    _write_json(governance_explain_path, governance_explain)

    bundle = {
        "summary": summary,
        "governance_explain": governance_explain,
        "artifacts": {
            "objective_tensor_logs": str(objective_path),
            "scalar_rewards": str(scalar_path),
            "econ_tensors": str(econ_path),
            "econ_deltas": str(econ_delta_path),
            "governance_report": str(governance_path),
            "governance_explain": str(governance_explain_path),
            "episode_rollup": str(episodes_path),
            "objective_scalar_plot": str(objective_plot_path),
            "econ_governance_plot": str(econ_plot_path),
        },
    }
    _write_json(output_dir / "summary.json", summary)
    _write_json(output_dir / "artifact_bundle.json", bundle)
    return bundle


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run golden path objective->econ->governance demo."
    )
    parser.add_argument(
        "--env",
        type=str,
        default="workcell",
        choices=["workcell"],
        help="Environment label used in artifacts.",
    )
    parser.add_argument(
        "--episodes", type=int, default=10, help="Number of episodes to synthesize."
    )
    parser.add_argument("--seed", type=int, default=0, help="Deterministic seed.")
    parser.add_argument(
        "--emit",
        type=str,
        default="artifacts/golden_path",
        help="Artifact output directory.",
    )
    parser.add_argument(
        "--regal-anomaly-episode",
        type=int,
        default=-1,
        help="Optional episode index for injected trajectory anomaly (default: disabled).",
    )
    parser.add_argument(
        "--fail-on-governance-failure",
        action="store_true",
        help="Exit with code 1 if any episode fails governance checks.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be > 0")
    output_dir = Path(args.emit)
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = run_golden_path(
        env=args.env,
        episodes=int(args.episodes),
        seed=int(args.seed),
        output_dir=output_dir,
        regal_anomaly_episode=int(args.regal_anomaly_episode),
    )

    summary = bundle["summary"]
    print("=== Golden Path Complete ===")
    print(f"Run ID: {summary['run_id']}")
    print(f"Env: {summary['env']}")
    print(f"Episodes: {summary['episodes']}")
    print(f"All governance passed: {summary['all_governance_passed']}")
    print(f"Mean scalar reward: {summary['mean_scalar_reward']:.4f}")
    print(f"Artifact bundle: {output_dir / 'artifact_bundle.json'}")
    _print_governance_explain(
        bundle.get("governance_explain", {}), output_dir / "governance_explain.json"
    )
    if args.fail_on_governance_failure and not bool(summary["all_governance_passed"]):
        print(
            "Governance failure gate enabled: exiting non-zero due to failed governance checks.",
            file=sys.stderr,
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
