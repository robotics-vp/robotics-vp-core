#!/usr/bin/env python3
"""Run a minimal real training-run + receipt-ingest readiness probe.

This script executes a tiny training flow through RegalTrainingRunner and the
training receipt-ingest path, then reports target future-training predicates.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.replay.dataset import ReplayDatasetBuilder
from src.replay.receipt_ingest import build_training_run_receipt_label_bundle
from src.shadow_runtime.control_plane import run_shadow_control_plane
from src.training.checkpoint_registry import build_checkpoint_record
from src.training.regal_training_runner import TrainingRunConfig, run_training_with_regality
from src.valuation.trajectory_audit import create_trajectory_audit


def _run_probe(output_root: Path, *, seed: int) -> Dict[str, Any]:
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    shadow_dir = output_root / "shadow_run"
    training_dir = output_root / "training_run"

    def _train(runner) -> None:
        run_shadow_control_plane(
            output_dir=shadow_dir,
            seed=seed,
            episodes=1,
            objective_profile_id="balanced_contract",
            include_regal=True,
            timestamp_base="2026-03-26T00:00:00+00:00",
        )
        trace_payload = json.loads((shadow_dir / "shadow_episode_traces.json").read_text(encoding="utf-8"))
        episode_log = trace_payload["episodes"][0]["episode_log"]
        episode_id = str(episode_log["metadata"]["episode_id"])

        episode_logs_dir = training_dir / "online_episode_logs"
        episode_logs_dir.mkdir(parents=True, exist_ok=True)
        episode_log_path = episode_logs_dir / f"{episode_id}.json"
        episode_log_path.write_text(json.dumps(episode_log), encoding="utf-8")

        dataset_dir = training_dir / "online_replay_dataset"
        dataset = (
            ReplayDatasetBuilder()
            .add_workcell_episode_log(
                episode_log_path,
                run_id="training_probe_001",
                source_domain="training_run",
            )
            .write(dataset_dir)
        )

        receipts_path = training_dir / "online_episode_receipts.jsonl"
        receipts_path.write_text(
            json.dumps(
                {
                    "run_id": "training_probe_001",
                    "episode_id": episode_id,
                    "source_domain": "training_run",
                    "predicted_value": 2.0,
                    "realized_value": 1.8,
                    "quoted_rate": 2.0,
                    "accepted_rate": 1.8,
                    "pricing_accepted": True,
                    "task_success": True,
                    "objective_satisfied": True,
                    "realized_reward": 1.2,
                    "failure_events": [],
                    "risk_events": [],
                    "incident_events": [],
                    "expected_adaptation_benefit": 0.1,
                    "realized_adaptation_benefit": 0.08,
                    "adaptation_compute_cost": 0.02,
                    "adaptation_risk_cost": 0.0,
                    "adaptation_review_required": False,
                    "marginal_frontier_gain_predicted": 0.04,
                    "marginal_frontier_gain_realized": 0.03,
                    "data_share_credit_predicted": 0.06,
                    "data_share_credit_realized": 0.05,
                    "downweight_recommended": False,
                    "human_review_label": "pass",
                    "override_label": None,
                    "scene_tracks_non_stub": True,
                    "scene_tracks_backend": "real",
                    "teacher_runtime_backend_selected": "real",
                    "semantic_memory_grounded": True,
                }
            )
            + "\n",
            encoding="utf-8",
        )

        checkpoint_path = training_dir / "checkpoint.pt"
        checkpoint_path.write_bytes(b"probe-checkpoint")
        promotion_eval_path = training_dir / "regal_promotion_eval.json"
        promotion_eval_path.write_text(
            json.dumps({"summary": {"eligible_nodes": 1}}, indent=2),
            encoding="utf-8",
        )

        runner.set_eligible_datapacks([episode_id])
        runner.set_sampler_config(seed=seed, config_sha="probe_cfg")
        runner.record_sample("shadow_task", datapack_id=episode_id, slice_id=episode_id)
        runner.add_trajectory_audit(
            create_trajectory_audit(
                episode_id=episode_id,
                num_steps=2,
                actions=[[0.0, 0.1], [0.1, 0.2]],
                rewards=[0.2, 0.3],
                reward_components={"throughput": [0.2, 0.3]},
            )
        )
        runner.update_step(2)
        runner.configure_training_runtime(
            training_kind="nightly_readiness_probe",
            config_digest="probe_cfg",
            replay_dataset_summary=dataset.to_summary(),
            source_domain_coverage={"source_domain_counts": {"training_run": 1}},
            receipt_label_coverage={"total_labels": 1},
            objective_profile_snapshot={"profile_id": "balanced_contract"},
            promotion_policy_snapshot={"policy_name": "promotion_default"},
        )
        runner.set_regal_result({"overall_status": "pass"}, context_sha="probe_ctx")
        runner.register_artifact("online_episode_logs", episode_logs_dir)
        runner.register_artifact("online_episode_receipts", receipts_path)
        runner.register_artifact("online_replay_dataset_manifest", dataset_dir / "manifest.json")
        runner.register_artifact("regal_promotion_eval", promotion_eval_path)
        runner.register_checkpoint(
            build_checkpoint_record(
                checkpoint_id="probe_checkpoint",
                model_family="probe",
                model_version="v1",
                path=checkpoint_path,
                step=2,
                epoch=1,
            )
        )

    run_training_with_regality(
        training_fn=_train,
        config=TrainingRunConfig(
            output_dir=str(training_dir),
            seed=seed,
            num_episodes=1,
            training_steps=2,
            fail_on_verify_error=False,
        ),
        plan_sha="probe_plan_sha",
        plan_id="nightly_readiness_probe",
    )

    bundle = build_training_run_receipt_label_bundle(training_dir)
    summary = dict(bundle.metadata.get("execution_precondition_summary", {}) or {})
    satisfied = dict(summary.get("satisfied_preconditions", {}) or {})

    target_counts = {
        "signal_bool::scene_tracks_non_stub": int(satisfied.get("signal_bool::scene_tracks_non_stub", 0)),
        "signal_bool::teacher_runtime_real": int(satisfied.get("signal_bool::teacher_runtime_real", 0)),
        "signal_bool::budget_settlement_live": int(satisfied.get("signal_bool::budget_settlement_live", 0)),
    }
    target_truth = {key: value > 0 for key, value in target_counts.items()}

    report = {
        "probe_root": str(output_root),
        "training_run_dir": str(training_dir),
        "execution_precondition_summary": summary,
        "target_predicates_satisfied_count": target_counts,
        "target_predicates_truth": target_truth,
        "target_false_predicates": [key for key, ok in target_truth.items() if not ok],
    }
    return report


def _write_outputs(report: Dict[str, Any], output_root: Path) -> None:
    json_path = output_root / "readiness_probe_summary.json"
    md_path = output_root / "readiness_probe_summary.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# Receipt Readiness Probe",
        "",
        f"- Probe root: `{report['probe_root']}`",
        f"- Training run dir: `{report['training_run_dir']}`",
        "",
        "## Target Predicates",
    ]
    for key, count in report["target_predicates_satisfied_count"].items():
        truth = "true" if report["target_predicates_truth"][key] else "false"
        lines.append(f"- `{key}`: count={count}, truth={truth}")

    false_preds = report.get("target_false_predicates", [])
    lines.extend(
        [
            "",
            "## False Predicates",
            "- none" if not false_preds else "",
        ]
    )
    if false_preds:
        lines.extend([f"- `{item}`" for item in false_preds])

    summary = report.get("execution_precondition_summary", {})
    lines.extend(
        [
            "",
            "## Execution Summary",
            f"- report_count: {summary.get('report_count', 0)}",
            f"- ready_count: {summary.get('ready_count', 0)}",
            f"- blocked_count: {summary.get('blocked_count', 0)}",
            f"- mean_readiness_score: {summary.get('mean_readiness_score', 0)}",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/economic_world_model/readiness_probe"),
        help="Directory where probe artifacts are written.",
    )
    parser.add_argument("--seed", type=int, default=17, help="Probe seed.")
    args = parser.parse_args()

    report = _run_probe(args.output_root, seed=args.seed)
    _write_outputs(report, args.output_root)

    print(json.dumps(report["target_predicates_satisfied_count"], indent=2, sort_keys=True))
    print("false_predicates=", ",".join(report["target_false_predicates"]) or "none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
