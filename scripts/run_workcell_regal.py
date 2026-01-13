#!/usr/bin/env python3
"""
Canonical workcell (manufacturing cell) runner with full regality compliance.

This is the PARAMOUNT env for regality - produces all required artifacts:
- RunManifestV1 + ledger.jsonl + exposure_manifest.json
- selection_manifest.json (Phase 2)
- orchestrator_state.json (Phase 1)
- trajectory_audit.json (Phase 3 - REQUIRED)
- verification_report.json (Phase 7)
- deploy_gate_inputs.json + deploy_gate_decision.json (Phase 6)

Calls verify_run() unconditionally and fails hard on any failed check.

Usage:
    python scripts/run_workcell_regal.py --output-dir artifacts/workcell_regal
    python scripts/run_workcell_regal.py --task kitting --episodes 5
    python scripts/run_workcell_regal.py --include-regal --include-econ
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.envs.workcell_env.env import WorkcellEnv
from src.envs.workcell_env.config import WorkcellEnvConfig, PRESETS
from src.envs.workcell_env.rewards.reward_breakdown import compute_workcell_reward_breakdown
from src.envs.workcell_env.trajectory_audit import WorkcellTrajectoryCollector

from src.contracts.schemas import (
    SemanticUpdatePlanV1,
    TaskGraphOp,
    PlanOpType,
    LedgerWindowV1,
    LedgerExposureV1,
    LedgerPolicyV1,
    LedgerProbeV1,
    RegalGatesV1,
    RegalPhaseV1,
    RegalContextV1,
    TrajectoryAuditV1,
    EconTensorV1,
)
from src.valuation.value_ledger import ValueLedger
from src.valuation.exposure_manifest import (
    ExposureTracker,
    write_exposure_manifest,
    write_selection_manifest,
)
from src.valuation.run_manifest import create_run_manifest, write_manifest
from src.valuation.valuation_verifier import verify_run, write_verification_report
from src.valuation.trajectory_audit import aggregate_trajectory_audits
from src.orchestrator.orchestrator_state import (
    OrchestratorStateTracker,
    write_orchestrator_state,
)
from src.deployment.deploy_gate import (
    create_deploy_gate_inputs,
    compute_deploy_decision,
    write_deploy_gate_inputs,
    write_deploy_gate_decision,
)
from src.determinism.determinism_context import set_determinism, get_context_summary
from src.utils.config_digest import sha256_json, sha256_file
from src.evaluation.audit_suite import AuditEvalSuite, AuditSuiteConfig, AuditScenario


def create_workcell_plan(task_type: str = "kitting") -> SemanticUpdatePlanV1:
    """Create a workcell training plan."""
    return SemanticUpdatePlanV1(
        plan_id=f"workcell_{task_type}_v1",
        source_commit="workcell_regal",
        task_graph_changes=[
            TaskGraphOp(op=PlanOpType.SET_WEIGHT, task_family=task_type, weight=0.8),
            TaskGraphOp(op=PlanOpType.SET_WEIGHT, task_family="assembly", weight=0.2),
        ],
        notes=f"Workcell plan for {task_type} task",
    )


def run_episode(
    env: WorkcellEnv,
    episode_id: str,
    seed: int,
    max_steps: int = 100,
) -> Tuple[Dict[str, Any], WorkcellTrajectoryCollector]:
    """Run a single episode and collect trajectory audit data.

    Returns:
        (episode_result, trajectory_collector)
    """
    collector = WorkcellTrajectoryCollector(
        episode_id=episode_id,
        velocity_threshold=10.0,
    )

    obs = env.reset(seed=seed)
    total_reward = 0.0
    success = False
    steps = 0

    for step in range(max_steps):
        # Simple random action (in real training, this would be policy output)
        action = {"action_type": "PICK", "target": f"part_{step % 3}"}

        obs, reward, terminated, truncated, info = env.step(action)

        # Compute reward breakdown
        task_info = info.get("task", {})
        reward_breakdown = compute_workcell_reward_breakdown(
            success=info.get("success", False),
            progress=task_info.get("progress", 0.0),
            time_cost=float(step),
            error_count=task_info.get("errors", 0),
            collision_count=task_info.get("collision_count", 0),
            items_picked=task_info.get("items_picked", 0),
            items_placed=task_info.get("items_placed", 0),
            items_total=task_info.get("items_total", 6),
        )

        # Add breakdown to info for collector
        info["reward_breakdown"] = reward_breakdown

        # Record step
        collector.record_step(
            action=list(range(7)),  # Simulate 7-DoF action
            obs=obs,
            reward=reward,
            info=info,
        )

        total_reward += reward
        steps += 1

        if terminated or truncated:
            success = info.get("success", False)
            break

    return {
        "episode_id": episode_id,
        "success": success,
        "total_reward": total_reward,
        "steps": steps,
        "task_info": info.get("task", {}),
    }, collector


def main():
    parser = argparse.ArgumentParser(description="Workcell regality runner")
    parser.add_argument("--output-dir", type=str, default="artifacts/workcell_regal")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--training-steps", type=int, default=500)

    # Task selection
    parser.add_argument("--task", type=str, default="kitting",
                        choices=["kitting", "peg_in_hole", "conveyor_sorting", "assembly"])

    # Quarantine
    parser.add_argument("--quarantine", type=str, default="",
                        help="Comma-separated datapack IDs to quarantine")

    # Regal
    parser.add_argument("--include-regal", action="store_true")
    parser.add_argument("--regal-ids", type=str, default="spec_guardian,world_coherence,reward_integrity")

    # Econ tensor
    parser.add_argument("--include-econ", action="store_true")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = str(uuid.uuid4())[:8]
    quarantine_ids = [x.strip() for x in args.quarantine.split(",") if x.strip()]

    print("=" * 60)
    print("WORKCELL REGAL RUNNER (Manufacturing Cell)")
    print("=" * 60)
    print(f"Run ID: {run_id}")
    print(f"Task: {args.task}")
    print(f"Episodes: {args.episodes}")
    print(f"Output: {output_dir}")
    if quarantine_ids:
        print(f"Quarantine: {quarantine_ids}")

    # Set determinism
    det_ctx = set_determinism(seed=args.seed)

    # Initialize trackers
    exposure_tracker = ExposureTracker(manifest_id=run_id, step_start=0)
    orchestrator_tracker = OrchestratorStateTracker(step=0)

    # Set quarantine
    if quarantine_ids:
        exposure_tracker.set_quarantine(quarantine_ids)

    # Set eligible datapacks (for selection manifest)
    all_datapacks = [f"dp_{run_id}_{i:03d}" for i in range(10)]
    exposure_tracker.set_eligible_datapacks(all_datapacks)
    exposure_tracker.set_sampler_config(seed=args.seed)

    # Create plan
    plan = create_workcell_plan(args.task)
    plan_path = output_dir / "plan.json"
    with open(plan_path, "w") as f:
        json.dump(plan.model_dump(mode="json"), f, indent=2)

    # Use file SHA to match verifier's computation
    plan_sha = sha256_file(str(plan_path))

    print(f"\n[1/8] Created plan: {plan.plan_id}")
    print(f"  Plan SHA: {plan_sha[:16]}")

    # Run audit before (synthetic)
    print(f"\n[2/8] Running pre-training audit...")
    audit_scenarios = [
        AuditScenario(f"{args.task}_01", args.task, args.task, num_episodes=2),
    ]
    audit_config = AuditSuiteConfig(
        suite_id="workcell_audit_v1",
        seed=args.seed,
        scenarios=audit_scenarios,
    )
    audit_suite = AuditEvalSuite(config=audit_config)
    audit_before = audit_suite.run(
        checkpoint_ref="baseline",
        policy_id="policy_baseline",
        output_dir=str(output_dir / "audit_before"),
    )
    print(f"  Success rate: {audit_before.success_rate:.2%}")

    baseline_weights = {"kitting": 0.5, "assembly": 0.5}
    baseline_weights_sha = sha256_json(baseline_weights)

    # Create env and run episodes
    print(f"\n[3/8] Running {args.episodes} training episodes...")
    config = WorkcellEnvConfig(max_steps=args.max_steps)
    env = WorkcellEnv(config=config)

    trajectory_audits: List[TrajectoryAuditV1] = []
    episode_results = []

    for ep_idx in range(args.episodes):
        ep_id = f"ep_{run_id}_{ep_idx:03d}"
        ep_seed = args.seed + ep_idx

        # Sample datapack (with quarantine enforcement)
        dp_id = all_datapacks[ep_idx % len(all_datapacks)]
        recorded = exposure_tracker.record_sample(args.task, dp_id, f"slice_{ep_idx}")

        if not recorded:
            print(f"  Episode {ep_idx}: QUARANTINED (dp={dp_id})")
            continue

        result, collector = run_episode(env, ep_id, ep_seed, args.max_steps)
        episode_results.append(result)

        # Build trajectory audit from collector
        audit = collector.build_audit()
        trajectory_audits.append(audit)

        print(f"  Episode {ep_idx}: success={result['success']}, reward={result['total_reward']:.3f}, steps={result['steps']}")
        print(f"    Events: {collector.event_counts}")

    env.close()

    # Record orchestrator decisions
    orchestrator_tracker.set_patience("task_weight_gate", 3)
    orchestrator_tracker.update_step(args.training_steps)

    # Aggregate trajectory audits
    print(f"\n[4/8] Aggregating trajectory audits...")
    aggregated_audit = None  # For regal evaluation
    if trajectory_audits:
        # aggregate_trajectory_audits returns SHA string directly
        trajectory_audit_sha = aggregate_trajectory_audits(trajectory_audits)
        audit_path = output_dir / "trajectory_audit.json"

        # Use last audit as representative for regal evaluation
        aggregated_audit = trajectory_audits[-1]

        # Compute aggregate stats
        total_return = sum(a.total_return for a in trajectory_audits)
        total_steps = sum(a.num_steps for a in trajectory_audits)

        with open(audit_path, "w") as f:
            audit_data = {
                "aggregate_sha": trajectory_audit_sha,
                "num_episodes": len(trajectory_audits),
                "total_return": total_return,
                "total_steps": total_steps,
                "episode_audits": [a.model_dump(mode="json") for a in trajectory_audits],
            }
            json.dump(audit_data, f, indent=2)
        print(f"  Aggregated {len(trajectory_audits)} audits")
        print(f"  Total return: {total_return:.3f}")
        print(f"  Trajectory audit SHA: {trajectory_audit_sha[:16]}")
    else:
        trajectory_audit_sha = None
        aggregated_audit = None
        print("  WARNING: No trajectory audits collected!")

    # Run regal if enabled
    regal_result = None
    regal_context_sha = None
    regal_config = None
    if args.include_regal:
        print(f"\n[4b/8] Running regal evaluation...")
        from src.regal.regal_evaluator import evaluate_regals

        regal_config = RegalGatesV1(
            enabled_regal_ids=[r.strip() for r in args.regal_ids.split(",")],
            patience=3,
            penalty_mode="warn",
            determinism_seed=args.seed,
        )

        regal_context = RegalContextV1(
            run_id=run_id,
            step=args.training_steps,
            plan_sha=plan_sha,
            trajectory_audit_sha=trajectory_audit_sha,
        )

        regal_result = evaluate_regals(
            config=regal_config,
            phase=RegalPhaseV1.POST_AUDIT,
            plan=plan,
            signals={},
            policy_config=None,
            context=regal_context,
            trajectory_audit=aggregated_audit,
        )

        regal_context_sha = regal_context.sha256()
        print(f"  All passed: {regal_result.all_passed}")
        for report in regal_result.reports:
            status = "PASS" if report.passed else "FAIL"
            print(f"    {report.regal_id}: {status}")

    # Compute econ tensor if enabled
    econ_tensor = None
    econ_basis_sha = None
    if args.include_econ:
        print(f"\n[4c/8] Computing econ tensor...")
        from src.economics.econ_basis_registry import get_default_basis
        from src.economics.econ_tensor import econ_to_tensor

        basis_def = get_default_basis()
        econ_basis_sha = basis_def.sha256

        success_rate = sum(1 for r in episode_results if r["success"]) / max(len(episode_results), 1)
        econ_data = {
            "mpl_units_per_hour": 15.0 * success_rate,
            "wage_parity": 0.9,
            "energy_cost": 1.5,
            "success_rate": success_rate,
        }
        econ_tensor = econ_to_tensor(econ_data, basis=basis_def.spec, source="workcell")
        print(f"  Econ tensor SHA: {econ_tensor.sha256()[:16]}")

    # Run audit after (synthetic)
    print(f"\n[5/8] Running post-training audit...")
    audit_config_b = AuditSuiteConfig(
        suite_id="workcell_audit_v1",
        seed=args.seed + 100,
        scenarios=audit_scenarios,
    )
    audit_suite_b = AuditEvalSuite(config=audit_config_b)
    audit_after = audit_suite_b.run(
        checkpoint_ref="trained",
        policy_id="policy_trained",
        output_dir=str(output_dir / "audit_after"),
    )
    print(f"  Success rate: {audit_after.success_rate:.2%}")

    delta_success = audit_after.success_rate - audit_before.success_rate
    print(f"  Delta success: {delta_success:+.2%}")

    final_weights = {"kitting": 0.8, "assembly": 0.2}
    final_weights_sha = sha256_json(final_weights)

    # Write artifacts
    print(f"\n[6/8] Writing artifacts...")

    # Exposure manifest
    exposure_tracker.update_step(args.training_steps)
    exposure_manifest = exposure_tracker.build_manifest()
    exposure_path = output_dir / "exposure_manifest.json"
    exposure_sha = write_exposure_manifest(str(exposure_path), exposure_manifest)
    print(f"  exposure_manifest.json: {exposure_sha[:16]}")

    # Selection manifest
    selection_manifest = exposure_tracker.build_selection_manifest()
    selection_path = output_dir / "selection_manifest.json"
    selection_sha = write_selection_manifest(str(selection_path), selection_manifest)
    print(f"  selection_manifest.json: {selection_sha[:16]}")
    print(f"    Rejected: {len(selection_manifest.rejected_datapacks)}")

    # Orchestrator state
    orchestrator_state = orchestrator_tracker.build_state()
    orchestrator_path = output_dir / "orchestrator_state.json"
    orchestrator_sha = write_orchestrator_state(str(orchestrator_path), orchestrator_state)
    print(f"  orchestrator_state.json: {orchestrator_sha[:16]}")

    # Deploy gate
    deploy_inputs = create_deploy_gate_inputs(
        regal_result=regal_result,
        audit_delta_success=delta_success,
        trajectory_audit_sha=trajectory_audit_sha,
        econ_tensor_sha=econ_tensor.sha256() if econ_tensor else None,
    )
    deploy_inputs_path = output_dir / "deploy_gate_inputs.json"
    deploy_inputs_sha = write_deploy_gate_inputs(str(deploy_inputs_path), deploy_inputs)

    deploy_decision = compute_deploy_decision(deploy_inputs, require_regal=args.include_regal)
    deploy_decision_path = output_dir / "deploy_gate_decision.json"
    deploy_decision_sha = write_deploy_gate_decision(str(deploy_decision_path), deploy_decision)
    print(f"  deploy_gate_inputs.json: {deploy_inputs_sha[:16]}")
    print(f"  deploy_gate_decision.json: {deploy_decision_sha[:16]}")

    # Ledger
    ts_start = datetime.now().isoformat()
    ts_end = datetime.now().isoformat()

    ledger = ValueLedger(str(output_dir / "ledger.jsonl"))
    record = ledger.create_record(
        run_id=run_id,
        plan_id=plan.plan_id,
        plan_sha=plan_sha,
        audit_before=audit_before,
        audit_after=audit_after,
        window=LedgerWindowV1(
            step_start=0,
            step_end=args.training_steps,
            ts_start=ts_start,
            ts_end=ts_end,
        ),
        exposure=LedgerExposureV1(
            datapack_ids=exposure_manifest.datapack_ids,
            slice_ids=exposure_manifest.slice_ids,
            exposure_manifest_sha=exposure_sha,
        ),
        policy=LedgerPolicyV1(
            policy_before="baseline",
            policy_after="trained",
        ),
        regal=regal_result,
        notes="Workcell regal training run",
    )
    ledger.append(record)
    print(f"  ledger.jsonl: record_id={record.record_id}")

    # Manifest
    manifest = create_run_manifest(
        run_id=run_id,
        plan_sha=plan_sha,
        audit_suite_id=audit_config.suite_id,
        audit_seed=args.seed,
        audit_config_sha=audit_before.config_sha,
        datapack_ids=exposure_manifest.datapack_ids,
        seeds=det_ctx.seed_bundle(),
        determinism_config=get_context_summary(),
        baseline_weights_sha=baseline_weights_sha,
        final_weights_sha=final_weights_sha,
        trajectory_audit_sha=trajectory_audit_sha,
        regal_context_sha=regal_context_sha,
        regal_config_sha=regal_config.sha256() if regal_config else None,
        orchestrator_state_sha=orchestrator_sha,
        selection_manifest_sha=selection_sha,
        deploy_gate_inputs_sha=deploy_inputs_sha,
        deploy_gate_decision_sha=deploy_decision_sha,
        econ_basis_sha=econ_basis_sha,
        econ_tensor_sha=econ_tensor.sha256() if econ_tensor else None,
        quarantine_manifest_sha=sha256_json(quarantine_ids) if quarantine_ids else None,
    )

    manifest_path = output_dir / "run_manifest.json"
    write_manifest(str(manifest_path), manifest)
    print(f"  run_manifest.json: {manifest.run_id}")

    # Run verification (UNCONDITIONAL)
    print(f"\n[7/8] Running verify_run() (UNCONDITIONAL)...")
    verification_report = verify_run(str(output_dir))

    # Write verification report
    verification_path = output_dir / "verification_report.json"
    verification_sha = write_verification_report(str(verification_path), verification_report)
    print(f"  verification_report.json: {verification_sha[:16]}")

    # Update manifest with verification SHA
    manifest.verification_report_sha = verification_sha
    write_manifest(str(manifest_path), manifest)

    # Print verification result
    print(f"\n[8/8] Verification Result:")
    print(f"  All passed: {verification_report.all_passed}")
    print(f"  Checks: {len(verification_report.checks)}")
    for check in verification_report.checks:
        status = "PASS" if check.passed else "FAIL"
        print(f"    [{status}] {check.check_id}: {check.message[:50]}")

    # Summary
    print("\n" + "=" * 60)
    print("WORKCELL REGAL RUN COMPLETE")
    print("=" * 60)
    print(f"\nArtifact SHAs (populated in manifest):")
    print(f"  selection_manifest_sha:   {selection_sha[:16]}")
    print(f"  orchestrator_state_sha:   {orchestrator_sha[:16]}")
    print(f"  trajectory_audit_sha:     {trajectory_audit_sha[:16] if trajectory_audit_sha else 'N/A'}")
    print(f"  verification_report_sha:  {verification_sha[:16]}")
    print(f"  deploy_gate_inputs_sha:   {deploy_inputs_sha[:16]}")

    print(f"\nDeploy Decision: {'ALLOW' if deploy_decision.allow_deploy else 'DENY'}")
    print(f"  Reason: {deploy_decision.reason}")

    print(f"\nOutput directory: {output_dir}")
    print(f"  - run_manifest.json")
    print(f"  - ledger.jsonl")
    print(f"  - exposure_manifest.json")
    print(f"  - selection_manifest.json")
    print(f"  - orchestrator_state.json")
    print(f"  - trajectory_audit.json")
    print(f"  - verification_report.json")
    print(f"  - deploy_gate_inputs.json")
    print(f"  - deploy_gate_decision.json")

    # FAIL HARD on verification failure
    if not verification_report.all_passed:
        print(f"\nERROR: Verification FAILED!")
        failed_checks = [c for c in verification_report.checks if not c.passed]
        for check in failed_checks:
            print(f"  FAILED: {check.check_id} - {check.message}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("ALL CHECKS PASSED - Manufacturing cell regality verified")
    print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
