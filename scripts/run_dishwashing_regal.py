#!/usr/bin/env python3
"""
Canonical dishwashing runner with full regality compliance.

REGALITY COMPLIANCE: FULL
-------------------------
Produces all required artifacts:
- RunManifestV1 + ledger.jsonl + exposure_manifest.json
- selection_manifest.json (Phase 2)
- orchestrator_state.json (Phase 1)
- trajectory_audit.json (Phase 3 - REQUIRED)
- verification_report.json (Phase 7)
- deploy_gate_inputs.json + deploy_gate_decision.json (Phase 6)

Calls verify_run() unconditionally and fails hard on any failed check.

Usage:
    python scripts/run_dishwashing_regal.py --output-dir artifacts/dishwashing_regal
    python scripts/run_dishwashing_regal.py --episodes 5 --econ-preset realistic
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import env from module, regality extensions from package
from src.envs.dishwashing_env import DishwashingEnv, summarize_episode_info
from src.envs.dishwashing_regal.rewards.reward_breakdown import compute_dishwashing_reward_breakdown
from src.envs.dishwashing_regal.trajectory_audit import DishwashingTrajectoryCollector
from src.config.internal_profile import get_internal_experiment_profile
from src.config.econ_params import load_econ_params

from src.contracts.schemas import (
    SemanticUpdatePlanV1,
    TaskGraphOp,
    PlanOpType,
    LedgerWindowV1,
    LedgerExposureV1,
    LedgerPolicyV1,
    TrajectoryAuditV1,
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


def create_dishwashing_plan() -> SemanticUpdatePlanV1:
    """Create a dishwashing training plan."""
    return SemanticUpdatePlanV1(
        plan_id="dishwashing_regal_v1",
        source_commit="dishwashing_regal",
        task_graph_changes=[
            TaskGraphOp(op=PlanOpType.SET_WEIGHT, task_family="dishwashing", weight=1.0),
        ],
        notes="Dishwashing plan with full regality compliance",
    )


def run_episode(
    env: DishwashingEnv,
    episode_id: str,
    seed: int,
    max_steps: int = 100,
) -> Tuple[Dict[str, Any], DishwashingTrajectoryCollector, List[Dict[str, Any]]]:
    """Run a single episode and collect trajectory audit data.

    Returns:
        (episode_result, trajectory_collector, info_history)
    """
    np.random.seed(seed)

    collector = DishwashingTrajectoryCollector(episode_id=episode_id)
    info_history = []

    obs = env.reset()
    prev_completed = 0
    prev_errors = 0
    prev_attempts = 0
    prev_energy = 0.0

    for step in range(max_steps):
        # Simple random policy: speed and care in [0.4, 0.8] range
        speed = np.random.uniform(0.4, 0.8)
        care = np.random.uniform(0.4, 0.8)
        action = np.array([speed, care])

        obs, info, done = env.step(action)

        # Compute deltas for trajectory collector
        delta_completed = env.completed - prev_completed
        delta_errors = env.errors - prev_errors
        delta_attempts = env.attempts - prev_attempts
        delta_energy = env.energy_Wh - prev_energy

        # Compute reward breakdown
        mpl_rate = (env.completed * 60.0) / max(env.t, 0.001)
        error_rate = env.errors / max(env.attempts, 1)

        reward_breakdown = compute_dishwashing_reward_breakdown(
            completed=delta_completed,
            attempts=delta_attempts,
            errors=delta_errors,
            speed=speed,
            care=care,
            energy_Wh=delta_energy,
            profit=info.get("profit", 0.0),
            mpl_rate=mpl_rate,
            error_rate=error_rate,
        )

        # Enrich info with deltas and breakdown
        enriched_info = dict(info)
        enriched_info["delta_completed"] = delta_completed
        enriched_info["delta_errors"] = delta_errors
        enriched_info["delta_attempts"] = delta_attempts
        enriched_info["delta_energy_Wh"] = delta_energy
        enriched_info["error_rate"] = error_rate
        enriched_info["reward_breakdown"] = reward_breakdown

        info_history.append(enriched_info)

        # Record step
        collector.record_step(action, obs, info.get("profit", 0.0), enriched_info)

        # Update prev values
        prev_completed = env.completed
        prev_errors = env.errors
        prev_attempts = env.attempts
        prev_energy = env.energy_Wh

        if done:
            break

    # Summarize episode
    summary = summarize_episode_info(info_history)

    return {
        "episode_id": episode_id,
        "success": summary.error_rate_episode < 0.12,  # SLA compliance
        "total_completed": env.completed,
        "total_errors": env.errors,
        "mpl_episode": summary.mpl_episode,
        "error_rate": summary.error_rate_episode,
        "energy_Wh": env.energy_Wh,
        "profit": summary.profit,
        "steps": env.steps,
        "termination_reason": summary.termination_reason,
    }, collector, info_history


def main():
    parser = argparse.ArgumentParser(description="Dishwashing regality runner")
    parser.add_argument("--output-dir", type=str, default="artifacts/dishwashing_regal")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--training-steps", type=int, default=500)
    parser.add_argument("--econ-preset", type=str, default="toy", choices=["toy", "realistic"])

    # Quarantine
    parser.add_argument("--quarantine", type=str, default="",
                        help="Comma-separated datapack IDs to quarantine")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = str(uuid.uuid4())[:8]
    quarantine_ids = [x.strip() for x in args.quarantine.split(",") if x.strip()]

    print("=" * 60)
    print("DISHWASHING REGAL RUNNER (Full Regality Compliance)")
    print("=" * 60)
    print(f"Run ID: {run_id}")
    print(f"Econ preset: {args.econ_preset}")
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
    plan = create_dishwashing_plan()
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
        AuditScenario("dishwashing_01", "dishwashing", "dishwashing", num_episodes=2),
    ]
    audit_config = AuditSuiteConfig(
        suite_id="dishwashing_audit_v1",
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

    baseline_weights = {"dishwashing": 1.0}
    baseline_weights_sha = sha256_json(baseline_weights)

    # Create env and run episodes
    print(f"\n[3/8] Running {args.episodes} training episodes...")
    profile = get_internal_experiment_profile("dishwashing")
    econ_params = load_econ_params(profile, preset=args.econ_preset)
    env = DishwashingEnv(econ_params)

    trajectory_audits: List[TrajectoryAuditV1] = []
    episode_results = []

    for ep_idx in range(args.episodes):
        ep_id = f"ep_{run_id}_{ep_idx:03d}"
        ep_seed = args.seed + ep_idx

        # Sample datapack (with quarantine enforcement)
        dp_id = all_datapacks[ep_idx % len(all_datapacks)]
        recorded = exposure_tracker.record_sample("dishwashing", dp_id, f"slice_{ep_idx}")

        if not recorded:
            print(f"  Episode {ep_idx}: QUARANTINED (dp={dp_id})")
            continue

        result, collector, _ = run_episode(env, ep_id, ep_seed, args.max_steps)
        episode_results.append(result)

        # Build trajectory audit from collector
        audit = collector.build_audit()
        trajectory_audits.append(audit)

        print(f"  Episode {ep_idx}: success={result['success']}, "
              f"MPL={result['mpl_episode']:.1f}/hr, "
              f"err_rate={result['error_rate']:.2%}, "
              f"profit=${result['profit']:.2f}")
        print(f"    Events: {collector.event_counts}")

    # Record orchestrator decisions
    orchestrator_tracker.set_patience("sla_gate", 3)
    orchestrator_tracker.update_step(args.training_steps)

    # Aggregate trajectory audits
    print(f"\n[4/8] Aggregating trajectory audits...")
    if trajectory_audits:
        trajectory_audit_sha = aggregate_trajectory_audits(trajectory_audits)
        audit_path = output_dir / "trajectory_audit.json"

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
        print("  WARNING: No trajectory audits collected!")

    # Run audit after (synthetic)
    print(f"\n[5/8] Running post-training audit...")
    audit_config_b = AuditSuiteConfig(
        suite_id="dishwashing_audit_v1",
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

    final_weights = {"dishwashing": 1.0}
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
        audit_delta_success=delta_success,
        trajectory_audit_sha=trajectory_audit_sha,
    )
    deploy_inputs_path = output_dir / "deploy_gate_inputs.json"
    deploy_inputs_sha = write_deploy_gate_inputs(str(deploy_inputs_path), deploy_inputs)

    deploy_decision = compute_deploy_decision(deploy_inputs, require_regal=False)
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
        notes="Dishwashing regal training run",
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
        orchestrator_state_sha=orchestrator_sha,
        selection_manifest_sha=selection_sha,
        deploy_gate_inputs_sha=deploy_inputs_sha,
        deploy_gate_decision_sha=deploy_decision_sha,
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
    print("DISHWASHING REGAL RUN COMPLETE")
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
    print("ALL CHECKS PASSED - Dishwashing regality verified")
    print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
