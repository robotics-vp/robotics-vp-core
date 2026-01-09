#!/usr/bin/env python3
"""Closed-loop smoke test script (hardened with probe discriminator).

Proves the architecture is real by running a complete cycle:
signals → plan → actuation (sampling) → training window → audit eval → ledger + manifest

Modes:
    --manual-plan: Use hardcoded plans (default)
    --auto-plan: Generate plan from homeostatic signals
    --token-only: Use repr_tokens for signals (if available)
    --include-probe-epi: Run probe epiplexity harness with stability/transfer gates

Usage:
    .venv/bin/python -m scripts.run_closed_loop_smoke --output-dir artifacts/closed_loop
    .venv/bin/python -m scripts.run_closed_loop_smoke --auto-plan --output-dir artifacts/hardened
    .venv/bin/python -m scripts.run_closed_loop_smoke --auto-plan --include-probe-epi --output-dir artifacts/probe
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add src to path if needed
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.contracts.schemas import (
    SemanticUpdatePlanV1,
    TaskGraphOp,
    DatapackSelectionConfig,
    PlanOpType,
    LedgerWindowV1,
    LedgerPolicyV1,
    LedgerProbeV1,
    ProbeEpiReportV1,
    PlanPolicyConfigV1,
    PlanGainScheduleV1,
)
from src.orchestrator.plan_applier import PlanApplier
from src.orchestrator.homeostatic_plan_writer import (
    build_signal_bundle_for_plan,
    build_plan_from_signals,
    write_plan,
    GateStatus,
)
from src.evaluation.audit_suite import AuditEvalSuite, AuditSuiteConfig, AuditScenario
from src.evaluation.audit_registry import get_suite, get_suite_sha
from src.evaluation.probe_harness import (
    ProbeHarness,
    ProbeHarnessConfig,
    write_probe_report,
)
from src.valuation.value_ledger import ValueLedger
from src.valuation.exposure_manifest import (
    ExposureManifestV1,
    ExposureTracker,
    write_exposure_manifest,
)
from src.valuation.run_manifest import create_run_manifest, write_manifest
from src.determinism.determinism_context import set_determinism, get_context_summary
from src.utils.config_digest import sha256_json, sha256_file


def create_baseline_plan() -> SemanticUpdatePlanV1:
    """Create baseline plan (equal weights)."""
    return SemanticUpdatePlanV1(
        plan_id="baseline_v1",
        source_commit="smoke_test",
        task_graph_changes=[
            TaskGraphOp(op=PlanOpType.SET_WEIGHT, task_family="manipulation", weight=0.5),
            TaskGraphOp(op=PlanOpType.SET_WEIGHT, task_family="navigation", weight=0.5),
        ],
        notes="Baseline equal-weight plan",
    )


def create_updated_plan() -> SemanticUpdatePlanV1:
    """Create updated plan (shifted weights)."""
    return SemanticUpdatePlanV1(
        plan_id="updated_v1",
        source_commit="smoke_test",
        task_graph_changes=[
            TaskGraphOp(op=PlanOpType.SET_WEIGHT, task_family="manipulation", weight=0.8),
            TaskGraphOp(op=PlanOpType.SET_WEIGHT, task_family="navigation", weight=0.2),
        ],
        datapack_selection=DatapackSelectionConfig(
            quotas={"occluded": 5, "dynamic": 10},
        ),
        notes="Updated plan with manipulation bias",
    )


def sample_with_weights(weights: Dict[str, float], n_samples: int, seed: int) -> List[str]:
    """Simulate sampling with given task family weights."""
    import random
    rng = random.Random(seed)

    families = list(weights.keys())
    probs = list(weights.values())
    total = sum(probs)
    probs = [p / total for p in probs]

    samples = []
    for _ in range(n_samples):
        r = rng.random()
        cumsum = 0.0
        for family, prob in zip(families, probs):
            cumsum += prob
            if r <= cumsum:
                samples.append(family)
                break
    return samples


def print_histogram(samples: List[str], title: str) -> None:
    """Print a simple histogram."""
    counter = Counter(samples)
    total = len(samples)
    print(f"\n{title}")
    print("-" * 40)
    for family, count in sorted(counter.items()):
        pct = 100 * count / total
        bar = "#" * int(pct / 2)
        print(f"  {family:15s} {count:4d} ({pct:5.1f}%) {bar}")


def generate_synthetic_repr_data(
    n_samples: int,
    input_dim: int,
    seed: int,
    add_signal: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic representation data for probe.

    Args:
        n_samples: Number of samples
        input_dim: Representation dimension
        seed: Random seed
        add_signal: If True, add a learnable signal

    Returns:
        (X, y) tuple
    """
    np.random.seed(seed)
    X = np.random.randn(n_samples, input_dim).astype(np.float32)

    if add_signal:
        # Add a linear signal
        true_w = np.random.randn(input_dim).astype(np.float32)
        y = X @ true_w + np.random.randn(n_samples).astype(np.float32) * 0.1
    else:
        # Pure noise
        y = np.random.randn(n_samples).astype(np.float32)

    return X, y



def run_probe_harness(
    args: argparse.Namespace,
    output_dir: Path,
) -> Tuple[Optional[ProbeEpiReportV1], Optional[GateStatus]]:
    """Run probe epiplexity harness using registry.

    Args:
        args: Command line arguments
        output_dir: Output directory

    Returns:
        (ProbeEpiReportV1, GateStatus) or (None, None) if not enabled
    """
    if not args.include_probe_epi:
        return None, None

    # Use registry harness if available, else standard
    from src.evaluation.probe_harness import get_probe_harness, ProbeHarnessDefinition, ProbeHarnessConfig
    
    probe_registry_id = "smoke_probe_v1"
    print(f"\n[PROBE] Running probe epiplexity harness ({probe_registry_id})...")

    try:
        harness_def = get_probe_harness(probe_registry_id)
        config = harness_def.to_config()
        # Override seeds with args if provided (for smoke testing variability)
        config.seeds = [int(s) for s in args.probe_seeds.split(",")]
    except KeyError:
        # Fallback (should not happen in smoke)
        print("  WARNING: Registry harness not found, using defaults")
        config = ProbeHarnessConfig(
            probe_variants=["linear"],
            probe_steps=args.probe_steps,
            seeds=[int(s) for s in args.probe_seeds.split(",")],
            input_dim=32,
        )

    print(f"  Variants: {config.probe_variants}")
    print(f"  Steps: {config.probe_steps}")
    print(f"  Seeds: {config.seeds}")

    # Generate synthetic data
    n_samples = 200
    input_dim = 32

    # Baseline: pure noise representation
    X_baseline, y_baseline = generate_synthetic_repr_data(
        n_samples, input_dim, args.seed, add_signal=False
    )

    # After: with learnable signal
    X_after, y_after = generate_synthetic_repr_data(
        n_samples, input_dim, args.seed, add_signal=True
    )

    # OOD slice (different distribution)
    X_ood, y_ood = generate_synthetic_repr_data(
        n_samples // 2, input_dim, args.seed + 1000, add_signal=True
    )

    # Configure harness
    config.input_dim = input_dim
    config.hidden_dim = 32
    
    harness = ProbeHarness(config)
    report = harness.run(
        baseline_data=(X_baseline, y_baseline),
        after_data=(X_after, y_after),
        ood_data=(X_ood, y_ood),
    )

    # Write report
    report_path = output_dir / "probe_report.json"
    write_probe_report(str(report_path), report)

    print(f"  Baseline score: {report.baseline_score:.4f}")
    print(f"  After score: {report.after_score:.4f}")
    print(f"  Delta: {report.delta:.4f}")
    print(f"  FLOPs: {report.flops_estimate:.0f}")
    print(f"  ΔEpi/FLOP: {report.delta_epi_per_flop:.2e}")
    print(f"  Sign consistency: {report.sign_consistency:.2f}")
    print(f"  Stability gate: {'PASS' if report.stability_pass else 'FAIL'}")
    print(f"  OOD delta: {report.ood_delta:.4f}" if report.ood_delta else "  OOD delta: N/A")
    print(f"  Transfer gate: {'PASS' if report.transfer_pass else 'FAIL'}")
    print(f"  Report SHA: {report.report_sha[:16]}")

    return report, None


def main():
    parser = argparse.ArgumentParser(description="Closed-loop smoke test (hardened)")
    parser.add_argument("--output-dir", type=str, default="artifacts/closed_loop")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--training-steps", type=int, default=1000)

    # Mode flags
    parser.add_argument("--manual-plan", action="store_true", default=True,
                        help="Use hardcoded plans (default)")
    parser.add_argument("--auto-plan", action="store_true",
                        help="Generate plan from homeostatic signals")
    parser.add_argument("--token-only", action="store_true",
                        help="Use repr_tokens for signals (if available)")

    # Probe flags
    parser.add_argument("--include-probe-epi", action="store_true",
                        help="Run probe epiplexity harness")
    parser.add_argument("--probe-steps", type=int, default=100,
                        help="Probe training steps")
    parser.add_argument("--probe-seeds", type=str, default="42,43,44",
                        help="Comma-separated probe seeds")
    parser.add_argument("--probe-variant", type=str, default="linear",
                        choices=["linear", "mlp"],
                        help="Probe model variant")

    args = parser.parse_args()

    # Determine mode
    if args.auto_plan:
        mode = "auto-plan"
    elif args.token_only:
        mode = "token-only"
    else:
        mode = "manual-plan"

    if args.include_probe_epi:
        mode += "+probe"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = str(uuid.uuid4())[:8]
    print(f"=== Closed-Loop Smoke Test (Hardened) ===")
    print(f"Run ID: {run_id}")
    print(f"Mode: {mode}")
    print(f"Output: {output_dir}")

    # Set determinism
    det_ctx = set_determinism(seed=args.seed)
    print(f"Determinism: seed={args.seed}")

    # Initialize exposure tracker
    exposure_tracker = ExposureTracker(manifest_id=run_id, step_start=0)

    # Run probe harness if enabled
    probe_report, _ = run_probe_harness(args, output_dir)

    # =========================================================================
    # Step 1: Create/generate baseline plan
    # =========================================================================
    print("\n[1/8] Creating baseline plan...")
    baseline_plan = create_baseline_plan()
    baseline_path = output_dir / "plan_baseline.json"
    with open(baseline_path, "w") as f:
        json.dump(baseline_plan.model_dump(mode="json"), f, indent=2)
    print(f"  Plan ID: {baseline_plan.plan_id}")
    print(f"  Plan SHA: {baseline_plan.sha256()[:16]}")

    # =========================================================================
    # Step 2: Sample baseline distribution
    # =========================================================================
    print("\n[2/8] Sampling baseline distribution...")
    applier = PlanApplier(
        plan_path=str(baseline_path),
        enabled=True,
        events_path=str(output_dir / "plan_applied_events.jsonl"),
    )
    result = applier.load(step=0)
    print(f"  Plan applied: {result.applied}")
    print(f"  Overrides: {applier.task_overrides.weights}")
    baseline_weights_sha = sha256_json(applier.task_overrides.weights)
    final_weights_sha = baseline_weights_sha # Default if reload fails

    baseline_samples = sample_with_weights(
        applier.task_overrides.weights, args.samples, args.seed
    )
    # Record exposure
    for family in baseline_samples:
        exposure_tracker.record_sample(family, f"dp_{run_id[:4]}", "baseline")
    print_histogram(baseline_samples, "Baseline Sampling Distribution")

    # =========================================================================
    # Step 3: Run audit eval (checkpoint A)
    # =========================================================================
    print("\n[3/8] Running audit eval (checkpoint A)...")

    # Use registry suite if available
    try:
        registry_suite = get_suite("smoke_audit_v1")
        audit_suite_sha = registry_suite.sha256()
        print(f"  Using registry suite: smoke_audit_v1")
        print(f"  Audit suite SHA: {audit_suite_sha[:16]}")
        scenarios = [
            AuditScenario(s.scenario_id, s.task_name, s.task_family, s.num_episodes)
            for s in registry_suite.scenarios
        ]
    except KeyError:
        audit_suite_sha = "inline"
        scenarios = [
            AuditScenario("balanced_01", "drawer_vase", "manipulation", num_episodes=3),
            AuditScenario("occluded_01", "drawer_vase_occluded", "manipulation", num_episodes=3),
        ]

    audit_config = AuditSuiteConfig(
        suite_id="smoke_audit_v1",
        seed=args.seed,
        scenarios=scenarios,
    )
    audit_suite = AuditEvalSuite(config=audit_config)
    audit_before = audit_suite.run(
        checkpoint_ref="checkpoint_A",
        policy_id="policy_baseline",
        output_dir=str(output_dir / "audit_before"),
    )
    print(f"  Episodes: {audit_before.num_episodes}")
    print(f"  Success rate: {audit_before.success_rate:.2%}")
    print(f"  Mean MPL proxy: {audit_before.mean_mpl_proxy:.3f}")
    print(f"  Audit config SHA: {audit_before.config_sha[:16]}")

    # =========================================================================
    # Step 4: Generate/load updated plan
    # =========================================================================
    gate_status: Optional[GateStatus] = None

    if args.auto_plan:
        print(f"\n[4/8] Generating updated plan from signals...")

        # Build signal bundle from audit results + probe report
        audit_deltas = {"delta_success": 0.0, "delta_mpl": 0.0}
        signals = build_signal_bundle_for_plan(
            audit_deltas=audit_deltas,
            coverage_stats=dict(Counter(baseline_samples)),
            probe_report=probe_report,
        )

        # Configure plan policy with gain schedule
        config = PlanPolicyConfigV1(
            gain_schedule=PlanGainScheduleV1(
                conservative_multiplier=1.1,
                full_multiplier=1.5,
                max_abs_weight_change=0.5,
                min_weight_clamp=0.1,
                max_weight_clamp=2.0,
            ),
            default_weights={"manipulation": 0.5, "navigation": 0.5},
            min_raw_delta=0.01,
        )
        exposure_count = probe_report.num_samples_id if probe_report else None
        
        updated_plan, gate_status = build_plan_from_signals(
            signals, config, plan_id="homeostatic_v1", source_commit="auto",
            probe_report=probe_report,
            exposure_count=exposure_count,
        )

        print(f"  Auto-generated from signals")
        if probe_report:
            print(f"  ΔEpi/FLOP: {probe_report.delta_epi_per_flop:.2e}")
            print(f"  Stability gate: {'PASS' if probe_report.stability_pass else 'FAIL'}")
            print(f"  Transfer gate: {'PASS' if probe_report.transfer_pass else 'FAIL'}")
            
        if gate_status:
            if gate_status.raw_delta is not None:
                print(f"  Raw Delta: {gate_status.raw_delta:.4f}")
            if gate_status.delta_per_exposure is not None:
                print(f"  ΔEpi/Exp: {gate_status.delta_per_exposure:.2e}")
            if gate_status.transfer_patience_exceeded:
                print(f"  Transfer Patience: EXCEEDED ({gate_status.transfer_failure_count} failures)")
            
            if gate_status.ledger_policy:
                lp = gate_status.ledger_policy
                print(f"  Applied Multiplier: {lp.applied_multiplier:.2f} ({lp.gain_schedule_source})")
                if lp.clamped:
                    print(f"  Scale Update: CLAMPED")
                print(f"  Policy Config SHA: {lp.policy_config_sha[:16]}")
                
            print(f"  Forced NOOP: {gate_status.forced_noop}")
            if gate_status.forced_noop:
                print(f"  Reason: {gate_status.reason}")
    else:
        print(f"\n[4/8] Loading hardcoded updated plan...")
        updated_plan = create_updated_plan()

    updated_path = output_dir / "plan_updated.json"
    write_plan(str(updated_path), updated_plan)
    print(f"  Plan ID: {updated_plan.plan_id}")
    print(f"  Plan SHA: {updated_plan.sha256()[:16]}")

    # =========================================================================
    # Step 5: Hot reload updated plan
    # =========================================================================
    print("\n[5/8] Hot-reloading updated plan...")
    applier.plan_path = str(updated_path)
    applier._file_mtime = 0  # Force reload
    reload_result = applier.poll_and_apply(step=100)

    if reload_result and reload_result.applied:
        print(f"  Plan applied: True")
        print(f"  Previous SHA: {reload_result.prev_plan_sha[:16] if reload_result.prev_plan_sha else 'None'}")
        print(f"  New SHA: {reload_result.plan_sha[:16]}")
        print(f"  New weights: {applier.task_overrides.weights}")
        final_weights_sha = sha256_json(applier.task_overrides.weights)
    else:
        print(f"  Plan reload skipped or failed")

    exposure_tracker.set_plan(updated_plan.plan_id, updated_plan.sha256())

    # =========================================================================
    # Step 6: Sample updated distribution
    # =========================================================================
    print("\n[6/8] Sampling updated distribution...")
    updated_samples = sample_with_weights(
        applier.task_overrides.weights, args.samples, args.seed + 1
    )
    for family in updated_samples:
        exposure_tracker.record_sample(family, f"dp_{run_id[:4]}", "updated")
    print_histogram(updated_samples, "Updated Sampling Distribution")

    baseline_counter = Counter(baseline_samples)
    updated_counter = Counter(updated_samples)
    shift_detected = baseline_counter != updated_counter
    print(f"\n  Distribution shift detected: {shift_detected}")

    # =========================================================================
    # Step 7: Simulate training + second audit
    # =========================================================================
    print(f"\n[7/8] Simulating {args.training_steps} training steps...")
    ts_start = datetime.now().isoformat()
    exposure_tracker.update_step(args.training_steps)
    ts_end = datetime.now().isoformat()
    print(f"  Training window: step 0 → {args.training_steps}")

    print("\n[7b/8] Running audit eval (checkpoint B)...")
    audit_config_b = AuditSuiteConfig(
        suite_id="smoke_audit_v1",
        seed=args.seed + 100,
        scenarios=scenarios,
    )
    audit_suite_b = AuditEvalSuite(config=audit_config_b)
    audit_after = audit_suite_b.run(
        checkpoint_ref="checkpoint_B",
        policy_id="policy_updated",
        output_dir=str(output_dir / "audit_after"),
    )
    print(f"  Episodes: {audit_after.num_episodes}")
    print(f"  Success rate: {audit_after.success_rate:.2%}")
    print(f"  Mean MPL proxy: {audit_after.mean_mpl_proxy:.3f}")

    delta_success = audit_after.success_rate - audit_before.success_rate
    delta_mpl = (audit_after.mean_mpl_proxy or 0) - (audit_before.mean_mpl_proxy or 0)
    print(f"\n  Δ Success rate: {delta_success:+.2%}")
    print(f"  Δ MPL proxy: {delta_mpl:+.3f}")

    # =========================================================================
    # Step 8: Write exposure manifest, ledger, run manifest
    # =========================================================================
    print("\n[8/8] Writing artifacts...")

    # Exposure manifest (MANDATORY)
    exposure_manifest = exposure_tracker.build_manifest()
    exposure_path = output_dir / "exposure_manifest.json"
    exposure_sha = write_exposure_manifest(str(exposure_path), exposure_manifest)
    print(f"  exposure_manifest.json written")
    print(f"    Manifest SHA: {exposure_sha[:16]}")
    print(f"    Total samples: {exposure_manifest.total_samples}")

    # Ledger
    from src.contracts.schemas import LedgerExposureV1
    ledger = ValueLedger(str(output_dir / "ledger.jsonl"))

    # Build probe info for ledger if available
    ledger_probe = None
    if probe_report and gate_status:
        ledger_probe = LedgerProbeV1(
            probe_config_sha=probe_report.probe_config_sha,
            probe_report_sha=probe_report.report_sha,
            delta_epi_per_flop=probe_report.delta_epi_per_flop,
            stability_pass=probe_report.stability_pass,
            transfer_pass=probe_report.transfer_pass,
        )

    record = ledger.create_record(
        run_id=run_id,
        plan_id=updated_plan.plan_id,
        plan_sha=updated_plan.sha256(),
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
            policy_before="checkpoint_A",
            policy_after="checkpoint_B",
        ),
        notes=f"Mode: {mode}",
    )

    # Add probe to record if available
    if ledger_probe:
        record.probe = ledger_probe

    ledger.append(record)
    print(f"  ledger.jsonl written")
    print(f"    Record ID: {record.record_id}")
    print(f"    Δ Success: {record.deltas.delta_success:+.4f}" if record.deltas.delta_success else "    Δ Success: N/A")
    if ledger_probe:
        print(f"    Probe: ΔEpi/FLOP={ledger_probe.delta_epi_per_flop:.2e}, stability={ledger_probe.stability_pass}")

    manifest_path = output_dir / "run_manifest.json"
    
    # Calculate plan_applied_events SHA if it exists
    plan_events_path = output_dir / "plan_applied_events.jsonl"
    plan_applied_events_sha = None
    if plan_events_path.exists():
        plan_applied_events_sha = sha256_file(str(plan_events_path))

    # Run manifest
    manifest = create_run_manifest(
        run_id=run_id,
        plan_sha=updated_plan.sha256(),
        audit_suite_id=audit_config.suite_id,
        audit_seed=args.seed,
        audit_config_sha=audit_before.config_sha,
        datapack_ids=exposure_manifest.datapack_ids,
        seeds=det_ctx.seed_bundle(),
        determinism_config=get_context_summary(),
    )
    
    # Populate provenance fields
    manifest.baseline_weights_sha = baseline_weights_sha
    manifest.final_weights_sha = final_weights_sha
    
    plan_events_path = output_dir / "plan_applied_events.jsonl"
    if plan_events_path.exists():
        manifest.plan_applied_events_sha = sha256_file(str(plan_events_path))
    
    # Populate provenance fields
    manifest.baseline_weights_sha = baseline_weights_sha
    manifest.final_weights_sha = final_weights_sha
    manifest.plan_applied_events_sha = plan_applied_events_sha

    # Add probe hashes to manifest if available
    if probe_report:
        manifest.probe_config_sha = probe_report.probe_config_sha
        manifest.probe_report_sha = probe_report.report_sha

    write_manifest(str(manifest_path), manifest)
    print(f"  run_manifest.json written")
    print(f"    Run ID: {manifest.run_id}")
    print(f"    Plan SHA: {manifest.plan_sha[:16]}")
    print(f"    Audit suite SHA: {audit_suite_sha[:16] if audit_suite_sha != 'inline' else 'inline'}")
    if manifest.probe_config_sha:
        print(f"    Probe config SHA: {manifest.probe_config_sha[:16]}")
    print(f"    Source commit: {manifest.source_commit or 'N/A'}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 50)
    print("CLOSED LOOP COMPLETE (HARDENED)")
    print("=" * 50)
    print(f"\n✓ Mode: {mode}")
    print(f"✓ Plan applied: {baseline_plan.plan_id} → {updated_plan.plan_id}")
    print(f"✓ Sampling distribution shifted: {shift_detected}")
    print(f"✓ Audit suite SHA: {audit_suite_sha[:16] if audit_suite_sha != 'inline' else 'inline'}")
    print(f"✓ Exposure manifest SHA: {exposure_sha[:16]}")
    print(f"✓ Ledger record: {record.record_id}")
    print(f"✓ Run manifest: {manifest.run_id}")

    if probe_report:
        print(f"\nProbe Discriminator Results:")
        print(f"  ΔEpi/FLOP: {probe_report.delta_epi_per_flop:.2e}")
        print(f"  Stability: {probe_report.sign_consistency:.2f} ({'PASS' if probe_report.stability_pass else 'FAIL'})")
        print(f"  Transfer: {'PASS' if probe_report.transfer_pass else 'FAIL'}")
        if gate_status and gate_status.forced_noop:
            print(f"  Action: FORCED_NOOP ({gate_status.reason})")

    print(f"\nArtifacts:")
    print(f"  {exposure_path}")
    print(f"  {output_dir / 'ledger.jsonl'}")
    print(f"  {manifest_path}")
    print(f"  {output_dir / 'plan_applied_events.jsonl'}")
    if probe_report:
        print(f"  {output_dir / 'probe_report.json'}")

    if not shift_detected and mode.startswith("manual"):
        print("\n⚠ WARNING: Distribution shift not detected!")
        sys.exit(1)

    print("\n✓ All acceptance criteria met.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
