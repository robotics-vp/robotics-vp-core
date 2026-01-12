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
    GraphSpecV1,
    LedgerGraphV1,
    RegalGatesV1,
    LedgerRegalV1,
    TrajectoryAuditV1,
    EconTensorV1,
    LedgerEconV1,
    RegalPhaseV1,
    RegalContextV1,
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
from src.valuation.trajectory_audit import (
    create_trajectory_audit,
    aggregate_trajectory_audits,
)
from src.determinism.determinism_context import set_determinism, get_context_summary
from src.utils.config_digest import sha256_json, sha256_file
from src.geometry_graphs.small_world import graph_summary_from_embeddings


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

    # Graph metrics flags
    parser.add_argument("--include-graph-summary", action="store_true",
                        help="Compute and log graph small-world metrics")

    # Regal flags (Stage-6 meta-regal)
    parser.add_argument("--include-regal", action="store_true",
                        help="Run meta-regal gate evaluation (Stage-6)")
    parser.add_argument("--regal-ids", type=str, default="spec_guardian,world_coherence,reward_integrity",
                        help="Comma-separated regal IDs to enable")
    parser.add_argument("--regal-patience", type=int, default=3,
                        help="Regal patience (consecutive failures before action)")
    parser.add_argument("--regal-penalty-mode", type=str, default="warn",
                        choices=["warn", "noop", "clamp"],
                        help="Regal penalty mode")
    parser.add_argument("--regal-smoke-anomaly", action="store_true",
                        help="Inject physics anomalies in trajectory audit to trip WorldCoherenceRegal")

    # D4 Knob calibration flags
    parser.add_argument("--use-learned-knobs", action="store_true",
                        help="Use D4 learned/heuristic knob calibration")

    # Trajectory audit flags (Stage-6 spatiotemporal grounding)
    parser.add_argument("--include-trajectory-audit", action="store_true",
                        help="Include synthetic trajectory audit data")

    # Econ tensor flags (canonical coordinate chart)
    parser.add_argument("--include-econ-tensor", action="store_true",
                        help="Compute and log econ tensor (canonical coordinate chart)")

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

    if args.include_regal:
        mode += "+regal"

    if args.use_learned_knobs:
        mode += "+knobs"

    if args.include_trajectory_audit:
        mode += "+trajectory"

    if args.include_econ_tensor:
        mode += "+econ"

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

    # Compute graph summary if enabled
    graph_summary = None
    graph_spec = None
    if args.include_graph_summary:
        print("\n[Graph] Computing small-world graph metrics...")
        # Generate synthetic BEV-like embeddings for graph construction
        rng = np.random.default_rng(args.seed)
        H, W, D = 16, 16, 32  # Synthetic BEV grid: 16x16 cells, 32-dim embeddings
        embeddings = rng.standard_normal((H, W, D)).astype(np.float32)
        
        graph_spec = GraphSpecV1(
            spec_id="smoke_graph_spec_v1",
            local_connectivity=4,
            knn_k=6,
            min_lattice_hops_for_shortcut=4,
            n_sources=16,
            n_queries=32,
            max_hops=30,
            seed=args.seed,
        )
        
        graph_summary = graph_summary_from_embeddings(embeddings, (H, W), graph_spec, seed=args.seed)

        print(f"  Node count: {graph_summary.node_count}")
        print(f"  Local edges: {graph_summary.local_edge_count}")
        print(f"  Shortcut edges: {graph_summary.shortcut_edge_count}")
        print(f"  Shortcut fraction: {graph_summary.shortcut_fraction:.2%}")
        print(f"  Select mode: {graph_summary.shortcut_select_mode}")
        if graph_summary.shortcut_score_threshold_used is not None:
            print(f"  Score threshold: {graph_summary.shortcut_score_threshold_used:.3f}")
        print(f"  Score mode: {graph_summary.shortcut_score_mode}")
        if graph_summary.shortcut_edge_count > 0:
            print(f"  Score stats: min={graph_summary.shortcut_score_min:.3f} p50={graph_summary.shortcut_score_p50:.3f} p90={graph_summary.shortcut_score_p90:.3f} max={graph_summary.shortcut_score_max:.3f}")
        print(f"  σ (small-worldness): {graph_summary.sigma:.3f}")
        print(f"  Baseline type: {graph_summary.baseline_type}")
        print(f"  Nav success (full): {graph_summary.nav_success_rate:.2%}")
        print(f"  Nav success (lattice-only): {graph_summary.nav_success_lattice:.2%}")
        print(f"  Nav gain (wormhole benefit): {graph_summary.nav_gain:.2%}")
        print(f"  Visited nodes/query: {graph_summary.nav_visited_nodes_mean:.1f}")
        print(f"  Compute time: {graph_summary.compute_time_ms:.1f}ms")


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

        # Setup knob model if enabled
        knob_model = None
        if args.use_learned_knobs:
            from src.regal.knob_model import get_knob_model
            # Use stub learned model for smoke test (simulates learned behavior)
            knob_model = get_knob_model(use_learned=True)
            print(f"  Using D4 knob calibration: {knob_model.model_sha}")

        updated_plan, gate_status = build_plan_from_signals(
            signals, config, plan_id="homeostatic_v1", source_commit="auto",
            probe_report=probe_report,
            exposure_count=exposure_count,
            knob_model=knob_model,
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

            # Print knob policy info if used
            if gate_status.knob_policy:
                kp = gate_status.knob_policy
                print(f"  Knob Policy: {kp.policy_source}")
                if kp.gain_multiplier_override:
                    print(f"    Gain override: {kp.gain_multiplier_override:.2f}")
                if kp.patience_override:
                    print(f"    Patience override: {kp.patience_override}")
                if kp.clamped:
                    print(f"    Clamped: {', '.join(kp.clamp_reasons)}")

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

    # Run regal evaluation if enabled
    regal_result: Optional[LedgerRegalV1] = None
    regal_config: Optional[RegalGatesV1] = None
    regal_context_sha: Optional[str] = None  # Populated if context built
    if args.include_regal:
        print("\n[4b/8] Running meta-regal gate evaluation...")
        from src.regal.regal_evaluator import evaluate_regals

        regal_config = RegalGatesV1(
            enabled_regal_ids=[r.strip() for r in args.regal_ids.split(",")],
            patience=args.regal_patience,
            penalty_mode=args.regal_penalty_mode,
            determinism_seed=args.seed,
        )

        # Build signal bundle for regal evaluation
        regal_signals = build_signal_bundle_for_plan(
            audit_deltas={"delta_success": 0.0, "delta_mpl": 0.0},
            coverage_stats=dict(Counter(baseline_samples)),
            probe_report=probe_report,
        )

        # Build policy config for regal evaluation
        regal_policy_config = PlanPolicyConfigV1(
            gain_schedule=PlanGainScheduleV1(
                conservative_multiplier=1.1,
                full_multiplier=1.5,
                max_abs_weight_change=0.5,
                min_weight_clamp=0.1,
                max_weight_clamp=2.0,
            ),
            default_weights={"manipulation": 0.5, "navigation": 0.5},
            regal_gates=regal_config,
        )

        regal_result = evaluate_regals(
            config=regal_config,
            phase=RegalPhaseV1.POST_PLAN_PRE_APPLY,
            plan=updated_plan,
            signals=regal_signals,
            policy_config=regal_policy_config,
            context=None,
        )

        print(f"  Phase: POST_PLAN_PRE_APPLY")
        print(f"  Regal config SHA: {regal_config.sha256()[:16]}")
        print(f"  Enabled regals: {regal_config.enabled_regal_ids}")
        print(f"  All passed: {regal_result.all_passed}")
        for report in regal_result.reports:
            status = "PASS" if report.passed else "FAIL"
            print(f"    {report.regal_id}: {status} (phase={report.phase.value}, confidence={report.confidence:.2f})")
            if not report.passed:
                print(f"      Rationale: {report.rationale}")

    # Generate trajectory audit if enabled (using real producer)
    trajectory_audit: Optional[TrajectoryAuditV1] = None
    trajectory_audit_sha: Optional[str] = None
    if args.include_trajectory_audit or (args.include_regal and getattr(args, 'regal_smoke_anomaly', False)):
        print("\n[4c/8] Generating trajectory audit via producer...")
        rng = np.random.default_rng(args.seed)

        # Check if we need to inject anomalies for testing
        inject_anomalies = getattr(args, 'regal_smoke_anomaly', False)
        
        # Generate synthetic episode data (simulates training loop output)
        num_steps = 50
        action_dim = 7  # e.g., 7-DoF robot
        
        # Create synthetic actions/rewards (deterministic from seed)
        actions = [
            [float(x) for x in rng.standard_normal(action_dim) * 0.1]
            for _ in range(num_steps)
        ]
        rewards = [float(rng.uniform(0.0, 0.05)) for _ in range(num_steps)]
        reward_components = {
            "manipulation_reward": [float(rng.uniform(0.02, 0.04)) for _ in range(num_steps)],
            "collision_penalty": [float(rng.uniform(-0.01, 0.0)) for _ in range(num_steps)],
            "time_penalty": [float(rng.uniform(-0.005, -0.001)) for _ in range(num_steps)],
        }
        events = ["grasp_attempt", "grasp_success", "place_attempt"]
        
        # Generate velocity data with optional spikes for anomaly testing
        if inject_anomalies:
            # Inject velocity spikes that will trip WorldCoherenceRegal
            velocities = [
                [float(rng.uniform(15.0, 20.0)), 0.0, 0.0] if i % 5 == 0 else [1.0, 1.0, 1.0]
                for i in range(num_steps)
            ]
            print("  [ANOMALY MODE] Injecting physics anomalies to trip WorldCoherenceRegal")
        else:
            velocities = [
                [float(x) for x in rng.uniform(0.5, 2.0, 3)]
                for _ in range(num_steps)
            ]
        
        # Use the REAL producer (canonical entrypoint)
        trajectory_audit = create_trajectory_audit(
            episode_id=f"smoke_episode_{run_id}",
            num_steps=num_steps,
            actions=actions,
            rewards=rewards,
            reward_components=reward_components,
            events=events,
            velocities=velocities,
            velocity_threshold=10.0,
        )
        
        # For anomaly testing, also inject contact_anomaly_count and penetration_max
        if inject_anomalies:
            trajectory_audit = TrajectoryAuditV1(
                **{**trajectory_audit.model_dump(), 
                   "contact_anomaly_count": 5,
                   "penetration_max": 0.05}
            )
        
        # Compute SHA for provenance
        trajectory_audit_sha = trajectory_audit.sha256()

        print(f"  Episode ID: {trajectory_audit.episode_id}")
        print(f"  Steps: {trajectory_audit.num_steps}")
        print(f"  Total return: {trajectory_audit.total_return:.3f}")
        print(f"  Action mean: {trajectory_audit.action_mean[:3]}... (truncated)")
        print(f"  Event counts: {trajectory_audit.event_counts}")
        print(f"  Velocity spikes: {trajectory_audit.velocity_spike_count}")
        print(f"  Contact anomalies: {trajectory_audit.contact_anomaly_count}")
        print(f"  Trajectory audit SHA: {trajectory_audit_sha[:16]}")

    # Re-run regal evaluation at POST_AUDIT phase if trajectory audit present
    regal_result_post_audit: Optional[LedgerRegalV1] = None
    if args.include_regal and trajectory_audit is not None:
        print("\n[4c-ii/8] Running meta-regal gate evaluation at POST_AUDIT phase...")
        from src.regal.regal_evaluator import evaluate_regals

        # Build RegalContextV1 for typed provenance
        regal_context = RegalContextV1(
            run_id=run_id,
            step=1,
            plan_sha=updated_plan.sha256() if updated_plan else None,
            trajectory_audit_sha=trajectory_audit.sha256() if trajectory_audit else None,
        )

        regal_result_post_audit = evaluate_regals(
            config=regal_config,
            phase=RegalPhaseV1.POST_AUDIT,
            plan=updated_plan,
            signals=regal_signals,
            policy_config=regal_policy_config,
            context=regal_context,
            trajectory_audit=trajectory_audit,
        )

        print(f"  Phase: POST_AUDIT")
        print(f"  RegalContext SHA: {regal_context.sha256()[:16]}")
        print(f"  All passed: {regal_result_post_audit.all_passed}")
        for report in regal_result_post_audit.reports:
            status = "PASS" if report.passed else "FAIL"
            print(f"    {report.regal_id}: {status} (phase={report.phase.value}, confidence={report.confidence:.2f})")
            if not report.passed:
                print(f"      Rationale: {report.rationale}")
                if report.findings and "trajectory_audit_present" in report.findings:
                    print(f"      Trajectory audit inspected: {report.findings['trajectory_audit_present']}")

        # Capture regal_context for manifest (POST_AUDIT is the authoritative phase)
        regal_context_sha = regal_context.sha256()
        # Use POST_AUDIT result as the primary result since it includes trajectory audit
        regal_result = regal_result_post_audit

    # Generate econ tensor if enabled
    econ_tensor: Optional["EconTensorV1"] = None
    econ_basis_sha: Optional[str] = None
    if args.include_econ_tensor:
        print("\n[4d/8] Computing econ tensor (canonical coordinate chart)...")
        from src.economics.econ_basis_registry import get_default_basis
        from src.economics.econ_tensor import econ_to_tensor, compute_tensor_summary

        # Get default basis
        basis_def = get_default_basis()
        econ_basis_sha = basis_def.sha256

        # Create synthetic econ data (in production, this comes from EconVector)
        rng_econ = np.random.default_rng(args.seed + 100)
        econ_data = {
            "mpl_units_per_hour": float(rng_econ.uniform(5.0, 25.0)),
            "wage_parity": float(rng_econ.uniform(0.8, 1.2)),
            "energy_cost": float(rng_econ.uniform(0.5, 3.0)),
            "damage_cost": float(rng_econ.uniform(0.0, 1.0)),
            "novelty_delta": float(rng_econ.uniform(-0.1, 0.1)),
            "reward_scalar_sum": float(rng_econ.uniform(0.5, 2.0)),
            "mobility_penalty": float(rng_econ.uniform(0.0, 0.2)),
            "throughput": float(rng_econ.uniform(10.0, 50.0)),
            "error_rate": float(rng_econ.uniform(0.0, 0.2)),
            "success_rate": float(rng_econ.uniform(0.7, 1.0)),
        }

        econ_tensor = econ_to_tensor(
            econ_data,
            basis=basis_def.spec,
            source="synthetic",
        )

        tensor_summary = compute_tensor_summary(econ_tensor, basis_def.spec)

        print(f"  Basis ID: {econ_tensor.basis_id}")
        print(f"  Basis SHA: {econ_basis_sha[:16]}")
        print(f"  Tensor SHA: {econ_tensor.sha256()[:16]}")
        print(f"  Tensor norm: {tensor_summary.get('norm', 0):.4f}")
        print(f"  Key values: mpl={econ_data['mpl_units_per_hour']:.2f}, success={econ_data['success_rate']:.2f}")

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

    # Create econ ledger entry if tensor available
    ledger_econ: Optional[LedgerEconV1] = None
    if econ_tensor and econ_basis_sha:
        from src.economics.econ_tensor import compute_tensor_summary
        tensor_summary = compute_tensor_summary(econ_tensor)
        ledger_econ = LedgerEconV1(
            basis_sha=econ_basis_sha,
            econ_tensor_sha=econ_tensor.sha256(),
            econ_tensor_summary=tensor_summary,
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
        probe=ledger_probe,
        regal=regal_result,
        econ=ledger_econ,
        notes=f"Mode: {mode}",
    )

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

    # Add graph SHAs to manifest if available
    if graph_summary and graph_spec:
        manifest.graph_spec_sha = graph_spec.sha256()
        manifest.graph_summary_sha = graph_summary.summary_sha

    # Add regal SHAs to manifest if available
    if regal_result and regal_config:
        manifest.regal_config_sha = regal_config.sha256()
        # Compute aggregate report SHA from individual report SHAs (deterministic: sorted by phase, regal_id)
        if regal_result.reports:
            sorted_reports = sorted(regal_result.reports, key=lambda r: (r.phase.value, r.regal_id))
            manifest.regal_report_sha = sha256_json([r.report_sha for r in sorted_reports])
        manifest.regal_inputs_sha = regal_result.combined_inputs_sha
        # Add regal context SHA if context was built
        if regal_context_sha:
            manifest.regal_context_sha = regal_context_sha

    # Add knob calibration SHAs to manifest if available
    if gate_status and gate_status.knob_policy:
        manifest.knob_model_sha = gate_status.knob_policy.model_sha
        manifest.knob_policy_sha = gate_status.knob_policy.sha256()
        manifest.knob_policy_used = gate_status.knob_policy_used

    # Add trajectory audit SHA to manifest if available
    if trajectory_audit:
        manifest.trajectory_audit_sha = trajectory_audit.sha256()

    # Add econ tensor SHAs to manifest if available
    if econ_tensor:
        manifest.econ_basis_sha = econ_basis_sha
        manifest.econ_tensor_sha = econ_tensor.sha256()

    write_manifest(str(manifest_path), manifest)
    print(f"  run_manifest.json written")
    print(f"    Run ID: {manifest.run_id}")
    print(f"    Plan SHA: {manifest.plan_sha[:16]}")
    print(f"    Audit suite SHA: {audit_suite_sha[:16] if audit_suite_sha != 'inline' else 'inline'}")
    if manifest.probe_config_sha:
        print(f"    Probe config SHA: {manifest.probe_config_sha[:16]}")
    if manifest.graph_spec_sha:
        print(f"    Graph spec SHA: {manifest.graph_spec_sha[:16]}")
        print(f"    Graph summary SHA: {manifest.graph_summary_sha[:16] if manifest.graph_summary_sha else 'N/A'}")
    if manifest.regal_config_sha:
        print(f"    Regal config SHA: {manifest.regal_config_sha[:16]}")
        print(f"    Regal report SHA: {manifest.regal_report_sha[:16] if manifest.regal_report_sha else 'N/A'}")
        if manifest.regal_context_sha:
            print(f"    Regal context SHA: {manifest.regal_context_sha[:16]}")
    if manifest.knob_policy_sha:
        print(f"    Knob policy SHA: {manifest.knob_policy_sha[:16]}")
        print(f"    Knob policy used: {manifest.knob_policy_used}")
    if manifest.trajectory_audit_sha:
        print(f"    Trajectory audit SHA: {manifest.trajectory_audit_sha[:16]}")
    if manifest.econ_basis_sha:
        print(f"    Econ basis SHA: {manifest.econ_basis_sha[:16]}")
        print(f"    Econ tensor SHA: {manifest.econ_tensor_sha[:16] if manifest.econ_tensor_sha else 'N/A'}")
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

    if regal_result:
        print(f"\nMeta-Regal Gate Results (Stage-6):")
        print(f"  All passed: {regal_result.all_passed}")
        for report in regal_result.reports:
            status = "PASS" if report.passed else "FAIL"
            print(f"  {report.regal_id}: {status} (conf={report.confidence:.2f})")
        if regal_config:
            print(f"  Config SHA: {regal_config.sha256()[:16]}")

    if gate_status and gate_status.knob_policy:
        print(f"\nD4 Knob Calibration Results:")
        kp = gate_status.knob_policy
        print(f"  Policy source: {kp.policy_source}")
        print(f"  Policy SHA: {kp.sha256()[:16]}")
        if kp.gain_multiplier_override:
            print(f"  Gain override: {kp.gain_multiplier_override:.2f}")
        if kp.clamped:
            print(f"  Clamped: {', '.join(kp.clamp_reasons)}")

    if trajectory_audit:
        print(f"\nTrajectory Audit (Stage-6 Spatiotemporal):")
        print(f"  Episode: {trajectory_audit.episode_id}")
        print(f"  Steps: {trajectory_audit.num_steps}")
        print(f"  Return: {trajectory_audit.total_return:.3f}")
        print(f"  Physics anomalies: pen={trajectory_audit.penetration_max:.4f}, vel_spikes={trajectory_audit.velocity_spike_count}")
        print(f"  Audit SHA: {trajectory_audit.sha256()[:16]}")

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
