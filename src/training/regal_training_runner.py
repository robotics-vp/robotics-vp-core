"""
Canonical training runner with full regality compliance.

All training scripts should use this wrapper to ensure:
- Full artifact production (manifest, ledger, exposure, selection, orchestrator state)
- Trajectory audit enforcement (required for training runs)
- Quarantine enforcement
- verify_run() called unconditionally

Phase C: Canonical runner wrapper for regality compliance.
"""
from __future__ import annotations

import json
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.contracts.schemas import (
    RunManifestV1,
    TrajectoryAuditV1,
    EconTensorV1,
    RegalContextV1,
    LedgerWindowV1,
    LedgerExposureV1,
    LedgerPolicyV1,
    OrchestratorStateV1,
    SelectionManifestV1,
    DeployGateInputsV1,
    DeployGateDecisionV1,
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


@dataclass
class TrainingRunConfig:
    """Configuration for a training run."""

    run_id: Optional[str] = None
    output_dir: str = "artifacts/training"
    seed: int = 42

    # Training params
    num_episodes: int = 10
    training_steps: int = 1000

    # Audit suite
    audit_suite_id: str = "default"
    audit_seed: int = 42

    # Quarantine
    quarantine_datapack_ids: List[str] = field(default_factory=list)

    # Regal
    enable_regal: bool = True
    regal_ids: List[str] = field(default_factory=lambda: ["spec_guardian", "world_coherence", "reward_integrity"])

    # Require trajectory audit (enforced)
    require_trajectory_audit: bool = True

    # Fail hard on verification failure
    fail_on_verify_error: bool = True


@dataclass
class TrainingRunResult:
    """Result of a training run."""

    run_id: str
    success: bool
    output_dir: Path

    # Artifact SHAs
    manifest_sha: Optional[str] = None
    ledger_sha: Optional[str] = None
    exposure_sha: Optional[str] = None
    selection_manifest_sha: Optional[str] = None
    orchestrator_state_sha: Optional[str] = None
    trajectory_audit_sha: Optional[str] = None
    verification_report_sha: Optional[str] = None
    deploy_gate_inputs_sha: Optional[str] = None

    # Verification result
    verify_all_passed: bool = False
    verify_failed_checks: List[str] = field(default_factory=list)

    # Deploy decision
    allow_deploy: bool = False
    deploy_reason: str = ""


class RegalTrainingRunner:
    """Canonical training runner with full regality compliance.

    Ensures all training runs produce required artifacts and pass verification.
    """

    def __init__(self, config: TrainingRunConfig):
        self.config = config
        self.run_id = config.run_id or str(uuid.uuid4())[:8]
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize trackers
        self.exposure_tracker = ExposureTracker(
            manifest_id=self.run_id,
            step_start=0,
        )
        self.orchestrator_tracker = OrchestratorStateTracker(step=0)

        # Set quarantine
        if config.quarantine_datapack_ids:
            self.exposure_tracker.set_quarantine(config.quarantine_datapack_ids)

        # Trajectory audits (aggregated per window)
        self._trajectory_audits: List[TrajectoryAuditV1] = []

        # Timestamps
        self._ts_start: Optional[str] = None
        self._ts_end: Optional[str] = None

        # Weights tracking
        self._baseline_weights_sha: Optional[str] = None
        self._final_weights_sha: Optional[str] = None

        # Regal results
        self._regal_result: Optional[Any] = None
        self._regal_context_sha: Optional[str] = None

        # Audit results
        self._audit_before: Optional[Any] = None
        self._audit_after: Optional[Any] = None

        # Econ tensor
        self._econ_tensor: Optional[EconTensorV1] = None
        self._econ_basis_sha: Optional[str] = None

    def start_training(self) -> None:
        """Called at start of training."""
        self._ts_start = datetime.now().isoformat()
        set_determinism(seed=self.config.seed)

    def record_sample(
        self,
        task_family: str,
        datapack_id: Optional[str] = None,
        slice_id: Optional[str] = None,
    ) -> bool:
        """Record a training sample (with quarantine enforcement).

        Returns:
            True if sample was recorded, False if excluded due to quarantine
        """
        return self.exposure_tracker.record_sample(task_family, datapack_id, slice_id)

    def set_eligible_datapacks(self, datapack_ids: List[str]) -> None:
        """Set eligible datapacks for selection manifest."""
        self.exposure_tracker.set_eligible_datapacks(datapack_ids)

    def set_sampler_config(self, seed: int, config_sha: Optional[str] = None) -> None:
        """Set sampler config for selection manifest."""
        self.exposure_tracker.set_sampler_config(seed, config_sha)

    def record_rejection(self, datapack_id: str, reason: str) -> None:
        """Record a datapack rejection."""
        self.exposure_tracker.record_rejection(datapack_id, reason)

    def add_trajectory_audit(self, audit: TrajectoryAuditV1) -> None:
        """Add a trajectory audit from an episode."""
        self._trajectory_audits.append(audit)

    def record_orchestrator_failure(self, gate_id: str) -> None:
        """Record an orchestrator gate failure."""
        self.orchestrator_tracker.record_failure(gate_id)

    def record_orchestrator_clamp(self, gate_id: str, trigger: str, value: Any) -> None:
        """Record an orchestrator clamp decision."""
        self.orchestrator_tracker.record_clamp(gate_id, trigger, value)

    def update_step(self, step: int) -> None:
        """Update current training step."""
        self.exposure_tracker.update_step(step)
        self.orchestrator_tracker.update_step(step)

    def set_weights(
        self,
        baseline_weights: Optional[Dict[str, float]] = None,
        final_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        """Set weight SHAs for manifest."""
        if baseline_weights:
            self._baseline_weights_sha = sha256_json(baseline_weights)
        if final_weights:
            self._final_weights_sha = sha256_json(final_weights)

    def set_audit_results(self, before: Any, after: Any) -> None:
        """Set audit results for ledger."""
        self._audit_before = before
        self._audit_after = after

    def set_regal_result(self, result: Any, context_sha: Optional[str] = None) -> None:
        """Set regal evaluation result."""
        self._regal_result = result
        self._regal_context_sha = context_sha

    def set_econ_tensor(self, tensor: EconTensorV1, basis_sha: str) -> None:
        """Set econ tensor for manifest."""
        self._econ_tensor = tensor
        self._econ_basis_sha = basis_sha

    def finalize(
        self,
        plan_sha: str,
        plan_id: str = "training_plan",
    ) -> TrainingRunResult:
        """Finalize training run and write all artifacts.

        Args:
            plan_sha: SHA of the training plan
            plan_id: Plan identifier

        Returns:
            TrainingRunResult with all artifact SHAs and verification result
        """
        self._ts_end = datetime.now().isoformat()

        # Aggregate trajectory audits (REQUIRED for training)
        trajectory_audit_sha: Optional[str] = None
        if self._trajectory_audits:
            # aggregate_trajectory_audits returns SHA string directly
            trajectory_audit_sha = aggregate_trajectory_audits(self._trajectory_audits)

            # Write trajectory audit file
            audit_path = self.output_dir / "trajectory_audit.json"
            with open(audit_path, "w") as f:
                audit_data = {
                    "aggregate_sha": trajectory_audit_sha,
                    "num_episodes": len(self._trajectory_audits),
                    "episode_audits": [a.model_dump(mode="json") for a in self._trajectory_audits],
                }
                json.dump(audit_data, f, indent=2)

        elif self.config.require_trajectory_audit:
            print("WARNING: No trajectory audits recorded but require_trajectory_audit=True")
            print("         Training run will fail verification!")

        # Build exposure manifest
        exposure_manifest = self.exposure_tracker.build_manifest()
        exposure_path = self.output_dir / "exposure_manifest.json"
        exposure_sha = write_exposure_manifest(str(exposure_path), exposure_manifest)

        # Build selection manifest
        selection_manifest = self.exposure_tracker.build_selection_manifest()
        selection_path = self.output_dir / "selection_manifest.json"
        selection_sha = write_selection_manifest(str(selection_path), selection_manifest)

        # Build orchestrator state
        orchestrator_state = self.orchestrator_tracker.build_state()
        orchestrator_path = self.output_dir / "orchestrator_state.json"
        orchestrator_sha = write_orchestrator_state(str(orchestrator_path), orchestrator_state)

        # Build deploy gate inputs
        deploy_inputs = create_deploy_gate_inputs(
            regal_result=self._regal_result,
            audit_delta_success=(
                self._audit_after.success_rate - self._audit_before.success_rate
                if self._audit_before and self._audit_after else None
            ),
            trajectory_audit_sha=trajectory_audit_sha,
            econ_tensor_sha=self._econ_tensor.sha256() if self._econ_tensor else None,
        )
        deploy_inputs_path = self.output_dir / "deploy_gate_inputs.json"
        deploy_inputs_sha = write_deploy_gate_inputs(str(deploy_inputs_path), deploy_inputs)

        # Compute deploy decision
        deploy_decision = compute_deploy_decision(deploy_inputs, require_regal=self.config.enable_regal)
        deploy_decision_path = self.output_dir / "deploy_gate_decision.json"
        deploy_decision_sha = write_deploy_gate_decision(str(deploy_decision_path), deploy_decision)

        # Create ledger
        ledger = ValueLedger(str(self.output_dir / "ledger.jsonl"))

        record = ledger.create_record(
            run_id=self.run_id,
            plan_id=plan_id,
            plan_sha=plan_sha,
            audit_before=self._audit_before,
            audit_after=self._audit_after,
            window=LedgerWindowV1(
                step_start=0,
                step_end=self.config.training_steps,
                ts_start=self._ts_start or "",
                ts_end=self._ts_end or "",
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
            regal=self._regal_result,
            notes=f"Regal training run: {self.run_id}",
        )
        ledger.append(record)
        ledger_sha = sha256_file(str(self.output_dir / "ledger.jsonl"))

        # Create manifest (will be updated with verification_report_sha after verify)
        manifest = create_run_manifest(
            run_id=self.run_id,
            plan_sha=plan_sha,
            audit_suite_id=self.config.audit_suite_id,
            audit_seed=self.config.audit_seed,
            audit_config_sha=self._audit_before.config_sha if self._audit_before else "",
            datapack_ids=exposure_manifest.datapack_ids,
            seeds={"audit": self.config.seed},
            determinism_config=get_context_summary(),
            baseline_weights_sha=self._baseline_weights_sha,
            final_weights_sha=self._final_weights_sha,
            trajectory_audit_sha=trajectory_audit_sha,
            regal_context_sha=self._regal_context_sha,
            orchestrator_state_sha=orchestrator_sha,
            selection_manifest_sha=selection_sha,
            deploy_gate_inputs_sha=deploy_inputs_sha,
            deploy_gate_decision_sha=deploy_decision_sha,
            econ_basis_sha=self._econ_basis_sha,
            econ_tensor_sha=self._econ_tensor.sha256() if self._econ_tensor else None,
            quarantine_manifest_sha=sha256_json(self.config.quarantine_datapack_ids) if self.config.quarantine_datapack_ids else None,
        )

        # Write initial manifest
        manifest_path = self.output_dir / "run_manifest.json"
        write_manifest(str(manifest_path), manifest)

        # Run verification (UNCONDITIONAL)
        print(f"\n[VERIFY] Running verify_run({self.output_dir})...")
        verification_report = verify_run(str(self.output_dir))

        # Write verification report
        verification_path = self.output_dir / "verification_report.json"
        verification_sha = write_verification_report(str(verification_path), verification_report)

        # Update manifest with verification_report_sha
        manifest.verification_report_sha = verification_sha
        write_manifest(str(manifest_path), manifest)
        manifest_sha = sha256_file(str(manifest_path))

        # Build result
        failed_checks = [c.check_id for c in verification_report.checks if not c.passed]

        result = TrainingRunResult(
            run_id=self.run_id,
            success=verification_report.all_passed,
            output_dir=self.output_dir,
            manifest_sha=manifest_sha,
            ledger_sha=ledger_sha,
            exposure_sha=exposure_sha,
            selection_manifest_sha=selection_sha,
            orchestrator_state_sha=orchestrator_sha,
            trajectory_audit_sha=trajectory_audit_sha,
            verification_report_sha=verification_sha,
            deploy_gate_inputs_sha=deploy_inputs_sha,
            verify_all_passed=verification_report.all_passed,
            verify_failed_checks=failed_checks,
            allow_deploy=deploy_decision.allow_deploy,
            deploy_reason=deploy_decision.reason,
        )

        # Print summary
        print(f"\n{'='*60}")
        print("TRAINING RUN FINALIZED")
        print(f"{'='*60}")
        print(f"Run ID: {self.run_id}")
        print(f"Output: {self.output_dir}")
        print(f"\nArtifact SHAs:")
        print(f"  manifest_sha:             {manifest_sha[:16] if manifest_sha else 'N/A'}")
        print(f"  exposure_sha:             {exposure_sha[:16]}")
        print(f"  selection_manifest_sha:   {selection_sha[:16]}")
        print(f"  orchestrator_state_sha:   {orchestrator_sha[:16]}")
        print(f"  trajectory_audit_sha:     {trajectory_audit_sha[:16] if trajectory_audit_sha else 'MISSING'}")
        print(f"  verification_report_sha:  {verification_sha[:16]}")
        print(f"  deploy_gate_inputs_sha:   {deploy_inputs_sha[:16]}")
        print(f"\nVerification: {'PASS' if verification_report.all_passed else 'FAIL'}")
        if failed_checks:
            print(f"  Failed checks: {failed_checks}")
        print(f"\nDeploy Decision: {'ALLOW' if deploy_decision.allow_deploy else 'DENY'}")
        print(f"  Reason: {deploy_decision.reason}")

        # Fail hard if configured
        if self.config.fail_on_verify_error and not verification_report.all_passed:
            print(f"\nERROR: Verification failed, exiting with error")
            sys.exit(1)

        return result


def run_training_with_regality(
    training_fn: Callable[[RegalTrainingRunner], None],
    config: TrainingRunConfig,
    plan_sha: str,
    plan_id: str = "training_plan",
) -> TrainingRunResult:
    """Run a training function with full regality compliance.

    Args:
        training_fn: Function that performs training, receives runner for recording
        config: Training run configuration
        plan_sha: SHA of the training plan
        plan_id: Plan identifier

    Returns:
        TrainingRunResult with all artifact SHAs and verification result
    """
    runner = RegalTrainingRunner(config)
    runner.start_training()

    # Run the training
    training_fn(runner)

    # Finalize and verify
    return runner.finalize(plan_sha=plan_sha, plan_id=plan_id)


__all__ = [
    "TrainingRunConfig",
    "TrainingRunResult",
    "RegalTrainingRunner",
    "run_training_with_regality",
]
