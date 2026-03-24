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
from typing import Any, Callable, Dict, List, Mapping, Optional

from src.contracts.schemas import (
    TrajectoryAuditV1,
    AuditAggregateV1,
    EconTensorV1,
    LedgerWindowV1,
    LedgerExposureV1,
    LedgerPolicyV1,
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
from src.training.checkpoint_registry import (
    CheckpointRecord,
    create_checkpoint_registry,
    write_checkpoint_registry,
)
from src.training.training_manifest import (
    TrainingRuntimeManifest,
    build_training_runtime_summary_markdown,
    write_training_runtime_manifest,
)


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

    # Runtime unification
    training_runtime_manifest_sha: Optional[str] = None
    checkpoint_registry_sha: Optional[str] = None
    runtime_status: str = "unknown"
    failure_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "success": bool(self.success),
            "output_dir": str(self.output_dir),
            "manifest_sha": self.manifest_sha,
            "ledger_sha": self.ledger_sha,
            "exposure_sha": self.exposure_sha,
            "selection_manifest_sha": self.selection_manifest_sha,
            "orchestrator_state_sha": self.orchestrator_state_sha,
            "trajectory_audit_sha": self.trajectory_audit_sha,
            "verification_report_sha": self.verification_report_sha,
            "deploy_gate_inputs_sha": self.deploy_gate_inputs_sha,
            "verify_all_passed": bool(self.verify_all_passed),
            "verify_failed_checks": list(self.verify_failed_checks),
            "allow_deploy": bool(self.allow_deploy),
            "deploy_reason": self.deploy_reason,
            "training_runtime_manifest_sha": self.training_runtime_manifest_sha,
            "checkpoint_registry_sha": self.checkpoint_registry_sha,
            "runtime_status": self.runtime_status,
            "failure_reason": self.failure_reason,
        }


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

        # Unified training runtime state
        self._training_kind: Optional[str] = None
        self._training_runtime_context: Dict[str, Any] = {}
        self._runtime_artifacts: Dict[str, str] = {}
        self._runtime_artifact_metadata: Dict[str, Dict[str, Any]] = {}
        self._checkpoint_records: List[CheckpointRecord] = []
        self._runtime_status: str = "initialized"
        self._failure_reason: Optional[str] = None
        self._runtime_started_at: Optional[str] = None
        self._runtime_ended_at: Optional[str] = None

    def start_training(self) -> None:
        """Called at start of training."""
        self._ts_start = datetime.now().isoformat()
        self._runtime_started_at = self._ts_start
        self._runtime_status = "running"
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

    def configure_training_runtime(
        self,
        *,
        training_kind: str,
        config_path: Optional[str] = None,
        config_digest: str = "",
        replay_dataset_dir: Optional[str] = None,
        replay_manifest_digest: Optional[str] = None,
        replay_dataset_summary: Optional[Mapping[str, Any]] = None,
        objective_profile_snapshot: Optional[Mapping[str, Any]] = None,
        promotion_policy_snapshot: Optional[Mapping[str, Any]] = None,
        source_domain_coverage: Optional[Mapping[str, Any]] = None,
        receipt_label_coverage: Optional[Mapping[str, Any]] = None,
        artifact_schema_compatibility: Optional[List[Mapping[str, Any]]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Store unified runtime metadata for training-manifest emission."""
        self._training_kind = str(training_kind)
        self._training_runtime_context = {
            "config_path": config_path,
            "config_digest": config_digest,
            "replay_dataset_dir": replay_dataset_dir,
            "replay_manifest_digest": replay_manifest_digest,
            "replay_dataset_summary": dict(replay_dataset_summary or {}),
            "objective_profile_snapshot": dict(objective_profile_snapshot or {}),
            "promotion_policy_snapshot": dict(promotion_policy_snapshot or {}),
            "source_domain_coverage": dict(source_domain_coverage or {}),
            "receipt_label_coverage": dict(receipt_label_coverage or {}),
            "artifact_schema_compatibility": [
                dict(row) for row in (artifact_schema_compatibility or [])
            ],
            "metadata": dict(metadata or {}),
        }

    def register_artifact(
        self,
        artifact_id: str,
        path: str | Path,
        *,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Register a training-runtime artifact for the unified manifest."""
        self._runtime_artifacts[str(artifact_id)] = str(path)
        self._runtime_artifact_metadata[str(artifact_id)] = dict(metadata or {})

    def register_checkpoint(self, checkpoint: CheckpointRecord) -> None:
        """Register a checkpoint for the unified checkpoint registry."""
        self._checkpoint_records.append(checkpoint)

    def set_runtime_status(self, status: str, *, failure_reason: Optional[str] = None) -> None:
        """Update runtime status/failure information for summary artifacts."""
        self._runtime_status = str(status)
        self._runtime_ended_at = datetime.now().isoformat()
        self._failure_reason = failure_reason

    def _json_object(self, path: Optional[str]) -> Dict[str, Any]:
        if not path:
            return {}
        candidate = Path(path)
        if not candidate.exists():
            return {}
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return dict(payload) if isinstance(payload, Mapping) else {}

    def _write_promotion_ledger_artifact(
        self,
        *,
        effective_status: str,
    ) -> tuple[Optional[str], Optional[str]]:
        promotion_evidence_path = str(
            self._runtime_artifacts.get("promotion_ledger_ref")
            or self._runtime_artifacts.get("regal_promotion_eval")
            or ""
        )
        if not promotion_evidence_path:
            return None, None
        promotion_summary = self._json_object(promotion_evidence_path).get("summary", {})
        if not isinstance(promotion_summary, Mapping):
            promotion_summary = {}
        ledger_payload = {
            "schema_version": "promotion_ledger_v1",
            "run_id": self.run_id,
            "training_kind": self._training_kind or "training_job",
            "status": effective_status,
            "promotion_evidence_ref": promotion_evidence_path,
            "promotion_evidence_digest": (
                sha256_file(promotion_evidence_path)
                if Path(promotion_evidence_path).exists()
                else None
            ),
            "promotion_policy_snapshot": dict(self._training_runtime_context.get("promotion_policy_snapshot", {}) or {}),
            "source_domain_coverage": dict(self._training_runtime_context.get("source_domain_coverage", {}) or {}),
            "receipt_label_coverage": dict(self._training_runtime_context.get("receipt_label_coverage", {}) or {}),
            "summary": dict(promotion_summary),
            "metadata": {
                "runtime_status": effective_status,
                "artifact_ids": sorted(self._runtime_artifacts),
            },
        }
        ledger_path = self.output_dir / "promotion_ledger_v1.json"
        ledger_path.write_text(
            json.dumps(ledger_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        ledger_sha = sha256_file(ledger_path)
        self._runtime_artifacts["promotion_ledger_ref"] = str(ledger_path)
        self._runtime_artifact_metadata["promotion_ledger_ref"] = {
            "schema_version": "promotion_ledger_v1",
            "live": True,
        }
        return str(ledger_path), ledger_sha

    def _write_budget_settlement_artifact(
        self,
        *,
        effective_status: str,
        promotion_ledger_path: Optional[str],
    ) -> tuple[Optional[str], Optional[str], bool]:
        metadata = dict(self._training_runtime_context.get("metadata", {}) or {})
        explicit_live = metadata.get("budget_settlement_live")
        online_receipts_path = self._runtime_artifacts.get("online_episode_receipts")
        receipt_bundle_path = self._runtime_artifacts.get("receipt_label_bundle")
        receipt_summary_path = self._runtime_artifacts.get("receipt_label_summary")
        live = (
            bool(explicit_live)
            if explicit_live is not None
            else bool(online_receipts_path and Path(online_receipts_path).exists())
        )
        payload = {
            "schema_version": "budget_settlement_v1",
            "run_id": self.run_id,
            "training_kind": self._training_kind or "training_job",
            "status": effective_status,
            "budget_settlement_live": live,
            "observed_receipts_ref": (
                str(online_receipts_path)
                if online_receipts_path and Path(online_receipts_path).exists()
                else None
            ),
            "receipt_label_bundle_ref": (
                str(receipt_bundle_path)
                if receipt_bundle_path and Path(receipt_bundle_path).exists()
                else None
            ),
            "receipt_label_summary_ref": (
                str(receipt_summary_path)
                if receipt_summary_path and Path(receipt_summary_path).exists()
                else None
            ),
            "promotion_ledger_ref": promotion_ledger_path,
            "source_domain_coverage": dict(self._training_runtime_context.get("source_domain_coverage", {}) or {}),
            "receipt_label_coverage": dict(self._training_runtime_context.get("receipt_label_coverage", {}) or {}),
            "metadata": {
                "runtime_status": effective_status,
                "explicit_override": explicit_live if explicit_live is not None else None,
            },
        }
        settlement_path = self.output_dir / "budget_settlement_v1.json"
        settlement_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        settlement_sha = sha256_file(settlement_path)
        self._runtime_artifacts["budget_settlement_report"] = str(settlement_path)
        self._runtime_artifact_metadata["budget_settlement_report"] = {
            "schema_version": "budget_settlement_v1",
            "live": live,
        }
        return str(settlement_path), settlement_sha, live

    def _write_training_runtime_artifacts(
        self,
        *,
        plan_sha: str,
        plan_id: str,
        status: Optional[str] = None,
    ) -> tuple[Optional[str], Optional[str]]:
        if not (self._training_kind or self._runtime_artifacts or self._checkpoint_records or self._training_runtime_context):
            return None, None

        effective_status = status or self._runtime_status
        checkpoint_registry_sha: Optional[str] = None
        checkpoint_registry_path: Optional[str] = None
        if self._checkpoint_records:
            checkpoint_registry = create_checkpoint_registry(
                run_id=self.run_id,
                training_kind=self._training_kind or "training_job",
                checkpoints=self._checkpoint_records,
                metadata={"runtime_status": effective_status},
            )
            checkpoint_registry_path = str(self.output_dir / "checkpoint_registry.json")
            checkpoint_registry_sha = write_checkpoint_registry(
                checkpoint_registry_path,
                checkpoint_registry,
            )
        promotion_ledger_path, promotion_ledger_sha = self._write_promotion_ledger_artifact(
            effective_status=effective_status,
        )
        budget_settlement_path, budget_settlement_sha, budget_settlement_live = self._write_budget_settlement_artifact(
            effective_status=effective_status,
            promotion_ledger_path=promotion_ledger_path,
        )
        runtime_manifest = TrainingRuntimeManifest(
            schema_version="training_runtime_manifest_v1",
            run_id=self.run_id,
            training_kind=self._training_kind or "training_job",
            status=effective_status,
            seed=self.config.seed,
            plan_id=plan_id,
            plan_sha=plan_sha,
            started_at=self._runtime_started_at or self._ts_start or "",
            ended_at=self._runtime_ended_at or self._ts_end or "",
            config_path=self._training_runtime_context.get("config_path"),
            config_digest=str(self._training_runtime_context.get("config_digest", "")),
            replay_dataset_dir=self._training_runtime_context.get("replay_dataset_dir"),
            replay_manifest_digest=self._training_runtime_context.get("replay_manifest_digest"),
            replay_dataset_summary=dict(self._training_runtime_context.get("replay_dataset_summary", {}) or {}),
            objective_profile_snapshot=dict(self._training_runtime_context.get("objective_profile_snapshot", {}) or {}),
            promotion_policy_snapshot=dict(self._training_runtime_context.get("promotion_policy_snapshot", {}) or {}),
            source_domain_coverage=dict(self._training_runtime_context.get("source_domain_coverage", {}) or {}),
            receipt_label_coverage=dict(self._training_runtime_context.get("receipt_label_coverage", {}) or {}),
            artifact_paths=dict(sorted(self._runtime_artifacts.items())),
            checkpoint_registry_path=checkpoint_registry_path,
            checkpoint_registry_digest=checkpoint_registry_sha,
            promotion_evidence_path=self._runtime_artifacts.get("regal_promotion_eval"),
            promotion_evidence_digest=(
                sha256_file(self._runtime_artifacts["regal_promotion_eval"])
                if self._runtime_artifacts.get("regal_promotion_eval")
                and Path(self._runtime_artifacts["regal_promotion_eval"]).exists()
                else None
            ),
            promotion_ledger_path=promotion_ledger_path,
            promotion_ledger_digest=promotion_ledger_sha,
            budget_settlement_path=budget_settlement_path,
            budget_settlement_digest=budget_settlement_sha,
            budget_settlement_live=budget_settlement_live,
            artifact_schema_compatibility=[
                dict(row)
                for row in list(self._training_runtime_context.get("artifact_schema_compatibility", []) or [])
            ],
            failure_reason=self._failure_reason,
            metadata={
                **dict(self._training_runtime_context.get("metadata", {}) or {}),
                "budget_settlement_live": budget_settlement_live,
            },
        )
        manifest_path = self.output_dir / "training_runtime_manifest.json"
        manifest_sha = write_training_runtime_manifest(manifest_path, runtime_manifest)
        summary_md = build_training_runtime_summary_markdown(
            runtime_manifest,
            checkpoint_rows=[record.to_dict() for record in self._checkpoint_records],
        )
        (self.output_dir / "training_runtime_summary.md").write_text(summary_md, encoding="utf-8")
        self._runtime_artifacts.setdefault("training_runtime_manifest", str(manifest_path))
        self._runtime_artifacts.setdefault(
            "training_runtime_summary",
            str(self.output_dir / "training_runtime_summary.md"),
        )
        return manifest_sha, checkpoint_registry_sha

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
        self._runtime_ended_at = self._ts_end
        if self._runtime_status not in {"failed", "verification_failed"}:
            self._runtime_status = "finalizing"

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

        audit_before = self._audit_before or self._placeholder_audit_aggregate()
        audit_after = self._audit_after or audit_before

        # Build deploy gate inputs
        deploy_inputs = create_deploy_gate_inputs(
            regal_result=(
                self._regal_result
                if getattr(self._regal_result, "all_passed", None) is not None
                else None
            ),
            audit_delta_success=(
                audit_after.success_rate - audit_before.success_rate
                if audit_before.success_rate is not None and audit_after.success_rate is not None
                else None
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
            audit_before=audit_before,
            audit_after=audit_after,
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
            regal=(
                self._regal_result
                if getattr(self._regal_result, "all_passed", None) is not None
                else None
            ),
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
            audit_config_sha=audit_before.config_sha,
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
        runtime_manifest_sha, checkpoint_registry_sha = self._write_training_runtime_artifacts(
            plan_sha=plan_sha,
            plan_id=plan_id,
            status="completed" if verification_report.all_passed else "verification_failed",
        )
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
            training_runtime_manifest_sha=runtime_manifest_sha,
            checkpoint_registry_sha=checkpoint_registry_sha,
            runtime_status="completed" if verification_report.all_passed else "verification_failed",
            failure_reason=None if verification_report.all_passed else "verification_failed",
        )

        # Print summary
        print(f"\n{'='*60}")
        print("TRAINING RUN FINALIZED")
        print(f"{'='*60}")
        print(f"Run ID: {self.run_id}")
        print(f"Output: {self.output_dir}")
        print("\nArtifact SHAs:")
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
            print("\nERROR: Verification failed, exiting with error")
            sys.exit(1)

        return result

    def _placeholder_audit_aggregate(self) -> AuditAggregateV1:
        """Build a deterministic placeholder audit aggregate when no legacy audit exists."""
        returns = [float(audit.total_return or 0.0) for audit in self._trajectory_audits]
        success_like = [
            1.0
            if (audit.contact_anomaly_count or 0) == 0 and (audit.velocity_spike_count or 0) == 0
            else 0.0
            for audit in self._trajectory_audits
        ]
        energy_like = [
            float(sum(abs(value) for value in (audit.action_mean or [])))
            for audit in self._trajectory_audits
        ]
        return AuditAggregateV1(
            audit_suite_id=self.config.audit_suite_id,
            seed=self.config.audit_seed,
            num_episodes=len(self._trajectory_audits),
            success_rate=(
                sum(success_like) / float(len(success_like))
                if success_like
                else 0.0
            ),
            mean_return=(
                sum(returns) / float(len(returns))
                if returns
                else 0.0
            ),
            mean_energy_Wh=(
                sum(energy_like) / float(len(energy_like))
                if energy_like
                else 0.0
            ),
            mean_mpl_proxy=(
                sum(max(0.0, value) for value in returns) / float(len(returns))
                if returns
                else 0.0
            ),
            per_task={},
            episodes_sha=sha256_json([audit.sha256() for audit in self._trajectory_audits]),
            config_sha=sha256_json(
                {
                    "audit_suite_id": self.config.audit_suite_id,
                    "audit_seed": self.config.audit_seed,
                    "trajectory_audit_count": len(self._trajectory_audits),
                }
            ),
            regal_context_sha=self._regal_context_sha,
        )


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
    try:
        training_fn(runner)
    except Exception as exc:
        runner.set_runtime_status("failed", failure_reason=f"{exc.__class__.__name__}: {exc}")
        runner._write_training_runtime_artifacts(  # noqa: SLF001 - shared internal helper
            plan_sha=plan_sha,
            plan_id=plan_id,
            status="failed",
        )
        raise

    # Finalize and verify
    return runner.finalize(plan_sha=plan_sha, plan_id=plan_id)


__all__ = [
    "TrainingRunConfig",
    "TrainingRunResult",
    "RegalTrainingRunner",
    "run_training_with_regality",
]
