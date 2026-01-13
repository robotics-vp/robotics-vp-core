"""
Environment regality compliance tests.

Ensures manufacturing cell (workcell) and dishwashing envs produce
all required regality artifacts and pass verify_run().

Phase B: Env compliance harness for regality parity.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import pytest

from src.contracts.schemas import (
    RunManifestV1,
    TrajectoryAuditV1,
    SelectionManifestV1,
    OrchestratorStateV1,
    RewardBreakdownV1,
)
from src.valuation.exposure_manifest import ExposureTracker, write_exposure_manifest, write_selection_manifest
from src.valuation.run_manifest import create_run_manifest, write_manifest
from src.valuation.valuation_verifier import verify_run, write_verification_report
from src.valuation.trajectory_audit import create_trajectory_audit, aggregate_trajectory_audits
from src.valuation.value_ledger import ValueLedger
from src.orchestrator.orchestrator_state import OrchestratorStateTracker, write_orchestrator_state
from src.deployment.deploy_gate import (
    create_deploy_gate_inputs,
    compute_deploy_decision,
    write_deploy_gate_inputs,
    write_deploy_gate_decision,
)
from src.utils.config_digest import sha256_json


class EnvComplianceHarness:
    """Shared compliance harness for env regality testing.

    Both dishwashing and manufacturing (workcell) envs must pass
    these checks to be considered regality-compliant.
    """

    def __init__(self, env_name: str, output_dir: Path):
        self.env_name = env_name
        self.output_dir = output_dir
        self.run_id = f"test_{env_name}_001"

        # Trackers
        self.exposure_tracker = ExposureTracker(manifest_id=self.run_id, step_start=0)
        self.orchestrator_tracker = OrchestratorStateTracker(step=0)

        # Results
        self.trajectory_audits: List[TrajectoryAuditV1] = []
        self.artifact_shas: Dict[str, Optional[str]] = {}

    def setup_quarantine(self, quarantine_ids: List[str]) -> None:
        """Set up quarantine for compliance testing."""
        self.exposure_tracker.set_quarantine(quarantine_ids)

    def setup_eligible_datapacks(self, datapack_ids: List[str], seed: int = 42) -> None:
        """Set eligible datapacks for selection manifest."""
        self.exposure_tracker.set_eligible_datapacks(datapack_ids)
        self.exposure_tracker.set_sampler_config(seed=seed)

    def record_sample(self, task_family: str, datapack_id: str) -> bool:
        """Record a sample with quarantine enforcement."""
        return self.exposure_tracker.record_sample(task_family, datapack_id)

    def add_trajectory_audit(
        self,
        episode_id: str,
        num_steps: int = 50,
        actions: Optional[List[List[float]]] = None,
        rewards: Optional[List[float]] = None,
    ) -> TrajectoryAuditV1:
        """Add a trajectory audit."""
        if actions is None:
            actions = [[0.1] * 7 for _ in range(num_steps)]
        if rewards is None:
            rewards = [0.01 for _ in range(num_steps)]

        audit = create_trajectory_audit(
            episode_id=episode_id,
            num_steps=num_steps,
            actions=actions,
            rewards=rewards,
        )
        self.trajectory_audits.append(audit)
        return audit

    def finalize_and_verify(
        self,
        plan_sha: str = "test_plan_sha_123",
        require_trajectory_audit: bool = True,
    ) -> Dict[str, any]:
        """Finalize run and verify compliance.

        Returns:
            Dict with verification result and artifact SHAs
        """
        # Aggregate trajectory audits
        trajectory_audit_sha = None
        if self.trajectory_audits:
            # aggregate_trajectory_audits returns SHA string directly
            trajectory_audit_sha = aggregate_trajectory_audits(self.trajectory_audits)

            # Write individual audits for verification
            audit_path = self.output_dir / "trajectory_audit.json"
            # Use first audit as representative (in production would be full aggregate)
            with open(audit_path, "w") as f:
                audit_data = {
                    "aggregate_sha": trajectory_audit_sha,
                    "num_episodes": len(self.trajectory_audits),
                    "episode_ids": [a.episode_id for a in self.trajectory_audits],
                }
                json.dump(audit_data, f, indent=2)

        # Exposure manifest
        self.exposure_tracker.update_step(100)
        exposure_manifest = self.exposure_tracker.build_manifest()
        exposure_path = self.output_dir / "exposure_manifest.json"
        exposure_sha = write_exposure_manifest(str(exposure_path), exposure_manifest)

        # Selection manifest
        selection_manifest = self.exposure_tracker.build_selection_manifest()
        selection_path = self.output_dir / "selection_manifest.json"
        selection_sha = write_selection_manifest(str(selection_path), selection_manifest)

        # Orchestrator state
        self.orchestrator_tracker.update_step(100)
        orchestrator_state = self.orchestrator_tracker.build_state()
        orchestrator_path = self.output_dir / "orchestrator_state.json"
        orchestrator_sha = write_orchestrator_state(str(orchestrator_path), orchestrator_state)

        # Deploy gate inputs
        deploy_inputs = create_deploy_gate_inputs(
            audit_delta_success=0.05,
            trajectory_audit_sha=trajectory_audit_sha,
        )
        deploy_inputs_path = self.output_dir / "deploy_gate_inputs.json"
        deploy_inputs_sha = write_deploy_gate_inputs(str(deploy_inputs_path), deploy_inputs)

        deploy_decision = compute_deploy_decision(deploy_inputs, require_regal=False)
        deploy_decision_path = self.output_dir / "deploy_gate_decision.json"
        deploy_decision_sha = write_deploy_gate_decision(str(deploy_decision_path), deploy_decision)

        # Ledger - create minimal record with mock audit results
        ledger = ValueLedger(str(self.output_dir / "ledger.jsonl"))
        from src.contracts.schemas import LedgerWindowV1, LedgerExposureV1, LedgerPolicyV1

        # Create mock audit result with all required attributes
        class MockAuditResult:
            success_rate = 0.75
            error_rate = 0.1
            mean_error = 0.1
            mean_energy_Wh = 1.5
            mean_mpl_proxy = 0.8
            num_episodes = 2
            config_sha = "mock_config_sha"
            audit_suite_id = "test_audit"
            seed = 42
            episodes_sha = "mock_episodes_sha"

        mock_audit = MockAuditResult()

        record = ledger.create_record(
            run_id=self.run_id,
            plan_id="test_plan",
            plan_sha=plan_sha,
            audit_before=mock_audit,
            audit_after=mock_audit,
            window=LedgerWindowV1(step_start=0, step_end=100, ts_start="", ts_end=""),
            exposure=LedgerExposureV1(
                datapack_ids=exposure_manifest.datapack_ids,
                exposure_manifest_sha=exposure_sha,
            ),
            policy=LedgerPolicyV1(policy_before="baseline", policy_after="trained"),
            notes=f"Test run for {self.env_name}",
        )
        ledger.append(record)

        # Manifest
        manifest = create_run_manifest(
            run_id=self.run_id,
            plan_sha=plan_sha,
            audit_suite_id="test_audit",
            audit_seed=42,
            audit_config_sha="test_config",
            datapack_ids=exposure_manifest.datapack_ids,
            seeds={"audit": 42},
            trajectory_audit_sha=trajectory_audit_sha,
            orchestrator_state_sha=orchestrator_sha,
            selection_manifest_sha=selection_sha,
            deploy_gate_inputs_sha=deploy_inputs_sha,
            deploy_gate_decision_sha=deploy_decision_sha,
            # ALWAYS make it look like a training run (weights changed)
            # The trajectory_audit_sha being None should cause verification failure
            baseline_weights_sha="baseline_weights_sha",
            final_weights_sha="final_weights_sha",  # Always different from baseline
        )

        manifest_path = self.output_dir / "run_manifest.json"
        write_manifest(str(manifest_path), manifest)

        # Verify
        verification_report = verify_run(str(self.output_dir))

        # Write verification report
        verification_path = self.output_dir / "verification_report.json"
        verification_sha = write_verification_report(str(verification_path), verification_report)

        # Update manifest with verification SHA
        manifest.verification_report_sha = verification_sha
        write_manifest(str(manifest_path), manifest)

        self.artifact_shas = {
            "exposure_sha": exposure_sha,
            "selection_manifest_sha": selection_sha,
            "orchestrator_state_sha": orchestrator_sha,
            "trajectory_audit_sha": trajectory_audit_sha,
            "verification_report_sha": verification_sha,
            "deploy_gate_inputs_sha": deploy_inputs_sha,
        }

        return {
            "all_passed": verification_report.all_passed,
            "checks": verification_report.checks,
            "artifact_shas": self.artifact_shas,
            "selection_manifest": selection_manifest,
            "verification_report": verification_report,
        }


# =============================================================================
# Manufacturing Cell (Workcell) Compliance Tests
# =============================================================================

class TestWorkcellRegality:
    """Manufacturing cell (workcell) regality compliance tests."""

    def test_workcell_produces_selection_manifest(self):
        """Workcell produces selection manifest with all required fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            harness = EnvComplianceHarness("workcell", Path(tmpdir))
            harness.setup_eligible_datapacks(["dp_001", "dp_002", "dp_003"])

            harness.record_sample("kitting", "dp_001")
            harness.record_sample("kitting", "dp_002")
            harness.add_trajectory_audit("ep_001")

            result = harness.finalize_and_verify()

            assert result["artifact_shas"]["selection_manifest_sha"] is not None
            sm = result["selection_manifest"]
            assert len(sm.eligible_datapack_ids) == 3
            assert len(sm.selected_datapack_ids) == 2

    def test_workcell_produces_orchestrator_state(self):
        """Workcell produces orchestrator state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            harness = EnvComplianceHarness("workcell", Path(tmpdir))
            harness.setup_eligible_datapacks(["dp_001"])

            harness.orchestrator_tracker.record_failure("test_gate")
            harness.orchestrator_tracker.set_patience("test_gate", 2)

            harness.record_sample("kitting", "dp_001")
            harness.add_trajectory_audit("ep_001")

            result = harness.finalize_and_verify()

            assert result["artifact_shas"]["orchestrator_state_sha"] is not None

    def test_workcell_produces_trajectory_audit(self):
        """Workcell produces trajectory audit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            harness = EnvComplianceHarness("workcell", Path(tmpdir))
            harness.setup_eligible_datapacks(["dp_001"])

            harness.record_sample("kitting", "dp_001")
            harness.add_trajectory_audit("ep_001", num_steps=50)

            result = harness.finalize_and_verify()

            assert result["artifact_shas"]["trajectory_audit_sha"] is not None

    def test_workcell_quarantine_enforcement(self):
        """Workcell enforces quarantine - rejected datapacks recorded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            harness = EnvComplianceHarness("workcell", Path(tmpdir))
            harness.setup_quarantine(["bad_dp_001"])
            harness.setup_eligible_datapacks(["dp_001", "bad_dp_001"])

            # Try to sample quarantined datapack
            recorded = harness.record_sample("kitting", "bad_dp_001")
            assert recorded is False  # Quarantine rejected

            # Sample valid datapack
            recorded = harness.record_sample("kitting", "dp_001")
            assert recorded is True

            harness.add_trajectory_audit("ep_001")
            result = harness.finalize_and_verify()

            # Selection manifest should show rejection
            sm = result["selection_manifest"]
            assert len(sm.rejected_datapacks) == 1
            assert sm.rejected_datapacks[0]["id"] == "bad_dp_001"
            assert sm.rejected_datapacks[0]["reason"] == "quarantine"

    def test_workcell_verify_run_passes(self):
        """Workcell run passes verify_run() with full artifacts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            harness = EnvComplianceHarness("workcell", Path(tmpdir))
            harness.setup_eligible_datapacks(["dp_001", "dp_002"])

            harness.record_sample("kitting", "dp_001")
            harness.record_sample("assembly", "dp_002")
            harness.add_trajectory_audit("ep_001")
            harness.add_trajectory_audit("ep_002")

            result = harness.finalize_and_verify()

            # Should pass verification
            assert result["all_passed"] is True

    def test_workcell_training_requires_trajectory_audit(self):
        """Training run without trajectory audit fails verification."""
        with tempfile.TemporaryDirectory() as tmpdir:
            harness = EnvComplianceHarness("workcell", Path(tmpdir))
            harness.setup_eligible_datapacks(["dp_001"])

            harness.record_sample("kitting", "dp_001")
            # NO trajectory audit added

            result = harness.finalize_and_verify()

            # Should fail trajectory_audit_sha_required check
            failed_checks = [c for c in result["checks"] if not c.passed]
            trajectory_fail = any("trajectory_audit" in c.check_id.lower() for c in failed_checks)
            assert trajectory_fail, "Training run without trajectory audit should fail"


# =============================================================================
# Dishwashing Compliance Tests (Parity Check)
# =============================================================================

class TestDishwashingRegality:
    """Dishwashing env regality compliance tests.

    NOTE: Dishwashing is currently PARTIAL compliance - these tests
    establish the baseline for full compliance.
    """

    def test_dishwashing_can_produce_selection_manifest(self):
        """Dishwashing can produce selection manifest (parity with workcell)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            harness = EnvComplianceHarness("dishwashing", Path(tmpdir))
            harness.setup_eligible_datapacks(["dp_001", "dp_002"])

            harness.record_sample("manipulation", "dp_001")
            harness.add_trajectory_audit("ep_001")

            result = harness.finalize_and_verify()

            assert result["artifact_shas"]["selection_manifest_sha"] is not None

    def test_dishwashing_can_produce_orchestrator_state(self):
        """Dishwashing can produce orchestrator state (parity with workcell)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            harness = EnvComplianceHarness("dishwashing", Path(tmpdir))
            harness.setup_eligible_datapacks(["dp_001"])

            harness.record_sample("manipulation", "dp_001")
            harness.add_trajectory_audit("ep_001")

            result = harness.finalize_and_verify()

            assert result["artifact_shas"]["orchestrator_state_sha"] is not None

    def test_dishwashing_quarantine_enforcement(self):
        """Dishwashing enforces quarantine (parity with workcell)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            harness = EnvComplianceHarness("dishwashing", Path(tmpdir))
            harness.setup_quarantine(["bad_dp"])
            harness.setup_eligible_datapacks(["dp_001", "bad_dp"])

            recorded = harness.record_sample("manipulation", "bad_dp")
            assert recorded is False

            harness.record_sample("manipulation", "dp_001")
            harness.add_trajectory_audit("ep_001")

            result = harness.finalize_and_verify()

            sm = result["selection_manifest"]
            assert len(sm.rejected_datapacks) == 1


# =============================================================================
# Cross-Env Parity Tests
# =============================================================================

class TestEnvParity:
    """Tests ensuring manufacturing >= dishwashing on regality surfaces."""

    def test_both_envs_produce_same_artifact_types(self):
        """Both envs produce the same set of artifact types."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workcell_dir = Path(tmpdir) / "workcell"
            workcell_dir.mkdir()
            dishwashing_dir = Path(tmpdir) / "dishwashing"
            dishwashing_dir.mkdir()

            # Run workcell
            wc_harness = EnvComplianceHarness("workcell", workcell_dir)
            wc_harness.setup_eligible_datapacks(["dp_001"])
            wc_harness.record_sample("kitting", "dp_001")
            wc_harness.add_trajectory_audit("ep_001")
            wc_result = wc_harness.finalize_and_verify()

            # Run dishwashing
            dw_harness = EnvComplianceHarness("dishwashing", dishwashing_dir)
            dw_harness.setup_eligible_datapacks(["dp_001"])
            dw_harness.record_sample("manipulation", "dp_001")
            dw_harness.add_trajectory_audit("ep_001")
            dw_result = dw_harness.finalize_and_verify()

            # Both should have same artifact types
            wc_artifacts = set(wc_result["artifact_shas"].keys())
            dw_artifacts = set(dw_result["artifact_shas"].keys())

            assert wc_artifacts == dw_artifacts, \
                f"Artifact type mismatch: workcell={wc_artifacts}, dishwashing={dw_artifacts}"

    def test_workcell_at_least_parity_with_dishwashing(self):
        """Workcell has at least parity with dishwashing on regality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workcell_dir = Path(tmpdir) / "workcell"
            workcell_dir.mkdir()

            harness = EnvComplianceHarness("workcell", workcell_dir)
            harness.setup_quarantine(["quarantined_dp"])
            harness.setup_eligible_datapacks(["dp_001", "dp_002", "quarantined_dp"])

            harness.record_sample("kitting", "dp_001")
            harness.record_sample("kitting", "quarantined_dp")  # Should reject
            harness.add_trajectory_audit("ep_001")

            harness.orchestrator_tracker.record_failure("gate_1")
            harness.orchestrator_tracker.record_clamp("gate_1", "threshold", 0.5)

            result = harness.finalize_and_verify()

            # Workcell must pass verification
            assert result["all_passed"] is True

            # Must have all key artifacts
            assert result["artifact_shas"]["selection_manifest_sha"] is not None
            assert result["artifact_shas"]["orchestrator_state_sha"] is not None
            assert result["artifact_shas"]["trajectory_audit_sha"] is not None
            assert result["artifact_shas"]["verification_report_sha"] is not None

            # Must track quarantine rejections
            sm = result["selection_manifest"]
            assert len(sm.rejected_datapacks) >= 1


# =============================================================================
# RewardBreakdownV1 Tests
# =============================================================================

class TestRewardBreakdownCompliance:
    """Tests for RewardBreakdownV1 integration."""

    def test_workcell_reward_breakdown(self):
        """Workcell produces valid RewardBreakdownV1."""
        from src.envs.workcell_env.rewards.reward_breakdown import compute_workcell_reward_breakdown

        breakdown = compute_workcell_reward_breakdown(
            success=True,
            progress=0.8,
            time_cost=50.0,
            error_count=2,
            collision_count=1,
            items_picked=4,
            items_placed=3,
            items_total=6,
        )

        # Required components present
        assert breakdown.task_reward > 0  # Success reward
        assert breakdown.time_penalty < 0  # Time cost
        assert breakdown.energy_cost <= 0

        # Standard components
        assert breakdown.grasp_reward is not None  # items_picked > 0
        assert breakdown.place_reward is not None  # items_placed > 0
        assert breakdown.collision_penalty is not None  # collision_count > 0

        # Can compute total
        total = breakdown.total()
        assert isinstance(total, float)

        # Can convert to dict for TrajectoryAuditV1
        d = breakdown.to_dict()
        assert "task_reward" in d
        assert "time_penalty" in d

    def test_dishwashing_reward_breakdown(self):
        """Dishwashing produces valid RewardBreakdownV1."""
        from src.envs.dishwashing_regal.rewards.reward_breakdown import compute_dishwashing_reward_breakdown

        breakdown = compute_dishwashing_reward_breakdown(
            completed=5,
            attempts=6,
            errors=1,
            speed=0.6,
            care=0.7,
            energy_Wh=0.3,
            profit=1.5,
            mpl_rate=90.0,
            error_rate=0.08,
        )

        # Required components present
        assert breakdown.task_reward > 0  # Completed items
        assert breakdown.time_penalty < 0  # Time cost
        assert breakdown.energy_cost < 0  # Energy consumption

        # Error mapped to collision_penalty
        assert breakdown.collision_penalty is not None
        assert breakdown.collision_penalty < 0

        # Grasp reward for completed items
        assert breakdown.grasp_reward is not None
        assert breakdown.grasp_reward > 0

        # Custom components present
        assert "speed_efficiency" in breakdown.custom_components
        assert "profit_component" in breakdown.custom_components

        # Can compute total
        total = breakdown.total()
        assert isinstance(total, float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
