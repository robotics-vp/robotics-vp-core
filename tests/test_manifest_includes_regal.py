"""Tests for run manifest regal and knob provenance fields.

Verifies that RunManifestV1 includes all required SHA references for
regal gates, knob calibration, and trajectory audit.
"""
import pytest
from typing import Dict, Any, Optional

from src.contracts.schemas import (
    RunManifestV1,
    RegalGatesV1,
    RegalReportV1,
    LedgerRegalV1,
    KnobPolicyV1,
    RegimeFeaturesV1,
    TrajectoryAuditV1,
)
from src.valuation.run_manifest import create_run_manifest, write_manifest
from src.utils.config_digest import sha256_json


class TestManifestRegalFields:
    """Tests for regal-related fields in RunManifestV1."""

    def test_manifest_has_regal_config_sha_field(self):
        """Test RunManifestV1 has regal_config_sha field."""
        manifest = create_run_manifest(
            run_id="test_run",
            plan_sha="plan_sha",
            audit_suite_id="suite_id",
            audit_seed=42,
            audit_config_sha="audit_sha",
            datapack_ids=[],
            seeds={},
            determinism_config={},
        )

        assert hasattr(manifest, "regal_config_sha")
        assert manifest.regal_config_sha is None  # Default

    def test_manifest_has_regal_report_sha_field(self):
        """Test RunManifestV1 has regal_report_sha field."""
        manifest = create_run_manifest(
            run_id="test_run",
            plan_sha="plan_sha",
            audit_suite_id="suite_id",
            audit_seed=42,
            audit_config_sha="audit_sha",
            datapack_ids=[],
            seeds={},
            determinism_config={},
        )

        assert hasattr(manifest, "regal_report_sha")
        assert manifest.regal_report_sha is None

    def test_manifest_has_regal_inputs_sha_field(self):
        """Test RunManifestV1 has regal_inputs_sha field."""
        manifest = create_run_manifest(
            run_id="test_run",
            plan_sha="plan_sha",
            audit_suite_id="suite_id",
            audit_seed=42,
            audit_config_sha="audit_sha",
            datapack_ids=[],
            seeds={},
            determinism_config={},
        )

        assert hasattr(manifest, "regal_inputs_sha")
        assert manifest.regal_inputs_sha is None

    def test_manifest_regal_fields_can_be_set(self):
        """Test regal fields can be populated."""
        manifest = create_run_manifest(
            run_id="test_run",
            plan_sha="plan_sha",
            audit_suite_id="suite_id",
            audit_seed=42,
            audit_config_sha="audit_sha",
            datapack_ids=["dp1"],
            seeds={"main": 42},
            determinism_config={"enabled": True},
        )

        # Set regal fields
        config = RegalGatesV1(
            enabled_regal_ids=["spec_guardian"],
            patience=3,
        )
        manifest.regal_config_sha = config.sha256()
        manifest.regal_report_sha = "dummy_report_sha"
        manifest.regal_inputs_sha = "dummy_inputs_sha"

        assert manifest.regal_config_sha is not None
        assert len(manifest.regal_config_sha) == 64
        assert manifest.regal_report_sha == "dummy_report_sha"
        assert manifest.regal_inputs_sha == "dummy_inputs_sha"


class TestManifestKnobFields:
    """Tests for D4 knob calibration fields in RunManifestV1."""

    def test_manifest_has_knob_model_sha_field(self):
        """Test RunManifestV1 has knob_model_sha field."""
        manifest = create_run_manifest(
            run_id="test_run",
            plan_sha="plan_sha",
            audit_suite_id="suite_id",
            audit_seed=42,
            audit_config_sha="audit_sha",
            datapack_ids=[],
            seeds={},
            determinism_config={},
        )

        assert hasattr(manifest, "knob_model_sha")
        assert manifest.knob_model_sha is None

    def test_manifest_has_knob_policy_sha_field(self):
        """Test RunManifestV1 has knob_policy_sha field."""
        manifest = create_run_manifest(
            run_id="test_run",
            plan_sha="plan_sha",
            audit_suite_id="suite_id",
            audit_seed=42,
            audit_config_sha="audit_sha",
            datapack_ids=[],
            seeds={},
            determinism_config={},
        )

        assert hasattr(manifest, "knob_policy_sha")
        assert manifest.knob_policy_sha is None

    def test_manifest_has_knob_policy_used_field(self):
        """Test RunManifestV1 has knob_policy_used field."""
        manifest = create_run_manifest(
            run_id="test_run",
            plan_sha="plan_sha",
            audit_suite_id="suite_id",
            audit_seed=42,
            audit_config_sha="audit_sha",
            datapack_ids=[],
            seeds={},
            determinism_config={},
        )

        assert hasattr(manifest, "knob_policy_used")
        assert manifest.knob_policy_used is None

    def test_manifest_knob_fields_can_be_set(self):
        """Test knob fields can be populated."""
        manifest = create_run_manifest(
            run_id="test_run",
            plan_sha="plan_sha",
            audit_suite_id="suite_id",
            audit_seed=42,
            audit_config_sha="audit_sha",
            datapack_ids=["dp1"],
            seeds={"main": 42},
            determinism_config={"enabled": True},
        )

        # Set knob fields
        policy = KnobPolicyV1(
            policy_source="learned",
            regime_features_sha="features_sha",
            model_sha="model_v1",
            gain_multiplier_override=1.5,
        )
        manifest.knob_model_sha = policy.model_sha
        manifest.knob_policy_sha = policy.sha256()
        manifest.knob_policy_used = policy.policy_source

        assert manifest.knob_model_sha == "model_v1"
        assert manifest.knob_policy_sha is not None
        assert len(manifest.knob_policy_sha) == 64
        assert manifest.knob_policy_used == "learned"


class TestManifestTrajectoryAuditField:
    """Tests for trajectory audit field in RunManifestV1."""

    def test_manifest_has_trajectory_audit_sha_field(self):
        """Test RunManifestV1 has trajectory_audit_sha field."""
        manifest = create_run_manifest(
            run_id="test_run",
            plan_sha="plan_sha",
            audit_suite_id="suite_id",
            audit_seed=42,
            audit_config_sha="audit_sha",
            datapack_ids=[],
            seeds={},
            determinism_config={},
        )

        assert hasattr(manifest, "trajectory_audit_sha")
        assert manifest.trajectory_audit_sha is None

    def test_manifest_trajectory_audit_can_be_set(self):
        """Test trajectory audit SHA can be populated."""
        manifest = create_run_manifest(
            run_id="test_run",
            plan_sha="plan_sha",
            audit_suite_id="suite_id",
            audit_seed=42,
            audit_config_sha="audit_sha",
            datapack_ids=["dp1"],
            seeds={"main": 42},
            determinism_config={"enabled": True},
        )

        # Create trajectory audit and set SHA
        audit = TrajectoryAuditV1(
            episode_id="ep_001",
            num_steps=100,
            total_return=1.5,
        )
        manifest.trajectory_audit_sha = audit.sha256()

        assert manifest.trajectory_audit_sha is not None
        assert len(manifest.trajectory_audit_sha) == 64


class TestTrajectoryAuditSchema:
    """Tests for TrajectoryAuditV1 schema."""

    def test_trajectory_audit_creation(self):
        """Test TrajectoryAuditV1 can be created."""
        audit = TrajectoryAuditV1(
            episode_id="ep_001",
            num_steps=100,
            total_return=1.5,
        )

        assert audit.episode_id == "ep_001"
        assert audit.num_steps == 100
        assert audit.total_return == 1.5

    def test_trajectory_audit_with_all_fields(self):
        """Test TrajectoryAuditV1 with all optional fields."""
        audit = TrajectoryAuditV1(
            episode_id="ep_001",
            num_steps=100,
            action_mean=[0.1, 0.2, 0.3],
            action_std=[0.01, 0.02, 0.03],
            state_bounds={"x": [-1.0, 1.0], "y": [-1.0, 1.0]},
            total_return=1.5,
            reward_components={"main": 1.0, "penalty": -0.5},
            events=["grasp", "place"],
            event_counts={"grasp": 5, "place": 3},
            penetration_max=0.005,
            velocity_spike_count=2,
            contact_anomaly_count=1,
            scene_tracks_sha="scene_sha",
            bev_summary_sha="bev_sha",
        )

        assert audit.action_mean == [0.1, 0.2, 0.3]
        assert audit.state_bounds["x"] == [-1.0, 1.0]
        assert audit.reward_components["main"] == 1.0
        assert audit.events == ["grasp", "place"]
        assert audit.penetration_max == 0.005
        assert audit.velocity_spike_count == 2

    def test_trajectory_audit_sha_deterministic(self):
        """Test TrajectoryAuditV1 SHA is deterministic."""
        audit1 = TrajectoryAuditV1(
            episode_id="ep_001",
            num_steps=100,
            total_return=1.5,
        )
        audit2 = TrajectoryAuditV1(
            episode_id="ep_001",
            num_steps=100,
            total_return=1.5,
        )

        assert audit1.sha256() == audit2.sha256()

    def test_trajectory_audit_sha_changes(self):
        """Test SHA changes with different values."""
        audit1 = TrajectoryAuditV1(
            episode_id="ep_001",
            num_steps=100,
            total_return=1.5,
        )
        audit2 = TrajectoryAuditV1(
            episode_id="ep_001",
            num_steps=100,
            total_return=2.0,  # Different return
        )

        assert audit1.sha256() != audit2.sha256()


class TestLedgerPlanPolicyKnobField:
    """Tests for knob_policy field in LedgerPlanPolicyV1."""

    def test_ledger_plan_policy_has_knob_policy_field(self):
        """Test LedgerPlanPolicyV1 has knob_policy field."""
        from src.contracts.schemas import LedgerPlanPolicyV1

        ledger_policy = LedgerPlanPolicyV1(
            policy_config_sha="config_sha",
            gain_schedule_sha="schedule_sha",
        )

        assert hasattr(ledger_policy, "knob_policy")
        assert ledger_policy.knob_policy is None

    def test_ledger_plan_policy_with_knob_policy(self):
        """Test LedgerPlanPolicyV1 can store KnobPolicyV1."""
        from src.contracts.schemas import LedgerPlanPolicyV1

        knob_policy = KnobPolicyV1(
            policy_source="heuristic_fallback",
            regime_features_sha="features_sha",
            gain_multiplier_override=1.2,
        )

        ledger_policy = LedgerPlanPolicyV1(
            policy_config_sha="config_sha",
            gain_schedule_sha="schedule_sha",
            knob_policy=knob_policy,
        )

        assert ledger_policy.knob_policy is not None
        assert ledger_policy.knob_policy.policy_source == "heuristic_fallback"
        assert ledger_policy.knob_policy.gain_multiplier_override == 1.2


class TestValueLedgerRecordKnobField:
    """Tests for knob_policy field in ValueLedgerRecordV1."""

    def test_value_ledger_record_has_knob_policy_field(self):
        """Test ValueLedgerRecordV1 can include knob_policy via ledger_policy."""
        from src.contracts.schemas import (
            ValueLedgerRecordV1,
            LedgerPlanPolicyV1,
            LedgerWindowV1,
            LedgerAuditV1,
            LedgerDeltasV1,
            LedgerExposureV1,
            LedgerPolicyV1,
        )

        knob_policy = KnobPolicyV1(
            policy_source="learned",
            regime_features_sha="features_sha",
            model_sha="model_v1",
        )

        ledger_plan_policy = LedgerPlanPolicyV1(
            policy_config_sha="config_sha",
            gain_schedule_sha="schedule_sha",
            knob_policy=knob_policy,
        )

        # Create minimal record
        record = ValueLedgerRecordV1(
            record_id="rec_001",
            run_id="run_001",
            plan_id="plan_001",
            plan_sha="plan_sha",
            audit=LedgerAuditV1(
                audit_suite_id="test_suite",
                audit_seed=42,
                audit_config_sha="config_sha",
                audit_results_before_sha="before_sha",
                audit_results_after_sha="after_sha",
            ),
            deltas=LedgerDeltasV1(),
            window=LedgerWindowV1(
                step_start=0,
                step_end=1000,
                ts_start="2024-01-01T00:00:00",
                ts_end="2024-01-01T01:00:00",
            ),
            exposure=LedgerExposureV1(
                datapack_ids=["dp1"],
                slice_ids=["s1"],
                exposure_manifest_sha="manifest_sha",
            ),
            policy=LedgerPolicyV1(
                policy_before="ckpt_a",
                policy_after="ckpt_b",
            ),
            plan_policy=ledger_plan_policy,
        )

        assert record.plan_policy is not None
        assert record.plan_policy.knob_policy is not None
        assert record.plan_policy.knob_policy.policy_source == "learned"
