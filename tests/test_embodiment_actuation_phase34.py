"""Tests for Phase 3.4 morphology, neural seams, and training rows."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch

from src.runtime.action_adapter_v2 import ActionAdapterV2
from src.runtime.observation_adapter_v2 import ObservationAdapterV2
from src.world_model.embodiment_actuation import (
    ActionProposalSeam,
    DriftCalibrationSeam,
    InverseRetargetingSeam,
    LocalContactDynamicsSeam,
    build_g1_morphology_profile,
    build_phase34_training_manifest,
    build_phase34_training_rows_from_state,
    compile_embodiment_actuation_with_receipts,
    load_phase34_training_rows_jsonl,
    scan_unitree_g1_public_evidence,
    smoke_forward_all_seams,
    unitree_g1_contract,
    write_phase34_training_rows_jsonl,
)


def _state():
    profile = build_g1_morphology_profile("g1_29dof")
    entry = profile.to_registry_entry()
    return compile_embodiment_actuation_with_receipts(
        episode_id="phase34_test",
        embodiment_registry_entry=entry,
        action_adapter=ActionAdapterV2(
            schema_id=entry.action_schema_id,
            channel_order=profile.joint_names(),
            control_hz=50.0,
            translator_ref=entry.translator_refs["retarget"],
            embodiment_id=entry.embodiment_id,
        ),
        observation_adapter=ObservationAdapterV2(
            schema_id=entry.observation_schema_id,
            proprio_fields=[f"q_{joint}" for joint in profile.joint_names()],
            sensor_refs=["proprio://unitree_g1"],
            sample_hz=50.0,
            embodiment_id=entry.embodiment_id,
        ),
        provider_contracts=[unitree_g1_contract()],
        joint_state={
            "joint_names": profile.joint_names(),
            "positions": [0.0 for _ in profile.joint_names()],
        },
        source_refs={"morphology_profile_id": profile.profile_id},
    )


def test_g1_morphology_profile_captures_29dof_shape_without_hardware_claim() -> None:
    profile = build_g1_morphology_profile("g1_29dof")

    assert profile.joint_count == 29
    assert profile.action_dimension == 29
    assert profile.group_counts()["legs"] == 12
    assert profile.group_counts()["waist"] == 3
    assert profile.group_counts()["arms"] == 14
    assert "actuator_latency_profile" in profile.unresolved_evidence
    assert profile.to_registry_entry().metadata["hardware_calibrated"] is False


def test_unitree_public_evidence_scan_extracts_config_shape(tmp_path: Path) -> None:
    root = tmp_path / "unitree_rl_gym"
    cfg_dir = root / "legged_gym" / "envs" / "g1"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "g1_config.py").write_text(
        "num_observations = 47\nnum_privileged_obs = 50\nnum_actions = 12\n"
        "friction_range = [0.1, 1.25]\nadded_mass_range = [-1., 3.]\n"
    )
    asset_dir = root / "resources" / "robots" / "g1_description"
    asset_dir.mkdir(parents=True)
    (asset_dir / "g1_29dof.urdf").write_text("<robot name='g1'/>")

    profile, receipts = scan_unitree_g1_public_evidence([root])

    assert profile.observation_dimension == 47
    assert profile.privileged_observation_dimension == 50
    assert profile.variant == "g1_29dof"
    statuses = {receipt.evidence_kind: receipt.status for receipt in receipts}
    assert statuses["locomotion_config"] == "observed"
    assert statuses["morphology_asset_visibility"] == "observed"
    assert statuses["remaining_calibration_blockers"] == "external_blocked"


def test_phase34_neural_seams_forward_cpu_shapes_are_finite() -> None:
    features16 = torch.zeros(2, 16)
    features20 = torch.zeros(2, 20)
    features32 = torch.zeros(2, 32)

    outputs = [
        LocalContactDynamicsSeam()(features16),
        InverseRetargetingSeam(target_action_dim=29)(features32),
        ActionProposalSeam(action_dim=29, chunk_len=3)(features32),
        DriftCalibrationSeam()(features20),
    ]

    assert outputs[0]["transition_risk"].shape == (2,)
    assert outputs[1]["target_action"].shape == (2, 29)
    assert outputs[2]["action_chunk"].shape == (2, 3, 29)
    assert outputs[3]["calibration_priority"].shape == (2,)
    for output in outputs:
        for tensor in output.values():
            assert torch.isfinite(tensor).all()


def test_phase34_training_rows_and_manifest_stay_non_promotional(tmp_path: Path) -> None:
    result = _state()
    rows = build_phase34_training_rows_from_state(result.state, result.receipts)
    manifest = build_phase34_training_manifest(rows, source_refs={"state_id": result.state.state_id})
    path = write_phase34_training_rows_jsonl(rows, tmp_path / "rows.jsonl")
    loaded = load_phase34_training_rows_jsonl(path)

    assert {row.seam_id for row in rows} == {
        "local_contact_dynamics",
        "inverse_retargeting",
        "action_proposal",
        "drift_calibration",
    }
    assert len(loaded) == len(rows)
    assert manifest.promotion_eligible is False
    assert "no_gpu_training_run" in manifest.blocker_reasons
    assert "no_benchmark_promotion_evidence" in manifest.blocker_reasons


def test_phase34_smoke_forward_all_seams_from_compiled_state() -> None:
    result = _state()
    summary = smoke_forward_all_seams(result.state)

    assert set(summary) == {
        "local_contact_dynamics",
        "inverse_retargeting",
        "action_proposal",
        "drift_calibration",
    }
    assert all(item["finite"] for item in summary.values())
    assert summary["action_proposal"]["output_shapes"]["action_chunk"][-1] == 29


def test_phase34_smoke_script_writes_rows_and_summary(tmp_path: Path) -> None:
    out_dir = tmp_path / "smoke"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/smoke_test_embodiment_phase34.py",
            "--out-dir",
            str(out_dir),
            "--variant",
            "g1_29dof",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stdout = json.loads(proc.stdout)
    summary = json.loads((out_dir / "phase34_smoke_summary.json").read_text())

    assert stdout["status"] == "ok"
    assert summary["status"] == "ok"
    assert summary["training_manifest"]["promotion_eligible"] is False
    assert (out_dir / "phase34_training_rows.jsonl").exists()
