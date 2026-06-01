from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from src.economics.economic_wm_entry import evaluate_economic_wm_entry_preflight
from src.regal.bio_neuro_anomaly import (
    build_anomaly_suspicion_receipt,
    build_governance_escalation_event,
)
from src.runtime.action_adapter_v2 import ActionAdapterV2
from src.runtime.observation_adapter_v2 import ObservationAdapterV2
from src.world_model.economic_world_model import (
    build_economic_wm_scaffold_report,
    build_regime_acknowledgment,
    build_regime_broadcast,
    load_bio_neuro_receipt_join_report,
    load_bio_neuro_receipt_join_rows,
)
from src.world_model.embodiment_actuation import (
    build_embodiment_bio_neuro_substrate,
    build_g1_morphology_profile,
    compile_embodiment_actuation_with_receipts,
    unitree_g1_contract,
)
from src.world_model.perception_grounding.bio_neuro_receipts import (
    build_active_sensing_receipt,
    build_self_disturbance_receipt,
)


def _embodiment_state():
    profile = build_g1_morphology_profile("g1_29dof")
    entry = profile.to_registry_entry()
    joints = profile.joint_names()
    return compile_embodiment_actuation_with_receipts(
        episode_id="bio_neuro_test",
        frame_index=3,
        embodiment_registry_entry=entry,
        action_adapter=ActionAdapterV2(
            schema_id=entry.action_schema_id,
            channel_order=joints,
            control_hz=50.0,
            translator_ref=entry.translator_refs["retarget"],
            embodiment_id=entry.embodiment_id,
        ),
        observation_adapter=ObservationAdapterV2(
            schema_id=entry.observation_schema_id,
            proprio_fields=[f"q_{joint}" for joint in joints],
            sensor_refs=["proprio://unitree_g1", "rgb://head_camera_shadow"],
            sample_hz=50.0,
            embodiment_id=entry.embodiment_id,
        ),
        provider_contracts=[unitree_g1_contract()],
        joint_state={
            "joint_names": joints,
            "positions": [0.0 for _ in joints],
            "velocities": [0.02 for _ in joints],
            "efforts": [0.0 for _ in joints],
        },
        metadata={"embodiment_id": entry.embodiment_id},
    ).state


def _economic_report():
    preflight = evaluate_economic_wm_entry_preflight(
        stage1_sweep_report={
            "status": "ok",
            "scenario_count": 3,
            "admission_count": 3,
            "rlds_episode_count": 3,
            "lerobot_row_count": 3,
            "benchmark_ready_count": 1,
            "shadow_only_count": 2,
            "promotion_eligible": False,
            "failures": [],
            "scenario_reports": [],
        }
    )
    return build_economic_wm_scaffold_report(preflight)


def test_embodiment_bio_neuro_substrate_emits_typed_denied_surfaces() -> None:
    state = _embodiment_state()
    bundle = build_embodiment_bio_neuro_substrate(
        state,
        perception_uncertainty={"occlusion": 0.8, "contact_ambiguity": 0.5},
        value_of_information_prior=0.25,
    )
    payload = bundle.to_dict()

    assert payload["self_motion_expectation"]["version"] == "self_motion_expectation_v1"
    assert payload["self_motion_expectation"]["authority_level"] == "none"
    assert payload["self_motion_expectation"]["promotion_eligible"] is False
    assert payload["active_sensing_proposals"][0]["action_type"] in {
        "reposition_sensor",
        "sensor_mode_switch_depth",
        "cautious_exploratory_contact",
    }
    assert payload["active_sensing_proposals"][0]["promotion_eligible"] is False
    assert payload["synergy_codebook_entries"]
    assert payload["synergy_codebook_entries"][0]["learned"] is False
    assert payload["interoceptive_state"]["version"] == "interoceptive_state_v1"
    assert payload["receipt"]["promotion_eligible"] is False


def test_perception_receipts_consume_expectations_without_owning_body_truth() -> None:
    bundle = build_embodiment_bio_neuro_substrate(_embodiment_state())
    expectation = bundle.self_motion_expectation.to_dict()

    receipt = build_self_disturbance_receipt(
        expectation,
        perception_state_id="perception_state_test",
        observed_delta={"joint_velocity_l1": 0.5, "contact_transition_risk": 0.1},
        temporal_alignment_quality=0.75,
    )
    active = build_active_sensing_receipt(
        bundle.active_sensing_proposals[0].to_dict(),
        perception_state_id="perception_state_test",
        uncertainty_before={"occlusion": 0.8},
        uncertainty_after={"occlusion": 0.4},
        executed=False,
    )

    assert receipt.to_dict()["version"] == "self_disturbance_receipt_v1"
    assert receipt.to_dict()["metadata"]["perception_truth_owner"] is True
    assert receipt.promotion_eligible is False
    assert active.outcome_status == "not_executed"
    assert active.actual_information_gain > 0.0
    assert active.promotion_eligible is False


def test_economic_regime_broadcast_is_low_bandwidth_and_acknowledged() -> None:
    report = _economic_report()
    broadcast = build_regime_broadcast(
        report.economic_state,
        report.allocation_envelope,
    )
    ack = build_regime_acknowledgment(
        broadcast,
        wm_id="embodiment_actuation",
        accepted_settings={
            "trust_posture": broadcast.posture_settings["trust_posture"]
        },
    )

    payload = broadcast.to_dict()
    assert payload["version"] == "regime_broadcast_v1"
    assert payload["high_bandwidth_control"] is False
    assert payload["reward_math_mutation"] is False
    assert payload["promotion_eligible"] is False
    assert ack.local_authority_preserved is True
    assert ack.lower_wm_truth_redefined is False
    assert ack.promotion_eligible is False


def test_regal_anomaly_receipts_abstain_and_escalate_without_live_control() -> None:
    abstained = build_anomaly_suspicion_receipt(
        domain="perception_grounding",
        anomaly_type="identity_switch",
        evidence_scores={},
    )
    supported = build_anomaly_suspicion_receipt(
        domain="embodiment_actuation",
        anomaly_type="contact_mismatch",
        evidence_scores={"contact_mismatch": 0.9, "telemetry_quality": 0.8},
    )
    escalation = build_governance_escalation_event([abstained, supported])

    assert abstained.abstained is True
    assert supported.abstained is False
    assert supported.live_control_allowed is False
    assert escalation.escalation_level == "operator_review"
    assert escalation.live_control_allowed is False
    assert escalation.promotion_eligible is False


def test_bio_neuro_substrate_check_script_writes_receipts(tmp_path: Path) -> None:
    out_dir = tmp_path / "bio_neuro"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/economic_world_model/check_bio_neuro_substrate.py",
            "--output-dir",
            str(out_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stdout = json.loads(proc.stdout)
    report = json.loads((out_dir / "bio_neuro_substrate_report_v1.json").read_text())

    assert stdout["status"] == "ok_bio_neuro_substrate_passed"
    assert report["promotion_eligible"] is False
    assert report["provider_or_hardware_proof"] is False
    assert (out_dir / "bio_neuro_substrate_receipts_v1.jsonl").exists()
    assert report["receipt_join_status"] == "ok_bio_neuro_receipts_joined"
    assert report["receipt_join_promotion_eligible"] is False

    join_report_path = out_dir / "bio_neuro_receipt_join_report_v1.json"
    join_rows_path = out_dir / "bio_neuro_receipt_join_rows_v1.jsonl"
    assert join_report_path.exists()
    assert join_rows_path.exists()

    join_report = load_bio_neuro_receipt_join_report(join_report_path)
    join_rows = load_bio_neuro_receipt_join_rows(join_rows_path)
    assert join_report.row_count == len(join_rows)
    assert join_report.promotion_eligible is False
    assert join_report.provider_or_hardware_proof is False
    assert join_report.trained_model_proof is False
    assert join_report.phase7_abstraction_expanded is False
    assert "efference_copy" in join_report.economic_consumption_slots
    assert "active_sensing_value_of_information" in (
        join_report.economic_consumption_slots
    )
    assert "anomaly_governance" in join_report.economic_consumption_slots
    assert "regime_broadcast_conditioning" in join_report.economic_consumption_slots
    assert all(row.promotion_eligible is False for row in join_rows)
    assert all(row.ready_for_training is False for row in join_rows)
