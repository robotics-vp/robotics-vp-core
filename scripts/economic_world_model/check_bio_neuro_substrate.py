#!/usr/bin/env python3
"""Emit local bio/neuro substrate receipts.

This is a structural check only. It proves the typed substrate can be built
locally from G1-facing scaffold state; it does not run providers, train, launch
pods, control hardware, or promote any output.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

from src.economics.economic_wm_entry import evaluate_economic_wm_entry_preflight  # noqa: E402
from src.regal.bio_neuro_anomaly import (  # noqa: E402
    build_anomaly_suspicion_receipt,
    build_governance_escalation_event,
)
from src.runtime.action_adapter_v2 import ActionAdapterV2  # noqa: E402
from src.runtime.observation_adapter_v2 import ObservationAdapterV2  # noqa: E402
from src.world_model.economic_world_model import (  # noqa: E402
    build_bio_neuro_receipt_join_from_paths,
    build_economic_wm_scaffold_report,
    build_regime_acknowledgment,
    build_regime_broadcast,
)
from src.world_model.embodiment_actuation import (  # noqa: E402
    build_embodiment_bio_neuro_substrate,
    build_g1_morphology_profile,
    compile_embodiment_actuation_with_receipts,
    unitree_g1_contract,
)
from src.world_model.perception_grounding.bio_neuro_receipts import (  # noqa: E402
    build_active_sensing_receipt,
    build_self_disturbance_receipt,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def _build_embodiment_state() -> Any:
    profile = build_g1_morphology_profile("g1_29dof")
    entry = profile.to_registry_entry()
    joint_names = profile.joint_names()
    return compile_embodiment_actuation_with_receipts(
        episode_id="bio_neuro_local_substrate_smoke",
        frame_index=1,
        embodiment_registry_entry=entry,
        action_adapter=ActionAdapterV2(
            schema_id=entry.action_schema_id,
            channel_order=joint_names,
            control_hz=50.0,
            translator_ref=entry.translator_refs["retarget"],
            embodiment_id=entry.embodiment_id,
        ),
        observation_adapter=ObservationAdapterV2(
            schema_id=entry.observation_schema_id,
            proprio_fields=[f"q_{joint}" for joint in joint_names],
            sensor_refs=["proprio://unitree_g1", "rgb://head_camera_shadow"],
            sample_hz=50.0,
            embodiment_id=entry.embodiment_id,
        ),
        provider_contracts=[unitree_g1_contract()],
        joint_state={
            "joint_names": joint_names,
            "positions": [0.0 for _ in joint_names],
            "velocities": [0.01 for _ in joint_names],
            "efforts": [0.0 for _ in joint_names],
            "timestamp_s": 1.0,
        },
        source_refs={"morphology_profile_id": profile.profile_id},
        metadata={"embodiment_id": entry.embodiment_id},
    )


def _build_economic_report() -> Any:
    preflight = evaluate_economic_wm_entry_preflight(
        stage1_sweep_report={
            "status": "ok",
            "scenario_count": 2,
            "admission_count": 2,
            "rlds_episode_count": 2,
            "lerobot_row_count": 2,
            "benchmark_ready_count": 0,
            "shadow_only_count": 2,
            "promotion_eligible": False,
            "failures": [],
            "scenario_reports": [],
        }
    )
    return build_economic_wm_scaffold_report(preflight)


def run_check(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    embodiment_result = _build_embodiment_state()
    substrate = build_embodiment_bio_neuro_substrate(
        embodiment_result.state,
        perception_uncertainty={
            "occlusion": 0.72,
            "contact_ambiguity": 0.48,
            "identity": 0.22,
        },
        value_of_information_prior=0.25,
    )
    expectation_payload = substrate.self_motion_expectation.to_dict()
    proposal_payload = substrate.active_sensing_proposals[0].to_dict()
    self_disturbance = build_self_disturbance_receipt(
        expectation_payload,
        perception_state_id="perception_state_shadow",
        observed_delta={"joint_velocity_l1": 0.25, "contact_transition_risk": 0.1},
        temporal_alignment_quality=0.8,
        external_change_score=0.1,
    )
    active_sensing_receipt = build_active_sensing_receipt(
        proposal_payload,
        perception_state_id="perception_state_shadow",
        uncertainty_before={"occlusion": 0.72, "contact_ambiguity": 0.48},
        uncertainty_after={"occlusion": 0.42, "contact_ambiguity": 0.44},
        executed=False,
    )

    economic_report = _build_economic_report()
    broadcast = build_regime_broadcast(
        economic_report.economic_state,
        economic_report.allocation_envelope,
    )
    acknowledgments = [
        build_regime_acknowledgment(
            broadcast,
            wm_id=wm_id,
            accepted_settings={
                "trust_posture": broadcast.posture_settings["trust_posture"],
                "compute_posture": broadcast.posture_settings["compute_posture"],
            },
        )
        for wm_id in (
            "perception_grounding",
            "embodiment_actuation",
            "sim_synth_physics",
        )
    ]
    anomaly_receipt = build_anomaly_suspicion_receipt(
        domain="embodiment_actuation",
        anomaly_type="self_disturbance_mismatch",
        evidence_scores={
            "self_disturbance_mismatch": self_disturbance.mismatch_magnitude,
            "telemetry_gap": 0.4,
        },
        evidence_refs={"self_disturbance_receipt": self_disturbance.receipt_id},
    )
    escalation = build_governance_escalation_event([anomaly_receipt])

    receipt_rows = [
        substrate.receipt.to_dict() if substrate.receipt is not None else {},
        expectation_payload,
        *[proposal.to_dict() for proposal in substrate.active_sensing_proposals],
        *[entry.to_dict() for entry in substrate.synergy_codebook_entries],
        substrate.interoceptive_state.to_dict()
        if substrate.interoceptive_state is not None
        else {},
        self_disturbance.to_dict(),
        active_sensing_receipt.to_dict(),
        broadcast.to_dict(),
        *[ack.to_dict() for ack in acknowledgments],
        anomaly_receipt.to_dict(),
        escalation.to_dict(),
    ]
    receipt_rows = [row for row in receipt_rows if row]
    receipts_path = output_dir / "bio_neuro_substrate_receipts_v1.jsonl"
    report_path = output_dir / "bio_neuro_substrate_report_v1.json"
    join_report_path = output_dir / "bio_neuro_receipt_join_report_v1.json"
    join_rows_path = output_dir / "bio_neuro_receipt_join_rows_v1.jsonl"
    _write_jsonl(receipts_path, receipt_rows)

    report = {
        "status": "ok_bio_neuro_substrate_passed",
        "version": "bio_neuro_substrate_report_v1",
        "surface_count": len(receipt_rows),
        "surface_versions": [str(row.get("version", "")) for row in receipt_rows],
        "promotion_eligible": False,
        "provider_or_hardware_proof": False,
        "trained_model_proof": False,
        "authority_level": "none",
        "missing_external_proof": [
            "provider_execution",
            "gpu_training",
            "unitree_sim_runtime",
            "unitree_hardware_runtime",
            "promotion_grade_benchmarks",
        ],
        "output_paths": {
            "receipts_path": str(receipts_path),
            "report_path": str(report_path),
            "receipt_join_report_path": str(join_report_path),
            "receipt_join_rows_path": str(join_rows_path),
        },
    }
    _write_json(report_path, report)
    join_report = build_bio_neuro_receipt_join_from_paths(
        receipts_path=receipts_path,
        output_dir=output_dir,
        source_report_path=report_path,
        report_path=join_report_path,
        join_rows_path=join_rows_path,
        metadata={"source": "check_bio_neuro_substrate"},
    )
    report["receipt_join_status"] = join_report.status
    report["receipt_join_row_count"] = join_report.row_count
    report["receipt_join_promotion_eligible"] = join_report.promotion_eligible
    report["receipt_join_phase7_abstraction_expanded"] = (
        join_report.phase7_abstraction_expanded
    )
    _write_json(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(run_check(Path(args.output_dir)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
