#!/usr/bin/env python3
"""CPU-local Phase 3.4 smoke for Embodiment / Actuation seams and rows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.runtime.action_adapter_v2 import ActionAdapterV2
from src.runtime.observation_adapter_v2 import ObservationAdapterV2
from src.world_model.embodiment_actuation import (
    build_g1_morphology_profile,
    build_phase34_training_manifest,
    build_phase34_training_rows_from_state,
    compile_embodiment_actuation_with_receipts,
    scan_unitree_g1_public_evidence,
    smoke_forward_all_seams,
    unitree_g1_contract,
    write_phase34_training_rows_jsonl,
)
from src.utils.json_safe import to_json_safe


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="artifacts/embodiment_phase34")
    parser.add_argument(
        "--scan-root",
        action="append",
        default=[],
        help="Optional local public-repo root to scan for Unitree G1 evidence.",
    )
    parser.add_argument("--variant", default="g1_29dof")
    return parser


def main() -> int:
    args = _parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.scan_root:
        morphology, evidence_receipts = scan_unitree_g1_public_evidence(args.scan_root, variant=args.variant)
    else:
        morphology = build_g1_morphology_profile(args.variant)
        evidence_receipts = []

    registry_entry = morphology.to_registry_entry()
    action_adapter = ActionAdapterV2(
        schema_id=registry_entry.action_schema_id,
        channel_order=morphology.joint_names(),
        control_hz=50.0,
        latency_ms=0.0,
        translator_ref=registry_entry.translator_refs.get("retarget"),
        embodiment_id=registry_entry.embodiment_id,
    )
    observation_adapter = ObservationAdapterV2(
        schema_id=registry_entry.observation_schema_id,
        proprio_fields=[f"q_{joint}" for joint in morphology.joint_names()],
        sensor_refs=["proprio://unitree_g1", "imu://unitree_g1"],
        sample_hz=50.0,
        latency_ms=0.0,
        embodiment_id=registry_entry.embodiment_id,
    )
    result = compile_embodiment_actuation_with_receipts(
        episode_id="phase34_smoke",
        embodiment_registry_entry=registry_entry,
        action_adapter=action_adapter,
        observation_adapter=observation_adapter,
        provider_contracts=[unitree_g1_contract(metadata={"source": "phase34_smoke"})],
        joint_state={
            "joint_names": morphology.joint_names(),
            "positions": [0.0 for _ in morphology.joint_names()],
            "velocities": [0.0 for _ in morphology.joint_names()],
        },
        source_refs={"morphology_profile_id": morphology.profile_id},
    )
    rows = build_phase34_training_rows_from_state(result.state, result.receipts)
    manifest = build_phase34_training_manifest(
        rows,
        source_refs={"state_id": result.state.state_id, "morphology_profile_id": morphology.profile_id},
    )
    rows_path = write_phase34_training_rows_jsonl(rows, out_dir / "phase34_training_rows.jsonl")
    seam_smoke = smoke_forward_all_seams(result.state)

    summary = {
        "status": "ok" if all(item["finite"] for item in seam_smoke.values()) else "failed",
        "morphology": morphology.to_dict(),
        "evidence_receipts": [receipt.to_dict() for receipt in evidence_receipts],
        "state_id": result.state.state_id,
        "receipt_count": len(result.receipts),
        "training_manifest": manifest.to_dict(),
        "rows_path": str(rows_path),
        "seam_smoke": seam_smoke,
        "promotion_eligible": manifest.promotion_eligible,
    }
    (out_dir / "phase34_smoke_summary.json").write_text(
        json.dumps(to_json_safe(summary), indent=2, sort_keys=True)
    )
    print(json.dumps({"status": summary["status"], "out_dir": str(out_dir)}, sort_keys=True))
    return 0 if summary["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
