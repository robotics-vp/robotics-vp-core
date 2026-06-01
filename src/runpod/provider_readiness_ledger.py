"""Provider bring-up readiness ledger for RunPod-backed WM lanes.

This module emits planning receipts only. It does not launch RunPod, download
weights, execute providers, train models, promote outputs, or grant runtime
authority.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

from src.runpod.launch_profiles import build_runpod_launch_manifest
from src.utils.config_digest import sha256_json
from src.utils.json_safe import to_json_safe

PROVIDER_READINESS_LEDGER_VERSION = "provider_readiness_ledger_v1"
PROVIDER_READINESS_REPORT_VERSION = "provider_readiness_report_v1"


def _mapping(payload: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    return dict(to_json_safe(dict(payload or {})))


@dataclass(frozen=True)
class ProviderReadinessEntry:
    """One external provider family and its expected proof receipts."""

    ledger_key: str
    provider_family: str
    owner_wm: str
    owner_subsystem: str
    runpod_profile: str
    command: str
    expected_receipts: list[str]
    unavailable_mode: str
    external_blockers: list[str]
    local_prerequisites: list[str] = field(default_factory=list)
    notes: str = ""
    authority_class: str = "provider_readiness_ledger_only"
    promotion_eligible: bool = False
    version: str = PROVIDER_READINESS_LEDGER_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "ledger_key": self.ledger_key,
            "version": self.version,
            "provider_family": self.provider_family,
            "owner_wm": self.owner_wm,
            "owner_subsystem": self.owner_subsystem,
            "runpod_profile": self.runpod_profile,
            "command": self.command,
            "expected_receipts": list(self.expected_receipts),
            "unavailable_mode": self.unavailable_mode,
            "external_blockers": list(self.external_blockers),
            "local_prerequisites": list(self.local_prerequisites),
            "notes": self.notes,
            "authority_class": self.authority_class,
            "promotion_eligible": bool(self.promotion_eligible),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProviderReadinessEntry":
        return cls(
            ledger_key=str(payload.get("ledger_key", "")),
            provider_family=str(payload.get("provider_family", "")),
            owner_wm=str(payload.get("owner_wm", "")),
            owner_subsystem=str(payload.get("owner_subsystem", "")),
            runpod_profile=str(payload.get("runpod_profile", "")),
            command=str(payload.get("command", "")),
            expected_receipts=[str(item) for item in list(payload.get("expected_receipts", []) or [])],
            unavailable_mode=str(payload.get("unavailable_mode", "provider_unavailable_receipt")),
            external_blockers=[str(item) for item in list(payload.get("external_blockers", []) or [])],
            local_prerequisites=[str(item) for item in list(payload.get("local_prerequisites", []) or [])],
            notes=str(payload.get("notes", "")),
            authority_class=str(payload.get("authority_class", "provider_readiness_ledger_only")),
            promotion_eligible=bool(payload.get("promotion_eligible", False)),
            version=str(payload.get("version", PROVIDER_READINESS_LEDGER_VERSION)),
        )


@dataclass(frozen=True)
class ProviderReadinessReport:
    """Receipt for current local provider bring-up readiness."""

    report_id: str
    generated_at: str
    status: str
    entries: list[ProviderReadinessEntry]
    local_prerequisite_status: dict[str, bool]
    provider_execution_attempted: bool = False
    provider_or_hardware_proof: bool = False
    promotion_eligible: bool = False
    artifact_refs: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    version: str = PROVIDER_READINESS_REPORT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "version": self.version,
            "generated_at": self.generated_at,
            "status": self.status,
            "entry_count": len(self.entries),
            "entries": [entry.to_dict() for entry in self.entries],
            "local_prerequisite_status": dict(self.local_prerequisite_status),
            "provider_execution_attempted": bool(self.provider_execution_attempted),
            "provider_or_hardware_proof": bool(self.provider_or_hardware_proof),
            "promotion_eligible": bool(self.promotion_eligible),
            "artifact_refs": _mapping(self.artifact_refs),
            "metadata": _mapping(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProviderReadinessReport":
        return cls(
            report_id=str(payload.get("report_id", "")),
            generated_at=str(payload.get("generated_at", "")),
            status=str(payload.get("status", "unknown")),
            entries=[
                ProviderReadinessEntry.from_dict(row)
                for row in list(payload.get("entries", []) or [])
                if isinstance(row, Mapping)
            ],
            local_prerequisite_status={
                str(key): bool(value)
                for key, value in dict(payload.get("local_prerequisite_status", {}) or {}).items()
            },
            provider_execution_attempted=bool(payload.get("provider_execution_attempted", False)),
            provider_or_hardware_proof=bool(payload.get("provider_or_hardware_proof", False)),
            promotion_eligible=bool(payload.get("promotion_eligible", False)),
            artifact_refs=_mapping(payload.get("artifact_refs")),
            metadata=_mapping(payload.get("metadata")),
            version=str(payload.get("version", PROVIDER_READINESS_REPORT_VERSION)),
        )


def default_provider_readiness_entries() -> list[ProviderReadinessEntry]:
    """Return the bounded provider families named by the multi-WM audit."""

    provider_command = "python3 scripts/runpod/prepare_launch_manifest.py --profile provider_bringup"
    return [
        ProviderReadinessEntry(
            ledger_key="sam_sam3d_grounding",
            provider_family="SAM/SAM3D",
            owner_wm="perception_grounding",
            owner_subsystem="open_vocabulary_segmentation_tracking",
            runpod_profile="provider_bringup",
            command=provider_command,
            expected_receipts=[
                "provider_availability_receipt_v1",
                "segmentation_tracking_truth_receipt_v1",
                "perception_benchmark_evidence_v1",
                "provider_unavailable_receipt_v1",
            ],
            unavailable_mode="emit provider_unavailable_receipt; keep promotion_eligible=false",
            external_blockers=["weights_access", "cuda_host", "real_video_or_image_fixture"],
            local_prerequisites=["runpodctl", "RUNPOD_API_KEY"],
            notes="SAM/SAM3D outputs are provider truth only after real runtime receipts exist.",
        ),
        ProviderReadinessEntry(
            ledger_key="dino_siglip_backbone",
            provider_family="DINO/SigLIP",
            owner_wm="perception_grounding",
            owner_subsystem="vision_backbone_projection",
            runpod_profile="provider_bringup",
            command=provider_command,
            expected_receipts=[
                "vision_backbone_provider_truth_receipt_v1",
                "projection_seam_input_receipt_v1",
                "provider_unavailable_receipt_v1",
            ],
            unavailable_mode="emit provider_unavailable_receipt; use deterministic local seam tests only",
            external_blockers=["weights_access", "cuda_host", "calibrated_frame_corpus"],
            local_prerequisites=["runpodctl", "RUNPOD_API_KEY"],
            notes="Do not treat deterministic latent defaults as backbone provider proof.",
        ),
        ProviderReadinessEntry(
            ledger_key="vjepa2_temporal_grounding",
            provider_family="V-JEPA2",
            owner_wm="perception_grounding_and_sim_synth_physics",
            owner_subsystem="temporal_grounding_and_predictive_state",
            runpod_profile="provider_bringup",
            command=provider_command,
            expected_receipts=[
                "vjepa2_runtime_truth_receipt_v1",
                "temporal_alignment_input_receipt_v1",
                "predictive_state_provider_receipt_v1",
                "provider_unavailable_receipt_v1",
            ],
            unavailable_mode="emit provider_unavailable_receipt; keep V-JEPA lanes scaffold-only",
            external_blockers=["upstream_runtime_checkout", "weights_access", "cuda_host", "video_window_corpus"],
            local_prerequisites=["runpodctl", "RUNPOD_API_KEY"],
            notes="V-JEPA2 is a component lane, not a separate truth owner.",
        ),
        ProviderReadinessEntry(
            ledger_key="openvla_teacher_runtime",
            provider_family="OpenVLA",
            owner_wm="perception_grounding",
            owner_subsystem="teacher_semantic_proposals",
            runpod_profile="provider_bringup",
            command=provider_command,
            expected_receipts=[
                "teacher_runtime_provider_truth_receipt_v1",
                "teacher_action_envelope_receipt_v1",
                "provider_unavailable_receipt_v1",
            ],
            unavailable_mode="emit explicit teacher fallback envelope; keep external/advisory",
            external_blockers=["weights_access", "cuda_host", "task_specific_teacher_corpus"],
            local_prerequisites=["runpodctl", "RUNPOD_API_KEY"],
            notes="External VLA proposals remain advisory sidecars.",
        ),
        ProviderReadinessEntry(
            ledger_key="isaac_unitree_runtime",
            provider_family="Isaac/Unitree",
            owner_wm="sim_synth_physics_and_embodiment_actuation",
            owner_subsystem="g1_runtime_execution",
            runpod_profile="g1_loop_run",
            command="python3 scripts/runpod/prepare_launch_manifest.py --profile g1_loop_run --volume-id \"$RUNPOD_VOLUME_ID\"",
            expected_receipts=[
                "isaac_unitree_backend_execution_receipt_v1",
                "unitree_asset_calibration_receipt_v1",
                "whole_body_replay_receipt_v1",
                "runtime_unavailable_receipt_v1",
            ],
            unavailable_mode="emit runtime_unavailable_receipt; do not call local shadow success G1 proof",
            external_blockers=["RUNPOD_VOLUME_ID", "isaac_runtime", "unitree_assets", "policy_checkpoint"],
            local_prerequisites=["runpodctl", "RUNPOD_API_KEY", "RUNPOD_VOLUME_ID"],
            notes="G1/bipedal whole-body stays primary; fixed-base curricula stay fallback/regression sources.",
        ),
        ProviderReadinessEntry(
            ledger_key="holosoma_runtime",
            provider_family="Holosoma",
            owner_wm="sim_synth_physics_and_embodiment_actuation",
            owner_subsystem="motion_retargeting_runtime",
            runpod_profile="g1_loop_run",
            command="python3 scripts/runpod/prepare_launch_manifest.py --profile g1_loop_run --volume-id \"$RUNPOD_VOLUME_ID\"",
            expected_receipts=[
                "holosoma_runtime_execution_receipt_v1",
                "motion_datapack_binding_receipt_v1",
                "whole_body_rollout_receipt_v1",
                "runtime_unavailable_receipt_v1",
            ],
            unavailable_mode="emit runtime_unavailable_receipt; keep shadow work orders planning-only",
            external_blockers=["RUNPOD_VOLUME_ID", "holosoma_runtime", "motion_corpus", "retargeting_assets"],
            local_prerequisites=["runpodctl", "RUNPOD_API_KEY", "RUNPOD_VOLUME_ID"],
            notes="Holosoma local smoke visibility is not native runtime execution proof.",
        ),
    ]


def local_runpod_prerequisite_status(
    *,
    volume_id: str | None = None,
    api_key: str | None = None,
) -> dict[str, bool]:
    """Inspect local RunPod launch prerequisites without launching anything."""

    resolved_api_key = api_key if api_key is not None else os.environ.get("RUNPOD_API_KEY", "")
    resolved_volume_id = volume_id if volume_id is not None else os.environ.get("RUNPOD_VOLUME_ID", "")
    return {
        "runpodctl_installed": shutil.which("runpodctl") is not None,
        "RUNPOD_API_KEY_set": bool(str(resolved_api_key or "").strip()),
        "RUNPOD_VOLUME_ID_set": bool(str(resolved_volume_id or "").strip()),
    }


def build_provider_readiness_report(
    *,
    volume_id: str | None = None,
    api_key: str | None = None,
    entries: Optional[list[ProviderReadinessEntry]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> ProviderReadinessReport:
    """Build an honest provider-readiness report for local planning."""

    resolved_entries = entries or default_provider_readiness_entries()
    prereq_status = local_runpod_prerequisite_status(volume_id=volume_id, api_key=api_key)
    provider_prereqs_ready = prereq_status["runpodctl_installed"] and prereq_status["RUNPOD_API_KEY_set"]
    loop_prereqs_ready = provider_prereqs_ready and prereq_status["RUNPOD_VOLUME_ID_set"]
    status = (
        "ready_to_prepare_provider_manifests"
        if provider_prereqs_ready
        else "blocked_local_runpod_prerequisites"
    )
    provider_manifest_preview = build_runpod_launch_manifest(
        profile_id="provider_bringup",
        run_id="runpod-provider-readiness-ledger-preview",
        volume_id=volume_id,
    )
    generated_at = datetime.now(timezone.utc).isoformat()
    report_seed = {
        "generated_at": generated_at,
        "entries": [entry.ledger_key for entry in resolved_entries],
        "prereq_status": prereq_status,
        "version": PROVIDER_READINESS_REPORT_VERSION,
    }
    return ProviderReadinessReport(
        report_id=f"provider_readiness_{sha256_json(report_seed)[:16]}",
        generated_at=generated_at,
        status=status,
        entries=resolved_entries,
        local_prerequisite_status=prereq_status,
        artifact_refs={},
        metadata={
            "boundary": "local readiness ledger only; no provider execution attempted",
            "provider_profiles_ready": bool(provider_prereqs_ready),
            "loop_or_training_profiles_ready": bool(loop_prereqs_ready),
            "provider_bringup_manifest_preview": {
                "run_id": provider_manifest_preview["run_id"],
                "profile_id": "provider_bringup",
                "status": provider_manifest_preview["status"],
                "pod_class": provider_manifest_preview["pod_class"],
                "epistemic_status": provider_manifest_preview["epistemic_status"],
            },
            **_mapping(metadata),
        },
    )


def write_provider_readiness_report(
    output_dir: str | Path,
    *,
    volume_id: str | None = None,
    api_key: str | None = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Write provider readiness JSON and Markdown receipts."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    report = build_provider_readiness_report(
        volume_id=volume_id,
        api_key=api_key,
        metadata=metadata,
    )
    json_path = output_path / "provider_readiness_report_v1.json"
    markdown_path = output_path / "provider_readiness_report_v1.md"
    payload = report.to_dict()
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    markdown_path.write_text(render_provider_readiness_markdown(report), encoding="utf-8")
    return {
        "status": report.status,
        "report_id": report.report_id,
        "entry_count": len(report.entries),
        "provider_execution_attempted": False,
        "promotion_eligible": False,
        "json_path": str(json_path),
        "markdown_path": str(markdown_path),
        "report": payload,
    }


def render_provider_readiness_markdown(report: ProviderReadinessReport) -> str:
    """Render a compact human-readable ledger."""

    lines = [
        "# Provider Readiness Ledger",
        "",
        f"- Report: `{report.report_id}`",
        f"- Status: `{report.status}`",
        "- Boundary: local readiness ledger only; no provider, GPU, hardware, training, or promotion proof",
        f"- Provider execution attempted: `{str(report.provider_execution_attempted).lower()}`",
        f"- Promotion eligible: `{str(report.promotion_eligible).lower()}`",
        "",
        "## Local Prerequisites",
        "",
    ]
    for key, ready in sorted(report.local_prerequisite_status.items()):
        lines.append(f"- `{key}`: `{str(ready).lower()}`")
    lines.extend(
        [
            "",
            "## Provider Families",
            "",
            "| Provider | Owner WM | RunPod profile | Unavailable mode |",
            "| --- | --- | --- | --- |",
        ]
    )
    for entry in report.entries:
        lines.append(
            f"| {entry.provider_family} | `{entry.owner_wm}` | `{entry.runpod_profile}` | {entry.unavailable_mode} |"
        )
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "PROVIDER_READINESS_LEDGER_VERSION",
    "PROVIDER_READINESS_REPORT_VERSION",
    "ProviderReadinessEntry",
    "ProviderReadinessReport",
    "build_provider_readiness_report",
    "default_provider_readiness_entries",
    "local_runpod_prerequisite_status",
    "render_provider_readiness_markdown",
    "write_provider_readiness_report",
]
