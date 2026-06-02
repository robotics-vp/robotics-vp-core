#!/usr/bin/env python3
"""Compile the cross-WM provider bring-up readiness ledger."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

if __package__ is None or __package__ == "":
    REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

from src.world_model.economic_world_model import (  # noqa: E402
    ProviderBringupLedgerEntry,
    ProviderBringupLedgerReport,
    build_provider_bringup_ledger,
    save_provider_bringup_ledger,
    validate_provider_bringup_ledger,
)

DEFAULT_OUTPUT_DIR = Path(
    "artifacts/economic_world_model/provider_bringup_readiness_ledger"
)


def _write_markdown(
    path: Path,
    *,
    report: ProviderBringupLedgerReport,
    entries: list[ProviderBringupLedgerEntry],
    validation: Mapping[str, Any],
) -> None:
    payload = report.to_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Provider Bring-Up Readiness Ledger",
        "",
        f"- Report ID: `{payload['report_id']}`",
        f"- Status: `{payload['status']}`",
        f"- Entry count: `{payload['entry_count']}`",
        f"- Required families covered: `{payload['covered_required_family_count']}/{payload['required_family_count']}`",
        f"- Launch allowed count: `{payload['launch_allowed_count']}`",
        f"- Provider bring-up ready count: `{payload['provider_bringup_ready_count']}`",
        f"- Local verification command count: `{payload['local_verification_available_count']}`",
        f"- RunPod template count: `{payload['runpod_template_count']}`",
        f"- Promotion eligible: `{str(payload['promotion_eligible']).lower()}`",
        f"- Provider executed: `{str(payload['provider_executed']).lower()}`",
        f"- RunPod launched: `{str(payload['runpod_launched']).lower()}`",
        f"- Safe for template storage: `{str(validation['safe_for_template_storage']).lower()}`",
        f"- Safe for launch: `{str(validation['safe_for_launch']).lower()}`",
        "",
        "## Provider Rows",
        "",
    ]
    for entry in entries:
        lines.extend(
            [
                f"### `{entry.provider_key}`",
                f"- family: `{entry.provider_family}`",
                f"- owner WM: `{entry.owner_wm}`",
                f"- run class: `{entry.run_class}`",
                f"- pod class: `{entry.pod_class}`",
                f"- runpod profile: `{entry.runpod_profile}`",
                f"- launch allowed: `{str(entry.launch_allowed).lower()}`",
                f"- provider bring-up ready: `{str(entry.provider_bringup_ready).lower()}`",
                f"- unavailable posture: `{entry.unavailable_posture}`",
                "- expected receipts:",
                *[f"  - `{receipt}`" for receipt in entry.expected_receipts],
                "- blocker codes:",
                *[f"  - `{blocker}`" for blocker in entry.blocker_codes],
                "- local verification:",
                *[
                    f"  - `{command}`"
                    for command in entry.local_verification_commands
                ],
                "",
            ]
        )
    lines.extend(
        [
            "## Boundary",
            "",
            "This ledger is template-only. It does not download weights, launch",
            "RunPod, execute providers, run GPU jobs, operate hardware, write",
            "weights, mutate reward math, grant authority, or claim promotion.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_validation_markdown(path: Path, validation: Mapping[str, Any]) -> None:
    lines = [
        "# Provider Bring-Up Ledger Validation",
        "",
        f"- Status: `{validation['status']}`",
        f"- Safe for template storage: `{str(validation['safe_for_template_storage']).lower()}`",
        f"- Safe for launch: `{str(validation['safe_for_launch']).lower()}`",
        f"- Error count: `{validation['error_count']}`",
        f"- Warning count: `{validation['warning_count']}`",
    ]
    if validation["errors"]:
        lines.extend(["", "## Errors"])
        lines.extend(f"- `{error}`" for error in validation["errors"])
    if validation["warnings"]:
        lines.extend(["", "## Warnings"])
        lines.extend(f"- `{warning}`" for warning in validation["warnings"])
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "Validation proves the ledger remains safe to store as blocked",
            "template evidence. It is intentionally not safe for launch.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_compile_provider_bringup_readiness_ledger(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    foundation_backlog_path: str | Path = "scripts/FOUNDATION_MODEL_BRINGUP_BACKLOG.json",
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    artifact_refs = {
        "report_path": str(output / "provider_bringup_ledger_report_v1.json"),
        "entries_path": str(output / "provider_bringup_ledger_entries_v1.jsonl"),
        "manifest_template_dir": str(output / "manifest_templates"),
        "markdown_path": str(output / "provider_bringup_readiness_ledger_v1.md"),
        "validation_path": str(output / "provider_bringup_ledger_validation_v1.json"),
        "validation_markdown_path": str(
            output / "provider_bringup_ledger_validation_v1.md"
        ),
    }
    report, entries = build_provider_bringup_ledger(
        foundation_backlog_path=foundation_backlog_path,
        artifact_refs=artifact_refs,
        metadata={"source": "compile_provider_bringup_readiness_ledger_script"},
    )
    saved_refs = save_provider_bringup_ledger(output, report=report, entries=entries)
    validation = validate_provider_bringup_ledger(report=report, entries=entries)
    validation_payload = {
        **validation,
        "report_id": report.report_id,
        "artifact_refs": {
            "report_path": saved_refs["report_path"],
            "entries_path": saved_refs["entries_path"],
            "manifest_template_dir": saved_refs["manifest_template_dir"],
        },
    }
    Path(artifact_refs["validation_path"]).write_text(
        json.dumps(validation_payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_markdown(
        Path(artifact_refs["markdown_path"]),
        report=report,
        entries=entries,
        validation=validation,
    )
    _write_validation_markdown(
        Path(artifact_refs["validation_markdown_path"]), validation_payload
    )
    payload = report.to_dict()
    payload["entries"] = [entry.to_dict() for entry in entries]
    payload["validation"] = validation_payload
    payload["artifact_refs"] = {
        **dict(payload.get("artifact_refs", {}) or {}),
        **saved_refs,
        "markdown_path": artifact_refs["markdown_path"],
        "validation_path": artifact_refs["validation_path"],
        "validation_markdown_path": artifact_refs["validation_markdown_path"],
    }
    Path(saved_refs["report_path"]).write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--foundation-backlog",
        default="scripts/FOUNDATION_MODEL_BRINGUP_BACKLOG.json",
    )
    args = parser.parse_args()
    payload = run_compile_provider_bringup_readiness_ledger(
        output_dir=args.output_dir,
        foundation_backlog_path=args.foundation_backlog,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    validation = payload["validation"]
    return (
        0
        if payload["status"] == "ok"
        and payload["all_entries_fail_closed"]
        and validation["safe_for_template_storage"]
        and not validation["safe_for_launch"]
        and payload["launch_allowed_count"] == 0
        and payload["provider_bringup_ready_count"] == 0
        and not payload["provider_executed"]
        and not payload["promotion_eligible"]
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
