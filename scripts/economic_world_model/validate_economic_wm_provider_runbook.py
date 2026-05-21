#!/usr/bin/env python3
"""Validate Economic WM provider runbook templates remain template-only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

if __package__ is None or __package__ == "":
    REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

from src.world_model.economic_world_model import (  # noqa: E402
    EconomicWMProviderRunbookValidationReport,
    validate_economic_wm_provider_runbook_from_path,
)


def _write_markdown(
    path: Path, report: EconomicWMProviderRunbookValidationReport
) -> None:
    payload = report.to_dict()
    lines = [
        "# Economic WM Provider Runbook Validation",
        "",
        f"- Validation ID: `{payload['validation_id']}`",
        f"- Runbook ID: `{payload['runbook_id']}`",
        f"- Status: `{payload['status']}`",
        f"- Safe for template storage: `{str(payload['safe_for_template_storage']).lower()}`",
        f"- Safe for launch: `{str(payload['safe_for_launch']).lower()}`",
        f"- Error count: `{payload['error_count']}`",
        f"- Warning count: `{payload['warning_count']}`",
        "",
        "## Aggregate counts",
    ]
    for key, value in payload["aggregate_counts"].items():
        lines.append(f"- `{key}`: {value}")
    if payload["errors"]:
        lines.extend(["", "## Errors"])
        lines.extend(f"- `{error}`" for error in payload["errors"])
    if payload["warnings"]:
        lines.extend(["", "## Warnings"])
        lines.extend(f"- `{warning}`" for warning in payload["warnings"])
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "This validation proves only that the runbook remains safe to store as template evidence. It is deliberately not safe for launch and does not prove provider, GPU, benchmark, or promotion execution.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _resolve_runbook(
    *,
    output_root: Path,
    runbook_path: Optional[str | Path],
    compile_if_missing: bool,
) -> tuple[Path, Optional[Path]]:
    if runbook_path is not None:
        resolved = Path(runbook_path)
    else:
        resolved = Path(
            "artifacts/economic_world_model/economic_wm_provider_runbook/economic_wm_provider_runbook_v1.json"
        )
    if resolved.exists():
        manifest_dir = resolved.parent / "manifest_templates"
        return resolved, manifest_dir if manifest_dir.exists() else None
    if not compile_if_missing:
        raise FileNotFoundError(resolved)

    from scripts.economic_world_model.compile_economic_wm_provider_runbook import (  # noqa: E501
        run_compile_economic_wm_provider_runbook,
    )

    payload = run_compile_economic_wm_provider_runbook(
        output_dir=output_root / "provider_runbook",
    )
    return (
        Path(payload["artifact_refs"]["runbook_path"]),
        Path(payload["artifact_refs"]["manifest_template_dir"]),
    )


def run_validate_economic_wm_provider_runbook(
    *,
    output_dir: str | Path,
    runbook_path: Optional[str | Path] = None,
    manifest_template_dir: Optional[str | Path] = None,
    compile_if_missing: bool = True,
) -> Dict[str, Any]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    resolved_runbook, inferred_manifest_dir = _resolve_runbook(
        output_root=output_root,
        runbook_path=runbook_path,
        compile_if_missing=compile_if_missing,
    )
    resolved_manifest_dir = (
        Path(manifest_template_dir)
        if manifest_template_dir is not None
        else inferred_manifest_dir
    )
    validation_path = output_root / "economic_wm_provider_runbook_validation_v1.json"
    markdown_path = output_root / "economic_wm_provider_runbook_validation_v1.md"
    report = validate_economic_wm_provider_runbook_from_path(
        runbook_path=resolved_runbook,
        output_path=validation_path,
        manifest_template_dir=resolved_manifest_dir,
        metadata={"source": "validate_economic_wm_provider_runbook_script"},
    )
    payload = report.to_dict()
    payload["artifact_refs"] = {
        **dict(payload.get("artifact_refs", {}) or {}),
        "validation_path": str(validation_path),
        "markdown_path": str(markdown_path),
        "runbook_path": str(resolved_runbook),
        **(
            {"manifest_template_dir": str(resolved_manifest_dir)}
            if resolved_manifest_dir
            else {}
        ),
    }
    validation_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_markdown(
        markdown_path, EconomicWMProviderRunbookValidationReport.from_dict(payload)
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="artifacts/economic_world_model/economic_wm_provider_runbook_validation",
        help="Directory for provider runbook validation artifacts.",
    )
    parser.add_argument("--runbook", default=None)
    parser.add_argument("--manifest-template-dir", default=None)
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Do not compile the runbook if the input runbook is missing.",
    )
    args = parser.parse_args()
    payload = run_validate_economic_wm_provider_runbook(
        output_dir=args.output_dir,
        runbook_path=args.runbook,
        manifest_template_dir=args.manifest_template_dir,
        compile_if_missing=not args.no_compile,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return (
        0
        if payload["status"] == "ok"
        and payload["safe_for_template_storage"]
        and not payload["safe_for_launch"]
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
