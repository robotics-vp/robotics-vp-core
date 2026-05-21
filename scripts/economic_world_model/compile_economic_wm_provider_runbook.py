#!/usr/bin/env python3
"""Compile template-only Economic WM provider/GPU runbook artifacts."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

if __package__ is None or __package__ == "":
    REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

from src.world_model.economic_world_model import (  # noqa: E402
    EconomicWMProviderRunbook,
    build_economic_wm_provider_runbook_from_contract_path,
)


def _git_value(args: list[str], default: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return default
    value = result.stdout.strip()
    return value or default


def _write_markdown(path: Path, runbook: EconomicWMProviderRunbook) -> None:
    payload = runbook.to_dict()
    lines = [
        "# Economic WM Provider Runbook Templates",
        "",
        f"- Runbook ID: `{payload['runbook_id']}`",
        f"- Contract ID: `{payload['contract_id']}`",
        f"- Authority class: `{payload['authority_class']}`",
        f"- Launch allowed: `{str(payload['launch_allowed']).lower()}`",
        f"- Provider bring-up ready: `{str(payload['provider_bringup_ready']).lower()}`",
        f"- GPU training ready: `{str(payload['gpu_training_ready']).lower()}`",
        f"- Promotion eligible: `{str(payload['promotion_eligible']).lower()}`",
        f"- Reward math mutation: `{str(payload['reward_math_mutation']).lower()}`",
        "",
        "## Aggregate counts",
    ]
    for key, value in payload["aggregate_counts"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Templates"])
    for template in payload["templates"]:
        lines.extend(
            [
                f"### `{template['requirement_key']}`",
                f"- template id: `{template['template_id']}`",
                f"- mode: `{template['mode']}`",
                f"- run class: `{template['run_class']}`",
                f"- pod class: `{template['pod_class']}`",
                f"- epistemic status: `{template['epistemic_status']}`",
                f"- blocker: `{template['blocker']}`",
                f"- launch allowed: `{str(template['launch_allowed']).lower()}`",
                f"- local verification available: `{str(template['local_verification_available']).lower()}`",
                f"- current evidence status: `{template['current_status']}`",
            ]
        )
        if template["blocked_by"]:
            lines.append("- blocked by:")
            lines.extend(f"  - `{blocker}`" for blocker in template["blocked_by"])
        if template["required_artifacts"]:
            lines.append("- required artifacts:")
            lines.extend(
                f"  - `{artifact}`" for artifact in template["required_artifacts"]
            )
        lines.append("- command templates:")
        lines.extend(f"  - `{command}`" for command in template["command_templates"])
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "These artifacts are run templates only. The guard command intentionally fails until a human or agent replaces it with a real non-stub provider, GPU training, or benchmark command and records a real run manifest.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _resolve_contract(
    *,
    output_root: Path,
    contract_path: Optional[str | Path],
    run_contract_if_missing: bool,
) -> Path:
    if contract_path is not None:
        resolved = Path(contract_path)
        if resolved.exists():
            return resolved
        if not run_contract_if_missing:
            raise FileNotFoundError(resolved)
    else:
        resolved = Path(
            "artifacts/economic_world_model/economic_wm_teacher_provider_contracts/economic_wm_teacher_provider_contract_v1.json"
        )
        if resolved.exists():
            return resolved
        if not run_contract_if_missing:
            raise FileNotFoundError(resolved)

    from scripts.economic_world_model.prepare_economic_wm_teacher_provider_contracts import (  # noqa: E501
        run_prepare_economic_wm_teacher_provider_contracts,
    )

    contract_output = output_root / "teacher_provider_contracts"
    payload = run_prepare_economic_wm_teacher_provider_contracts(
        output_dir=contract_output,
    )
    return Path(payload["artifact_refs"]["contract_path"])


def run_compile_economic_wm_provider_runbook(
    *,
    output_dir: str | Path,
    contract_path: Optional[str | Path] = None,
    run_contract_if_missing: bool = True,
) -> Dict[str, Any]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    resolved_contract = _resolve_contract(
        output_root=output_root,
        contract_path=contract_path,
        run_contract_if_missing=run_contract_if_missing,
    )
    runbook_path = output_root / "economic_wm_provider_runbook_v1.json"
    markdown_path = output_root / "economic_wm_provider_runbook_v1.md"
    manifest_dir = output_root / "manifest_templates"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    runbook = build_economic_wm_provider_runbook_from_contract_path(
        contract_path=resolved_contract,
        output_path=runbook_path,
        artifact_refs={"contract_path": str(resolved_contract)},
        metadata={"source": "compile_economic_wm_provider_runbook_script"},
    )
    commit_sha = _git_value(["rev-parse", "--short", "HEAD"], "unknown")
    branch = _git_value(["branch", "--show-current"], "unknown")
    manifest_paths: list[str] = []
    template_payloads = []
    for template in runbook.templates:
        template_payload = template.to_dict()
        manifest_stub = template.to_manifest_stub(commit_sha=commit_sha, branch=branch)
        template_payload["manifest_stub"] = manifest_stub
        template_payloads.append(template_payload)
        manifest_path = manifest_dir / f"{template.template_id}.manifest_template.json"
        manifest_path.write_text(
            json.dumps(manifest_stub, indent=2, sort_keys=True), encoding="utf-8"
        )
        manifest_paths.append(str(manifest_path))

    payload = runbook.to_dict()
    payload["templates"] = template_payloads
    payload["artifact_refs"] = {
        **dict(payload.get("artifact_refs", {}) or {}),
        "runbook_path": str(runbook_path),
        "markdown_path": str(markdown_path),
        "manifest_template_dir": str(manifest_dir),
        "manifest_template_paths": manifest_paths,
        "contract_path": str(resolved_contract),
    }
    runbook_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_markdown(markdown_path, EconomicWMProviderRunbook.from_dict(payload))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="artifacts/economic_world_model/economic_wm_provider_runbook",
        help="Directory for template-only Economic WM provider runbook artifacts.",
    )
    parser.add_argument(
        "--contract",
        default=None,
        help="Existing economic_wm_teacher_provider_contract_v1.json path.",
    )
    parser.add_argument(
        "--no-run-contract",
        action="store_true",
        help="Do not materialize a teacher/provider contract if the contract path is missing.",
    )
    args = parser.parse_args()
    payload = run_compile_economic_wm_provider_runbook(
        output_dir=args.output_dir,
        contract_path=args.contract,
        run_contract_if_missing=not args.no_run_contract,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["templates"] and not payload["promotion_eligible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
