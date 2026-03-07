#!/usr/bin/env python3
"""Nightly roadmap audit for economic-world-model readiness."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_ROOT = REPO_ROOT / "docs" / "economic_world_model"
ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "economic_world_model"
DEFAULT_JSON = ARTIFACT_ROOT / "nightly_audit_summary.json"
DEFAULT_MARKDOWN = ARTIFACT_ROOT / "nightly_audit_summary.md"

REQUIRED_DOCS = [
    "docs/economic_world_model/architecture_gap_analysis.md",
    "docs/economic_world_model/roadmap.md",
    "docs/economic_world_model/progress_log.md",
    "docs/economic_world_model/nightly_audit.md",
    "docs/economic_world_model/codex_skill.md",
    "docs/economic_world_model/implementation_notes.md",
    "docs/economic_world_model/AUTOMATION_SPEC.md",
]
REQUIRED_AUTOMATION_FILES = [
    "codex_skills/economic-world-model-roadmap/SKILL.md",
    "scripts/economic_world_model/run_nightly_codex_task.sh",
    "scripts/economic_world_model/update_status_issue.py",
    ".github/workflows/economic-world-model-nightly.yml",
]
REQUIRED_SCAFFOLDS = [
    "src/runtime/packets.py",
    "src/embodiment/registry.py",
]
DEFAULT_CHECKS = [
    ("agent_verify", "./scripts/agent/verify.sh"),
    ("compileall", "python3 -m compileall src scripts/economic_world_model -q"),
    (
        "targeted_pytest",
        "python3 -m pytest -q "
        "tests/test_runtime_packets.py "
        "tests/embodiment/test_registry.py "
        "tests/test_objective_runtime_builder.py "
        "tests/test_constraint_set.py "
        "tests/test_pricing_sentinel.py "
        "tests/test_value_ledger.py",
    ),
]


def _sha256_json(payload: Dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git_commit() -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode == 0:
        return proc.stdout.strip()
    return "unknown"


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def _exists(rel_path: str) -> bool:
    return (REPO_ROOT / rel_path).exists()


def _search(rel_path: str, pattern: str) -> bool:
    return pattern in _read_text(REPO_ROOT / rel_path)


def _progress_latest_date() -> Optional[str]:
    progress_text = _read_text(DOCS_ROOT / "progress_log.md")
    match = re.search(r"^##\s+(\d{4}-\d{2}-\d{2})$", progress_text, re.MULTILINE)
    if match:
        return match.group(1)
    return None


def _backlog_updated_at() -> Optional[str]:
    backlog_path = REPO_ROOT / "scripts" / "TRAINING_MIGRATION_BACKLOG.json"
    if not backlog_path.exists():
        return None
    payload = json.loads(backlog_path.read_text(encoding="utf-8"))
    return payload.get("updated_at")


def _run_check(name: str, command: str) -> Dict[str, Any]:
    proc = subprocess.run(
        command,
        cwd=REPO_ROOT,
        shell=True,
        capture_output=True,
        text=True,
        check=False,
    )
    stdout_lines = [line for line in proc.stdout.splitlines() if line.strip()]
    stderr_lines = [line for line in proc.stderr.splitlines() if line.strip()]
    return {
        "name": name,
        "command": command,
        "passed": proc.returncode == 0,
        "exit_code": proc.returncode,
        "stdout_tail": stdout_lines[-12:],
        "stderr_tail": stderr_lines[-12:],
    }


def _task_candidates() -> List[Dict[str, Any]]:
    return [
        {
            "id": "runtime_packet_sidecar_wiring",
            "title": "Wire RuntimePacket sidecars into shadow runtime and replay ingest",
            "classification": "additive_wiring",
            "rationale": (
                "The canonical packet scaffolds exist, but the live shadow runtime and replay bridge "
                "still emit objective/econ/constraint artifacts independently. Sidecar packet emission "
                "is the smallest change that materially increases stack legibility and auditability."
            ),
            "targets": [
                "src/shadow_runtime/control_plane.py",
                "src/replay/ingest.py",
                "tests/test_shadow_econ_runner.py",
                "tests/test_replay_schema.py",
                "tests/test_runtime_packets.py",
            ],
            "execute_now": True,
            "pending": _exists("src/runtime/packets.py")
            and (
                not _search("src/shadow_runtime/control_plane.py", "RuntimePacket")
                or not _search("src/replay/ingest.py", "RuntimePacket")
            ),
        },
        {
            "id": "adapter_v2_scaffold",
            "title": "Add ActionAdapterV2 and ObservationAdapterV2 schema contracts",
            "classification": "scaffolding_only",
            "rationale": (
                "Embodiment registry scaffolding now exists, but the repo still lacks a canonical "
                "action/observation schema layer with timing and provenance annotations."
            ),
            "targets": [
                "src/runtime/action_adapter_v2.py",
                "src/runtime/observation_adapter_v2.py",
                "tests/test_runtime_packets.py",
            ],
            "execute_now": True,
            "pending": not _exists("src/runtime/action_adapter_v2.py")
            or not _exists("src/runtime/observation_adapter_v2.py"),
        },
        {
            "id": "event_spine_spec",
            "title": "Draft EventSpine and GovernanceTrace spec before code wiring",
            "classification": "docs_only",
            "rationale": (
                "Dense temporal auditability remains the biggest missing precondition after packets and "
                "embodiment normalization. A doc-first spec keeps the next wiring pass small and additive."
            ),
            "targets": [
                "docs/economic_world_model/architecture_gap_analysis.md",
                "docs/economic_world_model/roadmap.md",
                "docs/economic_world_model/progress_log.md",
            ],
            "execute_now": True,
            "pending": True,
        },
    ]


def _next_task() -> Dict[str, Any]:
    for candidate in _task_candidates():
        if candidate["pending"]:
            return {
                "id": candidate["id"],
                "title": candidate["title"],
                "classification": candidate["classification"],
                "rationale": candidate["rationale"],
                "target_files": list(candidate["targets"]),
                "execute_now": bool(candidate["execute_now"]),
            }
    return {
        "id": "audit_only",
        "title": "No missing additive step detected; refresh docs and verification only",
        "classification": "docs_only",
        "rationale": "The current scan did not find a higher-priority missing additive scaffold.",
        "target_files": [],
        "execute_now": False,
    }


def _docs_status() -> List[Dict[str, Any]]:
    return [
        {"path": path, "present": _exists(path)}
        for path in REQUIRED_DOCS + REQUIRED_AUTOMATION_FILES
    ]


def _scaffold_status() -> List[Dict[str, Any]]:
    return [
        {"path": path, "present": _exists(path)}
        for path in REQUIRED_SCAFFOLDS
    ]


def _drift_signals(
    docs_status: List[Dict[str, Any]],
    scaffold_status: List[Dict[str, Any]],
) -> List[str]:
    signals: List[str] = []
    for row in docs_status:
        if not row["present"]:
            signals.append(f"missing_required_doc_or_automation:{row['path']}")
    for row in scaffold_status:
        if not row["present"]:
            signals.append(f"missing_scaffold:{row['path']}")

    latest_progress = _progress_latest_date()
    backlog_updated = _backlog_updated_at()
    if latest_progress and backlog_updated and backlog_updated > latest_progress:
        signals.append("training_backlog_newer_than_progress_log")

    if _exists("src/runtime/packets.py") and not _search("docs/economic_world_model/roadmap.md", "RuntimePacket"):
        signals.append("runtime_packet_missing_from_roadmap_doc")

    if _exists("src/embodiment/registry.py") and not _search(
        "docs/economic_world_model/architecture_gap_analysis.md", "EmbodimentRegistry"
    ):
        signals.append("embodiment_registry_missing_from_gap_analysis")

    return signals


def _execution_paths() -> Dict[str, Any]:
    return {
        "codex_cli_installed": shutil.which("codex") is not None,
        "local_nightly_runner": _exists("scripts/economic_world_model/run_nightly_codex_task.sh"),
        "cloud_workflow": _exists(".github/workflows/economic-world-model-nightly.yml"),
        "repo_skill": _exists("codex_skills/economic-world-model-roadmap/SKILL.md"),
        "codex_api_key_present": bool(os.environ.get("CODEX_API_KEY") or os.environ.get("OPENAI_API_KEY")),
    }


def _render_markdown(summary: Dict[str, Any]) -> str:
    verification_lines = []
    for row in summary["verification"]:
        status = "PASS" if row["passed"] else "FAIL"
        verification_lines.append(f"| {row['name']} | {status} | `{row['command']}` |")
    if not verification_lines:
        verification_lines.append("| verification | SKIPPED | checks disabled |")

    drift_lines = summary["roadmap_drift"]["signals"] or ["none"]
    docs_lines = [
        f"- {'present' if row['present'] else 'missing'} `{row['path']}`"
        for row in summary["docs_status"]
    ]
    scaffold_lines = [
        f"- {'present' if row['present'] else 'missing'} `{row['path']}`"
        for row in summary["scaffold_status"]
    ]
    next_task = summary["next_task"]
    execution = summary["execution_paths"]

    return "\n".join(
        [
            "# Economic World Model Nightly Audit",
            "",
            f"- Generated at: `{summary['generated_at']}`",
            f"- Commit: `{summary['git_commit']}`",
            f"- Status: `{summary['status']}`",
            f"- Summary digest: `{summary['summary_digest']}`",
            "",
            "## Verification",
            "",
            "| Check | Result | Command |",
            "| --- | --- | --- |",
            *verification_lines,
            "",
            "## Drift Signals",
            "",
            *[f"- {signal}" for signal in drift_lines],
            "",
            "## Docs And Automation Substrate",
            "",
            *docs_lines,
            "",
            "## Middleware Scaffolds",
            "",
            *scaffold_lines,
            "",
            "## Execution Readiness",
            "",
            f"- local_cli_runner: {'ready' if execution['codex_cli_installed'] and execution['local_nightly_runner'] else 'not_ready'}",
            f"- repo_skill: {'ready' if execution['repo_skill'] else 'missing'}",
            f"- github_cloud_workflow: {'ready' if execution['cloud_workflow'] else 'missing'}",
            f"- codex_api_key_present: {'yes' if execution['codex_api_key_present'] else 'no'}",
            "",
            "## Next Best Additive Task",
            "",
            f"- Title: {next_task['title']}",
            f"- Classification: `{next_task['classification']}`",
            f"- Rationale: {next_task['rationale']}",
            f"- Safe for automatic execution: `{'yes' if next_task['execute_now'] else 'no'}`",
            *[f"- Target: `{path}`" for path in next_task["target_files"]],
        ]
    )


def build_summary(skip_checks: bool = False) -> Dict[str, Any]:
    docs_status = _docs_status()
    scaffold_status = _scaffold_status()
    verification = [] if skip_checks else [_run_check(name, cmd) for name, cmd in DEFAULT_CHECKS]
    drift_signals = _drift_signals(docs_status, scaffold_status)
    next_task = _next_task()
    execution_paths = _execution_paths()

    status = "ok"
    if any(not row["passed"] for row in verification) or drift_signals:
        status = "attention"

    stable_digest_payload = {
        "git_commit": _git_commit(),
        "docs_status": docs_status,
        "scaffold_status": scaffold_status,
        "verification": [
            {"name": row["name"], "passed": row["passed"], "exit_code": row["exit_code"]}
            for row in verification
        ],
        "roadmap_drift": drift_signals,
        "next_task": next_task,
        "execution_paths": execution_paths,
        "status": status,
    }

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": stable_digest_payload["git_commit"],
        "status": status,
        "docs_status": docs_status,
        "scaffold_status": scaffold_status,
        "verification": verification,
        "roadmap_drift": {"signals": drift_signals},
        "next_task": next_task,
        "execution_paths": execution_paths,
        "progress_log_latest": _progress_latest_date(),
        "training_backlog_updated_at": _backlog_updated_at(),
        "summary_digest": _sha256_json(stable_digest_payload),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the economic-world-model nightly audit.")
    parser.add_argument("--output-json", default=str(DEFAULT_JSON))
    parser.add_argument("--output-markdown", default=str(DEFAULT_MARKDOWN))
    parser.add_argument("--skip-checks", action="store_true")
    args = parser.parse_args()

    summary = build_summary(skip_checks=args.skip_checks)
    output_json = Path(args.output_json)
    output_markdown = Path(args.output_markdown)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_markdown.write_text(_render_markdown(summary) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
