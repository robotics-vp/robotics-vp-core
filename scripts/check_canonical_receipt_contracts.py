#!/usr/bin/env python3
"""Hard-gate canonical receipt and provider-truth contract hygiene."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


REQUIRED_RECEIPT_TOKENS = (
    "receipt_kind",
    "authority_class",
    "decision_scope",
    "reward_math_mutation",
)

RECEIPT_SCAN_DIRS = (
    "src/economics",
    "src/evidence",
    "src/orchestrator",
    "src/phase_h",
    "src/rl",
    "src/semantic",
    "src/vla",
    "src/vision",
    "src/world_model/sim_synth_physics",
)

PROVIDER_TRUTH_HELPER_TOKENS = (
    "build_external_provider_truth",
    "coerce_external_provider_truth",
    "build_teacher_provider_truth",
    "build_scene_tracks_provider_truth",
)


def iter_python_files(repo_root: Path, relative_dirs: Sequence[str]) -> Iterable[Path]:
    for relative_dir in relative_dirs:
        root = repo_root / relative_dir
        if not root.exists():
            continue
        yield from sorted(root.rglob("*.py"))


def find_receipt_contract_issues(repo_root: Path, relative_dirs: Sequence[str] = RECEIPT_SCAN_DIRS) -> List[Dict[str, str]]:
    issues: List[Dict[str, str]] = []
    for path in iter_python_files(repo_root, relative_dirs):
        content = path.read_text(encoding="utf-8")
        if "receipt_kind" not in content:
            continue
        missing = [token for token in REQUIRED_RECEIPT_TOKENS if token not in content]
        if missing:
            issues.append(
                {
                    "kind": "receipt_contract_missing_fields",
                    "file": str(path.relative_to(repo_root)),
                    "detail": ", ".join(missing),
                }
            )
    return issues


def find_provider_truth_issues(repo_root: Path, relative_dirs: Sequence[str] = RECEIPT_SCAN_DIRS) -> List[Dict[str, str]]:
    issues: List[Dict[str, str]] = []
    for path in iter_python_files(repo_root, relative_dirs):
        content = path.read_text(encoding="utf-8")
        if "provider_truth" not in content:
            continue
        if "tests/" in str(path):
            continue
        if "authority_class" in content:
            continue
        if any(token in content for token in PROVIDER_TRUTH_HELPER_TOKENS):
            continue
        issues.append(
            {
                "kind": "provider_truth_missing_canonicalization",
                "file": str(path.relative_to(repo_root)),
                "detail": "provider_truth appears without canonical helper or authority metadata",
            }
        )
    return issues


def run_guardrail(repo_root: Path) -> Dict[str, object]:
    receipt_issues = find_receipt_contract_issues(repo_root)
    provider_truth_issues = find_provider_truth_issues(repo_root)
    issues = [*receipt_issues, *provider_truth_issues]
    return {
        "repo_root": str(repo_root),
        "issue_count": len(issues),
        "receipt_issue_count": len(receipt_issues),
        "provider_truth_issue_count": len(provider_truth_issues),
        "issues": issues,
        "all_passed": not issues,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=str,
        default=str(Path(__file__).resolve().parents[1]),
        help="Repository root to scan",
    )
    parser.add_argument(
        "--json-output",
        type=str,
        default="",
        help="Optional path to write a JSON report",
    )
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve()
    payload = run_guardrail(repo_root)

    if args.json_output:
        Path(args.json_output).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
