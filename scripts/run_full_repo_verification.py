#!/usr/bin/env python3
"""CLI wrapper for compile/lint/type/test verification with CI-friendly JSON output."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(command: list[str]) -> dict:
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout[-4000:],
        "stderr": completed.stderr[-4000:],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run additive full-repo verification")
    parser.add_argument("--skip-mypy", action="store_true")
    parser.add_argument("--skip-pytest", action="store_true")
    args = parser.parse_args()

    checks = [
        _run(["python3", "-m", "compileall", "src", "scripts", "tests", "-q"]),
        _run(["ruff", "check", "."]),
        _run(["ruff", "format", "--check", "."]),
    ]
    if not args.skip_mypy:
        checks.append(_run(["mypy", "src/"]))
    if not args.skip_pytest:
        checks.append(_run(["pytest", "tests/", "-q"]))

    payload = {
        "root": str(ROOT),
        "checks": checks,
        "all_passed": all(check["returncode"] == 0 for check in checks),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
