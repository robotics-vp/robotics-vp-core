#!/usr/bin/env python3
"""CLI wrapper for compile/lint/type/test verification with CI-friendly JSON output."""
from __future__ import annotations

import argparse
import json
import shutil
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


def _resolve_tool(executable: str, module: str, *module_args: str) -> list[str]:
    if shutil.which(executable):
        return [executable, *module_args]
    return ["python3", "-m", module, *module_args]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run additive full-repo verification")
    parser.add_argument("--skip-ruff", action="store_true")
    parser.add_argument("--skip-format", action="store_true")
    parser.add_argument("--skip-mypy", action="store_true")
    parser.add_argument("--skip-pytest", action="store_true")
    args = parser.parse_args()

    checks = [_run(["python3", "-m", "compileall", "src", "scripts", "tests", "-q"])]
    if not args.skip_ruff:
        checks.append(_run(_resolve_tool("ruff", "ruff", "check", ".")))
    if not args.skip_format:
        checks.append(_run(_resolve_tool("ruff", "ruff", "format", "--check", ".")))
    if not args.skip_mypy:
        checks.append(_run(_resolve_tool("mypy", "mypy", "src/")))
    if not args.skip_pytest:
        checks.append(_run(_resolve_tool("pytest", "pytest", "tests/", "-q")))

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
