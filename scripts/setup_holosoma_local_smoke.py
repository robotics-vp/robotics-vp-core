#!/usr/bin/env python3
"""Install or remove the lightweight local Holosoma smoke-test path shim."""

from __future__ import annotations

import argparse
import json
import os
import site
from pathlib import Path


DEFAULT_HOLOSOMA_ROOT = Path(os.environ.get("HOLOSOMA_ROOT", "/Users/amarmurray/code/holosoma"))
DEFAULT_PTH_NAME = "robotics_vp_holosoma_local.pth"
PATH_SHIM_VERSION = "holosoma_local_smoke_path_shim_v1"
HOLOSOMA_SUBPATHS = (
    "src/holosoma",
    "src/holosoma_inference",
    "src/holosoma_retargeting",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _smoke_requirements_path() -> Path:
    return _repo_root() / "requirements-holosoma-smoke.txt"


def _site_packages_dir(override: str | None) -> Path:
    if override:
        return Path(override).expanduser()
    return Path(site.getusersitepackages())


def _path_entries(holosoma_root: Path) -> list[Path]:
    return [(holosoma_root / subpath).resolve() for subpath in HOLOSOMA_SUBPATHS]


def _build_install_report(
    *,
    holosoma_root: Path,
    site_packages_dir: Path,
    pth_name: str,
    dry_run: bool,
) -> dict:
    entries = _path_entries(holosoma_root)
    missing_entries = [str(path) for path in entries if not path.exists()]
    pth_path = site_packages_dir / pth_name
    status = "blocked_missing_paths" if missing_entries else "dry_run" if dry_run else "ready_to_write"
    return {
        "version": PATH_SHIM_VERSION,
        "status": status,
        "holosoma_root": str(holosoma_root),
        "site_packages_dir": str(site_packages_dir),
        "pth_path": str(pth_path),
        "path_entries": [str(path) for path in entries],
        "missing_path_entries": missing_entries,
        "smoke_requirements_path": str(_smoke_requirements_path()),
        "smoke_dependency_install_hint": (
            "python3 -m pip install --user --no-cache-dir -r requirements-holosoma-smoke.txt"
        ),
        "post_install_check_hint": (
            "python3 scripts/local_holosoma_smoke.py --preflight-only "
            "--out-dir artifacts/holosoma_local_probe"
        ),
    }


def _write_pth(report: dict) -> dict:
    if report["missing_path_entries"]:
        return {**report, "written": False}
    site_packages_dir = Path(str(report["site_packages_dir"]))
    pth_path = Path(str(report["pth_path"]))
    site_packages_dir.mkdir(parents=True, exist_ok=True)
    content = "\n".join(str(path) for path in report["path_entries"]) + "\n"
    pth_path.write_text(content, encoding="utf-8")
    return {**report, "status": "installed", "written": True}


def _remove_pth(*, site_packages_dir: Path, pth_name: str, dry_run: bool) -> dict:
    pth_path = site_packages_dir / pth_name
    existed = pth_path.exists()
    if existed and not dry_run:
        pth_path.unlink()
    return {
        "version": PATH_SHIM_VERSION,
        "status": "dry_run_remove" if dry_run else "removed" if existed else "already_absent",
        "site_packages_dir": str(site_packages_dir),
        "pth_path": str(pth_path),
        "existed": existed,
        "removed": bool(existed and not dry_run),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create a lightweight user-site .pth shim for the existing local Holosoma checkout. "
            "This intentionally does not install the full Holosoma dependency tree."
        )
    )
    parser.add_argument("--holosoma-root", default=str(DEFAULT_HOLOSOMA_ROOT))
    parser.add_argument("--site-packages-dir", default=None)
    parser.add_argument("--pth-name", default=DEFAULT_PTH_NAME)
    parser.add_argument("--remove", action="store_true", help="Remove the configured .pth shim.")
    parser.add_argument("--dry-run", action="store_true", help="Print the planned change without writing.")
    args = parser.parse_args(argv)

    site_packages_dir = _site_packages_dir(args.site_packages_dir)
    if args.remove:
        report = _remove_pth(
            site_packages_dir=site_packages_dir,
            pth_name=str(args.pth_name),
            dry_run=bool(args.dry_run),
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    report = _build_install_report(
        holosoma_root=Path(str(args.holosoma_root)).expanduser().resolve(),
        site_packages_dir=site_packages_dir,
        pth_name=str(args.pth_name),
        dry_run=bool(args.dry_run),
    )
    if args.dry_run:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if not report["missing_path_entries"] else 2
    report = _write_pth(report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not report["missing_path_entries"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
