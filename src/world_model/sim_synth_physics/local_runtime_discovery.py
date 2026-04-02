"""Targeted local host discovery for Phase-1 runtime roots."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


def _candidate_base_dirs() -> list[Path]:
    home = Path.home()
    cwd = Path.cwd().resolve()
    candidates = [
        cwd,
        cwd.parent,
        cwd.parent.parent,
        home,
        home / "code",
        home / "src",
        home / "repos",
        home / "dev",
        home / "workspace",
        home / "projects",
    ]
    seen: set[Path] = set()
    existing: list[Path] = []
    for path in candidates:
        try:
            resolved = path.resolve()
        except Exception:
            continue
        if resolved in seen or not resolved.exists():
            continue
        seen.add(resolved)
        existing.append(resolved)
    return existing


def discover_named_root(names: Iterable[str]) -> dict[str, object]:
    checked_paths: list[str] = []
    for base in _candidate_base_dirs():
        for name in names:
            candidate = (base / name).resolve()
            checked_paths.append(str(candidate))
            if candidate.exists():
                return {
                    "ref": str(candidate),
                    "source": "autodiscovery",
                    "checked_paths": checked_paths,
                }
    return {
        "ref": "",
        "source": "",
        "checked_paths": checked_paths,
    }


__all__ = ["discover_named_root"]
