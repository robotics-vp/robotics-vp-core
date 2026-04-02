"""Local/symbolic reference evidence helpers for Phase-1 runtime surfaces."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


def _looks_like_local_path(ref: str) -> bool:
    if not ref:
        return False
    path = Path(ref)
    return (
        path.is_absolute()
        or ref.startswith(".")
        or "/" in ref
        or "\\" in ref
        or bool(path.suffix)
    )


def describe_ref_evidence(ref: Any) -> dict[str, Any]:
    cleaned = str(ref or "").strip()
    if not cleaned:
        return {
            "ref": "",
            "verification_status": "missing",
            "is_local_path": False,
            "ready": False,
            "verified": False,
        }
    if cleaned == "inline_motion_clips":
        return {
            "ref": cleaned,
            "verification_status": "inline_ref",
            "is_local_path": False,
            "ready": True,
            "verified": True,
        }
    if _looks_like_local_path(cleaned):
        exists = Path(cleaned).exists()
        return {
            "ref": cleaned,
            "verification_status": "local_path_exists" if exists else "local_path_missing",
            "is_local_path": True,
            "ready": exists,
            "verified": exists,
        }
    return {
        "ref": cleaned,
        "verification_status": "symbolic_ref",
        "is_local_path": False,
        "ready": True,
        "verified": False,
    }


def summarize_preflight_evidence(
    required_components: list[str],
    evidence_by_component: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    missing: list[str] = []
    unready: list[str] = []
    symbolic: list[str] = []
    verified: list[str] = []
    ready: list[str] = []
    for component in required_components:
        evidence = dict(evidence_by_component.get(component) or {})
        status = str(evidence.get("verification_status", "") or "")
        if status in {"missing", "local_path_missing"}:
            missing.append(component)
            continue
        if not bool(evidence.get("ready", False)):
            missing.append(component)
            unready.append(component)
            continue
        if bool(evidence.get("ready", False)):
            ready.append(component)
        if bool(evidence.get("verified", False)):
            verified.append(component)
        elif status in {"symbolic_ref"}:
            symbolic.append(component)
    preflight_status = "preflight_ready"
    if missing:
        preflight_status = "preflight_blocked"
    elif symbolic:
        preflight_status = "preflight_partial"
    return {
        "status": preflight_status,
        "missing_components": missing,
        "symbolic_components": symbolic,
        "verified_components": verified,
        "ready_components": ready,
        "unready_components": unready,
        "required_components": list(required_components),
    }


__all__ = ["describe_ref_evidence", "summarize_preflight_evidence"]
