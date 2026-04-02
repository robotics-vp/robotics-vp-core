"""Local/symbolic reference evidence helpers for Phase-1 runtime surfaces."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence


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


def _evidence_rank(evidence: Mapping[str, Any]) -> int:
    if bool(evidence.get("verified", False)):
        return 3
    if bool(evidence.get("ready", False)):
        return 2
    status = str(evidence.get("verification_status", "") or "")
    if status == "local_path_missing":
        return 0
    if status == "missing":
        return -1
    return 0


def summarize_ref_candidates(refs: Sequence[Any]) -> dict[str, Any]:
    unique_refs: list[str] = []
    seen: set[str] = set()
    for ref in refs:
        cleaned = str(ref or "").strip()
        if not cleaned or cleaned in seen:
            continue
        unique_refs.append(cleaned)
        seen.add(cleaned)

    evidence_rows = [describe_ref_evidence(ref) for ref in unique_refs]
    verified_refs = [str(row.get("ref", "") or "") for row in evidence_rows if bool(row.get("verified", False))]
    ready_refs = [str(row.get("ref", "") or "") for row in evidence_rows if bool(row.get("ready", False))]
    symbolic_refs = [
        str(row.get("ref", "") or "")
        for row in evidence_rows
        if str(row.get("verification_status", "") or "") == "symbolic_ref"
    ]
    missing_refs = [
        str(row.get("ref", "") or "")
        for row in evidence_rows
        if str(row.get("verification_status", "") or "") in {"missing", "local_path_missing"}
    ]
    return {
        "candidate_count": len(unique_refs),
        "ready_candidate_count": len(ready_refs),
        "verified_candidate_count": len(verified_refs),
        "symbolic_candidate_count": len(symbolic_refs),
        "missing_candidate_count": len(missing_refs),
        "primary_verified_ref": verified_refs[0] if verified_refs else "",
        "primary_ready_ref": ready_refs[0] if ready_refs else "",
        "primary_symbolic_ref": symbolic_refs[0] if symbolic_refs else "",
    }


def select_best_named_ref(named_candidates: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    best_source = ""
    best_ref = ""
    best_evidence = describe_ref_evidence("")
    best_rank = -2
    candidate_refs: list[str] = []
    for source, ref in named_candidates:
        cleaned = str(ref or "").strip()
        if not cleaned:
            continue
        candidate_refs.append(cleaned)
        evidence = describe_ref_evidence(cleaned)
        rank = _evidence_rank(evidence)
        if rank > best_rank:
            best_source = str(source or "")
            best_ref = cleaned
            best_evidence = evidence
            best_rank = rank
    if not best_ref:
        return {
            "ref": "",
            "source": "",
            "evidence": describe_ref_evidence(""),
            "summary": summarize_ref_candidates(candidate_refs),
        }
    return {
        "ref": best_ref,
        "source": best_source,
        "evidence": best_evidence,
        "summary": summarize_ref_candidates(candidate_refs),
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


__all__ = [
    "describe_ref_evidence",
    "select_best_named_ref",
    "summarize_preflight_evidence",
    "summarize_ref_candidates",
]
