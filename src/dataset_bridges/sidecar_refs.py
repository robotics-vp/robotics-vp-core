"""Helpers to preserve replay sidecar references in dataset-bridge exports."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict


_SIDECAR_ID_KEYWORDS = (
    "event",
    "ledger",
    "decision",
    "packet",
    "contract",
    "trace",
    "datapack",
    "counterfactual",
    "target",
    "belief",
    "evidence",
    "teacher",
    "reconstruction",
    "supervision",
    "branch",
    "hypothesis",
    "pricing",
)


def _is_ref_key(key: str) -> bool:
    if key.endswith(("_ref", "_refs", "_path", "_paths")):
        return True
    if key.endswith(("_id", "_ids")):
        lowered = key.casefold()
        return any(token in lowered for token in _SIDECAR_ID_KEYWORDS)
    return False


def _is_ref_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value)
    if isinstance(value, list):
        return len(value) > 0
    return False


def extract_sidecar_refs(record: Any) -> Dict[str, Any]:
    """Collect sidecar-ish refs from replay records plus metadata/provenance."""
    refs: Dict[str, Any] = {}

    for key, value in vars(record).items():
        if _is_ref_key(key) and _is_ref_value(value):
            refs[key] = list(value) if isinstance(value, list) else value

    for envelope_key in ("metadata", "provenance"):
        envelope = getattr(record, envelope_key, None)
        if not isinstance(envelope, Mapping):
            continue
        for key, value in envelope.items():
            if _is_ref_key(key) and _is_ref_value(value):
                refs[key] = list(value) if isinstance(value, list) else value

    return refs


__all__ = ["extract_sidecar_refs"]
