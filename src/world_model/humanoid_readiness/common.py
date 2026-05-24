"""Shared helpers for humanoid readiness local scaffolds."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from src.utils.config_digest import sha256_json
from src.utils.json_safe import to_json_safe


DENIED_LOCAL_AUTHORITIES = (
    "training_executed",
    "weights_written",
    "provider_executed",
    "hardware_executed",
    "unitree_sim_runtime_executed",
    "live_policy_control",
    "reward_math_mutation",
    "promotion_eligible",
)

POSTURE_TAGS = (
    "bipedal_whole_body",
    "stable_base_mobile_manipulator",
    "fixed_base_tabletop",
    "unknown",
)


def mapping(payload: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    return dict(to_json_safe(dict(payload or {})))


def strings(values: Optional[Sequence[Any]]) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        return [values] if values else []
    return [str(value) for value in values if value not in (None, "")]


def float_mapping(payload: Optional[Mapping[str, Any]]) -> dict[str, float]:
    output: dict[str, float] = {}
    for key, value in dict(payload or {}).items():
        try:
            output[str(key)] = float(value)
        except Exception:
            continue
    return output


def denied_gate_map(extra: Optional[Mapping[str, Any]] = None) -> dict[str, bool]:
    gates = {key: False for key in DENIED_LOCAL_AUTHORITIES}
    gates.update({str(key): bool(value) for key, value in dict(extra or {}).items()})
    return gates


def stable_id(prefix: str, payload: Mapping[str, Any]) -> str:
    return f"{prefix}_{sha256_json(mapping(payload))[:16]}"


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    serialized = [json.dumps(row, sort_keys=True) for row in rows]
    target.write_text("\n".join(serialized) + "\n", encoding="utf-8")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows
