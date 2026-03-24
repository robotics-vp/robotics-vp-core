"""Datapack metadata helpers for epiplexity metrics."""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional



def _get_attr(obj: Any, key: str, default: Any = None) -> Any:
    if hasattr(obj, key):
        return getattr(obj, key)
    if isinstance(obj, dict):
        return obj.get(key, default)
    return default


def _set_attr(obj: Any, key: str, value: Any) -> None:
    if hasattr(obj, key):
        setattr(obj, key, value)
    elif isinstance(obj, dict):
        obj[key] = value


def attach_epiplexity_result(datapack: Any, result: Any) -> None:
    """Attach a single epiplexity run result to datapack metadata."""
    epiplexity = _get_attr(datapack, "epiplexity") or {}
    repr_bucket = epiplexity.setdefault(result.key.repr_id, {})
    budget_bucket = repr_bucket.setdefault(result.key.compute_budget_id, {})
    budget_bucket[str(result.key.seed)] = {
        "S_T_proxy": result.S_T_proxy,
        "H_T_proxy": result.H_T_proxy,
        "epi_per_flop": result.epi_per_flop,
        "delta_epi_vs_baseline": result.delta_epi_vs_baseline,
        "repr_version_hash": result.key.repr_version_hash,
        "tokenizer_version": result.key.tokenizer_version,
        "transform_chain_hash": result.key.transform_chain_hash,
        "dataset_slice_id": result.key.dataset_slice_id,
        "probe_model_id": result.key.probe_model_id,
        "compute_budget_id": result.key.compute_budget_id,
        "seed": result.key.seed,
        "flops_estimate": float(getattr(result, "flops_estimate", 0.0) or 0.0),
        "compute_normalizer": getattr(result, "compute_normalizer", "flops_estimate"),
        "estimator_id": getattr(result, "estimator_id", ""),
        "estimator_config_sha": getattr(result, "estimator_config_sha", ""),
        "score_mode": getattr(result, "score_mode", "absolute"),
        "baseline_repr_id": getattr(result, "baseline_repr_id", None),
    }
    _set_attr(datapack, "epiplexity", epiplexity)


def attach_epiplexity_summary(
    datapack: Any,
    repr_id: str,
    budget_id: str,
    summary: Dict[str, Any],
    set_default: bool = False,
) -> None:
    epiplexity_summary = _get_attr(datapack, "epiplexity_summary") or {}
    repr_bucket = epiplexity_summary.setdefault(repr_id, {})
    repr_bucket[budget_id] = summary
    if set_default:
        epiplexity_summary["_default"] = {"repr_id": repr_id, "budget_id": budget_id}
    _set_attr(datapack, "epiplexity_summary", epiplexity_summary)


def set_epiplexity_default_selector(
    datapack: Any,
    repr_id: str,
    budget_id: str,
    reason: str = "best_delta_epi_vs_baseline",
) -> None:
    epiplexity_summary = _get_attr(datapack, "epiplexity_summary") or {}
    epiplexity_summary["_default"] = {
        "repr_id": str(repr_id),
        "budget_id": str(budget_id),
        "reason": str(reason),
    }
    _set_attr(datapack, "epiplexity_summary", epiplexity_summary)


def select_default_epiplexity_summary(
    summary: Dict[str, Any],
    *,
    primary_metric: str = "delta_epi_vs_baseline",
    fallback_metric: str = "epi_per_flop",
) -> Optional[Dict[str, str]]:
    if not isinstance(summary, dict):
        return None
    explicit = summary.get("_default")
    if (
        isinstance(explicit, dict)
        and explicit.get("repr_id")
        and explicit.get("budget_id")
        and isinstance(summary.get(explicit["repr_id"]), dict)
        and explicit["budget_id"] in summary.get(explicit["repr_id"], {})
    ):
        return {
            "repr_id": str(explicit["repr_id"]),
            "budget_id": str(explicit["budget_id"]),
        }

    best: Optional[Dict[str, str]] = None
    best_score: Optional[tuple[float, float]] = None
    for repr_id, budgets in summary.items():
        if repr_id == "_default" or not isinstance(budgets, dict):
            continue
        for budget_id, stats in budgets.items():
            if not isinstance(stats, dict):
                continue
            mean = stats.get("mean", {}) if isinstance(stats.get("mean"), dict) else {}
            primary = _coerce_optional_float(mean.get(primary_metric))
            fallback = _coerce_optional_float(mean.get(fallback_metric))
            score = (
                primary if primary is not None else float("-inf"),
                fallback if fallback is not None else float("-inf"),
            )
            if best_score is None or score > best_score:
                best = {"repr_id": str(repr_id), "budget_id": str(budget_id)}
                best_score = score
    return best


def build_epiplexity_overlay_record(datapack: Any) -> Optional[Dict[str, Any]]:
    pack_id = _get_attr(datapack, "pack_id")
    if not pack_id:
        return None
    epiplexity = _get_attr(datapack, "epiplexity")
    epiplexity_summary = _get_attr(datapack, "epiplexity_summary")
    if not epiplexity and not epiplexity_summary:
        return None
    return {
        "pack_id": str(pack_id),
        "task_name": _get_attr(datapack, "task_name"),
        "episode_id": _get_attr(datapack, "episode_id"),
        "epiplexity": epiplexity,
        "epiplexity_summary": epiplexity_summary,
    }


def write_epiplexity_overlays(datapacks: list[Any], overlay_path: str) -> int:
    overlay_rows = []
    seen: set[tuple[str, str]] = set()
    for datapack in datapacks:
        row = build_epiplexity_overlay_record(datapack)
        if row is None:
            continue
        key = (str(row.get("task_name") or ""), str(row["pack_id"]))
        if key in seen:
            continue
        seen.add(key)
        overlay_rows.append(row)
    if not overlay_rows:
        return 0
    parent = os.path.dirname(overlay_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(overlay_path, "w", encoding="utf-8") as handle:
        for row in overlay_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return len(overlay_rows)


def load_epiplexity_overlay_map(overlay_path: str) -> Dict[str, Dict[str, Any]]:
    overlays: Dict[str, Dict[str, Any]] = {}
    if not overlay_path or not os.path.exists(overlay_path):
        return overlays
    with open(overlay_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            pack_id = str(row.get("pack_id", "") or "")
            if not pack_id:
                continue
            overlays[pack_id] = dict(row)
    return overlays


def apply_epiplexity_overlay(target: Any, overlay: Dict[str, Any]) -> None:
    if not isinstance(overlay, dict):
        return
    if overlay.get("epiplexity") is not None:
        _set_attr(target, "epiplexity", overlay.get("epiplexity"))
    if overlay.get("epiplexity_summary") is not None:
        _set_attr(target, "epiplexity_summary", overlay.get("epiplexity_summary"))


def extract_epiplexity_summary_metric(
    datapack: Any,
    repr_id: Optional[str] = None,
    budget_id: Optional[str] = None,
    metric: str = "delta_epi_vs_baseline",
) -> Optional[float]:
    summary = _get_attr(datapack, "epiplexity_summary")
    if not isinstance(summary, dict):
        return None
    if repr_id is None or budget_id is None:
        default = select_default_epiplexity_summary(summary) or {}
        repr_id = repr_id or default.get("repr_id")
        budget_id = budget_id or default.get("budget_id")
    if not repr_id or not budget_id:
        return None
    repr_bucket = summary.get(repr_id, {})
    budget_bucket = repr_bucket.get(budget_id, {}) if isinstance(repr_bucket, dict) else {}
    mean = budget_bucket.get("mean", {}) if isinstance(budget_bucket, dict) else {}
    if metric in mean:
        metric_value = mean.get(metric)
        try:
            return float(metric_value) if metric_value is not None else None
        except Exception:
            return None
    return None


def extract_epiplexity_summary_confidence(
    datapack: Any,
    repr_id: Optional[str] = None,
    budget_id: Optional[str] = None,
) -> Optional[float]:
    summary = _get_attr(datapack, "epiplexity_summary")
    if not isinstance(summary, dict):
        return None
    if repr_id is None or budget_id is None:
        default = select_default_epiplexity_summary(summary) or {}
        repr_id = repr_id or default.get("repr_id")
        budget_id = budget_id or default.get("budget_id")
    if not repr_id or not budget_id:
        return None
    repr_bucket = summary.get(repr_id, {})
    budget_bucket = repr_bucket.get(budget_id, {}) if isinstance(repr_bucket, dict) else {}
    conf = budget_bucket.get("confidence") if isinstance(budget_bucket, dict) else None
    if conf is None:
        return None
    try:
        return float(conf)
    except Exception:
        return None


def _coerce_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


__all__ = [
    "apply_epiplexity_overlay",
    "attach_epiplexity_result",
    "attach_epiplexity_summary",
    "build_epiplexity_overlay_record",
    "extract_epiplexity_summary_metric",
    "extract_epiplexity_summary_confidence",
    "load_epiplexity_overlay_map",
    "select_default_epiplexity_summary",
    "set_epiplexity_default_selector",
    "write_epiplexity_overlays",
]
