"""Datapack metadata helpers for epiplexity metrics."""
from __future__ import annotations

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
        default = summary.get("_default", {}) if isinstance(summary, dict) else {}
        repr_id = repr_id or default.get("repr_id")
        budget_id = budget_id or default.get("budget_id")
    if not repr_id or not budget_id:
        return None
    repr_bucket = summary.get(repr_id, {})
    budget_bucket = repr_bucket.get(budget_id, {}) if isinstance(repr_bucket, dict) else {}
    mean = budget_bucket.get("mean", {}) if isinstance(budget_bucket, dict) else {}
    if metric in mean:
        try:
            return float(mean.get(metric))
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
        default = summary.get("_default", {}) if isinstance(summary, dict) else {}
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


__all__ = [
    "attach_epiplexity_result",
    "attach_epiplexity_summary",
    "extract_epiplexity_summary_metric",
    "extract_epiplexity_summary_confidence",
]
