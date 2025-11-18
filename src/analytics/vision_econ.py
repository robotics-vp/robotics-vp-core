"""
Analytics helpers for correlating vision metadata with EconVectors.
"""
from typing import Dict, List, Callable, Any

from src.ontology.models import Episode, EconVector


def _backend_from_episode(ep: Episode) -> str:
    vision_meta = {}
    if getattr(ep, "vision_config", None):
        vision_meta = ep.vision_config
    if not vision_meta and getattr(ep, "metadata", None):
        vision_meta = (ep.metadata or {}).get("vision_metadata", {})
    return str((vision_meta or {}).get("backend") or (vision_meta or {}).get("backend_id") or "unknown")


def _vision_conditions(ep: Episode) -> Dict[str, Any]:
    if getattr(ep, "vision_conditions", None):
        return dict(ep.vision_conditions)
    if getattr(ep, "metadata", None):
        return dict((ep.metadata or {}).get("vision_conditions", {}))
    return {}


def _camera_fov(ep: Episode) -> float:
    meta = {}
    if getattr(ep, "vision_config", None):
        meta = ep.vision_config
    if not meta and getattr(ep, "metadata", None):
        meta = (ep.metadata or {}).get("vision_metadata", {})
    intrinsics = meta.get("camera_intrinsics", {}) if meta else {}
    return float(intrinsics.get("fov_deg", 0.0))


def bucket_camera_fov(fov: float) -> str:
    if fov <= 0:
        return "unknown"
    if fov <= 60:
        return "narrow"
    if fov >= 100:
        return "wide"
    return "normal"


def _econ_metrics(econ: EconVector) -> Dict[str, float]:
    return {
        "mpl_units_per_hour": float(econ.mpl_units_per_hour),
        "wage_parity": float(econ.wage_parity),
        "energy_cost": float(econ.energy_cost),
        "reward_scalar_sum": float(econ.reward_scalar_sum),
    }


def _aggregate(records: List[Dict[str, Any]], key_fn: Callable[[Dict[str, Any]], str]) -> Dict[str, Dict[str, float]]:
    buckets: Dict[str, List[Dict[str, float]]] = {}
    for rec in records:
        key = key_fn(rec) or "unknown"
        buckets.setdefault(key, []).append(rec["metrics"])
    summary: Dict[str, Dict[str, float]] = {}
    for key in sorted(buckets.keys()):
        metrics_list = buckets[key]
        if not metrics_list:
            continue
        agg: Dict[str, float] = {}
        all_keys = sorted(metrics_list[0].keys())
        for m in all_keys:
            vals = [float(r.get(m, 0.0)) for r in metrics_list]
            agg[m] = float(sum(vals) / len(vals))
        summary[key] = agg
    return summary


def build_vision_econ_records(episodes: List[Episode], econ_vectors: List[EconVector]) -> List[Dict[str, Any]]:
    econ_map = {e.episode_id: e for e in econ_vectors}
    records: List[Dict[str, Any]] = []
    for ep in episodes:
        econ = econ_map.get(ep.episode_id)
        if econ is None:
            continue
        records.append(
            {
                "episode_id": ep.episode_id,
                "backend": _backend_from_episode(ep),
                "vision_conditions": _vision_conditions(ep),
                "camera_fov": _camera_fov(ep),
                "metrics": _econ_metrics(econ),
            }
        )
    return records


def summarize_vision_econ_correlations(episodes: List[Episode], econ_vectors: List[EconVector]) -> Dict[str, Dict[str, Dict[str, float]]]:
    records = build_vision_econ_records(episodes, econ_vectors)
    return {
        "by_backend": _aggregate(records, lambda r: r.get("backend", "unknown")),
        "by_lighting_tag": _aggregate(records, lambda r: r.get("vision_conditions", {}).get("lighting_tag", "unknown")),
        "by_occlusion_tag": _aggregate(records, lambda r: r.get("vision_conditions", {}).get("occlusion_tag", "unknown")),
        "by_fov_bucket": _aggregate(records, lambda r: bucket_camera_fov(r.get("camera_fov", 0.0))),
    }


def correlation_rows_for_csv(summary: Dict[str, Dict[str, Dict[str, float]]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for group, buckets in summary.items():
        for bucket, metrics in buckets.items():
            for metric, mean_val in metrics.items():
                rows.append(
                    {
                        "group": group,
                        "bucket": bucket,
                        "metric": metric,
                        "mean": mean_val,
                    }
                )
    # Deterministic ordering
    rows.sort(key=lambda r: (r["group"], r["bucket"], r["metric"]))
    return rows
