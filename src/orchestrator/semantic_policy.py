from __future__ import annotations

from typing import Mapping, Sequence

from src.ontology.store import OntologyStore


def apply_arh_penalty(
    econ_metrics: Mapping[str, float],
    *,
    suspicious_key: str = "anti_reward_hacking_suspicious",
    penalty_factor: float = 0.5,
) -> dict[str, float]:
    metrics = dict(econ_metrics)
    suspicious = metrics.get(suspicious_key, 0.0)
    try:
        suspicious_val = float(suspicious)
    except (TypeError, ValueError):
        suspicious_val = 1.0 if suspicious else 0.0
    if suspicious_val <= 0.0:
        return metrics

    mpl = metrics.get("mpl_units_per_hour", 0.0)
    try:
        mpl_val = float(mpl)
    except (TypeError, ValueError):
        mpl_val = 0.0
    adjusted = mpl_val * max(0.0, min(1.0, 1.0 - penalty_factor))
    metrics["mpl_units_per_hour_adjusted"] = adjusted
    metrics["anti_reward_hacking_penalty"] = penalty_factor
    return metrics


def detect_semantic_gaps(store: OntologyStore, required_tags: Sequence[str]) -> list[str]:
    required = {str(tag).strip().lower() for tag in required_tags if str(tag).strip()}
    if not required:
        return []
    observed: set[str] = set()
    for scenario in store.list_scenarios():
        for tag in scenario.get("datapack_tags") or []:
            if tag:
                observed.add(str(tag).strip().lower())
    return sorted(required - observed)
