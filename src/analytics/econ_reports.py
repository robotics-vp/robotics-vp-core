"""
Economics reports over ontology artifacts.
"""
import math
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.ontology.store import OntologyStore
from src.ontology.models import Task, Episode, EconVector, Datapack
from src.policies.registry import build_all_policies


def _percentiles(values: List[float], ps=(10, 50, 90)) -> Dict[str, float]:
    if not values:
        return {f"p{p}": 0.0 for p in ps}
    values_sorted = sorted(values)
    out = {}
    for p in ps:
        idx = int((p / 100) * (len(values_sorted) - 1))
        out[f"p{p}"] = float(values_sorted[idx])
    return out


def _quality_grade(score: float) -> str:
    if score >= 0.8:
        return "GRADE_5"
    if score >= 0.5:
        return "GRADE_3"
    return "FAILED"


def _extract_segmentation_boundaries(seg_record: Any) -> List[Dict[str, Any]]:
    if not seg_record:
        return []
    if isinstance(seg_record, dict):
        if "segment_boundary_tags" in seg_record:
            return seg_record.get("segment_boundary_tags", []) or []
        if "enrichment" in seg_record:
            return seg_record.get("enrichment", {}).get("segment_boundary_tags", []) or []
    return []


def _boundary_reason(boundary: Any) -> str:
    if isinstance(boundary, dict):
        return str(boundary.get("reason", "")).lower()
    return str(getattr(boundary, "reason", "")).lower()


def compute_task_econ_summary(
    store: OntologyStore,
    task_id: str,
    reward_model_scores: Optional[Dict[str, Any]] = None,
    segmentation_tags: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    task = store.get_task(task_id)
    if not task:
        return {"task_id": task_id, "error": "task_not_found"}
    episodes = store.list_episodes(task_id=task_id)
    econ_vectors = store.list_econ_vectors()
    ev_map = {e.episode_id: e for e in econ_vectors}
    mpl_vals: List[float] = []
    wage_parity_vals: List[float] = []
    energy_costs: List[float] = []
    damage_costs: List[float] = []
    rewards: List[float] = []
    mobility_penalties: List[float] = []
    precision_bonuses: List[float] = []
    stability_risks: List[float] = []
    sampler_counts: Dict[str, int] = {}
    phase_counts: Dict[str, int] = {}
    successes = 0
    failures = 0
    quality_adjusted_mpl: List[float] = []
    error_adjusted_energy: List[float] = []
    grade_counts: Dict[str, int] = {}
    recovery_fractions: List[float] = []
    recovery_present = 0
    mpl_without_recovery: List[float] = []

    rm_scores = reward_model_scores or {}
    seg_map = segmentation_tags or {}

    for ep in episodes:
        ev = ev_map.get(ep.episode_id)
        if ev:
            mpl_vals.append(ev.mpl_units_per_hour)
            wage_parity_vals.append(ev.wage_parity)
            energy_costs.append(ev.energy_cost)
            damage_costs.append(ev.damage_cost)
            rewards.append(ev.reward_scalar_sum)
            mobility_penalties.append(getattr(ev, "mobility_penalty", 0.0))
            precision_bonuses.append(getattr(ev, "precision_bonus", 0.0))
            stability_risks.append(getattr(ev, "stability_risk_score", 0.0))
            rm = rm_scores.get(ep.episode_id, {})
            qual = float(rm.get("quality_score", rm.get("quality", 0.0))) if rm else 0.0
            err_prob = float(rm.get("error_probability", 0.0)) if rm else 0.0
            if rm:
                quality_adjusted_mpl.append(ev.mpl_units_per_hour * max(0.0, min(1.0, qual)))
                error_adjusted_energy.append(ev.energy_cost * (1.0 + max(0.0, min(1.0, err_prob))))
                grade = _quality_grade(qual)
                grade_counts[grade] = grade_counts.get(grade, 0) + 1
            seg_rec = seg_map.get(ep.episode_id, {})
            boundaries = _extract_segmentation_boundaries(seg_rec)
            seg_ids = {b.get("segment_id", "") if isinstance(b, dict) else getattr(b, "segment_id", "") for b in boundaries}
            rec_events = [b for b in boundaries if _boundary_reason(b) in {"recovery", "failure"}]
            frac = (len(rec_events) / max(len(seg_ids), 1)) if seg_ids else 0.0
            if rec_events:
                recovery_present += 1
            recovery_fractions.append(frac)
            if ev:
                mpl_without_recovery.append(ev.mpl_units_per_hour * (1.0 - frac))
        if ep.status == "success":
            successes += 1
        elif ep.status == "failure":
            failures += 1
        md = getattr(ep, "metadata", {}) or {}
        strat = md.get("sampling_metadata", {}).get("strategy") if isinstance(md, dict) else None
        phase = md.get("curriculum_phase") if isinstance(md, dict) else None
        if strat:
            sampler_counts[strat] = sampler_counts.get(strat, 0) + 1
        if phase:
            phase_counts[phase] = phase_counts.get(phase, 0) + 1

    summary = {
        "task": {
            "task_id": task.task_id,
            "name": task.name,
            "human_mpl_units_per_hour": task.human_mpl_units_per_hour,
            "human_wage_per_hour": task.human_wage_per_hour,
            "default_energy_cost_per_wh": task.default_energy_cost_per_wh,
        },
        "counts": {
            "episodes": len(episodes),
            "econ_vectors": len(econ_vectors),
            "successes": successes,
            "failures": failures,
        },
        "mpl": {
            "mean": float(sum(mpl_vals) / len(mpl_vals)) if mpl_vals else 0.0,
            **_percentiles(mpl_vals),
        },
        "wage_parity": {
            "mean": float(sum(wage_parity_vals) / len(wage_parity_vals)) if wage_parity_vals else 0.0,
            **_percentiles(wage_parity_vals),
        },
        "energy_cost_dist": {
            "mean": float(sum(energy_costs) / len(energy_costs)) if energy_costs else 0.0,
            **_percentiles(energy_costs),
        },
        "damage_cost_dist": {
            "mean": float(sum(damage_costs) / len(damage_costs)) if damage_costs else 0.0,
            **_percentiles(damage_costs),
        },
        "energy_cost": {"mean": float(sum(energy_costs) / len(energy_costs)) if energy_costs else 0.0},
        "damage_cost": {"mean": float(sum(damage_costs) / len(damage_costs)) if damage_costs else 0.0},
        "reward_scalar_sum": {"mean": float(sum(rewards) / len(rewards)) if rewards else 0.0},
        "sampler_counts": sampler_counts,
        "curriculum_phase_counts": phase_counts,
    }
    if mobility_penalties:
        summary["mobility_penalty"] = {"mean": float(sum(mobility_penalties) / len(mobility_penalties)), **_percentiles(mobility_penalties)}
    if precision_bonuses:
        summary["precision_bonus"] = {"mean": float(sum(precision_bonuses) / len(precision_bonuses)), **_percentiles(precision_bonuses)}
    if stability_risks:
        summary["stability_risk_score"] = {"mean": float(sum(stability_risks) / len(stability_risks)), **_percentiles(stability_risks)}
    if quality_adjusted_mpl:
        summary["quality_adjusted_mpl"] = {
            "mean": float(sum(quality_adjusted_mpl) / len(quality_adjusted_mpl)),
            **_percentiles(quality_adjusted_mpl),
        }
        summary["quality_grades"] = dict(sorted(grade_counts.items(), key=lambda kv: kv[0]))
    if error_adjusted_energy:
        summary["error_adjusted_energy_cost"] = {
            "mean": float(sum(error_adjusted_energy) / len(error_adjusted_energy)),
            **_percentiles(error_adjusted_energy),
        }
    if recovery_fractions:
        summary["recovery_segments"] = {
            "fraction_with_recovery": float(recovery_present / max(len(recovery_fractions), 1)),
            "mean_recovery_fraction": float(sum(recovery_fractions) / len(recovery_fractions)),
            "mpl_without_recovery_mean": float(sum(mpl_without_recovery) / len(mpl_without_recovery))
            if mpl_without_recovery
            else 0.0,
        }
    return summary


def compute_datapack_mix_summary(store: OntologyStore, task_id: str) -> Dict[str, Any]:
    datapacks = store.list_datapacks(task_id=task_id)
    by_source: Dict[str, List[Datapack]] = {}
    for d in datapacks:
        by_source.setdefault(d.source_type, []).append(d)
    mix = {}
    for src, dps in by_source.items():
        novelty = [dp.novelty_score for dp in dps]
        quality = [dp.quality_score for dp in dps]
        mix[src] = {
            "count": len(dps),
            "avg_novelty": float(sum(novelty) / len(novelty)) if novelty else 0.0,
            "avg_quality": float(sum(quality) / len(quality)) if quality else 0.0,
        }
    def _safe_dt(dt):
        if isinstance(dt, datetime):
            return dt
        try:
            return datetime.fromisoformat(str(dt))
        except Exception:
            return datetime.fromtimestamp(0)

    recent = sorted(datapacks, key=lambda d: _safe_dt(d.created_at), reverse=True)[:5] if datapacks else []
    return {
        "task_id": task_id,
        "sources": mix,
        "recent": [
            {
                "datapack_id": dp.datapack_id,
                "source_type": dp.source_type,
                "modality": dp.modality,
                "created_at": _safe_dt(dp.created_at).isoformat(),
            }
            for dp in recent
        ],
    }


def compute_pricing_snapshot(
    store: OntologyStore,
    task_id: str,
    reward_model_scores: Optional[Dict[str, Any]] = None,
    segmentation_tags: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    task_summary = compute_task_econ_summary(store, task_id, reward_model_scores=reward_model_scores, segmentation_tags=segmentation_tags)
    dp_summary = compute_datapack_mix_summary(store, task_id)
    policies = build_all_policies()
    datapacks = store.list_datapacks(task_id=task_id)
    valuations = []
    for dp in datapacks:
        feats = policies.data_valuation.build_features(dp)
        score = policies.data_valuation.score(feats)
        valuations.append(float(score.get("valuation_score", 0.0)))
    datapack_value = sum(valuations) / len(valuations) if valuations else 0.0

    pricing_features = policies.pricing.build_features(
        task_econ=task_summary,
        datapack_value=datapack_value,
        semantic_context={"datapack_mix": dp_summary},
    )
    pricing = policies.pricing.evaluate(pricing_features)
    metadata = pricing.get("metadata", {}) if isinstance(pricing, dict) else {}
    return {
        "task_id": task_id,
        "human_unit_cost": metadata.get("human_unit_cost", 0.0),
        "robot_unit_cost": metadata.get("robot_unit_cost", 0.0),
        "implied_spread_per_unit": metadata.get("spread_per_unit", 0.0),
        "datapack_price_floor": metadata.get("datapack_price_floor", 0.0),
        "task_summary": task_summary,
        "datapack_mix": dp_summary,
    }
