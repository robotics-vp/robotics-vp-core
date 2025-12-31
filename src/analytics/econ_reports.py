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


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _arh_penalty_from_components(components: Dict[str, float], default_penalty: float = 0.5) -> float:
    if not components:
        return 0.0
    if "mpl_units_per_hour_adjusted" in components:
        return 0.0
    penalty = _safe_float(components.get("anti_reward_hacking_penalty"))
    if penalty > 0.0:
        return max(0.0, min(1.0, penalty))
    suspicious = _safe_float(components.get("anti_reward_hacking_suspicious"))
    if suspicious > 0.0:
        return max(0.0, min(1.0, default_penalty))
    return 0.0


def _adjusted_mpl(ev: EconVector) -> float:
    components = ev.components or {}
    if "mpl_units_per_hour_adjusted" in components:
        return _safe_float(components.get("mpl_units_per_hour_adjusted"))
    penalty = _arh_penalty_from_components(components)
    if penalty <= 0.0:
        return ev.mpl_units_per_hour
    return ev.mpl_units_per_hour * max(0.0, 1.0 - penalty)


def _adjusted_wage_parity(ev: EconVector, adjusted_mpl: float) -> float:
    if ev.mpl_units_per_hour <= 0.0:
        return ev.wage_parity
    scale = adjusted_mpl / max(ev.mpl_units_per_hour, 1e-6)
    return ev.wage_parity * scale


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
    datapacks = store.list_datapacks(task_id=task_id)
    datapack_map = {d.datapack_id: d for d in datapacks}
    mpl_vals: List[float] = []
    mpl_raw_vals: List[float] = []
    wage_parity_vals: List[float] = []
    wage_parity_raw_vals: List[float] = []
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
    datapack_rating_counts: Dict[str, int] = {}
    econ_by_rating: Dict[str, Dict[str, List[float]]] = {}

    rm_scores = reward_model_scores or {}
    seg_map = segmentation_tags or {}

    for dp in datapacks:
        rating = getattr(dp, "auditor_rating", None)
        if rating:
            datapack_rating_counts[rating] = datapack_rating_counts.get(rating, 0) + 1

    for ep in episodes:
        ev = ev_map.get(ep.episode_id)
        if ev:
            mpl_val = _adjusted_mpl(ev)
            wage_parity_val = _adjusted_wage_parity(ev, mpl_val)
            mpl_vals.append(mpl_val)
            mpl_raw_vals.append(ev.mpl_units_per_hour)
            wage_parity_vals.append(wage_parity_val)
            wage_parity_raw_vals.append(ev.wage_parity)
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
                quality_adjusted_mpl.append(mpl_val * max(0.0, min(1.0, qual)))
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
                mpl_without_recovery.append(mpl_val * (1.0 - frac))
        if ep.status == "success":
            successes += 1
        elif ep.status == "failure":
            failures += 1
        dp_ref = datapack_map.get(getattr(ep, "datapack_id", None))
        dp_rating = getattr(dp_ref, "auditor_rating", None) if dp_ref else None
        if dp_rating and ev:
            bucket = econ_by_rating.setdefault(
                dp_rating,
                {"mpl": [], "damage": [], "energy": [], "wage_parity": [], "novelty_delta": [], "score": []},
            )
            adj_mpl = _adjusted_mpl(ev)
            bucket["mpl"].append(adj_mpl)
            bucket["damage"].append(ev.damage_cost)
            bucket["energy"].append(ev.energy_cost)
            bucket["wage_parity"].append(_adjusted_wage_parity(ev, adj_mpl))
            bucket["novelty_delta"].append(ev.novelty_delta)
            dp_score = getattr(dp_ref, "auditor_score", None)
            if dp_score is not None:
                try:
                    bucket["score"].append(float(dp_score))
                except Exception:
                    pass
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
    if mpl_raw_vals:
        summary["mpl_raw"] = {
            "mean": float(sum(mpl_raw_vals) / len(mpl_raw_vals)),
            **_percentiles(mpl_raw_vals),
        }
    if wage_parity_raw_vals:
        summary["wage_parity_raw"] = {
            "mean": float(sum(wage_parity_raw_vals) / len(wage_parity_raw_vals)),
            **_percentiles(wage_parity_raw_vals),
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
    if datapack_rating_counts:
        total_dp = sum(datapack_rating_counts.values())
        shares = {k: (v / total_dp if total_dp else 0.0) for k, v in datapack_rating_counts.items()}
        summary["auditor"] = {
            "datapack_ratings": {"counts": dict(sorted(datapack_rating_counts.items())), "shares": dict(sorted(shares.items()))},
        }
    if econ_by_rating:
        corr = {}
        for rating, vals in econ_by_rating.items():
            count = len(vals.get("mpl", []))
            corr[rating] = {
                "count": count,
                "mean_mpl": float(sum(vals["mpl"]) / count) if count else 0.0,
                "mean_damage_cost": float(sum(vals["damage"]) / count) if count else 0.0,
                "mean_energy_cost": float(sum(vals["energy"]) / count) if count else 0.0,
                "mean_wage_parity": float(sum(vals["wage_parity"]) / count) if count else 0.0,
                "mean_novelty_delta": float(sum(vals["novelty_delta"]) / count) if count else 0.0,
            }
            scores = vals.get("score", [])
            if scores:
                corr[rating]["mean_auditor_score"] = float(sum(scores) / len(scores))
        summary.setdefault("auditor", {})
        summary["auditor"]["econ_by_rating"] = dict(sorted(corr.items(), key=lambda kv: kv[0]))
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
    rating_counts: Dict[str, int] = {}
    for dp in datapacks:
        rating = getattr(dp, "auditor_rating", None)
        if rating:
            rating_counts[rating] = rating_counts.get(rating, 0) + 1
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
        "auditor_ratings": {
            "counts": dict(sorted(rating_counts.items())),
            "shares": {
                k: (v / sum(rating_counts.values()) if rating_counts else 0.0) for k, v in sorted(rating_counts.items())
            },
        }
        if rating_counts
        else {},
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
