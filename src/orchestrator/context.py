from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np

from src.valuation.datapack_repo import DataPackRepo
from src.valuation.datapack_schema import DataPackMeta
from src.valuation.energy_response_model import load_energy_interventions


@dataclass
class OrchestratorContext:
    env_name: str
    engine_type: str
    task_type: str
    customer_segment: str
    market_region: str
    objective_vector: List[float]  # [w_mpl, w_error, w_energy, w_safety, w_novelty]
    wage_human: float
    energy_price_kWh: float
    mean_delta_mpl: float
    mean_delta_error: float
    mean_delta_j: float
    mean_trust: float
    mean_w_econ: float
    profile_summaries: Dict[str, Dict[str, float]]  # profile -> {mpl, error, energy_Wh, risk}


@dataclass
class OrchestratorResult:
    steps: List
    chosen_backend: str
    energy_profile_weights: Dict[str, float]
    objective_preset: str
    data_mix_weights: Dict[str, float]
    expected_delta_mpl: float
    expected_delta_error: float
    expected_delta_energy_Wh: float


def _aggregate_datapacks(dps: List[DataPackMeta]) -> Dict[str, float]:
    if not dps:
        return {
            "mean_delta_mpl": 0.0,
            "mean_delta_error": 0.0,
            "mean_delta_j": 0.0,
            "mean_trust": 0.0,
            "mean_w_econ": 0.0,
        }
    delta_mpl = np.mean([dp.attribution.delta_mpl for dp in dps])
    delta_error = np.mean([dp.attribution.delta_error for dp in dps])
    delta_j = np.mean([dp.attribution.delta_J for dp in dps])
    trust = np.mean([dp.attribution.trust_score for dp in dps])
    w_econ = np.mean([dp.attribution.w_econ for dp in dps])
    return {
        "mean_delta_mpl": float(delta_mpl),
        "mean_delta_error": float(delta_error),
        "mean_delta_j": float(delta_j),
        "mean_trust": float(trust),
        "mean_w_econ": float(w_econ),
    }


def _profile_summaries_from_interventions(interventions_path: str) -> Dict[str, Dict[str, float]]:
    if not interventions_path:
        return {}
    samples = load_energy_interventions(interventions_path)
    profiles: Dict[str, Dict[str, List[float]]] = {}
    for s in samples:
        profiles.setdefault(s.profile_name, {"mpl": [], "error": [], "energy_Wh": [], "risk": []})
        profiles[s.profile_name]["mpl"].append(s.mpl)
        profiles[s.profile_name]["error"].append(s.error_rate)
        profiles[s.profile_name]["energy_Wh"].append(s.energy_Wh)
        profiles[s.profile_name]["risk"].append(s.risk_metric)
    summaries = {}
    for name, vals in profiles.items():
        summaries[name] = {
            "mpl": float(np.mean(vals["mpl"])) if vals["mpl"] else 0.0,
            "error": float(np.mean(vals["error"])) if vals["error"] else 0.0,
            "energy_Wh": float(np.mean(vals["energy_Wh"])) if vals["energy_Wh"] else 0.0,
            "risk": float(np.mean(vals["risk"])) if vals["risk"] else 0.0,
        }
    return summaries


def build_orchestrator_context_from_datapacks(
    base_dir: str,
    env_name: str,
    engine_type: str,
    task_type: str,
    customer_segment: str,
    market_region: str,
    interventions_path: Optional[str] = None,
) -> OrchestratorContext:
    repo = DataPackRepo(base_dir=base_dir)
    datapacks = list(repo.iter_all(env_name) or [])
    agg = _aggregate_datapacks(datapacks)

    # Pull a representative objective profile if available
    objective_vector = [0.6, 0.2, 0.15, 0.05, 0.0]
    wage_human = 0.0
    energy_price_kWh = 0.0
    if datapacks:
        for dp in datapacks:
            if dp.objective_profile:
                op = dp.objective_profile
                objective_vector = op.objective_vector
                wage_human = getattr(op, "wage_human", 0.0)
                energy_price_kWh = getattr(op, "energy_price_kWh", 0.0)
                break

    profile_summaries = _profile_summaries_from_interventions(interventions_path) if interventions_path else {}

    return OrchestratorContext(
        env_name=env_name,
        engine_type=engine_type,
        task_type=task_type,
        customer_segment=customer_segment,
        market_region=market_region,
        objective_vector=objective_vector,
        wage_human=wage_human,
        energy_price_kWh=energy_price_kWh,
        mean_delta_mpl=agg["mean_delta_mpl"],
        mean_delta_error=agg["mean_delta_error"],
        mean_delta_j=agg["mean_delta_j"],
        mean_trust=agg["mean_trust"],
        mean_w_econ=agg["mean_w_econ"],
        profile_summaries=profile_summaries,
    )
