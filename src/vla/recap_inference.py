"""
RECAP inference utilities: load trained heads, score ontology episodes.

Deterministic, JSON-safe, and read-only; does not alter rewards or policies.
"""
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn

from src.vla.recap_heads import (
    AdvantageConditioningConfig,
    AdvantageConditioningHead,
    DistributionalValueConfig,
    DistributionalValueHead,
)
from src.vla.recap_features import (
    RecapFeatureConfig,
    build_feature_vector,
    compute_metric_stats,
    set_seeds,
)
from src.ontology.models import EpisodeEvent, EconVector
from src.utils.json_safe import to_json_safe


@dataclass
class RecapEpisodeScores:
    episode_id: str
    advantage_bin_probs_mean: List[float]
    advantage_bin_probs_max: List[float]
    metric_distributions: Dict[str, List[float]]
    recap_goodness_score: float
    num_events: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return to_json_safe(asdict(self))


def _build_feature_records(
    events: List[EpisodeEvent],
    econ: EconVector,
    metadata: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    md = metadata or {}
    reward_mean = 0.0
    if events:
        reward_mean = sum(float(e.reward_scalar) for e in events) / float(len(events))
    records: List[Dict[str, Any]] = []
    for ev in events:
        records.append(
            {
                "episode_id": ev.episode_id,
                "timestep": ev.timestep,
                "advantage": float(ev.reward_scalar) - reward_mean,
                "metrics": ev.reward_components,
                "sampler_strategy": md.get("sampling_metadata", {}).get("strategy"),
                "curriculum_phase": md.get("curriculum_phase"),
                "objective_preset": md.get("objective_preset"),
            }
        )
    return records


def _scores_from_logits(
    adv_logits: torch.Tensor,
    value_logits: torch.Tensor,
    metrics: List[str],
    num_atoms: int,
    value_supports: Dict[str, Tuple[float, float]],
) -> Tuple[List[float], List[float], Dict[str, List[float]], float]:
    probs = torch.softmax(adv_logits, dim=1)  # [N, bins]
    adv_mean = probs.mean(dim=0).cpu().tolist()
    adv_max = probs.max(dim=0).values.cpu().tolist()

    value_logits = value_logits.view(adv_logits.shape[0], len(metrics), num_atoms)
    value_probs = torch.softmax(value_logits, dim=2)
    metric_distributions: Dict[str, List[float]] = {}
    recap_score = 0.0
    for mi, metric in enumerate(metrics):
        dist = value_probs[:, mi, :].mean(dim=0).cpu().tolist()
        metric_distributions[metric] = [float(x) for x in dist]
        # Expected value for goodness score
        support = value_supports.get(metric, (0.0, 1.0))
        lin = torch.linspace(support[0], support[1], steps=num_atoms)
        expected = float((value_probs[:, mi, :] * lin.to(value_probs.device)).sum(dim=1).mean().item())
        if metric.lower().startswith("mpl"):
            recap_score += expected
        elif "energy" in metric.lower():
            recap_score -= 0.5 * expected
        elif "error" in metric.lower() or "damage" in metric.lower():
            recap_score -= expected
    return adv_mean, adv_max, metric_distributions, float(recap_score)


@dataclass
class RecapHeadsBundle:
    adv_head: AdvantageConditioningHead
    value_head: DistributionalValueHead
    feature_config: RecapFeatureConfig
    advantage_bins: List[float]
    metrics: List[str]
    num_atoms: int

    def device(self) -> torch.device:
        return next(self.adv_head.parameters()).device


def load_recap_heads(checkpoint_path: str, device: str = "cpu") -> RecapHeadsBundle:
    ckpt = torch.load(checkpoint_path, map_location=device)
    advantage_bins: List[float] = ckpt["advantage_bins"]
    metrics: List[str] = ckpt["metrics"]
    num_atoms: int = ckpt["num_atoms"]
    categories: Dict[str, List[str]] = ckpt.get("categories", {})
    value_supports: Dict[str, Tuple[float, float]] = ckpt.get("value_supports", {m: (0.0, 1.0) for m in metrics})
    feature_config = RecapFeatureConfig(metrics=metrics, categories=categories, value_supports=value_supports, num_atoms=num_atoms)
    adv_config = AdvantageConditioningConfig(advantage_bins=advantage_bins)
    val_config = DistributionalValueConfig(metrics=metrics, num_atoms=num_atoms, value_supports=value_supports)
    hidden_dim = int(ckpt.get("hidden_dim", 64))
    feature_dim = ckpt.get("feature_dim", feature_config.feature_dim())
    adv_head = AdvantageConditioningHead(adv_config, input_dim=feature_dim, hidden_dim=hidden_dim).to(device)
    val_head = DistributionalValueHead(val_config, input_dim=feature_dim, hidden_dim=hidden_dim).to(device)

    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else {}
    adv_state = {k.replace("adv_head.", ""): v for k, v in state.items() if k.startswith("adv_head.")}
    val_state = {k.replace("value_head.", ""): v for k, v in state.items() if k.startswith("value_head.")}
    if adv_state:
        adv_head.load_state_dict(adv_state, strict=False)
    if val_state:
        val_head.load_state_dict(val_state, strict=False)
    bundle = RecapHeadsBundle(
        adv_head=adv_head,
        value_head=val_head,
        feature_config=feature_config,
        advantage_bins=advantage_bins,
        metrics=metrics,
        num_atoms=num_atoms,
    )
    return bundle


def compute_recap_scores(
    bundle: RecapHeadsBundle,
    events: List[EpisodeEvent],
    econ: Optional[EconVector],
    metadata: Optional[Dict[str, Any]] = None,
    seed: int = 0,
) -> RecapEpisodeScores:
    set_seeds(seed)
    if not events:
        raise ValueError("No events provided for RECAP scoring.")
    records = _build_feature_records(events, econ, metadata)
    metric_stats = compute_metric_stats(records, bundle.feature_config.metrics)
    features = [build_feature_vector(r, bundle.feature_config.metrics, metric_stats, bundle.feature_config.categories) for r in records]

    device = bundle.device()
    feats_tensor = torch.tensor(features, dtype=torch.float32, device=device)
    adv_logits = bundle.adv_head(feats_tensor)
    value_logits = bundle.value_head(feats_tensor)
    adv_mean, adv_max, metric_dists, recap_score = _scores_from_logits(
        adv_logits, value_logits, bundle.metrics, bundle.num_atoms, bundle.feature_config.value_supports
    )
    return RecapEpisodeScores(
        episode_id=events[0].episode_id,
        advantage_bin_probs_mean=[float(x) for x in adv_mean],
        advantage_bin_probs_max=[float(x) for x in adv_max],
        metric_distributions=metric_dists,
        recap_goodness_score=float(recap_score),
        num_events=len(events),
        metadata={"value_supports": bundle.feature_config.value_supports},
    )
