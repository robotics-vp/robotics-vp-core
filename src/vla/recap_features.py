"""
Shared RECAP feature utilities for training/inference.
"""
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_recap_jsonl(paths: List[str]) -> List[Dict]:
    records: List[Dict] = []
    for path in sorted(paths):
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
    # Deterministic ordering by (episode_id, timestep, sampler_strategy)
    records.sort(key=lambda r: (r.get("episode_id", ""), r.get("timestep", 0), r.get("sampler_strategy") or ""))
    return records


def infer_metrics(records: List[Dict], override: List[str] = None) -> List[str]:
    if override:
        return list(override)
    metrics = set()
    for rec in records:
        metrics.update(rec.get("metrics", {}).keys())
    return sorted(metrics)


def collect_categories(records: List[Dict], field: str) -> List[str]:
    vals = sorted({rec.get(field) for rec in records if rec.get(field) is not None})
    return [str(v) for v in vals]


def compute_metric_stats(records: List[Dict], metrics: List[str]) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for m in metrics:
        vals = [float((rec.get("metrics", {}) or {}).get(m, 0.0)) for rec in records]
        vals = vals or [0.0]
        stats[m] = {
            "min": min(vals),
            "max": max(vals),
            "mean": float(sum(vals) / len(vals)),
        }
    return stats


def build_feature_vector(
    rec: Dict,
    metrics: List[str],
    metric_stats: Dict[str, Dict[str, float]],
    categories: Dict[str, List[str]],
) -> List[float]:
    features: List[float] = []
    metric_values = rec.get("metrics", {}) or {}
    for m in metrics:
        stats = metric_stats[m]
        val = float(metric_values.get(m, 0.0))
        rng = stats["max"] - stats["min"]
        norm = 0.0 if rng == 0 else (val - stats["mean"]) / max(rng, 1e-6)
        features.append(norm)

    for field, cats in categories.items():
        one_hot = [0.0] * len(cats)
        if cats:
            val = rec.get(field)
            if val is not None and str(val) in cats:
                idx = cats.index(str(val))
                one_hot[idx] = 1.0
        features.extend(one_hot)
    return features


def quantize_metric(value: float, support: Tuple[float, float], num_atoms: int) -> int:
    lo, hi = support
    if hi == lo:
        return 0
    pos = (value - lo) / max(hi - lo, 1e-6)
    pos = min(max(pos, 0.0), 1.0)
    return int(round(pos * (num_atoms - 1)))


@dataclass
class RecapFeatureConfig:
    metrics: List[str]
    categories: Dict[str, List[str]]
    value_supports: Dict[str, Tuple[float, float]]
    num_atoms: int

    def feature_dim(self) -> int:
        dim = len(self.metrics)
        for cats in self.categories.values():
            dim += len(cats)
        return dim


def summarize_vision_features(policy_features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert PolicyObservationBuilder features into compact recap-ready fields.
    """
    latent = (policy_features.get("vision_latent") or {}).get("latent", [])
    backend = policy_features.get("backend")
    stats: Dict[str, Any] = {"vision_backend": backend}
    if latent:
        arr = np.array(latent, dtype=float)
        stats.update(
            {
                "vision_latent_mean": float(arr.mean()),
                "vision_latent_min": float(arr.min()),
                "vision_latent_max": float(arr.max()),
            }
        )
    if policy_features.get("state_digest"):
        stats["vision_state_digest"] = policy_features["state_digest"]
    return stats
