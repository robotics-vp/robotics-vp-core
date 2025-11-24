"""
EconCorrelator: Statistical correlation between SIMA-2 tags and economic outcomes.

Per SIMA2_ECON_CORRELATION_SPEC.md:
- Computes conditional expectations: E[Damage | Tag]
- Derives TrustMatrix to inform sampling and orchestrator policies
- Advisory-only, does not modify rewards
"""
from collections import defaultdict
from dataclasses import dataclass, asdict
import json
import math
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_TRUST_MATRIX_PATH = Path(__file__).resolve().parents[2] / "results" / "sima2" / "trust_matrix.json"
TAG_TYPES = ("RiskTag", "OODTag", "RecoveryTag", "FragilityTag")
SIGMOID_K = 3.0


@dataclass
class TrustEntry:
    tag: str
    mean_damage: float
    mean_energy: float
    mean_mpl: float
    mean_error_rate: float
    count: int
    trust_score: float = 0.0
    correlation_strength: float = 0.0
    correlation_strength_label: str = "unknown"
    economic_impact: str = "unknown"
    trust_tier: str = "untrusted"
    sampling_multiplier: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EconCorrelator:
    """
    Compute statistical correlations between Stage 2 tags and Stage 3 EconVectors.

    Metrics:
    - RiskPremium: E[Damage | RiskTag=High] / E[Damage | RiskTag=Low]
    - RecoveryValue: E[MPL | RecoveryTag=True] - E[MPL | RecoveryTag=False]
    - OOD Penalty: E[SuccessRate | OODTag=High]
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.risk_premium_target = float(self.config.get("risk_premium_target", 2.0))
        self.min_samples_for_trust = int(self.config.get("min_samples_for_trust", 10))

    def compute_correlations(self, datapacks: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Compute tag-to-econ correlations from datapacks.

        Input format:
        - datapacks: List of dicts with:
          - segments: [Segment dicts with metadata]
          - econ_vector: {"damage": float, "mpl": float, "energy_wh": float, "success": bool}

        Output:
        - TrustMatrix: Dict[tag_name, TrustEntry.to_dict()]
        """
        accum = defaultdict(lambda: {"damage": [], "energy": [], "mpl": [], "error_rate": []})
        indicators: Dict[str, List[int]] = {tag: [] for tag in TAG_TYPES}
        damages_all: List[float] = []

        for dp in datapacks:
            segment_dicts = self._normalize_segments(dp.get("segments", []))
            metrics = self._extract_metrics(dp)
            damages_all.append(metrics["damage"])

            for tag in TAG_TYPES:
                indicator = int(self._tag_indicator(tag, dp, segment_dicts))
                indicators[tag].append(indicator)
                if indicator:
                    accum[tag]["damage"].append(metrics["damage"])
                    accum[tag]["energy"].append(metrics["energy"])
                    accum[tag]["mpl"].append(metrics["mpl"])
                    accum[tag]["error_rate"].append(metrics["error_rate"])

        trust_matrix: Dict[str, Dict[str, Any]] = {}
        for tag in TAG_TYPES:
            stats = accum[tag]
            count = len(stats["damage"])
            mean_damage = float(np.mean(stats["damage"])) if stats["damage"] else 0.0
            mean_energy = float(np.mean(stats["energy"])) if stats["energy"] else 0.0
            mean_mpl = float(np.mean(stats["mpl"])) if stats["mpl"] else 0.0
            mean_error_rate = float(np.mean(stats["error_rate"])) if stats["error_rate"] else 0.0

            corr_strength = 0.0
            if (
                len(damages_all) >= 2
                and len(indicators[tag]) == len(damages_all)
                and 0 < sum(indicators[tag]) < len(indicators[tag])
                and count >= self.min_samples_for_trust
            ):
                corr_strength = float(abs(self._pearson_corr(damages_all, indicators[tag])))

            trust_score = float(self._sigmoid(SIGMOID_K * corr_strength))
            if count < self.min_samples_for_trust:
                trust_score = 0.0
            trust_tier = self._trust_tier(trust_score)
            sampling_multiplier = self._sampling_multiplier(trust_score)

            trust_matrix[tag] = TrustEntry(
                tag=tag,
                mean_damage=mean_damage,
                mean_energy=mean_energy,
                mean_mpl=mean_mpl,
                mean_error_rate=mean_error_rate,
                count=count,
                trust_score=trust_score,
                correlation_strength=corr_strength,
                correlation_strength_label=self._classify_correlation(corr_strength),
                economic_impact=self._describe_impact(tag, mean_damage, mean_mpl, mean_error_rate),
                trust_tier=trust_tier,
                sampling_multiplier=sampling_multiplier,
            ).to_dict()

        # Persist artifact for downstream consumers
        self.save_trust_matrix(trust_matrix)
        return trust_matrix

    def _sigmoid(self, x: float) -> float:
        try:
            return float(1.0 / (1.0 + math.exp(-x)))
        except OverflowError:
            return 0.0 if x < 0 else 1.0

    def _pearson_corr(self, xs: List[float], ys: List[int]) -> float:
        if len(xs) != len(ys) or len(xs) < 2:
            return 0.0
        x_arr = np.asarray(xs, dtype=float)
        y_arr = np.asarray(ys, dtype=float)
        if np.allclose(x_arr, x_arr[0]) or np.allclose(y_arr, y_arr[0]):
            return 0.0
        corr = np.corrcoef(x_arr, y_arr)[0, 1]
        if np.isnan(corr):
            return 0.0
        return float(corr)

    def _classify_correlation(self, corr_strength: float) -> str:
        if corr_strength > 0.5:
            return "strong"
        if corr_strength > 0.25:
            return "medium"
        if corr_strength > 0.0:
            return "weak"
        return "none"

    def _describe_impact(self, tag: str, mean_damage: float, mean_mpl: float, mean_error_rate: float) -> str:
        if "RiskTag" in tag:
            return "high_damage_predictor" if mean_damage > 0 else "low_damage"
        if "RecoveryTag" in tag:
            return "resilience_marker" if mean_error_rate < 0.5 else "noisy_recovery"
        if "OODTag" in tag:
            return "distribution_shift_signal"
        if "FragilityTag" in tag:
            return "fragile_interaction_marker"
        return "unknown"

    def _sampling_multiplier(self, trust_score: float) -> float:
        if trust_score > 0.8:
            return 5.0
        if trust_score > 0.5:
            return 1.5
        return 1.0

    def _trust_tier(self, trust_score: float) -> str:
        if trust_score > 0.8:
            return "trusted"
        if trust_score > 0.5:
            return "provisional"
        return "untrusted"

    def _normalize_segments(self, segments: List[Any]) -> List[Dict[str, Any]]:
        normed: List[Dict[str, Any]] = []
        for seg in segments or []:
            if hasattr(seg, "to_dict"):
                seg_dict = seg.to_dict()
            elif isinstance(seg, dict):
                seg_dict = dict(seg)
            else:
                seg_dict = getattr(seg, "__dict__", {}) or {}
            meta = seg_dict.get("metadata") or {}
            if not isinstance(meta, dict):
                meta = {}
            seg_dict["metadata"] = meta
            normed.append(seg_dict)
        return normed

    def _extract_metrics(self, datapack: Dict[str, Any]) -> Dict[str, float]:
        econ = datapack.get("econ_vector", {}) or {}
        damage = float(econ.get("damage", econ.get("damage_cost", 0.0)))
        mpl = float(econ.get("mpl", econ.get("marginal_product_of_labor", 0.0)))
        energy = float(econ.get("energy_wh", econ.get("energy_cost_wh", econ.get("energy", 0.0))))
        success = bool(econ.get("success", econ.get("episode_success", True)))
        error_rate = float(econ.get("error_rate", 1.0 - float(success)))
        return {
            "damage": damage,
            "mpl": mpl,
            "energy": energy,
            "error_rate": error_rate,
        }

    def _tag_indicator(self, tag: str, datapack: Dict[str, Any], segments: List[Dict[str, Any]]) -> bool:
        tag_lower = tag.lower()
        tags = [str(t).lower() for t in datapack.get("tags", []) or []]

        # Direct tag hints on datapack
        if tag_lower.startswith("risk") and any("risk" in t for t in tags):
            return True
        if tag_lower.startswith("recovery") and any("recovery" in t for t in tags):
            return True
        if tag_lower.startswith("ood") and any("ood" in t for t in tags):
            return True
        if tag_lower.startswith("fragility") and any("fragile" in t for t in tags):
            return True

        enrichment = datapack.get("enrichment", {}) or {}
        if tag_lower.startswith("ood") and enrichment.get("ood_tags"):
            return True
        if tag_lower.startswith("recovery") and enrichment.get("recovery_tags"):
            return True
        if tag_lower.startswith("fragility") and enrichment.get("fragility_tags"):
            return True
        if tag_lower.startswith("risk") and enrichment.get("risk_tags"):
            return True

        for seg in segments:
            meta = seg.get("metadata", {}) or {}
            label = str(seg.get("label") or "").lower()
            outcome = str(seg.get("outcome") or "").lower()
            meta_tags = [str(t).lower() for t in meta.get("tags", []) or []]
            if tag_lower.startswith("risk"):
                if meta.get("failure_observed") or outcome == "failure" or meta.get("risk_level", "").lower() in {"high", "critical"}:
                    return True
            if tag_lower.startswith("recovery"):
                if meta.get("recovery_observed") or outcome in {"recovery", "recovered"}:
                    return True
            if tag_lower.startswith("fragility"):
                if meta.get("fragile_interaction") or "fragile" in meta_tags or "fragile" in label:
                    return True
            if tag_lower.startswith("ood"):
                if meta.get("ood_detected") or float(meta.get("ood_score", 0.0)) > 0.0 or any("ood" in t for t in meta_tags):
                    return True
        return False

    def save_trust_matrix(self, trust_matrix: Dict[str, Any], path: Optional[str] = None) -> Path:
        """Save TrustMatrix to JSON."""
        tm_path = Path(path) if path else DEFAULT_TRUST_MATRIX_PATH
        tm_path.parent.mkdir(parents=True, exist_ok=True)
        with tm_path.open("w") as f:
            json.dump(trust_matrix, f, indent=2, sort_keys=True)
        return tm_path
