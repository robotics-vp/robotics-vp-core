"""
EconCorrelator: Statistical correlation between SIMA-2 tags and economic outcomes.

Per SIMA2_ECON_CORRELATION_SPEC.md:
- Computes conditional expectations: E[Damage | Tag]
- Derives TrustMatrix to inform sampling and orchestrator policies
- Advisory-only, does not modify rewards
"""
import json
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_TRUST_MATRIX_PATH = Path(__file__).resolve().parents[2] / "results" / "sima2" / "trust_matrix.json"


@dataclass
class TrustEntry:
    tag: str
    mean_damage: float
    mean_energy: float
    count: int
    trust_score: float = 0.0
    correlation_strength: str = "unknown"
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
        tag_stats = defaultdict(lambda: {"damage": [], "energy": [], "mpl": [], "success": []})

        for dp in datapacks:
            segments = dp.get("segments", [])
            econ = dp.get("econ_vector", {})

            damage = float(econ.get("damage", 0.0))
            energy = float(econ.get("energy_wh", 0.0))
            mpl = float(econ.get("mpl", 1.0))
            success = bool(econ.get("success", True))

            # Aggregate by tag
            for seg in segments:
                if isinstance(seg, dict):
                    seg_meta = seg.get("metadata", {})

                    # RiskTag classification
                    if seg_meta.get("failure_observed"):
                        tag_stats["RiskTag"]["damage"].append(damage)
                        tag_stats["RiskTag"]["energy"].append(energy)
                        tag_stats["RiskTag"]["mpl"].append(mpl)
                        tag_stats["RiskTag"]["success"].append(success)

                    # RecoveryTag
                    if seg_meta.get("recovery_observed"):
                        tag_stats["RecoveryTag"]["damage"].append(damage)
                        tag_stats["RecoveryTag"]["energy"].append(energy)
                        tag_stats["RecoveryTag"]["mpl"].append(mpl)
                        tag_stats["RecoveryTag"]["success"].append(success)

        # Compute TrustEntry for each tag
        trust_matrix = {}
        for tag, stats in tag_stats.items():
            if len(stats["damage"]) < self.min_samples_for_trust:
                continue

            mean_damage = float(np.mean(stats["damage"])) if stats["damage"] else 0.0
            mean_energy = float(np.mean(stats["energy"])) if stats["energy"] else 0.0
            mean_mpl = float(np.mean(stats["mpl"])) if stats["mpl"] else 0.0
            success_rate = float(np.mean([int(s) for s in stats["success"]])) if stats["success"] else 0.0
            count = len(stats["damage"])

            trust_score = self._compute_trust_score(tag, mean_damage, success_rate)
            correlation_strength = self._classify_correlation(trust_score)
            economic_impact = self._describe_impact(tag, mean_damage, mean_mpl, success_rate)
            trust_tier, sampling_multiplier = self._trust_tier_and_multiplier(trust_score)

            trust_matrix[tag] = TrustEntry(
                tag=tag,
                mean_damage=mean_damage,
                mean_energy=mean_energy,
                count=count,
                trust_score=trust_score,
                correlation_strength=correlation_strength,
                economic_impact=economic_impact,
                trust_tier=trust_tier,
                sampling_multiplier=sampling_multiplier,
            ).to_dict()

        return trust_matrix

    def _compute_trust_score(self, tag: str, mean_damage: float, success_rate: float) -> float:
        """Compute trust score (0-1) from correlation strength."""
        if "RiskTag" in tag:
            # High damage or low success → strong signal
            score = min(1.0, (mean_damage / 5.0) + (1.0 - success_rate))
            return max(0.0, min(1.0, score))
        elif "RecoveryTag" in tag:
            # Recovery should improve success
            return max(0.0, min(1.0, success_rate))
        return 0.5

    def _classify_correlation(self, trust_score: float) -> str:
        if trust_score > 0.8:
            return "strong"
        elif trust_score > 0.5:
            return "medium"
        return "weak"

    def _describe_impact(self, tag: str, mean_damage: float, mean_mpl: float, success_rate: float) -> str:
        if "RiskTag" in tag:
            return f"high_damage_predictor"
        elif "RecoveryTag" in tag:
            return f"resilience_marker"
        return "unknown"

    def _trust_tier_and_multiplier(self, trust_score: float) -> Tuple[str, float]:
        if trust_score > 0.8:
            return "trusted", 5.0
        if trust_score > 0.5:
            return "provisional", 1.5
        return "untrusted", 1.0

    def save_trust_matrix(self, trust_matrix: Dict[str, Any], path: Optional[str] = None) -> Path:
        """Save TrustMatrix to JSON."""
        tm_path = Path(path) if path else DEFAULT_TRUST_MATRIX_PATH
        tm_path.parent.mkdir(parents=True, exist_ok=True)
        with tm_path.open("w") as f:
            json.dump(trust_matrix, f, indent=2, sort_keys=True)
        return tm_path
