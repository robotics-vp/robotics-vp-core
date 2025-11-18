"""
Datapack Auditor Policy (Phase G).

Acts as a "Credit Rating Agency" for datapacks before they enter the RL loop.
Predicts economic value and risk based on semantic tags and metadata.
"""
from typing import Any, Dict, List, Optional, Sequence

from src.policies.interfaces import DatapackAuditorPolicy


class HeuristicDatapackAuditor(DatapackAuditorPolicy):
    """
    Deterministic heuristic backend for Datapack Auditor.
    
    Ratings:
    - AAA: High value, low risk, high novelty (Gold standard)
    - AA:  Good value, safe
    - A:   Standard training data
    - BBB: Risky but potentially high reward (Frontier)
    - JUNK: High risk, low value, or incoherent
    """

    def build_features(
        self,
        datapack: Any,
        semantic_tags: Optional[Sequence[Any]] = None,
        econ_slice: Optional[Dict[str, Any]] = None,
        recap_scores: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Build features from datapack and context.
        """
        tags = semantic_tags or []
        econ = econ_slice or {}
        recap = recap_scores or {}
        
        # Helper to get field from object or dict
        def _get(item, key, default=None):
            if isinstance(item, dict):
                return item.get(key, default)
            return getattr(item, key, default)

        # Extract tag summaries
        fragility_tags = [t for t in tags if _get(t, "fragility_level")]
        risk_tags = [t for t in tags if _get(t, "risk_type")]
        recovery_tags = [t for t in tags if _get(t, "intervention_type") == "failure_recovery"]
        ood_tags = [t for t in tags if _get(t, "novelty_type") == "edge_case"]
        
        # Calculate aggregate metrics
        max_fragility = 0.0
        fragility_map = {"low": 0.2, "medium": 0.5, "high": 0.8, "critical": 1.0}
        for t in fragility_tags:
            level = _get(t, "fragility_level", "low")
            max_fragility = max(max_fragility, fragility_map.get(level, 0.0))
            
        max_risk = 0.0
        risk_map = {"low": 0.2, "medium": 0.5, "high": 0.8, "critical": 1.0}
        for t in risk_tags:
            level = _get(t, "severity", "low")
            max_risk = max(max_risk, risk_map.get(level, 0.0))
            
        has_recovery = len(recovery_tags) > 0
        is_ood = len(ood_tags) > 0
        
        datapack_id = _get(datapack, "datapack_id") or _get(datapack, "pack_id") or _get(datapack, "episode_id", "unknown")

        return {
            "datapack_id": datapack_id,
            "max_fragility": max_fragility,
            "max_risk": max_risk,
            "has_recovery": has_recovery,
            "is_ood": is_ood,
            "expected_mpl_gain": econ.get("expected_mpl_gain", 0.0),
            "novelty_score": econ.get("novelty_score", 0.0),
            "recap_quality": recap.get("quality_score", 0.5),
        }

    def evaluate(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate features to produce audit rating and predicted econ.
        """
        # 1. Predict Risk Score (0.0 safe -> 1.0 dangerous)
        base_risk = features["max_risk"]
        if features["max_fragility"] > 0.5:
            base_risk = max(base_risk, features["max_fragility"] * 0.8)
        
        # Recovery demonstrates control over risk, lowering effective risk
        if features["has_recovery"]:
            base_risk *= 0.5
            
        # OOD increases uncertainty/risk
        if features["is_ood"]:
            base_risk = min(1.0, base_risk * 1.2)
            
        predicted_risk_score = base_risk
        
        # 2. Predict Value (MPL Gain)
        base_value = features["expected_mpl_gain"]
        
        # Novelty boosts value
        if features["novelty_score"] > 0.5:
            base_value *= (1.0 + features["novelty_score"])
            
        # High risk reduces realized value (expected failure)
        if predicted_risk_score > 0.7:
            base_value *= 0.5
            
        predicted_delta_mpl = base_value
        
        # 3. Predict Costs
        # Heuristic: High risk = high potential damage
        predicted_damage_cost = predicted_risk_score * 100.0  # $100 max heuristic
        predicted_energy_wh = 10.0 # Baseline
        
        # 4. Assign Rating
        rating = "BBB" # Default
        
        # AAA: High Value, Low Risk, High Quality
        if predicted_delta_mpl > 5.0 and predicted_risk_score < 0.3:
            rating = "AAA"
        # AA: Good Value, Safe
        elif predicted_delta_mpl > 2.0 and predicted_risk_score < 0.4:
            rating = "AA"
        # A: Standard
        elif predicted_risk_score < 0.5:
            rating = "A"
        # JUNK: High Risk, Low Value
        elif predicted_risk_score > 0.8 and predicted_delta_mpl < 2.0:
            rating = "JUNK"
        # BBB: Frontier (High Risk, High Value)
        elif predicted_risk_score > 0.6 and predicted_delta_mpl > 5.0:
            rating = "BBB"
            
        score = max(0.0, predicted_delta_mpl) * (1.0 - min(1.0, predicted_risk_score))

        return {
            "rating": rating,
            "score": score,
            "predicted_econ": {
                "expected_delta_mpl": predicted_delta_mpl,
                "expected_energy_wh": predicted_energy_wh,
                "expected_damage_cost": predicted_damage_cost,
                "risk_score": predicted_risk_score,
            },
            "metadata": {
                "auditor_backend": "heuristic_v1",
                "features_used": list(features.keys()),
                "datapack_id": features.get("datapack_id"),
                "raw_score_basis": {
                    "predicted_delta_mpl": predicted_delta_mpl,
                    "predicted_risk_score": predicted_risk_score,
                },
            }
        }
