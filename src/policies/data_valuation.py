"""
Heuristic DataValuationPolicy wrapper around existing datapack fields.

Preserves current valuation/quality scores by forwarding through stored
datapack metadata (trust_score, w_econ, quality_score) without altering any
reward math or sampling behavior.
"""
from typing import Any, Dict, Optional, Sequence

from src.policies.interfaces import DataValuationPolicy
from src.utils.json_safe import to_json_safe


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val)
    except Exception:
        return default


def _as_dict(obj: Any) -> Dict[str, Any]:
    if hasattr(obj, "to_dict"):
        try:
            return obj.to_dict()
        except Exception:
            pass
    return obj if isinstance(obj, dict) else {}


class HeuristicDataValuationPolicy(DataValuationPolicy):
    def __init__(self, trust_matrix: Dict[str, Any] = None):
        self.trust_matrix = trust_matrix or {}

    def build_features(
        self,
        datapack: Any,
        econ_slice: Optional[Dict[str, Any]] = None,
        semantic_tags: Optional[Sequence[Any]] = None,
        recap_scores: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        dp_dict = _as_dict(datapack)
        attribution = _as_dict(getattr(datapack, "attribution", dp_dict.get("attribution", {})))
        pack_id = dp_dict.get("pack_id") or getattr(datapack, "pack_id", None) or getattr(datapack, "datapack_id", None)
        recap_score = None
        if recap_scores and pack_id:
            recap_score = _safe_float(recap_scores.get(pack_id, {}).get("recap_goodness_score"))

        features: Dict[str, Any] = {
            "pack_id": pack_id,
            "delta_mpl": _safe_float(attribution.get("delta_mpl")),
            "delta_error": _safe_float(attribution.get("delta_error")),
            "delta_ep": _safe_float(attribution.get("delta_ep")),
            "delta_J": _safe_float(attribution.get("delta_J")),
            "trust_score": _safe_float(attribution.get("trust_score"), 0.0),
            "w_econ": _safe_float(attribution.get("w_econ"), 0.0),
            "quality_score": _safe_float(dp_dict.get("quality_score", getattr(datapack, "quality_score", 0.0))),
            "novelty_score": _safe_float(dp_dict.get("novelty_score", getattr(datapack, "novelty_score", 0.0))),
            "semantic_tags": list(dp_dict.get("semantic_tags", getattr(datapack, "semantic_tags", []) or [])),
            "recap_goodness_score": recap_score,
            "econ_slice": econ_slice or {},
        }
        if semantic_tags:
            features["semantic_tags"] = [to_json_safe(getattr(t, "to_dict", lambda: t)()) for t in semantic_tags]
        features["trust_matrix_score"] = self._semantic_trust(features.get("semantic_tags", []))
        return to_json_safe(features)

    def score(self, features: Dict[str, Any]) -> Dict[str, Any]:
        quality_score = _safe_float(features.get("quality_score"))
        trust_score = _safe_float(features.get("trust_score"), 0.0)
        w_econ = _safe_float(features.get("w_econ"), 0.0)
        valuation = quality_score if quality_score is not None else trust_score * w_econ
        econ_slice = features.get("econ_slice") or {}
        mobility_penalty = _safe_float(econ_slice.get("mobility_penalty", {}).get("mean") if isinstance(econ_slice, dict) else 0.0)
        if mobility_penalty:
            valuation = max(0.0, valuation - 0.1 * mobility_penalty)
        trust_matrix_score = _safe_float(features.get("trust_matrix_score"), 1.0)
        valuation = valuation * max(trust_matrix_score, 0.1)
        valuation = _safe_float(valuation)
        metadata = {
            "novelty_score": _safe_float(features.get("novelty_score")),
            "delta_mpl": _safe_float(features.get("delta_mpl")),
            "delta_error": _safe_float(features.get("delta_error")),
            "delta_J": _safe_float(features.get("delta_J")),
            "recap_goodness_score": _safe_float(features.get("recap_goodness_score")),
            "semantic_tags": features.get("semantic_tags", []),
            "trust_matrix_score": trust_matrix_score,
        }
        return {
            "valuation_score": valuation,
            "metadata": to_json_safe(metadata),
        }

    def _semantic_trust(self, semantic_tags: Sequence[Any]) -> float:
        scores = []
        for tag in semantic_tags or []:
            tag_type = None
            if isinstance(tag, dict):
                tag_type = tag.get("tag_type") or tag.get("type")
            else:
                tag_type = getattr(tag, "tag_type", None)
            if tag_type and tag_type in self.trust_matrix:
                try:
                    scores.append(float(self.trust_matrix[tag_type].get("trust_score", 0.0)))
                except Exception:
                    continue
        if not scores:
            return float(self.trust_matrix.get("_default", {}).get("trust_score", 1.0)) if isinstance(self.trust_matrix, dict) else 1.0
        return sum(scores) / len(scores)
