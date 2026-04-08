"""Economic-input normalization for the sim/synth/physics world model."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from ..common import clip01, mapping, safe_float


def build_economic_input_context(
    economic_context: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    payload = mapping(economic_context)
    urgency_score = clip01(
        payload.get(
            "economic_urgency_score",
            payload.get("urgency_score", payload.get("priority_score", 0.0)),
        )
    )
    value_targets = mapping(
        payload.get("value_target_pack")
        or payload.get("value_targets")
        or payload.get("value_target_summary")
    )
    return {
        **payload,
        "economic_urgency_score": urgency_score,
        "has_value_targets": bool(value_targets),
        "value_target_count": len(value_targets),
        "value_target_keys": sorted(str(key) for key in value_targets),
        "shadow_price_signal": safe_float(
            payload.get("shadow_price_signal", payload.get("value_pressure", 0.0)),
            0.0,
        ),
    }
