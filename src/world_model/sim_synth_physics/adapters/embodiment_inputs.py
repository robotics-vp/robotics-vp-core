"""Embodiment-input normalization for the sim/synth/physics world model."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from ..common import clip01, mapping, safe_float, strings


def build_embodiment_input_context(
    embodiment_context: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    payload = mapping(embodiment_context)
    capability_profile = mapping(
        payload.get("capability_profile")
        or payload.get("capability_summary")
        or payload.get("capabilities")
    )
    control_constraints = mapping(
        payload.get("control_constraints")
        or payload.get("action_constraints")
        or payload.get("latency_envelope")
    )
    active_embodiments = strings(
        payload.get("active_embodiments")
        or payload.get("robot_families")
        or payload.get("target_embodiments")
    )
    return {
        **payload,
        "active_embodiments": active_embodiments,
        "active_embodiment_count": len(active_embodiments),
        "has_capability_profile": bool(capability_profile),
        "capability_profile": capability_profile,
        "control_constraints": control_constraints,
        "latency_budget_ms": safe_float(
            control_constraints.get("latency_budget_ms", payload.get("latency_budget_ms", 0.0)),
            0.0,
        ),
        "contact_risk_score": clip01(
            payload.get("contact_risk_score", control_constraints.get("contact_risk_score", 0.0))
        ),
    }
