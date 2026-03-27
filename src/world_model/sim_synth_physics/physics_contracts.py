"""Physics/backend execution contracts for the sim/synth/physics world model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

from .common import mapping


@dataclass(frozen=True)
class PhysicsExecutionContract:
    """Canonical backend-routing contract for one WM planning window."""

    contract_id: str
    requested_backend: str
    resolved_backend: str
    fidelity_tier: str
    domain_randomization_regime: str
    calibration_profile: str
    backend_selection_policy: str
    adapter_name: str
    route_status: str
    fallback_reason: str = ""
    receipt_kind: str = "physics_execution_contract_v1"
    authority_class: str = "bounded_authority"
    decision_scope: str = "sim_synth_backend_routing"
    reward_math_mutation: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "physics_execution_contract_v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "contract_id": self.contract_id,
            "requested_backend": self.requested_backend,
            "resolved_backend": self.resolved_backend,
            "fidelity_tier": self.fidelity_tier,
            "domain_randomization_regime": self.domain_randomization_regime,
            "calibration_profile": self.calibration_profile,
            "backend_selection_policy": self.backend_selection_policy,
            "adapter_name": self.adapter_name,
            "route_status": self.route_status,
            "fallback_reason": self.fallback_reason,
            "receipt_kind": self.receipt_kind,
            "authority_class": self.authority_class,
            "decision_scope": self.decision_scope,
            "reward_math_mutation": bool(self.reward_math_mutation),
            "metadata": mapping(self.metadata),
            "version": self.version,
        }
