"""Uncertainty calibration receipts for Phase-6 transport scaffolds."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

from src.utils.config_digest import sha256_json
from src.utils.json_safe import to_json_safe
from src.world_model.transport.bridge_contracts import WMTransportBridgeContract

WM_TRANSPORT_UNCERTAINTY_CALIBRATION_VERSION = "wm_transport_uncertainty_calibration_v1"


def _mapping(payload: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    return dict(to_json_safe(dict(payload or {})))


@dataclass(frozen=True)
class WMTransportUncertaintyCalibration:
    """Calibration proxy for one transport bridge contract."""

    calibration_id: str
    contract_id: str
    bridge_key: str
    source_confidence: float
    target_confidence: float
    ece_proxy: float
    brier_proxy: float
    calibration_score: float
    authority_class: str = "transport_uncertainty_calibration_only"
    training_executed: bool = False
    promotion_eligible: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = WM_TRANSPORT_UNCERTAINTY_CALIBRATION_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "calibration_id": self.calibration_id,
            "version": self.version,
            "contract_id": self.contract_id,
            "bridge_key": self.bridge_key,
            "source_confidence": float(self.source_confidence),
            "target_confidence": float(self.target_confidence),
            "ece_proxy": float(self.ece_proxy),
            "brier_proxy": float(self.brier_proxy),
            "calibration_score": float(self.calibration_score),
            "authority_class": self.authority_class,
            "training_executed": bool(self.training_executed),
            "promotion_eligible": bool(self.promotion_eligible),
            "metadata": _mapping(self.metadata),
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "WMTransportUncertaintyCalibration":
        return cls(
            calibration_id=str(payload.get("calibration_id", "")),
            contract_id=str(payload.get("contract_id", "")),
            bridge_key=str(payload.get("bridge_key", "")),
            source_confidence=float(payload.get("source_confidence", 0.0)),
            target_confidence=float(payload.get("target_confidence", 0.0)),
            ece_proxy=float(payload.get("ece_proxy", 1.0)),
            brier_proxy=float(payload.get("brier_proxy", 1.0)),
            calibration_score=float(payload.get("calibration_score", 0.0)),
            authority_class=str(
                payload.get("authority_class", "transport_uncertainty_calibration_only")
            ),
            training_executed=bool(payload.get("training_executed", False)),
            promotion_eligible=bool(payload.get("promotion_eligible", False)),
            metadata=_mapping(payload.get("metadata")),
            version=str(
                payload.get("version", WM_TRANSPORT_UNCERTAINTY_CALIBRATION_VERSION)
            ),
        )


def calibrate_wm_transport_uncertainty(
    contract: WMTransportBridgeContract,
) -> WMTransportUncertaintyCalibration:
    source_confidence = max(0.0, min(1.0, contract.uncertainty_profile.confidence))
    target_confidence = 1.0 if contract.target_endpoint.transformer_id else 0.0
    ece_proxy = abs(source_confidence - target_confidence)
    brier_proxy = (1.0 - source_confidence) ** 2
    calibration_score = max(0.0, min(1.0, 1.0 - 0.5 * ece_proxy - 0.5 * brier_proxy))
    payload = {
        "contract_id": contract.contract_id,
        "bridge_key": contract.bridge_key,
        "source_confidence": source_confidence,
        "target_confidence": target_confidence,
    }
    return WMTransportUncertaintyCalibration(
        calibration_id=f"wm_transport_uncertainty_{sha256_json(payload)[:16]}",
        contract_id=contract.contract_id,
        bridge_key=contract.bridge_key,
        source_confidence=source_confidence,
        target_confidence=target_confidence,
        ece_proxy=ece_proxy,
        brier_proxy=brier_proxy,
        calibration_score=calibration_score,
        metadata={
            "calibration_required": contract.uncertainty_profile.calibration_required,
            "training_claim": False,
        },
    )


__all__ = [
    "WM_TRANSPORT_UNCERTAINTY_CALIBRATION_VERSION",
    "WMTransportUncertaintyCalibration",
    "calibrate_wm_transport_uncertainty",
]
