"""Advisor interfaces for heuristic and learned shadow runtime paths."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from src.learning.data_value_models import DataValueModel, predict_data_value
from src.learning.pricing_models import PricingDeltaModel, predict_pricing_delta
from src.learning.regal_support_models import RegalSupportModel, predict_regal_support
from src.learning.replay_policy_trainer import load_policy_checkpoint
from src.replay.schema import ReplayEpisodeRecord, ReplayStepRecord
from src.regality.promotion_policy import PromotionMetrics, RegalPromotionPolicy


class AdvisorMode(str, Enum):
    """Supported advisor execution modes."""

    HEURISTIC_ONLY = "heuristic_only"
    LEARNED_ONLY = "learned_only"
    HEURISTIC_LEARNED_RESIDUAL = "heuristic_learned_residual"
    HEURISTIC_LEARNED_COMPARE_ONLY = "heuristic_learned_compare_only"
    LEARNED_COMPARE_ONLY = "learned_compare_only"
    HEURISTIC_PLUS_LEARNED_RESIDUAL = "heuristic_plus_learned_residual"
    LEARNED_SHADOW_ONLY = "learned_shadow_only"
    PROMOTED_GATE_ELIGIBLE = "promoted_gate_eligible"


@dataclass(frozen=True)
class AdvisorPromotionGuard:
    """Resolved execution authority for a learned advisor."""

    requested_mode: str
    effective_mode: str
    gate_eligible: bool
    reasons: list[str]
    node_id: Optional[str] = None
    stage: Optional[str] = None
    calibration_summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "requested_mode": self.requested_mode,
            "effective_mode": self.effective_mode,
            "gate_eligible": bool(self.gate_eligible),
            "reasons": list(self.reasons),
            "node_id": self.node_id,
            "stage": self.stage,
            "calibration_summary": dict(self.calibration_summary),
        }


@dataclass(frozen=True)
class AdvisorResult:
    """Structured advisor output with fallback metadata."""

    advisor_id: str
    mode: str
    model_version: Optional[str]
    config_digest: Optional[str]
    fallback_used: bool
    heuristic_output: Dict[str, Any]
    learned_output: Dict[str, Any]
    applied_output: Dict[str, Any]
    promotion_guard: Dict[str, Any] = field(default_factory=dict)
    inference_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "advisor_id": self.advisor_id,
            "mode": self.mode,
            "model_version": self.model_version,
            "config_digest": self.config_digest,
            "fallback_used": bool(self.fallback_used),
            "heuristic_output": dict(self.heuristic_output),
            "learned_output": dict(self.learned_output),
            "applied_output": dict(self.applied_output),
            "promotion_guard": dict(self.promotion_guard),
            "inference_metadata": dict(self.inference_metadata),
        }


class PolicyAdvisor:
    """Replay policy advisor that scores replay slices with BC error and uncertainty."""

    advisor_id = "policy_advisor"

    def __init__(
        self,
        *,
        mode: AdvisorMode | str = AdvisorMode.HEURISTIC_ONLY,
        checkpoint_path: Optional[str | Path] = None,
        device: Optional[str] = None,
        promotion_policy: Optional[RegalPromotionPolicy] = None,
        promotion_node_id: Optional[str] = None,
        promotion_metrics: Optional[PromotionMetrics] = None,
    ) -> None:
        self.mode = AdvisorMode(mode)
        self.checkpoint_path = str(checkpoint_path) if checkpoint_path else None
        self.device = torch.device(device or "cpu")
        self.model = None
        self.model_config = None
        self._checkpoint_payload: Dict[str, Any] = {}
        if checkpoint_path and Path(checkpoint_path).exists():
            self.model, self.model_config, self._checkpoint_payload = load_policy_checkpoint(checkpoint_path, device=self.device)
        self.guard = _resolve_promotion_guard(
            advisor_id=self.advisor_id,
            requested_mode=self.mode,
            model_available=self.model is not None,
            promotion_policy=promotion_policy,
            promotion_node_id=promotion_node_id,
            promotion_metrics=promotion_metrics,
            preferred_fallback=AdvisorMode.LEARNED_COMPARE_ONLY,
        )

    def summarize_episode(self, records: Sequence[ReplayStepRecord]) -> AdvisorResult:
        heuristic_output = {
            "available": False,
            "reason": "no_heuristic_policy_baseline",
        }
        effective_mode = AdvisorMode(self.guard.effective_mode)
        if effective_mode == AdvisorMode.HEURISTIC_ONLY or self.model is None or self.model_config is None:
            return AdvisorResult(
                advisor_id=self.advisor_id,
                mode=effective_mode.value,
                model_version=None,
                config_digest=None,
                fallback_used=self.model is None,
                heuristic_output=heuristic_output,
                learned_output={},
                applied_output={
                    "available": False,
                    "mean_action_mae": None,
                    "mean_confidence": None,
                    "mean_uncertainty": 1.0,
                },
                promotion_guard=self.guard.to_dict(),
                inference_metadata={"num_steps": len(records)},
            )

        obs = torch.as_tensor([_pad_vector(row.obs_vector, self.model_config.obs_dim) for row in records], dtype=torch.float32, device=self.device)
        condition = torch.as_tensor([_pad_vector(row.condition_vector_values, self.model_config.condition_dim) for row in records], dtype=torch.float32, device=self.device)
        targets = torch.as_tensor([_pad_vector(row.action_vector, self.model_config.action_dim) for row in records], dtype=torch.float32, device=self.device)
        skill_modes = [row.skill_mode for row in records]
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(obs, condition, skill_modes=skill_modes)
        predicted = outputs["action_mean"]
        confidence = outputs["confidence"]
        mae = torch.mean(torch.abs(predicted - targets), dim=-1)
        mean_mae = float(mae.mean().item()) if len(records) else 0.0
        mean_confidence = float(confidence.mean().item()) if len(records) else 0.0
        learned_output = {
            "available": True,
            "mean_action_mae": mean_mae,
            "mean_confidence": mean_confidence,
            "mean_uncertainty": max(0.0, 1.0 - mean_confidence),
            "head_usage": dict(outputs["head_usage"]),
        }
        applied_output = dict(learned_output)
        if effective_mode == AdvisorMode.HEURISTIC_ONLY:
            applied_output = {
                "available": False,
                "mean_action_mae": None,
                "mean_confidence": None,
                "mean_uncertainty": 1.0,
            }
        return AdvisorResult(
            advisor_id=self.advisor_id,
            mode=effective_mode.value,
            model_version=str(self.model_config.model_version),
            config_digest=str(self.model_config.config_digest),
            fallback_used=effective_mode != self.mode,
            heuristic_output=heuristic_output,
            learned_output=learned_output,
            applied_output=applied_output,
            promotion_guard=self.guard.to_dict(),
            inference_metadata={"num_steps": len(records), "checkpoint_path": self.checkpoint_path},
        )


class PricingAdvisor:
    """Shadow advisor over heuristic price ticks plus learned residuals."""

    advisor_id = "pricing_advisor"

    def __init__(
        self,
        *,
        mode: AdvisorMode | str = AdvisorMode.HEURISTIC_ONLY,
        checkpoint_path: Optional[str | Path] = None,
        device: Optional[str] = None,
        promotion_policy: Optional[RegalPromotionPolicy] = None,
        promotion_node_id: Optional[str] = None,
        promotion_metrics: Optional[PromotionMetrics] = None,
    ) -> None:
        self.mode = AdvisorMode(mode)
        self.checkpoint_path = str(checkpoint_path) if checkpoint_path else None
        self.device = torch.device(device or "cpu")
        self.model: Optional[PricingDeltaModel] = None
        self._checkpoint_payload: Dict[str, Any] = {}
        if checkpoint_path and Path(checkpoint_path).exists():
            self.model, self._checkpoint_payload = _load_residual_checkpoint(
                checkpoint_path=checkpoint_path,
                model_class=PricingDeltaModel,
                device=self.device,
            )
        self.guard = _resolve_promotion_guard(
            advisor_id=self.advisor_id,
            requested_mode=self.mode,
            model_available=self.model is not None,
            promotion_policy=promotion_policy,
            promotion_node_id=promotion_node_id or "pricing_truth_regal",
            promotion_metrics=promotion_metrics,
            preferred_fallback=AdvisorMode.HEURISTIC_PLUS_LEARNED_RESIDUAL,
        )

    def assess_episode(self, episode: ReplayEpisodeRecord) -> AdvisorResult:
        heuristic = {
            "net_customer_rate": float(episode.pricing_summary.get("net_customer_rate", 0.0)),
            "task_hour_price_tick": float(episode.pricing_summary.get("task_hour_price_tick", 0.0)),
            "confidence": float(episode.pricing_summary.get("confidence", 0.0)),
        }
        effective_mode = AdvisorMode(self.guard.effective_mode)
        if effective_mode == AdvisorMode.HEURISTIC_ONLY or self.model is None:
            return AdvisorResult(
                advisor_id=self.advisor_id,
                mode=effective_mode.value,
                model_version=None,
                config_digest=None,
                fallback_used=self.model is None and self.mode != AdvisorMode.HEURISTIC_ONLY,
                heuristic_output=heuristic,
                learned_output={},
                applied_output=dict(heuristic),
                promotion_guard=self.guard.to_dict(),
                inference_metadata={"episode_id": episode.episode_id},
            )
        prediction = predict_pricing_delta(self.model, episode, device=self.device)
        predicted_rate = heuristic["net_customer_rate"] + prediction.value
        learned_output = {
            "predicted_residual": float(prediction.value),
            "predicted_net_customer_rate": float(predicted_rate),
            "confidence": float(prediction.confidence),
        }
        applied = dict(heuristic)
        if effective_mode in {
            AdvisorMode.LEARNED_ONLY,
            AdvisorMode.LEARNED_SHADOW_ONLY,
            AdvisorMode.HEURISTIC_LEARNED_RESIDUAL,
            AdvisorMode.HEURISTIC_PLUS_LEARNED_RESIDUAL,
            AdvisorMode.PROMOTED_GATE_ELIGIBLE,
        }:
            applied["net_customer_rate"] = float(max(0.0, predicted_rate))
            applied["confidence"] = float((heuristic["confidence"] + prediction.confidence) / 2.0)
        return AdvisorResult(
            advisor_id=self.advisor_id,
            mode=effective_mode.value,
            model_version=str(self._checkpoint_payload.get("model_version")) if self._checkpoint_payload else None,
            config_digest=str(self._checkpoint_payload.get("config_digest")) if self._checkpoint_payload else None,
            fallback_used=effective_mode != self.mode,
            heuristic_output=heuristic,
            learned_output=learned_output,
            applied_output=applied,
            promotion_guard=self.guard.to_dict(),
            inference_metadata={"episode_id": episode.episode_id, "checkpoint_path": self.checkpoint_path},
        )


class DataValueAdvisor:
    """Shadow advisor for datapack credits and marginal frontier gain."""

    advisor_id = "data_value_advisor"

    def __init__(
        self,
        *,
        mode: AdvisorMode | str = AdvisorMode.HEURISTIC_ONLY,
        checkpoint_path: Optional[str | Path] = None,
        device: Optional[str] = None,
        promotion_policy: Optional[RegalPromotionPolicy] = None,
        promotion_node_id: Optional[str] = None,
        promotion_metrics: Optional[PromotionMetrics] = None,
    ) -> None:
        self.mode = AdvisorMode(mode)
        self.checkpoint_path = str(checkpoint_path) if checkpoint_path else None
        self.device = torch.device(device or "cpu")
        self.model: Optional[DataValueModel] = None
        self._checkpoint_payload: Dict[str, Any] = {}
        if checkpoint_path and Path(checkpoint_path).exists():
            self.model, self._checkpoint_payload = _load_residual_checkpoint(
                checkpoint_path=checkpoint_path,
                model_class=DataValueModel,
                device=self.device,
            )
        self.guard = _resolve_promotion_guard(
            advisor_id=self.advisor_id,
            requested_mode=self.mode,
            model_available=self.model is not None,
            promotion_policy=promotion_policy,
            promotion_node_id=promotion_node_id or "data_value_regal",
            promotion_metrics=promotion_metrics,
            preferred_fallback=AdvisorMode.HEURISTIC_PLUS_LEARNED_RESIDUAL,
        )

    def assess_episode(self, episode: ReplayEpisodeRecord) -> AdvisorResult:
        heuristic = {
            "data_share_credit": float(episode.datapack_summary.get("data_share_credit", 0.0)),
            "marginal_frontier_gain": float(episode.datapack_summary.get("marginal_frontier_gain", 0.0)),
            "quality_score": float(episode.datapack_summary.get("quality_score", 0.0)),
        }
        effective_mode = AdvisorMode(self.guard.effective_mode)
        if effective_mode == AdvisorMode.HEURISTIC_ONLY or self.model is None:
            return AdvisorResult(
                advisor_id=self.advisor_id,
                mode=effective_mode.value,
                model_version=None,
                config_digest=None,
                fallback_used=self.model is None and self.mode != AdvisorMode.HEURISTIC_ONLY,
                heuristic_output=heuristic,
                learned_output={},
                applied_output=dict(heuristic),
                promotion_guard=self.guard.to_dict(),
                inference_metadata={"episode_id": episode.episode_id},
            )
        prediction = predict_data_value(self.model, episode, device=self.device)
        learned_output = {
            "predicted_data_value": float(prediction.value),
            "confidence": float(prediction.confidence),
        }
        applied = dict(heuristic)
        if effective_mode in {
            AdvisorMode.LEARNED_ONLY,
            AdvisorMode.LEARNED_SHADOW_ONLY,
            AdvisorMode.HEURISTIC_LEARNED_RESIDUAL,
            AdvisorMode.HEURISTIC_PLUS_LEARNED_RESIDUAL,
            AdvisorMode.PROMOTED_GATE_ELIGIBLE,
        }:
            applied["data_share_credit"] = float(max(0.0, heuristic["data_share_credit"] + prediction.value))
        return AdvisorResult(
            advisor_id=self.advisor_id,
            mode=effective_mode.value,
            model_version=str(self._checkpoint_payload.get("model_version")) if self._checkpoint_payload else None,
            config_digest=str(self._checkpoint_payload.get("config_digest")) if self._checkpoint_payload else None,
            fallback_used=effective_mode != self.mode,
            heuristic_output=heuristic,
            learned_output=learned_output,
            applied_output=applied,
            promotion_guard=self.guard.to_dict(),
            inference_metadata={"episode_id": episode.episode_id, "checkpoint_path": self.checkpoint_path},
        )


class RegalSupportAdvisor:
    """Learned support scores for typed regal outputs."""

    advisor_id = "regal_support_advisor"

    def __init__(
        self,
        *,
        mode: AdvisorMode | str = AdvisorMode.HEURISTIC_ONLY,
        checkpoint_path: Optional[str | Path] = None,
        device: Optional[str] = None,
        promotion_policy: Optional[RegalPromotionPolicy] = None,
        promotion_node_id: Optional[str] = None,
        promotion_metrics: Optional[PromotionMetrics] = None,
    ) -> None:
        self.mode = AdvisorMode(mode)
        self.checkpoint_path = str(checkpoint_path) if checkpoint_path else None
        self.device = torch.device(device or "cpu")
        self.model: Optional[RegalSupportModel] = None
        self._checkpoint_payload: Dict[str, Any] = {}
        if checkpoint_path and Path(checkpoint_path).exists():
            self.model, self._checkpoint_payload = _load_residual_checkpoint(
                checkpoint_path=checkpoint_path,
                model_class=RegalSupportModel,
                device=self.device,
            )
        self.guard = _resolve_promotion_guard(
            advisor_id=self.advisor_id,
            requested_mode=self.mode,
            model_available=self.model is not None,
            promotion_policy=promotion_policy,
            promotion_node_id=promotion_node_id or "plausibility_regal",
            promotion_metrics=promotion_metrics,
            preferred_fallback=AdvisorMode.LEARNED_COMPARE_ONLY,
        )

    def assess_episode(self, episode: ReplayEpisodeRecord) -> AdvisorResult:
        heuristic = {
            "overall_status": str(episode.regal_summary.get("overall_status", "pass")),
            "deploy_recommendation": str(episode.regal_summary.get("deploy_recommendation", "allow_shadow")),
            "pricing_recommendation": str(episode.regal_summary.get("pricing_recommendation", "publish")),
        }
        effective_mode = AdvisorMode(self.guard.effective_mode)
        if effective_mode == AdvisorMode.HEURISTIC_ONLY or self.model is None:
            return AdvisorResult(
                advisor_id=self.advisor_id,
                mode=effective_mode.value,
                model_version=None,
                config_digest=None,
                fallback_used=self.model is None and self.mode != AdvisorMode.HEURISTIC_ONLY,
                heuristic_output=heuristic,
                learned_output={},
                applied_output=dict(heuristic),
                promotion_guard=self.guard.to_dict(),
                inference_metadata={"episode_id": episode.episode_id},
            )
        prediction = predict_regal_support(self.model, episode, device=self.device)
        learned_output = {
            "anomaly_support_score": float(prediction.value),
            "confidence": float(prediction.confidence),
        }
        applied = dict(heuristic)
        if effective_mode in {
            AdvisorMode.LEARNED_ONLY,
            AdvisorMode.LEARNED_SHADOW_ONLY,
            AdvisorMode.HEURISTIC_LEARNED_RESIDUAL,
            AdvisorMode.HEURISTIC_PLUS_LEARNED_RESIDUAL,
            AdvisorMode.PROMOTED_GATE_ELIGIBLE,
        } and prediction.value > 0.75:
            applied["deploy_recommendation"] = "require_review"
        return AdvisorResult(
            advisor_id=self.advisor_id,
            mode=effective_mode.value,
            model_version=str(self._checkpoint_payload.get("model_version")) if self._checkpoint_payload else None,
            config_digest=str(self._checkpoint_payload.get("config_digest")) if self._checkpoint_payload else None,
            fallback_used=effective_mode != self.mode,
            heuristic_output=heuristic,
            learned_output=learned_output,
            applied_output=applied,
            promotion_guard=self.guard.to_dict(),
            inference_metadata={"episode_id": episode.episode_id, "checkpoint_path": self.checkpoint_path},
        )


def _pad_vector(values: Sequence[float], target_dim: int) -> List[float]:
    payload = [float(value) for value in list(values)[:target_dim]]
    if len(payload) < target_dim:
        payload.extend([0.0] * (target_dim - len(payload)))
    return payload


def _load_residual_checkpoint(
    *,
    checkpoint_path: str | Path,
    model_class: type,
    device: torch.device,
) -> Tuple[Any, Dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = model_class(
        input_dim=int(checkpoint["input_dim"]),
        hidden_dim=int(checkpoint.get("hidden_dim", 64)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, dict(checkpoint)


def _resolve_promotion_guard(
    *,
    advisor_id: str,
    requested_mode: AdvisorMode,
    model_available: bool,
    promotion_policy: Optional[RegalPromotionPolicy],
    promotion_node_id: Optional[str],
    promotion_metrics: Optional[PromotionMetrics],
    preferred_fallback: AdvisorMode,
) -> AdvisorPromotionGuard:
    reasons: list[str] = []
    effective_mode = requested_mode
    gate_eligible = False
    stage = None
    calibration_summary: Dict[str, Any] = {}
    if promotion_metrics and promotion_metrics.calibration_summary is not None:
        calibration_summary = promotion_metrics.calibration_summary.to_dict()

    if requested_mode == AdvisorMode.HEURISTIC_ONLY:
        return AdvisorPromotionGuard(
            requested_mode=requested_mode.value,
            effective_mode=AdvisorMode.HEURISTIC_ONLY.value,
            gate_eligible=False,
            reasons=["heuristic_mode_requested"],
            node_id=promotion_node_id,
            calibration_summary=calibration_summary,
        )

    if not model_available:
        return AdvisorPromotionGuard(
            requested_mode=requested_mode.value,
            effective_mode=AdvisorMode.HEURISTIC_ONLY.value,
            gate_eligible=False,
            reasons=["checkpoint_missing_or_unavailable"],
            node_id=promotion_node_id,
            calibration_summary=calibration_summary,
        )

    if requested_mode == AdvisorMode.HEURISTIC_LEARNED_COMPARE_ONLY:
        effective_mode = AdvisorMode.LEARNED_COMPARE_ONLY
        reasons.append("legacy_compare_mode_alias")
    elif requested_mode == AdvisorMode.HEURISTIC_LEARNED_RESIDUAL:
        effective_mode = AdvisorMode.HEURISTIC_PLUS_LEARNED_RESIDUAL
        reasons.append("legacy_residual_mode_alias")
    elif requested_mode == AdvisorMode.LEARNED_ONLY:
        effective_mode = AdvisorMode.LEARNED_SHADOW_ONLY
        reasons.append("legacy_learned_only_alias")

    if effective_mode == AdvisorMode.PROMOTED_GATE_ELIGIBLE:
        if promotion_policy is None or promotion_node_id is None or promotion_metrics is None or promotion_node_id not in promotion_policy.nodes:
            effective_mode = preferred_fallback
            reasons.append("promotion_context_missing")
        else:
            decision = promotion_policy.evaluate_node(
                promotion_node_id,
                promotion_metrics,
                evidence_pointers={"advisor_id": advisor_id},
            )
            stage = promotion_policy.node_stage(promotion_node_id).value
            if promotion_policy.gate_eligible(promotion_node_id) and decision.outcome != "recommend_demote":
                gate_eligible = True
                reasons.extend(decision.reasons)
            else:
                effective_mode = preferred_fallback
                reasons.append("promotion_criteria_not_met")
                reasons.extend(decision.reasons)

    return AdvisorPromotionGuard(
        requested_mode=requested_mode.value,
        effective_mode=effective_mode.value,
        gate_eligible=gate_eligible,
        reasons=reasons or ["learned_mode_active"],
        node_id=promotion_node_id,
        stage=stage,
        calibration_summary=calibration_summary,
    )
