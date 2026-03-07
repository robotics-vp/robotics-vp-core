"""Advisor interfaces for heuristic and learned shadow runtime paths."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from src.learning.data_value_models import DataValueModel, predict_data_value
from src.learning.pricing_models import PricingDeltaModel, predict_pricing_delta
from src.learning.regal_support_models import RegalSupportModel, predict_regal_support
from src.learning.replay_policy_trainer import load_policy_checkpoint
from src.replay.schema import ReplayEpisodeRecord, ReplayStepRecord
from src.utils.config_digest import sha256_json


class AdvisorMode(str, Enum):
    """Supported advisor execution modes."""

    HEURISTIC_ONLY = "heuristic_only"
    LEARNED_ONLY = "learned_only"
    HEURISTIC_LEARNED_RESIDUAL = "heuristic_learned_residual"
    HEURISTIC_LEARNED_COMPARE_ONLY = "heuristic_learned_compare_only"


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
    ) -> None:
        self.mode = AdvisorMode(mode)
        self.checkpoint_path = str(checkpoint_path) if checkpoint_path else None
        self.device = torch.device(device or "cpu")
        self.model = None
        self.model_config = None
        self._checkpoint_payload: Dict[str, Any] = {}
        if checkpoint_path and Path(checkpoint_path).exists():
            self.model, self.model_config, self._checkpoint_payload = load_policy_checkpoint(checkpoint_path, device=self.device)

    def summarize_episode(self, records: Sequence[ReplayStepRecord]) -> AdvisorResult:
        heuristic_output = {
            "available": False,
            "reason": "no_heuristic_policy_baseline",
        }
        if self.mode == AdvisorMode.HEURISTIC_ONLY or self.model is None or self.model_config is None:
            return AdvisorResult(
                advisor_id=self.advisor_id,
                mode=self.mode.value,
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
        return AdvisorResult(
            advisor_id=self.advisor_id,
            mode=self.mode.value,
            model_version=str(self.model_config.model_version),
            config_digest=str(self.model_config.config_digest),
            fallback_used=False,
            heuristic_output=heuristic_output,
            learned_output=learned_output,
            applied_output=dict(learned_output),
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

    def assess_episode(self, episode: ReplayEpisodeRecord) -> AdvisorResult:
        heuristic = {
            "net_customer_rate": float(episode.pricing_summary.get("net_customer_rate", 0.0)),
            "task_hour_price_tick": float(episode.pricing_summary.get("task_hour_price_tick", 0.0)),
            "confidence": float(episode.pricing_summary.get("confidence", 0.0)),
        }
        if self.mode == AdvisorMode.HEURISTIC_ONLY or self.model is None:
            return AdvisorResult(
                advisor_id=self.advisor_id,
                mode=self.mode.value,
                model_version=None,
                config_digest=None,
                fallback_used=self.model is None and self.mode != AdvisorMode.HEURISTIC_ONLY,
                heuristic_output=heuristic,
                learned_output={},
                applied_output=dict(heuristic),
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
        if self.mode in {AdvisorMode.LEARNED_ONLY, AdvisorMode.HEURISTIC_LEARNED_RESIDUAL}:
            applied["net_customer_rate"] = float(max(0.0, predicted_rate))
            applied["confidence"] = float((heuristic["confidence"] + prediction.confidence) / 2.0)
        return AdvisorResult(
            advisor_id=self.advisor_id,
            mode=self.mode.value,
            model_version=str(self._checkpoint_payload.get("model_version")) if self._checkpoint_payload else None,
            config_digest=str(self._checkpoint_payload.get("config_digest")) if self._checkpoint_payload else None,
            fallback_used=False,
            heuristic_output=heuristic,
            learned_output=learned_output,
            applied_output=applied,
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

    def assess_episode(self, episode: ReplayEpisodeRecord) -> AdvisorResult:
        heuristic = {
            "data_share_credit": float(episode.datapack_summary.get("data_share_credit", 0.0)),
            "marginal_frontier_gain": float(episode.datapack_summary.get("marginal_frontier_gain", 0.0)),
            "quality_score": float(episode.datapack_summary.get("quality_score", 0.0)),
        }
        if self.mode == AdvisorMode.HEURISTIC_ONLY or self.model is None:
            return AdvisorResult(
                advisor_id=self.advisor_id,
                mode=self.mode.value,
                model_version=None,
                config_digest=None,
                fallback_used=self.model is None and self.mode != AdvisorMode.HEURISTIC_ONLY,
                heuristic_output=heuristic,
                learned_output={},
                applied_output=dict(heuristic),
                inference_metadata={"episode_id": episode.episode_id},
            )
        prediction = predict_data_value(self.model, episode, device=self.device)
        learned_output = {
            "predicted_data_value": float(prediction.value),
            "confidence": float(prediction.confidence),
        }
        applied = dict(heuristic)
        if self.mode in {AdvisorMode.LEARNED_ONLY, AdvisorMode.HEURISTIC_LEARNED_RESIDUAL}:
            applied["data_share_credit"] = float(max(0.0, heuristic["data_share_credit"] + prediction.value))
        return AdvisorResult(
            advisor_id=self.advisor_id,
            mode=self.mode.value,
            model_version=str(self._checkpoint_payload.get("model_version")) if self._checkpoint_payload else None,
            config_digest=str(self._checkpoint_payload.get("config_digest")) if self._checkpoint_payload else None,
            fallback_used=False,
            heuristic_output=heuristic,
            learned_output=learned_output,
            applied_output=applied,
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

    def assess_episode(self, episode: ReplayEpisodeRecord) -> AdvisorResult:
        heuristic = {
            "overall_status": str(episode.regal_summary.get("overall_status", "pass")),
            "deploy_recommendation": str(episode.regal_summary.get("deploy_recommendation", "allow_shadow")),
            "pricing_recommendation": str(episode.regal_summary.get("pricing_recommendation", "publish")),
        }
        if self.mode == AdvisorMode.HEURISTIC_ONLY or self.model is None:
            return AdvisorResult(
                advisor_id=self.advisor_id,
                mode=self.mode.value,
                model_version=None,
                config_digest=None,
                fallback_used=self.model is None and self.mode != AdvisorMode.HEURISTIC_ONLY,
                heuristic_output=heuristic,
                learned_output={},
                applied_output=dict(heuristic),
                inference_metadata={"episode_id": episode.episode_id},
            )
        prediction = predict_regal_support(self.model, episode, device=self.device)
        learned_output = {
            "anomaly_support_score": float(prediction.value),
            "confidence": float(prediction.confidence),
        }
        applied = dict(heuristic)
        if self.mode in {AdvisorMode.LEARNED_ONLY, AdvisorMode.HEURISTIC_LEARNED_RESIDUAL} and prediction.value > 0.75:
            applied["deploy_recommendation"] = "require_review"
        return AdvisorResult(
            advisor_id=self.advisor_id,
            mode=self.mode.value,
            model_version=str(self._checkpoint_payload.get("model_version")) if self._checkpoint_payload else None,
            config_digest=str(self._checkpoint_payload.get("config_digest")) if self._checkpoint_payload else None,
            fallback_used=False,
            heuristic_output=heuristic,
            learned_output=learned_output,
            applied_output=applied,
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
