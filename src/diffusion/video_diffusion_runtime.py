"""Runtime/provider contract for governed video diffusion planning and bring-up."""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from src.evidence.provider_truth import build_external_provider_truth

from .real_video_diffusion_stub import (
    DiffusionProposal,
    SyntheticEpisodeProposal,
    VideoDiffusionStub,
)


def _normalize_backend_policy(value: Optional[str]) -> str:
    policy = str(value or "auto").strip().lower()
    if policy not in {"auto", "real", "disabled", "stub"}:
        return "auto"
    return policy


def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def _infer_model_family(model_ref: str) -> str:
    normalized = str(model_ref or "").strip().lower()
    if "ltx" in normalized:
        return "ltx_video"
    if "cogvideo" in normalized:
        return "cogvideox"
    if "wan" in normalized:
        return "wan"
    if "svd" in normalized or "stable-video" in normalized:
        return "stable_video_diffusion"
    if "opensora" in normalized:
        return "opensora"
    return "unknown"


@dataclass(frozen=True)
class VideoDiffusionRuntimeConfig:
    """Configuration for the governed video diffusion runtime."""

    model_ref: str = ""
    device: str = "cuda"
    backend_policy: str = "auto"


@dataclass(frozen=True)
class VideoDiffusionRuntimeStatus:
    """Resolved runtime/provider truth for video diffusion planning."""

    provider_truth: Dict[str, Any]
    model_ref: str = ""
    model_family: str = ""
    materialization_mode: str = "plan_only"
    planning_backend: str = "governed_diffusion_router_v1"
    version: str = "video_diffusion_runtime_status_v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider_truth": dict(self.provider_truth),
            "model_ref": self.model_ref,
            "model_family": self.model_family,
            "materialization_mode": self.materialization_mode,
            "planning_backend": self.planning_backend,
            "version": self.version,
        }


class VideoDiffusionRuntime:
    """Governed diffusion planner with explicit real-or-unavailable provider truth."""

    def __init__(
        self,
        config: Optional[VideoDiffusionRuntimeConfig] = None,
        *,
        planner: Optional[VideoDiffusionStub] = None,
    ) -> None:
        self._config = config or VideoDiffusionRuntimeConfig()
        self._planner = planner or VideoDiffusionStub()
        self._pipeline: Any = None
        self._status = self._resolve_status()

    @property
    def backend_policy(self) -> str:
        return _normalize_backend_policy(self._config.backend_policy)

    @property
    def model_ref(self) -> str:
        env_ref = os.environ.get("VIDEO_DIFFUSION_MODEL_NAME") or os.environ.get(
            "VIDEO_DIFFUSION_MODEL_REF", ""
        )
        return str(self._config.model_ref or env_ref or "")

    def provider_truth(self) -> Dict[str, Any]:
        return dict(self._status.provider_truth)

    def status(self) -> Dict[str, Any]:
        return self._status.to_dict()

    def _provider_metadata(self) -> Dict[str, Any]:
        return {
            "device": str(self._config.device or "cuda"),
            "requires_companion_gpu": True,
            "diffusers_available": _has_module("diffusers"),
            "torch_available": _has_module("torch"),
            "planning_backend": "governed_diffusion_router_v1",
            "oss_candidates": [
                "Lightricks/LTX-Video",
                "THUDM/CogVideoX",
                "Wan-Video/Wan2.1",
                "stabilityai/stable-video-diffusion-img2vid",
            ],
        }

    def _try_load_real_pipeline(self, model_ref: str) -> tuple[Any, str]:
        if not model_ref:
            return None, "model_ref_missing"
        if not _has_module("torch"):
            return None, "torch_unavailable"
        if not _has_module("diffusers"):
            return None, "diffusers_unavailable"
        try:
            import torch
            from diffusers import DiffusionPipeline
        except Exception as exc:  # pragma: no cover - import failure path
            return None, f"diffusion_runtime_import_error:{exc}"

        if str(self._config.device).startswith("cuda") and not torch.cuda.is_available():
            return None, "cuda_unavailable"

        try:
            pipeline = DiffusionPipeline.from_pretrained(model_ref, local_files_only=True)
            if hasattr(pipeline, "to"):
                pipeline.to(self._config.device)
            return pipeline, ""
        except Exception as exc:
            return None, f"diffusers_local_load_error:{exc}"

    def _resolve_status(self) -> VideoDiffusionRuntimeStatus:
        policy = self.backend_policy
        model_ref = self.model_ref
        model_family = _infer_model_family(model_ref)
        metadata = self._provider_metadata()
        metadata["model_ref"] = model_ref
        metadata["model_family"] = model_family

        if policy == "disabled":
            provider_truth = build_external_provider_truth(
                provider_id="video_diffusion_runtime",
                provider_kind="video_diffusion",
                provider_name="governed_video_diffusion",
                available=False,
                backend_selected="disabled",
                fallback_mode="disabled",
                calibration_class="not_applicable",
                grounding_class="not_applicable",
                metadata={**metadata, "materialization_mode": "disabled"},
            )
            return VideoDiffusionRuntimeStatus(
                provider_truth=provider_truth,
                model_ref=model_ref,
                model_family=model_family,
                materialization_mode="disabled",
            )

        if policy == "stub":
            provider_truth = build_external_provider_truth(
                provider_id="video_diffusion_runtime",
                provider_kind="video_diffusion",
                provider_name="governed_video_diffusion",
                available=False,
                backend_selected="stub",
                fallback_mode="explicit_stub_requested",
                calibration_class="synthetic_stub",
                grounding_class="not_applicable",
                metadata={**metadata, "materialization_mode": "plan_only"},
            )
            return VideoDiffusionRuntimeStatus(
                provider_truth=provider_truth,
                model_ref=model_ref,
                model_family=model_family,
                materialization_mode="plan_only",
            )

        pipeline, failure_reason = self._try_load_real_pipeline(model_ref)
        if pipeline is not None:
            self._pipeline = pipeline
            provider_truth = build_external_provider_truth(
                provider_id="video_diffusion_runtime",
                provider_kind="video_diffusion",
                provider_name="governed_video_diffusion",
                available=True,
                backend_selected="real",
                fallback_mode="",
                calibration_class="runtime_package_loaded",
                grounding_class="not_applicable",
                confidence=1.0,
                metadata={**metadata, "materialization_mode": "diffusers_local_pipeline"},
            )
            return VideoDiffusionRuntimeStatus(
                provider_truth=provider_truth,
                model_ref=model_ref,
                model_family=model_family,
                materialization_mode="diffusers_local_pipeline",
            )

        if policy == "real":
            raise RuntimeError(
                "Video diffusion runtime unavailable "
                f"(policy=real, model_ref={model_ref or 'missing'}, reason={failure_reason or 'unknown'})"
            )

        provider_truth = build_external_provider_truth(
            provider_id="video_diffusion_runtime",
            provider_kind="video_diffusion",
            provider_name="governed_video_diffusion",
            available=False,
            backend_selected="heuristic_fallback",
            fallback_mode="heuristic_planning_only",
            calibration_class="routing_only",
            grounding_class="not_applicable",
            metadata={
                **metadata,
                "materialization_mode": "plan_only",
                "failure_reason": failure_reason or "real_provider_unavailable",
            },
        )
        return VideoDiffusionRuntimeStatus(
            provider_truth=provider_truth,
            model_ref=model_ref,
            model_family=model_family,
            materialization_mode="plan_only",
        )

    def _annotate_proposal(self, proposal: DiffusionProposal) -> DiffusionProposal:
        provider_truth = self.provider_truth()
        constraint_set = dict(proposal.constraint_set or {})
        constraint_set["diffusion_provider_truth"] = provider_truth
        proposal.constraint_set = constraint_set
        proposal.diffusion_provider_truth = provider_truth
        proposal.diffusion_backend_selected = str(provider_truth.get("backend_selected", "heuristic_fallback"))
        proposal.diffusion_backend_policy = self.backend_policy
        proposal.diffusion_model_ref = self.model_ref
        proposal.diffusion_materialization_mode = self._status.materialization_mode
        return proposal

    def propose_augmented_clips(
        self,
        *,
        episode_id: str,
        media_refs: list[str],
        semantic_tags: list[str],
        objective_preset: str = "balanced",
        energy_profile: str = "BASE",
        econ_context: Optional[Dict[str, float]] = None,
        constraint_set: Optional[Dict[str, Any]] = None,
        hypotheses: Optional[list[Dict[str, Any]]] = None,
        routing_context: Optional[Dict[str, Any]] = None,
        num_proposals: int = 3,
    ) -> list[DiffusionProposal]:
        if self.backend_policy == "disabled":
            return []
        proposals = self._planner.propose_augmented_clips(
            episode_id=episode_id,
            media_refs=media_refs,
            semantic_tags=semantic_tags,
            objective_preset=objective_preset,
            energy_profile=energy_profile,
            econ_context=econ_context,
            constraint_set=constraint_set,
            hypotheses=hypotheses,
            routing_context=routing_context,
            num_proposals=num_proposals,
        )
        return [self._annotate_proposal(proposal) for proposal in proposals]

    def propose_synthetic_episode(
        self,
        *,
        source_episode_id: str,
        semantic_tags: list[str],
        objective_preset: str = "balanced",
        energy_profile: str = "BASE",
        econ_context: Optional[Dict[str, float]] = None,
        constraint_set: Optional[Dict[str, Any]] = None,
    ) -> SyntheticEpisodeProposal:
        episode = self._planner.propose_synthetic_episode(
            source_episode_id=source_episode_id,
            semantic_tags=semantic_tags,
            objective_preset=objective_preset,
            energy_profile=energy_profile,
            econ_context=econ_context,
            constraint_set=constraint_set,
        )
        episode.diffusion_proposals = [
            self._annotate_proposal(proposal) for proposal in list(episode.diffusion_proposals)
        ]
        return episode


__all__ = [
    "VideoDiffusionRuntime",
    "VideoDiffusionRuntimeConfig",
    "VideoDiffusionRuntimeStatus",
]
