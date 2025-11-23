"""
Canonical construction of PolicyObservation from VisionFrame + low-dim state.

Required VisionFrame fields:
- backend (str), task_id (str), episode_id (str), timestep (int)
- optional rgb/depth/segmentation paths and camera_name
- metadata must be JSON-safe

PolicyObservation enriches the VisionLatent with state_summary and mirrors the
same task/episode/timestep identifiers.
"""
from typing import Dict, Any

from src.vision.interfaces import VisionFrame, PolicyObservation, VisionLatent
from src.observation.adapter import ObservationAdapter
from src.policies.interfaces import VisionEncoderPolicy


class PolicyObservationBuilder:
    """Single entry point for building PolicyObservation objects."""

    def __init__(self, encoder: VisionEncoderPolicy, observation_adapter: ObservationAdapter = None, use_observation_adapter: bool = False):
        self.encoder = encoder
        self.observation_adapter = observation_adapter
        self.use_observation_adapter = use_observation_adapter

    def encode_frame(self, frame: VisionFrame) -> VisionLatent:
        """
        Centralized VisionFrame -> VisionLatent path used by all policy heads.
        """
        if hasattr(self.encoder, "encode_frame"):
            return self.encoder.encode_frame(frame)  # type: ignore[attr-defined]
        if hasattr(self.encoder, "encode"):
            return self.encoder.encode(frame)  # type: ignore[attr-defined]
        raise TypeError("Encoder must implement encode(frame)")

    def build(self, frame: VisionFrame, state_summary: Dict[str, Any]) -> PolicyObservation:
        latent = self.encode_frame(frame)
        return PolicyObservation(
            task_id=frame.task_id,
            episode_id=frame.episode_id,
            timestep=frame.timestep,
            latent=latent,
            state_summary=state_summary,
            metadata={
                "backend": frame.backend,
                "backend_id": frame.backend_id,
                "state_digest": frame.state_digest,
                "camera_intrinsics": frame.camera_intrinsics,
                "camera_extrinsics": frame.camera_extrinsics,
            },
        )

    def build_policy_features(
        self,
        frame: VisionFrame,
        state_summary: Dict[str, Any],
        *,
        use_observation_adapter: bool = False,
        observation_adapter: ObservationAdapter = None,
        adapter_kwargs: Dict[str, Any] = None,
        condition_kwargs: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Produce a policy-ready feature dict shared by heuristic vs neural encoders.
        """
        obs = self.build(frame, state_summary)
        features = {
            "task_id": obs.task_id,
            "episode_id": obs.episode_id,
            "timestep": obs.timestep,
            "backend": frame.backend,
            "backend_id": frame.backend_id or frame.backend,
            "state_digest": frame.state_digest,
            "vision_latent": obs.latent.to_dict(),
            "state_summary": state_summary,
            "camera_intrinsics": frame.camera_intrinsics,
            "camera_extrinsics": frame.camera_extrinsics,
            "vision_metadata": frame.metadata,
        }
        if not use_observation_adapter and not self.use_observation_adapter:
            return features

        adapter = observation_adapter or self.observation_adapter
        if adapter is None:
            # Graceful fallback to legacy features for robustness
            return features

        adapter_payload = adapter_kwargs or {}
        if condition_kwargs:
            observation, condition = adapter.build_observation_and_condition(
                vision_frame=frame,
                vision_latent=obs.latent,
                reward_scalar=float(adapter_payload.get("reward_scalar", 0.0)),
                reward_components=adapter_payload.get("reward_components", {}) or {},
                econ_vector=adapter_payload.get("econ_vector"),
                semantic_snapshot=adapter_payload.get("semantic_snapshot"),
                recap_scores=adapter_payload.get("recap_scores"),
                descriptor=adapter_payload.get("descriptor"),
                episode_metadata=adapter_payload.get("episode_metadata", {}),
                raw_env_obs=adapter_payload.get("raw_env_obs"),
                condition_kwargs=condition_kwargs,
            )
        else:
            observation = adapter.build_observation(
                vision_frame=frame,
                vision_latent=obs.latent,
                reward_scalar=float(adapter_payload.get("reward_scalar", 0.0)),
                reward_components=adapter_payload.get("reward_components", {}) or {},
                econ_vector=adapter_payload.get("econ_vector"),
                semantic_snapshot=adapter_payload.get("semantic_snapshot"),
                recap_scores=adapter_payload.get("recap_scores"),
                descriptor=adapter_payload.get("descriptor"),
                episode_metadata=adapter_payload.get("episode_metadata", {}),
                raw_env_obs=adapter_payload.get("raw_env_obs"),
            )
            condition = None
        tensor = adapter.to_policy_tensor(observation, condition=condition, include_condition=condition is not None)
        features.update(
            {
                "adapter_observation": observation,
                "adapter_observation_dict": observation.to_dict(),
                "adapter_policy_tensor": tensor,
            }
        )
        if condition is not None:
            features["condition_vector"] = condition
            features["condition_vector_dict"] = condition.to_dict()
        return features
