"""
Canonical construction of PolicyObservation from VisionFrame + low-dim state.

Required VisionFrame fields:
- backend (str), task_id (str), episode_id (str), timestep (int)
- optional rgb/depth/segmentation paths and camera_name
- metadata must be JSON-safe

PolicyObservation enriches the VisionLatent with state_summary and mirrors the
same task/episode/timestep identifiers.
"""
from typing import Dict, Any, Optional, Sequence

import numpy as np

from src.observation.adapter import ObservationAdapter
from src.observation.config import load_observation_config
from src.policies.interfaces import VisionEncoderPolicy
from src.vision.bifpn_fusion import fuse_feature_pyramid
from src.vision.config import load_vision_config
from src.vision.interfaces import VisionFrame, PolicyObservation, VisionLatent
from src.vision.regnet_backbone import build_regnet_feature_pyramid, flatten_pyramid, pyramid_to_json_safe
from src.vision.spatial_rnn_adapter import run_spatial_rnn, tensor_to_json_safe


class PolicyObservationBuilder:
    """Single entry point for building PolicyObservation objects."""

    def __init__(
        self,
        encoder: VisionEncoderPolicy,
        observation_adapter: ObservationAdapter = None,
        use_observation_adapter: bool = False,
        observation_config: Optional[Dict[str, Any]] = None,
        vision_config: Optional[Dict[str, Any]] = None,
    ):
        self.encoder = encoder
        self.observation_adapter = observation_adapter
        self.use_observation_adapter = use_observation_adapter
        self.observation_config = observation_config if observation_config is not None else load_observation_config()
        self.vision_config = vision_config if vision_config is not None else load_vision_config()
        self.use_observation_components = bool(self.observation_config.get("use_observation_components", False))
        self.default_use_condition_vector = bool(self.observation_config.get("use_condition_vector", False))
        self.vision_backbone = str(self.vision_config.get("backbone", "stub"))
        self.vision_use_bifpn = bool(self.vision_config.get("use_bifpn", False))
        self.vision_use_spatial_rnn = bool(self.vision_config.get("use_spatial_rnn", False))

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
        use_condition_vector: Optional[bool] = None,
        use_observation_components: Optional[bool] = None,
        frame_sequence: Optional[Sequence[VisionFrame]] = None,
        vision_overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Produce a policy-ready feature dict shared by heuristic vs neural encoders.
        """
        frames = list(frame_sequence) if frame_sequence else [frame]
        primary_frame = frames[0]
        obs = self.build(primary_frame, state_summary)
        features = {
            "task_id": obs.task_id,
            "episode_id": obs.episode_id,
            "timestep": obs.timestep,
            "backend": primary_frame.backend,
            "backend_id": primary_frame.backend_id or primary_frame.backend,
            "state_digest": primary_frame.state_digest,
            "vision_latent": obs.latent.to_dict(),
            "state_summary": state_summary,
            "camera_intrinsics": primary_frame.camera_intrinsics,
            "camera_extrinsics": primary_frame.camera_extrinsics,
            "vision_metadata": primary_frame.metadata,
        }
        backbone_cfg = vision_overrides or {}
        backbone_choice = str(backbone_cfg.get("backbone", self.vision_backbone or "stub"))
        use_bifpn = bool(backbone_cfg.get("use_bifpn", self.vision_use_bifpn))
        use_spatial_rnn = bool(backbone_cfg.get("use_spatial_rnn", self.vision_use_spatial_rnn))
        feature_dim = int(backbone_cfg.get("feature_dim", self.vision_config.get("regnet_feature_dim", 8)))
        if backbone_choice == "regnet_bifpn_stub":
            pyramid_sequence = [build_regnet_feature_pyramid(f, feature_dim=feature_dim) for f in frames]
            fused_sequence = [fuse_feature_pyramid(pyr) if use_bifpn else pyr for pyr in pyramid_sequence]
            flattened = [flatten_pyramid(pyr) for pyr in fused_sequence]
            spatial_tensor = run_spatial_rnn(flattened) if (use_spatial_rnn and flattened) else None
            backbone_tensor = spatial_tensor if (use_spatial_rnn and spatial_tensor is not None) else (flattened[-1] if flattened else np.array([], dtype=np.float32))
            if fused_sequence:
                features["vision_pyramid_stub"] = pyramid_to_json_safe(fused_sequence[-1])
            if backbone_tensor is not None and getattr(backbone_tensor, "size", 0) > 0:
                features["vision_backbone_features"] = [float(x) for x in np.asarray(backbone_tensor, dtype=np.float32).tolist()]
            if use_spatial_rnn and spatial_tensor is not None:
                features["spatial_rnn_summary"] = tensor_to_json_safe(spatial_tensor)
            features["vision_backbone"] = backbone_choice

        adapter_enabled = use_observation_adapter or self.use_observation_adapter
        if not adapter_enabled:
            return features

        adapter = observation_adapter or self.observation_adapter
        if adapter is None:
            # Graceful fallback to legacy features for robustness
            return features

        adapter_payload = adapter_kwargs or {}
        condition_payload = dict(condition_kwargs or {})
        if use_condition_vector is None:
            use_condition_vector = (
                bool(condition_payload)
                or getattr(adapter, "use_condition_vector", False)
                or self.default_use_condition_vector
            )
        if use_observation_components is None:
            use_observation_components = bool(
                getattr(adapter, "use_observation_components", False)
                or self.use_observation_components
                or adapter_payload.get("use_observation_components", False)
            )
        condition_payload["enable_condition"] = bool(use_condition_vector)

        observation, condition, tensor, components = adapter.build_observation_and_components(
            vision_frame=primary_frame,
            vision_latent=obs.latent,
            reward_scalar=float(adapter_payload.get("reward_scalar", 0.0)),
            reward_components=adapter_payload.get("reward_components", {}) or {},
            econ_vector=adapter_payload.get("econ_vector"),
            semantic_snapshot=adapter_payload.get("semantic_snapshot"),
            recap_scores=adapter_payload.get("recap_scores"),
            descriptor=adapter_payload.get("descriptor"),
            episode_metadata=adapter_payload.get("episode_metadata", {}),
            raw_env_obs=adapter_payload.get("raw_env_obs"),
            condition_kwargs=condition_payload,
            include_condition=bool(use_condition_vector),
            use_components=bool(use_observation_components),
        )
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
        if use_observation_components:
            features["observation_components"] = components
            features["observation_components_dict"] = components.to_dict()
        return features
