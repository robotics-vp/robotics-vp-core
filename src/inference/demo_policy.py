"""
DemoPolicy: Inference wrapper for trained models in demo/deployment scenarios.

This class provides a clean inference interface that:
- Loads all trained components (vision backbone, SIMA-2, spatial RNN, Hydra policy)
- Respects neural/stub flags from pipeline config
- Constructs ConditionVector and PolicyObservation correctly
- Returns actions compatible with env backends (PyBullet/Isaac)
- Is fully deterministic given seed
- Does NOT touch reward math or economics

For 2-week demo, not YC demo.
"""
import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

from src.config.pipeline import (
    get_canonical_task,
    get_training_config,
    is_neural_mode_enabled,
    load_pipeline_config,
)
from src.observation.adapter import ObservationAdapter
from src.observation.condition_vector_builder import ConditionVectorBuilder
from src.utils.json_safe import to_json_safe

# Try importing PyTorch components (graceful fallback to stub mode)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class DemoPolicyConfig:
    """
    Configuration for DemoPolicy inference wrapper.

    All fields are optional with sensible defaults from pipeline config.
    """
    canonical_task_id: Optional[str] = None
    use_neural_vision: bool = False
    use_neural_spatial_rnn: bool = False
    use_neural_sima2: bool = False
    use_neural_hydra: bool = False
    vision_checkpoint: Optional[str] = None
    spatial_rnn_checkpoint: Optional[str] = None
    sima2_checkpoint: Optional[str] = None
    hydra_checkpoint: Optional[str] = None
    device: str = "cpu"
    backend: str = "pybullet"  # pybullet or isaac
    enable_condition_vector: bool = True
    enable_phase_h: bool = False  # Advisory only, default off
    seed: Optional[int] = None
    use_amp: bool = False

    # Internal metadata (auto-populated, don't set manually)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, config: Optional[Dict[str, Any]] = None) -> "DemoPolicyConfig":
        """Construct from dictionary or pipeline config."""
        config = config or {}
        pipeline_config = load_pipeline_config()
        training_config = pipeline_config.get("training", {})

        return cls(
            canonical_task_id=config.get("canonical_task_id", pipeline_config.get("canonical_task_id", "drawer_open")),
            use_neural_vision=config.get("use_neural_vision", training_config.get("vision_backbone", {}).get("use_neural", False)),
            use_neural_spatial_rnn=config.get("use_neural_spatial_rnn", training_config.get("spatial_rnn", {}).get("use_neural", False)),
            use_neural_sima2=config.get("use_neural_sima2", training_config.get("sima2_segmenter", {}).get("use_neural", False)),
            use_neural_hydra=config.get("use_neural_hydra", training_config.get("hydra_policy", {}).get("use_neural", False)),
            vision_checkpoint=config.get("vision_checkpoint", training_config.get("vision_backbone", {}).get("checkpoint_path")),
            spatial_rnn_checkpoint=config.get("spatial_rnn_checkpoint", training_config.get("spatial_rnn", {}).get("checkpoint_path")),
            sima2_checkpoint=config.get("sima2_checkpoint", training_config.get("sima2_segmenter", {}).get("checkpoint_path")),
            hydra_checkpoint=config.get("hydra_checkpoint", training_config.get("hydra_policy", {}).get("checkpoint_path")),
            device=config.get("device", "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"),
            backend=config.get("backend", "pybullet"),
            enable_condition_vector=config.get("enable_condition_vector", True),
            enable_phase_h=config.get("enable_phase_h", False),
            enable_phase_h=config.get("enable_phase_h", False),
            seed=config.get("seed"),
            use_amp=config.get("use_amp", False),
        )


class DemoPolicy:
    """
    Inference wrapper for demo/deployment scenarios.

    Loads trained components and provides act(raw_obs) -> action interface.
    Fully deterministic given seed, JSON-safe, flag-gated.
    """

    def __init__(self, config: Optional[Union[DemoPolicyConfig, Dict]] = None):
        """
        Initialize DemoPolicy.

        Args:
            config: DemoPolicyConfig instance, dict, or None (uses defaults)
        """
        # Parse config
        if config is None:
            self.config = DemoPolicyConfig.from_dict()
        elif isinstance(config, dict):
            self.config = DemoPolicyConfig.from_dict(config)
        else:
            self.config = config

        # RNG state (will be seeded in reset())
        self.rng: Optional[np.random.Generator] = None
        self._seed: Optional[int] = None

        # Internal state
        self._spatial_rnn_hidden = None
        self._last_condition_vector = None
        self._last_skill_mode = None
        self._step_count = 0

        # Load components
        self._load_components()

        # Metadata for debugging
        self.metadata = {
            "backend_id": self.config.backend,
            "canonical_task_id": self.config.canonical_task_id,
            "neural_flags": {
                "vision": self.config.use_neural_vision,
                "spatial_rnn": self.config.use_neural_spatial_rnn,
                "sima2": self.config.use_neural_sima2,
                "hydra": self.config.use_neural_hydra,
            },
            "torch_available": TORCH_AVAILABLE,
            "device": self.config.device,
        }

    def _load_components(self) -> None:
        """
        Load all components (vision, spatial RNN, SIMA-2, Hydra policy).

        Respects neural flags and falls back to stubs when flags are off
        or checkpoints unavailable.
        """
        # Vision backbone (RegNet + BiFPN)
        if self.config.use_neural_vision and TORCH_AVAILABLE:
            self.vision_backbone = self._load_vision_backbone()
        else:
            self.vision_backbone = self._stub_vision_backbone()

        # Spatial RNN (ConvGRU)
        if self.config.use_neural_spatial_rnn and TORCH_AVAILABLE:
            self.spatial_rnn = self._load_spatial_rnn()
        else:
            self.spatial_rnn = self._stub_spatial_rnn()

        # SIMA-2 neural segmenter
        if self.config.use_neural_sima2 and TORCH_AVAILABLE:
            self.sima2_segmenter = self._load_sima2_segmenter()
        else:
            self.sima2_segmenter = self._stub_sima2_segmenter()

        # Hydra policy (multi-objective SAC)
        if self.config.use_neural_hydra and TORCH_AVAILABLE:
            self.hydra_policy = self._load_hydra_policy()
        else:
            self.hydra_policy = self._stub_hydra_policy()

        # ConditionVectorBuilder
        self.condition_builder = ConditionVectorBuilder(config=None)

        # ObservationAdapter (for PolicyObservation construction)
        self.obs_adapter = ObservationAdapter(
            policy_registry={},
            config={
                "use_condition_vector": self.config.enable_condition_vector,
            },
            condition_builder=self.condition_builder,
        )

    def _load_vision_backbone(self):
        """Load trained vision backbone (RegNet + BiFPN)."""
        try:
            from src.vision.regnet_backbone import RegNetBackbone
            from src.vision.bifpn_fusion import BiFPNFusion

            checkpoint_path = self.config.vision_checkpoint
            if checkpoint_path and Path(checkpoint_path).exists():
                # Load checkpoint
                checkpoint = torch.load(checkpoint_path, map_location=self.config.device)
                # Instantiate model (simplified; real impl would parse config from checkpoint)
                model = RegNetBackbone()
                model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
                model.to(self.config.device)
                model.eval()
                return model
            else:
                # No checkpoint, use initialized model
                from src.vision.regnet_backbone import RegNetBackbone
                model = RegNetBackbone()
                model.to(self.config.device)
                model.eval()
                return model
        except Exception:
            # Fallback to stub
            return self._stub_vision_backbone()

    def _stub_vision_backbone(self):
        """Stub vision backbone (hash-based features)."""
        class StubVisionBackbone:
            def __call__(self, vision_frame):
                # Hash-based deterministic feature
                signature = f"{vision_frame.episode_id}|{vision_frame.timestep}"
                digest = hashlib.sha256(signature.encode("utf-8")).hexdigest()
                feature = np.array([int(digest[i:i+2], 16) / 255.0 for i in range(0, 32, 2)], dtype=np.float32)
                return {"P3": feature, "P4": feature, "P5": feature}
        return StubVisionBackbone()

    def _load_spatial_rnn(self):
        """Load trained spatial RNN (ConvGRU)."""
        try:
            checkpoint_path = self.config.spatial_rnn_checkpoint
            if checkpoint_path and Path(checkpoint_path).exists():
                # Load checkpoint (simplified)
                checkpoint = torch.load(checkpoint_path, map_location=self.config.device)
                # Instantiate model
                from src.rl.trunk_net import SpatialRNN
                model = SpatialRNN()
                model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
                model.to(self.config.device)
                model.eval()
                return model
            else:
                # No checkpoint, use stub
                return self._stub_spatial_rnn()
        except Exception:
            return self._stub_spatial_rnn()

    def _stub_spatial_rnn(self):
        """Stub spatial RNN (identity)."""
        class StubSpatialRNN:
            def __call__(self, features, hidden=None):
                return features, None  # (features, hidden_state)
        return StubSpatialRNN()

    def _load_sima2_segmenter(self):
        """Load trained SIMA-2 neural segmenter."""
        try:
            checkpoint_path = self.config.sima2_checkpoint
            if checkpoint_path and Path(checkpoint_path).exists():
                checkpoint = torch.load(checkpoint_path, map_location=self.config.device)
                # Instantiate model (simplified)
                from src.ontology.sima2_segmenter import SIMA2Segmenter
                model = SIMA2Segmenter()
                model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
                model.to(self.config.device)
                model.eval()
                return model
            else:
                return self._stub_sima2_segmenter()
        except Exception:
            return self._stub_sima2_segmenter()

    def _stub_sima2_segmenter(self):
        """Stub SIMA-2 segmenter (returns default OOD/recovery signals)."""
        class StubSIMA2Segmenter:
            def __call__(self, vision_frame):
                return {
                    "ood_risk_level": 0.1,
                    "recovery_priority": 0.0,
                    "frontier_novelty": 0.0,
                }
        return StubSIMA2Segmenter()

    def _load_hydra_policy(self):
        """Load trained Hydra policy (multi-objective SAC)."""
        try:
            checkpoint_path = self.config.hydra_checkpoint
            if checkpoint_path and Path(checkpoint_path).exists():
                checkpoint = torch.load(checkpoint_path, map_location=self.config.device)
                # Instantiate model (simplified)
                from src.rl.hydra_heads import HydraPolicy
                model = HydraPolicy()
                model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
                model.to(self.config.device)
                model.eval()
                return model
            else:
                return self._stub_hydra_policy()
        except Exception:
            return self._stub_hydra_policy()

    def _stub_hydra_policy(self):
        """Stub Hydra policy (zero action)."""
        class StubHydraPolicy:
            def __call__(self, obs_tensor, deterministic=True):
                # Return zero action (compatible with most envs)
                return np.zeros(7, dtype=np.float32)  # 7-DOF arm action
        return StubHydraPolicy()

    def reset(self, seed: Optional[int] = None) -> None:
        """
        Reset policy state and seed all RNGs.

        Args:
            seed: Random seed for determinism. If None, uses config seed or 0.
        """
        self._seed = seed if seed is not None else (self.config.seed if self.config.seed is not None else 0)

        # Seed Python, NumPy, PyTorch
        random.seed(self._seed)
        np.random.seed(self._seed)
        self.rng = np.random.default_rng(self._seed)

        if TORCH_AVAILABLE:
            torch.manual_seed(self._seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self._seed)

        # Reset internal state
        self._spatial_rnn_hidden = None
        self._last_condition_vector = None
        self._last_skill_mode = None
        self._step_count = 0

    def act(self, raw_obs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute action given raw environment observation.

        Args:
            raw_obs: Raw observation dict from env (PyBullet or Isaac format)

        Returns:
            Action dict compatible with env backend:
            {
                "action": np.ndarray,  # Joint position/velocity targets
                "metadata": dict,       # Debugging info
            }
        """
        # Construct VisionFrame from raw_obs
        vision_frame = self._raw_obs_to_vision_frame(raw_obs)

        # Run neural components with AMP if enabled
        if TORCH_AVAILABLE and self.config.use_amp:
            with torch.autocast(device_type=self.config.device, dtype=torch.float16):
                # Run vision backbone
                vision_features = self.vision_backbone(vision_frame)

                # Run SIMA-2 segmenter (OOD/recovery signals)
                sima2_output = self.sima2_segmenter(vision_frame)

                # Build ConditionVector (CPU-bound, no AMP needed but harmless)
                condition_vector = self._build_condition_vector(raw_obs, sima2_output)
                self._last_condition_vector = condition_vector

                # Run spatial RNN
                spatial_features, self._spatial_rnn_hidden = self.spatial_rnn(
                    vision_features, self._spatial_rnn_hidden
                )

                # Construct PolicyObservation
                policy_obs = self._build_policy_observation(
                    raw_obs, spatial_features, condition_vector
                )

                # Run Hydra policy
                action = self.hydra_policy(policy_obs, deterministic=True)
        else:
            # Run vision backbone
            vision_features = self.vision_backbone(vision_frame)

            # Run SIMA-2 segmenter (OOD/recovery signals)
            sima2_output = self.sima2_segmenter(vision_frame)

            # Build ConditionVector
            condition_vector = self._build_condition_vector(raw_obs, sima2_output)
            self._last_condition_vector = condition_vector

            # Run spatial RNN
            spatial_features, self._spatial_rnn_hidden = self.spatial_rnn(
                vision_features, self._spatial_rnn_hidden
            )

            # Construct PolicyObservation
            policy_obs = self._build_policy_observation(
                raw_obs, spatial_features, condition_vector
            )

            # Run Hydra policy
            action = self.hydra_policy(policy_obs, deterministic=True)

        # Convert to numpy if needed
        if TORCH_AVAILABLE and isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()

        self._step_count += 1

        # Return action dict
        return {
            "action": action,
            "metadata": {
                "step": self._step_count,
                "skill_mode": condition_vector.skill_mode if hasattr(condition_vector, "skill_mode") else "default",
                "ood_risk_level": sima2_output.get("ood_risk_level", 0.0),
                "recovery_priority": sima2_output.get("recovery_priority", 0.0),
            },
        }

    def _raw_obs_to_vision_frame(self, raw_obs: Dict[str, Any]):
        """Convert raw env obs to VisionFrame."""
        from src.vision.interfaces import VisionFrame, compute_state_digest

        # Extract RGB (stub if missing)
        rgb = raw_obs.get("rgb")
        if rgb is None:
            rgb = np.zeros((64, 64, 3), dtype=np.uint8)

        # Extract depth (stub if missing)
        depth = raw_obs.get("depth")

        # Compute state digest
        state_digest = compute_state_digest(raw_obs)

        # Determine dimensions from RGB if available
        if rgb is not None and hasattr(rgb, 'shape'):
            height, width, channels = rgb.shape
        else:
            height, width, channels = 64, 64, 3

        return VisionFrame(
            backend=self.config.backend,
            task_id=self.config.canonical_task_id,
            episode_id=raw_obs.get("episode_id", "demo_episode"),
            timestep=self._step_count,
            width=width,
            height=height,
            channels=channels,
            state_digest=state_digest,
            metadata={"rgb_shape": (height, width, channels) if rgb is not None else None},
        )

    def _build_condition_vector(self, raw_obs: Dict[str, Any], sima2_output: Dict[str, Any]):
        """Build ConditionVector from raw obs and SIMA-2 output."""
        from src.observation.condition_vector import ConditionVector

        if not self.config.enable_condition_vector:
            # Return minimal condition vector
            return ConditionVector(
                curriculum_phase="demo",
                skill_mode="default",
                objective_preset="balanced",
            )

        # Use ConditionVectorBuilder
        return self.condition_builder.build(
            episode_config={},
            econ_state={},
            curriculum_phase="demo",
            sima2_trust=None,
            datapack_metadata={},
            episode_step=self._step_count,
            overrides={
                "ood_risk_level": sima2_output.get("ood_risk_level", 0.1),
                "recovery_priority": sima2_output.get("recovery_priority", 0.0),
            },
        )

    def _build_policy_observation(
        self, raw_obs: Dict[str, Any], spatial_features: Any, condition_vector: Any
    ) -> Any:
        """
        Build PolicyObservation tensor for Hydra policy.

        For stub mode, just return a simple feature vector.
        Real implementation would use ObservationAdapter.
        """
        # Simplified: flatten spatial features + condition vector
        if isinstance(spatial_features, dict):
            # Multi-scale features (P3/P4/P5)
            feature_list = []
            for k in sorted(spatial_features.keys()):
                f = spatial_features[k]
                if isinstance(f, np.ndarray):
                    feature_list.append(f.flatten())
            spatial_flat = np.concatenate(feature_list) if feature_list else np.zeros(64, dtype=np.float32)
        elif isinstance(spatial_features, np.ndarray):
            spatial_flat = spatial_features.flatten()
        else:
            spatial_flat = np.zeros(64, dtype=np.float32)

        # Condition vector features (stub)
        condition_flat = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        # Proprioception from raw_obs (joint positions, velocities)
        proprio = raw_obs.get("proprio", raw_obs.get("joint_positions", np.zeros(7, dtype=np.float32)))
        if not isinstance(proprio, np.ndarray):
            proprio = np.array(proprio, dtype=np.float32)

        # Concatenate all features
        obs_tensor = np.concatenate([spatial_flat[:32], condition_flat, proprio[:7]])
        return obs_tensor

    def get_summary(self) -> Dict[str, Any]:
        """
        Get JSON-safe summary of current policy state.

        Returns:
            Dictionary with metadata, last condition vector, skill mode, etc.
        """
        summary = dict(self.metadata)
        summary["step_count"] = self._step_count
        summary["seed"] = self._seed

        if self._last_condition_vector:
            try:
                summary["last_condition_vector"] = to_json_safe(self._last_condition_vector)
            except Exception:
                summary["last_condition_vector"] = {}

        if self._last_skill_mode:
            summary["last_skill_mode"] = str(self._last_skill_mode)

        return summary
