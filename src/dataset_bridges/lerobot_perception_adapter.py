"""LeRobot → Perception Seam Training Data Adapters.

This module provides adapters that convert LeRobot-format multi-camera
robot manipulation episodes into typed training samples for perception
neural seams.

GPU-honest status
-----------------
This module is **adapter-usable now** — no GPU required for data intake
and schema work. The adapters construct training samples from external
data; actual training at scale is a GPU-era step.

Supported conversions
---------------------
- LeRobot multi-camera step → ``MultiProviderSample`` for EvidenceFusionSeam
- LeRobot episode window → ``VJEPATemporalSample`` for VJEPATemporalAlignmentSeam
- LeRobot step with task → ``VisionBackboneProjectionSample`` (placeholder)

Data sources
------------
Primary targets are DROID, Bridge V2, and ALOHA suite datasets in
LeRobot v3 format. See ``perception_external_data_roadmap.md`` for
full data source analysis.

Feature extraction
------------------
For prototype-trainable work, features can be:
- Raw flattened images (no GPU, but low quality)
- Frozen backbone features (requires GPU for inference, cached after)
- Placeholder random features (for schema verification only)

Production training requires frozen backbone feature extraction.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, List, Mapping, Optional, Sequence, Tuple

import torch

from src.replay.schema import ReplayEpisodeRecord, ReplayStepRecord
from src.training.perception_seam_data import (
    MultiProviderSample,
    ProviderObservation,
    VJEPATemporalSample,
    VisionBackboneProjectionSample,
)


# ---------------------------------------------------------------------------
# Feature extraction strategies (GPU-honest)
# ---------------------------------------------------------------------------


@dataclass
class FeatureExtractionConfig:
    """Configuration for feature extraction from camera images."""

    strategy: str = "placeholder"  # "placeholder" | "flattened" | "frozen_backbone"
    d_feature: int = 128
    backbone_name: Optional[str] = None  # e.g., "dinov2_vitl14"
    device: str = "cpu"


def _placeholder_features(
    image: Any,
    d_feature: int,
    seed_str: str,
) -> torch.Tensor:
    """Generate deterministic placeholder features for schema verification.

    NOT for training — only for verifying data loading pipeline works.
    """
    # Deterministic hash-based seed
    seed = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16)
    torch.manual_seed(seed)
    return torch.randn(d_feature)


def _flattened_features(
    image: Any,
    d_feature: int,
) -> torch.Tensor:
    """Flatten and project raw image to fixed dimension.

    Low quality but requires no GPU. Useful for CPU proof-of-life.
    """
    if isinstance(image, torch.Tensor):
        flat = image.flatten().float()
    elif hasattr(image, "__array__"):
        import numpy as np
        flat = torch.from_numpy(np.asarray(image)).flatten().float()
    else:
        # Fallback: treat as placeholder
        return torch.randn(d_feature)

    # Simple projection: take first d_feature elements or pad
    if flat.numel() >= d_feature:
        return flat[:d_feature]
    else:
        return torch.nn.functional.pad(flat, (0, d_feature - flat.numel()))


def extract_features(
    image: Any,
    config: FeatureExtractionConfig,
    *,
    seed_str: str = "",
) -> torch.Tensor:
    """Extract features from camera image according to config.

    Args:
        image: Raw image data (tensor, numpy array, or PIL Image).
        config: Feature extraction configuration.
        seed_str: Seed string for deterministic placeholder features.

    Returns:
        Feature tensor of shape ``(d_feature,)``.
    """
    if config.strategy == "placeholder":
        return _placeholder_features(image, config.d_feature, seed_str)
    elif config.strategy == "flattened":
        return _flattened_features(image, config.d_feature)
    elif config.strategy == "frozen_backbone":
        # Frozen backbone extraction requires GPU and model loading
        # This is a GPU-era capability; stub for now
        raise NotImplementedError(
            "frozen_backbone feature extraction requires GPU. "
            "Use 'placeholder' or 'flattened' for CPU verification."
        )
    else:
        raise ValueError(f"Unknown feature extraction strategy: {config.strategy}")


# ---------------------------------------------------------------------------
# Camera key discovery
# ---------------------------------------------------------------------------


def discover_camera_keys(step_obs: Mapping[str, Any]) -> List[str]:
    """Discover available camera keys from step observation dict.

    LeRobot datasets store camera observations under keys like:
    - ``images.exterior_image_1_left``
    - ``images.wrist_image_left``
    - ``images.image_0``, ``images.image_1``, etc.

    Returns:
        List of camera key names (without ``images.`` prefix).
    """
    camera_keys = []
    for key in step_obs.keys():
        if key.startswith("images."):
            camera_keys.append(key[7:])  # Remove "images." prefix
        elif key.startswith("observation.images."):
            camera_keys.append(key[19:])  # Remove "observation.images." prefix
    return sorted(camera_keys)


# ---------------------------------------------------------------------------
# LeRobot → MultiProviderSample adapter
# ---------------------------------------------------------------------------


def multi_provider_sample_from_lerobot_step(
    step: ReplayStepRecord,
    *,
    camera_keys: Optional[List[str]] = None,
    feature_config: Optional[FeatureExtractionConfig] = None,
    downstream_task_success: Optional[float] = None,
) -> MultiProviderSample:
    """Convert LeRobot step with multi-camera obs to MultiProviderSample.

    Each camera becomes a "provider" with vision_backbone kind. This enables
    held-out reconstruction training for EvidenceFusionSeam.

    Args:
        step: ReplayStepRecord from LeRobot bridge rehydration.
        camera_keys: Camera keys to use. If None, auto-discovers from step.obs.
        feature_config: Feature extraction configuration.
        downstream_task_success: Optional task success signal (e.g., reward).

    Returns:
        MultiProviderSample suitable for EvidenceFusionSeam training.
    """
    if feature_config is None:
        feature_config = FeatureExtractionConfig()

    if camera_keys is None:
        camera_keys = discover_camera_keys(step.obs)

    if not camera_keys:
        raise ValueError(
            f"No camera keys found in step obs. Keys: {list(step.obs.keys())}"
        )

    providers: List[ProviderObservation] = []
    for cam_key in camera_keys:
        # Try multiple key formats
        image = None
        for prefix in ["images.", "observation.images.", ""]:
            full_key = f"{prefix}{cam_key}"
            if full_key in step.obs:
                image = step.obs[full_key]
                break

        if image is None:
            # Camera not available in this step
            providers.append(ProviderObservation(
                provider_id=cam_key,
                provider_kind="vision_backbone",
                availability_status="unavailable",
                truth_class="unavailable",
                features=torch.zeros(feature_config.d_feature),
            ))
            continue

        # Extract features
        seed_str = f"{step.episode_id}_{step.step_idx}_{cam_key}"
        features = extract_features(image, feature_config, seed_str=seed_str)

        providers.append(ProviderObservation(
            provider_id=cam_key,
            provider_kind="vision_backbone",
            availability_status="available",
            truth_class="provider_backed",
            features=features,
            confidence=torch.tensor(1.0),  # Raw camera has full confidence
            metadata={"camera_key": cam_key},
        ))

    # Task success proxy
    task_success = downstream_task_success
    if task_success is None:
        # Use reward as proxy if available
        task_success = float(step.reward) if step.reward != 0.0 else None

    return MultiProviderSample(
        sample_id=step.record_id,
        scene_id=step.episode_id,
        frame_idx=step.step_idx,
        providers=providers,
        downstream_task_success=task_success,
        metadata={
            "task_id": step.task_id,
            "env_id": step.env_id,
            "source_domain": step.source_domain,
        },
    )


def multi_provider_samples_from_episode(
    episode: ReplayEpisodeRecord,
    steps: Sequence[ReplayStepRecord],
    *,
    camera_keys: Optional[List[str]] = None,
    feature_config: Optional[FeatureExtractionConfig] = None,
    stride: int = 1,
    max_samples: Optional[int] = None,
) -> List[MultiProviderSample]:
    """Convert full LeRobot episode to MultiProviderSamples.

    Args:
        episode: ReplayEpisodeRecord from LeRobot bridge.
        steps: Sequence of ReplayStepRecords for the episode.
        camera_keys: Camera keys to use. If None, auto-discovers.
        feature_config: Feature extraction configuration.
        stride: Sample every N steps (default: 1 = all steps).
        max_samples: Maximum samples to extract (default: None = all).

    Returns:
        List of MultiProviderSample for training.
    """
    ordered_steps = sorted(steps, key=lambda s: s.step_idx)

    # Auto-discover camera keys from first step
    if camera_keys is None and ordered_steps:
        camera_keys = discover_camera_keys(ordered_steps[0].obs)

    samples = []
    for i, step in enumerate(ordered_steps):
        if i % stride != 0:
            continue

        sample = multi_provider_sample_from_lerobot_step(
            step,
            camera_keys=camera_keys,
            feature_config=feature_config,
        )
        samples.append(sample)

        if max_samples is not None and len(samples) >= max_samples:
            break

    return samples


# ---------------------------------------------------------------------------
# LeRobot → VisionBackboneProjectionSample adapter
# ---------------------------------------------------------------------------


def _projection_target_from_feature(
    feature: torch.Tensor,
    *,
    d_out: int,
) -> torch.Tensor:
    """Build a deterministic CPU-safe proxy target from one backbone feature."""

    flat = feature.flatten().float()
    if flat.numel() >= d_out:
        return flat[:d_out]
    return torch.nn.functional.pad(flat, (0, d_out - flat.numel()))


def vision_backbone_projection_sample_from_lerobot_step(
    step: ReplayStepRecord,
    *,
    camera_keys: Optional[List[str]] = None,
    feature_config: Optional[FeatureExtractionConfig] = None,
    tokens_per_camera: int = 4,
    d_out: int = 128,
) -> VisionBackboneProjectionSample:
    """Convert one LeRobot step into a projection-head proof sample.

    This is intentionally a **prototype / schema** adapter, not a provider-
    credible training adapter. In the absence of object annotations or frozen
    backbone execution, each camera slot becomes a stable pseudo-identity and
    each camera contributes a small bundle of deterministic proxy tokens. That
    is enough to prove the LeRobot → projection-seam path without pretending we
    have real object-ID supervision.
    """

    if feature_config is None:
        feature_config = FeatureExtractionConfig(d_feature=1024)
    if camera_keys is None:
        camera_keys = discover_camera_keys(step.obs)
    if not camera_keys:
        raise ValueError(
            f"No camera keys found in step obs. Keys: {list(step.obs.keys())}"
        )

    backbone_tokens: list[torch.Tensor] = []
    identity_labels: list[int] = []
    cross_provider_targets: list[torch.Tensor] = []

    for camera_idx, cam_key in enumerate(camera_keys):
        image = None
        for prefix in ["images.", "observation.images.", ""]:
            full_key = f"{prefix}{cam_key}"
            if full_key in step.obs:
                image = step.obs[full_key]
                break
        if image is None:
            continue

        base_feature = extract_features(
            image,
            feature_config,
            seed_str=f"vision_projection_{step.episode_id}_{cam_key}",
        )
        for token_idx in range(tokens_per_camera):
            token_seed = int(
                hashlib.md5(
                    f"{step.episode_id}_{step.step_idx}_{cam_key}_{token_idx}".encode()
                ).hexdigest()[:8],
                16,
            )
            generator = torch.Generator().manual_seed(token_seed)
            token = base_feature + torch.randn(
                base_feature.shape,
                generator=generator,
            ) * 0.01
            backbone_tokens.append(token)
            identity_labels.append(camera_idx)
            cross_provider_targets.append(
                _projection_target_from_feature(token, d_out=d_out)
            )

    if not backbone_tokens:
        raise ValueError(
            f"No available camera observations found in step obs. Keys: {list(step.obs.keys())}"
        )

    scene_label = int(
        hashlib.md5(step.episode_id.encode()).hexdigest()[:8],
        16,
    )
    return VisionBackboneProjectionSample(
        sample_id=step.record_id,
        backbone_features=torch.stack(backbone_tokens, dim=0),
        object_identity_labels=torch.tensor(identity_labels, dtype=torch.long),
        scene_label=scene_label,
        cross_provider_embeddings=torch.stack(cross_provider_targets, dim=0),
    )


def vision_backbone_projection_samples_from_episode(
    episode: ReplayEpisodeRecord,
    steps: Sequence[ReplayStepRecord],
    *,
    camera_keys: Optional[List[str]] = None,
    feature_config: Optional[FeatureExtractionConfig] = None,
    stride: int = 1,
    max_samples: Optional[int] = None,
    tokens_per_camera: int = 4,
    d_out: int = 128,
) -> List[VisionBackboneProjectionSample]:
    """Convert one LeRobot episode into projection-head proof samples."""

    ordered_steps = sorted(steps, key=lambda s: s.step_idx)
    if camera_keys is None and ordered_steps:
        camera_keys = discover_camera_keys(ordered_steps[0].obs)

    samples: list[VisionBackboneProjectionSample] = []
    for idx, step in enumerate(ordered_steps):
        if idx % stride != 0:
            continue
        samples.append(
            vision_backbone_projection_sample_from_lerobot_step(
                step,
                camera_keys=camera_keys,
                feature_config=feature_config,
                tokens_per_camera=tokens_per_camera,
                d_out=d_out,
            )
        )
        if max_samples is not None and len(samples) >= max_samples:
            break
    return samples


# ---------------------------------------------------------------------------
# LeRobot → VJEPATemporalSample adapter
# ---------------------------------------------------------------------------


def vjepa_temporal_sample_from_episode_window(
    steps: Sequence[ReplayStepRecord],
    window_start: int,
    window_size: int,
    *,
    n_objects: int = 10,
    d_vjepa: int = 1024,
    d_wm: int = 128,
    d_out: int = 128,
    feature_config: Optional[FeatureExtractionConfig] = None,
) -> Optional[VJEPATemporalSample]:
    """Extract VJEPATemporalSample from episode temporal window.

    Args:
        steps: Sequence of ReplayStepRecords (must be sorted by step_idx).
        window_start: Starting step index for the window.
        window_size: Number of temporal steps (T).
        n_objects: Number of WM objects to simulate.
        d_vjepa: V-JEPA token dimension.
        d_wm: WM object token dimension.
        d_out: Output dimension for aligned temporal state.
        feature_config: Feature extraction configuration.

    Returns:
        VJEPATemporalSample if window is valid, None otherwise.

    Note:
        This adapter creates placeholder V-JEPA tokens and WM object tokens.
        Real training requires:
        - V-JEPA tokens from frozen V-JEPA model inference (GPU)
        - WM object tokens from upstream perception state
        - Future object states from subsequent frame observations

        For prototype-trainable verification, we use synthetic placeholders
        derived deterministically from frame data.
    """
    if feature_config is None:
        feature_config = FeatureExtractionConfig(d_feature=d_vjepa)

    ordered_steps = sorted(steps, key=lambda s: s.step_idx)

    # Check window validity
    if window_start < 0 or window_start + window_size > len(ordered_steps):
        return None

    window_steps = ordered_steps[window_start:window_start + window_size]
    T = len(window_steps)

    # Generate placeholder V-JEPA tokens per timestep
    # In production: would come from frozen V-JEPA inference
    n_vjepa_tokens = 196  # ViT-L patch count
    vjepa_tokens = torch.zeros(T, n_vjepa_tokens, d_vjepa)

    for t, step in enumerate(window_steps):
        seed_str = f"vjepa_{step.episode_id}_{step.step_idx}"
        seed = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16)
        torch.manual_seed(seed)
        vjepa_tokens[t] = torch.randn(n_vjepa_tokens, d_vjepa)

    # Generate placeholder WM object tokens
    # In production: would come from upstream perception WM state
    seed_str = f"wm_{window_steps[0].episode_id}_{window_start}"
    seed = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16)
    torch.manual_seed(seed)
    wm_object_tokens = torch.randn(n_objects, d_wm)

    # Generate future object states from subsequent frames
    # In production: would come from actual future observation encoding
    future_object_states = torch.zeros(T, n_objects, d_out)
    for t, step in enumerate(window_steps):
        seed_str = f"future_{step.episode_id}_{step.step_idx}"
        seed = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16)
        torch.manual_seed(seed)
        future_object_states[t] = torch.randn(n_objects, d_out)

    # Object validity (all valid for now)
    object_valid_mask = torch.ones(n_objects, dtype=torch.bool)

    # Temporal ordering labels
    temporal_ordering_labels = torch.arange(T)

    return VJEPATemporalSample(
        sample_id=f"{window_steps[0].episode_id}_w{window_start}",
        vjepa_tokens=vjepa_tokens,
        wm_object_tokens=wm_object_tokens,
        future_object_states=future_object_states,
        object_valid_mask=object_valid_mask,
        temporal_ordering_labels=temporal_ordering_labels,
    )


def vjepa_temporal_samples_from_episode(
    episode: ReplayEpisodeRecord,
    steps: Sequence[ReplayStepRecord],
    *,
    window_size: int = 4,
    stride: int = 2,
    max_samples: Optional[int] = None,
    **kwargs,
) -> List[VJEPATemporalSample]:
    """Extract multiple VJEPATemporalSamples from an episode.

    Args:
        episode: ReplayEpisodeRecord from LeRobot bridge.
        steps: Sequence of ReplayStepRecords for the episode.
        window_size: Number of temporal steps per sample.
        stride: Stride between window starts.
        max_samples: Maximum samples to extract.
        **kwargs: Passed to vjepa_temporal_sample_from_episode_window.

    Returns:
        List of VJEPATemporalSample for training.
    """
    ordered_steps = sorted(steps, key=lambda s: s.step_idx)
    samples = []

    window_start = 0
    while window_start + window_size <= len(ordered_steps):
        sample = vjepa_temporal_sample_from_episode_window(
            ordered_steps,
            window_start,
            window_size,
            **kwargs,
        )
        if sample is not None:
            samples.append(sample)

        if max_samples is not None and len(samples) >= max_samples:
            break

        window_start += stride

    return samples


# ---------------------------------------------------------------------------
# Dataset-level adapters
# ---------------------------------------------------------------------------


@dataclass
class LeRobotPerceptionAdapterConfig:
    """Configuration for LeRobot → perception seam data adaptation."""

    # Feature extraction
    feature_config: FeatureExtractionConfig = field(default_factory=FeatureExtractionConfig)

    # Multi-provider sampling
    camera_keys: Optional[List[str]] = None
    step_stride: int = 1
    max_samples_per_episode: Optional[int] = None

    # V-JEPA temporal sampling
    temporal_window_size: int = 4
    temporal_stride: int = 2
    n_objects: int = 10
    d_vjepa: int = 1024
    d_wm: int = 128
    d_out: int = 128

    # Vision-backbone projection proof sampling
    projection_tokens_per_camera: int = 4

    def __post_init__(self):
        pass


def adapt_lerobot_episodes_for_evidence_fusion(
    episodes: Sequence[Tuple[ReplayEpisodeRecord, Sequence[ReplayStepRecord]]],
    config: Optional[LeRobotPerceptionAdapterConfig] = None,
) -> List[MultiProviderSample]:
    """Adapt LeRobot episodes for EvidenceFusionSeam training.

    Args:
        episodes: Sequence of (episode, steps) tuples from LeRobot bridge.
        config: Adapter configuration.

    Returns:
        List of MultiProviderSample for training.
    """
    if config is None:
        config = LeRobotPerceptionAdapterConfig()

    all_samples = []
    for episode, steps in episodes:
        samples = multi_provider_samples_from_episode(
            episode,
            steps,
            camera_keys=config.camera_keys,
            feature_config=config.feature_config,
            stride=config.step_stride,
            max_samples=config.max_samples_per_episode,
        )
        all_samples.extend(samples)

    return all_samples


def adapt_lerobot_episodes_for_vjepa_temporal(
    episodes: Sequence[Tuple[ReplayEpisodeRecord, Sequence[ReplayStepRecord]]],
    config: Optional[LeRobotPerceptionAdapterConfig] = None,
) -> List[VJEPATemporalSample]:
    """Adapt LeRobot episodes for VJEPATemporalAlignmentSeam training.

    Args:
        episodes: Sequence of (episode, steps) tuples from LeRobot bridge.
        config: Adapter configuration.

    Returns:
        List of VJEPATemporalSample for training.
    """
    if config is None:
        config = LeRobotPerceptionAdapterConfig()

    all_samples = []
    for episode, steps in episodes:
        samples = vjepa_temporal_samples_from_episode(
            episode,
            steps,
            window_size=config.temporal_window_size,
            stride=config.temporal_stride,
            n_objects=config.n_objects,
            d_vjepa=config.d_vjepa,
            d_wm=config.d_wm,
            d_out=config.d_out,
        )
        all_samples.extend(samples)

    return all_samples


def adapt_lerobot_episodes_for_vision_backbone_projection(
    episodes: Sequence[Tuple[ReplayEpisodeRecord, Sequence[ReplayStepRecord]]],
    config: Optional[LeRobotPerceptionAdapterConfig] = None,
) -> List[VisionBackboneProjectionSample]:
    """Adapt LeRobot episodes for local vision-projection proof work."""

    if config is None:
        config = LeRobotPerceptionAdapterConfig(
            feature_config=FeatureExtractionConfig(d_feature=1024)
        )

    all_samples: list[VisionBackboneProjectionSample] = []
    for episode, steps in episodes:
        samples = vision_backbone_projection_samples_from_episode(
            episode,
            steps,
            camera_keys=config.camera_keys,
            feature_config=config.feature_config,
            stride=config.step_stride,
            max_samples=config.max_samples_per_episode,
            tokens_per_camera=config.projection_tokens_per_camera,
            d_out=config.d_out,
        )
        all_samples.extend(samples)
    return all_samples


__all__ = [
    # Config
    "FeatureExtractionConfig",
    "LeRobotPerceptionAdapterConfig",
    # Feature extraction
    "extract_features",
    "discover_camera_keys",
    # Single-step adapters
    "multi_provider_sample_from_lerobot_step",
    "multi_provider_samples_from_episode",
    "vision_backbone_projection_sample_from_lerobot_step",
    "vision_backbone_projection_samples_from_episode",
    # Temporal window adapters
    "vjepa_temporal_sample_from_episode_window",
    "vjepa_temporal_samples_from_episode",
    # Dataset-level adapters
    "adapt_lerobot_episodes_for_evidence_fusion",
    "adapt_lerobot_episodes_for_vjepa_temporal",
    "adapt_lerobot_episodes_for_vision_backbone_projection",
]
