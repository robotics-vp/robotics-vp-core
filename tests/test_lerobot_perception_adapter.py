"""Integration tests for LeRobot → Perception Seam training data adapters.

These tests verify:
1. Multi-camera LeRobot steps convert to valid MultiProviderSample records
2. Episode windows convert to valid VJEPATemporalSample records
3. Camera key discovery handles DROID/Bridge observation formats
4. Feature extraction strategies (placeholder, flattened) work on CPU

GPU-honest status: These tests run entirely on CPU using placeholder or
flattened feature extraction strategies.
"""

from __future__ import annotations

import uuid
from typing import List

import pytest
import torch
import numpy as np

from src.replay.schema import ReplayEpisodeRecord, ReplayStepRecord
from src.dataset_bridges.lerobot_perception_adapter import (
    FeatureExtractionConfig,
    LeRobotPerceptionAdapterConfig,
    adapt_lerobot_episodes_for_vision_backbone_projection,
    discover_camera_keys,
    extract_features,
    multi_provider_sample_from_lerobot_step,
    multi_provider_samples_from_episode,
    vision_backbone_projection_sample_from_lerobot_step,
    vision_backbone_projection_samples_from_episode,
    vjepa_temporal_sample_from_episode_window,
    vjepa_temporal_samples_from_episode,
    adapt_lerobot_episodes_for_evidence_fusion,
    adapt_lerobot_episodes_for_vjepa_temporal,
)
from src.training.perception_seam_data import (
    MultiProviderSample,
    ProviderObservation,
    VJEPATemporalSample,
    VisionBackboneProjectionSample,
)


# ---------------------------------------------------------------------------
# Test fixtures — Mock LeRobot-format data
# ---------------------------------------------------------------------------


def make_mock_lerobot_step(
    episode_id: str,
    step_idx: int,
    *,
    camera_format: str = "droid",  # "droid" | "bridge" | "aloha"
    image_shape: tuple = (180, 320, 3),
    with_state: bool = True,
    reward: float = 0.0,
    done: bool = False,
) -> ReplayStepRecord:
    """Create a mock ReplayStepRecord in LeRobot format.

    Args:
        episode_id: Episode identifier.
        step_idx: Step index within episode.
        camera_format: Which dataset format to mimic.
        image_shape: Shape of mock images.
        with_state: Include robot state observation.
        reward: Step reward.
        done: Episode termination flag.

    Returns:
        ReplayStepRecord with observations matching LeRobot format.
    """
    # Generate deterministic mock images
    rng = np.random.default_rng(hash((episode_id, step_idx)) % (2**32))

    obs = {}

    if camera_format == "droid":
        # DROID: 3 cameras
        obs["images.exterior_image_1_left"] = rng.integers(0, 255, image_shape, dtype=np.uint8)
        obs["images.exterior_image_2_left"] = rng.integers(0, 255, image_shape, dtype=np.uint8)
        obs["images.wrist_image_left"] = rng.integers(0, 255, image_shape, dtype=np.uint8)
        if with_state:
            obs["state"] = rng.random(7).astype(np.float32)

    elif camera_format == "bridge":
        # Bridge V2: 4 cameras (image_0 through image_3)
        for i in range(4):
            obs[f"images.image_{i}"] = rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)
        if with_state:
            obs["state"] = rng.random(8).astype(np.float32)

    elif camera_format == "aloha":
        # ALOHA: variable cameras, bimanual
        obs["images.cam_high"] = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
        obs["images.cam_left_wrist"] = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
        obs["images.cam_right_wrist"] = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
        if with_state:
            obs["state"] = rng.random(14).astype(np.float32)  # bimanual

    else:
        raise ValueError(f"Unknown camera format: {camera_format}")

    return ReplayStepRecord(
        run_id="test_run",
        episode_id=episode_id,
        step_idx=step_idx,
        obs=obs,
        obs_vector=[],
        action={"joint_positions": list(rng.random(7))},
        action_vector=list(rng.random(7)),
        reward=reward,
        reward_decomposition={},
        done=done,
        task_id="pick_and_place",
        env_id="tabletop",
        condition_vector={},
        condition_vector_values=[],
        skill_mode="autonomous",
        objective_tensor_summary={},
        objective_tensor_ref=None,
        econ_tensor_summary={},
        econ_tensor_ref=None,
        constraint_flags=[],
        pricing_tick_ref=None,
        ledger_event_ref=None,
        source_domain="lerobot_test",
        seed=42,
        timestamp=f"2024-01-01T00:00:{step_idx:02d}Z",
        metadata={},
        provenance={},
    )


def make_mock_lerobot_episode(
    num_steps: int = 10,
    camera_format: str = "droid",
    **kwargs,
) -> tuple[ReplayEpisodeRecord, List[ReplayStepRecord]]:
    """Create a mock episode with steps in LeRobot format.

    Returns:
        Tuple of (episode record, list of step records).
    """
    episode_id = str(uuid.uuid4())[:8]

    steps = [
        make_mock_lerobot_step(
            episode_id,
            i,
            camera_format=camera_format,
            reward=1.0 if i == num_steps - 1 else 0.0,
            done=i == num_steps - 1,
            **kwargs,
        )
        for i in range(num_steps)
    ]

    episode = ReplayEpisodeRecord(
        run_id="test_run",
        episode_id=episode_id,
        task_id="pick_and_place",
        env_id="tabletop",
        source_domain="lerobot_test",
        seed=42,
        status="done",
        started_at=steps[0].timestamp,
        ended_at=steps[-1].timestamp,
        total_steps=len(steps),
        total_reward=sum(s.reward for s in steps),
        skill_mode="autonomous",
        condition_vector={},
        condition_vector_values=[],
        objective_tensor_summary={},
        objective_tensor_ref=None,
        econ_tensor_summary={},
        econ_tensor_ref=None,
        pricing_summary={},
        pricing_tick_refs=[],
        constraint_flags=[],
        regal_summary={},
        datapack_summary={},
        ledger_event_ids=[],
        metadata={},
        provenance={},
    )

    return episode, steps


# ---------------------------------------------------------------------------
# Camera key discovery tests
# ---------------------------------------------------------------------------


class TestDiscoverCameraKeys:
    """Test camera key auto-discovery from various dataset formats."""

    def test_discover_droid_cameras(self):
        """DROID format: exterior_image_1_left, exterior_image_2_left, wrist_image_left."""
        step = make_mock_lerobot_step("ep1", 0, camera_format="droid")
        keys = discover_camera_keys(step.obs)

        assert len(keys) == 3
        assert "exterior_image_1_left" in keys
        assert "exterior_image_2_left" in keys
        assert "wrist_image_left" in keys

    def test_discover_bridge_cameras(self):
        """Bridge V2 format: image_0, image_1, image_2, image_3."""
        step = make_mock_lerobot_step("ep1", 0, camera_format="bridge")
        keys = discover_camera_keys(step.obs)

        assert len(keys) == 4
        for i in range(4):
            assert f"image_{i}" in keys

    def test_discover_aloha_cameras(self):
        """ALOHA format: cam_high, cam_left_wrist, cam_right_wrist."""
        step = make_mock_lerobot_step("ep1", 0, camera_format="aloha")
        keys = discover_camera_keys(step.obs)

        assert len(keys) == 3
        assert "cam_high" in keys
        assert "cam_left_wrist" in keys
        assert "cam_right_wrist" in keys

    def test_discover_no_cameras(self):
        """Handle steps with no camera observations."""
        keys = discover_camera_keys({"state": [1, 2, 3]})
        assert keys == []

    def test_discover_observation_images_prefix(self):
        """Handle observation.images.* prefix (alternative LeRobot format)."""
        obs = {
            "observation.images.camera0": np.zeros((64, 64, 3)),
            "observation.images.camera1": np.zeros((64, 64, 3)),
            "state": [1, 2, 3],
        }
        keys = discover_camera_keys(obs)

        assert len(keys) == 2
        assert "camera0" in keys
        assert "camera1" in keys


# ---------------------------------------------------------------------------
# Feature extraction tests
# ---------------------------------------------------------------------------


class TestFeatureExtraction:
    """Test feature extraction strategies on CPU."""

    def test_placeholder_features_deterministic(self):
        """Placeholder features should be deterministic given same seed."""
        config = FeatureExtractionConfig(strategy="placeholder", d_feature=128)
        image = np.zeros((64, 64, 3))

        f1 = extract_features(image, config, seed_str="test_seed")
        f2 = extract_features(image, config, seed_str="test_seed")
        f3 = extract_features(image, config, seed_str="different_seed")

        assert torch.allclose(f1, f2)
        assert not torch.allclose(f1, f3)

    def test_placeholder_features_shape(self):
        """Placeholder features have correct shape."""
        config = FeatureExtractionConfig(strategy="placeholder", d_feature=256)
        image = np.zeros((64, 64, 3))

        features = extract_features(image, config, seed_str="test")
        assert features.shape == (256,)

    def test_flattened_features_numpy(self):
        """Flattened features work with numpy arrays."""
        config = FeatureExtractionConfig(strategy="flattened", d_feature=128)
        image = np.random.rand(64, 64, 3).astype(np.float32)

        features = extract_features(image, config)

        assert features.shape == (128,)
        assert features.dtype == torch.float32

    def test_flattened_features_tensor(self):
        """Flattened features work with torch tensors."""
        config = FeatureExtractionConfig(strategy="flattened", d_feature=64)
        image = torch.rand(32, 32, 3)

        features = extract_features(image, config)

        assert features.shape == (64,)

    def test_flattened_features_padding(self):
        """Flattened features pad small images correctly."""
        config = FeatureExtractionConfig(strategy="flattened", d_feature=1000)
        image = np.random.rand(8, 8, 3).astype(np.float32)  # 192 elements

        features = extract_features(image, config)

        assert features.shape == (1000,)
        # Last elements should be padding zeros
        assert features[192:].abs().sum() == 0

    def test_frozen_backbone_not_implemented(self):
        """Frozen backbone raises NotImplementedError (GPU required)."""
        config = FeatureExtractionConfig(strategy="frozen_backbone", d_feature=768)
        image = np.zeros((224, 224, 3))

        with pytest.raises(NotImplementedError, match="requires GPU"):
            extract_features(image, config)

    def test_unknown_strategy_error(self):
        """Unknown strategy raises ValueError."""
        config = FeatureExtractionConfig(strategy="unknown", d_feature=128)
        image = np.zeros((64, 64, 3))

        with pytest.raises(ValueError, match="Unknown feature extraction strategy"):
            extract_features(image, config)


# ---------------------------------------------------------------------------
# MultiProviderSample adapter tests
# ---------------------------------------------------------------------------


class TestMultiProviderSampleAdapter:
    """Test LeRobot step → MultiProviderSample conversion."""

    def test_droid_step_to_multi_provider(self):
        """DROID step produces MultiProviderSample with 3 providers."""
        step = make_mock_lerobot_step("ep1", 5, camera_format="droid")
        config = FeatureExtractionConfig(strategy="placeholder", d_feature=128)

        sample = multi_provider_sample_from_lerobot_step(step, feature_config=config)

        assert isinstance(sample, MultiProviderSample)
        assert sample.sample_id == step.record_id
        assert sample.scene_id == step.episode_id
        assert sample.frame_idx == 5
        assert len(sample.providers) == 3

    def test_bridge_step_to_multi_provider(self):
        """Bridge V2 step produces MultiProviderSample with 4 providers."""
        step = make_mock_lerobot_step("ep1", 0, camera_format="bridge")
        config = FeatureExtractionConfig(strategy="flattened", d_feature=64)

        sample = multi_provider_sample_from_lerobot_step(step, feature_config=config)

        assert len(sample.providers) == 4
        for provider in sample.providers:
            assert provider.provider_kind == "vision_backbone"
            assert provider.features.shape == (64,)

    def test_providers_have_correct_structure(self):
        """Each provider has correct fields."""
        step = make_mock_lerobot_step("ep1", 0, camera_format="droid")

        sample = multi_provider_sample_from_lerobot_step(step)

        for provider in sample.providers:
            assert isinstance(provider, ProviderObservation)
            assert provider.provider_id is not None
            assert provider.provider_kind == "vision_backbone"
            assert provider.availability_status in ("available", "unavailable")
            assert isinstance(provider.features, torch.Tensor)

    def test_task_success_from_reward(self):
        """Task success is derived from reward when not explicitly provided."""
        step = make_mock_lerobot_step("ep1", 9, camera_format="droid", reward=1.0)

        sample = multi_provider_sample_from_lerobot_step(step)

        assert sample.downstream_task_success == 1.0

    def test_task_success_explicit(self):
        """Explicit task success overrides reward."""
        step = make_mock_lerobot_step("ep1", 0, camera_format="droid", reward=0.5)

        sample = multi_provider_sample_from_lerobot_step(
            step, downstream_task_success=0.8
        )

        assert sample.downstream_task_success == 0.8

    def test_metadata_preserved(self):
        """Sample metadata includes task/env/source info."""
        step = make_mock_lerobot_step("ep1", 0, camera_format="droid")

        sample = multi_provider_sample_from_lerobot_step(step)

        assert sample.metadata["task_id"] == "pick_and_place"
        assert sample.metadata["env_id"] == "tabletop"
        assert sample.metadata["source_domain"] == "lerobot_test"

    def test_explicit_camera_keys(self):
        """Explicit camera keys override auto-discovery."""
        step = make_mock_lerobot_step("ep1", 0, camera_format="droid")

        sample = multi_provider_sample_from_lerobot_step(
            step, camera_keys=["exterior_image_1_left", "wrist_image_left"]
        )

        assert len(sample.providers) == 2

    def test_missing_camera_marked_unavailable(self):
        """Cameras not in obs are marked unavailable."""
        step = make_mock_lerobot_step("ep1", 0, camera_format="droid")

        sample = multi_provider_sample_from_lerobot_step(
            step, camera_keys=["exterior_image_1_left", "nonexistent_camera"]
        )

        providers_by_id = {p.provider_id: p for p in sample.providers}
        assert providers_by_id["exterior_image_1_left"].availability_status == "available"
        assert providers_by_id["nonexistent_camera"].availability_status == "unavailable"

    def test_no_cameras_raises_error(self):
        """Step with no discoverable cameras raises error."""
        step = ReplayStepRecord(
            run_id="test",
            episode_id="ep1",
            step_idx=0,
            obs={"state": [1, 2, 3]},  # No images
            obs_vector=[],
            action={},
            action_vector=[],
            reward=0.0,
            reward_decomposition={},
            done=False,
            task_id="test",
            env_id="test",
            condition_vector={},
            condition_vector_values=[],
            skill_mode="test",
            objective_tensor_summary={},
            objective_tensor_ref=None,
            econ_tensor_summary={},
            econ_tensor_ref=None,
            constraint_flags=[],
            pricing_tick_ref=None,
            ledger_event_ref=None,
            source_domain="test",
            seed=0,
            timestamp="",
            metadata={},
            provenance={},
        )

        with pytest.raises(ValueError, match="No camera keys found"):
            multi_provider_sample_from_lerobot_step(step)


class TestMultiProviderSamplesFromEpisode:
    """Test episode → MultiProviderSample list conversion."""

    def test_full_episode_conversion(self):
        """Full episode produces samples for all steps."""
        episode, steps = make_mock_lerobot_episode(num_steps=10, camera_format="droid")

        samples = multi_provider_samples_from_episode(episode, steps)

        assert len(samples) == 10
        for i, sample in enumerate(samples):
            assert sample.frame_idx == i

    def test_stride_subsampling(self):
        """Stride parameter subsamples steps."""
        episode, steps = make_mock_lerobot_episode(num_steps=10)

        samples = multi_provider_samples_from_episode(episode, steps, stride=3)

        assert len(samples) == 4  # indices 0, 3, 6, 9
        assert [s.frame_idx for s in samples] == [0, 3, 6, 9]

    def test_max_samples_limit(self):
        """max_samples limits output."""
        episode, steps = make_mock_lerobot_episode(num_steps=100)

        samples = multi_provider_samples_from_episode(episode, steps, max_samples=5)

        assert len(samples) == 5


# ---------------------------------------------------------------------------
# VisionBackboneProjectionSample adapter tests
# ---------------------------------------------------------------------------


class TestVisionBackboneProjectionSampleAdapter:
    """Test LeRobot step → VisionBackboneProjectionSample conversion."""

    def test_droid_step_to_projection_sample(self):
        step = make_mock_lerobot_step("ep1", 5, camera_format="droid")
        config = FeatureExtractionConfig(strategy="placeholder", d_feature=1024)

        sample = vision_backbone_projection_sample_from_lerobot_step(
            step,
            feature_config=config,
        )

        assert isinstance(sample, VisionBackboneProjectionSample)
        assert sample.sample_id == step.record_id
        assert sample.backbone_features.shape == (12, 1024)
        assert sample.object_identity_labels.tolist() == [0] * 4 + [1] * 4 + [2] * 4
        assert sample.cross_provider_embeddings is not None
        assert sample.cross_provider_embeddings.shape == (12, 128)

    def test_projection_tokens_are_deterministic(self):
        step = make_mock_lerobot_step("ep1", 0, camera_format="droid")

        sample1 = vision_backbone_projection_sample_from_lerobot_step(step)
        sample2 = vision_backbone_projection_sample_from_lerobot_step(step)

        assert torch.allclose(sample1.backbone_features, sample2.backbone_features)
        assert torch.allclose(
            sample1.cross_provider_embeddings,
            sample2.cross_provider_embeddings,
        )

    def test_projection_episode_conversion_respects_stride(self):
        episode, steps = make_mock_lerobot_episode(num_steps=10, camera_format="droid")

        samples = vision_backbone_projection_samples_from_episode(
            episode,
            steps,
            stride=3,
        )

        assert len(samples) == 4
        assert all(isinstance(sample, VisionBackboneProjectionSample) for sample in samples)


# ---------------------------------------------------------------------------
# VJEPATemporalSample adapter tests
# ---------------------------------------------------------------------------


class TestVJEPATemporalSampleAdapter:
    """Test LeRobot episode window → VJEPATemporalSample conversion."""

    def test_window_extraction(self):
        """Valid window produces VJEPATemporalSample."""
        _, steps = make_mock_lerobot_episode(num_steps=20)

        sample = vjepa_temporal_sample_from_episode_window(
            steps, window_start=5, window_size=4
        )

        assert isinstance(sample, VJEPATemporalSample)
        assert sample.sample_id.endswith("_w5")
        assert sample.vjepa_tokens.shape[0] == 4  # T=4

    def test_window_shapes(self):
        """Sample tensors have correct shapes."""
        _, steps = make_mock_lerobot_episode(num_steps=20)

        sample = vjepa_temporal_sample_from_episode_window(
            steps,
            window_start=0,
            window_size=8,
            n_objects=5,
            d_vjepa=512,
            d_wm=64,
            d_out=128,
        )

        assert sample.vjepa_tokens.shape == (8, 196, 512)
        assert sample.wm_object_tokens.shape == (5, 64)
        assert sample.future_object_states.shape == (8, 5, 128)
        assert sample.object_valid_mask.shape == (5,)
        assert sample.temporal_ordering_labels.shape == (8,)

    def test_invalid_window_returns_none(self):
        """Invalid window (out of bounds) returns None."""
        _, steps = make_mock_lerobot_episode(num_steps=10)

        # Window extends past end
        sample = vjepa_temporal_sample_from_episode_window(
            steps, window_start=8, window_size=5
        )
        assert sample is None

        # Negative start
        sample = vjepa_temporal_sample_from_episode_window(
            steps, window_start=-1, window_size=4
        )
        assert sample is None

    def test_deterministic_tokens(self):
        """Tokens are deterministic given same episode/step."""
        _, steps = make_mock_lerobot_episode(num_steps=10)

        sample1 = vjepa_temporal_sample_from_episode_window(
            steps, window_start=2, window_size=4
        )
        sample2 = vjepa_temporal_sample_from_episode_window(
            steps, window_start=2, window_size=4
        )

        assert torch.allclose(sample1.vjepa_tokens, sample2.vjepa_tokens)
        assert torch.allclose(sample1.wm_object_tokens, sample2.wm_object_tokens)

    def test_temporal_ordering_labels(self):
        """Temporal ordering labels are sequential."""
        _, steps = make_mock_lerobot_episode(num_steps=20)

        sample = vjepa_temporal_sample_from_episode_window(
            steps, window_start=5, window_size=6
        )

        expected = torch.arange(6)
        assert torch.equal(sample.temporal_ordering_labels, expected)


class TestVJEPATemporalSamplesFromEpisode:
    """Test episode → VJEPATemporalSample list conversion."""

    def test_sliding_windows(self):
        """Episode produces sliding window samples."""
        episode, steps = make_mock_lerobot_episode(num_steps=20)

        samples = vjepa_temporal_samples_from_episode(
            episode, steps, window_size=4, stride=2
        )

        # Windows at 0, 2, 4, 6, 8, 10, 12, 14, 16 (9 windows)
        assert len(samples) == 9

    def test_max_samples_limit(self):
        """max_samples limits output."""
        episode, steps = make_mock_lerobot_episode(num_steps=100)

        samples = vjepa_temporal_samples_from_episode(
            episode, steps, window_size=4, stride=2, max_samples=5
        )

        assert len(samples) == 5

    def test_short_episode(self):
        """Episode shorter than window produces no samples."""
        episode, steps = make_mock_lerobot_episode(num_steps=3)

        samples = vjepa_temporal_samples_from_episode(
            episode, steps, window_size=4, stride=2
        )

        assert len(samples) == 0


# ---------------------------------------------------------------------------
# Dataset-level adapter tests
# ---------------------------------------------------------------------------


class TestDatasetLevelAdapters:
    """Test batch adaptation of multiple episodes."""

    def test_adapt_for_evidence_fusion(self):
        """Multiple episodes adapt to MultiProviderSamples."""
        episodes = [
            make_mock_lerobot_episode(num_steps=5, camera_format="droid"),
            make_mock_lerobot_episode(num_steps=7, camera_format="droid"),
            make_mock_lerobot_episode(num_steps=3, camera_format="droid"),
        ]

        samples = adapt_lerobot_episodes_for_evidence_fusion(episodes)

        assert len(samples) == 5 + 7 + 3
        assert all(isinstance(s, MultiProviderSample) for s in samples)

    def test_adapt_for_evidence_fusion_with_config(self):
        """Config controls sampling parameters."""
        episodes = [
            make_mock_lerobot_episode(num_steps=10, camera_format="bridge"),
        ]

        config = LeRobotPerceptionAdapterConfig(
            step_stride=2,
            max_samples_per_episode=3,
        )

        samples = adapt_lerobot_episodes_for_evidence_fusion(episodes, config)

        assert len(samples) == 3

    def test_adapt_for_vjepa_temporal(self):
        """Multiple episodes adapt to VJEPATemporalSamples."""
        episodes = [
            make_mock_lerobot_episode(num_steps=20, camera_format="droid"),
            make_mock_lerobot_episode(num_steps=15, camera_format="droid"),
        ]

        config = LeRobotPerceptionAdapterConfig(
            temporal_window_size=4,
            temporal_stride=4,
        )

        samples = adapt_lerobot_episodes_for_vjepa_temporal(episodes, config)

        # 20 steps, window=4, stride=4 → 5 windows
        # 15 steps, window=4, stride=4 → 3 windows
        assert len(samples) == 5 + 3

    def test_adapt_for_vision_backbone_projection(self):
        episodes = [
            make_mock_lerobot_episode(num_steps=5, camera_format="droid"),
            make_mock_lerobot_episode(num_steps=7, camera_format="droid"),
        ]
        config = LeRobotPerceptionAdapterConfig(
            feature_config=FeatureExtractionConfig(
                strategy="placeholder",
                d_feature=1024,
            ),
            step_stride=2,
            max_samples_per_episode=3,
            projection_tokens_per_camera=4,
            d_out=128,
        )

        samples = adapt_lerobot_episodes_for_vision_backbone_projection(
            episodes,
            config,
        )

        assert len(samples) == 3 + 3
        assert all(
            isinstance(sample, VisionBackboneProjectionSample)
            for sample in samples
        )


# ---------------------------------------------------------------------------
# Integration tests with realistic data shapes
# ---------------------------------------------------------------------------


class TestRealisticDataIntegration:
    """Test adapters with realistic DROID/Bridge data shapes."""

    def test_droid_realistic_shapes(self):
        """DROID-like data: 180x320 RGB, 3 cameras, 15 FPS."""
        step = make_mock_lerobot_step(
            "droid_ep",
            0,
            camera_format="droid",
            image_shape=(180, 320, 3),
        )

        config = FeatureExtractionConfig(strategy="flattened", d_feature=256)
        sample = multi_provider_sample_from_lerobot_step(step, feature_config=config)

        assert len(sample.providers) == 3
        for p in sample.providers:
            assert p.features.shape == (256,)

    def test_bridge_realistic_shapes(self):
        """Bridge V2-like data: 256x256 RGB, 4 cameras, 5 FPS."""
        step = make_mock_lerobot_step(
            "bridge_ep",
            0,
            camera_format="bridge",
        )

        config = FeatureExtractionConfig(strategy="flattened", d_feature=512)
        sample = multi_provider_sample_from_lerobot_step(step, feature_config=config)

        assert len(sample.providers) == 4
        for p in sample.providers:
            assert p.features.shape == (512,)

    def test_droid_episode_temporal_samples(self):
        """DROID episode (avg ~400 frames) produces many temporal samples."""
        episode, steps = make_mock_lerobot_episode(
            num_steps=100,  # Smaller than real DROID but realistic for test
            camera_format="droid",
        )

        # Match typical DROID params
        samples = vjepa_temporal_samples_from_episode(
            episode,
            steps,
            window_size=8,  # ~0.5s at 15 FPS
            stride=4,
            n_objects=10,
            d_vjepa=1024,  # ViT-L dimension
        )

        # 100 steps, window=8, stride=4 → 24 windows (starts at 0,4,8,...,92)
        assert len(samples) == 24
        assert samples[0].vjepa_tokens.shape == (8, 196, 1024)

    def test_mixed_dataset_batch(self):
        """Batch from mixed DROID + Bridge datasets."""
        droid_eps = [
            make_mock_lerobot_episode(num_steps=20, camera_format="droid"),
            make_mock_lerobot_episode(num_steps=15, camera_format="droid"),
        ]
        bridge_eps = [
            make_mock_lerobot_episode(num_steps=10, camera_format="bridge"),
        ]

        # Evidence fusion works on both (different camera counts)
        droid_samples = adapt_lerobot_episodes_for_evidence_fusion(droid_eps)
        bridge_samples = adapt_lerobot_episodes_for_evidence_fusion(bridge_eps)

        # DROID has 3 cameras per sample
        assert all(len(s.providers) == 3 for s in droid_samples)
        # Bridge has 4 cameras per sample
        assert all(len(s.providers) == 4 for s in bridge_samples)


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_episode(self):
        """Empty step list produces empty samples."""
        episode, _ = make_mock_lerobot_episode(num_steps=1)

        samples = multi_provider_samples_from_episode(episode, [])
        assert samples == []

    def test_single_step_episode(self):
        """Single-step episode works for multi-provider but not temporal."""
        episode, steps = make_mock_lerobot_episode(num_steps=1)

        mp_samples = multi_provider_samples_from_episode(episode, steps)
        assert len(mp_samples) == 1

        vjepa_samples = vjepa_temporal_samples_from_episode(
            episode, steps, window_size=4
        )
        assert len(vjepa_samples) == 0

    def test_unsorted_steps(self):
        """Steps are sorted by step_idx before processing."""
        episode, steps = make_mock_lerobot_episode(num_steps=10)

        # Shuffle steps
        import random
        shuffled = steps.copy()
        random.shuffle(shuffled)

        samples = multi_provider_samples_from_episode(episode, shuffled)

        # Should still be in order
        assert [s.frame_idx for s in samples] == list(range(10))

    def test_config_post_init(self):
        """LeRobotPerceptionAdapterConfig defaults are set."""
        config = LeRobotPerceptionAdapterConfig()

        assert config.feature_config is not None
        assert isinstance(config.feature_config, FeatureExtractionConfig)
        assert config.temporal_window_size == 4
        assert config.n_objects == 10
