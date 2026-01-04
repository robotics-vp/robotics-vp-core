"""
Dataset and Training Pairs for Process Reward.

Provides dataset construction for training HopNet and FusionNet:
- Before/after frame pairs with hop labels
- Multi-source label aggregation
- Dataloader-compatible interface
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np

from src.process_reward.schemas import (
    ProcessRewardConfig,
    EpisodeFeatures,
    FrameFeatures,
    HopLabel,
)
from src.process_reward.features import FeatureExtractor
from src.process_reward.hop_model import (
    LabelProvider,
    OracleDistanceLabelProvider,
    TaskSuccessLabelProvider,
)


@dataclass
class HopTrainingExample:
    """A single training example for HopNet.

    Attributes:
        before_features: Pooled features at before frame.
        after_features: Pooled features at after frame.
        init_features: Pooled features at init frame.
        goal_features: Pooled features at goal frame.
        instruction_embedding: Instruction embedding.
        hop_label: Ground truth hop value.
        label_confidence: Confidence in the label.
        label_source: Source of the label.
        episode_id: Source episode identifier.
        before_idx: Frame index of before state.
        after_idx: Frame index of after state.
    """
    before_features: np.ndarray  # (feature_dim,)
    after_features: np.ndarray  # (feature_dim,)
    init_features: np.ndarray  # (feature_dim,)
    goal_features: np.ndarray  # (feature_dim,)
    instruction_embedding: np.ndarray  # (instruction_dim,)
    hop_label: float
    label_confidence: float = 1.0
    label_source: str = "oracle"
    episode_id: str = ""
    before_idx: int = 0
    after_idx: int = 0


@dataclass
class FusionTrainingExample:
    """A single training example for FusionNet.

    Attributes:
        phi_I: Incremental perspective value.
        phi_F: Forward perspective value.
        phi_B: Backward perspective value.
        conf_I: Incremental confidence.
        conf_F: Forward confidence.
        conf_B: Backward confidence.
        disagreement: Perspective disagreement.
        t_ratio: Timestep ratio (t/T).
        context: Context features.
        mhn_features: Optional MHN features.
        target_weights: Optional ground truth weights (for supervised learning).
        target_phi: Optional ground truth fused phi.
        target_conf: Optional ground truth confidence.
    """
    phi_I: float
    phi_F: float
    phi_B: float
    conf_I: float
    conf_F: float
    conf_B: float
    disagreement: float
    t_ratio: float
    context: np.ndarray  # (context_dim,)
    mhn_features: Optional[np.ndarray] = None  # (5,)
    target_weights: Optional[np.ndarray] = None  # (3,)
    target_phi: Optional[float] = None
    target_conf: Optional[float] = None


class HopDataset:
    """Dataset for training HopNet.

    Collects before/after pairs from episodes with labels from LabelProviders.
    """

    def __init__(
        self,
        config: ProcessRewardConfig,
        label_providers: Optional[List[LabelProvider]] = None,
        sample_stride: int = 1,
        max_pairs_per_episode: Optional[int] = None,
    ):
        """Initialize.

        Args:
            config: Process reward configuration.
            label_providers: List of label providers. If None, uses oracle.
            sample_stride: Stride for sampling pairs (1 = all consecutive pairs).
            max_pairs_per_episode: Maximum pairs to sample per episode.
        """
        self.config = config
        self.label_providers = label_providers or [OracleDistanceLabelProvider()]
        self.sample_stride = sample_stride
        self.max_pairs_per_episode = max_pairs_per_episode

        self.examples: List[HopTrainingExample] = []
        self._feature_extractor = FeatureExtractor(config)

    def add_episode(
        self,
        scene_tracks_lite: Any,
        instruction_embedding: np.ndarray,
        goal_frame_idx: Optional[int] = None,
        episode_id: str = "",
        extra_labels: Optional[List[HopLabel]] = None,
    ) -> int:
        """Add an episode to the dataset.

        Args:
            scene_tracks_lite: Deserialized scene tracks.
            instruction_embedding: Instruction embedding.
            goal_frame_idx: Optional goal frame index.
            episode_id: Episode identifier.
            extra_labels: Optional additional labels.

        Returns:
            Number of examples added.
        """
        from src.process_reward.features import extract_features_from_scene_tracks_lite

        # Extract features
        episode_features = extract_features_from_scene_tracks_lite(
            scene_tracks_lite,
            self.config,
            goal_frame_idx=goal_frame_idx,
        )

        T = len(episode_features.frame_features)
        if T < 2:
            return 0

        # Get init and goal features
        init_features = episode_features.init_features.pooled
        goal_features = (
            episode_features.goal_features.pooled
            if episode_features.goal_features is not None
            else episode_features.frame_features[-1].pooled
        )

        # Collect labels from all providers
        all_labels: Dict[Tuple[int, int], List[HopLabel]] = {}

        for provider in self.label_providers:
            labels = provider.get_labels(episode_features, goal_idx=goal_frame_idx)
            for label in labels:
                key = (label.before_idx, label.after_idx)
                if key not in all_labels:
                    all_labels[key] = []
                all_labels[key].append(label)

        # Add extra labels
        if extra_labels:
            for label in extra_labels:
                key = (label.before_idx, label.after_idx)
                if key not in all_labels:
                    all_labels[key] = []
                all_labels[key].append(label)

        # Aggregate labels and create examples
        count = 0
        pairs = list(all_labels.keys())

        # Apply stride and max limit
        if self.sample_stride > 1:
            pairs = pairs[::self.sample_stride]
        if self.max_pairs_per_episode:
            pairs = pairs[:self.max_pairs_per_episode]

        for before_idx, after_idx in pairs:
            labels = all_labels[(before_idx, after_idx)]

            # Aggregate: weighted average by confidence
            total_weight = sum(l.confidence for l in labels)
            if total_weight == 0:
                continue

            hop_value = sum(l.hop_value * l.confidence for l in labels) / total_weight
            avg_confidence = total_weight / len(labels)

            # Determine primary source
            source = max(labels, key=lambda l: l.confidence).source

            example = HopTrainingExample(
                before_features=episode_features.frame_features[before_idx].pooled.copy(),
                after_features=episode_features.frame_features[after_idx].pooled.copy(),
                init_features=init_features.copy(),
                goal_features=goal_features.copy(),
                instruction_embedding=instruction_embedding.copy(),
                hop_label=hop_value,
                label_confidence=avg_confidence,
                label_source=source,
                episode_id=episode_id,
                before_idx=before_idx,
                after_idx=after_idx,
            )

            self.examples.append(example)
            count += 1

        return count

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> HopTrainingExample:
        return self.examples[idx]

    def get_batch(
        self,
        batch_size: int,
        shuffle: bool = True,
    ) -> Iterator[List[HopTrainingExample]]:
        """Iterate over batches.

        Args:
            batch_size: Batch size.
            shuffle: Whether to shuffle.

        Yields:
            Batches of HopTrainingExample.
        """
        indices = list(range(len(self.examples)))
        if shuffle:
            random.shuffle(indices)

        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i + batch_size]
            yield [self.examples[j] for j in batch_indices]

    def to_numpy_batch(
        self,
        examples: List[HopTrainingExample],
    ) -> Dict[str, np.ndarray]:
        """Convert batch of examples to numpy arrays.

        Args:
            examples: List of examples.

        Returns:
            Dict of numpy arrays suitable for training.
        """
        return {
            "before": np.stack([e.before_features for e in examples]),
            "after": np.stack([e.after_features for e in examples]),
            "init": np.stack([e.init_features for e in examples]),
            "goal": np.stack([e.goal_features for e in examples]),
            "instruction": np.stack([e.instruction_embedding for e in examples]),
            "hop_label": np.array([e.hop_label for e in examples], dtype=np.float32),
            "label_confidence": np.array([e.label_confidence for e in examples], dtype=np.float32),
        }

    def save(self, path: Union[str, Path]) -> None:
        """Save dataset to disk.

        Args:
            path: Path to save (without extension).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save features as npz
        np.savez_compressed(
            f"{path}_features.npz",
            before=np.stack([e.before_features for e in self.examples]),
            after=np.stack([e.after_features for e in self.examples]),
            init=np.stack([e.init_features for e in self.examples]),
            goal=np.stack([e.goal_features for e in self.examples]),
            instruction=np.stack([e.instruction_embedding for e in self.examples]),
            hop_label=np.array([e.hop_label for e in self.examples]),
            label_confidence=np.array([e.label_confidence for e in self.examples]),
        )

        # Save metadata as JSON
        metadata = []
        for e in self.examples:
            metadata.append({
                "episode_id": e.episode_id,
                "before_idx": e.before_idx,
                "after_idx": e.after_idx,
                "label_source": e.label_source,
            })

        with open(f"{path}_metadata.json", "w") as f:
            json.dump(metadata, f)

    @classmethod
    def load(cls, path: Union[str, Path], config: ProcessRewardConfig) -> "HopDataset":
        """Load dataset from disk.

        Args:
            path: Path to load (without extension).
            config: Process reward configuration.

        Returns:
            Loaded HopDataset.
        """
        path = Path(path)

        # Load features
        data = np.load(f"{path}_features.npz")

        # Load metadata
        with open(f"{path}_metadata.json", "r") as f:
            metadata = json.load(f)

        dataset = cls(config)

        for i, meta in enumerate(metadata):
            example = HopTrainingExample(
                before_features=data["before"][i],
                after_features=data["after"][i],
                init_features=data["init"][i],
                goal_features=data["goal"][i],
                instruction_embedding=data["instruction"][i],
                hop_label=float(data["hop_label"][i]),
                label_confidence=float(data["label_confidence"][i]),
                label_source=meta["label_source"],
                episode_id=meta["episode_id"],
                before_idx=meta["before_idx"],
                after_idx=meta["after_idx"],
            )
            dataset.examples.append(example)

        return dataset

    def statistics(self) -> Dict[str, Any]:
        """Get dataset statistics.

        Returns:
            Dictionary of statistics.
        """
        if not self.examples:
            return {"num_examples": 0}

        hops = np.array([e.hop_label for e in self.examples])
        confs = np.array([e.label_confidence for e in self.examples])
        sources = [e.label_source for e in self.examples]

        source_counts = {}
        for s in sources:
            source_counts[s] = source_counts.get(s, 0) + 1

        return {
            "num_examples": len(self.examples),
            "hop_mean": float(np.mean(hops)),
            "hop_std": float(np.std(hops)),
            "hop_min": float(np.min(hops)),
            "hop_max": float(np.max(hops)),
            "confidence_mean": float(np.mean(confs)),
            "confidence_min": float(np.min(confs)),
            "source_counts": source_counts,
        }


class FusionDataset:
    """Dataset for training FusionNet.

    Collects perspective values with target fusion outputs.
    """

    def __init__(self, config: ProcessRewardConfig):
        """Initialize.

        Args:
            config: Process reward configuration.
        """
        self.config = config
        self.examples: List[FusionTrainingExample] = []

    def add_from_episode_output(
        self,
        output: "ProcessRewardEpisodeOutput",
        target_weights: Optional[np.ndarray] = None,
    ) -> int:
        """Add examples from an episode output.

        Args:
            output: ProcessRewardEpisodeOutput.
            target_weights: Optional target weights (T, 3) for supervised learning.

        Returns:
            Number of examples added.
        """
        T = len(output.phi_star)
        count = 0

        for t in range(T):
            example = FusionTrainingExample(
                phi_I=float(output.perspectives.phi_I[t]),
                phi_F=float(output.perspectives.phi_F[t]),
                phi_B=float(output.perspectives.phi_B[t]),
                conf_I=float(output.perspectives.conf_I[t]),
                conf_F=float(output.perspectives.conf_F[t]),
                conf_B=float(output.perspectives.conf_B[t]),
                disagreement=float(output.diagnostics.disagreement[t]),
                t_ratio=t / max(T - 1, 1),
                context=np.zeros(8, dtype=np.float32),  # Placeholder
                target_phi=float(output.phi_star[t]),
                target_conf=float(output.conf[t]),
                target_weights=target_weights[t] if target_weights is not None else None,
            )
            self.examples.append(example)
            count += 1

        return count

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> FusionTrainingExample:
        return self.examples[idx]


def create_dataset_from_npz_files(
    npz_paths: List[Union[str, Path]],
    config: ProcessRewardConfig,
    instruction_embedding: Optional[np.ndarray] = None,
    label_providers: Optional[List[LabelProvider]] = None,
    max_episodes: Optional[int] = None,
) -> HopDataset:
    """Create HopDataset from exported NPZ files.

    Args:
        npz_paths: List of paths to NPZ files with scene_tracks_v1.
        config: Process reward configuration.
        instruction_embedding: Default instruction embedding to use.
        label_providers: Label providers for generating labels.
        max_episodes: Maximum number of episodes to load.

    Returns:
        HopDataset with examples from all files.
    """
    from src.vision.scene_ir_tracker.serialization import deserialize_scene_tracks_v1

    dataset = HopDataset(config, label_providers)

    if instruction_embedding is None:
        instruction_embedding = np.random.randn(config.instruction_embedding_dim).astype(np.float32)

    for i, path in enumerate(npz_paths):
        if max_episodes and i >= max_episodes:
            break

        try:
            data = dict(np.load(path, allow_pickle=False))
            scene_tracks = deserialize_scene_tracks_v1(data)

            episode_id = Path(path).stem
            dataset.add_episode(
                scene_tracks,
                instruction_embedding,
                episode_id=episode_id,
            )
        except Exception as e:
            import warnings
            warnings.warn(f"Failed to load {path}: {e}")
            continue

    return dataset
