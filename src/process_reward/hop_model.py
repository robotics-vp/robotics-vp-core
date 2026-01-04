"""
Hop Model (HopNet) for Process Reward.

Predicts "hop" (progress) between BEFORE and AFTER states, conditioned on init, goal, and instruction.
Supports multiple label sources via LabelProvider abstraction.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


from src.process_reward.schemas import (
    ProcessRewardConfig,
    FrameFeatures,
    EpisodeFeatures,
    HopLabel,
)


# -----------------------------------------------------------------------------
# Label Provider Abstraction
# -----------------------------------------------------------------------------

class LabelProvider(abc.ABC):
    """Abstract base class for hop label providers.

    Different label sources can implement this interface:
    - OracleDistanceLabelProvider: Uses feature-space distance to goal as proxy.
    - HumanLabelProvider: Human annotations.
    - LLMLabelProvider: LLM-generated labels.
    - TaskSuccessLabelProvider: Sparse task success signals.
    """

    @abc.abstractmethod
    def get_labels(
        self,
        episode_features: EpisodeFeatures,
        init_idx: int = 0,
        goal_idx: Optional[int] = None,
    ) -> List[HopLabel]:
        """Get hop labels for an episode.

        Args:
            episode_features: Features for the episode.
            init_idx: Index of initial frame.
            goal_idx: Optional index of goal frame.

        Returns:
            List of HopLabel for transitions in the episode.
        """
        pass


class OracleDistanceLabelProvider(LabelProvider):
    """Provides oracle labels based on feature-space distance to goal.

    This is a bootstrap method that uses the change in distance to goal
    as a proxy for progress.
    """

    def __init__(self, normalize: bool = True):
        """Initialize.

        Args:
            normalize: If True, normalize hop values to [-1, 1].
        """
        self.normalize = normalize

    def get_labels(
        self,
        episode_features: EpisodeFeatures,
        init_idx: int = 0,
        goal_idx: Optional[int] = None,
    ) -> List[HopLabel]:
        """Compute hop labels from feature distance to goal.

        hop[t] = (dist_to_goal[t] - dist_to_goal[t+1]) / max_dist
        Positive = moved closer to goal, Negative = moved away.

        Args:
            episode_features: Features for the episode.
            init_idx: Index of initial frame.
            goal_idx: Optional index of goal frame.

        Returns:
            List of HopLabel for each transition.
        """
        T = len(episode_features.frame_features)
        if T < 2:
            return []

        # Use goal features if available, else use last frame
        if goal_idx is not None and episode_features.goal_features is not None:
            goal_features = episode_features.goal_features.pooled
        else:
            goal_features = episode_features.frame_features[-1].pooled
            goal_idx = T - 1

        # Compute distances to goal
        distances = []
        for ff in episode_features.frame_features:
            dist = np.linalg.norm(ff.pooled - goal_features)
            distances.append(dist)
        distances = np.array(distances)

        # Max distance for normalization
        max_dist = max(distances.max(), 1e-6)

        # Compute hops
        labels = []
        for t in range(T - 1):
            # Hop = reduction in distance to goal
            hop_value = (distances[t] - distances[t + 1]) / max_dist

            if self.normalize:
                hop_value = np.clip(hop_value, -1.0, 1.0)

            labels.append(HopLabel(
                hop_value=float(hop_value),
                source="oracle",
                confidence=1.0,
                before_idx=t,
                after_idx=t + 1,
                metadata={"dist_before": float(distances[t]), "dist_after": float(distances[t + 1])},
            ))

        return labels


class TaskSuccessLabelProvider(LabelProvider):
    """Provides sparse labels based on task success signals.

    Distributes success signal across the episode based on position.
    """

    def __init__(
        self,
        success_value: float = 1.0,
        failure_value: float = -1.0,
        distribution: Literal["uniform", "linear", "final"] = "linear",
    ):
        """Initialize.

        Args:
            success_value: Value assigned for successful episodes.
            failure_value: Value assigned for failed episodes.
            distribution: How to distribute the signal across transitions.
        """
        self.success_value = success_value
        self.failure_value = failure_value
        self.distribution = distribution

    def get_labels(
        self,
        episode_features: EpisodeFeatures,
        init_idx: int = 0,
        goal_idx: Optional[int] = None,
        success: Optional[bool] = None,
    ) -> List[HopLabel]:
        """Compute hop labels from task success.

        Args:
            episode_features: Features for the episode.
            init_idx: Index of initial frame.
            goal_idx: Optional index of goal frame.
            success: Whether episode succeeded (from metadata).

        Returns:
            List of HopLabel for each transition.
        """
        T = len(episode_features.frame_features)
        if T < 2:
            return []

        # Use success flag from global stats if not provided
        if success is None:
            success = episode_features.global_stats.get("success", False)

        total_value = self.success_value if success else self.failure_value

        labels = []
        for t in range(T - 1):
            if self.distribution == "uniform":
                hop_value = total_value / (T - 1)
            elif self.distribution == "linear":
                # More weight towards the end
                weight = (t + 1) / sum(range(1, T))
                hop_value = total_value * weight
            else:  # final
                hop_value = total_value if t == T - 2 else 0.0

            labels.append(HopLabel(
                hop_value=float(hop_value),
                source="task_success",
                confidence=0.5,  # Lower confidence for sparse signals
                before_idx=t,
                after_idx=t + 1,
                metadata={"success": success},
            ))

        return labels


class ProxyLabelProvider(LabelProvider):
    """Provides proxy labels from a custom function.

    Useful for domain-specific heuristics.
    """

    def __init__(self, label_fn: Callable[[EpisodeFeatures, int, Optional[int]], List[HopLabel]]):
        """Initialize.

        Args:
            label_fn: Function that takes (episode_features, init_idx, goal_idx)
                and returns list of HopLabel.
        """
        self.label_fn = label_fn

    def get_labels(
        self,
        episode_features: EpisodeFeatures,
        init_idx: int = 0,
        goal_idx: Optional[int] = None,
    ) -> List[HopLabel]:
        """Delegate to custom function."""
        return self.label_fn(episode_features, init_idx, goal_idx)


# -----------------------------------------------------------------------------
# HopNet Model
# -----------------------------------------------------------------------------

if TORCH_AVAILABLE:

    class HopNet(nn.Module):
        """Neural network for predicting hop (progress) between states.

        Consumes:
            - Pooled features for before/after states.
            - Anchor features for init and goal.
            - Instruction embedding.

        Outputs:
            - hop_hat: Predicted hop in [-1, 1].
            - hop_uncertainty: Uncertainty estimate (log variance or confidence head).
        """

        def __init__(
            self,
            feature_dim: int = 32,
            instruction_dim: int = 64,
            hidden_dim: int = 128,
            num_layers: int = 3,
            output_uncertainty: bool = True,
            dropout: float = 0.1,
        ):
            """Initialize HopNet.

            Args:
                feature_dim: Dimension of state features.
                instruction_dim: Dimension of instruction embedding.
                hidden_dim: Hidden layer dimension.
                num_layers: Number of hidden layers.
                output_uncertainty: If True, output uncertainty estimate.
                dropout: Dropout probability.
            """
            super().__init__()

            self.feature_dim = feature_dim
            self.instruction_dim = instruction_dim
            self.output_uncertainty = output_uncertainty

            # Input: [before, after, init, goal, instruction]
            # Each state feature is feature_dim, instruction is instruction_dim
            input_dim = 4 * feature_dim + instruction_dim

            # Build MLP
            layers = []
            in_dim = input_dim
            for i in range(num_layers):
                layers.append(nn.Linear(in_dim, hidden_dim))
                layers.append(nn.LayerNorm(hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                in_dim = hidden_dim

            self.encoder = nn.Sequential(*layers)

            # Output heads
            self.hop_head = nn.Linear(hidden_dim, 1)  # hop prediction

            if output_uncertainty:
                self.uncertainty_head = nn.Linear(hidden_dim, 1)  # log variance or confidence
            else:
                self.uncertainty_head = None

        def forward(
            self,
            before: torch.Tensor,  # (B, feature_dim)
            after: torch.Tensor,  # (B, feature_dim)
            init: torch.Tensor,  # (B, feature_dim)
            goal: torch.Tensor,  # (B, feature_dim)
            instruction: torch.Tensor,  # (B, instruction_dim)
        ) -> Dict[str, torch.Tensor]:
            """Forward pass.

            Args:
                before: Features for before state.
                after: Features for after state.
                init: Features for initial state.
                goal: Features for goal state.
                instruction: Instruction embedding.

            Returns:
                Dict with "hop_hat" and optionally "hop_uncertainty".
            """
            # Concatenate inputs
            x = torch.cat([before, after, init, goal, instruction], dim=-1)

            # Encode
            h = self.encoder(x)

            # Predict hop
            hop_hat = torch.tanh(self.hop_head(h))  # [-1, 1]

            result = {"hop_hat": hop_hat.squeeze(-1)}

            if self.uncertainty_head is not None:
                # Output as log variance (can be converted to std)
                log_var = self.uncertainty_head(h)
                result["hop_uncertainty"] = log_var.squeeze(-1)

            return result


    class HopNetWrapper:
        """Wrapper for HopNet that handles numpy inputs and provides batch inference."""

        def __init__(
            self,
            config: ProcessRewardConfig,
            hop_net: Optional[HopNet] = None,
        ):
            """Initialize wrapper.

            Args:
                config: Process reward configuration.
                hop_net: Optional pretrained HopNet. If None, creates default.
            """
            self.config = config
            self.device = torch.device(config.device)

            if hop_net is not None:
                self.hop_net = hop_net.to(self.device)
            else:
                self.hop_net = HopNet(
                    feature_dim=config.feature_dim,
                    instruction_dim=config.instruction_embedding_dim,
                    hidden_dim=config.fusion_hidden_dim,
                    num_layers=config.fusion_num_layers,
                    output_uncertainty=True,
                ).to(self.device)

            self.hop_net.eval()

        def predict_hop(
            self,
            before_features: np.ndarray,  # (feature_dim,)
            after_features: np.ndarray,  # (feature_dim,)
            init_features: np.ndarray,  # (feature_dim,)
            goal_features: np.ndarray,  # (feature_dim,)
            instruction_embedding: np.ndarray,  # (instruction_dim,)
        ) -> Tuple[float, float]:
            """Predict hop for a single transition.

            Args:
                before_features: Features at before state.
                after_features: Features at after state.
                init_features: Features at init state.
                goal_features: Features at goal state.
                instruction_embedding: Instruction embedding.

            Returns:
                (hop_hat, uncertainty) tuple.
            """
            with torch.no_grad():
                before = torch.from_numpy(before_features).float().unsqueeze(0).to(self.device)
                after = torch.from_numpy(after_features).float().unsqueeze(0).to(self.device)
                init = torch.from_numpy(init_features).float().unsqueeze(0).to(self.device)
                goal = torch.from_numpy(goal_features).float().unsqueeze(0).to(self.device)
                instr = torch.from_numpy(instruction_embedding).float().unsqueeze(0).to(self.device)

                out = self.hop_net(before, after, init, goal, instr)

                hop_hat = out["hop_hat"].item()
                uncertainty = out.get("hop_uncertainty", torch.tensor([0.0])).item()

            return hop_hat, uncertainty

        def predict_episode_hops(
            self,
            episode_features: EpisodeFeatures,
            instruction_embedding: np.ndarray,
            goal_frame_idx: Optional[int] = None,
        ) -> Tuple[np.ndarray, np.ndarray]:
            """Predict hops for all transitions in an episode.

            Args:
                episode_features: Episode features.
                instruction_embedding: Instruction embedding.
                goal_frame_idx: Optional goal frame index.

            Returns:
                (hops, uncertainties) arrays of shape (T-1,).
            """
            T = len(episode_features.frame_features)
            if T < 2:
                return np.array([]), np.array([])

            init_features = episode_features.init_features.pooled
            goal_features = (
                episode_features.goal_features.pooled
                if episode_features.goal_features is not None
                else episode_features.frame_features[-1].pooled
            )

            hops = []
            uncertainties = []

            for t in range(T - 1):
                before = episode_features.frame_features[t].pooled
                after = episode_features.frame_features[t + 1].pooled

                hop, unc = self.predict_hop(
                    before, after, init_features, goal_features, instruction_embedding
                )
                hops.append(hop)
                uncertainties.append(unc)

            return np.array(hops, dtype=np.float32), np.array(uncertainties, dtype=np.float32)

        def load_checkpoint(self, path: str) -> None:
            """Load model checkpoint.

            Args:
                path: Path to checkpoint file.
            """
            checkpoint = torch.load(path, map_location=self.device)
            if "model_state_dict" in checkpoint:
                self.hop_net.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.hop_net.load_state_dict(checkpoint)
            self.hop_net.eval()

else:
    # Fallback when PyTorch not available
    class HopNet:
        """Placeholder when PyTorch not available."""
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch required for HopNet")

    class HopNetWrapper:
        """Placeholder when PyTorch not available."""
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch required for HopNetWrapper")


# -----------------------------------------------------------------------------
# Heuristic Hop Predictor (No PyTorch Required)
# -----------------------------------------------------------------------------

class HeuristicHopPredictor:
    """Heuristic hop predictor that doesn't require PyTorch.

    Uses simple feature-based rules to estimate progress.
    """

    def __init__(self, config: ProcessRewardConfig):
        """Initialize.

        Args:
            config: Process reward configuration.
        """
        self.config = config

    def predict_hop(
        self,
        before_features: np.ndarray,
        after_features: np.ndarray,
        init_features: np.ndarray,
        goal_features: np.ndarray,
    ) -> Tuple[float, float]:
        """Predict hop using distance-based heuristic.

        Args:
            before_features: Features at before state.
            after_features: Features at after state.
            init_features: Features at init state.
            goal_features: Features at goal state.

        Returns:
            (hop, uncertainty) tuple.
        """
        # Distance to goal
        dist_before = np.linalg.norm(before_features - goal_features)
        dist_after = np.linalg.norm(after_features - goal_features)

        # Normalize by initial distance
        dist_init = np.linalg.norm(init_features - goal_features)
        if dist_init < 1e-6:
            dist_init = 1.0

        # Hop = normalized reduction in distance
        hop = (dist_before - dist_after) / dist_init
        hop = np.clip(hop, -1.0, 1.0)

        # Uncertainty based on feature stability
        feature_change = np.linalg.norm(after_features - before_features)
        uncertainty = 1.0 - np.exp(-feature_change)  # Higher change = higher uncertainty

        return float(hop), float(uncertainty)

    def predict_episode_hops(
        self,
        episode_features: EpisodeFeatures,
        goal_frame_idx: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict hops for all transitions in an episode.

        Args:
            episode_features: Episode features.
            goal_frame_idx: Optional goal frame index.

        Returns:
            (hops, uncertainties) arrays of shape (T-1,).
        """
        T = len(episode_features.frame_features)
        if T < 2:
            return np.array([]), np.array([])

        init_features = episode_features.init_features.pooled
        goal_features = (
            episode_features.goal_features.pooled
            if episode_features.goal_features is not None
            else episode_features.frame_features[-1].pooled
        )

        hops = []
        uncertainties = []

        for t in range(T - 1):
            before = episode_features.frame_features[t].pooled
            after = episode_features.frame_features[t + 1].pooled

            hop, unc = self.predict_hop(before, after, init_features, goal_features)
            hops.append(hop)
            uncertainties.append(unc)

        return np.array(hops, dtype=np.float32), np.array(uncertainties, dtype=np.float32)


def create_hop_predictor(config: ProcessRewardConfig) -> Union["HopNetWrapper", HeuristicHopPredictor]:
    """Factory function to create appropriate hop predictor.

    Args:
        config: Process reward configuration.

    Returns:
        HopNetWrapper if PyTorch available and model path specified,
        else HeuristicHopPredictor.
    """
    if TORCH_AVAILABLE and config.hop_model_path:
        wrapper = HopNetWrapper(config)
        wrapper.load_checkpoint(config.hop_model_path)
        return wrapper
    elif TORCH_AVAILABLE:
        # Use untrained HopNet (for testing/initial bootstrap)
        return HopNetWrapper(config)
    else:
        return HeuristicHopPredictor(config)
