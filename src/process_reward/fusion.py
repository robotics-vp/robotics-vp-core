"""
Fusion Network for Process Reward.

Learns to combine three progress perspectives (Phi_I, Phi_F, Phi_B) into a single
fused potential Phi_star with confidence estimate.

Key requirements:
- NO simple averaging. Uses learned FusionNet.
- Orchestrator-controllable via FusionOverride hyperparameters.
- Outputs weights + confidence for PBRS.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

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
    FusionOverride,
    ProgressPerspectives,
    FusionDiagnostics,
    MHNSummary,
)
from src.process_reward.progress_perspectives import (
    compute_perspective_disagreement,
    compute_perspective_entropy,
)


# -----------------------------------------------------------------------------
# FusionNet (PyTorch)
# -----------------------------------------------------------------------------

if TORCH_AVAILABLE:

    class FusionNet(nn.Module):
        """Neural network for fusing three progress perspectives.

        Inputs:
            - [Phi_I, Phi_F, Phi_B]: Three perspective values at timestep t.
            - [conf_I, conf_F, conf_B]: Per-perspective confidences.
            - disagreement: Max difference between perspectives.
            - Context features: occlusion stats, IR stats, MHN features, t/T.

        Outputs:
            - weights: (3,) softmax weights for combining perspectives.
            - confidence: Scalar confidence in [0, 1].
        """

        def __init__(
            self,
            hidden_dim: int = 64,
            num_layers: int = 2,
            context_dim: int = 8,
            use_mhn_features: bool = True,
            dropout: float = 0.1,
        ):
            """Initialize FusionNet.

            Args:
                hidden_dim: Hidden layer dimension.
                num_layers: Number of hidden layers.
                context_dim: Dimension of context features.
                use_mhn_features: Whether to include MHN features.
                dropout: Dropout probability.
            """
            super().__init__()

            self.use_mhn_features = use_mhn_features

            # Input features:
            # - 3 phi values
            # - 3 confidence values
            # - 1 disagreement
            # - 1 timestamp ratio (t/T)
            # - context_dim context features (occlusion, IR, etc.)
            # - 5 MHN features (if enabled)
            base_input_dim = 3 + 3 + 1 + 1 + context_dim
            if use_mhn_features:
                base_input_dim += 5

            self.input_dim = base_input_dim

            # Encoder MLP
            layers = []
            in_dim = base_input_dim
            for i in range(num_layers):
                layers.append(nn.Linear(in_dim, hidden_dim))
                layers.append(nn.LayerNorm(hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                in_dim = hidden_dim

            self.encoder = nn.Sequential(*layers)

            # Weight head: outputs logits for 3 perspectives
            self.weight_head = nn.Linear(hidden_dim, 3)

            # Confidence head: outputs logit for confidence
            self.confidence_head = nn.Linear(hidden_dim, 1)

        def forward(
            self,
            phi_values: torch.Tensor,  # (B, 3) or (B, T, 3)
            conf_values: torch.Tensor,  # (B, 3) or (B, T, 3)
            disagreement: torch.Tensor,  # (B,) or (B, T)
            t_ratio: torch.Tensor,  # (B,) or (B, T) - timestep / total_steps
            context: torch.Tensor,  # (B, context_dim) or (B, T, context_dim)
            mhn_features: Optional[torch.Tensor] = None,  # (B, 5) or (B, T, 5)
            temperature: float = 1.0,
            candidate_mask: Optional[torch.Tensor] = None,  # (3,) or (B, 3)
        ) -> Dict[str, torch.Tensor]:
            """Forward pass.

            Args:
                phi_values: Perspective values [Phi_I, Phi_F, Phi_B].
                conf_values: Perspective confidences.
                disagreement: Disagreement between perspectives.
                t_ratio: Timestep ratio (t/T).
                context: Context features.
                mhn_features: Optional MHN features.
                temperature: Softmax temperature (from FusionOverride).
                candidate_mask: Boolean mask for enabled perspectives.

            Returns:
                Dict with "weights" (B, 3) and "confidence" (B,).
            """
            # Handle both (B,) and (B, T) inputs
            if phi_values.dim() == 2:
                # (B, 3) input
                batch_mode = True
                B = phi_values.shape[0]
            else:
                batch_mode = False
                B = 1
                phi_values = phi_values.unsqueeze(0)
                conf_values = conf_values.unsqueeze(0)
                disagreement = disagreement.unsqueeze(0)
                t_ratio = t_ratio.unsqueeze(0)
                context = context.unsqueeze(0)
                if mhn_features is not None:
                    mhn_features = mhn_features.unsqueeze(0)

            # Ensure proper shapes
            if disagreement.dim() == 1:
                disagreement = disagreement.unsqueeze(-1)
            if t_ratio.dim() == 1:
                t_ratio = t_ratio.unsqueeze(-1)

            # Concatenate inputs
            inputs = [phi_values, conf_values, disagreement, t_ratio, context]
            if self.use_mhn_features and mhn_features is not None:
                inputs.append(mhn_features)
            elif self.use_mhn_features:
                # Pad with zeros if MHN features expected but not provided
                inputs.append(torch.zeros(B, 5, device=phi_values.device))

            x = torch.cat(inputs, dim=-1)

            # Encode
            h = self.encoder(x)

            # Compute weight logits
            weight_logits = self.weight_head(h)  # (B, 3)

            # Apply candidate mask (set masked candidates to -inf)
            if candidate_mask is not None:
                if candidate_mask.dim() == 1:
                    candidate_mask = candidate_mask.unsqueeze(0).expand(B, -1)
                mask_value = torch.where(
                    candidate_mask,
                    torch.zeros_like(weight_logits),
                    torch.full_like(weight_logits, -1e9),
                )
                weight_logits = weight_logits + mask_value

            # Apply temperature and softmax
            weights = F.softmax(weight_logits / temperature, dim=-1)

            # Compute confidence
            conf_logit = self.confidence_head(h)
            confidence = torch.sigmoid(conf_logit).squeeze(-1)

            if not batch_mode:
                weights = weights.squeeze(0)
                confidence = confidence.squeeze(0)

            return {"weights": weights, "confidence": confidence}


    class FusionNetWrapper:
        """Wrapper for FusionNet that handles numpy inputs and applies FusionOverride."""

        def __init__(
            self,
            config: ProcessRewardConfig,
            fusion_net: Optional[FusionNet] = None,
        ):
            """Initialize wrapper.

            Args:
                config: Process reward configuration.
                fusion_net: Optional pretrained FusionNet.
            """
            self.config = config
            self.device = torch.device(config.device)

            if fusion_net is not None:
                self.fusion_net = fusion_net.to(self.device)
            else:
                self.fusion_net = FusionNet(
                    hidden_dim=config.fusion_hidden_dim,
                    num_layers=config.fusion_num_layers,
                    use_mhn_features=config.use_mhn_features,
                ).to(self.device)

            self.fusion_net.eval()

            # Previous weights for temporal smoothing
            self._prev_weights: Optional[np.ndarray] = None

        def fuse_perspectives(
            self,
            perspectives: ProgressPerspectives,
            context_features: np.ndarray,  # (T, context_dim)
            mhn_summary: Optional[MHNSummary] = None,
            override: Optional[FusionOverride] = None,
        ) -> Tuple[np.ndarray, np.ndarray, FusionDiagnostics]:
            """Fuse perspectives for an episode.

            Args:
                perspectives: Three progress perspectives.
                context_features: Per-timestep context features.
                mhn_summary: Optional MHN summary.
                override: Optional fusion overrides from orchestrator.

            Returns:
                (phi_star, confidence, diagnostics) tuple.
            """
            override = override or self.config.default_fusion_override
            T = len(perspectives.phi_I)

            # Prepare inputs
            phi_values = np.stack([
                perspectives.phi_I,
                perspectives.phi_F,
                perspectives.phi_B,
            ], axis=-1)  # (T, 3)

            conf_values = np.stack([
                perspectives.conf_I,
                perspectives.conf_F,
                perspectives.conf_B,
            ], axis=-1)  # (T, 3)

            disagreement = compute_perspective_disagreement(perspectives)
            t_ratio = np.arange(T) / max(T - 1, 1)

            # MHN features (broadcast to all timesteps)
            if mhn_summary is not None:
                mhn_features = np.tile(np.array([
                    mhn_summary.mean_tree_depth,
                    mhn_summary.mean_branch_factor,
                    mhn_summary.residual_mean,
                    mhn_summary.structural_difficulty,
                    mhn_summary.plausibility_score,
                ], dtype=np.float32), (T, 1))
            else:
                mhn_features = None

            # Candidate mask
            candidate_mask = torch.tensor(override.candidate_mask, dtype=torch.bool)

            # Process in batch
            with torch.no_grad():
                phi_t = torch.from_numpy(phi_values).float().to(self.device)
                conf_t = torch.from_numpy(conf_values).float().to(self.device)
                disag_t = torch.from_numpy(disagreement).float().to(self.device)
                ratio_t = torch.from_numpy(t_ratio).float().to(self.device)
                ctx_t = torch.from_numpy(context_features).float().to(self.device)

                mhn_t = None
                if mhn_features is not None:
                    mhn_t = torch.from_numpy(mhn_features).float().to(self.device)

                out = self.fusion_net(
                    phi_t, conf_t, disag_t, ratio_t, ctx_t,
                    mhn_features=mhn_t,
                    temperature=override.temperature,
                    candidate_mask=candidate_mask.to(self.device),
                )

                weights = out["weights"].cpu().numpy()  # (T, 3)
                confidence = out["confidence"].cpu().numpy()  # (T,)

            # Apply weight floor (only for enabled candidates)
            mask = np.array(override.candidate_mask, dtype=np.float32)
            weights = np.maximum(weights, override.min_weight_floor * mask)
            # Re-zero masked candidates (floor cannot resurrect them)
            weights = weights * mask
            weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-8)

            # Apply temporal smoothing
            if override.weight_smoothing > 0 and self._prev_weights is not None:
                alpha = override.weight_smoothing
                weights = alpha * self._prev_weights + (1 - alpha) * weights
                # Re-apply mask after smoothing (smoothing cannot resurrect masked candidates)
                weights = weights * mask
                weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-8)

            self._prev_weights = weights.copy()

            # Compute fused potential
            phi_star = np.sum(weights * phi_values, axis=-1)  # (T,)

            # Apply entropy penalty to confidence
            entropy = compute_perspective_entropy(perspectives)
            confidence = confidence * (1.0 - override.entropy_penalty * entropy)
            # Apply confidence cap from orchestrator policy
            confidence = np.clip(confidence, 0.0, override.confidence_cap)

            # Compute gating factor
            gating_factor = np.where(
                confidence >= override.risk_tolerance,
                np.ones_like(confidence),
                confidence / override.risk_tolerance,
            )

            diagnostics = FusionDiagnostics(
                weights=weights.astype(np.float32),
                entropy=entropy.astype(np.float32),
                disagreement=disagreement.astype(np.float32),
                gating_factor=gating_factor.astype(np.float32),
            )

            return phi_star.astype(np.float32), confidence.astype(np.float32), diagnostics

        def reset_temporal_state(self) -> None:
            """Reset temporal smoothing state (call at episode boundaries)."""
            self._prev_weights = None

        def load_checkpoint(self, path: str) -> None:
            """Load model checkpoint."""
            checkpoint = torch.load(path, map_location=self.device)
            if "model_state_dict" in checkpoint:
                self.fusion_net.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.fusion_net.load_state_dict(checkpoint)
            self.fusion_net.eval()

else:
    class FusionNet:
        """Placeholder when PyTorch not available."""
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch required for FusionNet")

    class FusionNetWrapper:
        """Placeholder when PyTorch not available."""
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch required for FusionNetWrapper")


# -----------------------------------------------------------------------------
# Heuristic Fusion (No PyTorch Required)
# -----------------------------------------------------------------------------

class HeuristicFusion:
    """Heuristic fusion that doesn't require PyTorch.

    Uses confidence-weighted combination with simple rules.
    Still respects FusionOverride parameters.
    """

    def __init__(self, config: ProcessRewardConfig):
        """Initialize.

        Args:
            config: Process reward configuration.
        """
        self.config = config
        self._prev_weights: Optional[np.ndarray] = None

    def fuse_perspectives(
        self,
        perspectives: ProgressPerspectives,
        context_features: np.ndarray,
        mhn_summary: Optional[MHNSummary] = None,
        override: Optional[FusionOverride] = None,
    ) -> Tuple[np.ndarray, np.ndarray, FusionDiagnostics]:
        """Fuse perspectives using heuristic rules.

        Args:
            perspectives: Three progress perspectives.
            context_features: Per-timestep context features (used for confidence).
            mhn_summary: Optional MHN summary.
            override: Optional fusion overrides from orchestrator.

        Returns:
            (phi_star, confidence, diagnostics) tuple.
        """
        override = override or self.config.default_fusion_override
        T = len(perspectives.phi_I)

        # Compute base weights from confidences
        conf_stack = np.stack([
            perspectives.conf_I,
            perspectives.conf_F,
            perspectives.conf_B,
        ], axis=-1)  # (T, 3)

        # Apply candidate mask - masked candidates get -inf before softmax
        mask = np.array(override.candidate_mask, dtype=np.float32)

        # Use log-space softmax with mask
        log_conf = np.log(conf_stack + 1e-8) / override.temperature
        # Set masked candidates to -inf
        log_conf = np.where(mask > 0, log_conf, -1e9)

        # Softmax
        exp_conf = np.exp(log_conf - log_conf.max(axis=-1, keepdims=True))
        weights = exp_conf / (exp_conf.sum(axis=-1, keepdims=True) + 1e-8)

        # Apply weight floor (only for enabled candidates)
        weights = np.maximum(weights, override.min_weight_floor * mask)
        # Re-zero masked candidates (floor cannot resurrect them)
        weights = weights * mask
        weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-8)

        # Apply temporal smoothing
        if override.weight_smoothing > 0 and self._prev_weights is not None:
            alpha = override.weight_smoothing
            weights = alpha * self._prev_weights + (1 - alpha) * weights
            # Re-apply mask after smoothing (smoothing cannot resurrect masked candidates)
            weights = weights * mask
            weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-8)

        self._prev_weights = weights.copy()

        # Compute fused potential
        phi_values = np.stack([
            perspectives.phi_I,
            perspectives.phi_F,
            perspectives.phi_B,
        ], axis=-1)  # (T, 3)

        phi_star = np.sum(weights * phi_values, axis=-1)

        # Compute confidence from weighted confidences and disagreement
        disagreement = compute_perspective_disagreement(perspectives)
        entropy = compute_perspective_entropy(perspectives)

        # Base confidence = weighted sum of per-perspective confidences
        base_conf = np.sum(weights * conf_stack, axis=-1)

        # Reduce confidence based on disagreement
        disagreement_penalty = 1.0 - disagreement  # Higher disagreement = lower confidence

        # Reduce confidence based on entropy
        entropy_penalty = 1.0 - override.entropy_penalty * entropy

        # Incorporate MHN plausibility if available
        mhn_factor = 1.0
        if mhn_summary is not None:
            mhn_factor = mhn_summary.plausibility_score

        confidence = base_conf * disagreement_penalty * entropy_penalty * mhn_factor
        # Apply confidence cap from orchestrator policy
        confidence = np.clip(confidence, 0.0, override.confidence_cap)

        # Compute gating factor
        gating_factor = np.where(
            confidence >= override.risk_tolerance,
            np.ones_like(confidence),
            confidence / override.risk_tolerance,
        )

        diagnostics = FusionDiagnostics(
            weights=weights.astype(np.float32),
            entropy=entropy.astype(np.float32),
            disagreement=disagreement.astype(np.float32),
            gating_factor=gating_factor.astype(np.float32),
        )

        return phi_star.astype(np.float32), confidence.astype(np.float32), diagnostics

    def reset_temporal_state(self) -> None:
        """Reset temporal smoothing state."""
        self._prev_weights = None


def create_fusion(
    config: ProcessRewardConfig,
) -> "FusionNetWrapper | HeuristicFusion":
    """Factory function to create appropriate fusion module.

    Args:
        config: Process reward configuration.

    Returns:
        FusionNetWrapper if PyTorch available, else HeuristicFusion.
    """
    if TORCH_AVAILABLE:
        return FusionNetWrapper(config)
    else:
        return HeuristicFusion(config)


def build_context_features(
    episode_features: "EpisodeFeatures",
    mhn_summary: Optional[MHNSummary] = None,
) -> np.ndarray:
    """Build context features for fusion from episode features.

    Args:
        episode_features: Episode features.
        mhn_summary: Optional MHN summary.

    Returns:
        (T, context_dim) array of context features.
    """
    from src.process_reward.schemas import EpisodeFeatures

    T = len(episode_features.frame_features)
    context_dim = 8  # Fixed dimension

    context = np.zeros((T, context_dim), dtype=np.float32)

    for t, ff in enumerate(episode_features.frame_features):
        vis_stats = ff.visibility_stats
        ir_stats = ff.ir_stats

        context[t, 0] = vis_stats.get("pct_visible", 1.0)
        context[t, 1] = vis_stats.get("pct_occluded", 0.0)
        context[t, 2] = vis_stats.get("num_bodies", 0) / max(vis_stats.get("num_tracks", 1), 1)
        context[t, 3] = vis_stats.get("num_objects", 0) / max(vis_stats.get("num_tracks", 1), 1)
        context[t, 4] = ir_stats.get("mean_ir_loss", 0.0)
        context[t, 5] = ir_stats.get("pct_converged", 1.0)
        context[t, 6] = t / max(T - 1, 1)  # Normalized timestep
        context[t, 7] = 1.0  # Reserved for future use

    return context
