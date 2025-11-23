"""
Shared trunk that fuses vision/state/condition features for Hydra policies.

Minimal, deterministic, and flag-free: callers control which encoders to pass.
"""
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.observation.condition_vector import ConditionVector


def _pad_or_trim(tensor: torch.Tensor, target_dim: int) -> torch.Tensor:
    """Adjust tensor last-dim to target_dim deterministically."""
    current = tensor.shape[-1]
    if current == target_dim:
        return tensor
    if current > target_dim:
        return tensor[..., :target_dim]
    pad_width = target_dim - current
    pad = torch.zeros(*tensor.shape[:-1], pad_width, device=tensor.device, dtype=tensor.dtype)
    return torch.cat([tensor, pad], dim=-1)


def _tensor_from_iterable(values: Any, device: torch.device, fallback_dim: int) -> torch.Tensor:
    """Convert lists/arrays to 1D tensor; fallback to zeros."""
    try:
        tensor = torch.as_tensor(values, device=device, dtype=torch.float32).flatten()
        if tensor.numel() == 0:
            raise ValueError("empty tensor")
        return tensor
    except Exception:
        return torch.zeros(fallback_dim, device=device, dtype=torch.float32)


class TrunkNet(nn.Module):
    """
    Simple trunk that fuses vision latent, state summary, and ConditionVector.

    This is intentionally lightweight and does not alter reward/econ math.
    """

    def __init__(
        self,
        vision_dim: int = 256,
        state_dim: int = 32,
        condition_dim: int = 32,
        hidden_dim: int = 256,
        use_condition_film: bool = True,
        use_condition_vector: bool = True,
        use_condition_vector_for_policy: bool = False,
        condition_fusion_mode: str = "film",
        condition_film_hidden_dim: int = 64,
        condition_context_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.vision_dim = vision_dim
        self.state_dim = state_dim
        self.condition_dim = condition_dim
        self.use_condition_film = use_condition_film
        self.use_condition_vector = use_condition_vector
        self.use_condition_vector_for_policy = use_condition_vector_for_policy
        self.condition_fusion_mode = (condition_fusion_mode or "film").lower()
        self.condition_context_dim = condition_context_dim or hidden_dim

        self.vision_proj = nn.Linear(max(1, vision_dim), hidden_dim)
        self.state_proj = nn.Linear(max(1, state_dim), hidden_dim)
        self.condition_proj = nn.Linear(max(1, condition_dim), hidden_dim)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.condition_film = nn.Sequential(
            nn.Linear(max(1, condition_dim), condition_film_hidden_dim),
            nn.ReLU(),
            nn.Linear(condition_film_hidden_dim, hidden_dim * 2),
        )
        self.condition_context = nn.Sequential(
            nn.Linear(max(1, condition_dim), condition_film_hidden_dim),
            nn.ReLU(),
            nn.Linear(condition_film_hidden_dim, self.condition_context_dim),
        )
        # Zero-init conditioning heads so enabling the flag keeps outputs identical until trained
        for layer in (self.condition_film[-1], self.condition_context[-1]):
            if hasattr(layer, "weight"):
                nn.init.zeros_(layer.weight)
            if hasattr(layer, "bias") and layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(
        self,
        obs: Any,
        condition: Optional[ConditionVector] = None,
        use_condition: Optional[bool] = None,
        use_condition_vector_for_policy: Optional[bool] = None,
    ) -> torch.Tensor:
        device = next(self.parameters()).device
        use_condition = self.use_condition_vector if use_condition is None else use_condition
        use_policy_condition = (
            self.use_condition_vector_for_policy if use_condition_vector_for_policy is None else use_condition_vector_for_policy
        )
        if not use_condition:
            condition = None
        vision_vec = self._extract_vision(obs, device)
        state_vec = self._extract_state(obs, device)
        condition_vec = self._extract_condition(condition, device)

        vision_embed = self.vision_proj(_pad_or_trim(vision_vec, self.vision_dim))
        state_embed = self.state_proj(_pad_or_trim(state_vec, self.state_dim))
        condition_embed = self.condition_proj(_pad_or_trim(condition_vec, self.condition_dim))

        if self.use_condition_film:
            gating = torch.sigmoid(condition_embed)
            vision_embed = vision_embed * gating
            state_embed = state_embed * gating

        fused = torch.cat([vision_embed, state_embed, condition_embed], dim=-1)
        trunk_features = self.fusion(fused)

        if use_policy_condition and condition is not None:
            conditioned = self._condition_policy_features(trunk_features, condition_embed)
            return trunk_features, conditioned
        return trunk_features

    def _extract_vision(self, obs: Any, device: torch.device) -> torch.Tensor:
        latent = getattr(obs, "latent", None)
        if hasattr(latent, "latent"):
            latent = getattr(latent, "latent")
        if latent is None and isinstance(obs, dict):
            latent = obs.get("vision_latent") or obs.get("latent")
        return _tensor_from_iterable(latent, device, self.vision_dim)

    def _extract_state(self, obs: Any, device: torch.device) -> torch.Tensor:
        state_summary = getattr(obs, "state_summary", None) or {}
        if isinstance(obs, dict):
            state_summary = obs.get("state_summary", state_summary)
        if isinstance(state_summary, dict):
            flat: list = []
            for key, val in sorted(state_summary.items(), key=lambda kv: str(kv[0])):
                if isinstance(val, dict):
                    for k2, v2 in sorted(val.items(), key=lambda kv: str(kv[0])):
                        flat.append(float(v2) if isinstance(v2, (int, float)) else 0.0)
                elif isinstance(val, (list, tuple)):
                    flat.extend([float(v) if isinstance(v, (int, float)) else 0.0 for v in val])
                elif isinstance(val, (int, float)):
                    flat.append(float(val))
            return _tensor_from_iterable(flat, device, self.state_dim)
        return _tensor_from_iterable(state_summary, device, self.state_dim)

    def _extract_condition(self, condition: Optional[ConditionVector], device: torch.device) -> torch.Tensor:
        if condition is None:
            return torch.zeros(self.condition_dim, device=device, dtype=torch.float32)
        try:
            vec = condition.to_vector()
            return torch.as_tensor(vec, device=device, dtype=torch.float32).flatten()
        except Exception:
            return torch.zeros(self.condition_dim, device=device, dtype=torch.float32)

    def _condition_policy_features(self, trunk_features: torch.Tensor, condition_embed: torch.Tensor) -> torch.Tensor:
        if not self.use_condition_vector_for_policy:
            return trunk_features
        if self.condition_fusion_mode == "concat":
            context = self.condition_context(condition_embed)
            return torch.cat([trunk_features, context], dim=-1)
        # Default: FiLM
        gamma_beta = self.condition_film(condition_embed)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)
        return trunk_features * (1 + gamma) + beta

    def condition_policy_features(self, trunk_features: torch.Tensor, condition: Optional[ConditionVector]) -> Optional[torch.Tensor]:
        """
        Apply the conditioning block to externally computed trunk features.
        Returns None if conditioning is disabled or no condition vector is provided.
        """
        if not self.use_condition_vector_for_policy or condition is None:
            return None
        device = trunk_features.device
        condition_vec = self._extract_condition(condition, device)
        condition_embed = self.condition_proj(_pad_or_trim(condition_vec, self.condition_dim))
        return self._condition_policy_features(trunk_features, condition_embed)
