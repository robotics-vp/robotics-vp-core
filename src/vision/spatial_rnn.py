"""
Spatial RNN for temporal coherence in visual feature sequences.

Implements:
- ConvGRU: Gated recurrent unit with spatial structure
- S4D: State-space model for long-range dependencies
- Deterministic forward pass with seed control
- Forward dynamics loss for next-step prediction

Architecture:
- Input: Sequence of multi-scale features (from RegNet/BiFPN)
- Hidden state: Maintains temporal coherence
- Output: Temporally-smoothed features per level + summary vector
"""
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

# Try importing PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    class ConvGRUCell(nn.Module):
        """
        Convolutional GRU cell for spatial features.

        Maintains spatial structure while applying gated recurrence.
        For 1D feature vectors, we treat them as [B, C, 1, 1] spatial tensors.
        """

        def __init__(self, input_dim: int, hidden_dim: int):
            super().__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim

            # Gates: reset and update
            self.conv_gates = nn.Conv2d(
                input_dim + hidden_dim,
                2 * hidden_dim,
                kernel_size=1,
                bias=True
            )

            # Candidate hidden state
            self.conv_candidate = nn.Conv2d(
                input_dim + hidden_dim,
                hidden_dim,
                kernel_size=1,
                bias=True
            )

        def forward(
            self,
            x: torch.Tensor,
            h_prev: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
            """
            Args:
                x: Input tensor [B, C, H, W] or [B, C]
                h_prev: Previous hidden state [B, hidden_dim, H, W]

            Returns:
                h: New hidden state [B, hidden_dim, H, W]
            """
            # Reshape 1D vectors to 2D spatial
            if x.dim() == 2:
                x = x.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]

            batch_size, _, height, width = x.shape

            # Initialize hidden state if None
            if h_prev is None:
                h_prev = torch.zeros(
                    batch_size, self.hidden_dim, height, width,
                    device=x.device, dtype=x.dtype
                )

            # Concatenate input and hidden state
            combined = torch.cat([x, h_prev], dim=1)

            # Compute gates
            gates = self.conv_gates(combined)
            reset_gate, update_gate = torch.split(gates, self.hidden_dim, dim=1)
            reset_gate = torch.sigmoid(reset_gate)
            update_gate = torch.sigmoid(update_gate)

            # Compute candidate hidden state
            combined_reset = torch.cat([x, reset_gate * h_prev], dim=1)
            candidate = torch.tanh(self.conv_candidate(combined_reset))

            # Update hidden state
            h = (1 - update_gate) * h_prev + update_gate * candidate

            return h

        def init_hidden(self, batch_size: int, height: int, width: int, device: torch.device) -> torch.Tensor:
            """Initialize hidden state."""
            return torch.zeros(batch_size, self.hidden_dim, height, width, device=device)


    class SpatialRNN(nn.Module):
        """
        Spatial RNN with ConvGRU or S4D backend.

        Processes sequences of multi-scale features with temporal coherence.
        """

        def __init__(
            self,
            hidden_dim: int,
            feature_dim: int,
            levels: Sequence[str] = ("P3", "P4", "P5"),
            mode: str = "convgru",
            seed: int = 0,
            use_checkpointing: bool = False,
        ):
            """
            Args:
                hidden_dim: Hidden state dimension
                feature_dim: Input feature dimension
                levels: Pyramid levels to process
                mode: "convgru" or "s4d" (currently only convgru implemented)
                seed: Random seed for deterministic initialization
            """
            super().__init__()

            # Set seed for deterministic initialization
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            self.hidden_dim = hidden_dim
            self.feature_dim = feature_dim
            self.levels = list(levels)
            self.mode = mode
            self.use_checkpointing = use_checkpointing

            if mode == "convgru":
                # Create ConvGRU cell per level
                self.cells = nn.ModuleDict({
                    level: ConvGRUCell(feature_dim, hidden_dim)
                    for level in self.levels
                })
            elif mode == "s4d":
                # S4D mode (placeholder - not implemented yet)
                raise NotImplementedError("S4D mode not yet implemented")
            else:
                raise ValueError(f"Unknown mode: {mode}")

            # Summary projection (aggregates all levels)
            total_dim = hidden_dim * len(self.levels)
            self.summary_proj = nn.Linear(total_dim, hidden_dim)

        def forward(
            self,
            sequence: Union[List[Dict[str, np.ndarray]], List[np.ndarray]],
            initial_state: Optional[Dict[str, torch.Tensor]] = None,
        ) -> Dict[str, torch.Tensor]:
            """
            Forward pass over sequence.

            Args:
                sequence: List of feature pyramids (dicts) or vectors (arrays)
                    - If dict: {level: np.ndarray} per timestep
                    - If np.ndarray: flat vector per timestep
                initial_state: Optional initial hidden states per level

            Returns:
                Dict with:
                    - Per-level final hidden states
                    - "summary": Aggregated summary vector
            """
            if not sequence:
                return {"summary": torch.zeros(1, self.hidden_dim)}

            device = next(self.parameters()).device

            # Initialize hidden states
            if initial_state is None:
                hidden_states = {level: None for level in self.levels}
            else:
                hidden_states = initial_state

            # Process sequence
            for t, item in enumerate(sequence):
                if isinstance(item, dict):
                    # Multi-level pyramid
                    for level in self.levels:
                        if level not in item:
                            continue
                        feat = item[level]
                        feat_tensor = torch.from_numpy(np.asarray(feat, dtype=np.float32)).unsqueeze(0).to(device)
                        if self.use_checkpointing:
                            from src.utils.training_env import checkpoint_if_enabled
                            # Need to ensure hidden state requires grad for checkpointing to work if it's the only input
                            # But here feat_tensor requires grad (usually) or is leaf.
                            # Checkpoint requires at least one input with requires_grad=True.
                            # feat_tensor is created from numpy, so it doesn't require grad by default unless we set it.
                            # But in training loop, we might be backpropagating through it?
                            # Actually, in SpatialRNN training, inputs are features from dataset, so they are leaves.
                            # But if we are training end-to-end, they might come from backbone.
                            # In Phase I, we are training SpatialRNN on pre-extracted features?
                            # scripts/train_spatial_rnn.py uses SpatialRNNDataset which yields numpy arrays.
                            # So inputs are leaves.
                            # If inputs are leaves and don't require grad, checkpointing will fail or do nothing useful.
                            # But wait, we are training the RNN parameters.
                            # Checkpointing saves memory by not storing intermediate activations.
                            # If inputs don't require grad, we can still checkpoint the module to save its internal activations.
                            # But torch.utils.checkpoint requires at least one input to have requires_grad=True.
                            # We can force feat_tensor.requires_grad_(True).
                            feat_tensor.requires_grad_(True)
                            hidden_states[level] = checkpoint_if_enabled(self.cells[level], feat_tensor, hidden_states.get(level), enabled=True)
                        else:
                            hidden_states[level] = self.cells[level](feat_tensor, hidden_states.get(level))
                else:
                    # Flat vector - split across levels
                    feat = np.asarray(item, dtype=np.float32).flatten()
                    feat_per_level = len(feat) // len(self.levels)
                    for i, level in enumerate(self.levels):
                        start = i * feat_per_level
                        end = start + feat_per_level if i < len(self.levels) - 1 else len(feat)
                        level_feat = feat[start:end]
                        if len(level_feat) < self.feature_dim:
                            # Pad if needed
                            level_feat = np.pad(level_feat, (0, self.feature_dim - len(level_feat)))
                        level_feat = level_feat[:self.feature_dim]
                        feat_tensor = torch.from_numpy(level_feat).unsqueeze(0).to(device)
                        hidden_states[level] = self.cells[level](feat_tensor, hidden_states.get(level))

            # Clamp to prevent NaN/Inf
            for level in hidden_states:
                if hidden_states[level] is not None:
                    hidden_states[level] = torch.clamp(hidden_states[level], min=-1e6, max=1e6)

            # Aggregate summary
            summary_parts = []
            for level in self.levels:
                if hidden_states.get(level) is not None:
                    # Global average pooling
                    pooled = hidden_states[level].mean(dim=[2, 3])  # [B, hidden_dim]
                    summary_parts.append(pooled)
                else:
                    summary_parts.append(torch.zeros(1, self.hidden_dim, device=device))

            summary_concat = torch.cat(summary_parts, dim=1)  # [B, hidden_dim * num_levels]
            summary = self.summary_proj(summary_concat)  # [B, hidden_dim]

            # Return final states + summary
            result = {level: hidden_states[level] for level in self.levels if hidden_states.get(level) is not None}
            result["summary"] = summary

            return result

        def compute_forward_loss(
            self,
            sequence: Union[List[Dict[str, np.ndarray]], List[np.ndarray]],
            targets: Union[List[Dict[str, np.ndarray]], List[np.ndarray]],
        ) -> Dict[str, float]:
            """
            Compute forward dynamics loss: || z_pred[t+1] - z_true[t+1] ||²

            Args:
                sequence: Input sequence (t=0 to T-1)
                targets: Target sequence (t=1 to T)

            Returns:
                Dict with loss components
            """
            device = next(self.parameters()).device

            if len(sequence) != len(targets):
                raise ValueError(f"Sequence and targets must have same length, got {len(sequence)} vs {len(targets)}")

            total_loss = 0.0
            per_level_loss = {level: 0.0 for level in self.levels}
            num_steps = len(sequence)

            hidden_states = {level: None for level in self.levels}

            for t in range(num_steps - 1):
                # Forward step
                current = sequence[t]
                target_next = targets[t + 1]

                # Process current timestep
                if isinstance(current, dict):
                    for level in self.levels:
                        if level not in current:
                            continue
                        feat = current[level]
                        feat_tensor = torch.from_numpy(np.asarray(feat, dtype=np.float32)).unsqueeze(0).to(device)
                        if self.use_checkpointing:
                            from src.utils.training_env import checkpoint_if_enabled
                            feat_tensor.requires_grad_(True)
                            hidden_states[level] = checkpoint_if_enabled(self.cells[level], feat_tensor, hidden_states.get(level), enabled=True)
                        else:
                            hidden_states[level] = self.cells[level](feat_tensor, hidden_states.get(level))
                else:
                    # Flat vector mode (simplified)
                    feat = np.asarray(current, dtype=np.float32).flatten()
                    feat_per_level = len(feat) // len(self.levels)
                    for i, level in enumerate(self.levels):
                        start = i * feat_per_level
                        end = start + feat_per_level if i < len(self.levels) - 1 else len(feat)
                        level_feat = feat[start:end]
                        if len(level_feat) < self.feature_dim:
                            level_feat = np.pad(level_feat, (0, self.feature_dim - len(level_feat)))
                        level_feat = level_feat[:self.feature_dim]
                        feat_tensor = torch.from_numpy(level_feat).unsqueeze(0).to(device)
                        hidden_states[level] = self.cells[level](feat_tensor, hidden_states.get(level))

                # Compute prediction error vs target
                if isinstance(target_next, dict):
                    for level in self.levels:
                        if level not in target_next or hidden_states.get(level) is None:
                            continue
                        pred = hidden_states[level].mean(dim=[2, 3])  # [B, hidden_dim]
                        target_feat = torch.from_numpy(np.asarray(target_next[level], dtype=np.float32)).to(device)
                        if target_feat.dim() == 1:
                            target_feat = target_feat.unsqueeze(0)
                        # Pad/truncate to match hidden_dim
                        if target_feat.shape[1] < self.hidden_dim:
                            target_feat = F.pad(target_feat, (0, self.hidden_dim - target_feat.shape[1]))
                        target_feat = target_feat[:, :self.hidden_dim]

                        loss = F.mse_loss(pred, target_feat)
                        per_level_loss[level] += loss.item()
                        total_loss += loss.item()
                else:
                    # Flat vector target
                    target_feat = torch.from_numpy(np.asarray(target_next, dtype=np.float32).flatten()).to(device)
                    # Aggregate prediction
                    pred_parts = []
                    for level in self.levels:
                        if hidden_states.get(level) is not None:
                            pooled = hidden_states[level].mean(dim=[2, 3])
                            pred_parts.append(pooled)
                    if pred_parts:
                        pred = torch.cat(pred_parts, dim=1).flatten()
                        # Match sizes
                        min_len = min(len(pred), len(target_feat))
                        loss = F.mse_loss(pred[:min_len], target_feat[:min_len])
                        total_loss += loss.item()

            # Average over timesteps
            avg_loss = total_loss / max(1, num_steps - 1)
            avg_per_level = {level: loss / max(1, num_steps - 1) for level, loss in per_level_loss.items()}

            return {
                "total_loss": float(avg_loss),
                "per_level_loss": {k: float(v) for k, v in avg_per_level.items()},
            }


def run_spatial_rnn(
    features: Sequence[Any],
    hidden_dim: int = 64,
    feature_dim: int = 8,
    levels: Sequence[str] = ("P3", "P4", "P5"),
    mode: str = "convgru",
    seed: int = 0,
    use_neural: bool = False,
) -> np.ndarray:
    """
    Run spatial RNN on feature sequence (convenience function).

    Args:
        features: Sequence of feature pyramids or flat vectors
        hidden_dim: Hidden state dimension
        feature_dim: Input feature dimension
        levels: Pyramid levels
        mode: "convgru" or "s4d"
        seed: Random seed
        use_neural: If True and PyTorch available, use neural RNN; else use stub

    Returns:
        Summary vector (numpy array)
    """
    if not use_neural or not TORCH_AVAILABLE:
        # Fallback to simple decay-based stub
        if not features:
            return np.array([], dtype=np.float32)

        # Handle first feature to initialize hidden state
        first_feat = features[0]
        if isinstance(first_feat, dict):
            # Flatten pyramid
            first_arr = np.concatenate([np.asarray(v, dtype=np.float32).flatten() for v in first_feat.values()])
        else:
            first_arr = np.asarray(first_feat, dtype=np.float32).flatten()

        hidden = np.zeros_like(first_arr)
        decay = 0.65
        input_scale = 0.35

        for idx, feat in enumerate(features):
            if isinstance(feat, dict):
                # Flatten pyramid
                arr = np.concatenate([np.asarray(v, dtype=np.float32).flatten() for v in feat.values()])
            else:
                arr = np.asarray(feat, dtype=np.float32).flatten()
            if len(arr) > len(hidden):
                arr = arr[:len(hidden)]
            elif len(arr) < len(hidden):
                arr = np.pad(arr, (0, len(hidden) - len(arr)))
            gate = 1.0 / float(idx + 1)
            hidden = decay * hidden + input_scale * arr + gate * arr.mean()
        return hidden.astype(np.float32)

    # Neural mode
    model = SpatialRNN(
        hidden_dim=hidden_dim,
        feature_dim=feature_dim,
        levels=levels,
        mode=mode,
        seed=seed,
    )
    model.eval()

    with torch.no_grad():
        outputs = model.forward(list(features))

    summary = outputs["summary"].squeeze(0).cpu().numpy()
    return summary.astype(np.float32)


def tensor_to_json_safe(tensor: Union[np.ndarray, "torch.Tensor"]) -> Dict[str, Any]:
    """Convert tensor to JSON-safe dict."""
    try:
        if TORCH_AVAILABLE and isinstance(tensor, torch.Tensor):
            arr = tensor.detach().cpu().numpy()
        else:
            arr = np.asarray(tensor, dtype=np.float32)
        return {"values": [float(x) for x in arr.flatten().tolist()]}
    except Exception:
        return {"values": []}
