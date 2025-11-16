"""
MLP encoder for observation → latent representation.

Architecture: state → 256 ReLU → 256 ReLU → 128 (LayerNorm)

This encoder learns what features matter for economic performance
(profit/quality tradeoff) instead of using hand-crafted features.
"""
import torch
import torch.nn as nn


class MLPEncoder(nn.Module):
    """
    MLP encoder for state observations.

    Maps raw state dict → learned latent representation.
    Ready to swap with video encoder later (same interface).
    """

    def __init__(self, obs_dim, latent_dim=128, hidden_dim=256):
        super().__init__()

        self.obs_dim = obs_dim
        self.latent_dim = latent_dim

        # Encoder backbone
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim)  # Stabilize latent distribution
        )

        # Initialize with Xavier (better for deep networks)
        for layer in self.encoder:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, obs):
        """
        Encode observation to latent.

        Args:
            obs: [B, obs_dim] tensor or dict

        Returns:
            latent: [B, latent_dim] tensor
        """
        if isinstance(obs, dict):
            obs = self._obs_dict_to_tensor(obs)

        return self.encoder(obs)

    def _obs_dict_to_tensor(self, obs_dict):
        """Convert observation dict to tensor."""
        # For dishwashing: [t, completed, attempts, errors]
        features = []
        for key in ['t', 'completed', 'attempts', 'errors']:
            features.append(obs_dict[key])
        return torch.FloatTensor(features).unsqueeze(0)


class ConsistencyHead(nn.Module):
    """
    Auxiliary head for consistency loss: predict next latent.

    Loss: ||f_ψ(o_{t+1}) - f_ψ(ô_{t+1})||
    """

    def __init__(self, latent_dim, hidden_dim=256):
        super().__init__()

        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, latent):
        """Predict next latent given current."""
        return self.predictor(latent)


class ContrastiveHead(nn.Module):
    """
    Auxiliary head for contrastive learning (SimCLR-style).

    Projects latent to contrastive space for InfoNCE loss.
    """

    def __init__(self, latent_dim, proj_dim=128):
        super().__init__()

        self.projector = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, proj_dim)
        )

    def forward(self, latent):
        """Project latent for contrastive learning."""
        return self.projector(latent)


class EncoderWithAuxiliaries(nn.Module):
    """
    Encoder with auxiliary self-supervised heads.

    Loss = L_RL + λ_c * L_contrastive + λ_k * L_consistency
    """

    def __init__(self, obs_dim, latent_dim=128, hidden_dim=256,
                 use_consistency=True, use_contrastive=True):
        super().__init__()

        self.encoder = MLPEncoder(obs_dim, latent_dim, hidden_dim)

        # Auxiliary heads (optional)
        self.use_consistency = use_consistency
        self.use_contrastive = use_contrastive

        if use_consistency:
            self.consistency_head = ConsistencyHead(latent_dim, hidden_dim)

        if use_contrastive:
            self.contrastive_head = ContrastiveHead(latent_dim, proj_dim=128)

    def encode(self, obs):
        """Main encoding (for RL)."""
        return self.encoder(obs)

    def predict_next(self, latent):
        """Predict next latent (for consistency loss)."""
        if not self.use_consistency:
            raise RuntimeError("Consistency head not enabled")
        return self.consistency_head(latent)

    def project_contrastive(self, latent):
        """Project for contrastive loss."""
        if not self.use_contrastive:
            raise RuntimeError("Contrastive head not enabled")
        return self.contrastive_head(latent)

    def compute_consistency_loss(self, latent_t, latent_t1_true):
        """
        Consistency loss: ||predicted_next - true_next||².

        Args:
            latent_t: Current latent [B, latent_dim]
            latent_t1_true: True next latent [B, latent_dim]

        Returns:
            loss: Scalar consistency loss
        """
        predicted_next = self.predict_next(latent_t)
        return nn.functional.mse_loss(predicted_next, latent_t1_true.detach())

    def compute_contrastive_loss(self, latent_batch, temperature=0.1):
        """
        InfoNCE contrastive loss (simplified version).

        Positive pairs: temporally close transitions
        Negative pairs: other samples in batch

        Args:
            latent_batch: [B, latent_dim]
            temperature: Softmax temperature

        Returns:
            loss: Scalar contrastive loss
        """
        # Project to contrastive space
        proj = self.project_contrastive(latent_batch)  # [B, proj_dim]

        # Normalize
        proj = nn.functional.normalize(proj, dim=1)

        # Similarity matrix
        sim_matrix = torch.matmul(proj, proj.T) / temperature  # [B, B]

        # Mask out diagonal (self-similarity)
        mask = torch.eye(sim_matrix.size(0), device=sim_matrix.device).bool()
        sim_matrix = sim_matrix.masked_fill(mask, float('-inf'))

        # Loss: encourage uniform similarity (simple version)
        # In full SimCLR: would have positive pairs from augmentation
        loss = -torch.log_softmax(sim_matrix, dim=1).mean()

        return loss
