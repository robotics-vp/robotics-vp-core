"""
Vision Encoder with Multiple Heads.

Combines CNN encoder with risk, affordance, no-go, and fragility heads.
"""

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    class nn:
        class Module:
            pass

from .risk_map_head import RiskMapHead
from .affordance_head import AffordanceHead
from .no_go_head import NoGoZoneHead
from .fragility_prior_head import FragilityPriorHead


class SimpleCNNEncoder(nn.Module if TORCH_AVAILABLE else object):
    """
    Simple CNN encoder for image features.
    """

    def __init__(self, in_channels=3, feature_dim=128):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")

        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),  # 64x64
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 32x32
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 16x16
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, feature_dim, 3, stride=1, padding=1),  # 16x16
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        """
        Args:
            x: (batch, C, H, W) image

        Returns:
            features: (batch, feature_dim, H', W')
        """
        return self.encoder(x)


class VisionEncoderWithHeads(nn.Module if TORCH_AVAILABLE else object):
    """
    Complete vision encoder with multiple prediction heads.

    Combines:
    - CNN encoder for feature extraction
    - Risk map head for collision risk
    - Affordance head for interaction affordances
    - No-go zone head for unsafe regions
    - Fragility prior head for fragile object detection

    Also produces a latent vector (z_V) for downstream tasks.
    """

    def __init__(
        self,
        in_channels=3,
        feature_dim=128,
        z_v_dim=128,
        map_size=(16, 16),
        use_risk_head=True,
        use_affordance_head=True,
        use_nogo_head=True,
        use_fragility_head=True
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for VisionEncoderWithHeads")

        super().__init__()

        self.feature_dim = feature_dim
        self.z_v_dim = z_v_dim
        self.map_size = map_size

        # CNN encoder
        self.encoder = SimpleCNNEncoder(in_channels, feature_dim)

        # Global average pooling for z_V
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.z_v_projection = nn.Linear(feature_dim, z_v_dim)

        # Prediction heads
        self.use_risk_head = use_risk_head
        self.use_affordance_head = use_affordance_head
        self.use_nogo_head = use_nogo_head
        self.use_fragility_head = use_fragility_head

        if use_risk_head:
            self.risk_head = RiskMapHead(feature_dim, out_size=map_size)

        if use_affordance_head:
            self.affordance_head = AffordanceHead(feature_dim, out_size=map_size)

        if use_nogo_head:
            self.nogo_head = NoGoZoneHead(feature_dim, out_size=map_size)

        if use_fragility_head:
            self.fragility_head = FragilityPriorHead(feature_dim, out_size=map_size)

    def forward(self, image, return_all=True):
        """
        Process image and compute all outputs.

        Args:
            image: (batch, C, H, W) input image (C=3 for RGB)
            return_all: Return all head outputs

        Returns:
            outputs: dict with:
                - z_v: (batch, z_v_dim) latent vector
                - risk_map: (batch, H', W') risk probability
                - affordance_map: (batch, H', W') affordance scores
                - nogo_mask: (batch, H', W') binary no-go mask
                - fragility_map: (batch, H', W') fragility heatmap
                - features: (batch, feature_dim, H', W') intermediate features
        """
        # Encode image
        features = self.encoder(image)  # (batch, feature_dim, H', W')

        # Compute z_V (global latent)
        pooled = self.global_pool(features).squeeze(-1).squeeze(-1)  # (batch, feature_dim)
        z_v = self.z_v_projection(pooled)  # (batch, z_v_dim)

        outputs = {
            'z_v': z_v,
            'features': features,
        }

        if not return_all:
            return outputs

        # Compute head outputs
        if self.use_risk_head:
            outputs['risk_map'] = self.risk_head(features)

        if self.use_affordance_head:
            outputs['affordance_map'] = self.affordance_head(features)

        if self.use_nogo_head:
            outputs['nogo_mask'] = self.nogo_head(features)

        if self.use_fragility_head:
            outputs['fragility_map'] = self.fragility_head(features)

        return outputs

    def compute_combined_safety_map(self, outputs):
        """
        Compute combined safety map from all heads.

        Args:
            outputs: dict from forward()

        Returns:
            safety_map: (batch, H', W') combined safety scores
        """
        batch_size = outputs['z_v'].shape[0]
        device = outputs['z_v'].device
        H, W = self.map_size

        # Initialize with ones (safe everywhere)
        safety_map = torch.ones(batch_size, H, W, device=device)

        # Reduce safety where risk is high
        if 'risk_map' in outputs:
            safety_map = safety_map * (1.0 - outputs['risk_map'])

        # Zero out no-go zones
        if 'nogo_mask' in outputs:
            safety_map = safety_map * (1.0 - outputs['nogo_mask'])

        # Reduce safety in fragile regions
        if 'fragility_map' in outputs:
            safety_map = safety_map * (1.0 - outputs['fragility_map'] * 0.5)

        return safety_map

    def compute_action_safety_score(self, outputs, action_direction):
        """
        Compute safety score for a proposed action.

        Args:
            outputs: dict from forward()
            action_direction: (batch, 2) normalized direction in image space

        Returns:
            safety_score: (batch,) score in [0, 1]
        """
        # Get combined safety map
        safety_map = self.compute_combined_safety_map(outputs)  # (batch, H, W)

        # Sample along action direction
        # For simplicity, use center of image as current position
        batch_size = safety_map.shape[0]
        H, W = self.map_size
        center = torch.tensor([H // 2, W // 2], device=safety_map.device, dtype=torch.float)

        # Sample points along direction
        num_samples = 10
        scores = []

        for i in range(num_samples):
            t = (i + 1) / num_samples
            # Sample position (simplified: 2D projection)
            sample_pos = center + action_direction * t * (H // 2)

            # Clamp to valid range
            sample_y = torch.clamp(sample_pos[:, 0].long(), 0, H - 1)
            sample_x = torch.clamp(sample_pos[:, 1].long(), 0, W - 1)

            # Get safety at sample points
            batch_indices = torch.arange(batch_size, device=safety_map.device)
            sample_safety = safety_map[batch_indices, sample_y, sample_x]
            scores.append(sample_safety)

        # Average safety along trajectory
        safety_score = torch.stack(scores, dim=-1).mean(dim=-1)

        return safety_score

    def get_latent_and_maps(self, image):
        """
        Convenience function to get z_V and all maps.

        Args:
            image: (batch, C, H, W) input image

        Returns:
            z_v: (batch, z_v_dim)
            risk_map: (batch, H', W')
            affordance_map: (batch, H', W')
            nogo_mask: (batch, H', W')
            fragility_map: (batch, H', W')
        """
        outputs = self.forward(image, return_all=True)

        z_v = outputs['z_v']
        risk_map = outputs.get('risk_map', None)
        affordance_map = outputs.get('affordance_map', None)
        nogo_mask = outputs.get('nogo_mask', None)
        fragility_map = outputs.get('fragility_map', None)

        return z_v, risk_map, affordance_map, nogo_mask, fragility_map

    def save(self, path):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'feature_dim': self.feature_dim,
            'z_v_dim': self.z_v_dim,
            'map_size': self.map_size,
            'use_risk_head': self.use_risk_head,
            'use_affordance_head': self.use_affordance_head,
            'use_nogo_head': self.use_nogo_head,
            'use_fragility_head': self.use_fragility_head,
        }, path)

    @classmethod
    def load(cls, path, device='cpu'):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        model = cls(
            feature_dim=checkpoint['feature_dim'],
            z_v_dim=checkpoint['z_v_dim'],
            map_size=checkpoint['map_size'],
            use_risk_head=checkpoint['use_risk_head'],
            use_affordance_head=checkpoint['use_affordance_head'],
            use_nogo_head=checkpoint['use_nogo_head'],
            use_fragility_head=checkpoint['use_fragility_head'],
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        return model


class VisionHeadTrainer:
    """
    Training utilities for vision heads.
    """

    def __init__(self, model, lr=1e-4, device='cpu'):
        self.model = model.to(device)
        self.device = device

        if TORCH_AVAILABLE:
            self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def train_step(self, images, gt_risk=None, gt_affordance=None, gt_nogo=None, gt_fragility=None):
        """
        Single training step.

        Args:
            images: (batch, C, H, W) input images
            gt_risk: Optional ground truth risk maps
            gt_affordance: Optional ground truth affordance maps
            gt_nogo: Optional ground truth no-go masks
            gt_fragility: Optional ground truth fragility maps

        Returns:
            losses: dict with individual losses
        """
        self.model.train()
        self.optimizer.zero_grad()

        outputs = self.model(images.to(self.device), return_all=True)

        losses = {}
        total_loss = 0.0

        # Risk loss
        if gt_risk is not None and 'risk_map' in outputs:
            risk_loss = F.binary_cross_entropy(outputs['risk_map'], gt_risk.to(self.device))
            losses['risk'] = risk_loss.item()
            total_loss = total_loss + risk_loss

        # Affordance loss
        if gt_affordance is not None and 'affordance_map' in outputs:
            aff_loss = F.binary_cross_entropy(outputs['affordance_map'], gt_affordance.to(self.device))
            losses['affordance'] = aff_loss.item()
            total_loss = total_loss + aff_loss

        # No-go loss
        if gt_nogo is not None and 'nogo_mask' in outputs:
            nogo_loss = F.binary_cross_entropy(
                self.model.nogo_head(outputs['features'], return_soft=True),
                gt_nogo.to(self.device)
            )
            losses['nogo'] = nogo_loss.item()
            total_loss = total_loss + nogo_loss

        # Fragility loss
        if gt_fragility is not None and 'fragility_map' in outputs:
            frag_loss = F.mse_loss(outputs['fragility_map'], gt_fragility.to(self.device))
            losses['fragility'] = frag_loss.item()
            total_loss = total_loss + frag_loss

        losses['total'] = total_loss.item()

        # Backprop
        if isinstance(total_loss, torch.Tensor):
            total_loss.backward()
            self.optimizer.step()

        return losses

    def evaluate(self, images, gt_risk=None, gt_nogo=None):
        """
        Evaluate model performance.

        Args:
            images: (batch, C, H, W)
            gt_risk: Ground truth risk maps
            gt_nogo: Ground truth no-go masks

        Returns:
            metrics: dict with evaluation metrics
        """
        self.model.eval()

        with torch.no_grad():
            outputs = self.model(images.to(self.device), return_all=True)

        metrics = {}

        # Risk map accuracy
        if gt_risk is not None and 'risk_map' in outputs:
            pred_risk = outputs['risk_map']
            # Binary accuracy at 0.5 threshold
            pred_binary = (pred_risk > 0.5).float()
            gt_binary = (gt_risk.to(self.device) > 0.5).float()
            accuracy = (pred_binary == gt_binary).float().mean()
            metrics['risk_accuracy'] = accuracy.item()

        # No-go IoU
        if gt_nogo is not None and 'nogo_mask' in outputs:
            pred_nogo = outputs['nogo_mask']
            intersection = (pred_nogo * gt_nogo.to(self.device)).sum()
            union = pred_nogo.sum() + gt_nogo.to(self.device).sum() - intersection
            iou = (intersection + 1e-6) / (union + 1e-6)
            metrics['nogo_iou'] = iou.item()

        return metrics
