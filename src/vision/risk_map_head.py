"""
Risk Map Head for Vision-Based HRL.

Predicts per-pixel fragility/collision risk probability.
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


class RiskMapHead(nn.Module if TORCH_AVAILABLE else object):
    """
    Predicts per-pixel collision/fragility risk.

    Input: Image features from encoder
    Output: Risk map [0, 1] where higher values indicate higher risk

    The risk map represents:
    - Proximity to fragile objects (vase)
    - Historical collision locations
    - Areas of high uncertainty
    """

    def __init__(
        self,
        in_channels=128,
        hidden_channels=64,
        out_size=(16, 16)
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for RiskMapHead")

        super().__init__()

        self.out_size = out_size

        # Convolutional layers
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid(),
        )

        # Adaptive upsampling to target size
        self.upsample = nn.Upsample(size=out_size, mode='bilinear', align_corners=False)

    def forward(self, features):
        """
        Predict risk map.

        Args:
            features: (batch, in_channels, H, W) encoded image features

        Returns:
            risk_map: (batch, out_H, out_W) in [0, 1]
        """
        x = self.conv(features)  # (batch, 1, H, W)
        x = self.upsample(x)      # (batch, 1, out_H, out_W)
        return x.squeeze(1)       # (batch, out_H, out_W)

    def compute_loss(self, pred_risk_map, gt_risk_map, collision_mask=None):
        """
        Compute training loss.

        Args:
            pred_risk_map: Predicted risk map (batch, H, W)
            gt_risk_map: Ground truth risk map (batch, H, W)
            collision_mask: Optional binary mask of actual collisions (batch, H, W)

        Returns:
            loss: Scalar loss
        """
        # Binary cross-entropy loss
        bce_loss = F.binary_cross_entropy(pred_risk_map, gt_risk_map)

        # Optional: Weight by collision events
        if collision_mask is not None:
            # Higher weight for actual collision regions
            weighted_loss = (collision_mask * 10.0 + 1.0) * F.binary_cross_entropy(
                pred_risk_map, gt_risk_map, reduction='none'
            )
            bce_loss = weighted_loss.mean()

        return bce_loss


class RiskMapGenerator:
    """
    Generates ground truth risk maps from environment state.

    Used for training the RiskMapHead.
    """

    def __init__(
        self,
        image_size=(128, 128),
        risk_radius=0.15,  # meters
        max_risk=1.0
    ):
        self.image_size = image_size
        self.risk_radius = risk_radius
        self.max_risk = max_risk

    def generate_from_state(self, vase_pos, camera_intrinsics=None):
        """
        Generate risk map from vase position.

        Args:
            vase_pos: (3,) vase position in world coordinates
            camera_intrinsics: Optional camera parameters for projection

        Returns:
            risk_map: (H, W) risk probability
        """
        H, W = self.image_size
        risk_map = np.zeros((H, W), dtype=np.float32)

        # Project vase position to image coordinates
        # For simplicity, assume orthographic projection from top-down view
        # In real implementation, use actual camera parameters

        # Map world coordinates to image
        # Assuming workspace: x in [-1, 1], y in [-1, 1]
        vase_x_img = int((vase_pos[0] + 1) / 2 * W)
        vase_y_img = int((vase_pos[1] + 1) / 2 * H)

        # Create Gaussian risk distribution around vase
        # Radius in pixels (assuming 1m = 64 pixels approximately)
        radius_pixels = int(self.risk_radius * 64)

        # Generate risk heatmap
        for dy in range(-radius_pixels * 2, radius_pixels * 2 + 1):
            for dx in range(-radius_pixels * 2, radius_pixels * 2 + 1):
                x = vase_x_img + dx
                y = vase_y_img + dy

                if 0 <= x < W and 0 <= y < H:
                    dist = np.sqrt(dx**2 + dy**2)
                    # Gaussian falloff
                    risk = np.exp(-dist**2 / (2 * radius_pixels**2))
                    risk_map[y, x] = max(risk_map[y, x], risk * self.max_risk)

        return risk_map

    def generate_from_collision_history(self, collision_points, image_size=None):
        """
        Generate risk map from historical collision data.

        Args:
            collision_points: List of (x, y, z) collision locations
            image_size: Optional override for image size

        Returns:
            risk_map: (H, W) risk probability
        """
        if image_size is None:
            H, W = self.image_size
        else:
            H, W = image_size

        risk_map = np.zeros((H, W), dtype=np.float32)

        for point in collision_points:
            # Project to image
            x_img = int((point[0] + 1) / 2 * W)
            y_img = int((point[1] + 1) / 2 * H)

            # Add risk spike
            radius = 10  # pixels
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    x = x_img + dx
                    y = y_img + dy

                    if 0 <= x < W and 0 <= y < H:
                        dist = np.sqrt(dx**2 + dy**2)
                        risk = np.exp(-dist**2 / (2 * (radius/2)**2))
                        risk_map[y, x] = max(risk_map[y, x], risk)

        return risk_map

    def generate_from_clearance(self, ee_pos, vase_pos, clearance_threshold=0.15):
        """
        Generate risk map based on current clearance.

        Args:
            ee_pos: (3,) end-effector position
            vase_pos: (3,) vase position
            clearance_threshold: Safety threshold

        Returns:
            risk_map: (H, W) risk probability
        """
        H, W = self.image_size

        # Base risk from vase
        risk_map = self.generate_from_state(vase_pos)

        # Increase risk near current EE if clearance is low
        clearance = np.linalg.norm(ee_pos - vase_pos)

        if clearance < clearance_threshold:
            # High risk zone around EE
            ee_x_img = int((ee_pos[0] + 1) / 2 * W)
            ee_y_img = int((ee_pos[1] + 1) / 2 * H)

            risk_factor = 1.0 - (clearance / clearance_threshold)  # 0 to 1

            radius = int(20 * risk_factor)
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    x = ee_x_img + dx
                    y = ee_y_img + dy

                    if 0 <= x < W and 0 <= y < H:
                        dist = np.sqrt(dx**2 + dy**2)
                        risk = np.exp(-dist**2 / (2 * (radius/2)**2)) * risk_factor
                        risk_map[y, x] = max(risk_map[y, x], risk)

        return risk_map
