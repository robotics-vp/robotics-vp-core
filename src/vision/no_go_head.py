"""
No-Go Zone Head for Vision-Based HRL.

Predicts binary mask of unsafe regions that should be avoided.
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


class NoGoZoneHead(nn.Module if TORCH_AVAILABLE else object):
    """
    Predicts binary mask of unsafe (no-go) zones.

    Input: Image features from encoder
    Output: Binary mask [0, 1] where 1 indicates no-go zone

    No-go zones include:
    - Fragile object locations (vase)
    - High collision risk areas
    - Out-of-bounds regions
    """

    def __init__(
        self,
        in_channels=128,
        hidden_channels=64,
        out_size=(16, 16),
        risk_threshold=0.7
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for NoGoZoneHead")

        super().__init__()

        self.out_size = out_size
        self.risk_threshold = risk_threshold

        # Shared convolutions
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
        )

        self.upsample = nn.Upsample(size=out_size, mode='bilinear', align_corners=False)

    def forward(self, features, return_soft=False):
        """
        Predict no-go zone mask.

        Args:
            features: (batch, in_channels, H, W) encoded image features
            return_soft: Return soft (sigmoid) mask instead of hard threshold

        Returns:
            no_go_mask: (batch, out_H, out_W) binary or soft mask
        """
        x = self.conv(features)  # (batch, 1, H, W)
        x = self.upsample(x)      # (batch, 1, out_H, out_W)
        x = x.squeeze(1)          # (batch, out_H, out_W)

        soft_mask = torch.sigmoid(x)

        if return_soft:
            return soft_mask
        else:
            # Hard threshold
            hard_mask = (soft_mask > self.risk_threshold).float()
            return hard_mask

    def forward_with_risk(self, features, risk_map=None):
        """
        Predict no-go zones considering external risk map.

        Args:
            features: (batch, in_channels, H, W) encoded features
            risk_map: Optional (batch, out_H, out_W) risk map

        Returns:
            no_go_mask: (batch, out_H, out_W) combined mask
            soft_risk: (batch, out_H, out_W) soft risk values
        """
        soft_mask = self.forward(features, return_soft=True)

        if risk_map is not None:
            # Combine with external risk
            # Max pooling to be conservative
            combined_risk = torch.maximum(soft_mask, risk_map)
        else:
            combined_risk = soft_mask

        hard_mask = (combined_risk > self.risk_threshold).float()

        return hard_mask, combined_risk

    def compute_loss(self, pred_mask, gt_mask, safety_weight=2.0):
        """
        Compute training loss with emphasis on safety.

        Args:
            pred_mask: Predicted soft mask (batch, H, W)
            gt_mask: Ground truth binary mask (batch, H, W)
            safety_weight: Weight for false negatives (missing no-go zones)

        Returns:
            loss: Scalar loss
        """
        # Weighted BCE - higher penalty for missing no-go zones (false negatives)
        pos_weight = torch.ones_like(gt_mask) * safety_weight
        loss = F.binary_cross_entropy(pred_mask, gt_mask, weight=pos_weight)
        return loss

    def compute_iou(self, pred_mask, gt_mask):
        """
        Compute Intersection over Union for no-go zones.

        Args:
            pred_mask: Predicted binary mask
            gt_mask: Ground truth binary mask

        Returns:
            iou: Scalar IoU score
        """
        pred_binary = (pred_mask > self.risk_threshold).float()

        intersection = (pred_binary * gt_mask).sum()
        union = pred_binary.sum() + gt_mask.sum() - intersection

        iou = (intersection + 1e-6) / (union + 1e-6)
        return iou


class NoGoZoneGenerator:
    """
    Generates ground truth no-go zone masks from environment state.
    """

    def __init__(
        self,
        image_size=(128, 128),
        object_radius=0.08,  # meters (vase radius + margin)
        workspace_bounds=(-1.0, 1.0, -1.0, 1.0)  # x_min, x_max, y_min, y_max
    ):
        self.image_size = image_size
        self.object_radius = object_radius
        self.workspace_bounds = workspace_bounds

    def generate_from_vase(self, vase_pos, safety_margin=0.1):
        """
        Generate no-go zone around vase.

        Args:
            vase_pos: (3,) vase world position
            safety_margin: Additional safety margin

        Returns:
            no_go_mask: (H, W) binary mask
        """
        H, W = self.image_size
        no_go_mask = np.zeros((H, W), dtype=np.float32)

        # Project vase to image
        vase_x_img = int((vase_pos[0] + 1) / 2 * W)
        vase_y_img = int((vase_pos[1] + 1) / 2 * H)

        # No-go radius in pixels
        total_radius = self.object_radius + safety_margin
        radius_pixels = int(total_radius * 64)

        # Create circular no-go zone
        for dy in range(-radius_pixels, radius_pixels + 1):
            for dx in range(-radius_pixels, radius_pixels + 1):
                x = vase_x_img + dx
                y = vase_y_img + dy

                if 0 <= x < W and 0 <= y < H:
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist <= radius_pixels:
                        no_go_mask[y, x] = 1.0

        return no_go_mask

    def generate_workspace_bounds_mask(self):
        """
        Generate mask for out-of-bounds workspace regions.

        Returns:
            bounds_mask: (H, W) binary mask (1 = out of bounds)
        """
        H, W = self.image_size
        bounds_mask = np.zeros((H, W), dtype=np.float32)

        x_min, x_max, y_min, y_max = self.workspace_bounds

        # Create valid workspace mask
        for y in range(H):
            for x in range(W):
                # Convert pixel to world coordinates
                world_x = (x / W) * 2 - 1
                world_y = (y / H) * 2 - 1

                # Check if outside bounds
                if (world_x < x_min or world_x > x_max or
                    world_y < y_min or world_y > y_max):
                    bounds_mask[y, x] = 1.0

        return bounds_mask

    def generate_combined_no_go(self, vase_pos, collision_history=None):
        """
        Generate combined no-go zone from multiple sources.

        Args:
            vase_pos: (3,) vase position
            collision_history: List of past collision locations

        Returns:
            combined_mask: (H, W) binary mask
        """
        # Base: vase no-go zone
        mask = self.generate_from_vase(vase_pos)

        # Add workspace bounds
        bounds_mask = self.generate_workspace_bounds_mask()
        mask = np.maximum(mask, bounds_mask)

        # Add historical collision zones
        if collision_history is not None:
            H, W = self.image_size
            for collision_point in collision_history:
                x_img = int((collision_point[0] + 1) / 2 * W)
                y_img = int((collision_point[1] + 1) / 2 * H)

                # Small no-go zone around collision
                radius = 10  # pixels
                for dy in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):
                        x = x_img + dx
                        y = y_img + dy

                        if 0 <= x < W and 0 <= y < H:
                            if np.sqrt(dx**2 + dy**2) <= radius:
                                mask[y, x] = 1.0

        return mask

    def validate_trajectory(self, waypoints, vase_pos):
        """
        Check if a trajectory passes through no-go zones.

        Args:
            waypoints: List of (x, y, z) waypoints
            vase_pos: (3,) vase position

        Returns:
            is_safe: bool
            violations: List of (waypoint_idx, distance_to_nogo)
        """
        no_go_mask = self.generate_from_vase(vase_pos)
        H, W = self.image_size

        is_safe = True
        violations = []

        for i, waypoint in enumerate(waypoints):
            x_img = int((waypoint[0] + 1) / 2 * W)
            y_img = int((waypoint[1] + 1) / 2 * H)

            # Check if in no-go zone
            if 0 <= x_img < W and 0 <= y_img < H:
                if no_go_mask[y_img, x_img] > 0.5:
                    is_safe = False
                    # Distance to edge of no-go zone
                    violations.append((i, 0.0))  # In the zone
            else:
                # Out of bounds
                is_safe = False
                violations.append((i, -1.0))  # Out of bounds

        return is_safe, violations
