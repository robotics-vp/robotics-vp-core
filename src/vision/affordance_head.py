"""
Affordance Head for Vision-Based HRL.

Predicts handle graspability and interaction affordances.
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


class AffordanceHead(nn.Module if TORCH_AVAILABLE else object):
    """
    Predicts interaction affordances (graspability, pushability, etc.).

    Input: Image features from encoder
    Output: Affordance map [0, 1] where higher values indicate more affordant regions

    Affordances include:
    - Drawer handle graspability
    - Push/pull affordances
    - Safe grasp points
    """

    def __init__(
        self,
        in_channels=128,
        hidden_channels=64,
        out_size=(16, 16),
        num_affordance_types=3  # grasp, push, pull
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for AffordanceHead")

        super().__init__()

        self.out_size = out_size
        self.num_affordance_types = num_affordance_types

        # Shared feature processing
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        # Affordance-specific heads
        self.affordance_convs = nn.ModuleList([
            nn.Conv2d(32, 1, 1) for _ in range(num_affordance_types)
        ])

        # Combined affordance (weighted sum)
        self.combined_conv = nn.Conv2d(32, 1, 1)

        self.upsample = nn.Upsample(size=out_size, mode='bilinear', align_corners=False)

    def forward(self, features, return_all=False):
        """
        Predict affordance map.

        Args:
            features: (batch, in_channels, H, W) encoded image features
            return_all: Return all affordance types separately

        Returns:
            affordance_map: (batch, out_H, out_W) combined affordance
            or if return_all:
            (combined_map, grasp_map, push_map, pull_map)
        """
        shared = self.shared_conv(features)  # (batch, 32, H, W)

        if return_all:
            affordance_maps = []
            for aff_conv in self.affordance_convs:
                aff_map = torch.sigmoid(aff_conv(shared))  # (batch, 1, H, W)
                aff_map = self.upsample(aff_map).squeeze(1)  # (batch, out_H, out_W)
                affordance_maps.append(aff_map)

            combined = torch.sigmoid(self.combined_conv(shared))
            combined = self.upsample(combined).squeeze(1)

            return combined, *affordance_maps

        else:
            combined = torch.sigmoid(self.combined_conv(shared))
            combined = self.upsample(combined).squeeze(1)
            return combined

    def compute_loss(self, pred_affordance, gt_affordance, mask=None):
        """
        Compute training loss.

        Args:
            pred_affordance: Predicted affordance map (batch, H, W)
            gt_affordance: Ground truth affordance map (batch, H, W)
            mask: Optional mask for valid regions

        Returns:
            loss: Scalar loss
        """
        if mask is not None:
            # Masked BCE loss
            pred_masked = pred_affordance * mask
            gt_masked = gt_affordance * mask
            loss = F.binary_cross_entropy(pred_masked, gt_masked)
        else:
            loss = F.binary_cross_entropy(pred_affordance, gt_affordance)

        return loss


class AffordanceMapGenerator:
    """
    Generates ground truth affordance maps from environment state.

    Used for training the AffordanceHead.
    """

    def __init__(
        self,
        image_size=(128, 128),
        handle_radius=0.05,  # meters
        affordance_spread=0.1  # meters
    ):
        self.image_size = image_size
        self.handle_radius = handle_radius
        self.affordance_spread = affordance_spread

    def generate_grasp_affordance(self, handle_pos):
        """
        Generate grasp affordance map.

        Args:
            handle_pos: (3,) handle position in world coordinates

        Returns:
            grasp_map: (H, W) grasp affordance probability
        """
        H, W = self.image_size
        grasp_map = np.zeros((H, W), dtype=np.float32)

        # Project handle to image
        handle_x_img = int((handle_pos[0] + 1) / 2 * W)
        handle_y_img = int((handle_pos[1] + 1) / 2 * H)

        # Grasp region around handle
        radius_pixels = int(self.handle_radius * 64)

        for dy in range(-radius_pixels * 3, radius_pixels * 3 + 1):
            for dx in range(-radius_pixels * 3, radius_pixels * 3 + 1):
                x = handle_x_img + dx
                y = handle_y_img + dy

                if 0 <= x < W and 0 <= y < H:
                    dist = np.sqrt(dx**2 + dy**2)
                    # Gaussian centered on handle
                    affordance = np.exp(-dist**2 / (2 * radius_pixels**2))
                    grasp_map[y, x] = affordance

        return grasp_map

    def generate_push_affordance(self, drawer_pos, drawer_normal):
        """
        Generate push affordance map.

        Args:
            drawer_pos: (3,) drawer center position
            drawer_normal: (3,) direction to push

        Returns:
            push_map: (H, W) push affordance probability
        """
        H, W = self.image_size
        push_map = np.zeros((H, W), dtype=np.float32)

        # Push region along drawer surface
        drawer_x_img = int((drawer_pos[0] + 1) / 2 * W)
        drawer_y_img = int((drawer_pos[1] + 1) / 2 * H)

        # Elongated Gaussian along push direction
        spread_pixels = int(self.affordance_spread * 64)

        for dy in range(-spread_pixels * 2, spread_pixels * 2 + 1):
            for dx in range(-spread_pixels * 2, spread_pixels * 2 + 1):
                x = drawer_x_img + dx
                y = drawer_y_img + dy

                if 0 <= x < W and 0 <= y < H:
                    # Distance perpendicular to push direction
                    # Simplified: assume push in -y direction
                    dist_perp = abs(dx)
                    dist_along = abs(dy)

                    # Higher affordance along push direction
                    affordance = np.exp(-dist_perp**2 / (spread_pixels**2)) * \
                                np.exp(-dist_along**2 / (spread_pixels * 2)**2)
                    push_map[y, x] = affordance

        return push_map

    def generate_safe_grasp_affordance(self, handle_pos, vase_pos, safety_margin=0.15):
        """
        Generate safe grasp affordance that avoids fragile objects.

        Args:
            handle_pos: (3,) handle position
            vase_pos: (3,) vase position
            safety_margin: Distance to maintain from vase

        Returns:
            safe_grasp_map: (H, W) safe grasp affordance
        """
        # Get base grasp affordance
        grasp_map = self.generate_grasp_affordance(handle_pos)

        H, W = self.image_size

        # Penalize regions near vase
        vase_x_img = int((vase_pos[0] + 1) / 2 * W)
        vase_y_img = int((vase_pos[1] + 1) / 2 * H)

        safety_pixels = int(safety_margin * 64)

        for dy in range(-safety_pixels * 2, safety_pixels * 2 + 1):
            for dx in range(-safety_pixels * 2, safety_pixels * 2 + 1):
                x = vase_x_img + dx
                y = vase_y_img + dy

                if 0 <= x < W and 0 <= y < H:
                    dist = np.sqrt(dx**2 + dy**2)
                    # Penalty factor
                    if dist < safety_pixels:
                        penalty = 1.0 - (dist / safety_pixels)
                        grasp_map[y, x] *= (1.0 - penalty)

        return grasp_map

    def generate_drawer_handle_detection(self, image, handle_color_hint=None):
        """
        Detect drawer handle from RGB image using simple heuristics.

        Args:
            image: (H, W, 3) RGB image
            handle_color_hint: Optional (3,) RGB color of handle

        Returns:
            detection_map: (H, W) handle likelihood
        """
        H, W, _ = image.shape
        detection_map = np.zeros((H, W), dtype=np.float32)

        # Simple color-based detection
        if handle_color_hint is not None:
            # Distance from hint color
            color_dist = np.linalg.norm(
                image.astype(np.float32) - handle_color_hint, axis=-1
            )
            # Normalize and invert (closer = higher score)
            max_dist = np.sqrt(3 * 255**2)
            detection_map = 1.0 - (color_dist / max_dist)
        else:
            # Use edges as proxy for handle
            # Simple gradient magnitude
            gray = np.mean(image, axis=-1)
            grad_x = np.gradient(gray, axis=1)
            grad_y = np.gradient(gray, axis=0)
            gradient_mag = np.sqrt(grad_x**2 + grad_y**2)

            # Normalize
            if gradient_mag.max() > 0:
                detection_map = gradient_mag / gradient_mag.max()

        return detection_map
