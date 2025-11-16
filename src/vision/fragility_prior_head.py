"""
Fragility Prior Head for Vision-Based HRL.

Draws heatmaps around detected fragile objects.
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


class FragilityPriorHead(nn.Module if TORCH_AVAILABLE else object):
    """
    Predicts fragility priors - heatmaps indicating fragile object locations
    and their fragility level.

    Input: Image features from encoder
    Output: Fragility heatmap [0, 1] where higher values indicate more fragile regions

    Used to:
    - Identify fragile objects (glass, ceramic, delicate items)
    - Estimate fragility level
    - Generate protective buffers around fragile objects
    """

    def __init__(
        self,
        in_channels=128,
        hidden_channels=64,
        out_size=(16, 16),
        num_fragility_levels=3  # low, medium, high
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for FragilityPriorHead")

        super().__init__()

        self.out_size = out_size
        self.num_fragility_levels = num_fragility_levels

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        # Detection head (where are fragile objects?)
        self.detection_head = nn.Sequential(
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid(),
        )

        # Fragility level head (how fragile?)
        self.fragility_head = nn.Sequential(
            nn.Conv2d(32, num_fragility_levels, 1),
            nn.Softmax(dim=1),
        )

        # Heatmap spread head (how far does the fragility extend?)
        self.spread_head = nn.Sequential(
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid(),
        )

        self.upsample = nn.Upsample(size=out_size, mode='bilinear', align_corners=False)

    def forward(self, features, return_all=False):
        """
        Predict fragility heatmap.

        Args:
            features: (batch, in_channels, H, W) encoded image features
            return_all: Return all components separately

        Returns:
            fragility_heatmap: (batch, out_H, out_W) combined heatmap
            or if return_all:
            (heatmap, detection_map, fragility_levels, spread_map)
        """
        shared = self.shared(features)

        # Detection: where are fragile objects?
        detection = self.detection_head(shared)  # (batch, 1, H, W)
        detection = self.upsample(detection).squeeze(1)  # (batch, out_H, out_W)

        # Fragility levels: how fragile?
        frag_levels = self.fragility_head(shared)  # (batch, num_levels, H, W)
        frag_levels = self.upsample(frag_levels)  # (batch, num_levels, out_H, out_W)

        # Weighted fragility score
        # Level weights: [0.3, 0.6, 1.0] for low, medium, high
        level_weights = torch.tensor(
            [0.3, 0.6, 1.0], device=features.device
        ).view(1, -1, 1, 1)
        fragility_score = (frag_levels * level_weights).sum(dim=1)  # (batch, out_H, out_W)

        # Spread: how far does fragility extend?
        spread = self.spread_head(shared)  # (batch, 1, H, W)
        spread = self.upsample(spread).squeeze(1)  # (batch, out_H, out_W)

        # Combined heatmap
        # Detection * fragility score * spread factor
        heatmap = detection * fragility_score * (1.0 + spread)

        # Normalize to [0, 1]
        heatmap = torch.clamp(heatmap, 0, 1)

        if return_all:
            return heatmap, detection, frag_levels, spread
        else:
            return heatmap

    def compute_loss(self, pred_heatmap, gt_heatmap, gt_detection=None, gt_levels=None):
        """
        Compute training loss.

        Args:
            pred_heatmap: Predicted heatmap (batch, H, W)
            gt_heatmap: Ground truth heatmap (batch, H, W)
            gt_detection: Optional detection ground truth (batch, H, W)
            gt_levels: Optional fragility levels ground truth (batch, num_levels, H, W)

        Returns:
            loss: Scalar loss
        """
        # Main heatmap loss (MSE for regression)
        heatmap_loss = F.mse_loss(pred_heatmap, gt_heatmap)

        total_loss = heatmap_loss

        # Optional: Add detection and level losses if ground truth available
        # This would require returning intermediate outputs during training

        return total_loss

    def generate_protective_buffer(self, fragility_heatmap, buffer_threshold=0.3):
        """
        Generate protective buffer zones around fragile regions.

        Args:
            fragility_heatmap: (batch, H, W) fragility scores
            buffer_threshold: Threshold for creating buffer

        Returns:
            buffer_mask: (batch, H, W) binary mask of buffer zones
        """
        # Dilate fragility regions to create buffer
        # Use max pooling as morphological dilation
        kernel_size = 5
        padding = kernel_size // 2

        heatmap_expanded = fragility_heatmap.unsqueeze(1)  # (batch, 1, H, W)
        dilated = F.max_pool2d(
            heatmap_expanded, kernel_size, stride=1, padding=padding
        ).squeeze(1)

        buffer_mask = (dilated > buffer_threshold).float()

        return buffer_mask


class FragilityPriorGenerator:
    """
    Generates ground truth fragility priors from environment state.
    """

    def __init__(
        self,
        image_size=(128, 128),
        default_fragility_radius=0.1,  # meters
        fragility_levels={'low': 0.3, 'medium': 0.6, 'high': 1.0}
    ):
        self.image_size = image_size
        self.default_radius = default_fragility_radius
        self.fragility_levels = fragility_levels

    def generate_for_vase(
        self,
        vase_pos,
        fragility_level='high',
        material_type='glass'
    ):
        """
        Generate fragility heatmap for a vase.

        Args:
            vase_pos: (3,) vase world position
            fragility_level: 'low', 'medium', or 'high'
            material_type: Material type for estimating fragility

        Returns:
            heatmap: (H, W) fragility heatmap
        """
        H, W = self.image_size
        heatmap = np.zeros((H, W), dtype=np.float32)

        # Get fragility score
        frag_score = self.fragility_levels.get(fragility_level, 0.6)

        # Material-based adjustments
        if material_type == 'glass':
            frag_score *= 1.2  # Glass is very fragile
        elif material_type == 'ceramic':
            frag_score *= 1.0
        elif material_type == 'metal':
            frag_score *= 0.3

        frag_score = min(frag_score, 1.0)

        # Project to image
        vase_x_img = int((vase_pos[0] + 1) / 2 * W)
        vase_y_img = int((vase_pos[1] + 1) / 2 * H)

        # Create Gaussian heatmap
        radius_pixels = int(self.default_radius * 64)

        for dy in range(-radius_pixels * 3, radius_pixels * 3 + 1):
            for dx in range(-radius_pixels * 3, radius_pixels * 3 + 1):
                x = vase_x_img + dx
                y = vase_y_img + dy

                if 0 <= x < W and 0 <= y < H:
                    dist = np.sqrt(dx**2 + dy**2)
                    # Gaussian with fragility score
                    value = frag_score * np.exp(-dist**2 / (2 * radius_pixels**2))
                    heatmap[y, x] = max(heatmap[y, x], value)

        return heatmap

    def generate_for_multiple_objects(self, objects):
        """
        Generate combined fragility heatmap for multiple objects.

        Args:
            objects: List of dicts with 'position', 'fragility_level', 'material'

        Returns:
            combined_heatmap: (H, W) combined fragility heatmap
        """
        H, W = self.image_size
        combined = np.zeros((H, W), dtype=np.float32)

        for obj in objects:
            obj_heatmap = self.generate_for_vase(
                obj['position'],
                obj.get('fragility_level', 'medium'),
                obj.get('material', 'ceramic')
            )
            # Max combine (most fragile wins)
            combined = np.maximum(combined, obj_heatmap)

        return combined

    def estimate_fragility_from_image(self, image, object_detector=None):
        """
        Estimate fragility from visual features (simplified).

        Args:
            image: (H, W, 3) RGB image
            object_detector: Optional object detection model

        Returns:
            estimated_heatmap: (H, W) estimated fragility
        """
        H, W, _ = image.shape
        heatmap = np.zeros((H, W), dtype=np.float32)

        # Simple heuristics based on color and texture
        # Glass/ceramic tends to have:
        # - Smooth gradients
        # - Reflective (bright) regions
        # - Specific colors (transparent, white, brown)

        # Convert to grayscale
        gray = np.mean(image, axis=-1)

        # High brightness regions (reflections)
        brightness = gray / 255.0
        bright_mask = brightness > 0.8

        # Smooth regions (low texture variance)
        from scipy.ndimage import uniform_filter

        local_mean = uniform_filter(gray, size=5)
        local_sqr_mean = uniform_filter(gray**2, size=5)
        variance = local_sqr_mean - local_mean**2
        smooth_mask = variance < 100  # Low variance = smooth

        # Combine signals
        fragility_signal = (bright_mask.astype(float) * 0.3 +
                           smooth_mask.astype(float) * 0.7)

        # Smooth the result
        from scipy.ndimage import gaussian_filter
        heatmap = gaussian_filter(fragility_signal, sigma=3)

        return heatmap.astype(np.float32)

    def adaptive_buffer_size(self, fragility_score, ee_velocity):
        """
        Compute adaptive safety buffer based on fragility and velocity.

        Args:
            fragility_score: float in [0, 1]
            ee_velocity: float (m/s)

        Returns:
            buffer_size: float (meters)
        """
        # Base buffer
        base_buffer = 0.05  # 5cm

        # Fragility contribution
        fragility_buffer = fragility_score * 0.1  # up to 10cm

        # Velocity contribution (higher velocity = larger buffer)
        velocity_buffer = ee_velocity * 0.2  # up to 20cm at 1m/s

        total_buffer = base_buffer + fragility_buffer + velocity_buffer

        return min(total_buffer, 0.3)  # Cap at 30cm
