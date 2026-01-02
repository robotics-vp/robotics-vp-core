"""
LPIPS Perceptual Loss Wrapper.

Provides LPIPS loss with fallback gradient-based approximation.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    nn = None  # type: ignore
    TORCH_AVAILABLE = False

# Try to import real LPIPS
_REAL_LPIPS_AVAILABLE = False
try:
    import lpips as lpips_lib  # type: ignore
    _REAL_LPIPS_AVAILABLE = True
except ImportError:
    pass


class LPIPSLoss:
    """LPIPS perceptual loss wrapper.
    
    Uses real LPIPS when available, falls back to gradient-based approximation.
    """
    
    def __init__(
        self,
        net: str = "alex",
        device: str = "cuda",
        use_fallback: bool = False,
        allow_fallback: bool = True,
    ):
        """Initialize LPIPS loss.
        
        Args:
            net: Network to use ("alex", "vgg", "squeeze").
            device: Device for computation.
            use_fallback: Force fallback even if LPIPS available.
            allow_fallback: If False, raise RuntimeError when LPIPS not available.
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for LPIPS")
        
        self.device = device if torch.cuda.is_available() else "cpu"
        self.use_fallback = use_fallback
        self.allow_fallback = allow_fallback
        self._lpips_fn = None
        
        if not use_fallback:
            self._try_load_lpips(net)
    
    def _try_load_lpips(self, net: str) -> None:
        """Attempt to load real LPIPS."""
        if not _REAL_LPIPS_AVAILABLE:
            msg = "LPIPS not installed. Install with: pip install lpips"
            if not self.allow_fallback:
                raise RuntimeError(msg)
            logger.info(f"{msg} Using gradient-based fallback.")
            self.use_fallback = True
            return
        
        try:
            self._lpips_fn = lpips_lib.LPIPS(net=net).to(self.device)
            self._lpips_fn.eval()
            for param in self._lpips_fn.parameters():
                param.requires_grad = False
            logger.info(f"LPIPS ({net}) loaded successfully")
        except Exception as e:
            msg = f"Failed to load LPIPS: {e}"
            if not self.allow_fallback:
                raise RuntimeError(msg)
            logger.warning(f"{msg}. Using fallback.")
            self.use_fallback = True
    
    @property
    def is_real(self) -> bool:
        """Returns True if using real LPIPS."""
        return self._lpips_fn is not None
    
    def __call__(
        self,
        img1: "torch.Tensor",
        img2: "torch.Tensor",
    ) -> "torch.Tensor":
        """Compute LPIPS loss.
        
        Args:
            img1: (B, 3, H, W) or (3, H, W) image in [-1, 1] or [0, 1].
            img2: Same shape as img1.
        
        Returns:
            Scalar loss tensor.
        """
        # Ensure batch dimension
        if img1.ndim == 3:
            img1 = img1.unsqueeze(0)
            img2 = img2.unsqueeze(0)
        
        # Normalize to [-1, 1] if in [0, 1]
        if img1.min() >= 0 and img1.max() <= 1:
            img1 = img1 * 2 - 1
            img2 = img2 * 2 - 1
        
        if self.use_fallback:
            return self._compute_fallback(img1, img2)
        return self._compute_real(img1, img2)
    
    def _compute_real(
        self,
        img1: "torch.Tensor",
        img2: "torch.Tensor",
    ) -> "torch.Tensor":
        """Compute real LPIPS loss."""
        if self._lpips_fn is None:
            return self._compute_fallback(img1, img2)
        
        img1 = img1.to(self.device)
        img2 = img2.to(self.device)
        
        return self._lpips_fn(img1, img2).mean()
    
    def _compute_fallback(
        self,
        img1: "torch.Tensor",
        img2: "torch.Tensor",
    ) -> "torch.Tensor":
        """Compute gradient-based perceptual approximation.
        
        Uses Sobel-like gradients as a simple perceptual proxy.
        """
        # Sobel kernels
        sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1],
        ], dtype=img1.dtype, device=img1.device).view(1, 1, 3, 3)
        
        sobel_y = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1],
        ], dtype=img1.dtype, device=img1.device).view(1, 1, 3, 3)
        
        # Compute gradients for each channel
        loss = torch.tensor(0.0, device=img1.device)
        
        for c in range(3):
            c1 = img1[:, c:c+1, :, :]
            c2 = img2[:, c:c+1, :, :]
            
            # Pad to handle borders
            c1_padded = torch.nn.functional.pad(c1, (1, 1, 1, 1), mode='replicate')
            c2_padded = torch.nn.functional.pad(c2, (1, 1, 1, 1), mode='replicate')
            
            # Compute gradients
            g1x = torch.nn.functional.conv2d(c1_padded, sobel_x)
            g1y = torch.nn.functional.conv2d(c1_padded, sobel_y)
            g2x = torch.nn.functional.conv2d(c2_padded, sobel_x)
            g2y = torch.nn.functional.conv2d(c2_padded, sobel_y)
            
            # L2 difference on gradients
            loss = loss + ((g1x - g2x) ** 2).mean()
            loss = loss + ((g1y - g2y) ** 2).mean()
        
        return loss / 6  # Normalize by 6 (3 channels × 2 directions)


if __name__ == "__main__":
    if TORCH_AVAILABLE:
        lpips = LPIPSLoss(use_fallback=False)
        print(f"LPIPS wrapper initialized (real={lpips.is_real})")
        
        # Test
        img1 = torch.rand(1, 3, 64, 64)
        img2 = torch.rand(1, 3, 64, 64)
        loss = lpips(img1, img2)
        print(f"Loss: {loss.item():.4f}")
