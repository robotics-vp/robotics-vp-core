"""
SAM3D-Objects Wrapper.

Provides a clean interface to SAM3D-Objects inference with:
- Real model integration when available
- Fallback stub for testing without weights
- CPU and GPU support
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Try to import real SAM3D-Objects
_REAL_SAM3D_OBJECTS_AVAILABLE = False
try:
    # Import paths may vary depending on installation
    from sam3d_objects import SAM3D_Objects  # type: ignore
    _REAL_SAM3D_OBJECTS_AVAILABLE = True
except ImportError:
    try:
        # Try alternate import path for submodule
        from third_party.sam3d_objects import SAM3D_Objects  # type: ignore
        _REAL_SAM3D_OBJECTS_AVAILABLE = True
    except ImportError:
        pass

# Try to import torch for tensor operations
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    TORCH_AVAILABLE = False


@dataclass
class ObjectInferenceResult:
    """Result from SAM3D-Objects inference.
    
    Attributes:
        object_id: Identifier for this object.
        shape_latent: (D,) shape latent embedding.
        appearance_latent: (D,) appearance/texture latent.
        position: (3,) world position.
        rotation: (4,) quaternion [x, y, z, w].
        scale: Uniform scale factor.
        geometry: Geometry dict with 'type' and representation data.
        confidence: Model confidence [0, 1].
    """
    object_id: str
    shape_latent: np.ndarray
    appearance_latent: np.ndarray
    position: np.ndarray
    rotation: np.ndarray
    scale: float
    geometry: Dict[str, Any]
    confidence: float = 1.0


class SAM3DObjectsInference:
    """Wrapper for SAM3D-Objects inference.
    
    Supports:
    - Real inference when SAM3D-Objects is installed and weights available
    - Fallback stub for testing
    - CPU and GPU execution
    """
    
    DEFAULT_WEIGHTS_PATH = "checkpoints/sam3d_objects/checkpoint.pth"
    LATENT_DIM = 256
    
    def __init__(
        self,
        weights_path: Optional[str] = None,
        device: str = "cuda",
        use_fallback: bool = False,
        allow_fallback: bool = True,
    ):
        """Initialize inference wrapper.
        
        Args:
            weights_path: Path to model weights. Uses default if None.
            device: Device for inference ("cuda" or "cpu").
            use_fallback: Force use of fallback stub even if model available.
            allow_fallback: If False, raise RuntimeError when deps/weights missing.
        """
        self.device = device if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
        self.use_fallback = use_fallback
        self.allow_fallback = allow_fallback
        self.weights_path = weights_path or self.DEFAULT_WEIGHTS_PATH
        self._model = None
        
        if not use_fallback:
            self._try_load_model()
    
    def _try_load_model(self) -> None:
        """Attempt to load the real model."""
        if not _REAL_SAM3D_OBJECTS_AVAILABLE:
            msg = (
                "SAM3D-Objects not installed. "
                "Install with: git clone https://github.com/facebookresearch/sam-3d-objects.git third_party/sam3d_objects && "
                "pip install -e third_party/sam3d_objects"
            )
            if not self.allow_fallback:
                raise RuntimeError(msg)
            logger.info(f"{msg} Using fallback stub.")
            self.use_fallback = True
            return
        
        if not Path(self.weights_path).exists():
            msg = (
                f"SAM3D-Objects weights not found at {self.weights_path}. "
                "Download weights from: https://github.com/facebookresearch/sam-3d-objects/releases"
            )
            if not self.allow_fallback:
                raise RuntimeError(msg)
            logger.warning(f"{msg} Using fallback stub.")
            self.use_fallback = True
            return
        
        try:
            # Load real model
            self._model = SAM3D_Objects.from_pretrained(self.weights_path)  # type: ignore
            if self.device == "cuda":
                self._model = self._model.cuda()
            self._model.eval()
            logger.info("SAM3D-Objects model loaded successfully")
        except Exception as e:
            msg = f"Failed to load SAM3D-Objects model: {e}"
            if not self.allow_fallback:
                raise RuntimeError(msg)
            logger.warning(f"{msg}. Using fallback.")
            self.use_fallback = True
    
    @property
    def is_real(self) -> bool:
        """Returns True if using real model, False if fallback."""
        return self._model is not None and not self.use_fallback
    
    def infer(
        self,
        rgb: np.ndarray,
        instance_masks: List[np.ndarray],
        point_map: Optional[np.ndarray] = None,
        class_names: Optional[List[str]] = None,
    ) -> List[ObjectInferenceResult]:
        """Run inference on RGB image with instance masks.
        
        Args:
            rgb: (H, W, 3) RGB image in [0, 255] uint8 or [0, 1] float.
            instance_masks: List of (H, W) boolean masks, one per object.
            point_map: Optional (H, W, 3) point map (depth in camera frame).
            class_names: Optional list of class names per mask.
        
        Returns:
            List of ObjectInferenceResult for each detected object.
        """
        if self.use_fallback:
            return self._infer_fallback(rgb, instance_masks, point_map, class_names)
        return self._infer_real(rgb, instance_masks, point_map, class_names)
    
    def _infer_real(
        self,
        rgb: np.ndarray,
        instance_masks: List[np.ndarray],
        point_map: Optional[np.ndarray],
        class_names: Optional[List[str]],
    ) -> List[ObjectInferenceResult]:
        """Run real SAM3D-Objects inference."""
        if self._model is None:
            return self._infer_fallback(rgb, instance_masks, point_map, class_names)
        
        # Normalize image if needed
        if rgb.dtype == np.uint8:
            rgb_float = rgb.astype(np.float32) / 255.0
        else:
            rgb_float = rgb.astype(np.float32)
        
        results = []
        
        for i, mask in enumerate(instance_masks):
            # Prepare inputs for SAM3D-Objects
            # This interface depends on actual SAM3D-Objects API
            try:
                with torch.no_grad():
                    # Convert to tensor
                    rgb_t = torch.from_numpy(rgb_float).permute(2, 0, 1).unsqueeze(0)
                    mask_t = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
                    
                    if self.device == "cuda":
                        rgb_t = rgb_t.cuda()
                        mask_t = mask_t.cuda()
                    
                    # Run inference (API may vary)
                    output = self._model(rgb_t, mask_t)
                    
                    # Extract latents and geometry
                    shape_latent = output["shape_latent"].cpu().numpy().flatten()
                    appearance_latent = output["appearance_latent"].cpu().numpy().flatten()
                    position = output["position"].cpu().numpy().flatten()
                    rotation = output.get("rotation", torch.tensor([0, 0, 0, 1])).cpu().numpy().flatten()
                    scale = float(output.get("scale", torch.tensor(1.0)).cpu().item())
                    
                    # Get geometry representation
                    if "gaussians" in output:
                        geometry = {
                            "type": "gaussians",
                            "means": output["gaussians"]["means"].cpu().numpy(),
                            "scales": output["gaussians"]["scales"].cpu().numpy(),
                            "rotations": output["gaussians"]["rotations"].cpu().numpy(),
                            "opacities": output["gaussians"]["opacities"].cpu().numpy(),
                            "colors": output["gaussians"]["colors"].cpu().numpy(),
                        }
                    elif "mesh" in output:
                        geometry = {
                            "type": "mesh",
                            "vertices": output["mesh"]["vertices"].cpu().numpy(),
                            "faces": output["mesh"]["faces"].cpu().numpy(),
                        }
                    else:
                        geometry = {"type": "latent_only"}
                    
                    class_name = class_names[i] if class_names and i < len(class_names) else None
                    obj_id = f"obj_{i}" if class_name is None else f"{class_name}_{i}"
                    
                    results.append(ObjectInferenceResult(
                        object_id=obj_id,
                        shape_latent=shape_latent,
                        appearance_latent=appearance_latent,
                        position=position,
                        rotation=rotation,
                        scale=scale,
                        geometry=geometry,
                        confidence=float(output.get("confidence", 0.9)),
                    ))
            except Exception as e:
                logger.warning(f"Real inference failed for object {i}: {e}. Using fallback.")
                results.append(self._create_fallback_result(rgb, mask, i, class_names))
        
        return results
    
    def _infer_fallback(
        self,
        rgb: np.ndarray,
        instance_masks: List[np.ndarray],
        point_map: Optional[np.ndarray],
        class_names: Optional[List[str]],
    ) -> List[ObjectInferenceResult]:
        """Generate fallback results for testing."""
        results = []
        for i, mask in enumerate(instance_masks):
            results.append(self._create_fallback_result(rgb, mask, i, class_names, point_map))
        return results
    
    def _create_fallback_result(
        self,
        rgb: np.ndarray,
        mask: np.ndarray,
        idx: int,
        class_names: Optional[List[str]],
        point_map: Optional[np.ndarray] = None,
    ) -> ObjectInferenceResult:
        """Create a single fallback result."""
        H, W = rgb.shape[:2]
        
        # Compute mask centroid
        ys, xs = np.where(mask)
        if len(ys) == 0:
            center_x, center_y, mask_size = 0.5, 0.5, 0.1
        else:
            center_x = xs.mean() / W
            center_y = ys.mean() / H
            mask_size = np.sqrt(len(ys)) / max(H, W)
        
        # Estimate depth
        if point_map is not None and len(ys) > 0:
            depth = float(point_map[mask, 2].mean())
        else:
            depth = 3.0
        
        # Generate deterministic latents based on position
        rng = np.random.RandomState(hash(f"obj_{idx}_{center_x:.3f}") % (2**31))
        shape_latent = rng.randn(self.LATENT_DIM).astype(np.float32) * 0.1
        appearance_latent = rng.randn(self.LATENT_DIM).astype(np.float32) * 0.1
        
        # Position in camera frame (approximate)
        position = np.array([
            (center_x - 0.5) * depth * 1.5,
            (center_y - 0.5) * depth * 1.5,
            depth,
        ], dtype=np.float32)
        
        # Identity rotation
        rotation = np.array([0, 0, 0, 1], dtype=np.float32)
        
        # Scale based on mask size
        scale = float(mask_size * depth)
        
        # Synthetic gaussian geometry
        num_gaussians = 64
        geometry = {
            "type": "gaussians",
            "means": rng.randn(num_gaussians, 3).astype(np.float32) * scale * 0.3,
            "scales": np.abs(rng.randn(num_gaussians, 3).astype(np.float32) * 0.1) + 0.01,
            "rotations": np.tile([0, 0, 0, 1], (num_gaussians, 1)).astype(np.float32),
            "opacities": np.ones(num_gaussians, dtype=np.float32) * 0.8,
            "colors": rng.rand(num_gaussians, 3).astype(np.float32) * 0.5 + 0.25,
        }
        
        class_name = class_names[idx] if class_names and idx < len(class_names) else None
        obj_id = f"obj_{idx}" if class_name is None else f"{class_name}_{idx}"
        
        return ObjectInferenceResult(
            object_id=obj_id,
            shape_latent=shape_latent,
            appearance_latent=appearance_latent,
            position=position,
            rotation=rotation,
            scale=scale,
            geometry=geometry,
            confidence=0.85,  # Lower confidence for fallback
        )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-weights", action="store_true", help="Download model weights")
    args = parser.parse_args()
    
    if args.download_weights:
        print("To download SAM3D-Objects weights:")
        print("1. Visit https://github.com/facebookresearch/sam-3d-objects")
        print("2. Follow the instructions in the README")
        print(f"3. Place weights at: {SAM3DObjectsInference.DEFAULT_WEIGHTS_PATH}")
    else:
        # Quick test
        wrapper = SAM3DObjectsInference(use_fallback=True)
        print(f"SAM3D-Objects wrapper initialized (real={wrapper.is_real})")
        
        # Test inference
        test_rgb = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        test_mask = np.zeros((256, 256), dtype=bool)
        test_mask[100:150, 100:150] = True
        
        results = wrapper.infer(test_rgb, [test_mask])
        print(f"Inference returned {len(results)} results")
        if results:
            print(f"  Shape latent dim: {results[0].shape_latent.shape}")
            print(f"  Position: {results[0].position}")
