"""
SAM3D-Objects Adapter.

Provides an adapter for SAM3D-Objects model for object reconstruction.
Uses stub implementation when actual model is not available.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ObjectPrediction:
    """Prediction from SAM3D-Objects for a single object.

    Attributes:
        object_id: Identifier for this object.
        shape_latent: Latent code for shape.
        appearance_latent: Latent code for appearance/texture.
        layout: Object pose and scale dict.
        geometry: Geometry representation (mesh vertices/faces or gaussian params).
        confidence: Model confidence for this prediction.
    """

    object_id: str
    shape_latent: np.ndarray
    appearance_latent: np.ndarray
    layout: Dict[str, Any]
    geometry: Dict[str, Any]
    confidence: float = 1.0

    def __post_init__(self) -> None:
        self.shape_latent = np.asarray(self.shape_latent, dtype=np.float32)
        self.appearance_latent = np.asarray(self.appearance_latent, dtype=np.float32)
        self.confidence = float(self.confidence)


@dataclass
class SAM3DObjectsConfig:
    """Configuration for SAM3D-Objects adapter.

    Attributes:
        model_path: Path to SAM3D-Objects checkpoint.
        use_point_map: Whether to use point map as input.
        latent_dim: Dimension of shape/appearance latents.
        output_gaussians: Whether to output gaussian splat representation.
        device: Device for model inference.
    """

    model_path: Optional[str] = None
    use_point_map: bool = False
    latent_dim: int = 256
    output_gaussians: bool = True
    device: str = "cuda"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_path": self.model_path,
            "use_point_map": self.use_point_map,
            "latent_dim": self.latent_dim,
            "output_gaussians": self.output_gaussians,
            "device": self.device,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SAM3DObjectsConfig":
        return cls(
            model_path=data.get("model_path"),
            use_point_map=data.get("use_point_map", False),
            latent_dim=data.get("latent_dim", 256),
            output_gaussians=data.get("output_gaussians", True),
            device=data.get("device", "cuda"),
        )


class SAM3DObjectsAdapter:
    """Adapter for SAM3D-Objects model.

    Provides a clean interface for running SAM3D-Objects inference.
    Falls back to stub implementation if model is not available.
    """

    def __init__(
        self,
        config: Optional[SAM3DObjectsConfig] = None,
        use_stub: bool = True,
    ):
        """Initialize adapter.

        Args:
            config: Configuration for the adapter.
            use_stub: If True, use stub implementation instead of real model.
        """
        self.config = config or SAM3DObjectsConfig()
        self.use_stub = use_stub
        self._model = None

        if not use_stub:
            self._load_model()

    def _load_model(self) -> None:
        """Load SAM3D-Objects model via third_party wrapper."""
        try:
            from third_party.sam3d_objects_wrapper import SAM3DObjectsInference
            
            self._wrapper = SAM3DObjectsInference(
                weights_path=self.config.model_path,
                device=self.config.device,
                use_fallback=False,
            )
            
            if self._wrapper.is_real:
                logger.info("SAM3D-Objects loaded via third_party wrapper")
                self.use_stub = False
            else:
                logger.info("SAM3D-Objects wrapper using fallback mode")
                self.use_stub = True
        except ImportError as e:
            logger.warning(f"Failed to import third_party wrapper: {e}. Using stub.")
            self.use_stub = True
        except Exception as e:
            logger.warning(f"Failed to load SAM3D-Objects: {e}. Using stub.")
            self.use_stub = True

    def infer(
        self,
        rgb: np.ndarray,
        instance_masks: List[np.ndarray],
        point_map: Optional[np.ndarray] = None,
        class_names: Optional[List[str]] = None,
    ) -> List[ObjectPrediction]:
        """Run inference on image with instance masks.

        Args:
            rgb: (H, W, 3) RGB image in [0, 255] uint8 or [0, 1] float.
            instance_masks: List of (H, W) boolean masks, one per object.
            point_map: Optional (H, W, 3) point map (depth in camera frame).
            class_names: Optional list of class names per mask.

        Returns:
            List of ObjectPrediction for each detected object.
        """
        if self.use_stub:
            return self._infer_stub(rgb, instance_masks, point_map, class_names)
        return self._infer_model(rgb, instance_masks, point_map, class_names)

    def _infer_model(
        self,
        rgb: np.ndarray,
        instance_masks: List[np.ndarray],
        point_map: Optional[np.ndarray],
        class_names: Optional[List[str]],
    ) -> List[ObjectPrediction]:
        """Run actual model inference via third_party wrapper."""
        if not hasattr(self, '_wrapper') or self._wrapper is None:
            return self._infer_stub(rgb, instance_masks, point_map, class_names)
        
        results = self._wrapper.infer(rgb, instance_masks, point_map, class_names)
        
        predictions = []
        for r in results:
            predictions.append(ObjectPrediction(
                object_id=r.object_id,
                shape_latent=r.shape_latent,
                appearance_latent=r.appearance_latent,
                layout={
                    "position": r.position.tolist(),
                    "rotation": r.rotation.tolist(),
                    "scale": r.scale,
                },
                geometry=r.geometry,
                confidence=r.confidence,
            ))
        return predictions

    def _infer_stub(
        self,
        rgb: np.ndarray,
        instance_masks: List[np.ndarray],
        point_map: Optional[np.ndarray],
        class_names: Optional[List[str]],
    ) -> List[ObjectPrediction]:
        """Generate stub predictions for testing.

        Creates synthetic predictions with reasonable defaults.
        """
        predictions = []
        H, W = rgb.shape[:2]

        for i, mask in enumerate(instance_masks):
            # Compute mask center and size
            ys, xs = np.where(mask)
            if len(ys) == 0:
                continue

            center_x = xs.mean() / W
            center_y = ys.mean() / H
            size = np.sqrt(len(ys)) / max(H, W)

            # Estimate depth from point map or use default
            if point_map is not None:
                masked_depths = point_map[mask, 2]
                depth = masked_depths.mean() if len(masked_depths) > 0 else 3.0
            else:
                depth = 3.0  # Default depth in meters

            # Generate synthetic latents (deterministic based on mask)
            rng = np.random.RandomState(hash(f"obj_{i}_{center_x:.3f}_{center_y:.3f}") % (2**31))
            shape_latent = rng.randn(self.config.latent_dim).astype(np.float32) * 0.1
            appearance_latent = rng.randn(self.config.latent_dim).astype(np.float32) * 0.1

            # Layout (pose estimation)
            layout = {
                "position": [
                    (center_x - 0.5) * depth * 2,  # x in world
                    (center_y - 0.5) * depth * 2,  # y in world
                    depth,  # z depth
                ],
                "rotation": [0.0, 0.0, 0.0, 1.0],  # quaternion (identity)
                "scale": float(size * depth),
            }

            # Generate synthetic geometry
            if self.config.output_gaussians:
                # Gaussian splat representation
                num_gaussians = 100
                geometry = {
                    "type": "gaussians",
                    "means": rng.randn(num_gaussians, 3).astype(np.float32) * layout["scale"],
                    "scales": np.abs(rng.randn(num_gaussians, 3).astype(np.float32) * 0.1),
                    "rotations": np.tile([0, 0, 0, 1], (num_gaussians, 1)).astype(np.float32),
                    "opacities": np.ones(num_gaussians, dtype=np.float32),
                    "colors": rng.rand(num_gaussians, 3).astype(np.float32),
                }
            else:
                # Simple mesh representation
                geometry = {
                    "type": "mesh",
                    "vertices": rng.randn(100, 3).astype(np.float32) * layout["scale"],
                    "faces": np.arange(99).reshape(-1, 3).astype(np.int32),
                }

            class_name = class_names[i] if class_names and i < len(class_names) else None
            obj_id = f"obj_{i}" if class_name is None else f"{class_name}_{i}"

            predictions.append(
                ObjectPrediction(
                    object_id=obj_id,
                    shape_latent=shape_latent,
                    appearance_latent=appearance_latent,
                    layout=layout,
                    geometry=geometry,
                    confidence=0.9,
                )
            )

        return predictions


def create_sam3d_objects_adapter(
    config: Optional[Dict[str, Any]] = None,
    use_stub: bool = True,
) -> SAM3DObjectsAdapter:
    """Factory function to create SAM3D-Objects adapter.

    Args:
        config: Configuration dict.
        use_stub: Whether to use stub implementation.

    Returns:
        Configured SAM3DObjectsAdapter.
    """
    cfg = SAM3DObjectsConfig.from_dict(config or {})
    return SAM3DObjectsAdapter(config=cfg, use_stub=use_stub)
