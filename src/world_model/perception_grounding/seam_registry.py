"""Neural seam registry for the Perception / Grounding WM.

Manages loading, caching, and checkpoint persistence of neural seam modules.
Each seam follows the ``disabled|auto|required`` promotion posture and can
be loaded from checkpoints or initialized fresh.

Seam types
----------
- ``EvidenceFusionSeam``: fuses heterogeneous provider evidence
- ``SAMCalibrationSeam``: calibrates SAM mask confidence
- ``VisionBackboneProjectionSeam``: projects DINOv2/SigLIP features
- ``DepthMetricCalibrationSeam``: calibrates depth to metric scale
- ``VJEPATemporalAlignmentSeam``: aligns V-JEPA predictions to WM tokens

Registry usage
--------------
The registry is the single source of truth for seam instances within a
planning window.  Resolvers in ``promotion.py`` check benchmark eligibility;
this registry handles the actual ``nn.Module`` lifecycle.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Type, Union

import torch
import torch.nn as nn

from .neural_seams import (
    DepthMetricCalibrationSeam,
    EvidenceFusionSeam,
    SAMCalibrationSeam,
    SceneGraphTransformerSeam,
    VisionBackboneProjectionSeam,
    VJEPATemporalAlignmentSeam,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Seam type registry
# ---------------------------------------------------------------------------

SEAM_TYPES: Dict[str, Type[nn.Module]] = {
    "evidence_fusion": EvidenceFusionSeam,
    "sam_calibration": SAMCalibrationSeam,
    "vision_backbone_projection": VisionBackboneProjectionSeam,
    "depth_metric_calibration": DepthMetricCalibrationSeam,
    "vjepa_temporal_alignment": VJEPATemporalAlignmentSeam,
    "scene_graph_transformer": SceneGraphTransformerSeam,
}

# Default hyperparameters per seam type
SEAM_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "evidence_fusion": {
        "d_model": 32,
        "n_heads": 2,
        "d_ff": 64,
        "dropout": 0.0,
    },
    "sam_calibration": {
        "d_mask": 256,
        "d_model": 128,
        "n_heads": 4,
        "d_ff": 256,
        "dropout": 0.1,
    },
    "vision_backbone_projection": {
        "d_backbone": 1024,
        "d_hidden": 512,
        "d_out": 128,
        "dropout": 0.1,
    },
    "depth_metric_calibration": {
        "d_depth": 1,
        "d_hidden": 128,
        "intrinsic_dim": 4,
        "dropout": 0.1,
    },
    "vjepa_temporal_alignment": {
        "d_vjepa": 1024,
        "d_wm_token": 128,
        "d_model": 256,
        "d_out": 128,
        "n_heads": 8,
        "d_ff": 512,
        "n_temporal_steps": 4,
        "dropout": 0.1,
    },
    "scene_graph_transformer": {
        "d_token": 128,
        "d_model": 128,
        "d_out": 128,
        "d_edge": 64,
        "n_heads": 4,
        "d_ff": 256,
        "n_layers": 2,
        "dropout": 0.1,
    },
}


# ---------------------------------------------------------------------------
# Seam instance descriptor
# ---------------------------------------------------------------------------


@dataclass
class SeamDescriptor:
    """Describes a loaded or loadable neural seam."""

    seam_type: str
    seam_id: str
    posture: str = "disabled"
    promotion_stage: str = "not_loaded"
    checkpoint_path: Optional[str] = None
    hyperparams: Dict[str, Any] = field(default_factory=dict)
    param_count: int = 0
    loaded: bool = False
    device: str = "cpu"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seam_type": self.seam_type,
            "seam_id": self.seam_id,
            "posture": self.posture,
            "promotion_stage": self.promotion_stage,
            "checkpoint_path": self.checkpoint_path,
            "hyperparams": dict(self.hyperparams),
            "param_count": self.param_count,
            "loaded": self.loaded,
            "device": self.device,
            "metadata": dict(self.metadata),
        }


# ---------------------------------------------------------------------------
# Seam registry
# ---------------------------------------------------------------------------


class PerceptionSeamRegistry:
    """Registry for neural seams in the Perception / Grounding WM.

    Manages:
    - Seam instance creation with type-safe hyperparameters
    - Checkpoint loading and saving
    - Device placement (CPU/GPU)
    - Posture tracking (disabled/auto/required)

    Thread-safety: NOT thread-safe.  Use one registry per planning loop.
    """

    def __init__(
        self,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        default_device: str = "cpu",
    ) -> None:
        """Initialize the seam registry.

        Args:
            checkpoint_dir: Directory for seam checkpoints.  If None,
                checkpoints are not persisted.
            default_device: Default device for seam modules.
        """
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.default_device = default_device
        self._seams: Dict[str, nn.Module] = {}
        self._descriptors: Dict[str, SeamDescriptor] = {}

    def register_seam(
        self,
        seam_type: str,
        seam_id: str,
        *,
        posture: str = "disabled",
        hyperparams: Optional[Dict[str, Any]] = None,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
    ) -> SeamDescriptor:
        """Register a seam type without loading the module.

        Args:
            seam_type: One of the SEAM_TYPES keys.
            seam_id: Unique identifier for this seam instance.
            posture: Promotion posture (disabled/auto/required).
            hyperparams: Override default hyperparameters.
            checkpoint_path: Path to checkpoint file.
            device: Device for this seam (overrides default).

        Returns:
            SeamDescriptor for the registered seam.
        """
        if seam_type not in SEAM_TYPES:
            raise ValueError(
                f"Unknown seam type: {seam_type}. "
                f"Available: {list(SEAM_TYPES.keys())}"
            )

        merged_params = dict(SEAM_DEFAULTS.get(seam_type, {}))
        if hyperparams:
            merged_params.update(hyperparams)

        descriptor = SeamDescriptor(
            seam_type=seam_type,
            seam_id=seam_id,
            posture=posture,
            promotion_stage="registered",
            checkpoint_path=checkpoint_path,
            hyperparams=merged_params,
            device=device or self.default_device,
        )
        self._descriptors[seam_id] = descriptor
        return descriptor

    def load_seam(
        self,
        seam_id: str,
        *,
        force_reload: bool = False,
    ) -> nn.Module:
        """Load a seam module, optionally from checkpoint.

        Args:
            seam_id: Identifier of a registered seam.
            force_reload: If True, reload even if already loaded.

        Returns:
            The loaded nn.Module.

        Raises:
            KeyError: If seam_id is not registered.
            RuntimeError: If loading fails.
        """
        if seam_id not in self._descriptors:
            raise KeyError(f"Seam not registered: {seam_id}")

        if seam_id in self._seams and not force_reload:
            return self._seams[seam_id]

        descriptor = self._descriptors[seam_id]
        seam_cls = SEAM_TYPES[descriptor.seam_type]

        # Create module with hyperparams
        try:
            seam = seam_cls(**descriptor.hyperparams)
        except TypeError as e:
            raise RuntimeError(
                f"Failed to create seam {seam_id} with params "
                f"{descriptor.hyperparams}: {e}"
            ) from e

        # Load checkpoint if available
        ckpt_path = self._resolve_checkpoint_path(descriptor)
        if ckpt_path and ckpt_path.exists():
            try:
                state_dict = torch.load(ckpt_path, map_location="cpu")
                seam.load_state_dict(state_dict)
                logger.info(f"Loaded seam {seam_id} from {ckpt_path}")
            except Exception as e:
                logger.warning(
                    f"Failed to load checkpoint for {seam_id} from "
                    f"{ckpt_path}: {e}. Using fresh initialization."
                )

        # Move to device
        device = torch.device(descriptor.device)
        seam = seam.to(device)

        # Update descriptor
        descriptor.loaded = True
        descriptor.promotion_stage = "loaded"
        descriptor.param_count = sum(
            p.numel() for p in seam.parameters() if p.requires_grad
        )

        self._seams[seam_id] = seam
        return seam

    def get_seam(self, seam_id: str) -> Optional[nn.Module]:
        """Get a loaded seam, or None if not loaded."""
        return self._seams.get(seam_id)

    def get_descriptor(self, seam_id: str) -> Optional[SeamDescriptor]:
        """Get the descriptor for a seam."""
        return self._descriptors.get(seam_id)

    def save_seam(
        self,
        seam_id: str,
        *,
        checkpoint_path: Optional[Union[str, Path]] = None,
    ) -> Optional[Path]:
        """Save a seam's state_dict to checkpoint.

        Args:
            seam_id: Identifier of a loaded seam.
            checkpoint_path: Override the descriptor's checkpoint path.

        Returns:
            Path to saved checkpoint, or None if save failed.
        """
        if seam_id not in self._seams:
            logger.warning(f"Cannot save unloaded seam: {seam_id}")
            return None

        seam = self._seams[seam_id]
        descriptor = self._descriptors[seam_id]

        if checkpoint_path:
            save_path = Path(checkpoint_path)
        else:
            save_path = self._resolve_checkpoint_path(descriptor, create_dir=True)

        if not save_path:
            logger.warning(f"No checkpoint path for seam: {seam_id}")
            return None

        try:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(seam.state_dict(), save_path)
            descriptor.checkpoint_path = str(save_path)
            logger.info(f"Saved seam {seam_id} to {save_path}")
            return save_path
        except Exception as e:
            logger.error(f"Failed to save seam {seam_id}: {e}")
            return None

    def unload_seam(self, seam_id: str) -> None:
        """Unload a seam module to free memory."""
        if seam_id in self._seams:
            del self._seams[seam_id]
            if seam_id in self._descriptors:
                self._descriptors[seam_id].loaded = False
                self._descriptors[seam_id].promotion_stage = "unloaded"

    def list_seams(self) -> Dict[str, SeamDescriptor]:
        """List all registered seams and their descriptors."""
        return dict(self._descriptors)

    def summary(self) -> Dict[str, Any]:
        """Summary of registry state for logging/receipts."""
        total_params = sum(
            d.param_count for d in self._descriptors.values() if d.loaded
        )
        return {
            "registered_count": len(self._descriptors),
            "loaded_count": len(self._seams),
            "total_params": total_params,
            "seams": {
                sid: desc.to_dict() for sid, desc in self._descriptors.items()
            },
        }

    def _resolve_checkpoint_path(
        self,
        descriptor: SeamDescriptor,
        create_dir: bool = False,
    ) -> Optional[Path]:
        """Resolve checkpoint path for a seam."""
        if descriptor.checkpoint_path:
            return Path(descriptor.checkpoint_path)

        if self.checkpoint_dir:
            path = self.checkpoint_dir / f"{descriptor.seam_id}.pt"
            if create_dir:
                path.parent.mkdir(parents=True, exist_ok=True)
            return path

        return None


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def create_default_registry(
    checkpoint_dir: Optional[Union[str, Path]] = None,
    device: str = "cpu",
) -> PerceptionSeamRegistry:
    """Create a registry with all seam types pre-registered.

    This is the standard entry point for perception compilation.
    All seams start in ``disabled`` posture; promotion resolvers
    determine which should be loaded based on benchmark signals.
    """
    registry = PerceptionSeamRegistry(
        checkpoint_dir=checkpoint_dir,
        default_device=device,
    )

    # Register all standard seams
    registry.register_seam(
        "evidence_fusion",
        "evidence_fusion_default",
        posture="disabled",
    )
    registry.register_seam(
        "sam_calibration",
        "sam_calibration_default",
        posture="disabled",
    )
    registry.register_seam(
        "vision_backbone_projection",
        "vision_backbone_projection_default",
        posture="disabled",
    )
    registry.register_seam(
        "depth_metric_calibration",
        "depth_metric_calibration_default",
        posture="disabled",
    )
    registry.register_seam(
        "vjepa_temporal_alignment",
        "vjepa_temporal_alignment_default",
        posture="disabled",
    )
    registry.register_seam(
        "scene_graph_transformer",
        "scene_graph_transformer_default",
        posture="disabled",
    )

    return registry


__all__ = [
    "PerceptionSeamRegistry",
    "SEAM_DEFAULTS",
    "SEAM_TYPES",
    "SeamDescriptor",
    "create_default_registry",
]
