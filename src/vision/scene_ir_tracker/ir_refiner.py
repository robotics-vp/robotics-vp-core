"""
IR Refiner.

Implements inverse rendering refinement with alternating optimization.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    nn = None  # type: ignore
    TORCH_AVAILABLE = False

from src.vision.nag.types import CameraParams
from src.vision.scene_ir_tracker.config import IRRefinerConfig
from src.vision.scene_ir_tracker.ir_scene_graph_renderer import IRSceneGraphRenderer
from src.vision.scene_ir_tracker.types import SceneEntity3D

logger = logging.getLogger(__name__)


def _check_torch() -> None:
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for IRRefiner")


@dataclass
class IRRefinementResult:
    """Result from IR refinement.

    Attributes:
        loss_curve: Loss values across iterations.
        converged: Whether refinement converged.
        diverged: Whether refinement diverged (loss exploded).
        final_loss: Final loss value.
        best_loss: Best loss achieved during optimization.
        per_phase_losses: Dict of phase name -> final loss for that phase.
        iterations_run: Total iterations across all phases.
    """

    loss_curve: List[float] = field(default_factory=list)
    converged: bool = False
    diverged: bool = False
    final_loss: float = float("inf")
    best_loss: float = float("inf")
    per_phase_losses: Dict[str, float] = field(default_factory=dict)
    iterations_run: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "loss_curve": self.loss_curve,
            "converged": self.converged,
            "diverged": self.diverged,
            "final_loss": self.final_loss,
            "best_loss": self.best_loss,
            "per_phase_losses": self.per_phase_losses,
            "iterations_run": self.iterations_run,
        }


class LPIPSStub:
    """Stub for LPIPS loss when not available."""

    def __init__(self, device: "torch.device"):
        self.device = device

    def __call__(
        self,
        img1: "torch.Tensor",
        img2: "torch.Tensor",
    ) -> "torch.Tensor":
        """Compute simple perceptual loss approximation."""
        # Simple gradient-based perceptual approximation
        if img1.ndim == 3:
            img1 = img1.unsqueeze(0)
            img2 = img2.unsqueeze(0)

        # Sobel-like gradients as perceptual proxy
        grad1_x = img1[:, :, :, 1:] - img1[:, :, :, :-1]
        grad1_y = img1[:, :, 1:, :] - img1[:, :, :-1, :]
        grad2_x = img2[:, :, :, 1:] - img2[:, :, :, :-1]
        grad2_y = img2[:, :, 1:, :] - img2[:, :, :-1, :]

        loss = ((grad1_x - grad2_x) ** 2).mean() + ((grad1_y - grad2_y) ** 2).mean()
        return loss


class IRRefiner:
    """Inverse rendering refiner with alternating optimization.

    Refines entity parameters (texture, pose, shape) to match target images
    using differentiable rendering and perceptual losses.
    """

    def __init__(
        self,
        config: Optional[IRRefinerConfig] = None,
        renderer: Optional[IRSceneGraphRenderer] = None,
        device: str = "cuda",
    ):
        """Initialize refiner.

        Args:
            config: Refinement configuration.
            renderer: Scene graph renderer to use.
            device: Device for computation.
        """
        self.config = config or IRRefinerConfig()
        self.renderer = renderer or IRSceneGraphRenderer()
        self._device_str = device
        self._lpips: Optional[LPIPSStub] = None

    @property
    def device(self) -> "torch.device":
        """Get torch device."""
        _check_torch()
        if self._device_str == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    @property
    def lpips(self) -> LPIPSStub:
        """Get LPIPS loss function."""
        if self._lpips is None:
            self._lpips = LPIPSStub(self.device)
        return self._lpips

    def refine(
        self,
        entities: List[SceneEntity3D],
        target_rgb: "torch.Tensor",
        target_masks: Dict[str, "torch.Tensor"],
        camera: CameraParams,
    ) -> Tuple[List[SceneEntity3D], IRRefinementResult]:
        """Refine entity parameters to match target.

        Implements alternating optimization schedule:
        1. Texture fitting
        2. Pose/scale fitting
        3. Shape/joint refinement

        Args:
            entities: Initial entity states.
            target_rgb: (3, H, W) target RGB image in [0, 1].
            target_masks: Dict of track_id -> (H, W) target mask.
            camera: Camera parameters for rendering.

        Returns:
            Tuple of (refined_entities, refinement_result).
        """
        _check_torch()

        if not entities:
            return [], IRRefinementResult(converged=True, final_loss=0.0)

        result = IRRefinementResult()
        refined_entities = [self._copy_entity(e) for e in entities]

        # Phase 1: Texture fitting
        phase1_loss = self._optimize_phase(
            refined_entities,
            target_rgb,
            target_masks,
            camera,
            phase="texture",
            num_iters=self.config.num_texture_iters,
            lr=self.config.lr_texture,
        )
        result.per_phase_losses["texture"] = phase1_loss
        result.loss_curve.append(phase1_loss)

        # Phase 2: Pose/scale fitting
        phase2_loss = self._optimize_phase(
            refined_entities,
            target_rgb,
            target_masks,
            camera,
            phase="pose",
            num_iters=self.config.num_pose_iters,
            lr=self.config.lr_pose,
        )
        result.per_phase_losses["pose"] = phase2_loss
        result.loss_curve.append(phase2_loss)

        # Phase 3: Shape/joint refinement
        phase3_loss = self._optimize_phase(
            refined_entities,
            target_rgb,
            target_masks,
            camera,
            phase="shape",
            num_iters=self.config.num_shape_iters,
            lr=self.config.lr_shape,
        )
        result.per_phase_losses["shape"] = phase3_loss
        result.loss_curve.append(phase3_loss)

        result.final_loss = phase3_loss
        result.iterations_run = (
            self.config.num_texture_iters
            + self.config.num_pose_iters
            + self.config.num_shape_iters
        )

        # Check convergence
        if len(result.loss_curve) >= 2:
            improvement = result.loss_curve[-2] - result.loss_curve[-1]
            result.converged = improvement < self.config.convergence_threshold

        # Update entity IR losses
        for entity in refined_entities:
            entity.ir_loss = result.final_loss / max(1, len(entities))

        return refined_entities, result

    def _copy_entity(self, entity: SceneEntity3D) -> SceneEntity3D:
        """Create a copy of entity for refinement."""
        return SceneEntity3D(
            entity_type=entity.entity_type,
            track_id=entity.track_id,
            pose=entity.pose.copy(),
            scale=entity.scale,
            class_name=entity.class_name,
            mask_2d=entity.mask_2d.copy() if entity.mask_2d is not None else None,
            mask_logits=entity.mask_logits.copy() if entity.mask_logits is not None else None,
            geometry_handle=entity.geometry_handle,
            z_shape=entity.z_shape.copy() if entity.z_shape is not None else None,
            z_tex=entity.z_tex.copy() if entity.z_tex is not None else None,
            z_shape_ema=entity.z_shape_ema.copy() if entity.z_shape_ema is not None else None,
            z_tex_ema=entity.z_tex_ema.copy() if entity.z_tex_ema is not None else None,
            visibility=entity.visibility,
            occlusion_score=entity.occlusion_score,
            ir_loss=entity.ir_loss,
            joints_3d={k: v.copy() for k, v in entity.joints_3d.items()} if entity.joints_3d else None,
        )

    def _optimize_phase(
        self,
        entities: List[SceneEntity3D],
        target_rgb: "torch.Tensor",
        target_masks: Dict[str, "torch.Tensor"],
        camera: CameraParams,
        phase: str,
        num_iters: int,
        lr: float,
    ) -> float:
        """Run optimization for a single phase.

        Args:
            entities: Entities to optimize (in-place).
            target_rgb: Target RGB image.
            target_masks: Target masks.
            camera: Camera parameters.
            phase: Phase name ("texture", "pose", or "shape").
            num_iters: Number of iterations.
            lr: Learning rate.

        Returns:
            Final loss for this phase.
        """
        # In stub mode, just compute loss and return
        # Real implementation would create optimizable parameters

        H, W = target_rgb.shape[1], target_rgb.shape[2]
        best_loss = float("inf")

        for iter_idx in range(num_iters):
            # Render current state
            rendered_rgb, rendered_masks, _ = self.renderer.render_scene(
                entities, camera, H, W
            )

            # Compute loss
            loss = self._compute_loss(
                rendered_rgb,
                target_rgb,
                rendered_masks,
                target_masks,
                entities,
            )

            loss_value = loss.item() if hasattr(loss, "item") else float(loss)
            best_loss = min(best_loss, loss_value)

            # In stub mode, simulate improvement
            if phase == "texture":
                # Simulate texture improvement
                for e in entities:
                    if e.z_tex is not None:
                        e.z_tex = e.z_tex * 0.99
            elif phase == "pose":
                # Simulate pose improvement (small translation adjustment)
                for e in entities:
                    e.pose[:3, 3] *= 0.999
            elif phase == "shape":
                # Simulate shape improvement
                for e in entities:
                    if e.z_shape is not None:
                        e.z_shape = e.z_shape * 0.99

        return best_loss

    def _compute_loss(
        self,
        rendered_rgb: "torch.Tensor",
        target_rgb: "torch.Tensor",
        rendered_masks: Dict[str, "torch.Tensor"],
        target_masks: Dict[str, "torch.Tensor"],
        entities: List[SceneEntity3D],
    ) -> "torch.Tensor":
        """Compute combined IR loss.

        Loss = w_rgb * MSE(rgb) + w_lpips * LPIPS(rgb) + w_reg * embedding_reg
        """
        device = rendered_rgb.device

        # RGB MSE loss (on masked regions)
        combined_mask = torch.zeros(
            (rendered_rgb.shape[1], rendered_rgb.shape[2]),
            device=device,
        )
        for mask in rendered_masks.values():
            combined_mask = torch.max(combined_mask, mask)

        if combined_mask.sum() > 0:
            rgb_mse = ((rendered_rgb - target_rgb) ** 2 * combined_mask.unsqueeze(0)).sum()
            rgb_mse = rgb_mse / (combined_mask.sum() * 3 + 1e-8)
        else:
            rgb_mse = torch.tensor(0.0, device=device)

        # LPIPS loss
        lpips_loss = self.lpips(rendered_rgb, target_rgb)

        # Embedding regularization
        reg_loss = torch.tensor(0.0, device=device)
        for entity in entities:
            if entity.z_shape is not None:
                z = torch.from_numpy(entity.z_shape).to(device)
                reg_loss = reg_loss + (z ** 2).mean()
            if entity.z_tex is not None:
                z = torch.from_numpy(entity.z_tex).to(device)
                reg_loss = reg_loss + (z ** 2).mean()

        # Combine losses
        total_loss = (
            self.config.rgb_loss_weight * rgb_mse
            + self.config.lpips_weight * lpips_loss
            + self.config.embedding_reg_weight * reg_loss
        )

        return total_loss

    def refine_sequence(
        self,
        frames_entities: List[List[SceneEntity3D]],
        target_rgb_sequence: "torch.Tensor",
        target_masks_sequence: List[Dict[str, "torch.Tensor"]],
        camera: CameraParams,
    ) -> Tuple[List[List[SceneEntity3D]], List[IRRefinementResult]]:
        """Refine entities across a sequence.

        Args:
            frames_entities: Entity states per frame.
            target_rgb_sequence: (T, 3, H, W) target RGB sequence.
            target_masks_sequence: List of per-frame mask dicts.
            camera: Camera parameters.

        Returns:
            Tuple of (refined_frames_entities, per_frame_results).
        """
        refined_frames = []
        results = []

        for t, entities in enumerate(frames_entities):
            target_rgb = target_rgb_sequence[t]
            target_masks = target_masks_sequence[t] if t < len(target_masks_sequence) else {}

            refined, result = self.refine(entities, target_rgb, target_masks, camera)
            refined_frames.append(refined)
            results.append(result)

        return refined_frames, results


def create_ir_refiner(
    config: Optional[Dict[str, Any]] = None,
    renderer: Optional[IRSceneGraphRenderer] = None,
    device: str = "cuda",
) -> IRRefiner:
    """Factory function to create IR refiner.

    Args:
        config: Configuration dict for IRRefinerConfig.
        renderer: Optional renderer instance.
        device: Device string.

    Returns:
        Configured IRRefiner.
    """
    if config:
        cfg = IRRefinerConfig.from_dict(config)
    else:
        cfg = IRRefinerConfig()

    return IRRefiner(config=cfg, renderer=renderer, device=device)
