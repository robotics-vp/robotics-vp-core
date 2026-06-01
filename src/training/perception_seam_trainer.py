"""Training loop infrastructure for perception neural seams.

This module provides the training orchestration for perception seams:
- Training loop with gradient accumulation
- Validation and benchmark gate evaluation
- Checkpoint management via PerceptionSeamRegistry
- Metrics logging and receipt emission
- Early stopping and learning rate scheduling

Training regime
---------------
All seams use supervised/contrastive/predictive objectives, NOT RL.
The training loop is standard supervised learning with:
1. Forward pass through seam
2. Loss computation from loss functions module
3. Backward pass and optimizer step
4. Periodic validation and checkpoint saving
5. Benchmark gate evaluation for promotion decisions

Integration with PerceptionSeamRegistry
---------------------------------------
The trainer uses PerceptionSeamRegistry for:
- Loading seam checkpoints at training start
- Saving checkpoints at training milestones
- Tracking promotion stage transitions

Receipt emission
----------------
Training emits receipts for:
- Training step metrics
- Validation results
- Benchmark gate evaluations
- Promotion/demotion decisions
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler
from torch.utils.data import DataLoader

from src.world_model.perception_grounding.seam_registry import (
    PerceptionSeamRegistry,
    SeamDescriptor,
)

from .perception_seam_losses import (
    SeamLossResult,
    annotation_bridge_projection_loss,
    evidence_fusion_loss,
    sam_calibration_loss,
    scene_graph_transformer_loss,
    vision_backbone_projection_loss,
    depth_metric_calibration_loss,
    vjepa_temporal_alignment_loss,
    get_seam_loss_fn,
)
from .perception_seam_data import (
    EvidenceFusionBatch,
    SAMCalibrationBatch,
    DepthCalibrationBatch,
    SceneGraphBatch,
    VJEPATemporalBatch,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------


@dataclass
class SeamTrainingConfig:
    """Configuration for seam training."""

    # Training parameters
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    batch_size: int = 32
    max_epochs: int = 100
    max_steps: Optional[int] = None
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0

    # Validation and checkpointing
    val_check_interval: int = 100
    checkpoint_interval: int = 500
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4

    # Learning rate scheduling
    lr_scheduler: str = "cosine"  # "cosine" or "none"
    warmup_steps: int = 100
    min_lr: float = 1e-6

    # Benchmark gates
    benchmark_gate_interval: int = 1000
    promotion_threshold: float = 0.8
    demotion_threshold: float = 0.5

    # Logging
    log_interval: int = 10
    log_dir: Optional[str] = None

    # Device
    device: str = "cpu"
    mixed_precision: bool = False


# ---------------------------------------------------------------------------
# Training state
# ---------------------------------------------------------------------------


@dataclass
class SeamTrainingState:
    """Mutable state tracked during training."""

    step: int = 0
    epoch: int = 0
    best_val_loss: float = float("inf")
    best_step: int = 0
    patience_counter: int = 0
    total_train_loss: float = 0.0
    total_train_samples: int = 0
    training_start_time: Optional[float] = None
    is_finished: bool = False
    finish_reason: str = ""


# ---------------------------------------------------------------------------
# Training receipts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SeamTrainingStepReceipt:
    """Receipt for a single training step."""

    receipt_id: str
    seam_id: str
    seam_type: str
    step: int
    epoch: int
    train_loss: float
    component_losses: Dict[str, float]
    metrics: Dict[str, float]
    learning_rate: float
    grad_norm: float
    timestamp: str
    version: str = "seam_training_step_receipt_v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "seam_id": self.seam_id,
            "seam_type": self.seam_type,
            "step": self.step,
            "epoch": self.epoch,
            "train_loss": float(self.train_loss),
            "component_losses": {k: float(v) for k, v in self.component_losses.items()},
            "metrics": {k: float(v) for k, v in self.metrics.items()},
            "learning_rate": float(self.learning_rate),
            "grad_norm": float(self.grad_norm),
            "timestamp": self.timestamp,
            "version": self.version,
        }


@dataclass(frozen=True)
class SeamValidationReceipt:
    """Receipt for validation evaluation."""

    receipt_id: str
    seam_id: str
    seam_type: str
    step: int
    val_loss: float
    component_losses: Dict[str, float]
    metrics: Dict[str, float]
    is_best: bool
    timestamp: str
    version: str = "seam_validation_receipt_v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "seam_id": self.seam_id,
            "seam_type": self.seam_type,
            "step": self.step,
            "val_loss": float(self.val_loss),
            "component_losses": {k: float(v) for k, v in self.component_losses.items()},
            "metrics": {k: float(v) for k, v in self.metrics.items()},
            "is_best": bool(self.is_best),
            "timestamp": self.timestamp,
            "version": self.version,
        }


@dataclass(frozen=True)
class BenchmarkGateReceipt:
    """Receipt for benchmark gate evaluation."""

    receipt_id: str
    seam_id: str
    seam_type: str
    step: int
    gate_score: float
    gate_passed: bool
    gate_threshold: float
    previous_stage: str
    new_stage: str
    gate_metrics: Dict[str, float]
    timestamp: str
    version: str = "benchmark_gate_receipt_v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "seam_id": self.seam_id,
            "seam_type": self.seam_type,
            "step": self.step,
            "gate_score": float(self.gate_score),
            "gate_passed": bool(self.gate_passed),
            "gate_threshold": float(self.gate_threshold),
            "previous_stage": self.previous_stage,
            "new_stage": self.new_stage,
            "gate_metrics": {k: float(v) for k, v in self.gate_metrics.items()},
            "timestamp": self.timestamp,
            "version": self.version,
        }


# ---------------------------------------------------------------------------
# Seam Trainer
# ---------------------------------------------------------------------------


class PerceptionSeamTrainer:
    """Training orchestrator for perception neural seams.

    Manages the full training lifecycle:
    1. Seam and optimizer initialization
    2. Training loop with gradient accumulation
    3. Validation and early stopping
    4. Checkpoint saving via registry
    5. Benchmark gate evaluation for promotion

    Example usage::

        registry = PerceptionSeamRegistry(checkpoint_dir="./checkpoints")
        registry.register_seam("evidence_fusion", "fusion_v1", posture="auto")

        trainer = PerceptionSeamTrainer(
            registry=registry,
            seam_id="fusion_v1",
            config=SeamTrainingConfig(learning_rate=1e-4, max_epochs=50),
        )

        trainer.fit(train_loader, val_loader)

        # Check if seam was promoted
        desc = registry.get_descriptor("fusion_v1")
        print(f"Promotion stage: {desc.promotion_stage}")
    """

    def __init__(
        self,
        registry: PerceptionSeamRegistry,
        seam_id: str,
        config: Optional[SeamTrainingConfig] = None,
        *,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[LRScheduler] = None,
        loss_fn: Optional[Callable[..., SeamLossResult]] = None,
        benchmark_evaluator: Optional[
            Callable[[nn.Module, DataLoader], Dict[str, float]]
        ] = None,
    ) -> None:
        """Initialize trainer.

        Args:
            registry: PerceptionSeamRegistry managing the seam.
            seam_id: ID of the seam to train.
            config: Training configuration.
            optimizer: Optional pre-configured optimizer.
            scheduler: Optional pre-configured LR scheduler.
            loss_fn: Optional custom loss function.
            benchmark_evaluator: Optional benchmark evaluator for promotion gates.
        """
        self.registry = registry
        self.seam_id = seam_id
        self.config = config or SeamTrainingConfig()

        # Load seam
        self.seam = registry.load_seam(seam_id)
        self.seam.to(self.config.device)

        # Get seam metadata
        descriptor = registry.get_descriptor(seam_id)
        if descriptor is None:
            raise ValueError(f"Unknown perception seam descriptor: {seam_id}")
        self.descriptor: SeamDescriptor = descriptor
        self.seam_type = self.descriptor.seam_type

        # Initialize optimizer
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = AdamW(
                self.seam.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )

        # Initialize scheduler
        self.scheduler: Optional[LRScheduler]
        if scheduler is not None:
            self.scheduler = scheduler
        elif self.config.lr_scheduler == "cosine":
            # Will be properly initialized when we know total steps
            self.scheduler = None
            self._scheduler_needs_init = True
        else:
            self.scheduler = None
            self._scheduler_needs_init = False

        # Loss function
        if loss_fn is not None:
            self.loss_fn = loss_fn
        else:
            self.loss_fn = get_seam_loss_fn(self.seam_type)

        # Benchmark evaluator
        self.benchmark_evaluator = benchmark_evaluator

        # Training state
        self.state = SeamTrainingState()

        # Receipts
        self.training_receipts: List[SeamTrainingStepReceipt] = []
        self.validation_receipts: List[SeamValidationReceipt] = []
        self.benchmark_receipts: List[BenchmarkGateReceipt] = []

        # Mixed precision
        self.scaler = torch.amp.GradScaler() if self.config.mixed_precision else None

    def _init_scheduler(self, total_steps: int) -> None:
        """Initialize LR scheduler after we know total steps."""
        if self.config.lr_scheduler == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps - self.config.warmup_steps,
                eta_min=self.config.min_lr,
            )
        self._scheduler_needs_init = False

    def _get_current_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]["lr"]

    def _warmup_lr(self) -> None:
        """Apply warmup learning rate."""
        if self.state.step < self.config.warmup_steps:
            warmup_factor = self.state.step / max(1, self.config.warmup_steps)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.config.learning_rate * warmup_factor

    def _compute_loss(
        self,
        batch: Any,
    ) -> SeamLossResult:
        """Compute loss for a batch based on seam type."""
        if self.seam_type == "annotation_bridge_projection":
            return self._compute_annotation_bridge_loss(batch)
        elif self.seam_type == "evidence_fusion":
            return self._compute_evidence_fusion_loss(batch)
        elif self.seam_type == "sam_calibration":
            return self._compute_sam_calibration_loss(batch)
        elif self.seam_type == "depth_metric_calibration":
            return self._compute_depth_calibration_loss(batch)
        elif self.seam_type == "vjepa_temporal_alignment":
            return self._compute_vjepa_temporal_loss(batch)
        elif self.seam_type == "vision_backbone_projection":
            return self._compute_vision_backbone_loss(batch)
        elif self.seam_type == "scene_graph_transformer":
            return self._compute_scene_graph_loss(batch)
        else:
            raise ValueError(f"Unknown seam type: {self.seam_type}")

    def _compute_evidence_fusion_loss(
        self, batch: EvidenceFusionBatch
    ) -> SeamLossResult:
        """Compute evidence fusion loss."""
        # Forward pass: seam takes 12-dim encoded metadata, not raw features
        weights, confidence = self.seam(batch.seam_input_features)

        # Compute loss
        return evidence_fusion_loss(
            predicted_weights=weights,
            predicted_confidence=confidence,
            held_out_provider_idx=batch.held_out_idx,
            held_out_reconstruction_target=batch.held_out_features,
            provider_features=batch.provider_features,
            task_correlation_target=batch.task_correlation_target,
            provider_availability_mask=batch.provider_availability,
        )

    def _compute_sam_calibration_loss(
        self, batch: SAMCalibrationBatch
    ) -> SeamLossResult:
        """Compute SAM calibration loss."""
        # Forward pass
        result = self.seam(
            batch.mask_features,
            batch.raw_confidence,
            batch.mask_valid,
        )

        # Compute loss
        return sam_calibration_loss(
            calibrated_confidence=result["calibrated_confidence"],
            epistemic_uncertainty=result["epistemic_uncertainty"],
            prompt_satisfaction=result["prompt_satisfaction"],
            downstream_mask_quality=batch.downstream_quality,
            provider_disagreement=batch.provider_disagreement,
            segmentation_iou=batch.segmentation_iou,
            mask_valid=batch.mask_valid,
        )

    def _compute_depth_calibration_loss(
        self, batch: DepthCalibrationBatch
    ) -> SeamLossResult:
        """Compute depth calibration loss."""
        # Forward pass
        result = self.seam(batch.relative_depth, batch.camera_intrinsics)

        # Compute loss
        return depth_metric_calibration_loss(
            metric_depth=result["metric_depth"],
            predicted_uncertainty=result["uncertainty"],
            predicted_scale=result["scale"],
            predicted_shift=result["shift"],
            ground_truth_depth=batch.ground_truth_depth,
            depth_valid_mask=batch.depth_valid_mask,
            previous_scale=batch.previous_scale,
            previous_shift=batch.previous_shift,
        )

    def _compute_vjepa_temporal_loss(self, batch: VJEPATemporalBatch) -> SeamLossResult:
        """Compute V-JEPA temporal alignment loss."""
        # Forward pass
        result = self.seam(batch.vjepa_tokens, batch.wm_object_tokens)

        # Compute loss
        return vjepa_temporal_alignment_loss(
            temporal_aligned=result["temporal_aligned"],
            temporal_confidence=result["temporal_confidence"],
            future_object_states=batch.future_object_states,
            object_valid_mask=batch.object_valid_mask,
            temporal_ordering_labels=batch.temporal_ordering_labels,
        )

    def _compute_vision_backbone_loss(self, batch: Any) -> SeamLossResult:
        """Compute vision backbone projection loss.

        Note: VisionBackboneProjectionSeam uses a different batch structure.
        This is a placeholder that assumes batch has the right fields.
        """
        # Forward pass
        projected = self.seam(batch.backbone_features)

        # Compute loss
        return vision_backbone_projection_loss(
            projected_features=projected,
            object_identity_labels=batch.object_identity_labels,
            scene_labels=getattr(batch, "scene_labels", None),
            cross_provider_embeddings=getattr(batch, "cross_provider_embeddings", None),
        )

    def _compute_annotation_bridge_loss(self, batch: SceneGraphBatch) -> SeamLossResult:
        """Compute annotation bridge projection loss.

        Reuses SceneGraphBatch: node_features are the object tokens,
        node_labels are the class targets.  confidence_target and
        affordance_target may be absent (graceful degradation).
        """
        result = self.seam(
            batch.node_features,
            node_mask=batch.node_mask,
        )
        return annotation_bridge_projection_loss(
            class_logits=result["class_logits"],
            confidence=result["confidence"],
            affordance_scores=result["affordance_scores"],
            class_labels=(
                batch.node_labels
                if batch.node_labels is not None
                else torch.zeros(
                    result["class_logits"].shape[:2],
                    dtype=torch.long,
                    device=result["class_logits"].device,
                )
            ),
            confidence_targets=getattr(batch, "node_confidence_target", None),
            affordance_targets=getattr(batch, "affordance_targets", None),
            node_mask=batch.node_mask,
        )

    def _compute_scene_graph_loss(self, batch: SceneGraphBatch) -> SeamLossResult:
        """Compute scene graph transformer loss."""
        # Forward pass
        result = self.seam(
            batch.node_features,
            batch.edge_index,
            batch.edge_type,
            edge_features=batch.edge_features,
            node_mask=batch.node_mask,
        )

        # Compute loss
        return scene_graph_transformer_loss(
            refined_tokens=result["refined_tokens"],
            edge_weights=result["edge_weights"],
            graph_confidence=result["graph_confidence"],
            input_tokens=batch.node_features,
            node_labels=batch.node_labels,
            edge_importance_target=batch.edge_importance,
            node_confidence_target=batch.node_confidence_target,
            node_mask=batch.node_mask,
            edge_mask=batch.edge_mask,
        )

    def _training_step(
        self, batch: Any
    ) -> Tuple[float, Dict[str, float], Dict[str, float]]:
        """Execute a single training step.

        Returns:
            Tuple of (loss_value, component_losses, metrics)
        """
        self.seam.train()

        # Move batch to device
        batch = self._batch_to_device(batch)

        # Forward + loss
        if self.config.mixed_precision:
            with torch.amp.autocast(device_type="cuda"):
                loss_result = self._compute_loss(batch)
        else:
            loss_result = self._compute_loss(batch)

        loss = loss_result.total_loss / self.config.gradient_accumulation_steps

        # Backward
        if self.config.mixed_precision and self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        # Gradient accumulation
        if (self.state.step + 1) % self.config.gradient_accumulation_steps == 0:
            # Gradient clipping
            if self.config.mixed_precision and self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
            _grad_norm = torch.nn.utils.clip_grad_norm_(
                self.seam.parameters(), self.config.max_grad_norm
            )

            # Optimizer step
            if self.config.mixed_precision and self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            self.optimizer.zero_grad()

            # LR scheduling
            if self.state.step < self.config.warmup_steps:
                self._warmup_lr()
            elif self.scheduler is not None:
                self.scheduler.step()
        else:
            _grad_norm = torch.tensor(0.0)

        return (
            loss_result.total_loss.item(),
            {k: v.item() for k, v in loss_result.component_losses.items()},
            loss_result.metrics,
        )

    def _validation_step(
        self, batch: Any
    ) -> Tuple[float, Dict[str, float], Dict[str, float]]:
        """Execute a single validation step."""
        self.seam.eval()
        batch = self._batch_to_device(batch)

        with torch.no_grad():
            loss_result = self._compute_loss(batch)

        return (
            loss_result.total_loss.item(),
            {k: v.item() for k, v in loss_result.component_losses.items()},
            loss_result.metrics,
        )

    def _batch_to_device(self, batch: Any) -> Any:
        """Move batch tensors to device."""
        device = torch.device(self.config.device)

        if hasattr(batch, "_fields"):  # NamedTuple
            return type(batch)(
                *(t.to(device) if isinstance(t, torch.Tensor) else t for t in batch)
            )

        # Dataclass
        for field_name in batch.__dataclass_fields__:
            value = getattr(batch, field_name)
            if isinstance(value, torch.Tensor):
                object.__setattr__(batch, field_name, value.to(device))

        return batch

    def _validate(
        self, val_loader: DataLoader
    ) -> Tuple[float, Dict[str, float], Dict[str, float]]:
        """Run validation loop."""
        total_loss = 0.0
        total_samples = 0
        accumulated_components: Dict[str, float] = {}
        accumulated_metrics: Dict[str, float] = {}

        for batch in val_loader:
            loss, components, metrics = self._validation_step(batch)
            batch_size = self._get_batch_size(batch)

            total_loss += loss * batch_size
            total_samples += batch_size

            for k, v in components.items():
                accumulated_components[k] = (
                    accumulated_components.get(k, 0.0) + v * batch_size
                )
            for k, v in metrics.items():
                accumulated_metrics[k] = (
                    accumulated_metrics.get(k, 0.0) + v * batch_size
                )

        avg_loss = total_loss / max(1, total_samples)
        avg_components = {
            k: v / max(1, total_samples) for k, v in accumulated_components.items()
        }
        avg_metrics = {
            k: v / max(1, total_samples) for k, v in accumulated_metrics.items()
        }

        return avg_loss, avg_components, avg_metrics

    def _get_batch_size(self, batch: Any) -> int:
        """Get batch size from batch object."""
        for field_name in getattr(batch, "__dataclass_fields__", {}):
            value = getattr(batch, field_name)
            if isinstance(value, torch.Tensor) and value.dim() > 0:
                return value.size(0)
        return 1

    def _evaluate_benchmark_gate(
        self,
        val_loader: DataLoader,
    ) -> BenchmarkGateReceipt:
        """Evaluate benchmark gate for promotion decision."""
        # Run validation
        val_loss, _, val_metrics = self._validate(val_loader)

        # Use custom evaluator if provided
        if self.benchmark_evaluator is not None:
            gate_metrics = self.benchmark_evaluator(self.seam, val_loader)
        else:
            # Default: use inverse of validation loss as gate score
            gate_metrics = {"val_loss": val_loss, **val_metrics}

        # Compute gate score (lower loss = higher score)
        gate_score = 1.0 / (1.0 + val_loss)

        # Determine promotion
        previous_stage = self.descriptor.promotion_stage
        gate_passed = gate_score >= self.config.promotion_threshold

        if gate_passed and previous_stage in ("registered", "raw_provider_output"):
            new_stage = "promoted"
        elif (
            gate_score < self.config.demotion_threshold and previous_stage == "promoted"
        ):
            new_stage = "demoted_to_shadow"
        else:
            new_stage = previous_stage

        # Update registry
        if new_stage != previous_stage:
            # Save checkpoint before stage change
            self.registry.save_seam(self.seam_id)
            # Update stage in descriptor (would need registry method)

        receipt = BenchmarkGateReceipt(
            receipt_id=f"gate_{uuid.uuid4().hex[:12]}",
            seam_id=self.seam_id,
            seam_type=self.seam_type,
            step=self.state.step,
            gate_score=gate_score,
            gate_passed=gate_passed,
            gate_threshold=self.config.promotion_threshold,
            previous_stage=previous_stage,
            new_stage=new_stage,
            gate_metrics=gate_metrics,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        self.benchmark_receipts.append(receipt)
        return receipt

    def _emit_training_receipt(
        self,
        loss: float,
        components: Dict[str, float],
        metrics: Dict[str, float],
        grad_norm: float,
    ) -> SeamTrainingStepReceipt:
        """Emit training step receipt."""
        receipt = SeamTrainingStepReceipt(
            receipt_id=f"train_{uuid.uuid4().hex[:12]}",
            seam_id=self.seam_id,
            seam_type=self.seam_type,
            step=self.state.step,
            epoch=self.state.epoch,
            train_loss=loss,
            component_losses=components,
            metrics=metrics,
            learning_rate=self._get_current_lr(),
            grad_norm=grad_norm,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self.training_receipts.append(receipt)
        return receipt

    def _emit_validation_receipt(
        self,
        val_loss: float,
        components: Dict[str, float],
        metrics: Dict[str, float],
        is_best: bool,
    ) -> SeamValidationReceipt:
        """Emit validation receipt."""
        receipt = SeamValidationReceipt(
            receipt_id=f"val_{uuid.uuid4().hex[:12]}",
            seam_id=self.seam_id,
            seam_type=self.seam_type,
            step=self.state.step,
            val_loss=val_loss,
            component_losses=components,
            metrics=metrics,
            is_best=is_best,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self.validation_receipts.append(receipt)
        return receipt

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        *,
        resume_from_checkpoint: bool = True,
    ) -> Dict[str, Any]:
        """Run full training loop.

        Args:
            train_loader: Training data loader.
            val_loader: Optional validation data loader.
            resume_from_checkpoint: Whether to resume from existing checkpoint.

        Returns:
            Training summary dict.
        """
        self.state.training_start_time = time.time()

        # Calculate total steps
        steps_per_epoch = len(train_loader)
        if self.config.max_steps is not None:
            total_steps = self.config.max_steps
        else:
            total_steps = steps_per_epoch * self.config.max_epochs

        # Initialize scheduler
        if self._scheduler_needs_init:
            self._init_scheduler(total_steps)

        logger.info(
            f"Starting training for seam {self.seam_id} ({self.seam_type}). "
            f"Total steps: {total_steps}, LR: {self.config.learning_rate}"
        )

        # Training loop
        while not self.state.is_finished:
            self.state.epoch += 1

            for batch in train_loader:
                # Training step
                loss, components, metrics = self._training_step(batch)

                self.state.step += 1
                self.state.total_train_loss += loss
                self.state.total_train_samples += self._get_batch_size(batch)

                # Logging
                if self.state.step % self.config.log_interval == 0:
                    avg_loss = self.state.total_train_loss / max(
                        1, self.state.total_train_samples
                    )
                    logger.info(
                        f"Step {self.state.step}/{total_steps} | "
                        f"Epoch {self.state.epoch} | "
                        f"Loss: {loss:.4f} | "
                        f"Avg Loss: {avg_loss:.4f} | "
                        f"LR: {self._get_current_lr():.2e}"
                    )
                    self._emit_training_receipt(loss, components, metrics, 0.0)

                # Validation
                if (
                    val_loader is not None
                    and self.state.step % self.config.val_check_interval == 0
                ):
                    val_loss, val_components, val_metrics = self._validate(val_loader)
                    is_best = val_loss < self.state.best_val_loss

                    if is_best:
                        self.state.best_val_loss = val_loss
                        self.state.best_step = self.state.step
                        self.state.patience_counter = 0
                        self.registry.save_seam(self.seam_id)
                    else:
                        self.state.patience_counter += 1

                    self._emit_validation_receipt(
                        val_loss, val_components, val_metrics, is_best
                    )

                    logger.info(
                        f"Validation | Step {self.state.step} | "
                        f"Val Loss: {val_loss:.4f} | "
                        f"Best: {self.state.best_val_loss:.4f} @ step {self.state.best_step}"
                    )

                    # Early stopping
                    if (
                        self.state.patience_counter
                        >= self.config.early_stopping_patience
                    ):
                        self.state.is_finished = True
                        self.state.finish_reason = "early_stopping"
                        break

                # Checkpoint
                if self.state.step % self.config.checkpoint_interval == 0:
                    self.registry.save_seam(self.seam_id)

                # Benchmark gate
                if (
                    val_loader is not None
                    and self.state.step % self.config.benchmark_gate_interval == 0
                ):
                    gate_receipt = self._evaluate_benchmark_gate(val_loader)
                    logger.info(
                        f"Benchmark Gate | Step {self.state.step} | "
                        f"Score: {gate_receipt.gate_score:.4f} | "
                        f"Passed: {gate_receipt.gate_passed} | "
                        f"Stage: {gate_receipt.previous_stage} -> {gate_receipt.new_stage}"
                    )

                # Check max steps
                if (
                    self.config.max_steps is not None
                    and self.state.step >= self.config.max_steps
                ):
                    self.state.is_finished = True
                    self.state.finish_reason = "max_steps"
                    break

            # Check max epochs
            if self.state.epoch >= self.config.max_epochs:
                self.state.is_finished = True
                self.state.finish_reason = "max_epochs"

        # Final save
        self.registry.save_seam(self.seam_id)

        # Training summary
        training_time = time.time() - self.state.training_start_time
        summary = {
            "seam_id": self.seam_id,
            "seam_type": self.seam_type,
            "total_steps": self.state.step,
            "total_epochs": self.state.epoch,
            "best_val_loss": self.state.best_val_loss,
            "best_step": self.state.best_step,
            "finish_reason": self.state.finish_reason,
            "training_time_seconds": training_time,
            "training_receipts_count": len(self.training_receipts),
            "validation_receipts_count": len(self.validation_receipts),
            "benchmark_receipts_count": len(self.benchmark_receipts),
        }

        logger.info(f"Training complete. Summary: {summary}")
        return summary

    def save_receipts(self, path: str | Path) -> None:
        """Save all receipts to JSON file."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        receipts = {
            "seam_id": self.seam_id,
            "seam_type": self.seam_type,
            "training_receipts": [r.to_dict() for r in self.training_receipts],
            "validation_receipts": [r.to_dict() for r in self.validation_receipts],
            "benchmark_receipts": [r.to_dict() for r in self.benchmark_receipts],
        }

        output_path.write_text(json.dumps(receipts, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Convenience training functions
# ---------------------------------------------------------------------------


def train_evidence_fusion_seam(
    registry: PerceptionSeamRegistry,
    seam_id: str,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    *,
    config: Optional[SeamTrainingConfig] = None,
) -> Dict[str, Any]:
    """Train an evidence fusion seam."""
    trainer = PerceptionSeamTrainer(
        registry=registry,
        seam_id=seam_id,
        config=config,
    )
    return trainer.fit(train_loader, val_loader)


def train_sam_calibration_seam(
    registry: PerceptionSeamRegistry,
    seam_id: str,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    *,
    config: Optional[SeamTrainingConfig] = None,
) -> Dict[str, Any]:
    """Train a SAM calibration seam."""
    trainer = PerceptionSeamTrainer(
        registry=registry,
        seam_id=seam_id,
        config=config,
    )
    return trainer.fit(train_loader, val_loader)


def train_depth_calibration_seam(
    registry: PerceptionSeamRegistry,
    seam_id: str,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    *,
    config: Optional[SeamTrainingConfig] = None,
) -> Dict[str, Any]:
    """Train a depth calibration seam."""
    trainer = PerceptionSeamTrainer(
        registry=registry,
        seam_id=seam_id,
        config=config,
    )
    return trainer.fit(train_loader, val_loader)


def train_vjepa_temporal_seam(
    registry: PerceptionSeamRegistry,
    seam_id: str,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    *,
    config: Optional[SeamTrainingConfig] = None,
) -> Dict[str, Any]:
    """Train a V-JEPA temporal alignment seam."""
    trainer = PerceptionSeamTrainer(
        registry=registry,
        seam_id=seam_id,
        config=config,
    )
    return trainer.fit(train_loader, val_loader)


__all__ = [
    # Config
    "SeamTrainingConfig",
    "SeamTrainingState",
    # Receipts
    "SeamTrainingStepReceipt",
    "SeamValidationReceipt",
    "BenchmarkGateReceipt",
    # Trainer
    "PerceptionSeamTrainer",
    # Convenience functions
    "train_depth_calibration_seam",
    "train_evidence_fusion_seam",
    "train_sam_calibration_seam",
    "train_vjepa_temporal_seam",
]
