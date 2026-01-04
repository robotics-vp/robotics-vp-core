"""
Scene IR Tracker Configuration.

Provides dataclass configurations for the Scene IR Tracker module.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional
import os


@dataclass
class IRRefinerConfig:
    """Configuration for inverse rendering refinement.

    Attributes:
        num_texture_iters: Iterations for texture fitting phase.
        num_pose_iters: Iterations for pose/scale fitting phase.
        num_shape_iters: Iterations for shape/joint refinement phase.
        lr_texture: Learning rate for texture optimization.
        lr_pose: Learning rate for pose optimization.
        lr_shape: Learning rate for shape optimization.
        rgb_loss_weight: Weight for RGB MSE loss.
        lpips_weight: Weight for LPIPS perceptual loss.
        embedding_reg_weight: Weight for embedding regularization (keep near mean).
        convergence_threshold: IR loss change threshold for convergence.
    """

    num_texture_iters: int = 50
    num_pose_iters: int = 30
    num_shape_iters: int = 20
    lr_texture: float = 1e-3
    lr_pose: float = 1e-4
    lr_shape: float = 1e-4
    rgb_loss_weight: float = 1.0
    lpips_weight: float = 0.1
    embedding_reg_weight: float = 0.01
    convergence_threshold: float = 1e-4
    
    # Guardrails
    max_pose_jump_m: float = 1.0  # Max position change per frame in meters
    max_scale_change_ratio: float = 2.0  # Max scale change ratio per frame
    latent_norm_max: float = 10.0  # Max L2 norm for latent vectors
    grad_clip_norm: float = 1.0  # Gradient clipping norm
    early_stop_patience: int = 10  # Stop if no improvement for this many iters
    divergence_loss_factor: float = 5.0  # If loss > factor * best_loss, mark diverged

    def __post_init__(self) -> None:
        self.num_texture_iters = max(1, int(self.num_texture_iters))
        self.num_pose_iters = max(1, int(self.num_pose_iters))
        self.num_shape_iters = max(1, int(self.num_shape_iters))
        self.lr_texture = float(self.lr_texture)
        self.lr_pose = float(self.lr_pose)
        self.lr_shape = float(self.lr_shape)
        self.rgb_loss_weight = float(self.rgb_loss_weight)
        self.lpips_weight = float(self.lpips_weight)
        self.embedding_reg_weight = float(self.embedding_reg_weight)
        self.convergence_threshold = float(self.convergence_threshold)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "num_texture_iters": self.num_texture_iters,
            "num_pose_iters": self.num_pose_iters,
            "num_shape_iters": self.num_shape_iters,
            "lr_texture": self.lr_texture,
            "lr_pose": self.lr_pose,
            "lr_shape": self.lr_shape,
            "rgb_loss_weight": self.rgb_loss_weight,
            "lpips_weight": self.lpips_weight,
            "embedding_reg_weight": self.embedding_reg_weight,
            "convergence_threshold": self.convergence_threshold,
            "max_pose_jump_m": self.max_pose_jump_m,
            "max_scale_change_ratio": self.max_scale_change_ratio,
            "latent_norm_max": self.latent_norm_max,
            "grad_clip_norm": self.grad_clip_norm,
            "early_stop_patience": self.early_stop_patience,
            "divergence_loss_factor": self.divergence_loss_factor,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IRRefinerConfig":
        """Create from dictionary."""
        return cls(
            num_texture_iters=data.get("num_texture_iters", 50),
            num_pose_iters=data.get("num_pose_iters", 30),
            num_shape_iters=data.get("num_shape_iters", 20),
            lr_texture=data.get("lr_texture", 1e-3),
            lr_pose=data.get("lr_pose", 1e-4),
            lr_shape=data.get("lr_shape", 1e-4),
            rgb_loss_weight=data.get("rgb_loss_weight", 1.0),
            lpips_weight=data.get("lpips_weight", 0.1),
            embedding_reg_weight=data.get("embedding_reg_weight", 0.01),
            convergence_threshold=data.get("convergence_threshold", 1e-4),
            max_pose_jump_m=data.get("max_pose_jump_m", 1.0),
            max_scale_change_ratio=data.get("max_scale_change_ratio", 2.0),
            latent_norm_max=data.get("latent_norm_max", 10.0),
            grad_clip_norm=data.get("grad_clip_norm", 1.0),
            early_stop_patience=data.get("early_stop_patience", 10),
            divergence_loss_factor=data.get("divergence_loss_factor", 5.0),
        )


@dataclass
class TrackingConfig:
    """Configuration for Kalman filter tracking.

    Attributes:
        kalman_process_noise: Process noise covariance diagonal.
        kalman_observation_noise: Observation noise covariance diagonal.
        association_distance_threshold: Max distance for track association.
        iou_threshold: Min 3D IoU for association (objects only).
        latent_similarity_weight: Weight for latent embedding similarity in cost.
        centroid_distance_weight: Weight for centroid distance in cost.
        iou_weight: Weight for 3D IoU in cost.
        max_age: Frames before track is considered dead.
        min_hits: Minimum hits before track is confirmed.
        ema_alpha: EMA decay for latent embedding updates.
    """

    kalman_process_noise: float = 0.1
    kalman_observation_noise: float = 0.5
    association_distance_threshold: float = 2.0
    iou_threshold: float = 0.1
    latent_similarity_weight: float = 0.3
    centroid_distance_weight: float = 0.5
    iou_weight: float = 0.2
    max_age: int = 5
    min_hits: int = 2
    ema_alpha: float = 0.9

    def __post_init__(self) -> None:
        self.kalman_process_noise = float(self.kalman_process_noise)
        self.kalman_observation_noise = float(self.kalman_observation_noise)
        self.association_distance_threshold = float(self.association_distance_threshold)
        self.iou_threshold = float(self.iou_threshold)
        self.latent_similarity_weight = float(self.latent_similarity_weight)
        self.centroid_distance_weight = float(self.centroid_distance_weight)
        self.iou_weight = float(self.iou_weight)
        self.max_age = max(1, int(self.max_age))
        self.min_hits = max(1, int(self.min_hits))
        self.ema_alpha = max(0.0, min(1.0, float(self.ema_alpha)))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "kalman_process_noise": self.kalman_process_noise,
            "kalman_observation_noise": self.kalman_observation_noise,
            "association_distance_threshold": self.association_distance_threshold,
            "iou_threshold": self.iou_threshold,
            "latent_similarity_weight": self.latent_similarity_weight,
            "centroid_distance_weight": self.centroid_distance_weight,
            "iou_weight": self.iou_weight,
            "max_age": self.max_age,
            "min_hits": self.min_hits,
            "ema_alpha": self.ema_alpha,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrackingConfig":
        """Create from dictionary."""
        return cls(
            kalman_process_noise=data.get("kalman_process_noise", 0.1),
            kalman_observation_noise=data.get("kalman_observation_noise", 0.5),
            association_distance_threshold=data.get("association_distance_threshold", 2.0),
            iou_threshold=data.get("iou_threshold", 0.1),
            latent_similarity_weight=data.get("latent_similarity_weight", 0.3),
            centroid_distance_weight=data.get("centroid_distance_weight", 0.5),
            iou_weight=data.get("iou_weight", 0.2),
            max_age=data.get("max_age", 5),
            min_hits=data.get("min_hits", 2),
            ema_alpha=data.get("ema_alpha", 0.9),
        )


@dataclass
class SceneIRTrackerConfig:
    """Configuration for the Scene IR Tracker.

    Attributes:
        device: Device for computation ("cpu" or "cuda").
        precision: Floating point precision ("float32" or "float16").
        batch_size: Batch size for processing.
        use_point_map: Whether to use point map input for SAM3D.
        body_joints_for_mhn: Joints to extract for MHN input.
        sam3d_objects_config: Configuration dict for SAM3D-Objects adapter.
        sam3d_body_config: Configuration dict for SAM3D-Body adapter.
        ir_refiner_config: Configuration for IR refinement.
        tracking_config: Configuration for Kalman tracking.
        use_stub_adapters: Use stub implementations (for testing without models).
    """

    device: str = "cuda"
    precision: Literal["float32", "float16"] = "float32"
    batch_size: int = 1
    use_point_map: bool = False
    body_joints_for_mhn: List[str] = field(
        default_factory=lambda: ["pelvis", "left_hand", "right_hand", "left_foot", "right_foot"]
    )
    sam3d_objects_config: Dict[str, Any] = field(default_factory=dict)
    sam3d_body_config: Dict[str, Any] = field(default_factory=dict)
    ir_refiner_config: IRRefinerConfig = field(default_factory=IRRefinerConfig)
    tracking_config: TrackingConfig = field(default_factory=TrackingConfig)
    use_stub_adapters: bool = True
    allow_fallbacks: bool = field(default_factory=lambda: os.environ.get("SCENE_IR_ALLOW_FALLBACKS", "0") == "1")

    def __post_init__(self) -> None:
        self.device = str(self.device)
        if self.precision not in ("float32", "float16"):
            self.precision = "float32"
        self.batch_size = max(1, int(self.batch_size))
        self.use_point_map = bool(self.use_point_map)
        self.use_stub_adapters = bool(self.use_stub_adapters)
        self.allow_fallbacks = bool(self.allow_fallbacks)

        # Convert nested configs from dicts if needed
        if isinstance(self.ir_refiner_config, dict):
            self.ir_refiner_config = IRRefinerConfig.from_dict(self.ir_refiner_config)
        if isinstance(self.tracking_config, dict):
            self.tracking_config = TrackingConfig.from_dict(self.tracking_config)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "device": self.device,
            "precision": self.precision,
            "batch_size": self.batch_size,
            "use_point_map": self.use_point_map,
            "body_joints_for_mhn": self.body_joints_for_mhn,
            "sam3d_objects_config": self.sam3d_objects_config,
            "sam3d_body_config": self.sam3d_body_config,
            "ir_refiner_config": self.ir_refiner_config.to_dict(),
            "tracking_config": self.tracking_config.to_dict(),
            "use_stub_adapters": self.use_stub_adapters,
            "allow_fallbacks": self.allow_fallbacks,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SceneIRTrackerConfig":
        """Create from dictionary."""
        ir_config = data.get("ir_refiner_config", {})
        if isinstance(ir_config, dict):
            ir_config = IRRefinerConfig.from_dict(ir_config)

        tracking_config = data.get("tracking_config", {})
        if isinstance(tracking_config, dict):
            tracking_config = TrackingConfig.from_dict(tracking_config)

        return cls(
            device=data.get("device", "cuda"),
            precision=data.get("precision", "float32"),
            batch_size=data.get("batch_size", 1),
            use_point_map=data.get("use_point_map", False),
            body_joints_for_mhn=data.get(
                "body_joints_for_mhn",
                ["pelvis", "left_hand", "right_hand", "left_foot", "right_foot"],
            ),
            sam3d_objects_config=data.get("sam3d_objects_config", {}),
            sam3d_body_config=data.get("sam3d_body_config", {}),
            ir_refiner_config=ir_config,
            tracking_config=tracking_config,
            use_stub_adapters=data.get("use_stub_adapters", True),
            allow_fallbacks=data.get("allow_fallbacks", os.environ.get("SCENE_IR_ALLOW_FALLBACKS", "0") == "1"),
        )
