"""
SAM3D-Body Adapter.

Provides an adapter for SAM3D-Body model for human body reconstruction.
Uses stub implementation when actual model is not available.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _stable_seed(key: str, seed: Optional[int]) -> int:
    payload = f"{seed}:{key}" if seed is not None else key
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little", signed=False)

# Standard body joint names (SMPL-like skeleton)
BODY_JOINT_NAMES = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hand",
    "right_hand",
]


@dataclass
class BodyPrediction:
    """Prediction from SAM3D-Body for a single person.

    Attributes:
        body_id: Identifier for this body.
        mhr_params: Motion Hierarchy Representation parameters.
        joints_3d: Dict of joint_name -> (3,) world position.
        mesh_vertices: (V, 3) mesh vertex positions.
        mesh_faces: (F, 3) mesh face indices.
        camera_params: Estimated camera parameters dict.
        shape_latent: Body shape latent (e.g., SMPL beta).
        pose_latent: Body pose latent (e.g., SMPL theta).
        confidence: Model confidence for this prediction.
    """

    body_id: str
    joints_3d: Dict[str, np.ndarray]
    mesh_vertices: np.ndarray
    mesh_faces: np.ndarray
    mhr_params: Dict[str, Any]
    camera_params: Dict[str, Any]
    shape_latent: np.ndarray
    pose_latent: np.ndarray
    confidence: float = 1.0

    def __post_init__(self) -> None:
        self.mesh_vertices = np.asarray(self.mesh_vertices, dtype=np.float32)
        self.mesh_faces = np.asarray(self.mesh_faces, dtype=np.int32)
        self.shape_latent = np.asarray(self.shape_latent, dtype=np.float32)
        self.pose_latent = np.asarray(self.pose_latent, dtype=np.float32)
        self.confidence = float(self.confidence)

    @property
    def pelvis_position(self) -> np.ndarray:
        """Get pelvis position."""
        return self.joints_3d.get("pelvis", np.zeros(3, dtype=np.float32))

    def get_pose_matrix(self) -> np.ndarray:
        """Get 4x4 world_from_body transform centered at pelvis."""
        pose = np.eye(4, dtype=np.float32)
        pose[:3, 3] = self.pelvis_position
        return pose


@dataclass
class SAM3DBodyConfig:
    """Configuration for SAM3D-Body adapter.

    Attributes:
        model_path: Path to SAM3D-Body checkpoint.
        use_keypoints: Whether to use 2D keypoint input.
        shape_latent_dim: Dimension of shape latent.
        pose_latent_dim: Dimension of pose latent.
        output_mesh: Whether to output mesh vertices.
        stub_seed: Optional seed for deterministic stub outputs.
        device: Device for model inference.
    """

    model_path: Optional[str] = None
    use_keypoints: bool = True
    shape_latent_dim: int = 10
    pose_latent_dim: int = 72
    output_mesh: bool = True
    stub_seed: Optional[int] = None
    device: str = "cuda"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_path": self.model_path,
            "use_keypoints": self.use_keypoints,
            "shape_latent_dim": self.shape_latent_dim,
            "pose_latent_dim": self.pose_latent_dim,
            "output_mesh": self.output_mesh,
            "stub_seed": self.stub_seed,
            "device": self.device,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SAM3DBodyConfig":
        return cls(
            model_path=data.get("model_path"),
            use_keypoints=data.get("use_keypoints", True),
            shape_latent_dim=data.get("shape_latent_dim", 10),
            pose_latent_dim=data.get("pose_latent_dim", 72),
            output_mesh=data.get("output_mesh", True),
            stub_seed=data.get("stub_seed"),
            device=data.get("device", "cuda"),
        )


class SAM3DBodyAdapter:
    """Adapter for SAM3D-Body model.

    Provides a clean interface for running SAM3D-Body inference.
    Falls back to stub implementation if model is not available.
    """

    def __init__(
        self,
        config: Optional[SAM3DBodyConfig] = None,
        use_stub: bool = True,
    ):
        """Initialize adapter.

        Args:
            config: Configuration for the adapter.
            use_stub: If True, use stub implementation instead of real model.
        """
        self.config = config or SAM3DBodyConfig()
        self.use_stub = use_stub
        self._model = None

        if not use_stub:
            self._load_model()

    def _load_model(self) -> None:
        """Load SAM3D-Body model via third_party wrapper."""
        try:
            from third_party.sam3d_body_wrapper import SAM3DBodyInference
            
            self._wrapper = SAM3DBodyInference(
                weights_path=self.config.model_path,
                device=self.config.device,
                use_fallback=False,
            )
            
            if self._wrapper.is_real:
                logger.info("SAM3D-Body loaded via third_party wrapper")
                self.use_stub = False
            else:
                logger.info("SAM3D-Body wrapper using fallback mode")
                self.use_stub = True
        except ImportError as e:
            logger.warning(f"Failed to import third_party wrapper: {e}. Using stub.")
            self.use_stub = True
        except Exception as e:
            logger.warning(f"Failed to load SAM3D-Body: {e}. Using stub.")
            self.use_stub = True

    def infer(
        self,
        rgb: np.ndarray,
        person_mask: np.ndarray,
        keypoints: Optional[np.ndarray] = None,
        body_id: str = "body_0",
    ) -> BodyPrediction:
        """Run inference on image with person mask.

        Args:
            rgb: (H, W, 3) RGB image in [0, 255] uint8 or [0, 1] float.
            person_mask: (H, W) boolean mask for the person.
            keypoints: Optional (J, 3) keypoints with (x, y, confidence).
            body_id: Identifier for this body.

        Returns:
            BodyPrediction for the detected person.
        """
        if self.use_stub:
            return self._infer_stub(rgb, person_mask, keypoints, body_id)
        return self._infer_model(rgb, person_mask, keypoints, body_id)

    def _infer_model(
        self,
        rgb: np.ndarray,
        person_mask: np.ndarray,
        keypoints: Optional[np.ndarray],
        body_id: str,
    ) -> BodyPrediction:
        """Run actual model inference via third_party wrapper."""
        if not hasattr(self, '_wrapper') or self._wrapper is None:
            return self._infer_stub(rgb, person_mask, keypoints, body_id)
        
        result = self._wrapper.infer(rgb, person_mask, keypoints, body_id)
        
        return BodyPrediction(
            body_id=result.body_id,
            joints_3d=result.joints_3d,
            mesh_vertices=result.mesh_vertices,
            mesh_faces=result.mesh_faces,
            mhr_params={
                "root_orient": [0.0, 0.0, 0.0],
                "body_pose": result.pose_latent[:63].tolist() if len(result.pose_latent) >= 63 else [0.0] * 63,
                "transl": result.pelvis_position.tolist(),
            },
            camera_params=result.camera_params,
            shape_latent=result.shape_latent,
            pose_latent=result.pose_latent,
            confidence=result.confidence,
        )

    def _infer_stub(
        self,
        rgb: np.ndarray,
        person_mask: np.ndarray,
        keypoints: Optional[np.ndarray],
        body_id: str,
    ) -> BodyPrediction:
        """Generate stub prediction for testing.

        Creates a synthetic body prediction with T-pose skeleton.
        """
        H, W = rgb.shape[:2]

        # Compute mask center for positioning
        ys, xs = np.where(person_mask)
        if len(ys) > 0:
            center_x = xs.mean() / W
            center_y = ys.mean() / H
            mask_height = (ys.max() - ys.min()) / H
        else:
            center_x, center_y = 0.5, 0.5
            mask_height = 0.5

        # Estimate depth from mask size (larger mask = closer)
        depth = 3.0 / (mask_height + 0.1)
        depth = np.clip(depth, 1.5, 10.0)

        # Pelvis world position
        pelvis_x = (center_x - 0.5) * depth * 1.5
        pelvis_y = (center_y - 0.5) * depth * 1.5
        pelvis_z = depth

        # Generate T-pose skeleton
        rng = np.random.RandomState(_stable_seed(body_id, self.config.stub_seed))
        joints_3d = self._generate_tpose_skeleton(
            pelvis_position=np.array([pelvis_x, pelvis_y, pelvis_z]),
            height=1.7,  # Average human height
            noise_scale=0.02,
            rng=rng,
        )

        # Generate synthetic mesh (simplified)
        if self.config.output_mesh:
            mesh_vertices, mesh_faces = self._generate_stub_mesh(joints_3d, rng)
        else:
            mesh_vertices = np.zeros((1, 3), dtype=np.float32)
            mesh_faces = np.zeros((0, 3), dtype=np.int32)

        # Generate latents
        shape_latent = rng.randn(self.config.shape_latent_dim).astype(np.float32) * 0.5
        pose_latent = rng.randn(self.config.pose_latent_dim).astype(np.float32) * 0.1

        # MHR params (motion hierarchy representation)
        mhr_params = {
            "root_orient": [0.0, 0.0, 0.0],
            "body_pose": pose_latent[:63].tolist() if len(pose_latent) >= 63 else [0.0] * 63,
            "transl": joints_3d["pelvis"].tolist(),
        }

        # Camera params (rough estimate)
        camera_params = {
            "focal_length": H,
            "principal_point": [W / 2, H / 2],
            "rotation": [0.0, 0.0, 0.0],
            "translation": [0.0, 0.0, depth],
        }

        return BodyPrediction(
            body_id=body_id,
            joints_3d=joints_3d,
            mesh_vertices=mesh_vertices,
            mesh_faces=mesh_faces,
            mhr_params=mhr_params,
            camera_params=camera_params,
            shape_latent=shape_latent,
            pose_latent=pose_latent,
            confidence=0.85,
        )

    def _generate_tpose_skeleton(
        self,
        pelvis_position: np.ndarray,
        height: float = 1.7,
        noise_scale: float = 0.01,
        rng: Optional[np.random.RandomState] = None,
    ) -> Dict[str, np.ndarray]:
        """Generate T-pose skeleton joints.

        Args:
            pelvis_position: (3,) pelvis world position.
            height: Human height in meters.
            noise_scale: Noise to add to joint positions.
            rng: Random state for reproducibility.

        Returns:
            Dict of joint_name -> (3,) world position.
        """
        rng = rng or np.random.RandomState(0)

        # Relative offsets from pelvis in T-pose (normalized by height)
        offsets = {
            "pelvis": np.array([0.0, 0.0, 0.0]),
            "left_hip": np.array([-0.1, 0.0, 0.0]),
            "right_hip": np.array([0.1, 0.0, 0.0]),
            "spine1": np.array([0.0, 0.1, 0.0]),
            "left_knee": np.array([-0.1, -0.25, 0.0]),
            "right_knee": np.array([0.1, -0.25, 0.0]),
            "spine2": np.array([0.0, 0.2, 0.0]),
            "left_ankle": np.array([-0.1, -0.45, 0.0]),
            "right_ankle": np.array([0.1, -0.45, 0.0]),
            "spine3": np.array([0.0, 0.3, 0.0]),
            "left_foot": np.array([-0.1, -0.47, 0.05]),
            "right_foot": np.array([0.1, -0.47, 0.05]),
            "neck": np.array([0.0, 0.4, 0.0]),
            "left_collar": np.array([-0.08, 0.38, 0.0]),
            "right_collar": np.array([0.08, 0.38, 0.0]),
            "head": np.array([0.0, 0.5, 0.0]),
            "left_shoulder": np.array([-0.2, 0.35, 0.0]),
            "right_shoulder": np.array([0.2, 0.35, 0.0]),
            "left_elbow": np.array([-0.35, 0.35, 0.0]),
            "right_elbow": np.array([0.35, 0.35, 0.0]),
            "left_wrist": np.array([-0.5, 0.35, 0.0]),
            "right_wrist": np.array([0.5, 0.35, 0.0]),
            "left_hand": np.array([-0.55, 0.35, 0.0]),
            "right_hand": np.array([0.55, 0.35, 0.0]),
        }

        joints_3d = {}
        for name, offset in offsets.items():
            pos = pelvis_position + offset * height
            pos += rng.randn(3) * noise_scale
            joints_3d[name] = pos.astype(np.float32)

        return joints_3d

    def _generate_stub_mesh(
        self,
        joints_3d: Dict[str, np.ndarray],
        rng: np.random.RandomState,
    ) -> tuple:
        """Generate a simple mesh around skeleton.

        Returns (vertices, faces) tuple.
        """
        # Create a simple blob around the pelvis
        pelvis = joints_3d["pelvis"]
        num_vertices = 50
        vertices = rng.randn(num_vertices, 3).astype(np.float32) * 0.3 + pelvis

        # Simple triangle faces (not a proper mesh, just for testing)
        num_faces = (num_vertices - 2) // 3
        faces = np.arange(num_faces * 3).reshape(-1, 3).astype(np.int32)
        faces = np.clip(faces, 0, num_vertices - 1)

        return vertices, faces

    def infer_batch(
        self,
        rgb: np.ndarray,
        person_masks: List[np.ndarray],
        keypoints_list: Optional[List[Optional[np.ndarray]]] = None,
    ) -> List[BodyPrediction]:
        """Run inference on multiple people in same image.

        Args:
            rgb: (H, W, 3) RGB image.
            person_masks: List of (H, W) boolean masks.
            keypoints_list: Optional list of keypoints per person.

        Returns:
            List of BodyPrediction for each person.
        """
        predictions = []
        for i, mask in enumerate(person_masks):
            kpts = None
            if keypoints_list and i < len(keypoints_list):
                kpts = keypoints_list[i]
            pred = self.infer(rgb, mask, kpts, body_id=f"body_{i}")
            predictions.append(pred)
        return predictions


def create_sam3d_body_adapter(
    config: Optional[Dict[str, Any]] = None,
    use_stub: bool = True,
) -> SAM3DBodyAdapter:
    """Factory function to create SAM3D-Body adapter.

    Args:
        config: Configuration dict.
        use_stub: Whether to use stub implementation.

    Returns:
        Configured SAM3DBodyAdapter.
    """
    cfg = SAM3DBodyConfig.from_dict(config or {})
    return SAM3DBodyAdapter(config=cfg, use_stub=use_stub)
