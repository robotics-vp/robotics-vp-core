"""
SAM3D-Body Wrapper.

Provides a clean interface to SAM3D-Body inference with:
- Real model integration when available
- Fallback stub for testing without weights
- CPU and GPU support
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Try to import real SAM3D-Body
_REAL_SAM3D_BODY_AVAILABLE = False
try:
    from sam3d_body import SAM3D_Body  # type: ignore
    _REAL_SAM3D_BODY_AVAILABLE = True
except ImportError:
    try:
        from third_party.sam3d_body import SAM3D_Body  # type: ignore
        _REAL_SAM3D_BODY_AVAILABLE = True
    except ImportError:
        pass

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    TORCH_AVAILABLE = False


# Standard SMPL-like joint names
JOINT_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hand", "right_hand",
]


@dataclass
class BodyInferenceResult:
    """Result from SAM3D-Body inference.
    
    Attributes:
        body_id: Identifier for this person.
        joints_3d: Dict of joint_name -> (3,) world position.
        mesh_vertices: (V, 3) mesh vertices.
        mesh_faces: (F, 3) face indices.
        shape_latent: SMPL beta parameters (10,).
        pose_latent: SMPL theta parameters (72,).
        camera_params: Dict with focal, principal point, rotation, translation.
        confidence: Model confidence [0, 1].
    """
    body_id: str
    joints_3d: Dict[str, np.ndarray]
    mesh_vertices: np.ndarray
    mesh_faces: np.ndarray
    shape_latent: np.ndarray
    pose_latent: np.ndarray
    camera_params: Dict[str, Any]
    confidence: float = 1.0
    
    @property
    def pelvis_position(self) -> np.ndarray:
        """Get pelvis position."""
        return self.joints_3d.get("pelvis", np.zeros(3, dtype=np.float32))
    
    def get_pose_matrix(self) -> np.ndarray:
        """Get 4x4 world_from_body transform centered at pelvis."""
        pose = np.eye(4, dtype=np.float32)
        pose[:3, 3] = self.pelvis_position
        return pose


class SAM3DBodyInference:
    """Wrapper for SAM3D-Body inference."""
    
    DEFAULT_WEIGHTS_PATH = "checkpoints/sam3d_body/checkpoint.pth"
    SHAPE_DIM = 10
    POSE_DIM = 72
    
    def __init__(
        self,
        weights_path: Optional[str] = None,
        device: str = "cuda",
        use_fallback: bool = False,
        allow_fallback: bool = True,
    ):
        """Initialize inference wrapper.
        
        Args:
            weights_path: Path to model weights.
            device: Device for inference.
            use_fallback: Force fallback stub.
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
        if not _REAL_SAM3D_BODY_AVAILABLE:
            msg = (
                "SAM3D-Body not installed. "
                "Install with: git clone https://github.com/facebookresearch/sam-3d-body.git third_party/sam3d_body && "
                "pip install -e third_party/sam3d_body"
            )
            if not self.allow_fallback:
                raise RuntimeError(msg)
            logger.info(f"{msg} Using fallback stub.")
            self.use_fallback = True
            return
        
        if not Path(self.weights_path).exists():
            msg = (
                f"SAM3D-Body weights not found at {self.weights_path}. "
                "Download weights from: https://github.com/facebookresearch/sam-3d-body/releases"
            )
            if not self.allow_fallback:
                raise RuntimeError(msg)
            logger.warning(f"{msg} Using fallback stub.")
            self.use_fallback = True
            return
        
        try:
            self._model = SAM3D_Body.from_pretrained(self.weights_path)  # type: ignore
            if self.device == "cuda":
                self._model = self._model.cuda()
            self._model.eval()
            logger.info("SAM3D-Body model loaded successfully")
        except Exception as e:
            msg = f"Failed to load SAM3D-Body: {e}"
            if not self.allow_fallback:
                raise RuntimeError(msg)
            logger.warning(f"{msg}. Using fallback.")
            self.use_fallback = True
    
    @property
    def is_real(self) -> bool:
        """Returns True if using real model."""
        return self._model is not None and not self.use_fallback
    
    def infer(
        self,
        rgb: np.ndarray,
        person_mask: np.ndarray,
        keypoints: Optional[np.ndarray] = None,
        body_id: str = "body_0",
    ) -> BodyInferenceResult:
        """Run inference on RGB image with person mask."""
        if self.use_fallback:
            return self._infer_fallback(rgb, person_mask, keypoints, body_id)
        return self._infer_real(rgb, person_mask, keypoints, body_id)
    
    def _infer_real(
        self,
        rgb: np.ndarray,
        person_mask: np.ndarray,
        keypoints: Optional[np.ndarray],
        body_id: str,
    ) -> BodyInferenceResult:
        """Run real SAM3D-Body inference."""
        if self._model is None:
            return self._infer_fallback(rgb, person_mask, keypoints, body_id)
        
        if rgb.dtype == np.uint8:
            rgb = rgb.astype(np.float32) / 255.0
        
        try:
            with torch.no_grad():
                rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
                mask_t = torch.from_numpy(person_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
                
                if self.device == "cuda":
                    rgb_t = rgb_t.cuda()
                    mask_t = mask_t.cuda()
                
                output = self._model(rgb_t, mask_t)
                
                # Extract results
                joints_3d = {}
                joints_tensor = output["joints_3d"].cpu().numpy()[0]  # (J, 3)
                for j, name in enumerate(JOINT_NAMES):
                    if j < len(joints_tensor):
                        joints_3d[name] = joints_tensor[j].astype(np.float32)
                
                return BodyInferenceResult(
                    body_id=body_id,
                    joints_3d=joints_3d,
                    mesh_vertices=output["vertices"].cpu().numpy()[0],
                    mesh_faces=output["faces"].cpu().numpy(),
                    shape_latent=output["betas"].cpu().numpy().flatten()[:self.SHAPE_DIM],
                    pose_latent=output["body_pose"].cpu().numpy().flatten()[:self.POSE_DIM],
                    camera_params={
                        "focal_length": float(output.get("focal_length", rgb.shape[0])),
                        "principal_point": [rgb.shape[1] / 2, rgb.shape[0] / 2],
                    },
                    confidence=float(output.get("confidence", 0.9)),
                )
        except Exception as e:
            logger.warning(f"Real inference failed: {e}. Using fallback.")
            return self._infer_fallback(rgb, person_mask, keypoints, body_id)
    
    def _infer_fallback(
        self,
        rgb: np.ndarray,
        person_mask: np.ndarray,
        keypoints: Optional[np.ndarray],
        body_id: str,
    ) -> BodyInferenceResult:
        """Generate fallback result."""
        H, W = rgb.shape[:2]
        
        # Compute mask bounds
        ys, xs = np.where(person_mask)
        if len(ys) > 0:
            center_x = xs.mean() / W
            center_y = ys.mean() / H
            mask_height = (ys.max() - ys.min()) / H
        else:
            center_x, center_y = 0.5, 0.5
            mask_height = 0.5
        
        # Estimate depth from mask size
        depth = 3.0 / (mask_height + 0.1)
        depth = np.clip(depth, 1.5, 10.0)
        
        # Pelvis position
        pelvis = np.array([
            (center_x - 0.5) * depth * 1.5,
            (center_y - 0.5) * depth * 1.5,
            depth,
        ], dtype=np.float32)
        
        # Generate T-pose skeleton
        rng = np.random.RandomState(hash(body_id) % (2**31))
        joints_3d = self._generate_tpose(pelvis, height=1.7, rng=rng)
        
        # Generate simple mesh
        mesh_vertices, mesh_faces = self._generate_stub_mesh(joints_3d, rng)
        
        # Latents
        shape_latent = rng.randn(self.SHAPE_DIM).astype(np.float32) * 0.5
        pose_latent = rng.randn(self.POSE_DIM).astype(np.float32) * 0.1
        
        return BodyInferenceResult(
            body_id=body_id,
            joints_3d=joints_3d,
            mesh_vertices=mesh_vertices,
            mesh_faces=mesh_faces,
            shape_latent=shape_latent,
            pose_latent=pose_latent,
            camera_params={
                "focal_length": H,
                "principal_point": [W / 2, H / 2],
            },
            confidence=0.85,
        )
    
    def _generate_tpose(
        self,
        pelvis: np.ndarray,
        height: float = 1.7,
        rng: Optional[np.random.RandomState] = None,
    ) -> Dict[str, np.ndarray]:
        """Generate T-pose skeleton."""
        rng = rng or np.random.RandomState(0)
        
        offsets = {
            "pelvis": [0.0, 0.0, 0.0],
            "left_hip": [-0.1, 0.0, 0.0],
            "right_hip": [0.1, 0.0, 0.0],
            "spine1": [0.0, 0.1, 0.0],
            "left_knee": [-0.1, -0.25, 0.0],
            "right_knee": [0.1, -0.25, 0.0],
            "spine2": [0.0, 0.2, 0.0],
            "left_ankle": [-0.1, -0.45, 0.0],
            "right_ankle": [0.1, -0.45, 0.0],
            "spine3": [0.0, 0.3, 0.0],
            "left_foot": [-0.1, -0.47, 0.05],
            "right_foot": [0.1, -0.47, 0.05],
            "neck": [0.0, 0.4, 0.0],
            "left_collar": [-0.08, 0.38, 0.0],
            "right_collar": [0.08, 0.38, 0.0],
            "head": [0.0, 0.5, 0.0],
            "left_shoulder": [-0.2, 0.35, 0.0],
            "right_shoulder": [0.2, 0.35, 0.0],
            "left_elbow": [-0.35, 0.35, 0.0],
            "right_elbow": [0.35, 0.35, 0.0],
            "left_wrist": [-0.5, 0.35, 0.0],
            "right_wrist": [0.5, 0.35, 0.0],
            "left_hand": [-0.55, 0.35, 0.0],
            "right_hand": [0.55, 0.35, 0.0],
        }
        
        joints = {}
        for name, offset in offsets.items():
            pos = pelvis + np.array(offset, dtype=np.float32) * height
            pos += rng.randn(3) * 0.01
            joints[name] = pos.astype(np.float32)
        
        return joints
    
    def _generate_stub_mesh(
        self,
        joints: Dict[str, np.ndarray],
        rng: np.random.RandomState,
    ) -> tuple:
        """Generate simple mesh around skeleton."""
        pelvis = joints["pelvis"]
        num_verts = 102  # Divisible by 3 for faces
        vertices = rng.randn(num_verts, 3).astype(np.float32) * 0.3 + pelvis
        # Create triangles from consecutive vertex indices
        num_tris = num_verts // 3
        faces = np.arange(num_tris * 3).reshape(-1, 3).astype(np.int32)
        return vertices, faces
    
    def infer_batch(
        self,
        rgb: np.ndarray,
        person_masks: List[np.ndarray],
    ) -> List[BodyInferenceResult]:
        """Run inference on multiple people."""
        return [
            self.infer(rgb, mask, body_id=f"body_{i}")
            for i, mask in enumerate(person_masks)
        ]


if __name__ == "__main__":
    wrapper = SAM3DBodyInference(use_fallback=True)
    print(f"SAM3D-Body wrapper initialized (real={wrapper.is_real})")
    
    test_rgb = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    test_mask = np.zeros((256, 256), dtype=bool)
    test_mask[50:200, 100:150] = True
    
    result = wrapper.infer(test_rgb, test_mask)
    print(f"Joints: {len(result.joints_3d)}")
    print(f"Pelvis: {result.pelvis_position}")
