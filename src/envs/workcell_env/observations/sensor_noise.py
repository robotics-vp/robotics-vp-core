"""
Deterministic sensor noise models for workcell rendering.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from src.vision.nag.types import CameraParams


@dataclass(frozen=True)
class SensorNoiseConfig:
    """Configurable, deterministic sensor noise knobs."""

    depth_gaussian_std_m: float = 0.0
    depth_quantization_m: float = 0.0
    depth_dropout_prob: float = 0.0

    motion_blur_alpha: float = 0.0

    seg_erode_px: int = 0
    seg_dilate_px: int = 0
    seg_dropout_prob: float = 0.0
    seg_swap_prob: float = 0.0

    camera_jitter_std_m: float = 0.0
    camera_jitter_std_deg: float = 0.0

    rgb_gamma_std: float = 0.0
    rgb_brightness_std: float = 0.0
    rgb_contrast_std: float = 0.0

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "SensorNoiseConfig":
        payload = dict(data or {})
        return cls(
            depth_gaussian_std_m=float(payload.get("depth_gaussian_std_m", 0.0)),
            depth_quantization_m=float(payload.get("depth_quantization_m", 0.0)),
            depth_dropout_prob=float(payload.get("depth_dropout_prob", 0.0)),
            motion_blur_alpha=float(payload.get("motion_blur_alpha", 0.0)),
            seg_erode_px=int(payload.get("seg_erode_px", 0)),
            seg_dilate_px=int(payload.get("seg_dilate_px", 0)),
            seg_dropout_prob=float(payload.get("seg_dropout_prob", 0.0)),
            seg_swap_prob=float(payload.get("seg_swap_prob", 0.0)),
            camera_jitter_std_m=float(payload.get("camera_jitter_std_m", 0.0)),
            camera_jitter_std_deg=float(payload.get("camera_jitter_std_deg", 0.0)),
            rgb_gamma_std=float(payload.get("rgb_gamma_std", 0.0)),
            rgb_brightness_std=float(payload.get("rgb_brightness_std", 0.0)),
            rgb_contrast_std=float(payload.get("rgb_contrast_std", 0.0)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "depth_gaussian_std_m": self.depth_gaussian_std_m,
            "depth_quantization_m": self.depth_quantization_m,
            "depth_dropout_prob": self.depth_dropout_prob,
            "motion_blur_alpha": self.motion_blur_alpha,
            "seg_erode_px": self.seg_erode_px,
            "seg_dilate_px": self.seg_dilate_px,
            "seg_dropout_prob": self.seg_dropout_prob,
            "seg_swap_prob": self.seg_swap_prob,
            "camera_jitter_std_m": self.camera_jitter_std_m,
            "camera_jitter_std_deg": self.camera_jitter_std_deg,
            "rgb_gamma_std": self.rgb_gamma_std,
            "rgb_brightness_std": self.rgb_brightness_std,
            "rgb_contrast_std": self.rgb_contrast_std,
        }


def apply_sensor_noise(
    *,
    rgb_frames: List[np.ndarray],
    depth_frames: Optional[List[np.ndarray]],
    seg_frames: Optional[List[np.ndarray]],
    camera_params: CameraParams,
    seed: Optional[int],
    config: Mapping[str, Any] | None,
) -> Tuple[List[np.ndarray], Optional[List[np.ndarray]], Optional[List[np.ndarray]], CameraParams]:
    cfg = SensorNoiseConfig.from_dict(config)
    rng = np.random.RandomState(_stable_seed("sensor_noise", seed))

    rgb_frames = _apply_rgb_noise(rgb_frames, rng, cfg)
    depth_frames = _apply_depth_noise(depth_frames, rng, cfg)
    seg_frames = _apply_seg_noise(seg_frames, rng, cfg)
    camera_params = _apply_camera_jitter(camera_params, rng, cfg)

    return rgb_frames, depth_frames, seg_frames, camera_params


def _stable_seed(key: str, seed: Optional[int]) -> int:
    payload = f"{seed}:{key}" if seed is not None else key
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little", signed=False)


def _apply_rgb_noise(
    frames: List[np.ndarray],
    rng: np.random.RandomState,
    cfg: SensorNoiseConfig,
) -> List[np.ndarray]:
    if not frames:
        return frames
    out: List[np.ndarray] = []
    prev: Optional[np.ndarray] = None
    for frame in frames:
        img = np.asarray(frame).astype(np.float32) / 255.0
        if cfg.rgb_gamma_std > 0.0:
            gamma = max(0.1, 1.0 + rng.normal(0.0, cfg.rgb_gamma_std))
            img = np.power(img, gamma)
        if cfg.rgb_contrast_std > 0.0:
            contrast = 1.0 + rng.normal(0.0, cfg.rgb_contrast_std)
            img = (img - 0.5) * contrast + 0.5
        if cfg.rgb_brightness_std > 0.0:
            img = img + rng.normal(0.0, cfg.rgb_brightness_std)
        img = np.clip(img, 0.0, 1.0)
        if cfg.motion_blur_alpha > 0.0 and prev is not None:
            alpha = np.clip(cfg.motion_blur_alpha, 0.0, 1.0)
            img = (1.0 - alpha) * img + alpha * prev
        prev = img
        out.append((img * 255.0).astype(np.uint8))
    return out


def _apply_depth_noise(
    frames: Optional[List[np.ndarray]],
    rng: np.random.RandomState,
    cfg: SensorNoiseConfig,
) -> Optional[List[np.ndarray]]:
    if not frames:
        return frames
    out: List[np.ndarray] = []
    for frame in frames:
        depth = np.asarray(frame).astype(np.float32)
        if cfg.depth_gaussian_std_m > 0.0:
            depth = depth + rng.normal(0.0, cfg.depth_gaussian_std_m, size=depth.shape).astype(np.float32)
        if cfg.depth_quantization_m > 0.0:
            step = max(cfg.depth_quantization_m, 1e-9)
            depth = np.round(depth / step) * step
        if cfg.depth_dropout_prob > 0.0:
            mask = rng.rand(*depth.shape) < cfg.depth_dropout_prob
            depth[mask] = 0.0
        out.append(depth.astype(np.float32))
    return out


def _apply_seg_noise(
    frames: Optional[List[np.ndarray]],
    rng: np.random.RandomState,
    cfg: SensorNoiseConfig,
) -> Optional[List[np.ndarray]]:
    if not frames:
        return frames
    out: List[np.ndarray] = []
    for frame in frames:
        seg = np.asarray(frame).astype(np.int32)
        ids = [int(x) for x in np.unique(seg) if x != 0]
        if cfg.seg_dropout_prob > 0.0:
            for obj_id in ids:
                if rng.rand() < cfg.seg_dropout_prob:
                    seg[seg == obj_id] = 0
        if cfg.seg_swap_prob > 0.0 and len(ids) >= 2:
            if rng.rand() < cfg.seg_swap_prob:
                a, b = rng.choice(ids, size=2, replace=False)
                mask_a = seg == a
                mask_b = seg == b
                seg[mask_a] = b
                seg[mask_b] = a
        if cfg.seg_erode_px > 0 or cfg.seg_dilate_px > 0:
            seg = _morph_segmentation(seg, cfg.seg_erode_px, cfg.seg_dilate_px)
        out.append(seg.astype(np.int32))
    return out


def _apply_camera_jitter(
    camera_params: CameraParams,
    rng: np.random.RandomState,
    cfg: SensorNoiseConfig,
) -> CameraParams:
    if cfg.camera_jitter_std_m <= 0.0 and cfg.camera_jitter_std_deg <= 0.0:
        return camera_params
    world_from_cam = np.asarray(camera_params.world_from_cam).copy()
    for idx in range(world_from_cam.shape[0]):
        trans = rng.normal(0.0, cfg.camera_jitter_std_m, size=3) if cfg.camera_jitter_std_m > 0.0 else np.zeros(3)
        rot_deg = rng.normal(0.0, cfg.camera_jitter_std_deg, size=3) if cfg.camera_jitter_std_deg > 0.0 else np.zeros(3)
        delta = _transform_from_jitter(trans, rot_deg)
        world_from_cam[idx] = world_from_cam[idx] @ delta
    return CameraParams(
        fx=float(camera_params.fx),
        fy=float(camera_params.fy),
        cx=float(camera_params.cx),
        cy=float(camera_params.cy),
        height=int(camera_params.height),
        width=int(camera_params.width),
        world_from_cam=world_from_cam,
        near=float(camera_params.near),
        far=float(camera_params.far),
        camera_id=camera_params.camera_id,
    )


def _transform_from_jitter(translation: np.ndarray, rot_deg: np.ndarray) -> np.ndarray:
    rot = _rotation_matrix_from_rpy(np.deg2rad(rot_deg))
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = rot.astype(np.float32)
    transform[:3, 3] = translation.astype(np.float32)
    return transform


def _rotation_matrix_from_rpy(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rot_x = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float32)
    rot_y = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float32)
    rot_z = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float32)
    return rot_z @ rot_y @ rot_x


def _morph_segmentation(seg: np.ndarray, erode_px: int, dilate_px: int) -> np.ndarray:
    out = seg.copy()
    ids = [int(x) for x in np.unique(seg) if x != 0]
    for obj_id in ids:
        mask = out == obj_id
        for _ in range(max(erode_px, 0)):
            mask = _binary_erode(mask)
        for _ in range(max(dilate_px, 0)):
            expanded = _binary_dilate(mask)
            expanded &= (out == 0)
            mask = mask | expanded
        out[out == obj_id] = 0
        out[mask] = obj_id
    return out


def _binary_dilate(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    H, W = mask.shape
    out = np.zeros_like(mask, dtype=bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            out |= padded[1 + dy : 1 + dy + H, 1 + dx : 1 + dx + W]
    return out


def _binary_erode(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    H, W = mask.shape
    out = np.ones_like(mask, dtype=bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            out &= padded[1 + dy : 1 + dy + H, 1 + dx : 1 + dx + W]
    return out
