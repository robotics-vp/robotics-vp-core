"""Token providers for channel-set encoding."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import hashlib
import math

import numpy as np
import torch
import torch.nn.functional as F

from src.embodiment.artifacts import (
    EMBODIMENT_PROFILE_PREFIX,
    SKILL_SEGMENTS_PREFIX,
)
from src.scene.vector_scene.encoding import ordered_scene_tensors, SceneGraphEncoder
from src.envs.lsd3d_env.gaussian_scene import GaussianScene
from src.vision.scene_ir_tracker.serialization import deserialize_scene_tracks_v1, SceneTracksLite
from src.process_reward.schemas import MHNSummary


@dataclass
class TokenProviderOutput:
    channel_name: str
    tokens: torch.Tensor
    mask: torch.Tensor
    metadata: Dict[str, Any]


class BaseTokenProvider:
    channel_name: str

    def provide(
        self,
        episodes: Sequence[Any] | Any,
        target_len: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> TokenProviderOutput:
        raise NotImplementedError


class EmbodimentTokenProvider(BaseTokenProvider):
    """Convert embodiment artifacts into time-aligned tokens."""

    def __init__(self, channel_name: str = "embodiment", debug: bool = False, allow_synthetic: bool = False):
        self.channel_name = channel_name
        self.debug = debug
        self.allow_synthetic = allow_synthetic

    def provide(
        self,
        episodes: Sequence[Any] | Any,
        target_len: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> TokenProviderOutput:
        episode_list = _normalize_episode_list(episodes)
        tokens_list: List[torch.Tensor] = []
        masks: List[torch.Tensor] = []
        metadata: Dict[str, Any] = {"feature_names": None, "debug": self.debug}

        for ep in episode_list:
            payload = _extract_embodiment_payload(ep)
            if payload is None:
                if not self.allow_synthetic:
                    raise ValueError("Embodiment artifacts missing for required channel")
                payload = _synthetic_embodiment_payload()
            features, names = _embodiment_features(payload)
            token = torch.tensor(features, dtype=torch.float32)
            if target_len is not None:
                token = _resample_tokens(token, target_len)
            tokens_list.append(token)
            masks.append(torch.ones(token.shape[0], dtype=torch.float32))
            metadata["feature_names"] = names

        tokens = _stack_tokens(tokens_list)
        mask = _stack_masks(masks, tokens.shape[1])
        if device is not None:
            tokens = tokens.to(device)
            mask = mask.to(device)

        return TokenProviderOutput(
            channel_name=self.channel_name,
            tokens=tokens,
            mask=mask,
            metadata=metadata,
        )


class SceneGraphTokenProvider(BaseTokenProvider):
    """Encode scene graphs into per-timestep tokens."""

    def __init__(
        self,
        channel_name: str = "geometry_scene_graph",
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        pos_dim: int = 32,
        pool_method: str = "mean",
    ) -> None:
        self.channel_name = channel_name
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.pos_dim = pos_dim
        self.pool_method = pool_method
        self._encoder: Optional[SceneGraphEncoder] = None

    def provide(
        self,
        episodes: Sequence[Any] | Any,
        target_len: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> TokenProviderOutput:
        episode_list = _normalize_episode_list(episodes)
        tokens_list: List[torch.Tensor] = []
        masks: List[torch.Tensor] = []

        for ep in episode_list:
            graphs = _extract_scene_graphs(ep)
            if not graphs:
                raise ValueError("Scene graph data missing for required channel")
            per_step = []
            for graph in graphs:
                scene_tensors = _scene_graph_to_tensors(graph, pos_dim=self.pos_dim, device=device)
                if self._encoder is None:
                    self._encoder = SceneGraphEncoder(
                        node_input_dim=scene_tensors["node_features"].shape[-1],
                        obj_input_dim=scene_tensors["object_features"].shape[-1],
                        hidden_dim=self.hidden_dim,
                        num_layers=self.num_layers,
                        num_heads=self.num_heads,
                        pos_dim=self.pos_dim,
                        pool_method=self.pool_method,
                    )
                    if device is not None:
                        self._encoder = self._encoder.to(device)
                with torch.no_grad():
                    output = self._encoder(scene_tensors)
                scene_latent = output["scene_latent"].detach()
                if scene_latent.dim() == 1:
                    scene_latent = scene_latent.unsqueeze(0)
                per_step.append(scene_latent.squeeze(0))
            token = torch.stack(per_step, dim=0)
            if target_len is not None:
                token = _resample_tokens(token, target_len)
            tokens_list.append(token)
            masks.append(torch.ones(token.shape[0], dtype=torch.float32))

        tokens = _stack_tokens(tokens_list)
        mask = _stack_masks(masks, tokens.shape[1])
        if device is not None:
            tokens = tokens.to(device)
            mask = mask.to(device)

        return TokenProviderOutput(
            channel_name=self.channel_name,
            tokens=tokens,
            mask=mask,
            metadata={},
        )


@dataclass
class GeometryBEVConfig:
    resolution_m: float = 0.2
    extent_m: float = 5.0
    max_tracks: int = 64
    patch_size: int = 4
    embed_dim: int = 128
    seed: int = 0
    return_grid: bool = False
    debug_rotation_check: bool = False
    rotation_check_degrees: int = 90
    rotation_check_tolerance: float = 1e-4
    rotation_check_max_episodes: int = 1


class GeometryBEVProvider(BaseTokenProvider):
    """Map-first BEV tokens from SceneIR tracks."""

    def __init__(
        self,
        channel_name: str = "geometry_bev",
        config: Optional[GeometryBEVConfig] = None,
        allow_synthetic: bool = False,
    ) -> None:
        self.channel_name = channel_name
        self.config = config or GeometryBEVConfig()
        self.allow_synthetic = allow_synthetic
        self._proj: Optional[torch.Tensor] = None

    def provide(
        self,
        episodes: Sequence[Any] | Any,
        target_len: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> TokenProviderOutput:
        episode_list = _normalize_episode_list(episodes)
        tokens_list: List[torch.Tensor] = []
        masks: List[torch.Tensor] = []
        grids: List[np.ndarray] = []
        stats_list: List[Dict[str, float]] = []

        feature_names = _bev_feature_names()
        frame_spec = {
            "origin": "ego",
            "axes": "x_forward_y_left_z_up",
            "resolution_m": float(self.config.resolution_m),
            "extent_m": float(self.config.extent_m),
        }
        frame_sources: List[str] = []

        rotation_checked = 0
        for ep in episode_list:
            scene_tracks = _extract_scene_tracks(ep)
            if scene_tracks is None:
                if not self.allow_synthetic:
                    raise ValueError("geometry_bev channel requires scene_tracks")
                scene_tracks = _synthetic_scene_tracks()

            mhn_features = _extract_mhn_features(ep, scene_tracks.num_frames, scene_tracks.num_tracks)
            bev_grid, stats, frame_source = _build_bev_grid(scene_tracks, ep, mhn_features, self.config)
            if self.config.debug_rotation_check and rotation_checked < self.config.rotation_check_max_episodes:
                _check_bev_rotation_consistency(scene_tracks, ep, mhn_features, bev_grid, self.config)
                rotation_checked += 1
            frame_sources.append(frame_source)
            if target_len is not None and bev_grid.shape[0] != target_len:
                bev_grid = _resample_bev_grid(bev_grid, target_len)
            token, self._proj = _bev_grid_to_token(bev_grid, self.config, self._proj)
            if target_len is not None and token.shape[0] != target_len:
                token = _resample_tokens(token, target_len)
            tokens_list.append(token)
            masks.append(torch.ones(token.shape[0], dtype=torch.float32))
            stats_list.append(stats)
            if self.config.return_grid:
                grids.append(bev_grid)

        tokens = _stack_tokens(tokens_list)
        mask = _stack_masks(masks, tokens.shape[1])
        metadata: Dict[str, Any] = {
            "feature_names": feature_names,
            "frame_spec": frame_spec,
            "bev_stats": stats_list,
            "frame_sources": frame_sources,
        }
        if self.config.return_grid:
            metadata["bev_grid"] = _stack_bev_grids(grids, tokens.shape[1])

        if device is not None:
            tokens = tokens.to(device)
            mask = mask.to(device)
            if "bev_grid" in metadata:
                metadata["bev_grid"] = metadata["bev_grid"].to(device)

        return TokenProviderOutput(
            channel_name=self.channel_name,
            tokens=tokens,
            mask=mask,
            metadata=metadata,
        )


class RGBVisionTokenProvider(BaseTokenProvider):
    """Deterministic RGB token provider with layout flexibility."""

    def __init__(
        self,
        channel_name: str = "vision_rgb",
        token_dim: int = 64,
        pool_size: Tuple[int, int] = (4, 4),
        seed: int = 0,
        allow_synthetic: bool = False,
    ) -> None:
        self.channel_name = channel_name
        self.token_dim = token_dim
        self.pool_size = pool_size
        self.seed = seed
        self.allow_synthetic = allow_synthetic
        self._proj = _deterministic_projection(3 * pool_size[0] * pool_size[1] + 6, token_dim, seed)

    def provide(
        self,
        episodes: Sequence[Any] | Any,
        target_len: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> TokenProviderOutput:
        episode_list = _normalize_episode_list(episodes)
        tokens_list: List[torch.Tensor] = []
        masks: List[torch.Tensor] = []
        metadata: Dict[str, Any] = {"pool_size": self.pool_size}

        for ep in episode_list:
            features_payload = _extract_rgb_features(ep)
            if features_payload is not None:
                token = _rgb_features_to_tokens(features_payload)
                metadata["rgb_features_v1"] = features_payload
            else:
                frames = _extract_rgb_frames(ep)
                if frames is None:
                    frames = _proxy_render_from_geometry(ep)
                if frames is None:
                    if not self.allow_synthetic:
                        raise ValueError("vision_rgb channel requires frames or proxy render")
                    frames = _synthetic_rgb_frames(ep)

                token = _rgb_frames_to_tokens(frames, self.pool_size, self._proj)
            if target_len is not None:
                token = _resample_tokens(token, target_len)
            tokens_list.append(token)
            masks.append(torch.ones(token.shape[0], dtype=torch.float32))

        tokens = _stack_tokens(tokens_list)
        mask = _stack_masks(masks, tokens.shape[1])
        if device is not None:
            tokens = tokens.to(device)
            mask = mask.to(device)

        return TokenProviderOutput(
            channel_name=self.channel_name,
            tokens=tokens,
            mask=mask,
            metadata=metadata,
        )


class GaussianSceneTokenProvider(BaseTokenProvider):
    """Optional Gaussian scene summary tokens."""

    def __init__(
        self,
        channel_name: str = "geometry_gaussian_scene",
        token_dim: Optional[int] = None,
        seed: int = 0,
    ) -> None:
        self.channel_name = channel_name
        self.token_dim = token_dim
        self.seed = seed
        self._proj = None

    def provide(
        self,
        episodes: Sequence[Any] | Any,
        target_len: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> TokenProviderOutput:
        episode_list = _normalize_episode_list(episodes)
        tokens_list: List[torch.Tensor] = []
        masks: List[torch.Tensor] = []

        for ep in episode_list:
            scenes = _extract_gaussian_scenes(ep)
            if not scenes:
                raise ValueError("GaussianScene channel requested but data missing")
            per_step = []
            for scene in scenes:
                features = _gaussian_scene_features(scene)
                if self.token_dim is not None:
                    if self._proj is None:
                        self._proj = _deterministic_projection(features.shape[-1], self.token_dim, self.seed)
                    feature_tensor = torch.tensor(features, dtype=torch.float32) @ self._proj
                else:
                    feature_tensor = torch.tensor(features, dtype=torch.float32)
                per_step.append(feature_tensor)
            token = torch.stack(per_step, dim=0)
            if target_len is not None:
                token = _resample_tokens(token, target_len)
            tokens_list.append(token)
            masks.append(torch.ones(token.shape[0], dtype=torch.float32))

        tokens = _stack_tokens(tokens_list)
        mask = _stack_masks(masks, tokens.shape[1])
        if device is not None:
            tokens = tokens.to(device)
            mask = mask.to(device)

        return TokenProviderOutput(
            channel_name=self.channel_name,
            tokens=tokens,
            mask=mask,
            metadata={},
        )


# ---- Helpers ----

def _normalize_episode_list(episodes: Sequence[Any] | Any) -> List[Any]:
    if isinstance(episodes, (list, tuple)):
        return list(episodes)
    return [episodes]


def _stack_tokens(tokens_list: List[torch.Tensor]) -> torch.Tensor:
    if not tokens_list:
        raise ValueError("No tokens produced")
    max_len = max(t.shape[0] for t in tokens_list)
    padded = []
    for t in tokens_list:
        if t.shape[0] < max_len:
            pad_len = max_len - t.shape[0]
            pad = torch.zeros((pad_len, t.shape[1]), dtype=t.dtype)
            padded.append(torch.cat([t, pad], dim=0))
        else:
            padded.append(t)
    return torch.stack(padded, dim=0)


def _stack_masks(masks: List[torch.Tensor], target_len: int) -> torch.Tensor:
    if not masks:
        return torch.zeros((0, target_len), dtype=torch.float32)
    padded = []
    for m in masks:
        if m.shape[0] < target_len:
            pad_len = target_len - m.shape[0]
            pad = torch.zeros((pad_len,), dtype=m.dtype)
            padded.append(torch.cat([m, pad], dim=0))
        else:
            padded.append(m[:target_len])
    return torch.stack(padded, dim=0)


def _resample_tokens(tokens: torch.Tensor, target_len: int) -> torch.Tensor:
    if tokens.shape[0] == target_len:
        return tokens
    tokens_t = tokens.transpose(0, 1).unsqueeze(0)
    resized = F.interpolate(tokens_t, size=target_len, mode="linear", align_corners=False)
    return resized.squeeze(0).transpose(0, 1)


def _deterministic_projection(input_dim: int, output_dim: int, seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    weight = rng.standard_normal((input_dim, output_dim)).astype(np.float32)
    weight = weight / max(1.0, math.sqrt(input_dim))
    return torch.tensor(weight, dtype=torch.float32)


def _extract_scene_tracks(ep: Any) -> Optional[SceneTracksLite]:
    payload = None
    if hasattr(ep, "scene_tracks"):
        payload = getattr(ep, "scene_tracks")
    if isinstance(ep, dict):
        payload = ep.get("scene_tracks") or ep.get("scene_tracks_v1") or payload
        if payload is None:
            path = ep.get("scene_tracks_path") or ep.get("scene_tracks_npz")
            if path:
                payload = _load_npz(str(path))
    if payload is None:
        return None
    if isinstance(payload, SceneTracksLite):
        return payload
    if isinstance(payload, dict):
        return deserialize_scene_tracks_v1(_coerce_scene_tracks_payload(payload))
    raise ValueError("Unsupported scene_tracks payload type")


def _coerce_scene_tracks_payload(payload: Dict[str, Any]) -> Dict[str, np.ndarray]:
    coerced: Dict[str, np.ndarray] = {}
    for key, value in payload.items():
        if isinstance(value, np.ndarray):
            coerced[key] = value
        else:
            coerced[key] = np.asarray(value)
    return coerced


def _synthetic_scene_tracks(T: int = 5, K: int = 3) -> SceneTracksLite:
    track_ids = np.array([f"track_{k}" for k in range(K)], dtype="U16")
    entity_types = np.zeros((K,), dtype=np.int32)
    class_ids = np.full((K,), -1, dtype=np.int32)
    poses_R = np.repeat(np.eye(3, dtype=np.float32)[None, None, ...], T, axis=0)
    poses_R = np.repeat(poses_R, K, axis=1)
    poses_t = np.zeros((T, K, 3), dtype=np.float32)
    for k in range(K):
        poses_t[:, k, 0] = np.linspace(0.1 * k, 0.3 * k, T)
        poses_t[:, k, 1] = 0.2 * k
    scales = np.ones((T, K), dtype=np.float32)
    visibility = np.ones((T, K), dtype=np.float32)
    occlusion = np.zeros((T, K), dtype=np.float32)
    ir_loss = np.zeros((T, K), dtype=np.float32)
    converged = np.ones((T, K), dtype=bool)
    return SceneTracksLite(
        track_ids=track_ids,
        entity_types=entity_types,
        class_ids=class_ids,
        poses_R=poses_R,
        poses_t=poses_t,
        scales=scales,
        visibility=visibility,
        occlusion=occlusion,
        ir_loss=ir_loss,
        converged=converged,
    )


def _extract_mhn_features(ep: Any, T: int, K: int) -> Dict[str, Optional[np.ndarray]]:
    track_features = None
    summary = None
    if isinstance(ep, dict):
        track_features = ep.get("mhn_track_features")
        summary = ep.get("mhn_summary", None)
        if summary is None:
            summary = ep.get("mhn_features", None)
    else:
        track_features = getattr(ep, "mhn_track_features", None)
        summary = getattr(ep, "mhn_summary", None)
        if summary is None:
            summary = getattr(ep, "mhn_features", None)

    local_features = None
    if track_features is not None:
        arr = np.asarray(track_features, dtype=np.float32)
        if arr.ndim == 3:
            local = arr
        elif arr.ndim == 2 and arr.shape[0] == K:
            local = np.broadcast_to(arr[None, ...], (T, K, arr.shape[1]))
        elif arr.ndim == 2 and arr.shape[0] == T:
            local = None
            summary = arr
        elif arr.ndim == 1:
            local = None
            summary = arr
        else:
            local = None

        if local is not None:
            if local.shape[0] != T:
                local = np.broadcast_to(local[:1], (T, K, local.shape[2]))
            if local.shape[1] != K:
                local = local[:, :K]
            if local.shape[2] >= 3:
                local_features = local[:, :, :3]
            else:
                pad = np.zeros((local.shape[0], local.shape[1], 3 - local.shape[2]), dtype=np.float32)
                local_features = np.concatenate([local, pad], axis=2)

    global_features = None
    if summary is not None:
        if isinstance(summary, dict):
            summary_obj = MHNSummary.from_dict(summary)
            vec = np.array(
                [
                    float(getattr(summary_obj, "mean_tree_depth", 0.0)),
                    float(getattr(summary_obj, "mean_branch_factor", 0.0)),
                    float(getattr(summary_obj, "plausibility_score", 1.0)),
                ],
                dtype=np.float32,
            )
            global_features = np.broadcast_to(vec, (T, 3)).copy()
        else:
            arr = np.asarray(summary, dtype=np.float32)
            if arr.ndim == 1:
                arr = np.broadcast_to(arr, (T, arr.shape[0]))
            if arr.shape[0] != T:
                arr = np.broadcast_to(arr[:1], (T, arr.shape[-1]))
            if arr.shape[1] >= 3:
                global_features = arr[:, :3]
            else:
                global_features = np.pad(arr, ((0, 0), (0, 3 - arr.shape[1])))

    if global_features is None:
        global_features = np.zeros((T, 3), dtype=np.float32)

    return {"global": global_features, "local": local_features}


def _bev_feature_names() -> List[str]:
    return [
        "occupancy",
        "height_mean",
        "velocity_x",
        "velocity_y",
        "track_confidence",
        "occlusion_mean",
        "mhn_local_depth",
        "mhn_local_branch",
        "mhn_local_plausibility",
        "mhn_global_depth",
        "mhn_global_branch",
        "mhn_global_plausibility",
    ]


def _build_bev_grid(
    scene_tracks: SceneTracksLite,
    episode: Any,
    mhn_features: Dict[str, Optional[np.ndarray]],
    config: GeometryBEVConfig,
) -> Tuple[np.ndarray, Dict[str, float], str]:
    poses_t = np.asarray(scene_tracks.poses_t, dtype=np.float32)
    T, K = poses_t.shape[:2]
    visibility = np.asarray(getattr(scene_tracks, "visibility", np.ones((T, K))), dtype=np.float32)
    occlusion = np.asarray(getattr(scene_tracks, "occlusion", np.zeros((T, K))), dtype=np.float32)
    ir_loss = np.asarray(getattr(scene_tracks, "ir_loss", np.zeros((T, K))), dtype=np.float32)
    converged = np.asarray(getattr(scene_tracks, "converged", np.ones((T, K), dtype=bool)))

    ego_pos, ego_yaw, frame_source = _compute_ego_pose(scene_tracks, episode)
    extent = float(config.extent_m)
    resolution = float(config.resolution_m)
    grid_size = max(1, int(np.ceil(2.0 * extent / resolution)))
    num_features = len(_bev_feature_names())

    grid = np.zeros((T, grid_size, grid_size, num_features), dtype=np.float32)
    occupancy = np.zeros((T, grid_size, grid_size), dtype=np.float32)
    height_sum = np.zeros_like(occupancy)
    vel_x_sum = np.zeros_like(occupancy)
    vel_y_sum = np.zeros_like(occupancy)
    conf_sum = np.zeros_like(occupancy)
    occ_sum = np.zeros_like(occupancy)
    local_mhn_sum = np.zeros((T, grid_size, grid_size, 3), dtype=np.float32)
    local_mhn = mhn_features.get("local")
    global_mhn = mhn_features.get("global")
    if global_mhn is None:
        global_mhn = np.zeros((T, 3), dtype=np.float32)

    prev_pos = None
    track_counts = []
    for t in range(T):
        pos = poses_t[t]
        rel = pos - ego_pos[t][None, :]
        rel = _rotate_xy(rel, -ego_yaw[t])
        if prev_pos is None:
            vel = np.zeros_like(rel)
        else:
            vel = rel - prev_pos
        prev_pos = rel

        conf = visibility[t] * (1.0 - occlusion[t]) * (1.0 / (1.0 + ir_loss[t]))
        conf = conf * converged[t].astype(np.float32)
        track_idx = np.arange(K)
        if K > config.max_tracks:
            order = np.argsort(-conf, kind="mergesort")
            track_idx = order[: config.max_tracks]

        track_counts.append(float(len(track_idx)))
        for k in track_idx:
            x, y, z = rel[k]
            ix = int(np.floor((x + extent) / resolution))
            iy = int(np.floor((y + extent) / resolution))
            if ix < 0 or iy < 0 or ix >= grid_size or iy >= grid_size:
                continue
            occupancy[t, iy, ix] += 1.0
            height_sum[t, iy, ix] += z
            vel_x_sum[t, iy, ix] += vel[k, 0]
            vel_y_sum[t, iy, ix] += vel[k, 1]
            conf_sum[t, iy, ix] += conf[k]
            occ_sum[t, iy, ix] += occlusion[t, k]
            if local_mhn is not None:
                local_mhn_sum[t, iy, ix] += local_mhn[t, k]

        occ_mask = occupancy[t] > 0
        if np.any(occ_mask):
            height_sum[t, occ_mask] /= occupancy[t, occ_mask]
            vel_x_sum[t, occ_mask] /= occupancy[t, occ_mask]
            vel_y_sum[t, occ_mask] /= occupancy[t, occ_mask]
            conf_sum[t, occ_mask] /= occupancy[t, occ_mask]
            occ_sum[t, occ_mask] /= occupancy[t, occ_mask]
            if local_mhn is not None:
                local_mhn_sum[t, occ_mask] /= occupancy[t, occ_mask][..., None]
            grid[t, :, :, 9][occ_mask] = global_mhn[t, 0]
            grid[t, :, :, 10][occ_mask] = global_mhn[t, 1]
            grid[t, :, :, 11][occ_mask] = global_mhn[t, 2]

        grid[t, :, :, 0] = occupancy[t] / max(1.0, float(config.max_tracks))
        grid[t, :, :, 1] = height_sum[t]
        grid[t, :, :, 2] = vel_x_sum[t]
        grid[t, :, :, 3] = vel_y_sum[t]
        grid[t, :, :, 4] = conf_sum[t]
        grid[t, :, :, 5] = occ_sum[t]
        if local_mhn is not None:
            grid[t, :, :, 6:9] = local_mhn_sum[t]

    occupied_ratio = float(np.mean(occupancy > 0))
    stats = {
        "occupied_ratio": occupied_ratio,
        "track_count_mean": float(np.mean(track_counts)) if track_counts else 0.0,
        "track_count_max": float(np.max(track_counts)) if track_counts else 0.0,
        "grid_size": float(grid_size),
    }
    return grid, stats, frame_source


def _compute_ego_pose(scene_tracks: SceneTracksLite, episode: Any) -> Tuple[np.ndarray, np.ndarray, str]:
    poses = None
    if isinstance(episode, dict):
        poses = episode.get("ego_poses") or episode.get("ego_pose")
        if poses is None and isinstance(episode.get("robot_state"), dict):
            poses = episode["robot_state"].get("ee_pose")
    else:
        poses = getattr(episode, "ego_poses", None) or getattr(episode, "ego_pose", None)

    if poses is not None:
        arr = np.asarray(poses, dtype=np.float32)
        if arr.ndim == 2 and arr.shape == (4, 4):
            pos = arr[:3, 3][None, :]
            yaw = np.array([_yaw_from_rot(arr[:3, :3])], dtype=np.float32)
            return pos, yaw, "ego_pose"
        if arr.ndim == 3 and arr.shape[1:] == (4, 4):
            pos = arr[:, :3, 3]
            yaw = np.array([_yaw_from_rot(r[:3, :3]) for r in arr], dtype=np.float32)
            return pos, yaw, "ego_poses"
        if arr.ndim == 2 and arr.shape[1] >= 3:
            pos = arr[:, :3]
            yaw = np.zeros((arr.shape[0],), dtype=np.float32)
            return pos, yaw, "ego_positions"
        if arr.ndim == 1 and arr.shape[0] >= 3:
            pos = arr[:3][None, :]
            yaw = np.zeros((1,), dtype=np.float32)
            return pos, yaw, "ego_position"

    entity_types = np.asarray(getattr(scene_tracks, "entity_types", np.array([], dtype=np.int32)))
    body_indices = np.where(entity_types == 1)[0]
    if body_indices.size > 0:
        visibility = np.asarray(getattr(scene_tracks, "visibility", np.ones((scene_tracks.num_frames, scene_tracks.num_tracks))))
        scores = visibility[:, body_indices].mean(axis=0)
        best = body_indices[int(np.argmax(scores))]
        pos = scene_tracks.poses_t[:, best, :]
        yaw = np.array([_yaw_from_rot(scene_tracks.poses_R[t, best]) for t in range(scene_tracks.num_frames)], dtype=np.float32)
        return pos, yaw, "body_track"

    zeros = np.zeros((scene_tracks.num_frames, 3), dtype=np.float32)
    yaw = np.zeros((scene_tracks.num_frames,), dtype=np.float32)
    return zeros, yaw, "world"


def _yaw_from_rot(R: np.ndarray) -> float:
    return float(np.arctan2(R[1, 0], R[0, 0]))


def _rotate_xy(points: np.ndarray, yaw: float) -> np.ndarray:
    cos_y = float(np.cos(yaw))
    sin_y = float(np.sin(yaw))
    rot = np.array([[cos_y, -sin_y], [sin_y, cos_y]], dtype=np.float32)
    xy = points[:, :2] @ rot.T
    out = points.copy()
    out[:, :2] = xy
    return out


def _rotate_scene_tracks(scene_tracks: SceneTracksLite, yaw_deg: float) -> SceneTracksLite:
    yaw = float(np.deg2rad(yaw_deg))
    cos_y = float(np.cos(yaw))
    sin_y = float(np.sin(yaw))
    rot = np.array([[cos_y, -sin_y, 0.0], [sin_y, cos_y, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    poses_t = np.asarray(scene_tracks.poses_t, dtype=np.float32) @ rot.T
    poses_R = np.asarray(scene_tracks.poses_R, dtype=np.float32)
    poses_R_rot = np.einsum("ij,tkjl->tkil", rot, poses_R)
    return SceneTracksLite(
        track_ids=scene_tracks.track_ids,
        entity_types=scene_tracks.entity_types,
        class_ids=scene_tracks.class_ids,
        poses_R=poses_R_rot.astype(np.float32),
        poses_t=poses_t.astype(np.float32),
        scales=scene_tracks.scales,
        visibility=scene_tracks.visibility,
        occlusion=scene_tracks.occlusion,
        ir_loss=scene_tracks.ir_loss,
        converged=scene_tracks.converged,
    )


def _rotate_episode_ego(episode: Any, yaw_deg: float) -> Any:
    if not isinstance(episode, dict):
        return episode
    yaw = float(np.deg2rad(yaw_deg))
    cos_y = float(np.cos(yaw))
    sin_y = float(np.sin(yaw))
    rot = np.array([[cos_y, -sin_y, 0.0], [sin_y, cos_y, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    updated = dict(episode)
    ego_poses = updated.get("ego_poses")
    if ego_poses is not None:
        arr = np.asarray(ego_poses, dtype=np.float32)
        if arr.ndim == 3 and arr.shape[1:] == (4, 4):
            new = arr.copy()
            new[:, :3, :3] = np.einsum("ij,tjk->tik", rot, arr[:, :3, :3])
            new[:, :3, 3] = (rot @ arr[:, :3, 3].T).T
            updated["ego_poses"] = new
            return updated
    ego_pose = updated.get("ego_pose")
    if ego_pose is not None:
        arr = np.asarray(ego_pose, dtype=np.float32)
        if arr.shape == (4, 4):
            new = arr.copy()
            new[:3, :3] = rot @ arr[:3, :3]
            new[:3, 3] = rot @ arr[:3, 3]
            updated["ego_pose"] = new
            return updated
    ego_positions = updated.get("ego_positions")
    if ego_positions is not None:
        arr = np.asarray(ego_positions, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[1] >= 3:
            arr[:, :3] = (rot @ arr[:, :3].T).T
            updated["ego_positions"] = arr
            return updated
    robot_state = updated.get("robot_state")
    if isinstance(robot_state, dict) and "ee_pose" in robot_state:
        ee_pose = np.asarray(robot_state["ee_pose"], dtype=np.float32)
        if ee_pose.ndim == 1 and ee_pose.shape[0] >= 3:
            ee_pose = ee_pose.copy()
            ee_pose[:3] = rot @ ee_pose[:3]
            robot_state = dict(robot_state)
            robot_state["ee_pose"] = ee_pose
            updated["robot_state"] = robot_state
    return updated


def _check_bev_rotation_consistency(
    scene_tracks: SceneTracksLite,
    episode: Any,
    mhn_features: Dict[str, Optional[np.ndarray]],
    reference_grid: np.ndarray,
    config: GeometryBEVConfig,
) -> None:
    degrees = int(config.rotation_check_degrees)
    if degrees % 90 != 0:
        return
    rot_tracks = _rotate_scene_tracks(scene_tracks, degrees)
    rot_episode = _rotate_episode_ego(episode, degrees)
    rot_grid, _, _ = _build_bev_grid(rot_tracks, rot_episode, mhn_features, config)
    k = (degrees // 90) % 4
    rot_back = np.rot90(rot_grid, k=-k, axes=(1, 2))
    diff = float(np.max(np.abs(reference_grid - rot_back)))
    if diff > float(config.rotation_check_tolerance):
        raise ValueError(f"BEV rotation consistency check failed (max diff {diff:.6f})")


def _resample_bev_grid(grid: np.ndarray, target_len: int) -> np.ndarray:
    if grid.shape[0] == target_len:
        return grid
    grid_t = torch.from_numpy(grid).permute(1, 2, 3, 0).unsqueeze(0)
    resized = F.interpolate(grid_t, size=target_len, mode="nearest")
    resized = resized.squeeze(0).permute(3, 0, 1, 2).contiguous()
    return resized.cpu().numpy()


def _bev_grid_to_token(
    grid: np.ndarray,
    config: GeometryBEVConfig,
    proj: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    T, H, W, C = grid.shape
    patch = max(1, int(config.patch_size))
    pooled = _pool_bev_grid(grid, patch)
    flat = pooled.reshape(T, -1).astype(np.float32)
    if proj is None or proj.shape[0] != flat.shape[1] or proj.shape[1] != config.embed_dim:
        proj = _deterministic_projection(flat.shape[1], int(config.embed_dim), int(config.seed))
    tokens = torch.tensor(flat, dtype=torch.float32) @ proj
    return tokens, proj


def _pool_bev_grid(grid: np.ndarray, patch: int) -> np.ndarray:
    if patch <= 1:
        return grid
    T, H, W, C = grid.shape
    patch = min(patch, H, W)
    pooled_h = max(1, H // patch)
    pooled_w = max(1, W // patch)
    trimmed = grid[:, : pooled_h * patch, : pooled_w * patch, :]
    reshaped = trimmed.reshape(T, pooled_h, patch, pooled_w, patch, C)
    return reshaped.mean(axis=(2, 4))


def _stack_bev_grids(grids: List[np.ndarray], target_len: int) -> torch.Tensor:
    if not grids:
        return torch.zeros((0, target_len, 0, 0, 0), dtype=torch.float32)
    max_h = max(g.shape[1] for g in grids)
    max_w = max(g.shape[2] for g in grids)
    max_c = max(g.shape[3] for g in grids)
    padded = []
    for g in grids:
        if g.shape[0] != target_len:
            g = _resample_bev_grid(g, target_len)
        pad_h = max_h - g.shape[1]
        pad_w = max_w - g.shape[2]
        pad_c = max_c - g.shape[3]
        if pad_h or pad_w or pad_c:
            g = np.pad(g, ((0, 0), (0, pad_h), (0, pad_w), (0, pad_c)), mode="constant")
        padded.append(g)
    stacked = np.stack(padded, axis=0)
    return torch.tensor(stacked, dtype=torch.float32)


def _extract_embodiment_payload(ep: Any) -> Optional[Dict[str, Any]]:
    if hasattr(ep, "embodiment_profile"):
        emb = getattr(ep, "embodiment_profile")
        if emb is not None:
            return _payload_from_embodiment_summary(emb)
    if isinstance(ep, dict):
        if "embodiment_profile" in ep:
            return _payload_from_embodiment_summary(ep["embodiment_profile"])
        if "embodiment_profile_npz" in ep:
            return _payload_from_paths(ep)
    return None


def _payload_from_embodiment_summary(summary: Any) -> Optional[Dict[str, Any]]:
    if summary is None:
        return None
    if isinstance(summary, dict):
        paths = {
            "embodiment_profile_npz": summary.get("embodiment_profile_npz") or summary.get("embodiment_profile_path"),
            "skill_segments_npz": summary.get("skill_segments_npz") or summary.get("skill_segments_path"),
            "cost_summary": summary.get("cost_summary"),
            "value_summary": summary.get("value_summary"),
            "drift_score": summary.get("drift_score"),
            "w_embodiment": summary.get("w_embodiment"),
        }
        return _payload_from_paths(paths)
    # EmbodimentProfileSummary dataclass
    paths = {
        "embodiment_profile_npz": getattr(summary, "embodiment_profile_npz", None),
        "skill_segments_npz": getattr(summary, "skill_segments_npz", None),
        "cost_summary": getattr(summary, "cost_summary", None),
        "value_summary": getattr(summary, "value_summary", None),
        "drift_score": getattr(summary, "drift_score", None),
        "w_embodiment": getattr(summary, "w_embodiment", None),
    }
    return _payload_from_paths(paths)


def _payload_from_paths(paths: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    profile_path = paths.get("embodiment_profile_npz")
    if not profile_path:
        return None
    payload = {
        "profile": _load_npz(profile_path),
        "segments": _safe_load_npz(paths.get("skill_segments_npz")),
        "cost_summary": paths.get("cost_summary"),
        "value_summary": paths.get("value_summary"),
        "drift_score": paths.get("drift_score"),
        "w_embodiment": paths.get("w_embodiment"),
    }
    return payload


def _load_npz(path: str) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    return {k: data[k] for k in data.files}


def _safe_load_npz(path: Optional[str]) -> Optional[Dict[str, np.ndarray]]:
    if not path:
        return None
    try:
        return _load_npz(path)
    except Exception:
        return None


def _embodiment_features(payload: Dict[str, Any]) -> Tuple[np.ndarray, List[str]]:
    profile = payload.get("profile") or {}
    contact_matrix = profile.get(f"{EMBODIMENT_PROFILE_PREFIX}contact_matrix")
    contact_conf = profile.get(f"{EMBODIMENT_PROFILE_PREFIX}contact_confidence")
    contact_impossible = profile.get(f"{EMBODIMENT_PROFILE_PREFIX}contact_impossible")
    visibility = profile.get(f"{EMBODIMENT_PROFILE_PREFIX}visibility")
    occlusion = profile.get(f"{EMBODIMENT_PROFILE_PREFIX}occlusion")
    contact_distance = profile.get(f"{EMBODIMENT_PROFILE_PREFIX}contact_distance")

    if contact_matrix is None:
        raise ValueError("Embodiment profile missing contact_matrix")

    T, K = contact_matrix.shape[:2]
    denom = max(1.0, float(K * max(K - 1, 1)))

    contact_count = contact_matrix.reshape(T, -1).sum(axis=1) / denom
    contact_conf_mean = _masked_mean(contact_conf, contact_matrix)
    impossible_count = contact_impossible.reshape(T, -1).sum(axis=1) / denom if contact_impossible is not None else np.zeros(T)
    visibility_mean = visibility.mean(axis=1) if visibility is not None else np.ones(T)
    occlusion_mean = occlusion.mean(axis=1) if occlusion is not None else np.zeros(T)
    distance_mean = contact_distance.mean(axis=(1, 2)) if contact_distance is not None else np.zeros(T)

    segment_id, segment_type, segment_conf, segment_energy, segment_risk, segment_success = _segment_features(
        payload.get("segments"), T
    )

    drift_score = float(payload.get("drift_score", 0.0) or 0.0)
    w_embodiment = float(payload.get("w_embodiment", 1.0) or 1.0)

    cost_summary = payload.get("cost_summary")
    value_summary = payload.get("value_summary")
    cost_total = _sum_numeric(cost_summary)
    value_total = _sum_numeric(value_summary)

    features = np.stack(
        [
            contact_count,
            contact_conf_mean,
            visibility_mean,
            occlusion_mean,
            impossible_count,
            distance_mean,
            segment_id,
            segment_type,
            segment_conf,
            segment_energy,
            segment_risk,
            segment_success,
            np.full(T, drift_score),
            np.full(T, w_embodiment),
            np.full(T, cost_total),
            np.full(T, value_total),
        ],
        axis=1,
    ).astype(np.float32)

    names = [
        "contact_rate",
        "contact_conf_mean",
        "visibility_mean",
        "occlusion_mean",
        "impossible_rate",
        "contact_distance_mean",
        "segment_id",
        "segment_type",
        "segment_conf",
        "segment_energy",
        "segment_risk",
        "segment_success",
        "drift_score",
        "w_embodiment",
        "cost_total",
        "value_total",
    ]
    return features, names


def _segment_features(segments: Optional[Dict[str, Any]], T: int) -> Tuple[np.ndarray, ...]:
    if not segments:
        zeros = np.zeros(T, dtype=np.float32)
        return (
            np.full(T, -1, dtype=np.float32),
            np.full(T, -1, dtype=np.float32),
            zeros,
            zeros,
            zeros,
            zeros,
        )
    bounds = segments.get(f"{SKILL_SEGMENTS_PREFIX}segment_bounds")
    seg_type = segments.get(f"{SKILL_SEGMENTS_PREFIX}segment_type")
    seg_conf = segments.get(f"{SKILL_SEGMENTS_PREFIX}segment_confidence")
    seg_energy = segments.get(f"{SKILL_SEGMENTS_PREFIX}segment_energy_Wh")
    seg_risk = segments.get(f"{SKILL_SEGMENTS_PREFIX}segment_risk")
    seg_success = segments.get(f"{SKILL_SEGMENTS_PREFIX}segment_success")

    if bounds is None:
        zeros = np.zeros(T, dtype=np.float32)
        return (
            np.full(T, -1, dtype=np.float32),
            np.full(T, -1, dtype=np.float32),
            zeros,
            zeros,
            zeros,
            zeros,
        )

    segment_id = np.full(T, -1, dtype=np.float32)
    segment_type = np.full(T, -1, dtype=np.float32)
    segment_conf = np.zeros(T, dtype=np.float32)
    segment_energy = np.zeros(T, dtype=np.float32)
    segment_risk = np.zeros(T, dtype=np.float32)
    segment_success = np.zeros(T, dtype=np.float32)

    for idx, (start, end) in enumerate(bounds):
        start_i = max(0, int(start))
        end_i = min(T - 1, int(end))
        if end_i < start_i:
            continue
        segment_id[start_i : end_i + 1] = float(idx)
        segment_type[start_i : end_i + 1] = float(seg_type[idx]) if seg_type is not None else 0.0
        if seg_conf is not None:
            segment_conf[start_i : end_i + 1] = float(seg_conf[idx])
        if seg_energy is not None:
            segment_energy[start_i : end_i + 1] = float(seg_energy[idx])
        if seg_risk is not None:
            segment_risk[start_i : end_i + 1] = float(seg_risk[idx])
        if seg_success is not None:
            segment_success[start_i : end_i + 1] = float(seg_success[idx])

    return segment_id, segment_type, segment_conf, segment_energy, segment_risk, segment_success


def _masked_mean(values: Optional[np.ndarray], mask: np.ndarray) -> np.ndarray:
    if values is None:
        return np.zeros(mask.shape[0], dtype=np.float32)
    flat = values.reshape(mask.shape[0], -1)
    mask_flat = mask.reshape(mask.shape[0], -1).astype(bool)
    means = []
    for t in range(mask.shape[0]):
        if mask_flat[t].any():
            means.append(float(np.mean(flat[t][mask_flat[t]])))
        else:
            means.append(0.0)
    return np.array(means, dtype=np.float32)


def _sum_numeric(obj: Any) -> float:
    if not isinstance(obj, dict):
        return 0.0
    total = 0.0
    for val in obj.values():
        if isinstance(val, (int, float)):
            total += float(val)
        elif isinstance(val, dict):
            total += _sum_numeric(val)
    return total


def _synthetic_embodiment_payload() -> Dict[str, Any]:
    T = 8
    K = 2
    contact_matrix = np.zeros((T, K, K), dtype=bool)
    contact_matrix[:, 0, 1] = True
    contact_conf = np.ones((T, K, K), dtype=np.float32) * 0.8
    contact_impossible = np.zeros((T, K, K), dtype=bool)
    visibility = np.ones((T, K), dtype=np.float32)
    occlusion = np.zeros((T, K), dtype=np.float32)
    profile = {
        f"{EMBODIMENT_PROFILE_PREFIX}contact_matrix": contact_matrix,
        f"{EMBODIMENT_PROFILE_PREFIX}contact_confidence": contact_conf,
        f"{EMBODIMENT_PROFILE_PREFIX}contact_impossible": contact_impossible,
        f"{EMBODIMENT_PROFILE_PREFIX}visibility": visibility,
        f"{EMBODIMENT_PROFILE_PREFIX}occlusion": occlusion,
    }
    segments = {
        f"{SKILL_SEGMENTS_PREFIX}segment_bounds": np.array([[0, T - 1]], dtype=np.int32),
        f"{SKILL_SEGMENTS_PREFIX}segment_type": np.array([1], dtype=np.int32),
        f"{SKILL_SEGMENTS_PREFIX}segment_confidence": np.array([0.9], dtype=np.float32),
        f"{SKILL_SEGMENTS_PREFIX}segment_energy_Wh": np.array([0.1], dtype=np.float32),
        f"{SKILL_SEGMENTS_PREFIX}segment_risk": np.array([0.0], dtype=np.float32),
        f"{SKILL_SEGMENTS_PREFIX}segment_success": np.array([1.0], dtype=np.float32),
    }
    return {
        "profile": profile,
        "segments": segments,
        "cost_summary": {"total_Wh": 0.1},
        "value_summary": {"delta_mpl": 0.0},
        "drift_score": 0.0,
        "w_embodiment": 1.0,
    }


def _extract_scene_graphs(ep: Any) -> List[Any]:
    if hasattr(ep, "scene_graphs"):
        graphs = getattr(ep, "scene_graphs")
        if graphs:
            return list(graphs)
    if hasattr(ep, "scene_graph"):
        graph = getattr(ep, "scene_graph")
        if graph is not None:
            return [graph]
    if isinstance(ep, dict):
        if "scene_graphs" in ep and ep["scene_graphs"] is not None:
            return list(ep["scene_graphs"])
        if "scene_graph" in ep and ep["scene_graph"] is not None:
            return [ep["scene_graph"]]
        if "scene_graph_tensors" in ep and ep["scene_graph_tensors"] is not None:
            payload = ep["scene_graph_tensors"]
            if isinstance(payload, dict):
                return [payload]
            return list(payload)
    return []


def _scene_graph_to_tensors(graph: Any, pos_dim: int, device: Optional[torch.device]) -> Dict[str, torch.Tensor]:
    if isinstance(graph, dict) and "node_features" in graph:
        tensors = {
            key: torch.as_tensor(val) if not isinstance(val, torch.Tensor) else val
            for key, val in graph.items()
        }
    else:
        tensors = ordered_scene_tensors(graph, pos_dim=pos_dim, device=device)
    if tensors["node_features"].dim() == 2:
        return tensors
    return {k: (v.squeeze(0) if isinstance(v, torch.Tensor) and v.dim() > 2 else v) for k, v in tensors.items()}


def _extract_rgb_frames(ep: Any) -> Optional[np.ndarray]:
    if isinstance(ep, dict):
        for key in ("vision_rgb", "rgb_frames", "vision_rgb_frames"):
            if key in ep and ep[key] is not None:
                return np.asarray(ep[key])
    if hasattr(ep, "rgb_frames"):
        frames = getattr(ep, "rgb_frames")
        if frames is not None:
            return np.asarray(frames)
    return None


def _extract_rgb_features(ep: Any) -> Optional[Dict[str, Any]]:
    if isinstance(ep, dict):
        for key in ("rgb_features_v1", "vision_rgb_features", "rgb_features"):
            if key in ep and ep[key] is not None:
                payload = ep[key]
                if isinstance(payload, dict):
                    return payload
    if hasattr(ep, "rgb_features_v1"):
        payload = getattr(ep, "rgb_features_v1")
        if isinstance(payload, dict):
            return payload
    return None


def _proxy_render_from_geometry(ep: Any, height: int = 64, width: int = 64) -> Optional[np.ndarray]:
    graph_payload = None
    if isinstance(ep, dict):
        graph_payload = ep.get("scene_graph") or ep.get("geometry_scene_graph")
    if graph_payload is None:
        return None
    seed = _hash_seed(str(graph_payload))
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(1, height, width, 3), dtype=np.uint8)


def _synthetic_rgb_frames(ep: Any, height: int = 64, width: int = 64) -> np.ndarray:
    seed = _hash_seed(str(getattr(ep, "episode_id", None) or ep))
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(1, height, width, 3), dtype=np.uint8)


def _hash_seed(payload: str) -> int:
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _rgb_frames_to_tokens(frames: np.ndarray, pool_size: Tuple[int, int], proj: torch.Tensor) -> torch.Tensor:
    arr = np.asarray(frames)
    if arr.ndim == 4:
        arr = arr[None, ...]
    if arr.shape[-1] == 3:
        arr = np.transpose(arr, (0, 1, 4, 2, 3))
    tensor = torch.tensor(arr, dtype=torch.float32)
    if tensor.max() > 1.5:
        tensor = tensor / 255.0
    B, T, C, H, W = tensor.shape
    flat = tensor.reshape(B * T, C, H, W)
    pooled = F.adaptive_avg_pool2d(flat, pool_size)
    mean = flat.mean(dim=(2, 3))
    std = flat.std(dim=(2, 3), unbiased=False)
    feat = torch.cat([pooled.flatten(1), mean, std], dim=1)
    tokens = feat @ proj
    return tokens.reshape(B, T, -1).squeeze(0)


def _rgb_features_to_tokens(payload: Dict[str, Any]) -> torch.Tensor:
    features_temporal = payload.get("features_temporal")
    if features_temporal is not None:
        arr = np.asarray(features_temporal, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, ...]
        return torch.tensor(arr, dtype=torch.float32)
    features = payload.get("features")
    if features is None:
        raise ValueError("rgb_features_v1 missing features")
    arr = np.asarray(features, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, ...]
    return torch.tensor(arr, dtype=torch.float32)


def _extract_gaussian_scenes(ep: Any) -> List[GaussianScene]:
    if hasattr(ep, "gaussian_scene"):
        scene = getattr(ep, "gaussian_scene")
        if scene is not None:
            return [scene]
    if hasattr(ep, "gaussian_scenes"):
        scenes = getattr(ep, "gaussian_scenes")
        if scenes:
            return list(scenes)
    if isinstance(ep, dict):
        scene = ep.get("gaussian_scene")
        if scene is not None:
            return [scene]
        scenes = ep.get("gaussian_scenes")
        if scenes:
            return list(scenes)
    return []


def _gaussian_scene_features(scene: GaussianScene) -> np.ndarray:
    means = scene.means
    colors = scene.colors
    opacities = scene.opacities
    normals = scene.normals
    n_gaussians = float(scene.num_gaussians)

    mean_pos = means.mean(axis=0) if means.size else np.zeros(3)
    std_pos = means.std(axis=0) if means.size else np.zeros(3)
    mean_color = colors.mean(axis=0) if colors.size else np.zeros(3)
    mean_opacity = np.array([opacities.mean()]) if opacities.size else np.zeros(1)
    mean_normal = normals.mean(axis=0) if normals.size else np.zeros(3)

    return np.concatenate(
        [
            np.array([n_gaussians], dtype=np.float32),
            mean_pos.astype(np.float32),
            std_pos.astype(np.float32),
            mean_color.astype(np.float32),
            mean_opacity.astype(np.float32),
            mean_normal.astype(np.float32),
        ]
    ).astype(np.float32)


__all__ = [
    "TokenProviderOutput",
    "BaseTokenProvider",
    "EmbodimentTokenProvider",
    "SceneGraphTokenProvider",
    "RGBVisionTokenProvider",
    "GaussianSceneTokenProvider",
]
