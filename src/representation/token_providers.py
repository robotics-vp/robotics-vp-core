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
