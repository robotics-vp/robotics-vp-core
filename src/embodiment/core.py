"""Core embodiment computation from scene tracks and semantic fusion.

Embodiment outputs are advisory: used for datapack enrichment and optional
sampling/quality weighting, with no reward math changes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from src.embodiment.artifacts import (
    AffordanceGraphArtifact,
    EmbodimentProfileArtifact,
    EmbodimentSummary,
    SkillSegmentsArtifact,
)
from src.embodiment.config import EmbodimentConfig
from src.orchestrator.semantic_fusion import SEMANTIC_FUSION_PREFIX, SemanticFusionResult
from src.vision.scene_ir_tracker.serialization import SceneTracksLite, deserialize_scene_tracks_v1


TOOL_LABEL_KEYWORDS = ("hand", "gripper", "tool", "arm", "finger", "end_effector", "effector")


@dataclass
class EmbodimentResult:
    profile: EmbodimentProfileArtifact
    affordance_graph: AffordanceGraphArtifact
    skill_segments: SkillSegmentsArtifact
    cost_breakdown: Dict[str, Any]
    value_attribution: Dict[str, Any]
    drift_report: Dict[str, Any]
    calibration_targets: Dict[str, Any]
    summary: EmbodimentSummary
    w_embodiment: float
    trust_override_candidate: bool


@dataclass
class EmbodimentInputs:
    scene_tracks: Any
    semantic_fusion: Optional[Any] = None
    mhn_summary: Optional[Any] = None
    process_reward: Optional[Any] = None
    action_stream: Optional[Sequence[Any]] = None
    joint_state: Optional[Sequence[Any]] = None
    task_constraints: Optional[Dict[str, Any]] = None
    backend_tags: Optional[Dict[str, Any]] = None
    failure_events: Optional[Dict[str, Any]] = None
    episode_metrics: Optional[Dict[str, Any]] = None
    econ_attribution: Optional[Dict[str, float]] = None


def compute_embodiment(
    inputs: EmbodimentInputs,
    config: Optional[EmbodimentConfig] = None,
) -> EmbodimentResult:
    cfg = config or EmbodimentConfig()

    scene_tracks = _ensure_scene_tracks_lite(inputs.scene_tracks)
    poses_t = np.asarray(scene_tracks.poses_t, dtype=np.float32)
    T, K = poses_t.shape[:2]

    visibility = _ensure_array(getattr(scene_tracks, "visibility", None), (T, K), fill=1.0)
    occlusion = _ensure_array(getattr(scene_tracks, "occlusion", None), (T, K), fill=0.0)

    scales = getattr(scene_tracks, "scales", None)
    scale_mean = _safe_scale_mean(scales, K, cfg.contact_base_radius)
    contact_threshold = cfg.contact_base_radius + cfg.contact_scale_factor * (
        scale_mean[:, None] + scale_mean[None, :]
    )

    dist = _pairwise_distance(poses_t)
    contact_candidate = dist <= contact_threshold[None, ...]
    contact_candidate = _clear_diagonal(contact_candidate)

    observed = (visibility >= cfg.visibility_threshold) & (occlusion <= cfg.occlusion_threshold)
    contact_matrix = contact_candidate & observed[:, :, None] & observed[:, None, :]

    contact_confidence = _contact_confidence(dist, contact_threshold, visibility, occlusion)
    contact_confidence = contact_confidence * contact_candidate.astype(np.float32)

    impossible_mask = _impossible_contact_mask(
        contact_candidate,
        visibility,
        occlusion,
        cfg,
    )

    contact_counts = contact_matrix.sum(axis=0).astype(np.float32)
    contact_any = np.any(contact_matrix, axis=(1, 2))
    contact_coverage = float(np.mean(contact_any)) if contact_any.size else 0.0

    track_labels, _semantic_label_ids, semantic_confidence_mean = _infer_semantics(
        inputs.semantic_fusion,
        scene_tracks,
    )

    affordance_graph = _build_affordance_graph(
        track_ids=scene_tracks.track_ids,
        track_labels=track_labels,
        track_class_ids=scene_tracks.class_ids,
        contact_matrix=contact_matrix,
        contact_confidence=contact_confidence,
        cfg=cfg,
    )

    segments = _segment_contacts(contact_matrix, contact_confidence, cfg)
    segment_bounds, segment_types, segment_pairs = segments

    energy_per_step, time_step_s = _estimate_energy_per_step(
        inputs.action_stream,
        inputs.joint_state,
        inputs.episode_metrics,
        inputs.task_constraints,
        T,
    )

    segment_energy = _segment_energy(segment_bounds, energy_per_step)
    segment_risk = _segment_risk(segment_bounds, impossible_mask, inputs.failure_events)
    segment_success = _segment_success(segment_bounds, inputs.process_reward, T)

    skill_segments = SkillSegmentsArtifact(
        segment_bounds=segment_bounds,
        segment_type=segment_types,
        segment_confidence=_segment_confidence(segment_bounds, contact_matrix, contact_confidence),
        segment_contact_pairs=segment_pairs,
        segment_energy_Wh=segment_energy,
        segment_risk=segment_risk,
        segment_success=segment_success,
        segment_labels=_segment_labels(segment_types),
    )

    cost_breakdown = _build_cost_breakdown(
        segment_bounds=segment_bounds,
        segment_energy=segment_energy,
        segment_risk=segment_risk,
        segment_pairs=segment_pairs,
        time_step_s=time_step_s,
        impossible_mask=impossible_mask,
    )

    value_attribution = _build_value_attribution(
        segment_bounds=segment_bounds,
        segment_confidence=skill_segments.segment_confidence,
        segment_pairs=segment_pairs,
        econ_attribution=inputs.econ_attribution,
        track_labels=track_labels,
    )

    drift_report = _build_drift_report(
        contact_matrix=contact_matrix,
        track_labels=track_labels,
        task_constraints=inputs.task_constraints,
        backend_tags=inputs.backend_tags,
    )

    calibration_targets = _build_calibration_targets(
        contact_coverage=contact_coverage,
        drift_report=drift_report,
        failure_events=inputs.failure_events,
        task_constraints=inputs.task_constraints,
        cfg=cfg,
    )

    impossible_contacts = int(impossible_mask.sum())
    drift_score = float(drift_report.get("drift_score", 0.0))

    mhn_plausibility = _extract_mhn_plausibility(inputs.mhn_summary)

    w_embodiment = _compute_w_embodiment(
        semantic_confidence_mean=semantic_confidence_mean,
        contact_coverage=contact_coverage,
        mhn_plausibility=mhn_plausibility,
        drift_score=drift_score,
        impossible_contacts=impossible_contacts,
        total_contact_events=int(contact_candidate.sum()),
        cfg=cfg,
    )

    trust_override = bool(
        drift_score >= cfg.trust_override_drift_threshold
        or impossible_contacts >= cfg.trust_override_impossible_contacts
    )

    missing_inputs = _missing_inputs(inputs)

    summary = EmbodimentSummary(
        w_embodiment=w_embodiment,
        embodiment_quality_score=w_embodiment,
        contact_coverage_pct=contact_coverage,
        semantic_confidence_mean=semantic_confidence_mean,
        physically_impossible_contacts=impossible_contacts,
        drift_score=drift_score,
        trust_override_candidate=trust_override,
        missing_inputs=missing_inputs,
        diagnostics={
            "contact_event_count": int(contact_candidate.sum()),
            "contact_pair_count": int(np.sum(contact_counts > 0)),
            "mhn_plausibility": float(mhn_plausibility),
        },
    )

    profile = EmbodimentProfileArtifact(
        contact_matrix=contact_matrix,
        contact_confidence=contact_confidence,
        contact_impossible=impossible_mask,
        track_ids=np.asarray(scene_tracks.track_ids),
        track_class_ids=np.asarray(scene_tracks.class_ids),
        track_labels=track_labels,
        contact_distance=dist,
        visibility=visibility,
        occlusion=occlusion,
        contact_counts=contact_counts,
    )

    return EmbodimentResult(
        profile=profile,
        affordance_graph=affordance_graph,
        skill_segments=skill_segments,
        cost_breakdown=cost_breakdown,
        value_attribution=value_attribution,
        drift_report=drift_report,
        calibration_targets=calibration_targets,
        summary=summary,
        w_embodiment=w_embodiment,
        trust_override_candidate=trust_override,
    )


def _ensure_scene_tracks_lite(scene_tracks: Any) -> SceneTracksLite:
    if isinstance(scene_tracks, SceneTracksLite):
        return scene_tracks
    if isinstance(scene_tracks, dict):
        return deserialize_scene_tracks_v1(scene_tracks)
    if hasattr(scene_tracks, "poses_t"):
        return scene_tracks  # type: ignore[return-value]
    raise ValueError("Unsupported scene_tracks input; expected SceneTracksLite or dict")


def _ensure_array(value: Optional[np.ndarray], shape: Tuple[int, int], fill: float) -> np.ndarray:
    if value is None:
        return np.full(shape, fill, dtype=np.float32)
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape != shape:
        try:
            return np.broadcast_to(arr, shape).astype(np.float32)
        except Exception:
            return np.full(shape, fill, dtype=np.float32)
    return arr


def _safe_scale_mean(scales: Optional[np.ndarray], k: int, default_scale: float) -> np.ndarray:
    if scales is None:
        return np.full((k,), default_scale, dtype=np.float32)
    arr = np.asarray(scales, dtype=np.float32)
    if arr.ndim == 1 and arr.shape[0] == k:
        mean = arr
    else:
        mean = np.nanmean(arr, axis=0) if arr.size else np.zeros((k,), dtype=np.float32)
    mean = np.nan_to_num(mean, nan=default_scale)
    mean = np.clip(mean, 1e-4, None)
    return mean.astype(np.float32)


def _pairwise_distance(poses_t: np.ndarray) -> np.ndarray:
    diff = poses_t[:, :, None, :] - poses_t[:, None, :, :]
    return np.linalg.norm(diff, axis=-1)


def _clear_diagonal(mask: np.ndarray) -> np.ndarray:
    if mask.ndim < 3:
        return mask
    diag = np.eye(mask.shape[1], dtype=bool)[None, ...]
    return np.where(diag, False, mask)


def _contact_confidence(
    dist: np.ndarray,
    threshold: np.ndarray,
    visibility: np.ndarray,
    occlusion: np.ndarray,
) -> np.ndarray:
    denom = threshold[None, ...] + 1e-6
    base = 1.0 - (dist / denom)
    base = np.clip(base, 0.0, 1.0)
    vis_score = np.minimum(visibility[:, :, None], visibility[:, None, :])
    occ_score = 1.0 - np.maximum(occlusion[:, :, None], occlusion[:, None, :])
    conf = base * np.clip(vis_score, 0.0, 1.0) * np.clip(occ_score, 0.0, 1.0)
    return conf.astype(np.float32)


def _impossible_contact_mask(
    contact_candidate: np.ndarray,
    visibility: np.ndarray,
    occlusion: np.ndarray,
    cfg: EmbodimentConfig,
) -> np.ndarray:
    low_vis = visibility <= cfg.low_visibility_threshold
    high_occ = occlusion >= cfg.impossible_occlusion_threshold
    bad_obs = low_vis | high_occ
    impossible = contact_candidate & bad_obs[:, :, None] & bad_obs[:, None, :]
    return impossible.astype(bool)


def _extract_semantic_fusion_arrays(
    semantic_fusion: Any,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if semantic_fusion is None:
        return None, None
    if isinstance(semantic_fusion, SemanticFusionResult):
        return semantic_fusion.fused_class_probs, semantic_fusion.fused_confidence
    if isinstance(semantic_fusion, dict):
        probs = semantic_fusion.get(f"{SEMANTIC_FUSION_PREFIX}fused_class_probs")
        conf = semantic_fusion.get(f"{SEMANTIC_FUSION_PREFIX}fused_confidence")
        if probs is None and "fused_class_probs" in semantic_fusion:
            probs = semantic_fusion.get("fused_class_probs")
        if conf is None and "fused_confidence" in semantic_fusion:
            conf = semantic_fusion.get("fused_confidence")
        if probs is not None:
            probs = np.asarray(probs, dtype=np.float32)
        if conf is not None:
            conf = np.asarray(conf, dtype=np.float32)
        return probs, conf
    return None, None


def _infer_semantics(
    semantic_fusion: Any,
    scene_tracks: SceneTracksLite,
) -> Tuple[np.ndarray, Optional[np.ndarray], float]:
    probs, conf = _extract_semantic_fusion_arrays(semantic_fusion)
    if probs is None:
        labels = _fallback_labels(scene_tracks)
        return labels, None, 0.2

    probs = np.asarray(probs, dtype=np.float32)
    conf = np.asarray(conf, dtype=np.float32) if conf is not None else None
    if probs.ndim == 2:
        probs = probs[None, ...]
    mean_probs = np.mean(probs, axis=0)
    label_ids = np.argmax(mean_probs, axis=-1)
    labels = _labels_from_scene_tracks(scene_tracks, label_ids)
    if conf is not None and conf.size:
        sem_conf = float(np.mean(conf))
    else:
        sem_conf = float(np.mean(np.max(mean_probs, axis=-1)))
    return labels, label_ids.astype(np.int32), sem_conf


def _labels_from_scene_tracks(scene_tracks: SceneTracksLite, label_ids: np.ndarray) -> np.ndarray:
    class_names = getattr(scene_tracks, "class_names", None)
    class_ids = np.asarray(scene_tracks.class_ids)
    labels: List[str] = []
    for idx, class_id in enumerate(class_ids):
        if class_names and 0 <= int(class_id) < len(class_names):
            labels.append(str(class_names[int(class_id)]))
        elif label_ids is not None and idx < len(label_ids):
            labels.append(f"class_{int(label_ids[idx])}")
        else:
            labels.append(f"track_{idx}")
    return np.array(labels, dtype="U64")


def _fallback_labels(scene_tracks: SceneTracksLite) -> np.ndarray:
    class_names = getattr(scene_tracks, "class_names", None)
    class_ids = np.asarray(scene_tracks.class_ids)
    labels: List[str] = []
    for idx, class_id in enumerate(class_ids):
        if class_names and 0 <= int(class_id) < len(class_names):
            labels.append(str(class_names[int(class_id)]))
        else:
            labels.append(f"track_{idx}")
    return np.array(labels, dtype="U64")


def _build_affordance_graph(
    *,
    track_ids: np.ndarray,
    track_labels: np.ndarray,
    track_class_ids: np.ndarray,
    contact_matrix: np.ndarray,
    contact_confidence: np.ndarray,
    cfg: EmbodimentConfig,
) -> AffordanceGraphArtifact:
    T, K = contact_matrix.shape[:2]
    edges: List[Tuple[int, int]] = []
    edge_types: List[int] = []
    edge_conf: List[float] = []
    edge_support: List[float] = []

    for i in range(K):
        for j in range(i + 1, K):
            freq = float(np.mean(contact_matrix[:, i, j])) if T > 0 else 0.0
            if freq < cfg.min_contact_frequency:
                continue
            conf = float(np.mean(contact_confidence[:, i, j]))
            edges.append((i, j))
            edge_conf.append(conf)
            edge_support.append(freq)
            edge_types.append(1 if _is_tool_label(track_labels[i]) or _is_tool_label(track_labels[j]) else 0)

    edge_index = np.array(edges, dtype=np.int32) if edges else np.zeros((0, 2), dtype=np.int32)
    edge_type = np.array(edge_types, dtype=np.int32) if edge_types else np.zeros((0,), dtype=np.int32)
    edge_confidence = np.array(edge_conf, dtype=np.float32) if edge_conf else np.zeros((0,), dtype=np.float32)
    edge_support_arr = np.array(edge_support, dtype=np.float32) if edge_support else np.zeros((0,), dtype=np.float32)

    return AffordanceGraphArtifact(
        node_ids=np.asarray(track_ids),
        edge_index=edge_index,
        edge_type=edge_type,
        edge_confidence=edge_confidence,
        edge_support=edge_support_arr,
        node_class_ids=np.asarray(track_class_ids),
        node_labels=np.asarray(track_labels),
    )


def _segment_contacts(
    contact_matrix: np.ndarray,
    contact_confidence: np.ndarray,
    cfg: EmbodimentConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    T, K = contact_matrix.shape[:2]
    if T == 0 or K == 0:
        return (
            np.zeros((0, 2), dtype=np.int32),
            np.zeros((0,), dtype=np.int32),
            np.zeros((0, 2), dtype=np.int32),
        )

    conf = contact_confidence.copy()
    diag = np.eye(K, dtype=bool)[None, ...]
    conf = np.where(diag, -1.0, conf)
    flat_conf = conf.reshape(T, K * K)
    primary_idx = np.argmax(flat_conf, axis=1)
    primary_i = primary_idx // K
    primary_j = primary_idx % K
    has_contact = np.any(contact_matrix, axis=(1, 2))
    primary_i = np.where(has_contact, primary_i, -1)
    primary_j = np.where(has_contact, primary_j, -1)

    segments: List[Tuple[int, int, int, int, int]] = []
    start = 0
    current_state = bool(has_contact[0])
    current_pair = (int(primary_i[0]), int(primary_j[0]))

    for t in range(1, T):
        state = bool(has_contact[t])
        pair = (int(primary_i[t]), int(primary_j[t]))
        if state != current_state or pair != current_pair:
            segments.append((start, t - 1, 1 if current_state else 0, current_pair[0], current_pair[1]))
            start = t
            current_state = state
            current_pair = pair

    segments.append((start, T - 1, 1 if current_state else 0, current_pair[0], current_pair[1]))

    segments = _merge_short_segments(segments, cfg.segment_min_length)

    bounds = np.array([[s[0], s[1]] for s in segments], dtype=np.int32)
    seg_type = np.array([s[2] for s in segments], dtype=np.int32)
    seg_pairs = np.array([[s[3], s[4]] for s in segments], dtype=np.int32)
    return bounds, seg_type, seg_pairs


def _merge_short_segments(
    segments: List[Tuple[int, int, int, int, int]],
    min_length: int,
) -> List[Tuple[int, int, int, int, int]]:
    if not segments or min_length <= 1:
        return segments
    merged: List[Tuple[int, int, int, int, int]] = []
    for seg in segments:
        start, end, seg_type, i, j = seg
        if end - start + 1 < min_length and merged:
            prev = merged.pop()
            merged.append((prev[0], end, prev[2], prev[3], prev[4]))
        else:
            merged.append(seg)
    return merged


def _segment_energy(segment_bounds: np.ndarray, energy_per_step: float) -> np.ndarray:
    energies = []
    for start, end in segment_bounds:
        duration = int(end - start + 1)
        energies.append(max(0.0, energy_per_step * duration))
    return np.asarray(energies, dtype=np.float32)


def _segment_risk(
    segment_bounds: np.ndarray,
    impossible_mask: np.ndarray,
    failure_events: Optional[Dict[str, Any]],
) -> np.ndarray:
    risks = []
    for start, end in segment_bounds:
        if end < start:
            risks.append(0.0)
            continue
        window = impossible_mask[start : end + 1]
        ratio = float(np.mean(window.astype(np.float32))) if window.size else 0.0
        failure_penalty = 0.0
        if failure_events:
            failure_penalty = 0.1 * float(failure_events.get("safety_clamps", 0))
            failure_penalty += 0.1 * float(failure_events.get("resets", 0))
        risks.append(min(1.0, ratio + failure_penalty))
    return np.asarray(risks, dtype=np.float32)


def _segment_success(
    segment_bounds: np.ndarray,
    process_reward: Optional[Any],
    num_frames: int,
) -> np.ndarray:
    success = []
    conf_arr = _extract_process_reward_conf(process_reward, num_frames)
    reward_arr = _extract_process_reward_rshape(process_reward, num_frames)
    for start, end in segment_bounds:
        if end < start:
            success.append(0.0)
            continue
        conf = float(np.mean(conf_arr[start : end + 1])) if conf_arr is not None else 0.5
        reward = float(np.mean(reward_arr[start : end + 1])) if reward_arr is not None else 0.0
        score = np.clip(conf + max(0.0, reward), 0.0, 1.0)
        success.append(float(score))
    return np.asarray(success, dtype=np.float32)


def _segment_confidence(
    segment_bounds: np.ndarray,
    contact_matrix: np.ndarray,
    contact_confidence: np.ndarray,
) -> np.ndarray:
    confs = []
    for start, end in segment_bounds:
        if end < start:
            confs.append(0.0)
            continue
        window_contacts = contact_matrix[start : end + 1]
        window_conf = contact_confidence[start : end + 1]
        if window_contacts.size == 0:
            confs.append(0.0)
            continue
        if np.any(window_contacts):
            confs.append(float(np.mean(window_conf[window_contacts])))
        else:
            confs.append(float(np.mean(window_conf)))
    return np.asarray(confs, dtype=np.float32)


def _segment_labels(segment_types: np.ndarray) -> np.ndarray:
    labels = ["contact" if t == 1 else "free" for t in segment_types.tolist()]
    return np.array(labels, dtype="U16")


def _extract_process_reward_conf(process_reward: Optional[Any], num_frames: int) -> Optional[np.ndarray]:
    if process_reward is None:
        return None
    if hasattr(process_reward, "conf"):
        arr = np.asarray(getattr(process_reward, "conf"), dtype=np.float32)
        return _pad_or_trim(arr, num_frames)
    if isinstance(process_reward, dict):
        for key in ("conf", "conf_t", "conf_mean"):
            if key in process_reward:
                arr = np.asarray(process_reward.get(key), dtype=np.float32)
                return _pad_or_trim(arr, num_frames)
    return None


def _extract_process_reward_rshape(process_reward: Optional[Any], num_frames: int) -> Optional[np.ndarray]:
    if process_reward is None:
        return None
    if hasattr(process_reward, "r_shape"):
        arr = np.asarray(getattr(process_reward, "r_shape"), dtype=np.float32)
        return _pad_or_trim(arr, num_frames)
    if isinstance(process_reward, dict):
        for key in ("r_shape", "r_shape_t", "r_shape_step"):
            if key in process_reward:
                arr = np.asarray(process_reward.get(key), dtype=np.float32)
                return _pad_or_trim(arr, num_frames)
    return None


def _pad_or_trim(arr: np.ndarray, length: int) -> np.ndarray:
    arr = arr.reshape(-1)
    if arr.size == length:
        return arr
    if arr.size > length:
        return arr[:length]
    if arr.size == 0:
        return np.zeros((length,), dtype=np.float32)
    pad = np.full((length - arr.size,), float(arr[-1]), dtype=np.float32)
    return np.concatenate([arr, pad])


def _estimate_energy_per_step(
    action_stream: Optional[Sequence[Any]],
    joint_state: Optional[Sequence[Any]],
    episode_metrics: Optional[Dict[str, Any]],
    task_constraints: Optional[Dict[str, Any]],
    num_frames: int,
) -> Tuple[float, float]:
    time_step_s = 0.1
    if task_constraints:
        if "time_step_s" in task_constraints:
            time_step_s = float(task_constraints.get("time_step_s", time_step_s))

    if joint_state:
        energies = []
        for entry in joint_state:
            if isinstance(entry, dict) and "energy_estimate_Wh" in entry:
                try:
                    energies.append(float(entry.get("energy_estimate_Wh")))
                except Exception:
                    continue
        if energies:
            return float(np.mean(energies)), time_step_s

    action_energy = _estimate_energy_per_step_from_actions(action_stream)
    if action_energy is not None:
        return action_energy, time_step_s

    total_energy = None
    if episode_metrics and "energy_Wh" in episode_metrics:
        try:
            total_energy = float(episode_metrics.get("energy_Wh"))
        except Exception:
            total_energy = None
    if total_energy is not None and num_frames > 0:
        return max(0.0, total_energy / float(num_frames)), time_step_s

    return 0.0, time_step_s


def _build_cost_breakdown(
    *,
    segment_bounds: np.ndarray,
    segment_energy: np.ndarray,
    segment_risk: np.ndarray,
    segment_pairs: np.ndarray,
    time_step_s: float,
    impossible_mask: np.ndarray,
) -> Dict[str, Any]:
    segments: List[Dict[str, Any]] = []
    total_energy = float(np.sum(segment_energy)) if segment_energy.size else 0.0
    total_time = 0.0
    collision_risk = 0.0

    for idx, (start, end) in enumerate(segment_bounds.tolist()):
        duration = max(0, int(end - start + 1))
        time_s = duration * time_step_s
        total_time += time_s
        window = impossible_mask[start : end + 1]
        collision = float(np.mean(window.astype(np.float32))) if window.size else 0.0
        collision_risk = max(collision_risk, collision)
        segments.append(
            {
                "segment_id": idx,
                "start": int(start),
                "end": int(end),
                "time_s": float(time_s),
                "energy_Wh": float(segment_energy[idx]) if idx < len(segment_energy) else 0.0,
                "expected_rework_risk": float(segment_risk[idx]) if idx < len(segment_risk) else 0.0,
                "collision_risk": float(collision),
                "contact_pair": [int(x) for x in segment_pairs[idx].tolist()] if idx < len(segment_pairs) else [-1, -1],
            }
        )

    return {
        "episode": {
            "energy_Wh": total_energy,
            "time_s": float(total_time),
            "expected_rework_risk": float(np.mean(segment_risk)) if segment_risk.size else 0.0,
            "collision_risk": float(collision_risk),
        },
        "segments": segments,
    }


def _build_value_attribution(
    *,
    segment_bounds: np.ndarray,
    segment_confidence: np.ndarray,
    segment_pairs: np.ndarray,
    econ_attribution: Optional[Dict[str, float]],
    track_labels: np.ndarray,
) -> Dict[str, Any]:
    totals = {
        "delta_mpl": float(econ_attribution.get("delta_mpl", 0.0)) if econ_attribution else 0.0,
        "delta_error": float(econ_attribution.get("delta_error", 0.0)) if econ_attribution else 0.0,
        "delta_ep": float(econ_attribution.get("delta_ep", 0.0)) if econ_attribution else 0.0,
    }

    weights = segment_confidence.astype(np.float32)
    if weights.size == 0 or float(np.sum(weights)) <= 0:
        weights = np.ones((len(segment_bounds),), dtype=np.float32)
    weights = weights / max(float(np.sum(weights)), 1e-6)

    segments: List[Dict[str, Any]] = []
    for idx, (start, end) in enumerate(segment_bounds.tolist()):
        pair = segment_pairs[idx] if idx < len(segment_pairs) else np.array([-1, -1])
        label = _format_pair_label(pair, track_labels)
        segments.append(
            {
                "segment_id": idx,
                "start": int(start),
                "end": int(end),
                "contact_pair": [int(x) for x in pair.tolist()],
                "label": label,
                "delta_mpl": float(totals["delta_mpl"] * weights[idx]),
                "delta_error": float(totals["delta_error"] * weights[idx]),
                "delta_ep": float(totals["delta_ep"] * weights[idx]),
                "why_valuable": _why_valuable(label),
            }
        )

    return {
        "totals": totals,
        "segments": segments,
        "attribution_source": "econ_attribution" if econ_attribution else "proxy",
    }


def _build_drift_report(
    *,
    contact_matrix: np.ndarray,
    track_labels: np.ndarray,
    task_constraints: Optional[Dict[str, Any]],
    backend_tags: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    contact_counts = _contact_distribution(contact_matrix, track_labels)
    total = sum(contact_counts.values()) or 1.0
    distribution = {k: v / total for k, v in contact_counts.items()}

    drift_score = 0.0
    distribution_drift = None
    constraint_drift = None
    violations: List[str] = []

    if task_constraints:
        prior = task_constraints.get("contact_distribution_prior")
        if isinstance(prior, dict) and prior:
            distribution_drift = _l1_drift(distribution, prior)
            drift_score = max(drift_score, distribution_drift)

        allowed = task_constraints.get("allowed_contacts")
        forbidden = task_constraints.get("forbidden_contacts")
        constraint_drift, violations = _constraint_drift(distribution, allowed, forbidden)
        drift_score = max(drift_score, constraint_drift)

    sim_mismatch = _sim_backend_mismatch(task_constraints, backend_tags)
    drift_score = max(drift_score, sim_mismatch)

    return {
        "contact_distribution": distribution,
        "contact_distribution_drift": distribution_drift,
        "constraint_drift_score": constraint_drift,
        "constraint_violations": violations,
        "sim_backend_mismatch_score": sim_mismatch,
        "drift_score": drift_score,
    }


def _build_calibration_targets(
    *,
    contact_coverage: float,
    drift_report: Dict[str, Any],
    failure_events: Optional[Dict[str, Any]],
    task_constraints: Optional[Dict[str, Any]],
    cfg: EmbodimentConfig,
) -> Dict[str, Any]:
    deltas = {
        "friction": 0.0,
        "damping": 0.0,
        "mass": 0.0,
        "restitution": 0.0,
    }
    confidence = 0.1

    if task_constraints and "expected_contact_rate" in task_constraints:
        expected = float(task_constraints.get("expected_contact_rate", 0.0))
        tol = float(task_constraints.get("expected_contact_rate_tolerance", cfg.expected_contact_rate_tolerance))
        if contact_coverage < expected - tol:
            deltas["friction"] += cfg.friction_delta_step
        elif contact_coverage > expected + tol:
            deltas["friction"] -= cfg.friction_delta_step
        confidence = max(confidence, 0.3)

    if failure_events:
        clamps = float(failure_events.get("safety_clamps", 0))
        if clamps > 0:
            deltas["damping"] += cfg.damping_delta_step
            deltas["restitution"] -= cfg.restitution_delta_step
            confidence = max(confidence, 0.4)

    drift_score = float(drift_report.get("drift_score", 0.0))
    if drift_score > 0.3:
        deltas["mass"] += cfg.mass_delta_step
        confidence = max(confidence, 0.4)

    return {
        "recommended_deltas": deltas,
        "confidence": confidence,
        "reason": "heuristic",
    }


def _contact_distribution(contact_matrix: np.ndarray, track_labels: np.ndarray) -> Dict[str, float]:
    T, K = contact_matrix.shape[:2]
    counts: Dict[str, float] = {}
    for i in range(K):
        for j in range(i + 1, K):
            count = float(np.sum(contact_matrix[:, i, j])) if T > 0 else 0.0
            if count <= 0:
                continue
            label = _format_pair_label(np.array([i, j]), track_labels)
            counts[label] = counts.get(label, 0.0) + count
    return counts


def _format_pair_label(pair: np.ndarray, track_labels: np.ndarray) -> str:
    i, j = int(pair[0]), int(pair[1])
    if i < 0 or j < 0:
        return "unknown"
    label_i = track_labels[i] if 0 <= i < len(track_labels) else f"track_{i}"
    label_j = track_labels[j] if 0 <= j < len(track_labels) else f"track_{j}"
    return f"{label_i}|{label_j}"


def _why_valuable(label: str) -> str:
    lower = label.lower()
    if any(keyword in lower for keyword in TOOL_LABEL_KEYWORDS):
        return "tool_contact"
    if "grasp" in lower or "handle" in lower:
        return "precision_grasp"
    if "support" in lower or "surface" in lower:
        return "stable_support"
    return "contact_alignment"


def _l1_drift(dist: Dict[str, float], prior: Dict[str, float]) -> float:
    keys = set(dist) | set(prior)
    drift = 0.0
    for key in keys:
        drift += abs(float(dist.get(key, 0.0)) - float(prior.get(key, 0.0)))
    return 0.5 * drift


def _constraint_drift(
    distribution: Dict[str, float],
    allowed: Optional[Iterable[Any]],
    forbidden: Optional[Iterable[Any]],
) -> Tuple[float, List[str]]:
    violations: List[str] = []
    if not distribution:
        return 0.0, violations

    allowed_set = {str(x).lower() for x in (allowed or [])}
    forbidden_set = {str(x).lower() for x in (forbidden or [])}

    if not allowed_set and not forbidden_set:
        return 0.0, violations

    violation_score = 0.0
    for label, prob in distribution.items():
        lower = str(label).lower()
        if forbidden_set and lower in forbidden_set:
            violation_score += float(prob)
            violations.append(label)
        if allowed_set and lower not in allowed_set:
            violation_score += float(prob)
            violations.append(label)
    violation_score = min(1.0, violation_score)
    return violation_score, sorted(set(violations))


def _sim_backend_mismatch(
    task_constraints: Optional[Dict[str, Any]],
    backend_tags: Optional[Dict[str, Any]],
) -> float:
    if not task_constraints or not backend_tags:
        return 0.0
    expected_backend = task_constraints.get("expected_backend")
    backend = backend_tags.get("backend") or backend_tags.get("engine_type")
    if expected_backend and backend and str(expected_backend) != str(backend):
        return 1.0
    expected_hash = task_constraints.get("expected_physics_profile_hash")
    actual_hash = backend_tags.get("physics_profile_hash")
    if expected_hash and actual_hash and str(expected_hash) != str(actual_hash):
        return 1.0
    return 0.0


def _estimate_energy_per_step_from_actions(action_stream: Optional[Sequence[Any]]) -> Optional[float]:
    if not action_stream:
        return None
    magnitudes = []
    for action in action_stream:
        if isinstance(action, dict):
            delta = action.get("delta_position") or action.get("target_position") or action.get("position")
            if isinstance(delta, (list, tuple)) and len(delta) == 3:
                try:
                    magnitudes.append(float(np.linalg.norm(delta)))
                except Exception:
                    continue
        elif isinstance(action, (list, tuple)) and len(action) == 3:
            try:
                magnitudes.append(float(np.linalg.norm(action)))
            except Exception:
                continue
    if magnitudes:
        return float(np.mean(magnitudes)) * 0.02
    return None


def _compute_w_embodiment(
    *,
    semantic_confidence_mean: float,
    contact_coverage: float,
    mhn_plausibility: float,
    drift_score: float,
    impossible_contacts: int,
    total_contact_events: int,
    cfg: EmbodimentConfig,
) -> float:
    base = (
        cfg.w_semantic_weight * float(semantic_confidence_mean)
        + cfg.w_contact_weight * float(contact_coverage)
        + cfg.w_mhn_weight * float(mhn_plausibility)
    )
    base = max(0.0, min(1.0, base))

    drift_penalty = min(1.0, float(drift_score)) * cfg.drift_penalty_weight
    impossible_rate = float(impossible_contacts) / max(float(total_contact_events), 1.0)
    impossible_penalty = min(1.0, impossible_rate) * cfg.impossible_penalty_weight

    w_embodiment = base * (1.0 - drift_penalty) * (1.0 - impossible_penalty)
    return float(max(0.0, min(1.0, w_embodiment)))


def _is_tool_label(label: Any) -> bool:
    try:
        text = str(label).lower()
    except Exception:
        return False
    return any(keyword in text for keyword in TOOL_LABEL_KEYWORDS)


def _extract_mhn_plausibility(mhn_summary: Optional[Any]) -> float:
    if mhn_summary is None:
        return 1.0
    return float(getattr(mhn_summary, "plausibility_score", 1.0))


def _missing_inputs(inputs: EmbodimentInputs) -> List[str]:
    missing = []
    if inputs.semantic_fusion is None:
        missing.append("semantic_fusion")
    if inputs.mhn_summary is None:
        missing.append("mhn_summary")
    if inputs.action_stream is None:
        missing.append("action_stream")
    if inputs.joint_state is None:
        missing.append("joint_state")
    if inputs.task_constraints is None:
        missing.append("task_constraints")
    if inputs.backend_tags is None:
        missing.append("backend_tags")
    if inputs.failure_events is None:
        missing.append("failure_events")
    if inputs.episode_metrics is None:
        missing.append("episode_metrics")
    if inputs.econ_attribution is None:
        missing.append("econ_attribution")
    return missing


__all__ = [
    "EmbodimentConfig",
    "EmbodimentInputs",
    "EmbodimentResult",
    "compute_embodiment",
]
