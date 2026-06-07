"""LeRobot video/camera receipt bridge for replay and perception samples.

The adapter is intentionally local and receipt-oriented: it normalizes
video/camera logistics receipts into canonical replay rows and CPU-safe
perception seam samples without decoding video, running providers, training,
writing weights, or making promotion claims.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Optional, Sequence

from src.dataset_bridges.lerobot_bridge import replay_episode_from_lerobot
from src.dataset_bridges.lerobot_perception_adapter import (
    FeatureExtractionConfig,
    LeRobotPerceptionAdapterConfig,
    adapt_lerobot_episodes_for_evidence_fusion,
    adapt_lerobot_episodes_for_vision_backbone_projection,
    adapt_lerobot_episodes_for_vjepa_temporal,
)
from src.replay.schema import ReplayEpisodeRecord, ReplayStepRecord
from src.training.perception_seam_data import (
    MultiProviderSample,
    VJEPATemporalSample,
    VisionBackboneProjectionSample,
)
from src.utils.json_safe import to_json_safe


LEROBOT_VIDEO_RECEIPT_BRIDGE_VERSION = "lerobot_video_receipt_bridge_v1"
DEFAULT_SOURCE_DOMAIN = "lerobot_video_receipt_bridge"

_REF_KEYS = (
    "runtime_packet_ref",
    "event_spine_ref",
    "decision_ledger_ref",
    "objective_tensor_ref",
    "econ_tensor_ref",
    "pricing_tick_ref",
    "ledger_event_ref",
    "event_refs",
    "decision_refs",
)


@dataclass(frozen=True)
class LeRobotVideoReceiptBridgeReport:
    """Summary receipt for local video-receipt replay/perception bridging."""

    report_id: str
    dataset_id: str
    status: str
    video_receipt_count: int
    replay_episode_count: int
    replay_step_count: int
    camera_key_count: int
    evidence_fusion_sample_count: int
    vjepa_temporal_sample_count: int
    vision_backbone_projection_sample_count: int
    provider_executed: bool = False
    gpu_training_executed: bool = False
    video_decoding_executed: bool = False
    weights_downloaded: bool = False
    unitree_hardware_truth: bool = False
    promotion_eligible: bool = False
    phase7_authority_granted: bool = False
    unavailable_posture_count: int = 0
    remaining_blockers: list[str] = field(default_factory=list)
    artifact_refs: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    version: str = LEROBOT_VIDEO_RECEIPT_BRIDGE_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "version": self.version,
            "dataset_id": self.dataset_id,
            "status": self.status,
            "video_receipt_count": int(self.video_receipt_count),
            "replay_episode_count": int(self.replay_episode_count),
            "replay_step_count": int(self.replay_step_count),
            "camera_key_count": int(self.camera_key_count),
            "evidence_fusion_sample_count": int(self.evidence_fusion_sample_count),
            "vjepa_temporal_sample_count": int(self.vjepa_temporal_sample_count),
            "vision_backbone_projection_sample_count": int(
                self.vision_backbone_projection_sample_count
            ),
            "provider_executed": bool(self.provider_executed),
            "gpu_training_executed": bool(self.gpu_training_executed),
            "video_decoding_executed": bool(self.video_decoding_executed),
            "weights_downloaded": bool(self.weights_downloaded),
            "unitree_hardware_truth": bool(self.unitree_hardware_truth),
            "promotion_eligible": bool(self.promotion_eligible),
            "phase7_authority_granted": bool(self.phase7_authority_granted),
            "unavailable_posture_count": int(self.unavailable_posture_count),
            "remaining_blockers": list(self.remaining_blockers),
            "artifact_refs": _json_mapping(self.artifact_refs),
            "metadata": _json_mapping(self.metadata),
        }


@dataclass(frozen=True)
class LeRobotVideoReceiptPerceptionBundle:
    """In-memory result for one local video-receipt bridge pass."""

    report: LeRobotVideoReceiptBridgeReport
    lerobot_rows: list[dict[str, Any]]
    episodes: list[ReplayEpisodeRecord]
    steps: list[ReplayStepRecord]
    evidence_fusion_samples: list[MultiProviderSample]
    vjepa_temporal_samples: list[VJEPATemporalSample]
    vision_backbone_projection_samples: list[VisionBackboneProjectionSample]

    def to_summary(self) -> dict[str, Any]:
        return self.report.to_dict()


def _json_mapping(payload: Mapping[str, Any]) -> dict[str, Any]:
    safe = to_json_safe(dict(payload))
    if isinstance(safe, dict):
        return dict(safe)
    return {}


def _stable_id(prefix: str, payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(to_json_safe(dict(payload)), sort_keys=True).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(encoded).hexdigest()[:16]}"


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _frames(receipt: Mapping[str, Any]) -> list[dict[str, Any]]:
    value = receipt.get("frames") or receipt.get("frame_receipts")
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        rows = [_mapping(row) for row in value]
        return [row for row in rows if row]
    return [dict(receipt)]


def _strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, Iterable):
        return [str(item) for item in value if str(item)]
    return [str(value)] if str(value) else []


def _int_value(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float_value(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_camera_key(value: str) -> str:
    key = value.strip()
    for prefix in ("observation.images.", "images."):
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def _camera_key_from_path(path: str) -> str | None:
    for part in PurePosixPath(path).parts:
        if part.startswith("observation.images."):
            return _normalize_camera_key(part)
        if part.startswith("images."):
            return _normalize_camera_key(part)
    return None


def _camera_keys(
    receipt: Mapping[str, Any],
    frame: Mapping[str, Any],
) -> list[str]:
    keys = _strings(frame.get("camera_keys")) or _strings(receipt.get("camera_keys"))
    cameras = _mapping(frame.get("cameras")) or _mapping(receipt.get("cameras"))
    keys.extend(_normalize_camera_key(str(key)) for key in cameras.keys())
    single_camera = frame.get("camera_key") or receipt.get("camera_key")
    if single_camera is not None:
        keys.append(_normalize_camera_key(str(single_camera)))
    source_path = str(
        frame.get("source_path")
        or frame.get("source_video_ref")
        or receipt.get("source_path")
        or receipt.get("source_video_ref")
        or ""
    )
    path_key = _camera_key_from_path(source_path)
    if path_key is not None:
        keys.append(path_key)
    ordered = sorted({key for key in keys if key})
    return ordered or ["camera_0"]


def _camera_payloads(
    receipt: Mapping[str, Any],
    frame: Mapping[str, Any],
    camera_keys: Sequence[str],
) -> dict[str, dict[str, Any]]:
    cameras = _mapping(frame.get("cameras")) or _mapping(receipt.get("cameras"))
    payloads: dict[str, dict[str, Any]] = {}
    for camera_key in camera_keys:
        raw = cameras.get(camera_key) or cameras.get(f"images.{camera_key}") or {}
        payloads[camera_key] = _mapping(raw)
    return payloads


def _camera_available(
    camera: Mapping[str, Any],
    receipt: Mapping[str, Any],
    frame: Mapping[str, Any],
) -> bool:
    for key in ("available", "video_file_exists", "file_exists", "present"):
        value = camera.get(key, frame.get(key, receipt.get(key)))
        if value is not None:
            return bool(value)
    source_ref = (
        camera.get("source_video_ref")
        or camera.get("source_path")
        or frame.get("source_video_ref")
        or frame.get("source_path")
        or receipt.get("source_video_ref")
        or receipt.get("source_path")
    )
    return bool(source_ref)


def _combined_sidecars(
    receipt: Mapping[str, Any],
    frame: Mapping[str, Any],
    *,
    receipt_id: str,
    frame_receipt_id: str,
) -> dict[str, Any]:
    sidecars: dict[str, Any] = {
        "video_receipt_ref": receipt_id,
        "video_frame_receipt_ref": frame_receipt_id,
    }
    for source in (receipt, frame, _mapping(receipt.get("sidecars")), _mapping(frame.get("sidecars"))):
        for key in _REF_KEYS:
            value = source.get(key)
            if value is not None:
                sidecars[key] = list(value) if isinstance(value, list) else value
    return sidecars


def lerobot_rows_from_video_receipts(
    receipts: Sequence[Mapping[str, Any]],
    *,
    dataset_id: str = "lerobot_video_receipts",
    default_run_id: str = "lerobot_video_receipt_bridge",
    default_task_id: str = "unknown_task",
    default_env_id: str = "bipedal_whole_body_unitree_g1_shadow_replay",
    default_source_domain: str = DEFAULT_SOURCE_DOMAIN,
) -> list[dict[str, Any]]:
    """Normalize local LeRobot video/camera receipts into LeRobot-like rows."""

    rows: list[dict[str, Any]] = []
    for receipt_index, receipt in enumerate(receipts):
        receipt_id = str(
            receipt.get("receipt_id")
            or receipt.get("video_receipt_id")
            or _stable_id("lerobot_video_receipt", {"index": receipt_index, **dict(receipt)})
        )
        episode_id = str(
            receipt.get("episode_id")
            or receipt.get("source_episode_id")
            or f"{dataset_id.replace('/', '_')}_episode_{receipt_index:06d}"
        )
        run_id = str(receipt.get("run_id") or default_run_id)
        task_id = str(receipt.get("task") or receipt.get("task_id") or default_task_id)
        env_id = str(
            receipt.get("environment") or receipt.get("env_id") or default_env_id
        )
        for local_index, frame in enumerate(_frames(receipt)):
            frame_index = _int_value(
                frame.get("frame_index", frame.get("step_idx", frame.get("frame"))),
                local_index,
            )
            frame_receipt_id = str(
                frame.get("receipt_id")
                or frame.get("frame_receipt_id")
                or f"{receipt_id}:frame:{frame_index:06d}"
            )
            timestamp = str(
                frame.get("timestamp", receipt.get("timestamp", f"{frame_index:.6f}"))
            )
            camera_keys = _camera_keys(receipt, frame)
            camera_payloads = _camera_payloads(receipt, frame, camera_keys)
            observation: dict[str, Any] = {}
            unavailable_count = 0
            for camera_key in camera_keys:
                camera = camera_payloads[camera_key]
                available = _camera_available(camera, receipt, frame)
                unavailable_count += int(not available)
                observation[f"images.{camera_key}"] = {
                    "camera_key": camera_key,
                    "receipt_id": receipt_id,
                    "frame_receipt_id": frame_receipt_id,
                    "source_video_ref": str(
                        camera.get("source_video_ref")
                        or frame.get("source_video_ref")
                        or receipt.get("source_video_ref")
                        or receipt.get("source_path")
                        or ""
                    ),
                    "source_path": str(
                        camera.get("source_path")
                        or frame.get("source_path")
                        or receipt.get("source_path")
                        or ""
                    ),
                    "frame_index": frame_index,
                    "timestamp": timestamp,
                    "available": available,
                    "decoded": bool(camera.get("decoded", False)),
                    "video_decode_executed": False,
                    "provider_executed": False,
                    "truth_class": "advisory_evidence" if available else "unavailable",
                    "modality": str(camera.get("modality", receipt.get("modality", "video"))),
                    "unavailable_reason": str(
                        camera.get(
                            "unavailable_reason",
                            "video_ref_available_not_decoded" if available else "camera_receipt_unavailable",
                        )
                    ),
                }
            if "state" in frame or "observation.state" in frame:
                observation["observation.state"] = frame.get(
                    "observation.state", frame.get("state")
                )
            if "state" in receipt and "observation.state" not in observation:
                observation["observation.state"] = receipt.get("state")

            sidecars = _combined_sidecars(
                receipt,
                frame,
                receipt_id=receipt_id,
                frame_receipt_id=frame_receipt_id,
            )
            rows.append(
                {
                    "episode_id": str(frame.get("episode_id", episode_id) or episode_id),
                    "frame_index": frame_index,
                    "timestamp": timestamp,
                    "observation": observation,
                    "action": _mapping(frame.get("action") or receipt.get("action")),
                    "reward": _float_value(
                        frame.get("reward_proxy", frame.get("reward")),
                        _float_value(receipt.get("reward_proxy", receipt.get("reward")), 0.0),
                    ),
                    "done": bool(frame.get("done", receipt.get("done", False))),
                    "task": str(frame.get("task", task_id) or task_id),
                    "environment": str(frame.get("environment", env_id) or env_id),
                    "source_domain": default_source_domain,
                    "metadata": {
                        "run_id": run_id,
                        "seed": _int_value(receipt.get("seed"), 0),
                        "skill_mode": str(
                            receipt.get(
                                "skill_mode",
                                "external_video_receipt_shadow_import",
                            )
                        ),
                        "source_dataset_id": dataset_id,
                        "source_receipt_id": receipt_id,
                        "source_frame_receipt_id": frame_receipt_id,
                        "source_frame_index": frame_index,
                        "camera_keys": list(camera_keys),
                        "camera_count": len(camera_keys),
                        "camera_unavailable_count": unavailable_count,
                        "perception_sample_truth_class": "advisory_evidence",
                        "feature_posture": "cpu_placeholder_schema_verification",
                        "video_receipt_bridge": {
                            "version": LEROBOT_VIDEO_RECEIPT_BRIDGE_VERSION,
                            "provider_executed": False,
                            "gpu_training_executed": False,
                            "video_decoding_executed": False,
                            "weights_downloaded": False,
                            "promotion_eligible": False,
                            "phase7_authority_granted": False,
                        },
                        "benchmark_gate": {
                            "ready": False,
                            "reason": "video_receipt_bridge_schema_only_not_provider_truth",
                        },
                        "future_training_signals": {
                            "lerobot_video_receipt_bridge": True,
                            "training_ready": False,
                            "promotion_eligible": False,
                            "provider_truth": False,
                            "unitree_hardware_truth": False,
                        },
                        "internal_sidecars": sidecars,
                    },
                }
            )
    return rows


def replay_episodes_from_lerobot_video_receipts(
    receipts: Sequence[Mapping[str, Any]],
    *,
    dataset_id: str = "lerobot_video_receipts",
    default_run_id: str = "lerobot_video_receipt_bridge",
    default_task_id: str = "unknown_task",
    default_env_id: str = "bipedal_whole_body_unitree_g1_shadow_replay",
    default_source_domain: str = DEFAULT_SOURCE_DOMAIN,
) -> tuple[list[ReplayEpisodeRecord], list[ReplayStepRecord], list[dict[str, Any]]]:
    """Convert video receipts into replay episodes and steps."""

    rows = lerobot_rows_from_video_receipts(
        receipts,
        dataset_id=dataset_id,
        default_run_id=default_run_id,
        default_task_id=default_task_id,
        default_env_id=default_env_id,
        default_source_domain=default_source_domain,
    )
    by_episode: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_episode.setdefault(str(row["episode_id"]), []).append(row)

    episodes: list[ReplayEpisodeRecord] = []
    steps: list[ReplayStepRecord] = []
    for episode_rows in by_episode.values():
        episode, episode_steps = replay_episode_from_lerobot(
            episode_rows,
            default_run_id=default_run_id,
            default_source_domain=default_source_domain,
        )
        episodes.append(episode)
        steps.extend(episode_steps)
    return episodes, steps, rows


def adapt_lerobot_video_receipts_for_perception(
    receipts: Sequence[Mapping[str, Any]],
    *,
    dataset_id: str = "lerobot_video_receipts",
    config: Optional[LeRobotPerceptionAdapterConfig] = None,
    projection_config: Optional[LeRobotPerceptionAdapterConfig] = None,
) -> LeRobotVideoReceiptPerceptionBundle:
    """Build replay rows and CPU-safe perception samples from video receipts."""

    if config is None:
        config = LeRobotPerceptionAdapterConfig(
            feature_config=FeatureExtractionConfig(
                strategy="placeholder",
                d_feature=128,
            ),
            temporal_window_size=2,
            temporal_stride=1,
            d_vjepa=128,
            d_wm=32,
            d_out=64,
        )
    if projection_config is None:
        projection_config = LeRobotPerceptionAdapterConfig(
            feature_config=FeatureExtractionConfig(
                strategy="placeholder",
                d_feature=1024,
            ),
            step_stride=config.step_stride,
            max_samples_per_episode=config.max_samples_per_episode,
            projection_tokens_per_camera=2,
            d_out=128,
        )

    episodes, steps, rows = replay_episodes_from_lerobot_video_receipts(
        receipts,
        dataset_id=dataset_id,
    )
    episode_pairs = [
        (episode, [step for step in steps if step.episode_id == episode.episode_id])
        for episode in episodes
    ]
    evidence_samples = adapt_lerobot_episodes_for_evidence_fusion(
        episode_pairs,
        config,
    )
    vjepa_samples = adapt_lerobot_episodes_for_vjepa_temporal(episode_pairs, config)
    projection_samples = adapt_lerobot_episodes_for_vision_backbone_projection(
        episode_pairs,
        projection_config,
    )
    camera_keys = sorted(
        {
            key
            for row in rows
            for key in _strings(_mapping(row.get("metadata")).get("camera_keys"))
        }
    )
    unavailable_posture_count = sum(
        int(
            _mapping(row.get("metadata")).get("camera_unavailable_count", 0)
        )
        for row in rows
    )
    report = LeRobotVideoReceiptBridgeReport(
        report_id=_stable_id(
            "lerobot_video_receipt_bridge_report",
            {
                "dataset_id": dataset_id,
                "receipt_count": len(receipts),
                "row_count": len(rows),
                "camera_keys": camera_keys,
            },
        ),
        dataset_id=dataset_id,
        status="ok_local_video_receipts_replay_perception_schema_only",
        video_receipt_count=len(receipts),
        replay_episode_count=len(episodes),
        replay_step_count=len(steps),
        camera_key_count=len(camera_keys),
        evidence_fusion_sample_count=len(evidence_samples),
        vjepa_temporal_sample_count=len(vjepa_samples),
        vision_backbone_projection_sample_count=len(projection_samples),
        unavailable_posture_count=unavailable_posture_count,
        remaining_blockers=[
            "video_not_decoded",
            "provider_backbone_not_executed",
            "gpu_training_not_run",
            "promotion_benchmark_not_run",
            "unitree_hardware_truth_not_present",
        ],
        metadata={
            "camera_keys": camera_keys,
            "evidence_fusion_sample_ids": [sample.sample_id for sample in evidence_samples],
            "vjepa_temporal_sample_ids": [sample.sample_id for sample in vjepa_samples],
            "vision_backbone_projection_sample_ids": [
                sample.sample_id for sample in projection_samples
            ],
            "feature_strategy": config.feature_config.strategy,
            "projection_feature_strategy": projection_config.feature_config.strategy,
            "source_domain": DEFAULT_SOURCE_DOMAIN,
        },
    )
    return LeRobotVideoReceiptPerceptionBundle(
        report=report,
        lerobot_rows=rows,
        episodes=episodes,
        steps=steps,
        evidence_fusion_samples=evidence_samples,
        vjepa_temporal_samples=vjepa_samples,
        vision_backbone_projection_samples=projection_samples,
    )


def write_lerobot_video_receipt_bridge_artifacts(
    receipts: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
    *,
    dataset_id: str = "lerobot_video_receipts",
    config: Optional[LeRobotPerceptionAdapterConfig] = None,
    projection_config: Optional[LeRobotPerceptionAdapterConfig] = None,
) -> dict[str, Any]:
    """Write local bridge receipts and replay rows to an artifact directory."""

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    bundle = adapt_lerobot_video_receipts_for_perception(
        receipts,
        dataset_id=dataset_id,
        config=config,
        projection_config=projection_config,
    )
    rows_path = output_root / "lerobot_video_receipt_rows.jsonl"
    episodes_path = output_root / "replay_episodes.jsonl"
    steps_path = output_root / "replay_steps.jsonl"
    report_path = output_root / "lerobot_video_receipt_bridge_report_v1.json"
    _write_jsonl(rows_path, bundle.lerobot_rows)
    _write_jsonl(episodes_path, [episode.to_dict() for episode in bundle.episodes])
    _write_jsonl(steps_path, [step.to_dict() for step in bundle.steps])
    artifact_refs = {
        "report_path": str(report_path),
        "lerobot_rows_path": str(rows_path),
        "replay_episodes_path": str(episodes_path),
        "replay_steps_path": str(steps_path),
    }
    report = LeRobotVideoReceiptBridgeReport(
        **{
            **bundle.report.__dict__,
            "artifact_refs": artifact_refs,
        }
    )
    report_path.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report.to_dict()


def build_fixture_lerobot_video_receipts() -> list[dict[str, Any]]:
    """Small deterministic local fixture used by tests and compiler smoke."""

    return [
        {
            "receipt_id": "fixture_video_receipt_001",
            "dataset_id": "fixture/lerobot_video",
            "run_id": "fixture_lerobot_video_run",
            "episode_id": "fixture_video_episode_001",
            "task": "g1_shadow_pick_place",
            "environment": "bipedal_whole_body_unitree_g1_shadow_replay",
            "camera_keys": ["front", "wrist"],
            "source_path": "videos/chunk-000/observation.images.front/file-000.mp4",
            "runtime_packet_ref": "runtime_packet_fixture.json",
            "event_spine_ref": "event_spine_fixture.json",
            "decision_ledger_ref": "decision_ledger_fixture.json",
            "objective_tensor_ref": "objective_tensor_fixture.json",
            "econ_tensor_ref": "econ_tensor_fixture.json",
            "event_refs": ["event_fixture_001"],
            "decision_refs": ["decision_fixture_001"],
            "frames": [
                {
                    "frame_index": 0,
                    "timestamp": "0.000000",
                    "reward_proxy": 0.0,
                    "cameras": {
                        "front": {
                            "source_video_ref": "front.mp4",
                            "available": True,
                        },
                        "wrist": {
                            "source_video_ref": "wrist.mp4",
                            "available": True,
                        },
                    },
                    "action": {"joint_position_delta": [0.0, 0.1]},
                },
                {
                    "frame_index": 1,
                    "timestamp": "0.100000",
                    "reward_proxy": 0.25,
                    "cameras": {
                        "front": {
                            "source_video_ref": "front.mp4",
                            "available": True,
                        },
                        "wrist": {
                            "source_video_ref": "wrist.mp4",
                            "available": True,
                        },
                    },
                    "action": {"joint_position_delta": [0.1, 0.2]},
                },
                {
                    "frame_index": 2,
                    "timestamp": "0.200000",
                    "reward_proxy": 1.0,
                    "done": True,
                    "cameras": {
                        "front": {
                            "source_video_ref": "front.mp4",
                            "available": True,
                        },
                        "wrist": {
                            "source_video_ref": "wrist.mp4",
                            "available": False,
                            "unavailable_reason": "fixture_missing_wrist_frame",
                        },
                    },
                    "action": {"joint_position_delta": [0.2, 0.3]},
                },
            ],
        }
    ]


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(to_json_safe(dict(row)), sort_keys=True) + "\n")


__all__ = [
    "DEFAULT_SOURCE_DOMAIN",
    "LEROBOT_VIDEO_RECEIPT_BRIDGE_VERSION",
    "LeRobotVideoReceiptBridgeReport",
    "LeRobotVideoReceiptPerceptionBundle",
    "adapt_lerobot_video_receipts_for_perception",
    "build_fixture_lerobot_video_receipts",
    "lerobot_rows_from_video_receipts",
    "replay_episodes_from_lerobot_video_receipts",
    "write_lerobot_video_receipt_bridge_artifacts",
]
