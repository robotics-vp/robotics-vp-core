"""D4RT-style reconstruction sidecars for geometry and camera grounding."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from src.utils.config_digest import sha256_json
from src.utils.json_safe import to_json_safe


def _mapping(payload: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    return dict(to_json_safe(dict(payload or {})))


def _float_mapping(payload: Optional[Mapping[str, Any]]) -> Dict[str, float]:
    values: Dict[str, float] = {}
    for key, value in dict(payload or {}).items():
        try:
            values[str(key)] = float(value)
        except Exception:
            continue
    return values


def _strings(values: Optional[Sequence[Any]]) -> list[str]:
    return [str(value) for value in (values or []) if str(value)]


@dataclass(frozen=True)
class CameraCalibrationRecord:
    """Calibration completeness for one camera participating in reconstruction."""

    camera_name: str
    intrinsics_ref: Optional[str] = None
    extrinsics_ref: Optional[str] = None
    intrinsics: Dict[str, Any] = field(default_factory=dict)
    calibrated: bool = False
    calibration_source: str = "unknown"
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "camera_name": self.camera_name,
            "intrinsics_ref": self.intrinsics_ref,
            "extrinsics_ref": self.extrinsics_ref,
            "intrinsics": _mapping(self.intrinsics),
            "calibrated": bool(self.calibrated),
            "calibration_source": self.calibration_source,
            "confidence": float(self.confidence),
            "metadata": _mapping(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CameraCalibrationRecord":
        return cls(
            camera_name=str(payload.get("camera_name", "")),
            intrinsics_ref=payload.get("intrinsics_ref"),
            extrinsics_ref=payload.get("extrinsics_ref"),
            intrinsics=_mapping(payload.get("intrinsics")),
            calibrated=bool(payload.get("calibrated", False)),
            calibration_source=str(payload.get("calibration_source", "unknown")),
            confidence=float(payload.get("confidence", 0.0)),
            metadata=_mapping(payload.get("metadata")),
        )


@dataclass(frozen=True)
class FourDReconstructionSidecar:
    """Geometry/camera grounding sidecar for real-video-to-sim loops."""

    sidecar_id: str
    episode_id: str
    source_type: str
    world_frame: str
    media_refs: list[str] = field(default_factory=list)
    frame_window: Dict[str, Any] = field(default_factory=dict)
    calibrations: list[CameraCalibrationRecord] = field(default_factory=list)
    geometry_refs: Dict[str, Any] = field(default_factory=dict)
    evidence_refs: Dict[str, Any] = field(default_factory=dict)
    quality: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "four_d_reconstruction_sidecar_v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sidecar_id": self.sidecar_id,
            "episode_id": self.episode_id,
            "source_type": self.source_type,
            "world_frame": self.world_frame,
            "media_refs": list(self.media_refs),
            "frame_window": _mapping(self.frame_window),
            "calibrations": [record.to_dict() for record in self.calibrations],
            "geometry_refs": _mapping(self.geometry_refs),
            "evidence_refs": _mapping(self.evidence_refs),
            "quality": _float_mapping(self.quality),
            "metadata": _mapping(self.metadata),
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FourDReconstructionSidecar":
        return cls(
            sidecar_id=str(payload.get("sidecar_id", "")),
            episode_id=str(payload.get("episode_id", "")),
            source_type=str(payload.get("source_type", "")),
            world_frame=str(payload.get("world_frame", "world")),
            media_refs=_strings(payload.get("media_refs")),
            frame_window=_mapping(payload.get("frame_window")),
            calibrations=[
                CameraCalibrationRecord.from_dict(item)
                for item in payload.get("calibrations", []) or []
            ],
            geometry_refs=_mapping(payload.get("geometry_refs")),
            evidence_refs=_mapping(payload.get("evidence_refs")),
            quality=_float_mapping(payload.get("quality")),
            metadata=_mapping(payload.get("metadata")),
            version=str(payload.get("version", "four_d_reconstruction_sidecar_v1")),
        )


def build_four_d_reconstruction_sidecar(
    *,
    episode_id: str,
    source_type: str,
    media_refs: Optional[Sequence[Any]] = None,
    sensor_bundle_meta: Optional[Mapping[str, Any]] = None,
    frame_count: Optional[int] = None,
    frame_range: Optional[Sequence[Any]] = None,
    geometry_refs: Optional[Mapping[str, Any]] = None,
    evidence_refs: Optional[Mapping[str, Any]] = None,
    quality: Optional[Mapping[str, Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    world_frame: str = "world",
) -> FourDReconstructionSidecar:
    """Build a 4D reconstruction sidecar from available camera and geometry artifacts."""

    calibrations = _calibration_records_from_sensor_bundle(sensor_bundle_meta)
    calibration_score = 0.0
    if calibrations:
        calibration_score = sum(record.confidence for record in calibrations) / float(len(calibrations))

    quality_payload = {
        **_float_mapping(quality),
        "calibration_score": float(calibration_score),
        "camera_count": float(len(calibrations)),
        "grounding_completeness": float(_grounding_completeness(calibrations, geometry_refs, evidence_refs)),
    }
    frame_window = {
        "frame_count": int(frame_count or 0),
        "frame_range": [int(frame_range[0]), int(frame_range[1])] if frame_range and len(frame_range) >= 2 else [],
    }
    payload = {
        "episode_id": str(episode_id),
        "source_type": str(source_type),
        "world_frame": str(world_frame),
        "media_refs": _strings(media_refs),
        "frame_window": frame_window,
        "calibrations": [record.to_dict() for record in calibrations],
        "geometry_refs": _mapping(geometry_refs),
        "evidence_refs": _mapping(evidence_refs),
        "quality": quality_payload,
        "metadata": {
            "sensor_bundle_present": bool(sensor_bundle_meta),
            **_mapping(metadata),
        },
        "version": "four_d_reconstruction_sidecar_v1",
    }
    sidecar_id = f"recon4d_{sha256_json(payload)[:16]}"
    return FourDReconstructionSidecar(
        sidecar_id=sidecar_id,
        episode_id=str(episode_id),
        source_type=str(source_type),
        world_frame=str(world_frame),
        media_refs=_strings(media_refs),
        frame_window=frame_window,
        calibrations=calibrations,
        geometry_refs=_mapping(geometry_refs),
        evidence_refs=_mapping(evidence_refs),
        quality=quality_payload,
        metadata={
            "sensor_bundle_present": bool(sensor_bundle_meta),
            **_mapping(metadata),
        },
    )


def save_four_d_reconstruction_sidecar(path: Path, sidecar: FourDReconstructionSidecar) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sidecar.to_dict(), indent=2))


def load_four_d_reconstruction_sidecar(path: Path) -> FourDReconstructionSidecar:
    return FourDReconstructionSidecar.from_dict(json.loads(path.read_text()))


def _calibration_records_from_sensor_bundle(
    sensor_bundle_meta: Optional[Mapping[str, Any]],
) -> list[CameraCalibrationRecord]:
    if not isinstance(sensor_bundle_meta, Mapping):
        return []
    cameras = sensor_bundle_meta.get("cameras") or []
    intrinsics = sensor_bundle_meta.get("intrinsics") if isinstance(sensor_bundle_meta.get("intrinsics"), Mapping) else {}
    extrinsics = sensor_bundle_meta.get("extrinsics") if isinstance(sensor_bundle_meta.get("extrinsics"), Mapping) else {}
    records: list[CameraCalibrationRecord] = []
    for camera_name in cameras:
        name = str(camera_name)
        intr_ref = intrinsics.get(name)
        extr_ref = extrinsics.get(name)
        calibrated = bool(intr_ref and extr_ref)
        confidence = 1.0 if calibrated else (0.5 if (intr_ref or extr_ref) else 0.0)
        records.append(
            CameraCalibrationRecord(
                camera_name=name,
                intrinsics_ref=str(intr_ref) if intr_ref else None,
                extrinsics_ref=str(extr_ref) if extr_ref else None,
                calibrated=calibrated,
                calibration_source="sensor_bundle",
                confidence=float(confidence),
                metadata={"depth_unit": sensor_bundle_meta.get("depth_unit", "unknown")},
            )
        )
    return records


def _grounding_completeness(
    calibrations: Sequence[CameraCalibrationRecord],
    geometry_refs: Optional[Mapping[str, Any]],
    evidence_refs: Optional[Mapping[str, Any]],
) -> float:
    components: list[float] = []
    if calibrations:
        components.append(sum(1.0 for record in calibrations if record.calibrated) / float(len(calibrations)))
    if geometry_refs:
        present = sum(1.0 for value in dict(geometry_refs).values() if value)
        components.append(min(1.0, present / 4.0))
    if evidence_refs:
        present = sum(1.0 for value in dict(evidence_refs).values() if value)
        components.append(min(1.0, present / 4.0))
    return float(sum(components) / float(len(components))) if components else 0.0


__all__ = [
    "CameraCalibrationRecord",
    "FourDReconstructionSidecar",
    "build_four_d_reconstruction_sidecar",
    "load_four_d_reconstruction_sidecar",
    "save_four_d_reconstruction_sidecar",
]
