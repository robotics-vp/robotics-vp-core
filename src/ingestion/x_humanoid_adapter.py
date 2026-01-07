"""Non-blocking ingest adapter for X-Humanoid-style robotized clips."""
from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from src.controllers.synthetic_weight_controller import SyntheticWeightController
from src.embodiment.config import EmbodimentConfig
from src.embodiment.runner import run_embodiment_for_rollouts
from src.motor_backend.rollout_capture import EpisodeMetadata, EpisodeRollout, RolloutBundle
from src.orchestrator.semantic_fusion_runner import run_semantic_fusion_for_rollouts
from src.vision.map_first_supervision.node import MapFirstSupervisionNode
from src.vision.map_first_supervision.config import MapFirstSupervisionConfig
from src.vision.scene_ir_tracker.io.scene_tracks_runner import run_scene_tracks
from src.vla.semantic_evidence import build_vla_semantic_evidence_stub, save_vla_semantic_evidence_npz


@dataclass
class XHumanoidClipSpec:
    clip_path: str
    episode_id: Optional[str] = None
    task_id: str = "x_humanoid"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class XHumanoidIngestConfig:
    output_root: Path = Path("data/x_humanoid")
    ontology_root: Path = Path("data/ontology")
    camera: str = "front"
    scene_tracks_mode: str = "rgb"
    allow_low_quality: bool = True
    min_scene_tracks_quality: float = 0.2
    semantic_fusion_emit: bool = True
    map_first_enabled: bool = True
    vla_enabled: bool = True
    embodiment_enabled: bool = True
    enforce_gate: bool = False
    admission_threshold: float = 0.1
    lambda_target: float = 0.2
    trust_default: float = 0.5
    w_econ_default: float = 1.0
    use_symlink: bool = True


@dataclass
class XHumanoidIngestResult:
    episode_id: str
    episode_dir: Path
    scene_tracks_path: Optional[Path]
    map_first_path: Optional[Path]
    semantic_fusion_path: Optional[Path]
    embodiment_summary: Optional[Dict[str, Any]]
    admission_score: float
    admission_allowed: bool
    admission_override_candidate: bool


class XHumanoidIngestAdapter:
    """Ingest robotized RGB clips and run the Stage-2 vision/embodiment chain."""

    def __init__(self, config: Optional[XHumanoidIngestConfig] = None) -> None:
        self.config = config or XHumanoidIngestConfig()

    def ingest(self, clips: Iterable[XHumanoidClipSpec]) -> List[XHumanoidIngestResult]:
        results: List[XHumanoidIngestResult] = []
        for idx, clip in enumerate(clips):
            results.append(self._ingest_clip(clip, idx))
        return results

    def _ingest_clip(self, clip: XHumanoidClipSpec, index: int) -> XHumanoidIngestResult:
        cfg = self.config
        episode_id = clip.episode_id or f"xhum_{uuid.uuid4().hex[:8]}"
        episode_dir = cfg.output_root / f"episode_{index:03d}_{episode_id}"
        episode_dir.mkdir(parents=True, exist_ok=True)

        rgb_path = episode_dir / "rgb.mp4"
        self._materialize_clip(Path(clip.clip_path), rgb_path)

        trajectory_path = episode_dir / "trajectory.npz"
        metadata_path = episode_dir / "metadata.json"

        self._write_metadata(metadata_path, clip, episode_id, trajectory_path, rgb_path)

        scene_tracks_path = None
        try:
            result = run_scene_tracks(
                datapack_path=episode_dir,
                output_path=episode_dir,
                camera=cfg.camera,
                mode=cfg.scene_tracks_mode,
                ontology_root=cfg.ontology_root,
                min_quality=cfg.min_scene_tracks_quality,
                allow_low_quality=cfg.allow_low_quality,
            )
            scene_tracks_path = result.scene_tracks_path
        except Exception:
            scene_tracks_path = None

        if scene_tracks_path is not None:
            self._write_trajectory_payload(trajectory_path, episode_id, scene_tracks_path)

        map_first_path = None
        if cfg.map_first_enabled and scene_tracks_path is not None:
            map_first_path = episode_dir / "map_first_supervision_v1.npz"
            try:
                scene_tracks = dict(np.load(scene_tracks_path, allow_pickle=False))
                node = MapFirstSupervisionNode(MapFirstSupervisionConfig())
                node.run(scene_tracks, episode_assets=None, output_path=str(map_first_path))
            except Exception:
                map_first_path = None

        if cfg.vla_enabled and scene_tracks_path is not None:
            try:
                scene_tracks = dict(np.load(scene_tracks_path, allow_pickle=False))
                evidence = build_vla_semantic_evidence_stub(
                    scene_tracks=scene_tracks,
                    semantic_tags=list(clip.metadata.get("semantic_tags", [])),
                    instruction=str(clip.metadata.get("instruction", "")),
                )
                evidence_path = trajectory_path.with_name(f"{trajectory_path.stem}_vla_semantic_evidence_v1.npz")
                save_vla_semantic_evidence_npz(evidence_path, evidence)
            except Exception:
                pass

        semantic_fusion_path = None
        if cfg.semantic_fusion_emit:
            bundle = RolloutBundle(
                scenario_id=f"xhumanoid_{episode_id}",
                episodes=[
                    EpisodeRollout(
                        metadata=EpisodeMetadata(
                            episode_id=episode_id,
                            task_id=clip.task_id,
                            robot_family="x_humanoid",
                            seed=None,
                            env_params=clip.metadata,
                        ),
                        trajectory_path=trajectory_path,
                        rgb_video_path=rgb_path,
                        depth_video_path=None,
                        metrics={},
                    )
                ],
            )
            summaries = run_semantic_fusion_for_rollouts(bundle, emit_semantic_fusion=True)
            if summaries:
                semantic_fusion_path = Path(summaries[0].get("semantic_fusion_path")) if summaries[0].get("semantic_fusion_path") else None

        embodiment_summary = None
        if cfg.embodiment_enabled:
            bundle = RolloutBundle(
                scenario_id=f"xhumanoid_{episode_id}",
                episodes=[
                    EpisodeRollout(
                        metadata=EpisodeMetadata(
                            episode_id=episode_id,
                            task_id=clip.task_id,
                            robot_family="x_humanoid",
                            seed=None,
                            env_params=clip.metadata,
                        ),
                        trajectory_path=trajectory_path,
                        rgb_video_path=rgb_path,
                        depth_video_path=None,
                        metrics={},
                    )
                ],
            )
            summaries = run_embodiment_for_rollouts(
                bundle,
                output_dir=episode_dir,
                summary_path=episode_dir / "embodiment_summary.jsonl",
                config=EmbodimentConfig(),
                task_constraints=clip.metadata.get("task_constraints"),
            )
            if summaries:
                embodiment_summary = summaries[0]

        admission_score, admission_allowed, override_candidate = self._admission_gate(
            embodiment_summary,
            trust=float(clip.metadata.get("trust_score", cfg.trust_default)),
            w_econ=float(clip.metadata.get("w_econ", cfg.w_econ_default)),
        )

        self._update_metadata_with_gate(
            metadata_path,
            admission_score=admission_score,
            admission_allowed=admission_allowed,
            override_candidate=override_candidate,
        )

        return XHumanoidIngestResult(
            episode_id=episode_id,
            episode_dir=episode_dir,
            scene_tracks_path=scene_tracks_path,
            map_first_path=map_first_path,
            semantic_fusion_path=semantic_fusion_path,
            embodiment_summary=embodiment_summary,
            admission_score=admission_score,
            admission_allowed=admission_allowed,
            admission_override_candidate=override_candidate,
        )

    def _materialize_clip(self, clip_path: Path, dest: Path) -> None:
        if dest.exists():
            return
        if clip_path.is_dir():
            for candidate in (clip_path / "rgb.mp4", clip_path / "rgb_front.mp4"):
                if candidate.exists():
                    clip_path = candidate
                    break
            if clip_path.is_dir():
                raise FileNotFoundError(f"No rgb.mp4 found in {clip_path}")
        if not clip_path.exists():
            raise FileNotFoundError(f"Clip path not found: {clip_path}")
        if self.config.use_symlink:
            try:
                dest.symlink_to(clip_path)
                return
            except Exception:
                pass
        shutil.copyfile(clip_path, dest)

    def _write_metadata(
        self,
        metadata_path: Path,
        clip: XHumanoidClipSpec,
        episode_id: str,
        trajectory_path: Path,
        rgb_path: Path,
    ) -> None:
        payload = {
            "metadata": {
                "episode_id": episode_id,
                "task_id": clip.task_id,
                "robot_family": "x_humanoid",
                "seed": None,
                "env_params": dict(clip.metadata),
            },
            "trajectory_path": str(trajectory_path),
            "rgb_video_path": str(rgb_path),
            "depth_video_path": None,
            "metrics": {},
        }
        metadata_path.write_text(json.dumps(payload, indent=2))

    def _write_trajectory_payload(self, trajectory_path: Path, episode_id: str, scene_tracks_path: Path) -> None:
        payload = {
            "episode_id": episode_id,
            "scene_tracks_path": str(scene_tracks_path),
        }
        np.savez_compressed(trajectory_path, trajectory=payload)

    def _update_metadata_with_gate(
        self,
        metadata_path: Path,
        admission_score: float,
        admission_allowed: bool,
        override_candidate: bool,
    ) -> None:
        if not metadata_path.exists():
            return
        try:
            payload = json.loads(metadata_path.read_text())
        except Exception:
            return
        metrics = payload.get("metrics", {})
        if not isinstance(metrics, dict):
            metrics = {}
        metrics.update(
            {
                "embodiment_admission_score": float(admission_score),
                "embodiment_admission_allowed": bool(admission_allowed),
                "embodiment_admission_override_candidate": bool(override_candidate),
            }
        )
        payload["metrics"] = metrics
        metadata_path.write_text(json.dumps(payload, indent=2))

    def _admission_gate(self, summary: Optional[Dict[str, Any]], trust: float, w_econ: float) -> tuple[float, bool, bool]:
        cfg = self.config
        controller = SyntheticWeightController(default_lambda=cfg.lambda_target)
        weights = controller.compute_weights(
            trust=np.array([trust], dtype=np.float32),
            econ=np.array([w_econ], dtype=np.float32),
            n_real=1,
            mode="trust_econ_lambda",
            lambda_target=cfg.lambda_target,
        )
        quality = float(weights.get("quality", np.array([0.0], dtype=np.float32))[0])
        w_embodiment = float(summary.get("w_embodiment", 1.0)) if summary else 1.0
        admission_score = quality * w_embodiment
        override_candidate = bool(summary.get("trust_override_candidate", False)) if summary else False
        admission_allowed = admission_score >= cfg.admission_threshold
        if cfg.enforce_gate and override_candidate:
            admission_allowed = False
        if not cfg.enforce_gate:
            # Non-blocking by default: admission is advisory unless explicitly enforced.
            admission_allowed = True
        return admission_score, admission_allowed, override_candidate


__all__ = [
    "XHumanoidClipSpec",
    "XHumanoidIngestConfig",
    "XHumanoidIngestResult",
    "XHumanoidIngestAdapter",
]
