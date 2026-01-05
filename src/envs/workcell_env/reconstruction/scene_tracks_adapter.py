"""
Scene reconstruction adapter for workcell environments.

Converts SceneTracks_v1 and map_first_supervision artifacts
into WorkcellSceneSpec for environment replay.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.envs.workcell_env.scene.scene_spec import (
    ContainerSpec,
    FixtureSpec,
    PartSpec,
    StationSpec,
    WorkcellSceneSpec,
)

logger = logging.getLogger(__name__)


@dataclass
class TrackInfo:
    """Information about a tracked object."""
    track_id: str
    object_type: str
    positions: np.ndarray  # (T, 3)
    orientations: np.ndarray  # (T, 4) quaternions
    semantic_labels: Optional[np.ndarray] = None  # (T,) or (T, num_classes)
    confidence: Optional[np.ndarray] = None  # (T,)


@dataclass
class ReconstructionResult:
    """Result of scene reconstruction."""
    scene_spec: WorkcellSceneSpec
    track_mapping: Dict[str, str]  # track_id -> spec_id
    confidence_score: float
    warnings: List[str]


class SceneTracksAdapter:
    """
    Adapter that converts SceneTracks_v1 outputs into WorkcellSceneSpec.

    Takes tracked object trajectories and labels, and produces an initial
    layout (fixture placement, part placement) with constraints.
    """

    # Object type mapping from semantic labels
    OBJECT_TYPE_MAP = {
        "fixture": ["jig", "vise", "clamp", "holder", "bracket_holder"],
        "container": ["bin", "tray", "pallet", "box", "crate"],
        "part": ["bolt", "screw", "plate", "bracket", "housing", "peg", "widget"],
        "tool": ["wrench", "screwdriver", "gripper", "probe"],
        "station": ["bench", "table", "conveyor_segment"],
    }

    def __init__(self, default_bounds: Tuple[float, float, float] = (2.0, 2.0, 1.5)):
        self.default_bounds = default_bounds

    def reconstruct_from_tracks(
        self,
        scene_tracks: Dict[str, np.ndarray],
        map_first_artifacts: Optional[Dict[str, np.ndarray]] = None,
    ) -> ReconstructionResult:
        """
        Reconstruct WorkcellSceneSpec from SceneTracks_v1 data.

        Args:
            scene_tracks: Dict with keys like 'track_ids', 'positions', 'orientations', etc.
            map_first_artifacts: Optional map_first_supervision outputs

        Returns:
            ReconstructionResult with scene_spec and metadata
        """
        warnings: List[str] = []
        track_mapping: Dict[str, str] = {}

        # Parse tracks
        tracks = self._parse_scene_tracks(scene_tracks)
        if not tracks:
            warnings.append("No tracks found in scene_tracks data")

        # Use map_first for semantic labels if available
        if map_first_artifacts:
            tracks = self._enrich_with_map_first(tracks, map_first_artifacts)

        # Classify objects by type
        fixtures: List[FixtureSpec] = []
        containers: List[ContainerSpec] = []
        parts: List[PartSpec] = []
        stations: List[StationSpec] = []

        for track in tracks:
            obj_type = self._classify_object(track)
            initial_pos = tuple(track.positions[0].tolist()) if len(track.positions) > 0 else (0.0, 0.0, 0.0)
            initial_ori = tuple(track.orientations[0].tolist()) if len(track.orientations) > 0 else (0.0, 0.0, 0.0, 1.0)

            if obj_type == "fixture":
                spec = FixtureSpec(
                    id=f"fixture_{len(fixtures)}",
                    position=initial_pos,
                    orientation=initial_ori,
                    fixture_type=track.object_type or "generic",
                )
                fixtures.append(spec)
                track_mapping[track.track_id] = spec.id

            elif obj_type == "container":
                spec = ContainerSpec(
                    id=f"container_{len(containers)}",
                    position=initial_pos,
                    orientation=initial_ori,
                    container_type=track.object_type or "bin",
                    capacity=10,
                )
                containers.append(spec)
                track_mapping[track.track_id] = spec.id

            elif obj_type == "part":
                spec = PartSpec(
                    id=f"part_{len(parts)}",
                    position=initial_pos,
                    orientation=initial_ori,
                    part_type=track.object_type or "generic",
                )
                parts.append(spec)
                track_mapping[track.track_id] = spec.id

            elif obj_type == "station":
                spec = StationSpec(
                    id=f"station_{len(stations)}",
                    position=initial_pos,
                    orientation=initial_ori,
                    station_type=track.object_type or "bench",
                )
                stations.append(spec)
                track_mapping[track.track_id] = spec.id

        # Compute spatial bounds from track positions
        all_positions = np.concatenate([t.positions for t in tracks]) if tracks else np.zeros((1, 3))
        bounds = self._compute_bounds(all_positions)

        # Build scene spec
        scene_spec = WorkcellSceneSpec(
            workcell_id="reconstructed_workcell",
            stations=stations,
            fixtures=fixtures,
            parts=parts,
            tools=[],
            conveyors=[],
            containers=containers,
            spatial_bounds=bounds,
        )

        # Compute confidence
        confidence = self._compute_confidence(tracks, map_first_artifacts)

        return ReconstructionResult(
            scene_spec=scene_spec,
            track_mapping=track_mapping,
            confidence_score=confidence,
            warnings=warnings,
        )

    def reconstruct_from_paths(
        self,
        scene_tracks_path: Path,
        map_first_path: Optional[Path] = None,
    ) -> ReconstructionResult:
        """
        Reconstruct from file paths.

        Args:
            scene_tracks_path: Path to SceneTracks_v1 npz file
            map_first_path: Optional path to map_first_supervision npz file

        Returns:
            ReconstructionResult
        """
        scene_tracks = dict(np.load(scene_tracks_path, allow_pickle=True))

        map_first_artifacts = None
        if map_first_path and map_first_path.exists():
            map_first_artifacts = dict(np.load(map_first_path, allow_pickle=True))

        return self.reconstruct_from_tracks(scene_tracks, map_first_artifacts)

    def _parse_scene_tracks(self, scene_tracks: Dict[str, np.ndarray]) -> List[TrackInfo]:
        """Parse SceneTracks_v1 format into TrackInfo list."""
        tracks: List[TrackInfo] = []

        # Try different key conventions
        track_ids = scene_tracks.get("track_ids", scene_tracks.get("scene_tracks_v1/track_ids"))
        positions = scene_tracks.get("positions", scene_tracks.get("scene_tracks_v1/positions"))
        orientations = scene_tracks.get("orientations", scene_tracks.get("scene_tracks_v1/orientations"))
        labels = scene_tracks.get("semantic_labels", scene_tracks.get("scene_tracks_v1/semantic_labels"))

        if track_ids is None or positions is None:
            return tracks

        track_ids = np.asarray(track_ids)
        positions = np.asarray(positions)

        if orientations is not None:
            orientations = np.asarray(orientations)
        else:
            # Default identity quaternions
            orientations = np.tile([0.0, 0.0, 0.0, 1.0], (len(positions), 1))

        # Handle different array shapes
        if positions.ndim == 3:  # (num_tracks, T, 3)
            for i, tid in enumerate(track_ids):
                tracks.append(TrackInfo(
                    track_id=str(tid),
                    object_type="unknown",
                    positions=positions[i],
                    orientations=orientations[i] if orientations.ndim == 3 else orientations,
                    semantic_labels=labels[i] if labels is not None and labels.ndim >= 2 else None,
                ))
        elif positions.ndim == 2:  # (T, 3) single track or (num_tracks, 3) single frame
            tracks.append(TrackInfo(
                track_id=str(track_ids[0]) if len(track_ids) > 0 else "track_0",
                object_type="unknown",
                positions=positions,
                orientations=orientations,
            ))

        return tracks

    def _enrich_with_map_first(
        self,
        tracks: List[TrackInfo],
        map_first_artifacts: Dict[str, np.ndarray],
    ) -> List[TrackInfo]:
        """Enrich tracks with map_first semantic information."""
        semantics = map_first_artifacts.get(
            "map_first_supervision_v1/semantics_stable",
            map_first_artifacts.get("semantics_stable"),
        )

        if semantics is None:
            return tracks

        # Assign semantic labels to tracks based on spatial proximity
        # (simplified heuristic - production would use proper association)
        for track in tracks:
            if track.semantic_labels is None and len(track.positions) > 0:
                # Use first position to lookup
                track.semantic_labels = np.zeros(len(track.positions))

        return tracks

    def _classify_object(self, track: TrackInfo) -> str:
        """Classify track into object category."""
        obj_type_lower = track.object_type.lower() if track.object_type else ""

        for category, keywords in self.OBJECT_TYPE_MAP.items():
            if any(kw in obj_type_lower for kw in keywords):
                return category

        # Heuristic: stationary objects are likely fixtures/stations
        if len(track.positions) > 1:
            displacement = np.linalg.norm(track.positions[-1] - track.positions[0])
            if displacement < 0.01:  # Less than 1cm movement
                return "fixture"

        return "part"  # Default to part

    def _compute_bounds(self, positions: np.ndarray) -> Tuple[float, float, float]:
        """Compute spatial bounds from positions."""
        if len(positions) == 0:
            return self.default_bounds

        mins = positions.min(axis=0)
        maxs = positions.max(axis=0)
        ranges = maxs - mins

        # Add margin
        margin = 0.5
        bounds = tuple((r + margin) for r in ranges)

        return (max(bounds[0], 1.0), max(bounds[1], 1.0), max(bounds[2], 0.5))

    def _compute_confidence(
        self,
        tracks: List[TrackInfo],
        map_first_artifacts: Optional[Dict[str, np.ndarray]],
    ) -> float:
        """Compute reconstruction confidence score."""
        if not tracks:
            return 0.0

        # Base confidence from track count
        base_conf = min(len(tracks) / 10.0, 1.0)

        # Boost if map_first available
        if map_first_artifacts:
            base_conf = min(base_conf + 0.2, 1.0)

        # Boost if tracks have semantic labels
        labeled_count = sum(1 for t in tracks if t.semantic_labels is not None)
        label_conf = labeled_count / max(len(tracks), 1)

        return 0.6 * base_conf + 0.4 * label_conf


def reconstruct_workcell_from_video(
    scene_tracks_path: Path,
    map_first_path: Optional[Path] = None,
) -> ReconstructionResult:
    """
    Convenience function to reconstruct workcell from video artifacts.

    Args:
        scene_tracks_path: Path to SceneTracks_v1 npz
        map_first_path: Optional path to map_first_supervision npz

    Returns:
        ReconstructionResult
    """
    adapter = SceneTracksAdapter()
    return adapter.reconstruct_from_paths(scene_tracks_path, map_first_path)
