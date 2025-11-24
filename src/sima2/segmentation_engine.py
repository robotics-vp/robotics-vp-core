"""
Config-driven segmentation engine for SIMA-2 rollouts.
"""
import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.sima2.config import extract_provenance, get_segmentation_config, load_sima2_config
from src.sima2.semantic_tag_propagator import SegmentationBuilder
from src.sima2.tags.semantic_tags import SegmentBoundaryTag, SubtaskTag
from src.sima2.heuristic_segmenter import HeuristicSegmenter, DetectedPrimitive


@dataclass
class Segment:
    segment_id: str
    episode_id: str
    label: str
    start_t: int
    end_t: int
    outcome: str
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "segment_id": self.segment_id,
            "episode_id": self.episode_id,
            "label": self.label,
            "start_t": self.start_t,
            "end_t": self.end_t,
            "outcome": self.outcome,
            "confidence": float(self.confidence),
            "metadata": self.metadata,
        }
        if self.provenance:
            data["provenance"] = self.provenance
        return data


class SegmentationEngine:
    """
    Converts SIMA-2 rollouts into segments using config-driven thresholds.

    Mode selection (via config):
    - use_heuristic_segmenter=True: Use physics-based heuristic detection
    - use_heuristic_segmenter=False: Use existing boundary-based builder (default for compatibility)
    """

    def __init__(self, config_path: Optional[str] = None, segmentation_config: Optional[Dict[str, Any]] = None):
        self.config = load_sima2_config(config_path)
        self.raw_segmentation_config = segmentation_config or {}
        self.segmentation_config = get_segmentation_config(segmentation_config or self.config)
        self.builder = SegmentationBuilder(self.segmentation_config)

        # Flag-gated heuristic segmenter
        # Check both the raw config and the normalized config for the flag
        self.use_heuristic = bool(
            self.raw_segmentation_config.get("use_heuristic_segmenter", False) or
            self.segmentation_config.get("use_heuristic_segmenter", False)
        )
        self.heuristic_segmenter = HeuristicSegmenter(self.segmentation_config) if self.use_heuristic else None

    def segment_rollout(self, rollout: Dict[str, Any]) -> Dict[str, Any]:
        """
        Segment a rollout and return enriched rollout plus segment artifacts.

        Depending on use_heuristic flag:
        - If True: Use heuristic segmenter (physics-based)
        - If False: Use existing boundary builder (compatibility mode)
        """
        rollout_copy = copy.deepcopy(rollout)
        episode_id = rollout_copy.get("episode_id") or rollout_copy.get("task") or "sima2_episode"
        provenance = extract_provenance(rollout_copy, self.config)

        if self.use_heuristic and self.heuristic_segmenter:
            # Heuristic mode: detect from raw physics
            detected_prims = self.heuristic_segmenter.segment(rollout_copy)
            segments = self._segments_from_detected(
                detected_prims, episode_id, provenance, rollout_copy.get("metadata", {})
            )
            boundaries = self._boundaries_from_segments(segments)
            subtask_tags = []
        else:
            # Compatibility mode: use existing builder
            boundaries, subtask_tags = self.builder.build(rollout_copy)
            segments = self._segments_from_boundaries(
                boundaries, subtask_tags, episode_id=episode_id, provenance=provenance, metadata=rollout_copy.get("metadata", {})
            )

        rollout_copy["segments"] = [seg.to_dict() for seg in segments]
        rollout_copy.setdefault("metadata", {})
        rollout_copy["metadata"].update(provenance)
        rollout_copy["metadata"]["segmentation_mode"] = "heuristic" if self.use_heuristic else "boundary"
        rollout_copy.update(provenance)
        seg_events = self._segments_to_events(segments)
        if rollout_copy.get("events"):
            rollout_copy["events"] = seg_events + list(rollout_copy.get("events") or [])
        else:
            rollout_copy["events"] = seg_events

        return {
            "segments": segments,
            "segment_boundaries": boundaries,
            "subtask_tags": subtask_tags,
            "rollout": rollout_copy,
        }

    def _segments_from_boundaries(
        self,
        boundaries: List[SegmentBoundaryTag],
        subtask_tags: List[SubtaskTag],
        episode_id: str,
        provenance: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> List[Segment]:
        subtask_lookup = {st.segment_id: st.subtask_label for st in subtask_tags}
        segments: Dict[str, Dict[str, Any]] = {}
        for b in boundaries:
            seg = segments.setdefault(
                b.segment_id,
                {
                    "start": b.timestep,
                    "end": b.timestep,
                    "label": b.subtask_label or subtask_lookup.get(b.segment_id) or "segment",
                    "failure": False,
                    "recovery": False,
                },
            )
            if b.reason == "start":
                seg["start"] = min(seg["start"], b.timestep)
            seg["end"] = max(seg["end"], b.timestep)
            if b.reason == "failure":
                seg["failure"] = True
            if b.reason == "recovery":
                seg["recovery"] = True

        objects_present = ((metadata or {}).get("objects_present") or [])
        segment_list: List[Segment] = []
        for idx, (seg_id, seg_meta) in enumerate(sorted(segments.items(), key=lambda item: item[0])):
            outcome = "success"
            if seg_meta.get("failure") and seg_meta.get("recovery"):
                outcome = "recovered"
            elif seg_meta.get("failure"):
                outcome = "failure"
            elif seg_meta.get("recovery"):
                outcome = "recovery"
            seg_metadata = {
                "objects_present": list(objects_present),
                "segment_index": idx,
                "recovery_observed": seg_meta.get("recovery", False),
                "failure_observed": seg_meta.get("failure", False),
            }
            seg_metadata.update(provenance)
            segment_list.append(
                Segment(
                    segment_id=seg_id,
                    episode_id=episode_id,
                    label=str(seg_meta.get("label") or "segment"),
                    start_t=int(seg_meta.get("start")),
                    end_t=int(seg_meta.get("end")),
                    outcome=outcome,
                    confidence=0.9,
                    metadata=seg_metadata,
                    provenance=provenance,
                )
            )
        return sorted(segment_list, key=lambda s: (s.start_t, s.segment_id))

    def _segments_from_detected(
        self,
        detected_prims: List[DetectedPrimitive],
        episode_id: str,
        provenance: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> List[Segment]:
        """Convert heuristically-detected primitives into Segment objects."""
        segments: List[Segment] = []
        objects_present = metadata.get("objects_present", [])

        for idx, prim in enumerate(detected_prims):
            outcome = "success"
            if prim.failure and prim.recovery:
                outcome = "recovered"
            elif prim.failure:
                outcome = "failure"
            elif prim.recovery:
                outcome = "recovery"

            seg_id = f"{episode_id}_seg{idx}_{prim.label}"
            seg_metadata = {
                "objects_present": list(objects_present),
                "segment_index": idx,
                "recovery_observed": prim.recovery,
                "failure_observed": prim.failure,
                "object_name": prim.object_name,
            }
            if prim.metadata:
                seg_metadata.update(prim.metadata)
            seg_metadata.update(provenance)

            segments.append(
                Segment(
                    segment_id=seg_id,
                    episode_id=episode_id,
                    label=prim.label,
                    start_t=prim.start_t,
                    end_t=prim.end_t,
                    outcome=outcome,
                    confidence=prim.confidence,
                    metadata=seg_metadata,
                    provenance=provenance,
                )
            )

        return segments

    def _segments_to_events(self, segments: List[Segment]) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        for seg in segments:
            events.append(
                {
                    "primitive_id": seg.segment_id,
                    "action": seg.label,
                    "timestep": seg.start_t,
                    "end_timestep": seg.end_t,
                    "success": seg.outcome in {"success", "recovered"},
                    "status": seg.outcome,
                    "tags": [seg.label, seg.outcome],
                    "metadata": seg.metadata,
                    "segment_id": seg.segment_id,
                    "sima2_backend_id": seg.provenance.get("sima2_backend_id") if seg.provenance else None,
                    "sima2_model_version": seg.provenance.get("sima2_model_version") if seg.provenance else None,
                }
            )
        return events

    def _boundaries_from_segments(self, segments: List[Segment]) -> List[SegmentBoundaryTag]:
        boundaries: List[SegmentBoundaryTag] = []
        for seg in segments:
            boundaries.append(
                SegmentBoundaryTag(
                    episode_id=seg.episode_id,
                    segment_id=seg.segment_id,
                    timestep=seg.start_t,
                    reason="start",
                    subtask_label=seg.label,
                )
            )
            boundaries.append(
                SegmentBoundaryTag(
                    episode_id=seg.episode_id,
                    segment_id=seg.segment_id,
                    timestep=seg.end_t,
                    reason="end",
                    subtask_label=seg.label,
                )
            )
            meta = seg.metadata or {}
            if meta.get("failure_observed") or seg.outcome == "failure":
                boundaries.append(
                    SegmentBoundaryTag(
                        episode_id=seg.episode_id,
                        segment_id=seg.segment_id,
                        timestep=seg.end_t,
                        reason="failure",
                        subtask_label=seg.label,
                    )
                )
            if meta.get("recovery_observed") or seg.outcome in {"recovery", "recovered"}:
                boundaries.append(
                    SegmentBoundaryTag(
                        episode_id=seg.episode_id,
                        segment_id=seg.segment_id,
                        timestep=seg.end_t,
                        reason="recovery",
                        subtask_label=seg.label,
                    )
                )
        return sorted(boundaries, key=lambda b: (b.timestep, b.segment_id, b.reason))
