"""
Config-driven segmentation engine for SIMA-2 rollouts.
"""
import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.sima2.config import extract_provenance, get_segmentation_config, load_sima2_config
from src.sima2.semantic_tag_propagator import SegmentationBuilder
from src.sima2.tags.semantic_tags import SegmentBoundaryTag, SubtaskTag


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
    """

    def __init__(self, config_path: Optional[str] = None, segmentation_config: Optional[Dict[str, Any]] = None):
        self.config = load_sima2_config(config_path)
        self.segmentation_config = get_segmentation_config(segmentation_config or self.config)
        self.builder = SegmentationBuilder(self.segmentation_config)

    def segment_rollout(self, rollout: Dict[str, Any]) -> Dict[str, Any]:
        """
        Segment a rollout and return enriched rollout plus segment artifacts.
        """
        rollout_copy = copy.deepcopy(rollout)
        episode_id = rollout_copy.get("episode_id") or rollout_copy.get("task") or "sima2_episode"
        provenance = extract_provenance(rollout_copy, self.config)
        boundaries, subtask_tags = self.builder.build(rollout_copy)
        segments = self._segments_from_boundaries(
            boundaries, subtask_tags, episode_id=episode_id, provenance=provenance, metadata=rollout_copy.get("metadata", {})
        )

        rollout_copy["segments"] = [seg.to_dict() for seg in segments]
        rollout_copy.setdefault("metadata", {})
        rollout_copy["metadata"].update(provenance)
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
