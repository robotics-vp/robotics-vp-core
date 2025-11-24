"""
Phase I SIMA-2 segmenter dataset loader.

Provides deterministic segment descriptors derived from Stage 2 primitives,
SIMA-2 stress outputs, and ROS → Stage2 bridge metadata.
"""
from typing import Any, Dict, List

from src.datasets.base import Phase1DatasetBase, set_deterministic_seeds


class Sima2SegmenterDataset(Phase1DatasetBase):
    name = "sima2_segmenter_phase1"

    def __init__(self, *args, seed: int = 0, **kwargs) -> None:
        set_deterministic_seeds(seed)
        super().__init__(*args, seed=seed, **kwargs)

    def _augment_sample(self, sample: Dict[str, Any], idx: int) -> Dict[str, Any]:
        seg = sample.get("stage2_segments", {})
        stress = sample.get("sima2_stress", {})
        ros = sample.get("ros_stage2", {})

        sample["segments"] = self._build_segments(seg, stress, ros, idx)
        return sample

    def _build_segments(
        self, seg: Dict[str, Any], stress: Dict[str, Any], ros: Dict[str, Any], idx: int
    ) -> List[Dict[str, Any]]:
        tags: List[str] = []
        for candidate in (seg.get("tags") or [], stress.get("tag"), ros.get("stage2_tags") or []):
            if isinstance(candidate, list):
                tags.extend([str(t) for t in candidate])
            elif candidate:
                tags.append(str(candidate))
        tags = sorted(set(tags))
        base_segment = {
            "segment_id": seg.get("segment_id", f"seg_{idx}"),
            "tags": tags,
            "risk_level": seg.get("risk_level", 0.0),
            "stress_severity": stress.get("severity", 0.0),
            "ros_source": ros.get("source"),
        }
        return [base_segment]

