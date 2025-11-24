#!/usr/bin/env python3
"""
Smoke test for heuristic SIMA-2 segmentation (Stage 5.1).
"""
import copy
from pathlib import Path
import sys
from typing import Iterable, Sequence, Tuple

REPO_ROOT = Path(__file__).parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.sima2.client import Sima2Client
from src.sima2.segmentation_engine import SegmentationEngine


def _intervals_from_primitives(prims: Sequence[dict]) -> Sequence[Tuple[int, int]]:
    sorted_prims = sorted(prims or [], key=lambda p: int(p.get("timestep", 0)))
    intervals = []
    for idx, prim in enumerate(sorted_prims):
        start = int(prim.get("timestep", 0))
        next_start = int(sorted_prims[idx + 1].get("timestep", start + 1)) if idx + 1 < len(sorted_prims) else start + 2
        end = max(start + 1, next_start)
        intervals.append((start, end))
    return intervals


def _coverage(intervals: Iterable[Tuple[int, int]]) -> set:
    covered = set()
    for start, end in intervals:
        for t in range(int(start), int(end)):
            covered.add(t)
    return covered


def _compute_iou(heuristic_segments: Sequence, ground_truth_intervals: Sequence[Tuple[int, int]]) -> float:
    heur_intervals = []
    for seg in heuristic_segments:
        if hasattr(seg, "start_t"):
            start = int(getattr(seg, "start_t"))
            end = int(getattr(seg, "end_t"))
        else:
            seg_dict = seg if isinstance(seg, dict) else {}
            start = int(seg_dict.get("start_t", 0))
            end = int(seg_dict.get("end_t", start + 1))
        heur_intervals.append((start, end))
    if not heur_intervals and not ground_truth_intervals:
        return 1.0
    union = _coverage(heur_intervals) | _coverage(ground_truth_intervals)
    if not union:
        return 1.0
    intersection = _coverage(heur_intervals) & _coverage(ground_truth_intervals)
    return len(intersection) / len(union)


def run_single_template(template: str) -> None:
    client = Sima2Client(task_id="drawer_open", template=template, seed=0)
    rollout = client.run_episode({"episode_index": 0, "seed": 0, "template": template})
    engine = SegmentationEngine(segmentation_config={"use_heuristic_segmenter": True, "temporal_decay_window": 2})

    result_a = engine.segment_rollout(rollout)
    result_b = engine.segment_rollout(rollout)
    assert result_a["segments"], f"No segments generated for template={template}"
    assert result_a["segment_boundaries"], "Heuristic path must emit boundaries"
    signature_a = [(s.start_t, s.end_t, s.label) for s in result_a["segments"]]
    signature_b = [(s.start_t, s.end_t, s.label) for s in result_b["segments"]]
    assert signature_a == signature_b, "Segmentation must be deterministic"

    gt_intervals = _intervals_from_primitives(rollout.get("primitives") or [])
    iou = _compute_iou(result_a["segments"], gt_intervals)
    assert iou >= 0.9, f"Heuristic IoU too low for {template}: {iou}"

    rollout_no_labels = copy.deepcopy(rollout)
    for prim in rollout_no_labels.get("primitives", []):
        prim.pop("action", None)
    no_label_segments = engine.segment_rollout(rollout_no_labels)["segments"]
    assert no_label_segments, "Segmentation should not depend on client primitive labels"


def main():
    for template in ("success", "failure", "recovery", "mixed"):
        run_single_template(template)
    print("[smoke_test_segmentation_real] PASS")


if __name__ == "__main__":
    main()
