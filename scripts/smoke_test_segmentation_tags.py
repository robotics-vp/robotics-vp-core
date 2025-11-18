#!/usr/bin/env python3
"""
Smoke test for semantic segmentation tags generation.
"""
import tempfile
from pathlib import Path
import sys

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from src.sima2.semantic_tag_propagator import SegmentationBuilder
from src.sima2.tags.semantic_tags import SegmentBoundaryTag


def _build_datapack():
    return {
        "episode_id": "ep_segment_demo",
        "task": "drawer_vase",
        "primitives": [
            {"timestep": 0, "object": "drawer", "action": "approach", "risk": "low"},
            {"timestep": 3, "object": "drawer", "action": "pull", "risk": "medium"},
            {"timestep": 6, "object": "vase_inside", "action": "avoid", "risk": "high", "status": "failure"},
            {"timestep": 9, "object": "vase_inside", "action": "recover", "risk": "medium", "status": "recovery"},
        ],
    }


def _assert_sorted(boundaries):
    for i in range(1, len(boundaries)):
        prev = boundaries[i - 1]
        cur = boundaries[i]
        assert (prev.timestep, prev.segment_id) <= (cur.timestep, cur.segment_id)


def main():
    with tempfile.TemporaryDirectory() as _:
        builder = SegmentationBuilder()
        datapack = _build_datapack()
        b1, s1 = builder.build(datapack)
        b2, s2 = builder.build(datapack)

        assert b1 and s1, "Expected segmentation output"
        assert len(set(bt.segment_id for bt in b1)) >= 2, "Expected multiple segments"
        _assert_sorted(b1)
        for b in b1:
            assert isinstance(b, SegmentBoundaryTag)
            assert b.reason in {"start", "end", "failure", "recovery"}
        assert b1 == b2 and s1 == s2, "Determinism violated"
        print("[smoke_test_segmentation_tags] PASS")


if __name__ == "__main__":
    main()
