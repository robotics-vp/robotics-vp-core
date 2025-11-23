#!/usr/bin/env python3
"""
Smoke test for SIMA-2 OODTag and RecoveryTag firing semantics.
Deterministic and JSON-safe by construction.
"""
from pathlib import Path
import sys

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from src.sima2.tags.ood_recovery_tags import detect_ood_from_segment, detect_recovery_from_segments


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_visual_ood():
    seg = {
        "segment_id": "seg_v",
        "start_t": 0,
        "end_t": 4,
        "outcome": "success",
        "metadata": {"embedding_distance": 0.96},
    }
    tag = detect_ood_from_segment(seg, config={"trust_matrix": {"OODTag": {"trust_score": 0.9}}})
    _assert(tag is not None, "Expected visual OODTag")
    _assert(tag.source == "visual", "Expected visual source")
    _assert(abs(tag.severity - 1.0) < 1e-6, "Expected critical visual severity")


def test_temporal_ood_and_suppression():
    seg = {
        "segment_id": "seg_t",
        "start_t": 0,
        "end_t": 40,
        "outcome": "success",
        "metadata": {"duration": 40, "label": "grasp"},
    }
    config = {"trust_matrix": {"OODTag": {"trust_score": 0.9}}, "mean_durations": {"grasp": 5}}
    tag = detect_ood_from_segment(seg, config=config)
    _assert(tag is not None, "Expected temporal OODTag")
    _assert(tag.source == "temporal", "Expected temporal source")
    _assert(tag.severity >= 1.0, "Expected max temporal severity")

    suppressed = detect_ood_from_segment(seg, config={"trust_matrix": {"OODTag": {"trust_score": 0.1}}})
    _assert(suppressed is None, "Expected suppression when trust <= 0.5")


def test_recovery_pattern():
    segments = [
        {"segment_id": "seg0", "start_t": 0, "end_t": 5, "outcome": "failure", "metadata": {"failure_observed": True, "failure_severity": "critical"}},
        {"segment_id": "seg1", "start_t": 5, "end_t": 9, "outcome": "recovery", "label": "regrasp", "metadata": {"recovery_observed": True, "duration": 4}},
        {"segment_id": "seg2", "start_t": 9, "end_t": 12, "outcome": "success", "metadata": {}},
    ]
    tags = detect_recovery_from_segments(segments, config={"trust_matrix": {"RecoveryTag": {"trust_score": 0.9}}})
    _assert(len(tags) == 1, "Expected one RecoveryTag")
    _assert(tags[0].value_add == "high", "Expected high value-add for fast critical recovery")
    _assert(tags[0].correction_type == "regrasp", "Expected correction type passthrough")

    suppressed = detect_recovery_from_segments(segments, config={"trust_matrix": {"RecoveryTag": {"trust_score": 0.1}}})
    _assert(len(suppressed) == 0, "Expected suppression when recovery trust low")


def main():
    test_visual_ood()
    test_temporal_ood_and_suppression()
    test_recovery_pattern()
    print("[smoke_test_sima2_tags] PASS")


if __name__ == "__main__":
    main()
