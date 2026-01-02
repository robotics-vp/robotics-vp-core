import sys
import os
from typing import Dict, Any, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sima2.heuristic_segmenter import HeuristicSegmenter

def make_dense_rollout():
    # Simulate a dense trace:
    # 0-5: Approach (vel > 0, contact=F, width=0.08)
    # 6-10: Grasp (vel=0, contact=T, width closing)
    # 11-15: Pull (vel > 0, contact=T, width closed)
    
    signals = []
    
    # Approach
    for t in range(0, 6):
        signals.append({
            "timestep": t,
            "gripper_width": 0.08,
            "ee_velocity": 0.01,
            "contact": False,
            "object": "drawer_handle"
        })
        
    # Grasp (closing)
    for t in range(6, 11):
        width = 0.08 - (0.01 * (t - 6)) # 0.08 -> 0.04
        signals.append({
            "timestep": t,
            "gripper_width": width,
            "ee_velocity": 0.0,
            "contact": True,
            "object": "drawer_handle"
        })
        
    # Pull
    for t in range(11, 16):
        signals.append({
            "timestep": t,
            "gripper_width": 0.04,
            "ee_velocity": 0.1,
            "contact": True,
            "object": "drawer_handle"
        })
        
    return {
        "primitives": signals, # Using 'primitives' key as source for now
        "events": []
    }

def test_heuristic_segmentation():
    print("Testing HeuristicSegmenter...")
    segmenter = HeuristicSegmenter()
    rollout = make_dense_rollout()
    
    segments = segmenter.segment(rollout)
    
    print(f"Detected {len(segments)} segments")
    for seg in segments:
        print(f"  {seg.start_t}-{seg.end_t}: {seg.label} (conf={seg.confidence})")
        
    # Assertions
    # Expect 3 segments roughly: approach, grasp, pull
    # Note: HeuristicSegmenter might merge if logic is sticky or thresholds differ.
    
    labels = [s.label for s in segments]
    print(f"Labels: {labels}")
    
    assert "approach" in labels
    assert "grasp" in labels
    assert "pull" in labels
    
    print("[smoke_test_sima2_heuristic_segmentation] PASS")

if __name__ == "__main__":
    test_heuristic_segmentation()
