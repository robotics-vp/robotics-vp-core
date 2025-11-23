#!/usr/bin/env python3
"""
Smoke test for RegNet + BiFPN stubs.
"""
import json
import sys
from pathlib import Path

import numpy as np

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from src.vision.bifpn_fusion import fuse_feature_pyramid
from src.vision.interfaces import VisionFrame
from src.vision.regnet_backbone import build_regnet_feature_pyramid, flatten_pyramid, pyramid_to_json_safe


def main():
    frame = VisionFrame(backend="stub", task_id="task_regnet", episode_id="ep_regnet", timestep=0, state_digest="sig_regnet")
    pyramid1 = build_regnet_feature_pyramid(frame, feature_dim=6)
    pyramid2 = build_regnet_feature_pyramid(frame, feature_dim=6)
    assert pyramid1.keys() == pyramid2.keys()
    for lvl in pyramid1:
        assert np.allclose(pyramid1[lvl], pyramid2[lvl])
    fused = fuse_feature_pyramid(pyramid1)
    assert set(fused.keys()) == set(pyramid1.keys())
    for lvl in fused:
        assert fused[lvl].shape == pyramid1[lvl].shape
    flat = flatten_pyramid(fused)
    assert flat.size == sum(len(v) for v in fused.values())
    json.dumps(pyramid_to_json_safe(fused))
    print("[smoke_test_regnet_bifpn_stub] All checks passed.")


if __name__ == "__main__":
    main()
