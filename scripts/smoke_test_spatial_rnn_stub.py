#!/usr/bin/env python3
"""
Smoke test for spatial RNN stub.
"""
import json
import sys
from pathlib import Path

import numpy as np

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from src.vision.spatial_rnn_adapter import run_spatial_rnn, tensor_to_json_safe


def main():
    seq = [np.arange(4, dtype=float), np.ones(4, dtype=float) * 0.5, np.linspace(0.0, 1.0, 4, dtype=float)]
    fused1 = run_spatial_rnn(seq)
    fused2 = run_spatial_rnn(seq)
    assert np.allclose(fused1, fused2)
    json.dumps(tensor_to_json_safe(fused1))
    assert fused1.shape == seq[0].shape
    print("[smoke_test_spatial_rnn_stub] All checks passed.")


if __name__ == "__main__":
    main()
