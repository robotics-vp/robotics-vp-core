#!/usr/bin/env python3
"""
Smoke test for json_safe utility and downstream serialization targets.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.json_safe import to_json_safe, json_dumps_safe  # type: ignore
from src.vla.transformer_planner import VLAPlan  # type: ignore
from src.diffusion.real_video_diffusion_stub import DiffusionProposal, proposal_to_dict  # type: ignore

try:
    import torch  # type: ignore
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


def test_vla_plan_serialization():
    plan = VLAPlan(
        skill_sequence=[0, 1],
        skill_params=[np.array([0.1, 0.2])],
        timing_horizons=[5],
        confidence=[0.9, 0.8],
        instruction="open drawer safely",
    )
    safe_plan = to_json_safe(plan)
    json_str = json.dumps(safe_plan)
    assert json_str
    assert "skill_sequence" in json_str


def test_diffusion_proposal_serialization():
    proposal = DiffusionProposal(
        proposal_id="p1",
        episode_id="ep1",
        media_refs=["/tmp/a.mp4"],
        augmentation_type="safe_speed",
        semantic_tags=["safe", "fragile"],
        objective_preset="balanced",
        energy_profile="BASE",
        econ_context={"wage_human": 18.0},
        confidence=0.7,
        estimated_novelty=0.5,
        rationale="stub",
        timestamp=time.time(),
    )
    safe_proposal = proposal_to_dict(proposal)
    json_str = json.dumps(safe_proposal)
    assert "proposal_id" in json_str


def test_tensor_serialization():
    tensor = None
    if TORCH_AVAILABLE:
        tensor = torch.randn(2, 2)
        converted = to_json_safe(tensor, include_tensors=True)
        json_str = json.dumps(converted)
        assert json_str
    else:
        converted = to_json_safe({"placeholder": [1, 2, 3]})
    assert converted is not None


def main():
    test_vla_plan_serialization()
    test_diffusion_proposal_serialization()
    test_tensor_serialization()
    # Round-trip example via helper
    payload = {"nested": {"tensor": np.array([1.0, 2.0])}}
    dumped = json_dumps_safe(payload)
    assert dumped
    print("[smoke] json_safe utility verified.")


if __name__ == "__main__":
    main()
