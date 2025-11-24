#!/usr/bin/env python3
"""
End-to-end smoke: TFD → ConditionVector → ConditionedVisionAdapter.

Asserts:
- ConditionVector reflects TFD safety intent (lower risk tolerance)
- Base vision latent (z_v) remains unchanged across conditions
- Risk map scales with safety emphasis and stays bounded
- Deterministic outputs for fixed inputs/seeds
"""
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.observation.condition_vector_builder import ConditionVectorBuilder
from src.tfd.compiler import TextFrontDoor
from src.vision.conditioned_adapter import ConditionedVisionAdapter
from src.vision.interfaces import VisionFrame
from src.vision.regnet_backbone import flatten_pyramid


def build_condition(text: str):
    builder = ConditionVectorBuilder()
    tfd = TextFrontDoor()
    instruction = tfd.process_instruction(text)
    return builder.build(
        episode_config={"task_id": "fragile_task", "env_id": "sim", "backend_id": "isaac"},
        econ_state={"target_mpl": 50.0, "current_wage_parity": 1.0, "energy_budget_wh": 25.0},
        curriculum_phase="warmup",
        sima2_trust={"trust_score": 0.6},
        datapack_metadata={"tags": {"FragilityTag": 1.0}},
        episode_step=5,
        overrides=None,
        econ_slice=None,
        semantic_tags=None,
        recap_scores=None,
        trust_summary=None,
        episode_metadata={"sampler_strategy": "balanced"},
        advisory_context=None,
        tfd_instruction=instruction.to_dict() if hasattr(instruction, "to_dict") else instruction,
        enable_tfd_integration=True,
    )


def make_frame() -> VisionFrame:
    return VisionFrame(
        backend="isaac",
        backend_id="isaac",
        task_id="fragile_task",
        episode_id="ep_tfd_chain",
        timestep=0,
        width=4,
        height=4,
        channels=3,
        camera_intrinsics={"resolution": [4, 4]},
        camera_extrinsics={"frame": "world"},
    )


def flatten_z_v(z_v: dict) -> np.ndarray:
    return flatten_pyramid(z_v).astype(np.float32)


def main() -> int:
    cautious_cv = build_condition("Be very careful around fragile items")
    neutral_cv = build_condition("Go fast")
    frame = make_frame()
    adapter = ConditionedVisionAdapter(config={"enable_conditioning": True, "feature_dim": 8})

    cautious_out = adapter.forward(frame, cautious_cv)
    neutral_out = adapter.forward(frame, neutral_cv)

    # Base latent unchanged
    assert np.allclose(flatten_z_v(cautious_out["z_v"]), flatten_z_v(neutral_out["z_v"])), "z_v must be invariant to conditioning"

    assert cautious_cv.ood_risk_level < neutral_cv.ood_risk_level, "TFD safety intent should lower risk tolerance"

    # Risk emphasis increases under cautious instruction
    cautious_norm = float(np.linalg.norm(cautious_out["risk_map"]))
    neutral_norm = float(np.linalg.norm(neutral_out["risk_map"]))
    assert cautious_norm > neutral_norm, "Risk map should scale up for safety-focused TFD instruction"

    # Determinism for fixed inputs
    cautious_repeat = adapter.forward(frame, cautious_cv)
    assert np.allclose(cautious_out["risk_map"], cautious_repeat["risk_map"]), "Conditioned vision must be deterministic"
    assert np.allclose(flatten_z_v(cautious_out["z_v"]), flatten_z_v(cautious_repeat["z_v"])), "Base latent determinism failed"

    # Bounds
    assert np.max(cautious_out["risk_map"]) <= adapter.max_scale * 0.5 + 1e-6, "Risk map should stay within bounded scale"

    print("[smoke_test_tfd_condition_chain] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
