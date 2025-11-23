#!/usr/bin/env python3
"""
Smoke test for ObservationComponents + adapter integration.
"""
import json
import sys
from pathlib import Path

import numpy as np

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from src.observation.adapter import ObservationAdapter
from src.observation.components import ObservationComponents
from src.policies.registry import build_all_policies
from src.vision.interfaces import VisionFrame, VisionLatent


def build_inputs():
    frame = VisionFrame(
        backend="stub",
        task_id="task_components",
        episode_id="ep_components",
        timestep=0,
        state_digest="digest_components",
        camera_intrinsics={"fov_deg": 90.0},
        camera_extrinsics={"frame": "world"},
    )
    latent = VisionLatent(
        backend="stub",
        task_id=frame.task_id,
        episode_id=frame.episode_id,
        timestep=frame.timestep,
        latent=[0.1, 0.2, 0.3],
        model_name="vision-stub",
    )
    descriptor = {
        "pack_id": "dp_components",
        "sampling_metadata": {"strategy": "balanced", "phase": "warmup"},
        "semantic_tags": ["frontier"],
    }
    adapter_kwargs = {
        "reward_scalar": 1.0,
        "reward_components": {"mpl_component": 0.5},
        "econ_vector": None,
        "semantic_snapshot": None,
        "recap_scores": None,
        "descriptor": descriptor,
        "episode_metadata": {"episode_id": "ep_components", "curriculum_phase": "warmup"},
    }
    return frame, latent, adapter_kwargs


def main():
    policies = build_all_policies()
    adapter = ObservationAdapter(
        policies,
        config={"use_condition_vector": True, "use_observation_components": True},
        use_condition_vector=True,
    )
    frame, latent, adapter_kwargs = build_inputs()

    observation, condition = adapter.build_observation_and_condition(
        vision_frame=frame,
        vision_latent=latent,
        reward_scalar=adapter_kwargs["reward_scalar"],
        reward_components=adapter_kwargs["reward_components"],
        econ_vector=adapter_kwargs["econ_vector"],
        semantic_snapshot=adapter_kwargs["semantic_snapshot"],
        recap_scores=adapter_kwargs["recap_scores"],
        descriptor=adapter_kwargs["descriptor"],
        episode_metadata=adapter_kwargs["episode_metadata"],
    )
    tensor_legacy = adapter.to_policy_tensor(observation, condition=condition, include_condition=True)

    observation2, condition2, tensor_components, components = adapter.build_observation_and_components(
        vision_frame=frame,
        vision_latent=latent,
        include_condition=True,
        use_components=True,
        **adapter_kwargs,
    )
    assert np.allclose(tensor_legacy, tensor_components), "Component path must preserve tensor outputs"
    comp_dict = components.to_dict()
    json.dumps(comp_dict)
    restored = ObservationComponents.from_dict(comp_dict)
    assert restored.to_dict() == comp_dict

    _, _, tensor_no_components, components_empty = adapter.build_observation_and_components(
        vision_frame=frame,
        vision_latent=latent,
        include_condition=True,
        use_components=False,
        **adapter_kwargs,
    )
    assert np.allclose(tensor_legacy, tensor_no_components)
    assert components_empty.to_dict().get("vision_features") is None

    print("[smoke_test_observation_components] All checks passed.")


if __name__ == "__main__":
    main()
