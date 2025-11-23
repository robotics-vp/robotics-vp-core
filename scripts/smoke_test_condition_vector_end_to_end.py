#!/usr/bin/env python3
"""
Smoke test for end-to-end ConditionVector plumbing.

Validates:
- Deterministic ConditionVector build
- JSON-safe serialization
- Threading through ObservationAdapter + PolicyObservationBuilder
- Attachment into TrunkNet features
- Safe fallback when the condition vector is disabled
"""
import json
import numpy as np
import torch

from src.observation.condition_vector_builder import ConditionVectorBuilder
from src.observation.adapter import ObservationAdapter
from src.vision.policy_observation_builder import PolicyObservationBuilder
from src.vision.interfaces import VisionFrame, VisionLatent
from src.rl.trunk_net import TrunkNet


class StubEncoder:
    mode = "stub"
    backbone_name = "stub_encoder"

    def encode(self, frame: VisionFrame) -> VisionLatent:
        latent = [0.1, 0.2, 0.3, frame.timestep * 0.01]
        return VisionLatent(
            backend=frame.backend,
            task_id=frame.task_id,
            episode_id=frame.episode_id,
            timestep=frame.timestep,
            latent=latent,
            model_name=self.backbone_name,
            metadata={"source": "stub"},
        )


class StubSemanticSnapshot:
    def __init__(self) -> None:
        self.semantic_tags = [{"fragile": 1.0}, "frontier"]
        self.metadata = {"ood_score": 0.2, "recovery_score": 0.1}


class StubRecapScores:
    def __init__(self) -> None:
        self.advantage_bin_probs_mean = [0.1, 0.2, 0.7]
        self.metric_distributions = {"quality": [0.2, 0.3, 0.5]}
        self.metadata = {"value_supports": {"quality": (0.0, 1.0)}}
        self.recap_goodness_score = 0.72


class StubEconVector:
    def __init__(self) -> None:
        self.mpl_units_per_hour = 12.0
        self.wage_parity = 0.9
        self.energy_cost = 1.5
        self.damage_cost = 0.05
        self.source_domain = "stub"
        self.metadata = {"econ_slice": True}


def _build_adapter(use_condition_vector: bool) -> ObservationAdapter:
    trust_loader = lambda: {"fragile": {"trust_score": 0.8}, "frontier": {"trust_score": 0.7}}
    adapter_cfg = {"use_condition_vector": use_condition_vector, "condition_vector": {"skill_mode_order": ["frontier_exploration", "safety_critical", "efficiency_throughput", "recovery_heavy", "default"]}}
    return ObservationAdapter(policy_registry=None, trust_matrix_loader=trust_loader, recap_loader=None, config=adapter_cfg)


def run_enabled_path():
    adapter = _build_adapter(use_condition_vector=True)
    encoder = StubEncoder()
    builder = PolicyObservationBuilder(encoder, observation_adapter=adapter, use_observation_adapter=True)

    frame = VisionFrame(
        backend="pybullet",
        task_id="task_stub",
        episode_id="ep_001",
        timestep=5,
        backend_id="pybullet_v1",
        camera_intrinsics={"fx": 1.0},
        camera_extrinsics={"tx": 0.0},
        state_digest="digest_stub",
    )
    state_summary = {"speed": 0.6, "errors": 1}

    adapter_kwargs = {
        "reward_scalar": 1.0,
        "reward_components": {"mpl_component": 0.5, "error_penalty": 0.1},
        "econ_vector": StubEconVector(),
        "semantic_snapshot": StubSemanticSnapshot(),
        "recap_scores": StubRecapScores(),
        "descriptor": {
            "pack_id": "pack_stub",
            "semantic_tags": ["frontier"],
            "sampling_metadata": {"strategy": "frontier_prioritized", "frontier_score": 2.0},
            "metadata": {"backend_id": "pybullet", "pack_tier": 2},
        },
        "episode_metadata": {"episode_id": "ep_001", "sampler_strategy": "frontier_prioritized"},
        "raw_env_obs": {"t": 5, "completed": 0, "attempts": 1, "errors": 0},
    }

    features_a = builder.build_policy_features(
        frame,
        state_summary,
        use_observation_adapter=True,
        adapter_kwargs=adapter_kwargs,
        use_condition_vector=True,
    )
    features_b = builder.build_policy_features(
        frame,
        state_summary,
        use_observation_adapter=True,
        adapter_kwargs=adapter_kwargs,
        use_condition_vector=True,
    )
    condition_a = features_a["condition_vector"]
    condition_b = features_b["condition_vector"]

    assert condition_a.to_dict() == condition_b.to_dict(), "ConditionVector should be deterministic"
    assert np.allclose(condition_a.to_vector(), condition_b.to_vector()), "ConditionVector numeric form should be deterministic"
    json.dumps(condition_a.to_dict())

    trunk = TrunkNet(
        vision_dim=len(features_a["adapter_observation"].vision.latent),
        state_dim=len(state_summary),
        condition_dim=len(condition_a.to_vector()),
        hidden_dim=32,
        use_condition_film=True,
    )
    with torch.no_grad():
        fused = trunk(features_a["adapter_observation"], condition_a)
    assert fused.shape[-1] == 32, "TrunkNet output should match hidden_dim"

    print("[PASS] Condition vector enabled path is deterministic and threaded into trunk features.")
    return len(features_a["adapter_policy_tensor"]), condition_a


def run_disabled_path():
    adapter = _build_adapter(use_condition_vector=False)
    encoder = StubEncoder()
    builder = PolicyObservationBuilder(encoder, observation_adapter=adapter, use_observation_adapter=True)
    frame = VisionFrame(
        backend="pybullet",
        task_id="task_stub",
        episode_id="ep_002",
        timestep=1,
        backend_id="pybullet_v1",
        state_digest="digest_stub",
    )
    state_summary = {"speed": 0.5}
    adapter_kwargs = {
        "reward_scalar": 0.0,
        "reward_components": {},
        "econ_vector": StubEconVector(),
        "semantic_snapshot": None,
        "recap_scores": None,
        "descriptor": {"pack_id": "pack_stub_b", "sampling_metadata": {"strategy": "balanced"}},
        "episode_metadata": {"episode_id": "ep_002", "sampler_strategy": "balanced"},
        "raw_env_obs": {"t": 1},
    }
    features = builder.build_policy_features(
        frame,
        state_summary,
        use_observation_adapter=True,
        adapter_kwargs=adapter_kwargs,
        use_condition_vector=False,
    )
    assert "condition_vector" not in features, "ConditionVector should be omitted when disabled"
    print("[PASS] Condition vector disabled path falls back cleanly.")
    return len(features["adapter_policy_tensor"])


def main():
    enabled_dim, condition = run_enabled_path()
    disabled_dim = run_disabled_path()
    assert enabled_dim > disabled_dim, "Policy tensor should grow when condition vector is included"
    print(f"[SUMMARY] Enabled tensor dim={enabled_dim}, disabled tensor dim={disabled_dim}")
    print(f"[SUMMARY] ConditionVector skill_mode={condition.skill_mode}, phase={condition.curriculum_phase}")


if __name__ == "__main__":
    main()
