"""
Smoke test: TFD → ConditionVector → ConditionedVisionAdapter → RL policy features.

Asserts:
- TFD instructions modulate ConditionVector risk tolerance
- ConditionedVisionAdapter changes risk_map/feature scales based on ConditionVector
- Deterministic outputs across runs
- Flag gating preserves legacy vision path when disabled
"""
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.observation.adapter import ObservationAdapter  # noqa: E402
from src.tfd.compiler import TextFrontDoor  # noqa: E402
from src.vision.policy_observation_builder import PolicyObservationBuilder  # noqa: E402
from src.vision.interfaces import VisionFrame, VisionLatent  # noqa: E402


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


class StubEconVector:
    def __init__(self) -> None:
        self.mpl_units_per_hour = 12.0
        self.wage_parity = 0.9
        self.energy_cost = 1.5
        self.damage_cost = 0.05
        self.source_domain = "stub"
        self.metadata = {"econ_slice": True}


def build_builder() -> PolicyObservationBuilder:
    trust_loader = lambda: {}
    adapter_cfg = {"use_condition_vector": True}
    adapter = ObservationAdapter(policy_registry=None, trust_matrix_loader=trust_loader, recap_loader=None, config=adapter_cfg)
    encoder = StubEncoder()
    return PolicyObservationBuilder(encoder, observation_adapter=adapter, use_observation_adapter=True)


def make_frame() -> VisionFrame:
    return VisionFrame(
        backend="pybullet",
        task_id="vision_task",
        episode_id="ep_tfd",
        timestep=3,
        backend_id="pybullet_v1",
        camera_intrinsics={"fx": 1.0},
        camera_extrinsics={"tx": 0.0},
        state_digest="digest_stub",
    )


def build_features(builder: PolicyObservationBuilder, instruction_text: str, *, enable_conditioned: bool) -> dict:
    tfd = TextFrontDoor()
    instruction = tfd.process_instruction(instruction_text)
    frame = make_frame()
    state_summary = {"speed": 0.6, "errors": 1}
    adapter_kwargs = {
        "reward_scalar": 1.0,
        "reward_components": {"mpl_component": 0.5},
        "econ_vector": StubEconVector(),
        "semantic_snapshot": None,
        "recap_scores": None,
        "descriptor": {
            "task_id": frame.task_id,
            "env_id": "sim_env",
            "backend_id": frame.backend,
            "pack_id": "pack_stub",
            "semantic_tags": {"fragile": 1.0},
            "sampling_metadata": {"strategy": "frontier_prioritized", "frontier_score": 2.0},
        },
        "episode_metadata": {"episode_id": frame.episode_id, "sampler_strategy": "frontier_prioritized", "skill_id": "skill_stub"},
        "raw_env_obs": {"t": frame.timestep, "completed": 0, "attempts": 1, "errors": 0},
    }
    condition_kwargs = {"tfd_instruction": instruction, "enable_tfd_integration": True}
    vision_overrides = {"enable_conditioned_vision": enable_conditioned, "conditioned_vision_enable_conditioning": True}
    return builder.build_policy_features(
        frame,
        state_summary,
        use_observation_adapter=True,
        adapter_kwargs=adapter_kwargs,
        condition_kwargs=condition_kwargs,
        use_condition_vector=True,
        vision_overrides=vision_overrides,
    )


def assert_risk_modulation(builder: PolicyObservationBuilder):
    cautious = build_features(builder, "be cautious near glass", enable_conditioned=True)
    speed = build_features(builder, "go fast", enable_conditioned=True)
    cv_cautious = cautious["condition_vector"]
    cv_speed = speed["condition_vector"]
    assert cv_cautious.ood_risk_level < cv_speed.ood_risk_level, "TFD should lower risk tolerance for cautious instruction"

    risk_cautious = np.asarray(cautious["conditioned_vision"]["risk_map"])
    risk_speed = np.asarray(speed["conditioned_vision"]["risk_map"])
    assert not np.allclose(risk_cautious, risk_speed), "Conditioned vision risk map should change with ConditionVector"

    fused_cautious = np.asarray(cautious["conditioned_vision_vector"])
    fused_speed = np.asarray(speed["conditioned_vision_vector"])
    assert not np.allclose(fused_cautious, fused_speed), "Conditioned vision features should reflect ConditionVector changes"


def assert_determinism(builder: PolicyObservationBuilder):
    first = build_features(builder, "be cautious near glass", enable_conditioned=True)
    second = build_features(builder, "be cautious near glass", enable_conditioned=True)
    risk_first = np.asarray(first["conditioned_vision"]["risk_map"])
    risk_second = np.asarray(second["conditioned_vision"]["risk_map"])
    fused_first = np.asarray(first["conditioned_vision_vector"])
    fused_second = np.asarray(second["conditioned_vision_vector"])
    assert np.allclose(risk_first, risk_second), "Risk maps should be deterministic across runs"
    assert np.allclose(fused_first, fused_second), "Conditioned vision vectors should be deterministic"


def assert_flag_gating(builder: PolicyObservationBuilder):
    enabled = build_features(builder, "be cautious near glass", enable_conditioned=True)
    disabled = build_features(builder, "be cautious near glass", enable_conditioned=False)
    assert "conditioned_vision" in enabled, "Conditioned vision should be present when flag enabled"
    assert "conditioned_vision" not in disabled, "Conditioned vision should be skipped when flag disabled"
    assert enabled["vision_latent"] == disabled["vision_latent"], "Base vision latent must be unchanged by conditioning flag"


def main() -> int:
    builder = build_builder()
    assert_risk_modulation(builder)
    assert_determinism(builder)
    assert_flag_gating(builder)
    print("✅ TFD → Conditioned Vision chain smoke test passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
