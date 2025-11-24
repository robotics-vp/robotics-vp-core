#!/usr/bin/env python3
"""
Smoke test for IsaacAdapter contract.

Assertions:
- VisionFrame populates RGB/Depth/Segmentation paths
- ProprioFrame matches schema and carries energy estimate/contact sensors
- Deterministic outputs with fixed seed
- Compatible with ConditionedVisionAdapter and PolicyObservationBuilder
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.env.isaac_adapter import IsaacAdapter
from src.observation.condition_vector import ConditionVector
from src.vision.interfaces import VisionLatent
from src.vision.policy_observation_builder import PolicyObservationBuilder


class StubVisionEncoder:
    mode = "stub"
    backbone_name = "stub"

    def encode_frame(self, frame):
        return VisionLatent(
            backend=frame.backend,
            task_id=frame.task_id,
            episode_id=frame.episode_id,
            timestep=frame.timestep,
            latent=[float(frame.width), float(frame.height), float(frame.channels)],
            model_name="stub_encoder",
        )


def make_condition() -> ConditionVector:
    return ConditionVector(
        task_id="drawer",
        env_id="isaac_env",
        backend_id="isaac",
        target_mpl=60.0,
        current_wage_parity=1.0,
        energy_budget_wh=50.0,
        skill_mode="precision",
        ood_risk_level=0.4,
        recovery_priority=0.2,
        novelty_tier=1,
        sima2_trust_score=0.7,
        recap_goodness_bucket="silver",
        objective_preset="balanced",
    )


def synthetic_observation() -> dict:
    return {
        "rgb": np.zeros((4, 4, 3), dtype=np.uint8),
        "depth": np.ones((4, 4), dtype=np.float32),
        "segmentation": np.zeros((4, 4), dtype=np.uint8),
        "joint_positions": [0.1, -0.2],
        "joint_velocities": [0.05, -0.03],
        "joint_torques": [1.2, 0.8],
        "contact_forces": [0.0, 1.0],
        "end_effector_pose": {"position": [0.1, 0.0, 0.2], "orientation": [0, 0, 0, 1]},
        "dt": 0.1,
        "camera_intrinsics": {"resolution": [4, 4], "fov_deg": 90.0},
        "camera_extrinsics": {"frame": "world", "translation": [0.0, 0.0, 1.0]},
        "tf": {"base_link": {"translation": [0, 0, 0], "rotation": [0, 0, 0, 1]}},
        "action": {"ee_delta": [0.0, 0.0, -0.01]},
    }


def main() -> int:
    with tempfile.TemporaryDirectory() as tmpdir:
        adapter = IsaacAdapter(config={"seed": 42, "backend": "isaac"}, output_root=str(Path(tmpdir) / "isaac_out"))
        obs = synthetic_observation()
        condition = make_condition()

        result_a = adapter.adapt(obs, episode_id="isaac_ep", task_id="drawer_task", timestep=0, condition_vector=condition)
        result_b = adapter.adapt(obs, episode_id="isaac_ep", task_id="drawer_task", timestep=0, condition_vector=condition)

        vf = result_a["vision_frame"]
        assert vf.rgb_path and vf.depth_path and vf.segmentation_path, "VisionFrame should persist all modalities"
        assert vf.width > 0 and vf.height > 0, "VisionFrame dimensions should be set"
        assert vf.backend_id == "isaac", "VisionFrame should mark backend_id=isaac"
        assert vf.metadata.get("domain_randomization"), "Domain randomization metadata should be present"

        pf = result_a["proprio_frame"]
        assert len(pf.joint_positions) == 2, "ProprioFrame should carry joint positions"
        assert pf.energy_estimate_Wh > 0, "Energy estimate should be positive from torques"
        assert pf.contact_sensors, "Contact sensors should be propagated"
        proxies = result_a.get("energy_proxies", {})
        assert proxies.get("energy_proxy_Wh", 0.0) >= pf.energy_estimate_Wh, "Energy proxies should include energy_Wh"

        # Determinism
        assert vf.to_dict() == result_b["vision_frame"].to_dict(), "VisionFrame must be deterministic"
        assert pf.to_dict() == result_b["proprio_frame"].to_dict(), "ProprioFrame must be deterministic"
        assert np.allclose(result_a["vision_features"]["risk_map"], result_b["vision_features"]["risk_map"]), "Vision features should be deterministic"

        # Conditioned vision compatibility through policy observation builder
        builder = PolicyObservationBuilder(
            encoder=StubVisionEncoder(),
            use_observation_adapter=False,
            vision_config={"enable_conditioned_vision": True, "conditioned_vision_enable_conditioning": True},
        )
        state_summary = {
            "energy_proxy_Wh": proxies.get("energy_proxy_Wh", 0.0),
            "torque_abs_sum": proxies.get("torque_abs_sum", 0.0),
        }
        features_a = builder.build_policy_features(
            vf,
            state_summary,
            condition_vector=condition,
            vision_overrides={"enable_conditioned_vision": True, "conditioned_vision_enable_conditioning": True},
        )
        features_b = builder.build_policy_features(
            vf,
            state_summary,
            condition_vector=condition,
            vision_overrides={"enable_conditioned_vision": True, "conditioned_vision_enable_conditioning": True},
        )

        assert features_a["backend_tags"]["backend_id"] == "isaac", "Backend tags should carry isaac backend_id"
        assert features_a["conditioned_vision_vector"] is not None, "Conditioned vision vector should be present"
        assert np.allclose(features_a["conditioned_vision_vector"], features_b["conditioned_vision_vector"]), "Conditioned vision must be deterministic"
        assert features_a["condition_vector_dict"]["task_id"] == "drawer", "ConditionVector should be threaded through builder"
        assert features_a["conditioned_vision_risk_map"] is not None, "Risk map should be present"
        assert result_a["domain_randomization"], "Domain randomization metadata should be non-empty"

        features = result_a["vision_features"]
        assert "z_v" in features and "risk_map" in features and "affordance_map" in features, "ConditionedVisionAdapter outputs missing keys"

    print("[smoke_test_isaac_adapter_contract] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
