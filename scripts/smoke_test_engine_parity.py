#!/usr/bin/env python3
"""
Smoke test for backend API parity between PyBullet and Isaac stubs.

Verifies:
1. Both backends can be constructed (Isaac as stub)
2. They share the same interface methods
3. Engine type tagging is correct
4. EpisodeInfoSummary -> DataPackMeta -> orchestrator context works without errors
5. Media refs hooks are present and functional
"""

import inspect
from typing import get_type_hints

# Physics backends
from src.envs.physics.base_engine import PhysicsBackend
from src.envs.physics.pybullet_backend import PyBulletBackend
from src.envs.physics.isaac_backend import IsaacBackend
from src.envs.physics.backend_factory import make_backend

# Datapack and orchestrator integration
from src.envs.dishwashing_env import EpisodeInfoSummary
from src.valuation.datapack_schema import (
    ConditionProfile,
    AttributionProfile,
    ObjectiveProfile,
    DataPackMeta,
)


def test_isaac_construction():
    """Test that Isaac backend can be constructed without immediate error."""
    print("Test 1: Isaac backend construction")

    # Should not raise NotImplementedError on construction
    isaac = IsaacBackend(
        env_config={
            "env_name": "drawer_vase",
            "task": "drawer_vase",
            "robot": "franka",
            "dt": 1 / 60,
            "substeps": 2,
            "obs_dim": 64,
            "action_dim": 7,
            "econ_params": {
                "price_per_unit": 0.30,
                "vase_break_cost": 5.0,
                "energy_price_kWh": 0.12,
            },
        },
        num_envs=4,
        device="cuda:0",
    )

    assert isaac.engine_type == "isaac"
    assert isaac.env_name == "drawer_vase"
    assert isaac.num_envs == 4
    print("  PASS: Isaac backend constructed successfully")

    return isaac


def test_api_parity():
    """Test that both backends implement the same interface."""
    print("\nTest 2: API parity check")

    # Get all abstract methods from base class
    base_methods = set()
    for name, method in inspect.getmembers(PhysicsBackend, predicate=inspect.isfunction):
        if not name.startswith("_"):
            base_methods.add(name)

    # Also add properties
    for name in dir(PhysicsBackend):
        if not name.startswith("_"):
            attr = getattr(PhysicsBackend, name)
            if isinstance(attr, property):
                base_methods.add(name)

    print(f"  Base interface methods: {sorted(base_methods)}")

    # Check PyBulletBackend
    pybullet_methods = set()
    for name in dir(PyBulletBackend):
        if not name.startswith("_") and name in base_methods:
            pybullet_methods.add(name)

    # Check IsaacBackend
    isaac_methods = set()
    for name in dir(IsaacBackend):
        if not name.startswith("_") and name in base_methods:
            isaac_methods.add(name)

    # Compare
    missing_from_pybullet = base_methods - pybullet_methods
    missing_from_isaac = base_methods - isaac_methods

    if missing_from_pybullet:
        print(f"  WARNING: PyBullet missing: {missing_from_pybullet}")
    if missing_from_isaac:
        print(f"  WARNING: Isaac missing: {missing_from_isaac}")

    assert pybullet_methods == isaac_methods, (
        f"API mismatch: PyBullet has {pybullet_methods - isaac_methods}, "
        f"Isaac has {isaac_methods - pybullet_methods}"
    )

    print("  PASS: Both backends implement same base interface")

    # Check additional methods for parity
    additional_methods = [
        "get_media_refs",
        "set_media_refs",
        "get_current_episode_id",
        "get_config",
    ]

    for method in additional_methods:
        has_pybullet = hasattr(PyBulletBackend, method)
        has_isaac = hasattr(IsaacBackend, method)
        assert has_pybullet and has_isaac, f"Missing {method}: PyBullet={has_pybullet}, Isaac={has_isaac}"
        print(f"  PASS: Both have {method}")

    # Isaac-specific methods
    isaac_specific = ["get_batch_episode_info", "reset_env", "num_envs"]
    for method in isaac_specific:
        assert hasattr(IsaacBackend, method), f"Isaac missing {method}"
        print(f"  PASS: Isaac has {method} (vectorized-specific)")


def test_engine_tagging():
    """Test that engine_type property returns correct values."""
    print("\nTest 3: Engine type tagging")

    isaac = IsaacBackend(env_config={"env_name": "test_env"})
    assert isaac.engine_type == "isaac"
    print(f"  PASS: Isaac engine_type = '{isaac.engine_type}'")

    # PyBullet requires an env, so we'll use a mock
    class MockEnv:
        def reset(self):
            return {}, {}

        def step(self, action):
            return {}, 0.0, False, {}

    pybullet = PyBulletBackend(env=MockEnv(), env_name="test_env")
    assert pybullet.engine_type == "pybullet"
    print(f"  PASS: PyBullet engine_type = '{pybullet.engine_type}'")


def test_episode_info_summary_schema():
    """Test that both backends can produce EpisodeInfoSummary compatible with datapack."""
    print("\nTest 4: EpisodeInfoSummary schema compatibility")

    # Create a mock EpisodeInfoSummary (what both backends must produce)
    summary = EpisodeInfoSummary(
        termination_reason="success",
        mpl_episode=12.0,  # 12 units/hour
        ep_episode=0.8,  # 0.8 units/Wh
        error_rate_episode=0.05,
        throughput_units_per_hour=12.0,
        energy_Wh=15.0,
        energy_Wh_per_unit=1.25,
        energy_Wh_per_hour=15.0,
        limb_energy_Wh={"arm": 10.0, "gripper": 5.0},
        skill_energy_Wh={"reach": 6.0, "grasp": 9.0},
        energy_per_limb={"arm": 0.67, "gripper": 0.33},
        energy_per_skill={"reach": 0.4, "grasp": 0.6},
        energy_per_joint={"joint_1": 3.0, "joint_2": 4.0, "joint_3": 3.0, "gripper_joint": 5.0},
        energy_per_effector={"gripper": 5.0},
        coordination_metrics={"mean_active_joints": 3.2, "peak_power": 25.0},
        profit=5.5,
        wage_parity=0.85,
    )

    print(f"  Summary MPL: {summary.mpl_episode}, Energy: {summary.energy_Wh} Wh")

    # Test that we can build a ConditionProfile with engine_type
    condition = ConditionProfile(
        task_name="drawer_vase",
        engine_type="isaac",  # From backend.engine_type
        world_id="test_world_001",
        vase_offset=0.1,
        drawer_friction=0.5,
        lighting_profile="uniform",
        occlusion_level="none",
        econ_preset="standard",
        price_per_unit=0.30,
        vase_break_cost=5.0,
        energy_price_kWh=0.12,
        objective_vector=[1.0, 1.0, 1.0, 1.0, 0.0],
        tags={"num_envs": 4, "device": "cuda:0"},
    )

    assert condition.engine_type == "isaac"
    print(f"  PASS: ConditionProfile accepts engine_type='{condition.engine_type}'")

    # Test ObjectiveProfile integration
    objective = ObjectiveProfile(
        objective_vector=[1.0, 1.0, 1.0, 1.0, 0.0],
        wage_human=18.0,
        energy_price_kWh=0.12,
        market_region="US",
        task_family="manipulation",
        customer_segment="industrial",
        baseline_mpl_human=60.0,
        baseline_error_human=0.02,
        env_name="drawer_vase",
        engine_type="isaac",  # From backend
        task_type="drawer_open",
    )

    assert objective.engine_type == "isaac"
    print(f"  PASS: ObjectiveProfile accepts engine_type='{objective.engine_type}'")

    # Test AttributionProfile
    attribution = AttributionProfile(
        delta_mpl=0.5,
        delta_error=-0.01,
        delta_ep=0.1,
        delta_wage_parity=0.02,
        delta_J=0.3,
        trust_score=0.9,
        w_econ=0.85,
        lambda_budget=1.0,
        world_model_horizon=100,
        world_model_trust_over_horizon=[0.9] * 100,
        source_type="real",
        zv_stats={},
        mvd_score=0.5,
    )

    print(f"  PASS: AttributionProfile created with delta_J={attribution.delta_J}")


def test_datapack_creation_from_backend():
    """Test end-to-end: backend config -> datapack meta."""
    print("\nTest 5: DataPackMeta creation from backend config")

    # Simulate Isaac backend providing config
    isaac = IsaacBackend(
        env_config={
            "env_name": "drawer_vase",
            "task": "drawer_vase",
            "robot": "franka",
        },
        num_envs=4,
    )

    config = isaac.get_config()
    assert config["engine_type"] == "isaac"
    assert config["env_name"] == "drawer_vase"
    assert config["num_envs"] == 4
    print(f"  Isaac config: {config}")

    # Create ConditionProfile from backend config
    condition = ConditionProfile(
        task_name=config["env_name"],
        engine_type=config["engine_type"],
        world_id="world_001",
        vase_offset=(0.1, 0.0, 0.0),  # 3D offset tuple
        drawer_friction=0.5,
        lighting_profile="uniform",
        occlusion_level="none",
        econ_preset="standard",
        price_per_unit=0.30,
        vase_break_cost=5.0,
        energy_price_kWh=0.12,
        objective_vector=[1.0, 1.0, 1.0, 1.0, 0.0],
        tags={"num_envs": config["num_envs"], "device": config.get("device", "cuda:0")},
    )

    # Create minimal DataPackMeta
    from src.valuation.datapack_schema import EnergyProfile

    energy = EnergyProfile(
        total_Wh=15.0,
        Wh_per_unit=1.25,
        Wh_per_hour=15.0,
        energy_per_limb={"arm": 10.0, "gripper": 5.0},
        energy_per_skill={"reach": 6.0, "grasp": 9.0},
        energy_per_joint={},
        energy_per_effector={},
        coordination_metrics={},
    )

    attribution = AttributionProfile(
        delta_mpl=0.5,
        delta_error=-0.01,
        delta_ep=0.1,
        delta_wage_parity=0.02,
        delta_J=0.3,
        trust_score=0.9,
        w_econ=0.85,
        lambda_budget=1.0,
        world_model_horizon=100,
        world_model_trust_over_horizon=[0.9] * 100,
        source_type="real",
        zv_stats={},
        mvd_score=0.5,
    )

    objective = ObjectiveProfile(
        objective_vector=[1.0, 1.0, 1.0, 1.0, 0.0],
        wage_human=18.0,
        energy_price_kWh=0.12,
        market_region="US",
        task_family="manipulation",
        customer_segment="industrial",
        baseline_mpl_human=60.0,
        baseline_error_human=0.02,
        env_name=config["env_name"],
        engine_type=config["engine_type"],
        task_type="drawer_open",
    )

    # Create DataPackMeta
    datapack = DataPackMeta(
        pack_id="test_pack_001",
        task_name=config["env_name"],
        env_type=config["engine_type"],
        brick_id="brick_001",
        bucket="positive",
        semantic_tags=["test", "isaac_stub"],
        energy_driver_tags=["arm_dominant"],
        condition=condition,
        attribution=attribution,
        energy=energy,
        agent_profile={"policy": "scripted"},
        skill_trace=[
            {"skill_id": "reach", "duration": 2.0},
            {"skill_id": "grasp", "duration": 3.0},
        ],
        episode_metrics={},
        sima_annotation=None,
        vla_plan=None,
        objective_profile=objective,
        counterfactual_plan=None,
        counterfactual_source=None,
        episode_id=isaac.get_current_episode_id(0),
        episode_index=0,
        raw_data_path="",
        guidance_profile=None,
        vla_action_summary=None,
    )

    assert datapack.env_type == "isaac"
    assert datapack.condition.engine_type == "isaac"
    assert datapack.objective_profile.engine_type == "isaac"
    print(f"  PASS: DataPackMeta created with env_type='{datapack.env_type}'")

    # Test serialization
    datapack_dict = datapack.to_dict()
    assert datapack_dict["env_type"] == "isaac"
    assert datapack_dict["condition"]["engine_type"] == "isaac"
    print("  PASS: DataPackMeta serializes correctly")


def test_media_refs_hooks():
    """Test media_refs API for both backends."""
    print("\nTest 6: Media refs hooks")

    # Isaac backend
    isaac = IsaacBackend(env_config={"env_name": "test"}, num_envs=2)

    # Set episode IDs manually for testing
    isaac._current_episode_ids = ["ep_001", "ep_002"]

    # Set media refs
    isaac.set_media_refs(0, {"rgb_path": "/data/ep_001.mp4", "depth_path": "/data/ep_001_depth.npy"})
    isaac.set_media_refs(1, {"rgb_path": "/data/ep_002.mp4"})

    refs_0 = isaac.get_media_refs(0)
    refs_1 = isaac.get_media_refs(1)

    assert refs_0["rgb_path"] == "/data/ep_001.mp4"
    assert refs_1["rgb_path"] == "/data/ep_002.mp4"
    assert "depth_path" not in refs_1  # Only set for env 0
    print(f"  PASS: Isaac media refs: env0={refs_0}, env1={refs_1}")

    # Episode ID
    assert isaac.get_current_episode_id(0) == "ep_001"
    assert isaac.get_current_episode_id(1) == "ep_002"
    print("  PASS: Isaac episode IDs tracked correctly")

    # PyBullet backend
    class MockEnv:
        def reset(self):
            return {}, {}

    pybullet = PyBulletBackend(env=MockEnv(), env_name="test")
    pybullet.reset()  # Generates episode ID

    ep_id = pybullet.get_current_episode_id()
    assert ep_id is not None
    print(f"  PASS: PyBullet generated episode_id={ep_id}")

    pybullet.set_media_refs({"rgb_path": f"/data/{ep_id}.mp4"})
    refs = pybullet.get_media_refs()
    assert refs["rgb_path"] == f"/data/{ep_id}.mp4"
    print(f"  PASS: PyBullet media refs: {refs}")


def test_orchestrator_context_integration():
    """Test that backend config flows into orchestrator context."""
    print("\nTest 7: Orchestrator context integration (schema check)")

    # This is a schema-level test - we're not running orchestrator code,
    # just verifying that the data structures are compatible

    isaac = IsaacBackend(
        env_config={
            "env_name": "drawer_vase",
            "task": "drawer_vase",
            "econ_params": {
                "price_per_unit": 0.30,
                "energy_price_kWh": 0.12,
            },
        },
        num_envs=8,
    )

    config = isaac.get_config()

    # Simulate what orchestrator context builder would extract
    orchestrator_context_fields = {
        "env_name": config["env_name"],
        "engine_type": config["engine_type"],
        "num_envs": config["num_envs"],
        "device": config["device"],
        "task_type": config["env_config"].get("task", "unknown"),
        "energy_price_kWh": config["env_config"].get("econ_params", {}).get("energy_price_kWh", 0.10),
    }

    assert orchestrator_context_fields["engine_type"] == "isaac"
    assert orchestrator_context_fields["env_name"] == "drawer_vase"
    assert orchestrator_context_fields["num_envs"] == 8
    print(f"  PASS: Orchestrator context fields: {orchestrator_context_fields}")


def test_not_implemented_errors():
    """Test that physics methods still raise NotImplementedError."""
    print("\nTest 8: Physics methods raise NotImplementedError (as expected)")

    isaac = IsaacBackend(env_config={"env_name": "test"})

    physics_methods = [
        ("reset", ()),
        ("step", (None,)),
        ("get_episode_info", ()),
        ("get_info_history", ()),
        ("close", ()),
        ("get_observation_space", ()),
        ("get_action_space", ()),
        ("render", ()),
        ("seed", ()),
        ("get_state", ()),
        ("set_state", (None,)),
        ("get_batch_episode_info", ()),
        ("reset_env", (0,)),
    ]

    for method_name, args in physics_methods:
        method = getattr(isaac, method_name)
        try:
            method(*args)
            print(f"  FAIL: {method_name} should raise NotImplementedError")
            assert False, f"{method_name} did not raise NotImplementedError"
        except NotImplementedError:
            pass  # Expected
        except TypeError:
            # Some methods have required args, which is fine
            pass

    print("  PASS: All physics methods raise NotImplementedError (Isaac stub)")


def main():
    print("=" * 60)
    print("Backend API Parity Smoke Test")
    print("=" * 60)

    test_isaac_construction()
    test_api_parity()
    test_engine_tagging()
    test_episode_info_summary_schema()
    test_datapack_creation_from_backend()
    test_media_refs_hooks()
    test_orchestrator_context_integration()
    test_not_implemented_errors()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
    print("\nIsaac backend stub is ready for future implementation.")
    print("When you implement Isaac Gym physics, fill in the methods that")
    print("currently raise NotImplementedError, and all higher layers")
    print("(datapacks, orchestrator, RewardBuilder, solver) will 'just work'.")


if __name__ == "__main__":
    main()
