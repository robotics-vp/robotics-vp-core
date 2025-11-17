#!/usr/bin/env python3
"""
Test DataPack Repository and Skill Adapter.

Demonstrates unified 2.0-energy schema with:
- DataPackMeta construction from EpisodeInfoSummary
- DataPackRepo JSONL storage and queries
- SkillDatapackAdapter for skill-centric analysis
- SIMA/VLA annotation support
- Energy driver tag filtering
"""

import os
import sys
import json
import shutil
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.envs.dishwashing_env import EpisodeInfoSummary
from src.config.econ_params import EconParams
from src.valuation.datapacks import build_datapack_meta_from_episode, wrap_legacy_datapack
from src.valuation.datapack_repo import DataPackRepo
from src.valuation.datapack_schema import SimaAnnotation, DATAPACK_SCHEMA_VERSION
from src.hrl.skill_datapack_adapter import SkillDataPackAdapter


def create_fake_episode_summary(
    mpl: float = 10.0,
    error_rate: float = 0.05,
    ep: float = 2.0,
    energy_wh: float = 0.5,
    termination_reason: str = "success"
) -> EpisodeInfoSummary:
    """Create a fake EpisodeInfoSummary for testing."""
    return EpisodeInfoSummary(
        termination_reason=termination_reason,
        mpl_episode=mpl,
        ep_episode=ep,
        error_rate_episode=error_rate,
        throughput_units_per_hour=mpl,
        energy_Wh=energy_wh,
        energy_Wh_per_unit=energy_wh / max(mpl, 1.0),
        energy_Wh_per_hour=energy_wh * 10,
        limb_energy_Wh={
            "shoulder": 0.2,
            "elbow": 0.15,
            "wrist": 0.1,
            "gripper": 0.05,
        },
        skill_energy_Wh={
            "grasp": 0.2,
            "lift": 0.15,
            "place": 0.15,
        },
        energy_per_limb={
            "shoulder": {"Wh": 0.2, "fraction": 0.4},
            "elbow": {"Wh": 0.15, "fraction": 0.3},
            "wrist": {"Wh": 0.1, "fraction": 0.2},
            "gripper": {"Wh": 0.05, "fraction": 0.1},
        },
        energy_per_skill={
            "grasp": {"Wh": 0.2, "fraction": 0.4},
            "lift": {"Wh": 0.15, "fraction": 0.3},
            "place": {"Wh": 0.15, "fraction": 0.3},
        },
        energy_per_joint={
            "joint_0": {"Wh": 0.1, "torque_integral": 5.0},
            "joint_1": {"Wh": 0.2, "torque_integral": 8.0},
            "joint_2": {"Wh": 0.15, "torque_integral": 6.0},
        },
        energy_per_effector={
            "gripper": {"Wh": 0.05, "activation_time": 2.0},
        },
        coordination_metrics={
            "mean_active_joints": 2.5,
            "peak_power": 10.0,
        },
        profit=mpl * 5.0 - error_rate * 50.0,
        wage_parity=0.9,
    )


def test_datapack_construction():
    """Test building DataPackMeta from EpisodeInfoSummary."""
    print("\n" + "=" * 70)
    print("TEST 1: DataPack Construction from EpisodeInfoSummary")
    print("=" * 70)

    econ = EconParams(
        price_per_unit=5.0,
        damage_cost=50.0,
        energy_Wh_per_attempt=0.5,
        time_step_s=60.0,
        base_rate=2.0,
        p_min=0.02,
        k_err=0.12,
        q_speed=1.2,
        q_care=1.5,
        care_cost=0.25,
        max_steps=240,
        max_catastrophic_errors=3,
        max_error_rate_sla=0.10,
        min_steps_for_sla=5,
        zero_throughput_patience=10,
        preset="toy"
    )

    # Create episode summary
    summary = create_fake_episode_summary(mpl=15.0, error_rate=0.03, ep=3.0)

    # Build DataPackMeta
    dp = build_datapack_meta_from_episode(
        summary, econ,
        condition_profile={"task": "drawer_vase", "tags": ["test"], "engine_type": "pybullet"},
        agent_profile={"policy": "test_policy", "version": "v1"},
        brick_id="test_pack_001",
        env_type="drawer_vase",
        baseline_mpl=10.0,
        baseline_error=0.05,
        baseline_ep=2.0,
        skill_trace=[
            {"t": 0, "skill_id": 0, "params": {}, "duration": 10},
            {"t": 10, "skill_id": 1, "params": {}, "duration": 15},
            {"t": 25, "skill_id": 3, "params": {}, "duration": 20},
        ],
        sima_annotation=SimaAnnotation(
            instruction="open the drawer without hitting the vase",
            step_narrations=["Locating drawer handle", "Planning approach", "Grasping handle"],
            sima_agent_id="sima_v1",
            source_world="pyb_drawer_v1"
        ),
        vla_plan={
            "instruction": "open the drawer carefully",
            "skill_sequence": [0, 1, 3],
            "confidence": [0.9, 0.85, 0.88]
        }
    )

    print(f"Schema version: {dp.schema_version}")
    print(f"Pack ID: {dp.pack_id}")
    print(f"Env type: {dp.env_type}")
    print(f"Bucket: {dp.bucket}")
    print(f"Delta J: {dp.attribution.delta_J:.4f}")
    print(f"Delta MPL: {dp.attribution.delta_mpl:.4f}")
    print(f"Delta Error: {dp.attribution.delta_error:.4f}")
    print(f"Delta EP: {dp.attribution.delta_ep:.4f}")
    print(f"Trust score: {dp.attribution.trust_score:.4f}")
    print(f"Energy (total Wh): {dp.energy.total_Wh:.4f}")
    print(f"Energy driver tags: {dp.energy_driver_tags}")
    print(f"Skill trace: {len(dp.skill_trace)} skills")
    print(f"SIMA annotation: {dp.sima_annotation is not None}")
    print(f"VLA plan: {dp.vla_plan is not None}")

    assert dp.schema_version == DATAPACK_SCHEMA_VERSION
    assert dp.env_type == "drawer_vase"
    assert dp.bucket in ["positive", "negative"]
    assert dp.sima_annotation is not None
    assert dp.vla_plan is not None
    print("\nPASS: DataPack construction works correctly")


def test_datapack_repo():
    """Test DataPackRepo storage and queries."""
    print("\n" + "=" * 70)
    print("TEST 2: DataPackRepo Storage and Queries")
    print("=" * 70)

    # Create temp directory
    test_dir = "data/datapacks_test"
    os.makedirs(test_dir, exist_ok=True)

    try:
        repo = DataPackRepo(base_dir=test_dir)

        econ = EconParams(
            price_per_unit=5.0,
            damage_cost=50.0,
            energy_Wh_per_attempt=0.5,
            time_step_s=60.0,
            base_rate=2.0,
            p_min=0.02,
            k_err=0.12,
            q_speed=1.2,
            q_care=1.5,
            care_cost=0.25,
            max_steps=240,
            max_catastrophic_errors=3,
            max_error_rate_sla=0.10,
            min_steps_for_sla=5,
            zero_throughput_patience=10,
            preset="toy"
        )

        # Create diverse datapacks
        datapacks = []
        baseline_mpl = 10.0
        baseline_error = 0.05
        baseline_ep = 2.0

        for i in range(20):
            # Vary performance
            mpl = baseline_mpl + np.random.randn() * 3
            error = max(0, baseline_error + np.random.randn() * 0.02)
            ep = baseline_ep + np.random.randn() * 0.5

            summary = create_fake_episode_summary(mpl=mpl, error_rate=error, ep=ep)

            # Vary skill traces
            if i % 3 == 0:
                skill_trace = [
                    {"t": 0, "skill_id": 0, "params": {}, "duration": 10},
                    {"t": 10, "skill_id": 3, "params": {}, "duration": 20},
                    {"t": 30, "skill_id": 4, "params": {}, "duration": 15},
                ]
            elif i % 3 == 1:
                skill_trace = [
                    {"t": 0, "skill_id": 1, "params": {}, "duration": 12},
                    {"t": 12, "skill_id": 2, "params": {}, "duration": 18},
                    {"t": 30, "skill_id": 5, "params": {}, "duration": 10},
                ]
            else:
                skill_trace = [
                    {"t": 0, "skill_id": 0, "params": {}, "duration": 8},
                    {"t": 8, "skill_id": 1, "params": {}, "duration": 10},
                    {"t": 18, "skill_id": 3, "params": {}, "duration": 25},
                ]

            dp = build_datapack_meta_from_episode(
                summary, econ,
                condition_profile={"task": "drawer_vase", "tags": [], "engine_type": "pybullet"},
                agent_profile={"policy": "test", "version": "v1"},
                brick_id=f"test_{i:04d}",
                env_type="drawer_vase",
                baseline_mpl=baseline_mpl,
                baseline_error=baseline_error,
                baseline_ep=baseline_ep,
                skill_trace=skill_trace
            )
            datapacks.append(dp)

        # Append to repo
        repo.append_batch(datapacks)

        # Query tests
        print(f"\nTotal datapacks: {len(datapacks)}")

        # Get statistics
        stats = repo.get_statistics("drawer_vase")
        print(f"\nRepository statistics:")
        print(f"  Total: {stats['total']}")
        print(f"  Positive: {stats['positive']}")
        print(f"  Negative: {stats['negative']}")
        print(f"  Positive ratio: {stats['positive_ratio']:.2%}")
        print(f"  Delta J mean: {stats['delta_j_mean']:.4f}")
        print(f"  Trust mean: {stats['trust_mean']:.4f}")

        # Query positive only
        positive = repo.query(task_name="drawer_vase", bucket="positive")
        print(f"\nPositive datapacks: {len(positive)}")

        # Query by skill
        skill_3_packs = repo.query(task_name="drawer_vase", skill_id=3, limit=100)
        print(f"Datapacks with GRASP_HANDLE (skill 3): {len(skill_3_packs)}")

        # Query by trust
        high_trust = repo.query(task_name="drawer_vase", min_trust=0.5)
        print(f"High-trust datapacks (>=0.5): {len(high_trust)}")

        assert stats['total'] == 20
        assert stats['positive'] + stats['negative'] == 20
        print("\nPASS: DataPackRepo storage and queries work correctly")

    finally:
        # Cleanup
        shutil.rmtree(test_dir, ignore_errors=True)


def test_skill_datapack_adapter():
    """Test SkillDatapackAdapter queries."""
    print("\n" + "=" * 70)
    print("TEST 3: SkillDatapackAdapter Queries")
    print("=" * 70)

    test_dir = "data/datapacks_test_adapter"
    os.makedirs(test_dir, exist_ok=True)

    try:
        repo = DataPackRepo(base_dir=test_dir)
        adapter = SkillDataPackAdapter(repo)

        econ = EconParams(
            price_per_unit=5.0,
            damage_cost=50.0,
            energy_Wh_per_attempt=0.5,
            time_step_s=60.0,
            base_rate=2.0,
            p_min=0.02,
            k_err=0.12,
            q_speed=1.2,
            q_care=1.5,
            care_cost=0.25,
            max_steps=240,
            max_catastrophic_errors=3,
            max_error_rate_sla=0.10,
            min_steps_for_sla=5,
            zero_throughput_patience=10,
            preset="toy"
        )

        # Create datapacks with specific characteristics
        datapacks = []
        for i in range(30):
            mpl = 10.0 + np.random.randn() * 2
            error = max(0, 0.05 + np.random.randn() * 0.01)
            ep = 2.0 + np.random.randn() * 0.3

            summary = create_fake_episode_summary(mpl=mpl, error_rate=error, ep=ep)

            # Assign specific skills
            skill_id = i % 6
            skill_trace = [
                {"t": 0, "skill_id": skill_id, "params": {}, "duration": 20}
            ]

            dp = build_datapack_meta_from_episode(
                summary, econ,
                condition_profile={"task": "drawer_vase", "tags": [], "engine_type": "pybullet"},
                agent_profile={"policy": "test"},
                brick_id=f"adapter_test_{i:04d}",
                env_type="drawer_vase",
                baseline_mpl=10.0,
                baseline_error=0.05,
                baseline_ep=2.0,
                skill_trace=skill_trace
            )
            datapacks.append(dp)

        repo.append_batch(datapacks)

        # Test skill statistics
        print("\nPer-skill statistics:")
        for skill_id in range(6):
            stats = adapter.get_skill_statistics("drawer_vase", skill_id)
            print(f"  Skill {skill_id} ({stats['skill_name']}): "
                  f"usage={stats['total_usage']}, "
                  f"success_rate={stats['success_rate']:.2%}, "
                  f"mean_ΔJ={stats['delta_j_mean']:.4f}")

        # Test energy query
        energy_tags = ["energy_driver:fragility_cautious"]
        energy_filtered = adapter.query_by_energy_driver_tags("drawer_vase", energy_tags)
        print(f"\nDatapacks with fragility_cautious tag: {len(energy_filtered)}")

        # Test attribution range query
        high_mpl = adapter.query_by_attribution_ranges(
            "drawer_vase",
            min_delta_mpl=1.0,
            bucket="positive"
        )
        print(f"Positive datapacks with ΔMPL >= 1.0: {len(high_mpl)}")

        # Test pack IDs
        pack_ids = adapter.get_datapack_ids_for_skill("drawer_vase", skill_id=3)
        print(f"Pack IDs for skill 3: {len(pack_ids)} datapacks")

        # Test energy summary
        energy_summary = adapter.summarize_energy_usage("drawer_vase")
        print(f"\nEnergy summary:")
        print(f"  Mean total Wh: {energy_summary['total_wh_mean']:.4f}")
        print(f"  Mean Wh/unit: {energy_summary['wh_per_unit_mean']:.4f}")

        print("\nPASS: SkillDatapackAdapter queries work correctly")

    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def test_sima_vla_annotations():
    """Test SIMA and VLA annotation support."""
    print("\n" + "=" * 70)
    print("TEST 4: SIMA and VLA Annotations")
    print("=" * 70)

    econ = EconParams(
        price_per_unit=5.0,
        damage_cost=50.0,
        energy_Wh_per_attempt=0.5,
        time_step_s=60.0,
        base_rate=2.0,
        p_min=0.02,
        k_err=0.12,
        q_speed=1.2,
        q_care=1.5,
        care_cost=0.25,
        max_steps=240,
        max_catastrophic_errors=3,
        max_error_rate_sla=0.10,
        min_steps_for_sla=5,
        zero_throughput_patience=10,
        preset="toy"
    )

    summary = create_fake_episode_summary()

    # Create with full SIMA annotation
    sima = SimaAnnotation(
        instruction="carefully open the drawer while avoiding the fragile vase",
        step_narrations=[
            "I'm approaching the drawer handle",
            "Now I'm gripping the handle firmly",
            "Pulling the drawer open slowly",
            "Monitoring clearance from the vase",
            "Successfully opened the drawer"
        ],
        sima_agent_id="sima_v2",
        source_world="pyb_drawer_v1",
        derived_skill_plan=[0, 1, 2, 3, 4, 5]
    )
    sima.compute_stats()

    vla_plan = {
        "instruction": "open drawer carefully",
        "skill_sequence": [0, 1, 2, 3, 4, 5],
        "skill_params": [[0.15, 0.8, 0.5, 0.5, 100]],
        "timing_horizons": [10, 10, 20, 30, 40, 20],
        "confidence": [0.95, 0.92, 0.88, 0.90, 0.85, 0.93]
    }

    dp = build_datapack_meta_from_episode(
        summary, econ,
        condition_profile={"task": "drawer_vase", "tags": []},
        agent_profile={"policy": "vla_transformer"},
        env_type="drawer_vase",
        sima_annotation=sima,
        vla_plan=vla_plan
    )

    print(f"SIMA Annotation:")
    print(f"  Instruction: {dp.sima_annotation.instruction}")
    print(f"  Narration count: {dp.sima_annotation.narration_count}")
    print(f"  Avg narration length: {dp.sima_annotation.average_narration_length:.1f}")
    print(f"  Derived skill plan: {dp.sima_annotation.derived_skill_plan}")

    print(f"\nVLA Plan:")
    print(f"  Instruction: {dp.vla_plan['instruction']}")
    print(f"  Skill sequence: {dp.vla_plan['skill_sequence']}")
    print(f"  Confidence: {dp.vla_plan['confidence']}")

    # Serialize and deserialize
    dp_dict = dp.to_dict()
    dp_restored = dp.from_dict(dp_dict)

    assert dp_restored.sima_annotation is not None
    assert dp_restored.sima_annotation.instruction == sima.instruction
    assert dp_restored.vla_plan is not None
    assert dp_restored.vla_plan['skill_sequence'] == vla_plan['skill_sequence']

    print("\nPASS: SIMA and VLA annotations work correctly")


def test_legacy_conversion():
    """Test conversion from legacy 2.0-energy dict format."""
    print("\n" + "=" * 70)
    print("TEST 5: Legacy 2.0-Energy Dict Conversion")
    print("=" * 70)

    # Create a legacy-style dict (as produced by build_datapack_from_episode)
    legacy_dict = {
        "schema_version": "2.0-energy",
        "env_type": "dishwashing",
        "brick_id": "legacy_001",
        "episode_metrics": {
            "termination_reason": "success",
            "mpl_episode": 12.0,
            "ep_episode": 2.5,
            "error_rate_episode": 0.04,
            "energy_Wh": 0.6,
        },
        "econ_params": {
            "price_per_unit": 5.0,
            "damage_cost": 50.0,
            "energy_Wh_per_attempt": 0.5,
            "time_step_s": 0.01,
            "max_steps": 1000,
            "preset": "dishwashing"
        },
        "condition_profile": {"task": "dishwashing", "tags": ["phase_b"]},
        "agent_profile": {"policy": "ppo_baseline"},
        "tags": ["phase_b", "test"],
        "attribution": {
            "delta_mpl": 12.0,
            "delta_error": 0.04,
            "delta_ep": 2.5,
            "novelty": None,
            "trust": 0.85,
            "econ_weight": 1.2
        },
        "energy": {
            "total_Wh": 0.6,
            "Wh_per_unit": 0.05,
            "Wh_per_hour": 6.0,
            "limb_energy_Wh": {"arm": 0.4, "gripper": 0.2},
            "skill_energy_Wh": {"wash": 0.6},
            "energy_per_limb": {"arm": {"Wh": 0.4}},
            "energy_per_skill": {"wash": {"Wh": 0.6}},
            "energy_per_joint": {},
            "energy_per_effector": {},
            "coordination_metrics": {}
        },
        "semantic_energy_drivers": ["energy_driver:throughput_push"]
    }

    # Convert to DataPackMeta
    dp = wrap_legacy_datapack(legacy_dict)

    print(f"Converted from legacy format:")
    print(f"  Schema version: {dp.schema_version}")
    print(f"  Env type: {dp.env_type}")
    print(f"  Bucket: {dp.bucket}")
    print(f"  Trust score: {dp.attribution.trust_score}")
    print(f"  W_econ: {dp.attribution.w_econ}")
    print(f"  Energy tags: {dp.energy_driver_tags}")
    print(f"  Total Wh: {dp.energy.total_Wh}")

    # Convert back to legacy
    back_to_legacy = dp.to_legacy_energy_dict()
    print(f"\nConverted back to legacy:")
    print(f"  Schema version: {back_to_legacy['schema_version']}")
    print(f"  Env type: {back_to_legacy['env_type']}")
    print(f"  Trust: {back_to_legacy['attribution']['trust']}")

    assert dp.schema_version == "2.0-energy"
    assert dp.env_type == "dishwashing"
    assert dp.attribution.trust_score == 0.85
    print("\nPASS: Legacy conversion works correctly")


def main():
    print("=" * 70)
    print("DATAPACK REPOSITORY AND SKILL ADAPTER TESTS")
    print("=" * 70)
    print(f"Schema version: {DATAPACK_SCHEMA_VERSION}")

    test_datapack_construction()
    test_datapack_repo()
    test_skill_datapack_adapter()
    test_sima_vla_annotations()
    test_legacy_conversion()

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)
    print("\nThe unified datapack system supports:")
    print("  - Phase B dishwashing datapacks")
    print("  - Phase C drawer_vase datapacks")
    print("  - 2.0-energy schema with energy_per_limb/skill/joint/effector")
    print("  - SIMA natural language annotations")
    print("  - VLA skill plan metadata")
    print("  - Two-bucket taxonomy (positive/negative)")
    print("  - Energy driver tag filtering")
    print("  - Attribution range queries (ΔMPL, Δerror, ΔEP, ΔJ)")
    print("  - Per-skill statistics and training data extraction")


if __name__ == "__main__":
    main()
