#!/usr/bin/env python3
"""
Smoke test for datapack -> RL episode descriptor ingestion.
Adds no training/reward changes; purely validation.
"""
import json
import math
import os
import sys
from dataclasses import asdict, is_dataclass
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.valuation.datapack_schema import DataPackMeta, ConditionProfile, GuidanceProfile, ObjectiveProfile  # type: ignore  # noqa: E501
from src.rl.episode_sampling import datapack_to_rl_episode_descriptor, sampler_stub


def load_stage1_datapacks() -> List[DataPackMeta]:
    path = "results/stage1_pipeline/datapacks.json"
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        raw = json.load(f)
    dps = []
    if isinstance(raw, list):
        for entry in raw:
            try:
                # Use from_dict method to properly reconstruct nested objects
                dp = DataPackMeta.from_dict(entry)
                dps.append(dp)
            except Exception as e:
                print(f"Warning: Failed to load datapack: {e}")
                continue
    return dps


def synthesize_datapacks() -> List[DataPackMeta]:
    guidance = GuidanceProfile(
        is_good=True,
        quality_label="good",
        env_name="drawer_vase",
        engine_type="pybullet",
        task_type="fragility",
        customer_segment="balanced",
        objective_vector=[1.0, 1.0, 1.0, 1.0, 0.0],
        main_driver="throughput_gain",
        delta_mpl=0.0,
        delta_error=0.0,
        delta_energy_Wh=0.0,
        delta_J=0.0,
        semantic_tags=["dummy"],
    )
    op = ObjectiveProfile(
        objective_vector=[1.0, 1.0, 1.0, 1.0, 0.0],
        env_name="drawer_vase",
        engine_type="pybullet",
        task_type="fragility",
    )  # type: ignore  # noqa: E501
    cond = ConditionProfile(engine_type="pybullet", lighting_profile="default")  # type: ignore
    dps = []
    for i in range(3):
        dp = DataPackMeta(
            task_name="drawer_vase",
            env_type="drawer_vase",
            bucket="positive",
            condition=cond,
            guidance_profile=guidance,
            objective_profile=op,
        )
        dps.append(dp)
    return dps


def has_nan(obj) -> bool:
    try:
        return math.isnan(obj)  # type: ignore
    except Exception:
        return False


def convert_for_json(obj):
    if is_dataclass(obj):
        return {k: convert_for_json(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: convert_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_for_json(v) for v in obj]
    return obj


def main():
    dps = load_stage1_datapacks()
    if not dps:
        dps = synthesize_datapacks()
        print("[smoke] No stage1 datapacks found; using synthetic samples.")
    else:
        print(f"[smoke] Loaded {len(dps)} stage1 datapacks.")

    descriptors = [datapack_to_rl_episode_descriptor(dp) for dp in dps]

    for desc in descriptors:
        # Check new descriptor format fields
        assert "objective_vector" in desc, "Missing objective_vector"
        assert "env_name" in desc, "Missing env_name"
        assert "backend" in desc, "Missing backend"
        assert "tier" in desc, "Missing tier"
        assert "trust_score" in desc, "Missing trust_score"
        assert "sampling_weight" in desc, "Missing sampling_weight"

        flat_desc = convert_for_json(desc)
        flat_str = json.dumps(flat_desc, default=str)
        assert flat_str
        for v in flat_desc.get("objective_vector", []):
            assert not has_nan(v)

    sampled = sampler_stub(dps)
    assert sampled
    print("[smoke] descriptors:", list(sampled.values())[0])
    print("[smoke] Datapack RL ingestion smoke passed.")


if __name__ == "__main__":
    main()
