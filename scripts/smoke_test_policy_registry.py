#!/usr/bin/env python3
"""
Smoke test for Phase G policy registry.

Instantiates all policies in heuristic mode and validates expected methods are present.
"""
import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from src.policies.registry import build_all_policies  # noqa: E402


REQUIRED_METHODS = {
    "data_valuation": ["build_features", "score"],
    "pricing": ["build_features", "evaluate"],
    "safety_risk": ["build_features", "evaluate"],
    "energy_cost": ["build_features", "evaluate"],
    "episode_quality": ["build_features", "evaluate"],
    "sampler_weights": ["build_features", "evaluate"],
    "orchestrator": ["advise"],
    "meta_advisor": ["build_features", "evaluate"],
    "vision_encoder": ["encode", "batch_encode"],
}


def ensure_methods(name, obj, methods):
    missing = [m for m in methods if not hasattr(obj, m)]
    if missing:
        raise AssertionError(f"{name} missing methods: {missing}")


def main():
    bundle = build_all_policies()
    bundle_dict = bundle.to_dict()
    for name, methods in REQUIRED_METHODS.items():
        policy = bundle_dict.get(name)
        if policy is None:
            raise AssertionError(f"{name} policy is None")
        ensure_methods(name, policy, methods)
    print("[smoke_test_policy_registry] OK")


if __name__ == "__main__":
    main()
