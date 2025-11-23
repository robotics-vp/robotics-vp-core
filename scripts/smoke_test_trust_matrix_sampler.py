#!/usr/bin/env python3
"""
Smoke test for trust-aware sampler weights.
"""
from pathlib import Path
import sys

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from src.policies.sampler_weights import HeuristicSamplerWeightPolicy


def _descriptor(ep_id: str, tag_type: str):
    return {
        "descriptor": {"episode_id": ep_id, "semantic_tags": [{"tag_type": tag_type}], "trust_score": 0.9, "sampling_weight": 1.0},
        "semantic_tags": [{"tag_type": tag_type}],
    }


def main():
    trust_matrix = {
        "OODTag": {"trust_score": 0.9},
        "RecoveryTag": {"trust_score": 0.6},
        "NoveltyTag": {"trust_score": 0.2},
    }
    policy = HeuristicSamplerWeightPolicy(trust_matrix=trust_matrix)
    features = policy.build_features([
        _descriptor("ep_trusted", "OODTag"),
        _descriptor("ep_provisional", "RecoveryTag"),
        _descriptor("ep_untrusted", "NoveltyTag"),
    ])

    weights = policy.evaluate(features, strategy="balanced")
    assert weights["ep_trusted"] > weights["ep_provisional"] > weights["ep_untrusted"], "Expected trusted > provisional > untrusted"
    trusted_ratio = weights["ep_trusted"] / max(weights["ep_untrusted"], 1e-6)
    provisional_ratio = weights["ep_provisional"] / max(weights["ep_untrusted"], 1e-6)
    assert trusted_ratio >= 4.5, "Trusted multiplier should be ~5x"
    assert 1.3 <= provisional_ratio <= 2.2, "Provisional multiplier should be ~1.5x"

    print("[smoke_test_trust_matrix_sampler] PASS")


if __name__ == "__main__":
    main()
