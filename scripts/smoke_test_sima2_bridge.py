#!/usr/bin/env python3
"""
Smoke test for SIMA-2 → Datapack bridge stubs.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sima2.datapack_adapter import (  # type: ignore
    sima2_episode_to_datapack_meta_stub,
    sima2_semantic_tokens_from_rollout_stub,
)
from src.valuation.datapack_schema import DataPackMeta  # type: ignore


def main():
    rollout = {
        "task_name": "drawer_vase",
        "env_type": "sima2_sim",
        "semantic_rollout": ["plan", "act", "recover"],
        "meta": {"run_id": "sima2_demo"},
    }
    dp = sima2_episode_to_datapack_meta_stub(rollout)
    assert isinstance(dp, DataPackMeta)
    assert dp.task_name == "drawer_vase"
    tokens = sima2_semantic_tokens_from_rollout_stub(rollout)
    assert tokens == rollout["semantic_rollout"]
    print("[smoke] SIMA-2 bridge stubs ready.")


if __name__ == "__main__":
    main()
