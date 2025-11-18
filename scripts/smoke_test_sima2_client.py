#!/usr/bin/env python3
"""
Smoke test for SIMA-2 stub client rollouts.
"""
import json
from pathlib import Path
import sys

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from src.sima2.client import Sima2Client


def main():
    client = Sima2Client(task_id="drawer_vase")
    r1 = client.run_episode({"task_id": "drawer_vase", "episode_index": 0})
    r2 = client.run_episode({"task_id": "drawer_vase", "episode_index": 1})
    assert r1["episode_id"] != r2["episode_id"]
    assert r1["primitives"]
    # Determinism check
    repeat = client.run_episode({"task_id": "drawer_vase", "episode_index": 0})
    assert r1 == repeat
    # Shape sanity
    for key in ("primitives", "actions", "observations"):
        assert isinstance(r1[key], list) and r1[key]
    print("[smoke_test_sima2_client] PASS")


if __name__ == "__main__":
    main()
