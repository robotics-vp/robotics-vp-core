#!/usr/bin/env python3
"""
Generate deterministic SIMA-2 rollouts using the stub client.
"""
import argparse
import json
from pathlib import Path
import sys

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from src.sima2.client import Sima2Client


def main():
    parser = argparse.ArgumentParser(description="Run SIMA-2 stub rollouts.")
    parser.add_argument("--task-id", type=str, default="drawer_vase")
    parser.add_argument("--num-episodes", type=int, default=2)
    parser.add_argument("--output-jsonl", type=str, default="")
    args = parser.parse_args()

    out_path = Path(args.output_jsonl or f"results/sima2/rollouts_{args.task_id}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    client = Sima2Client(task_id=args.task_id)
    with out_path.open("w") as f:
        for idx in range(args.num_episodes):
            rollout = client.run_episode({"task_id": args.task_id, "episode_index": idx})
            f.write(json.dumps(rollout, sort_keys=True))
            f.write("\n")
    print(f"[run_sima2_rollouts] wrote {args.num_episodes} episodes to {out_path}")


if __name__ == "__main__":
    main()
