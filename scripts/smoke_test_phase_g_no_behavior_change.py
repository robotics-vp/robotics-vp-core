#!/usr/bin/env python3
"""
Phase G integration smoke: ensures registry wiring does not change behavior.

Runs the standard smoke bundle and optionally a short SAC+ontology run to
verify losses remain finite. Advisory-only; does not alter any reward math.
"""
import argparse
import subprocess
import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))


def main():
    parser = argparse.ArgumentParser(description="Phase G behavior-preserving smoke")
    parser.add_argument("--run-train", action="store_true", help="Optionally run a short SAC + ontology logging job")
    parser.add_argument("--train-episodes", type=int, default=2)
    args = parser.parse_args()

    print("[phase_g_smoke] Running core smoke bundle via run_all_smokes.py")
    subprocess.run(["python3", "scripts/run_all_smokes.py"], check=True)

    if args.run_train:
        print("[phase_g_smoke] Running optional SAC training smoke")
        subprocess.run(
            [
                "python3",
                "scripts/train_sac_with_ontology_logging.py",
                "--episodes",
                str(args.train_episodes),
                "--seed",
                "0",
            ],
            check=True,
        )
    print("[phase_g_smoke] Completed")


if __name__ == "__main__":
    main()
