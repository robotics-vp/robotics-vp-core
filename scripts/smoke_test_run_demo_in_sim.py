#!/usr/bin/env python3
"""
Smoke test for run_demo_in_sim.py script.

Tests:
1. Script runs without error for minimal config
2. Output directory is created
3. episodes.jsonl is created and valid
4. steps.jsonl is created and valid
5. Metrics fields exist in episode summaries
6. Script exits with code 0

Exit code 0 on success, 1 on failure.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_run_demo_in_sim():
    """Run run_demo_in_sim.py with minimal config and verify outputs."""
    print("="*80)
    print("[smoke_test_run_demo_in_sim] Starting smoke test")
    print("="*80)
    print()

    # Output directory
    output_dir = ROOT / "results" / "smoke_test_demo_sim"

    # Clean up previous run
    if output_dir.exists():
        shutil.rmtree(output_dir)

    # Run script with minimal config
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_demo_in_sim.py"),
        "--env-backend", "pybullet",
        "--num-episodes", "1",
        "--max-steps", "5",
        "--seed", "0",
        "--output-dir", str(output_dir),
    ]

    print(f"[smoke_test_run_demo_in_sim] Running command:")
    print(f"  {' '.join(cmd)}")
    print()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(ROOT),
        )

        print("[smoke_test_run_demo_in_sim] Script output:")
        print(result.stdout)
        if result.stderr:
            print("[smoke_test_run_demo_in_sim] Script stderr:")
            print(result.stderr)
        print()

        # Check exit code
        assert result.returncode == 0, f"Script exited with code {result.returncode}"
        print("  ✓ Script exited with code 0")

        # Check output directory exists
        assert output_dir.exists(), f"Output directory {output_dir} not created"
        print(f"  ✓ Output directory created: {output_dir}")

        # Check episodes.jsonl exists
        episodes_path = output_dir / "episodes.jsonl"
        assert episodes_path.exists(), f"episodes.jsonl not found at {episodes_path}"
        print(f"  ✓ episodes.jsonl created")

        # Parse episodes.jsonl
        with open(episodes_path, "r") as f:
            episodes = [json.loads(line) for line in f]

        assert len(episodes) > 0, "episodes.jsonl is empty"
        print(f"  ✓ episodes.jsonl has {len(episodes)} episode(s)")

        # Check episode structure
        ep = episodes[0]
        required_fields = [
            "episode_id",
            "seed",
            "success",
            "steps",
            "total_reward",
            "econ_summary",
            "ood_stats",
            "recovery_stats",
            "skill_mode_counts",
        ]
        for field in required_fields:
            assert field in ep, f"Episode missing required field: {field}"
        print(f"  ✓ Episode summary has all required fields")

        # Check econ_summary structure
        econ_fields = ["avg_mpl", "avg_energy", "total_errors"]
        for field in econ_fields:
            assert field in ep["econ_summary"], f"econ_summary missing field: {field}"
        print(f"  ✓ econ_summary has all required fields")

        # Check ood_stats structure
        ood_fields = ["max_ood_risk", "ood_step_count", "ood_step_fraction"]
        for field in ood_fields:
            assert field in ep["ood_stats"], f"ood_stats missing field: {field}"
        print(f"  ✓ ood_stats has all required fields")

        # Check recovery_stats structure
        recovery_fields = ["max_recovery_priority", "recovery_step_count", "recovery_step_fraction"]
        for field in recovery_fields:
            assert field in ep["recovery_stats"], f"recovery_stats missing field: {field}"
        print(f"  ✓ recovery_stats has all required fields")

        # Check steps.jsonl exists
        steps_path = output_dir / "steps.jsonl"
        assert steps_path.exists(), f"steps.jsonl not found at {steps_path}"
        print(f"  ✓ steps.jsonl created")

        # Parse steps.jsonl
        with open(steps_path, "r") as f:
            steps = [json.loads(line) for line in f]

        assert len(steps) > 0, "steps.jsonl is empty"
        print(f"  ✓ steps.jsonl has {len(steps)} step(s)")

        # Check step structure
        step = steps[0]
        step_fields = [
            "episode_id",
            "step",
            "action_summary",
            "ood_step_flags",
            "recovery_step_flags",
            "reward_scalar",
            "econ_step_summary",
        ]
        for field in step_fields:
            assert field in step, f"Step missing required field: {field}"
        print(f"  ✓ Step log has all required fields")

        # Check JSON-safe
        try:
            json.dumps(ep)
            json.dumps(step)
        except Exception as e:
            raise AssertionError(f"Outputs not JSON-safe: {e}")
        print(f"  ✓ All outputs are JSON-safe")

        print()
        print("="*80)
        print("[smoke_test_run_demo_in_sim] All tests passed ✓")
        print("="*80)
        return 0

    except subprocess.TimeoutExpired:
        print()
        print("="*80)
        print("[smoke_test_run_demo_in_sim] Test failed: Script timeout (>60s)")
        print("="*80)
        return 1

    except AssertionError as e:
        print()
        print("="*80)
        print(f"[smoke_test_run_demo_in_sim] Test failed: {e}")
        print("="*80)
        return 1

    except Exception as e:
        print()
        print("="*80)
        print(f"[smoke_test_run_demo_in_sim] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        print("="*80)
        return 1


def main():
    """Run smoke test."""
    return test_run_demo_in_sim()


if __name__ == "__main__":
    sys.exit(main())
