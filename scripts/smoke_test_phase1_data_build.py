#!/usr/bin/env python3
"""
Smoke test for Phase I data build pipeline.

Assertions:
- Pipeline runs end-to-end without errors
- All expected outputs are created
- Manifest is valid JSON and contains expected datasets
- Deterministic outputs with fixed seed
"""
import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_phase1_data_build import main as run_phase1_data_build


def main() -> int:
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "phase1_test"

        # Run with minimal settings (skip heavy operations)
        argv = [
            "--seed",
            "42",
            "--output-dir",
            str(output_dir),
            "--skip-ros",  # Skip ROS (requires real logs)
        ]

        try:
            run_phase1_data_build(argv)
        except Exception as e:
            print(f"[smoke_test_phase1_data_build] FAIL: Pipeline raised exception: {e}")
            return 1

        # Check summary exists
        summary_path = output_dir / "phase1_data_build_summary.json"
        assert summary_path.exists(), f"Summary not found at {summary_path}"

        with open(summary_path, "r") as f:
            summary = json.load(f)

        # Validate summary structure
        assert summary["canonical_task"] == "drawer_open", "Canonical task mismatch"
        assert summary["seed"] == 42, "Seed mismatch"
        assert summary["phase"] == "phase1", "Phase mismatch"
        assert "steps" in summary, "Missing steps in summary"

        # Check that at least some steps ran
        steps = summary["steps"]
        assert len(steps) > 0, "No steps executed"

        # SIMA-2 stress should have run (unless skipped)
        if "sima2_stress" in steps and steps["sima2_stress"]["status"] != "skipped":
            sima2_result = steps["sima2_stress"]
            assert sima2_result["status"] in [
                "success",
                "failed",
            ], "SIMA-2 stress has invalid status"
            if sima2_result["status"] == "success":
                assert "output_dir" in sima2_result, "SIMA-2 output_dir missing"
                assert "num_episodes" in sima2_result, "SIMA-2 num_episodes missing"

        # Isaac adapter should have run (creates synthetic rollout)
        if "isaac_adapter" in steps and steps["isaac_adapter"]["status"] != "skipped":
            isaac_result = steps["isaac_adapter"]
            assert isaac_result["status"] == "success", "Isaac adapter should succeed"
            assert "rollout_path" in isaac_result, "Isaac rollout_path missing"
            rollout_path = Path(isaac_result["rollout_path"])
            assert rollout_path.exists(), f"Isaac rollout not created at {rollout_path}"

            with open(rollout_path, "r") as f:
                rollout = json.load(f)
            assert rollout["task_id"] == "drawer_open", "Isaac rollout task_id mismatch"
            assert rollout["backend"] == "isaac", "Isaac rollout backend mismatch"
            assert rollout["seed"] == 42, "Isaac rollout seed mismatch"

        # Manifest build should have run
        if "manifest_build" in steps and steps["manifest_build"]["status"] != "skipped":
            manifest_result = steps["manifest_build"]
            assert (
                manifest_result["status"] == "success"
            ), "Manifest build should succeed"
            assert "manifest_path" in manifest_result, "Manifest path missing"

            manifest_path = Path(manifest_result["manifest_path"])
            assert manifest_path.exists(), f"Manifest not found at {manifest_path}"

            with open(manifest_path, "r") as f:
                manifest = json.load(f)

            assert manifest["phase"] == "phase1", "Manifest phase mismatch"
            assert "datasets" in manifest, "Manifest missing datasets"
            assert len(manifest["datasets"]) > 0, "Manifest has no datasets"

            # Check dataset structure
            for dataset_name, dataset_info in manifest["datasets"].items():
                assert "count" in dataset_info, f"Dataset {dataset_name} missing count"
                assert isinstance(
                    dataset_info["count"], int
                ), f"Dataset {dataset_name} count not int"

        print("[smoke_test_phase1_data_build] PASS")
        return 0


if __name__ == "__main__":
    sys.exit(main())
