#!/usr/bin/env python3
"""
Smoke test for Stage 1 pipeline: Video → Diffusion → VLA → DataPackMeta.

Validates:
- Pipeline runs without errors
- Generates valid datapacks
- Schema correctness
- Output files are created
"""

import json
import os
import shutil
import sys
from pathlib import Path

# Add repo root to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from scripts.run_stage1_pipeline import run_stage1_pipeline
from src.valuation.datapack_schema import DataPackMeta


def test_stage1_pipeline():
    """Test that Stage 1 pipeline runs and produces valid output."""
    print("=" * 70)
    print("Smoke Test: Stage 1 Pipeline")
    print("=" * 70)

    output_dir = "results/smoke_test_stage1"

    # Clean up previous test output
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    try:
        # Run pipeline with minimal parameters
        print("\n[1/5] Running Stage 1 pipeline...")
        stats = run_stage1_pipeline(
            num_videos=1,
            proposals_per_video=1,
            objective_preset="balanced",
            output_dir=output_dir,
        )
        print("✓ Pipeline completed successfully")

        # Check outputs
        print("\n[2/5] Checking output files...")
        datapacks_path = os.path.join(output_dir, "datapacks.json")
        log_path = os.path.join(output_dir, "pipeline_log.json")
        stats_path = os.path.join(output_dir, "pipeline_stats.json")

        assert os.path.exists(datapacks_path), "datapacks.json not created"
        assert os.path.exists(log_path), "pipeline_log.json not created"
        assert os.path.exists(stats_path), "pipeline_stats.json not created"
        print("✓ All output files present")

        # Load and validate datapacks
        print("\n[3/5] Validating datapacks...")
        with open(datapacks_path, "r") as f:
            datapacks_data = json.load(f)

        assert len(datapacks_data) >= 1, "No datapacks created"
        print(f"✓ Created {len(datapacks_data)} datapacks")

        # Validate schema
        print("\n[4/5] Validating datapack schema...")
        for i, dp_dict in enumerate(datapacks_data):
            # Try to reconstruct DataPackMeta from dict
            try:
                dp = DataPackMeta.from_dict(dp_dict)

                # Check required fields
                assert dp.pack_id, "pack_id is missing"
                assert dp.task_name, "task_name is missing"
                assert dp.condition is not None, "condition is None"
                assert dp.attribution is not None, "attribution is None"
                assert dp.objective_profile is not None, "objective_profile is None"
                assert dp.guidance_profile is not None, "guidance_profile is None"

                # Check attribution has required fields
                assert hasattr(dp.attribution, 'tier'), "attribution missing tier field"
                assert hasattr(dp.attribution, 'trust_score'), "attribution missing trust_score"
                assert hasattr(dp.attribution, 'delta_J'), "attribution missing delta_J"

                # Check guidance profile has required fields
                assert hasattr(dp.guidance_profile, 'is_good'), "guidance_profile missing is_good"
                assert hasattr(dp.guidance_profile, 'main_driver'), "guidance_profile missing main_driver"
                assert hasattr(dp.guidance_profile, 'semantic_tags'), "guidance_profile missing semantic_tags"

                # Check objective profile has required fields
                assert hasattr(dp.objective_profile, 'objective_vector'), "objective_profile missing objective_vector"
                assert len(dp.objective_profile.objective_vector) >= 4, "objective_vector too short"

                print(f"  ✓ Datapack {i+1}: {dp.pack_id[:20]}... (tier={dp.attribution.tier})")

            except Exception as e:
                print(f"  ✗ Datapack {i+1} failed schema validation: {e}")
                raise

        print("✓ All datapacks have valid schema")

        # Validate statistics
        print("\n[5/5] Validating statistics...")
        assert stats['total_videos'] == 1, f"Expected 1 video, got {stats['total_videos']}"
        assert stats['total_datapacks'] >= 1, f"Expected >=1 datapacks, got {stats['total_datapacks']}"
        assert 'tier_distribution' in stats, "Missing tier_distribution"
        assert 'avg_trust_score' in stats, "Missing avg_trust_score"
        print("✓ Statistics validated")

        print("\n" + "=" * 70)
        print("Stage 1 Pipeline Smoke Test: PASSED")
        print("=" * 70)
        print(f"Summary:")
        print(f"  Videos processed: {stats['total_videos']}")
        print(f"  Datapacks created: {stats['total_datapacks']}")
        print(f"  Tier distribution: {stats['tier_distribution']}")
        print(f"  Avg trust score: {stats['avg_trust_score']:.3f}")

        return True

    except Exception as e:
        print("\n" + "=" * 70)
        print("Stage 1 Pipeline Smoke Test: FAILED")
        print("=" * 70)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Cleanup
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
            print(f"\nCleaned up test directory: {output_dir}")


if __name__ == "__main__":
    success = test_stage1_pipeline()
    sys.exit(0 if success else 1)
