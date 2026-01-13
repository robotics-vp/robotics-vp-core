#!/usr/bin/env python3
"""
Smoke test script for workcell environment suite.

Runs basic tests for kitting and peg-in-hole tasks with deterministic seeds.
Exits nonzero on failure.

REGALITY COMPLIANCE: BASIC
--------------------------
This is a basic smoke test for env functionality only.
Does NOT produce: manifest, ledger, trajectory audit, selection manifest, etc.

For FULL regality compliance, use:
    python scripts/run_workcell_regal.py --output-dir artifacts/workcell_regal
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_config_loading() -> bool:
    """Test configuration loading and presets."""
    print("=" * 60)
    print("TEST: Config loading and presets")
    print("=" * 60)

    try:
        from src.envs.workcell_env.config import WorkcellEnvConfig, PRESETS

        # Test default config
        config = WorkcellEnvConfig()
        assert config.topology_type == "ASSEMBLY_BENCH"
        assert config.max_steps > 0
        print(f"  Default config: topology={config.topology_type}, max_steps={config.max_steps}")

        # Test presets
        for name, preset in PRESETS.items():
            assert isinstance(preset, WorkcellEnvConfig)
            print(f"  Preset '{name}': topology={preset.topology_type}")

        # Test serialization roundtrip
        config_dict = config.to_dict()
        config_restored = WorkcellEnvConfig.from_dict(config_dict)
        assert config == config_restored
        print("  Serialization roundtrip: OK")

        print("PASS: Config loading\n")
        return True

    except Exception as e:
        print(f"FAIL: {e}")
        traceback.print_exc()
        return False


def test_scene_generation() -> bool:
    """Test procedural scene generation."""
    print("=" * 60)
    print("TEST: Scene generation")
    print("=" * 60)

    try:
        from src.envs.workcell_env.config import WorkcellEnvConfig
        from src.envs.workcell_env.scene.generators import WorkcellSceneGenerator

        generator = WorkcellSceneGenerator()

        # Test with default config
        config = WorkcellEnvConfig()
        scene = generator.generate(config, seed=42)

        assert scene is not None
        assert scene.workcell_id is not None
        print(f"  Generated scene: {scene.workcell_id}")
        print(f"    Stations: {len(scene.stations)}")
        print(f"    Fixtures: {len(scene.fixtures)}")
        print(f"    Parts: {len(scene.parts)}")
        print(f"    Containers: {len(scene.containers)}")

        # Test deterministic generation
        scene2 = generator.generate(config, seed=42)
        assert scene.workcell_id == scene2.workcell_id
        print("  Deterministic generation: OK")

        print("PASS: Scene generation\n")
        return True

    except Exception as e:
        print(f"FAIL: {e}")
        traceback.print_exc()
        return False


def test_task_compilation() -> bool:
    """Test prompt-to-task compilation."""
    print("=" * 60)
    print("TEST: Task compilation from prompts")
    print("=" * 60)

    try:
        from src.envs.workcell_env.compiler import WorkcellTaskCompiler

        compiler = WorkcellTaskCompiler()

        # Test various prompts
        prompts = [
            ("Pack 6 items into a tray", "kitting"),
            ("Sort 20 widgets into 3 bins by color", "sorting"),
            ("Insert peg into hole with 1mm tolerance", "peg_in_hole"),
            ("Assemble bracket with 2 screws", "assembly"),
            ("Inspect 5 parts for defects", "inspection"),
        ]

        for prompt, expected_type in prompts:
            result = compiler.compile_from_prompt(prompt, seed=42)
            print(f"  '{prompt[:40]}...'")
            print(f"    -> task_type={result.inferred_task_type}, "
                  f"nodes={len(result.task_graph.nodes)}")
            assert result.inferred_task_type == expected_type, \
                f"Expected {expected_type}, got {result.inferred_task_type}"

        print("PASS: Task compilation\n")
        return True

    except Exception as e:
        print(f"FAIL: {e}")
        traceback.print_exc()
        return False


def test_env_reset_step() -> bool:
    """Test environment reset and step."""
    print("=" * 60)
    print("TEST: Environment reset and step")
    print("=" * 60)

    try:
        from src.envs.workcell_env.env import WorkcellEnv
        from src.envs.workcell_env.config import WorkcellEnvConfig

        config = WorkcellEnvConfig(max_steps=10)
        env = WorkcellEnv(config=config)

        # Test reset
        obs = env.reset(seed=42)
        assert obs is not None
        print(f"  Reset: obs keys = {list(obs.keys())[:5]}...")

        # Test step
        action = {"action_type": "PICK", "target": "part_0"}
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs is not None
        assert isinstance(reward, (int, float))
        print(f"  Step: reward={reward:.3f}, terminated={terminated}")

        # Run a few more steps
        for i in range(5):
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

        print(f"  Ran {i+2} steps total")

        env.close()
        print("PASS: Environment reset and step\n")
        return True

    except Exception as e:
        print(f"FAIL: {e}")
        traceback.print_exc()
        return False


def test_difficulty_features() -> bool:
    """Test difficulty feature computation."""
    print("=" * 60)
    print("TEST: Difficulty features")
    print("=" * 60)

    try:
        from src.envs.workcell_env.config import WorkcellEnvConfig, PRESETS
        from src.envs.workcell_env.difficulty.difficulty_features import compute_difficulty_features

        for name, config in PRESETS.items():
            features = compute_difficulty_features(config)
            composite = features.composite_difficulty()
            print(f"  {name}: composite_difficulty={composite:.2f}")
            assert 0.0 <= composite <= 1.0

        print("PASS: Difficulty features\n")
        return True

    except Exception as e:
        print(f"FAIL: {e}")
        traceback.print_exc()
        return False


def test_analytics() -> bool:
    """Test analytics integration."""
    print("=" * 60)
    print("TEST: Analytics integration")
    print("=" * 60)

    try:
        from src.analytics.workcell_analytics import (
            compute_episode_metrics,
            compute_suite_report,
            format_suite_report,
        )

        # Create mock episode data
        episodes = [
            {"success": True, "total_reward": 10.0, "steps": 50,
             "items_completed": 5, "items_total": 6, "errors": 0},
            {"success": True, "total_reward": 8.0, "steps": 60,
             "items_completed": 4, "items_total": 6, "errors": 1},
            {"success": False, "total_reward": 3.0, "steps": 100,
             "items_completed": 2, "items_total": 6, "errors": 2},
        ]

        metrics_list = []
        for i, ep in enumerate(episodes):
            m = compute_episode_metrics(f"ep_{i}", "kitting", ep)
            metrics_list.append(m)
            print(f"  Episode {i}: quality={m.quality_score:.2f}, wage=${m.implied_wage:.2f}/hr")

        report = compute_suite_report(metrics_list)
        print(f"  Suite success rate: {report.success_rate:.0%}")
        print(f"  Mean quality: {report.mean_quality_score:.2f}")

        # Test formatting
        report_str = format_suite_report(report)
        assert "WORKCELL SUITE REPORT" in report_str

        print("PASS: Analytics integration\n")
        return True

    except Exception as e:
        print(f"FAIL: {e}")
        traceback.print_exc()
        return False


def main() -> int:
    """Run all smoke tests."""
    print("\n" + "=" * 60)
    print("WORKCELL ENVIRONMENT SUITE - SMOKE TESTS")
    print("=" * 60 + "\n")

    tests = [
        ("Config loading", test_config_loading),
        ("Scene generation", test_scene_generation),
        ("Task compilation", test_task_compilation),
        ("Environment reset/step", test_env_reset_step),
        ("Difficulty features", test_difficulty_features),
        ("Analytics", test_analytics),
    ]

    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, passed))
        except Exception as e:
            print(f"ERROR in {name}: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, p in results if p)
    total = len(results)

    for name, p in results:
        status = "PASS" if p else "FAIL"
        print(f"  [{status}] {name}")

    print(f"\nTotal: {passed}/{total} passed")

    if passed == total:
        print("\nALL SMOKE TESTS PASSED")
        return 0
    else:
        print("\nSOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
