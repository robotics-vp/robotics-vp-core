#!/usr/bin/env python3
"""
Example: Build NAG scene from LSD vector scene backend.

Demonstrates the NAG overlay workflow:
1. Create camera parameters (static or time-varying)
2. Build NAGScene from LSD backend episode
3. Generate counterfactual clips with NAG edits
4. Analyze edit vectors for econ impact

Usage:
    python scripts/example_nag_from_lsd.py [--stub] [--seed 42]
"""
from __future__ import annotations

import argparse
import logging
import sys
from typing import Dict, Any, List

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def create_mock_lsd_episode(num_frames: int = 10, seed: int = 42) -> Dict[str, Any]:
    """Create a mock LSD backend episode for demonstration."""
    rng = np.random.default_rng(seed)

    # Mock scene graph with objects
    class MockObject:
        def __init__(self, id: int, x: float, y: float, class_name: str):
            self.id = id
            self.x = x
            self.y = y
            self.length = rng.uniform(0.5, 2.0)
            self.width = rng.uniform(0.5, 1.5)
            self.speed = rng.uniform(0, 1.0)
            self.heading = rng.uniform(0, 2 * np.pi)

            class ClassId:
                def __init__(self, name):
                    self.name = name
            self.class_id = ClassId(class_name)

    class MockSceneGraph:
        def __init__(self):
            self.objects = [
                MockObject(0, 5.0, 5.0, "ROBOT"),
                MockObject(1, 10.0, 8.0, "HUMAN"),
                MockObject(2, 3.0, 12.0, "PALLET"),
            ]

        def bounding_box(self):
            return (0, 0, 20, 20)

    # Mock Gaussian scene
    class MockGaussianScene:
        def __init__(self):
            self.num_gaussians = 50
            self.means = rng.uniform(0, 20, size=(50, 3)).astype(np.float32)
            self.colors = rng.uniform(0, 1, size=(50, 3)).astype(np.float32)
            self.opacities = rng.uniform(0.3, 1.0, size=(50,)).astype(np.float32)
            self.covs = rng.uniform(0.01, 0.1, size=(50, 6)).astype(np.float32)
            self.scales = rng.uniform(0.1, 0.5, size=(50, 3)).astype(np.float32)

    return {
        "episode_id": f"lsd_example_{seed}",
        "scene_id": "warehouse_demo",
        "num_frames": num_frames,
        "gaussian_scene": MockGaussianScene(),
        "scene_graph": MockSceneGraph(),
        "difficulty_features": {
            "graph_density": 0.5,
            "route_length": 30.0,
            "tilt": 0.0,
            "num_dynamic_agents": 3,
        },
        "mpl_metrics": {
            "mpl_units_per_hour": 45.0,
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Example NAG-from-LSD workflow",
    )
    parser.add_argument(
        "--stub",
        action="store_true",
        help="Use stub renderer (no real 3DGS rendering)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--num-counterfactuals",
        type=int,
        default=2,
        help="Number of counterfactual clips to generate",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save output datapacks (optional)",
    )
    parser.add_argument(
        "--filter-by-motion-hierarchy",
        action="store_true",
        help="Drop or mark NAG clips with low motion plausibility from MHN.",
    )
    parser.add_argument(
        "--mh-max-residual-mean",
        type=float,
        default=1.5,
        help="Maximum residual mean for motion plausibility checks.",
    )
    parser.add_argument(
        "--mh-min-plausibility-score",
        type=float,
        default=0.1,
        help="Minimum plausibility score for motion plausibility checks.",
    )
    args = parser.parse_args()

    try:
        import torch
    except ImportError:
        logger.error("PyTorch is required. Install with: pip install torch")
        sys.exit(1)

    from src.vision.nag import (
        CameraParams,
        NAGFromLSDConfig,
        NAGEditPolicyConfig,
        build_nag_scene_from_lsd_rollout,
        generate_nag_counterfactuals_for_lsd_episode,
        create_camera_from_lsd_config,
        SplattingGaussianRenderer,
        SplattingRendererConfig,
    )

    device = torch.device("cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"Stub renderer: {args.stub}")
    logger.info(f"Seed: {args.seed}")

    # Step 1: Create mock LSD episode
    logger.info("\n=== Step 1: Create mock LSD episode ===")
    episode = create_mock_lsd_episode(num_frames=10, seed=args.seed)
    logger.info(f"Episode ID: {episode['episode_id']}")
    logger.info(f"Scene graph objects: {len(episode['scene_graph'].objects)}")
    logger.info(f"Gaussian scene: {episode['gaussian_scene'].num_gaussians} gaussians")

    # Step 2: Configure NAG
    logger.info("\n=== Step 2: Configure NAG ===")
    nag_config = NAGFromLSDConfig(
        atlas_size=(128, 128),  # Smaller for demo
        max_iters=50,  # Fewer iterations for demo
        max_nodes=4,
        image_size=(128, 128),
        fov_deg=60.0,
        use_stub_renderer=args.stub,
        enable_motion_plausibility_filter=args.filter_by_motion_hierarchy,
        motion_plausibility_max_residual_mean=args.mh_max_residual_mean,
        motion_plausibility_min_score=args.mh_min_plausibility_score,
    )
    logger.info(f"NAG config: atlas_size={nag_config.atlas_size}, max_iters={nag_config.max_iters}")

    # Step 3: Create camera
    logger.info("\n=== Step 3: Create camera ===")
    camera = create_camera_from_lsd_config(nag_config, scene_bbox=(0, 0, 20, 20))
    logger.info(f"Camera: {camera.width}x{camera.height}, fov={camera.fov_deg}deg")
    logger.info(f"Static camera: {camera.is_static}")

    # Optional: Create time-varying camera
    if not camera.is_static:
        logger.info(f"Camera frames: {camera.num_frames}")

    # Step 4: Create renderer (if not using stub)
    renderer = None
    if not args.stub:
        logger.info("\n=== Step 4: Create renderer ===")
        renderer = SplattingGaussianRenderer(SplattingRendererConfig(use_gpu=False))
        logger.info("Created SplattingGaussianRenderer")

    # Step 5: Build NAG scene from LSD episode
    logger.info("\n=== Step 5: Build NAG scene ===")
    try:
        scene = build_nag_scene_from_lsd_rollout(
            backend_episode=episode,
            camera=camera,
            config=nag_config,
            device=device,
            renderer=renderer,
        )
        logger.info(f"NAG scene built successfully!")
        logger.info(f"Background node: {scene.background_node_id}")
        logger.info(f"Foreground nodes: {scene.get_foreground_nodes()}")
        logger.info(f"Metadata: {scene.metadata}")
    except Exception as e:
        logger.error(f"Failed to build NAG scene: {e}")
        sys.exit(1)

    # Step 6: Generate counterfactuals
    logger.info("\n=== Step 6: Generate counterfactuals ===")
    edit_config = NAGEditPolicyConfig(
        num_counterfactuals=args.num_counterfactuals,
        prob_remove=0.2,
        prob_duplicate=0.2,
        prob_pose_shift=0.4,
        prob_color_shift=0.2,
        max_edits_per_counterfactual=2,
    )

    datapacks = generate_nag_counterfactuals_for_lsd_episode(
        backend_episode=episode,
        camera=camera,
        nag_config=nag_config,
        edit_config=edit_config,
        device=device,
        renderer=renderer,
        seed=args.seed,
    )

    logger.info(f"Generated {len(datapacks)} counterfactual datapacks")
    for i, dp in enumerate(datapacks):
        logger.info(f"\n  Datapack {i+1}: {dp.counterfactual_id}")
        logger.info(f"    Frames shape: {dp.frames.shape}")
        logger.info(f"    Edits: {[e.get('edit_type') for e in dp.nag_edit_vector]}")
        logger.info(f"    NAG difficulty features: {dp.difficulty_features}")

    # Step 7: Analyze NAG surface (demo)
    logger.info("\n=== Step 7: Analyze NAG edit surface ===")
    from src.analytics.econ_reports import compute_nag_edit_surface_summary

    nag_datapacks_dicts = [dp.to_dict() for dp in datapacks]
    surface_summary = compute_nag_edit_surface_summary(nag_datapacks_dicts)

    logger.info(f"Edit type distribution: {surface_summary.get('edit_type_distribution', {})}")
    logger.info(f"Counterfactual impact: {surface_summary.get('counterfactual_impact', {})}")

    # Optional: Save datapacks
    if args.output_dir:
        import json
        from pathlib import Path

        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for dp in datapacks:
            dp_dict = dp.to_dict()
            dp_dict['frames_path'] = f"{dp.counterfactual_id}_frames.npy"

            # Save metadata
            with open(output_path / f"{dp.counterfactual_id}.json", 'w') as f:
                json.dump(dp_dict, f, indent=2)

            # Save frames
            np.save(output_path / dp_dict['frames_path'], dp.frames)

        logger.info(f"\nSaved datapacks to {output_path}")

    logger.info("\n=== Example complete! ===")


if __name__ == "__main__":
    main()
