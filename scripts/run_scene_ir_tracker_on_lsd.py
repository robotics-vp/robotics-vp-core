#!/usr/bin/env python3
"""
Run Scene IR Tracker on LSD Vector Scene episodes.

This script:
1. Loads or generates LSD sample episodes
2. Runs the scene IR tracker on each
3. Writes overlays/video and dumps metrics
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def create_synthetic_episode(
    num_frames: int = 50,
    height: int = 256,
    width: int = 256,
    num_objects: int = 3,
    num_bodies: int = 1,
    seed: int = 42,
) -> Dict[str, Any]:
    """Create synthetic episode data for testing."""
    rng = np.random.RandomState(seed)

    frames = []
    instance_masks = []
    class_labels = []

    for t in range(num_frames):
        # Generate synthetic frame
        frame = rng.randint(0, 255, (height, width, 3), dtype=np.uint8)
        frames.append(frame)

        # Generate synthetic masks
        frame_masks = {}
        frame_labels = {}

        # Objects
        for i in range(num_objects):
            mask = np.zeros((height, width), dtype=bool)
            cx = int(width * (0.2 + 0.6 * rng.rand()))
            cy = int(height * (0.2 + 0.6 * rng.rand()))
            radius = int(20 + 30 * rng.rand())
            yy, xx = np.ogrid[:height, :width]
            circle = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
            mask[circle] = True
            frame_masks[f"obj_{i}"] = mask
            frame_labels[f"obj_{i}"] = "box"

        # Bodies
        for i in range(num_bodies):
            mask = np.zeros((height, width), dtype=bool)
            cx = int(width * (0.3 + 0.4 * rng.rand()))
            cy = int(height * 0.5)
            # Ellipse for body
            yy, xx = np.ogrid[:height, :width]
            ellipse = ((xx - cx) / 30) ** 2 + ((yy - cy) / 50) ** 2 <= 1
            mask[ellipse] = True
            frame_masks[f"body_{i}"] = mask
            frame_labels[f"body_{i}"] = "person"

        instance_masks.append(frame_masks)
        class_labels.append(frame_labels)

    return {
        "frames": frames,
        "instance_masks": instance_masks,
        "class_labels": class_labels,
        "num_frames": num_frames,
    }


def run_tracker_on_episode(
    episode: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Run scene IR tracker on an episode."""
    from src.vision.nag.types import CameraParams
    from src.vision.scene_ir_tracker import SceneIRTracker, SceneIRTrackerConfig

    # Create camera
    height, width = episode["frames"][0].shape[:2]
    camera = CameraParams.from_single_pose(
        position=(0.0, 0.0, -5.0),
        look_at=(0.0, 0.0, 0.0),
        up=(0.0, 1.0, 0.0),
        fov_deg=60.0,
        width=width,
        height=height,
    )

    # Create tracker
    tracker_config = SceneIRTrackerConfig.from_dict(config)
    tracker = SceneIRTracker(tracker_config)

    # Run tracker
    scene_tracks = tracker.process_episode(
        frames=episode["frames"],
        instance_masks=episode["instance_masks"],
        camera=camera,
        class_labels=episode.get("class_labels"),
    )

    return {
        "scene_tracks": scene_tracks.to_dict(),
        "summary": scene_tracks.summary(),
        "metrics": scene_tracks.metrics.to_dict(),
    }


def save_metrics(
    results: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """Save tracker metrics to JSON."""
    summaries = [r["summary"] for r in results]
    all_metrics = {
        "num_episodes": len(results),
        "episode_summaries": summaries,
        "aggregate": {
            "mean_ir_loss": float(np.mean([s["mean_ir_loss"] for s in summaries])),
            "total_id_switches": sum(s["id_switch_count"] for s in summaries),
            "mean_occlusion_rate": float(np.mean([s["occlusion_rate"] for s in summaries])),
        },
    }

    with open(output_path / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    logger.info(f"Saved metrics to {output_path / 'metrics.json'}")


def save_overlays(
    episode: Dict[str, Any],
    result: Dict[str, Any],
    output_path: Path,
    episode_idx: int,
) -> None:
    """Save overlay images of tracked entities over RGB frames."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        logger.warning("PIL not available, skipping overlays")
        return

    frames = episode["frames"]
    tracks_data = result.get("scene_tracks", {})
    frame_entities = tracks_data.get("frames", [])

    overlay_dir = output_path / f"episode_{episode_idx:04d}" / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    for t, frame in enumerate(frames):
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img, mode='RGBA')

        # Draw entity info if available
        if t < len(frame_entities):
            entities = frame_entities[t]
            for ent in entities:
                if isinstance(ent, dict):
                    track_id = ent.get("track_id", "?")
                    entity_type = ent.get("entity_type", "obj")
                    ir_loss = ent.get("ir_loss", 0.0)
                    
                    # Draw label
                    color = (0, 255, 0, 128) if entity_type == "body" else (0, 0, 255, 128)
                    label = f"{track_id}: L={ir_loss:.3f}"
                    draw.text((10, 10 + entities.index(ent) * 15), label, fill=color)

        img.save(overlay_dir / f"frame_{t:04d}.png")

    logger.info(f"Saved overlays to {overlay_dir}")


def save_loss_curves(
    results: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """Save per-phase loss curves to JSON."""
    loss_curves = []
    
    for ep_idx, r in enumerate(results):
        metrics = r.get("metrics", {})
        ir_losses = metrics.get("ir_loss_per_frame", [])
        
        loss_curves.append({
            "episode": ep_idx,
            "ir_loss_per_frame": ir_losses,
            "mean_ir_loss": float(np.mean(ir_losses)) if ir_losses else 0.0,
            "min_ir_loss": float(np.min(ir_losses)) if ir_losses else 0.0,
            "max_ir_loss": float(np.max(ir_losses)) if ir_losses else 0.0,
        })

    with open(output_path / "loss_curves.json", "w") as f:
        json.dump(loss_curves, f, indent=2)

    logger.info(f"Saved loss curves to {output_path / 'loss_curves.json'}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Scene IR Tracker on LSD episodes")
    parser.add_argument(
        "--num-episodes", type=int, default=1,
        help="Number of synthetic episodes to process"
    )
    parser.add_argument(
        "--num-frames", type=int, default=20,
        help="Frames per episode"
    )
    parser.add_argument(
        "--output", type=str, default="tmp/scene_ir_tracker_output",
        help="Output directory"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device for computation"
    )
    parser.add_argument(
        "--save-overlays", action="store_true",
        help="Save overlay images"
    )
    parser.add_argument(
        "--save-loss-curves", action="store_true",
        help="Save per-phase loss curves"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Verbose output"
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Processing {args.num_episodes} episodes...")

    results = []
    episodes = []
    for ep_idx in range(args.num_episodes):
        logger.info(f"Episode {ep_idx + 1}/{args.num_episodes}")

        # Create synthetic episode
        episode = create_synthetic_episode(
            num_frames=args.num_frames,
            seed=args.seed + ep_idx,
        )
        episodes.append(episode)

        # Run tracker
        config = {
            "device": args.device,
            "use_stub_adapters": True,
        }
        result = run_tracker_on_episode(episode, config)
        results.append(result)

        logger.info(f"  Tracks: {result['summary']['num_tracks']}")
        logger.info(f"  Mean IR loss: {result['summary']['mean_ir_loss']:.4f}")
        logger.info(f"  ID switches: {result['summary']['id_switch_count']}")

        # Save overlays if requested
        if args.save_overlays:
            save_overlays(episode, result, output_path, ep_idx)

    # Save metrics
    save_metrics(results, output_path)

    # Save loss curves if requested
    if args.save_loss_curves:
        save_loss_curves(results, output_path)

    logger.info("Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())

