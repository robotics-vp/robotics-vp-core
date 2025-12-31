#!/usr/bin/env python3
"""
Visual debugging utilities for LSD Vector Scene environments.

This script:
1. Loads or generates an LSD Vector Scene
2. Produces 2D top-down plots of the scene graph
3. Optionally renders Gaussian scene views
4. Helps verify alignment between graph, mesh, and Gaussians

Usage:
    # Basic visualization with default config:
    python scripts/debug_lsd_vector_scene_viz.py --output-dir outputs/viz

    # Load specific config:
    python scripts/debug_lsd_vector_scene_viz.py \
        --config-path configs/lsd_vector_scene/smoke_warehouse.yaml \
        --output-dir outputs/viz_warehouse

    # Headless mode (no display):
    python scripts/debug_lsd_vector_scene_viz.py --no-show --output-dir outputs/viz
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Optional matplotlib import
try:
    import matplotlib
    matplotlib.use("Agg")  # Use non-interactive backend by default
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Circle, FancyArrowPatch
    from matplotlib.collections import LineCollection
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("WARNING: matplotlib not available. Visualization will be limited.")


def plot_scene_graph_2d(
    graph,
    ax=None,
    show_nodes: bool = True,
    show_edges: bool = True,
    show_objects: bool = True,
    title: str = "Scene Graph (Top-Down View)",
) -> Any:
    """
    Plot a 2D top-down view of the scene graph.

    Args:
        graph: SceneGraph instance
        ax: matplotlib axes (creates new if None)
        show_nodes: Whether to show node polylines
        show_edges: Whether to show edge connections
        show_objects: Whether to show scene objects

    Returns:
        matplotlib axes
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Cannot plot: matplotlib not available")
        return None

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # Color scheme
    node_colors = {
        "CORRIDOR": "lightblue",
        "ROOM": "lightgreen",
        "AISLE": "lightyellow",
        "ZONE": "lightgray",
        "DOOR": "orange",
        "INTERSECTION": "pink",
    }
    object_colors = {
        "HUMAN": "red",
        "ROBOT": "blue",
        "FORKLIFT": "purple",
        "PALLET": "brown",
        "SHELF": "gray",
        "OBSTACLE": "black",
    }

    # Plot nodes as polylines
    if show_nodes and hasattr(graph, "nodes"):
        for node in graph.nodes:
            polyline = node.polyline
            if len(polyline) >= 2:
                xs = [p[0] for p in polyline]
                ys = [p[1] for p in polyline]

                # Get color based on node type
                node_type_str = str(node.node_type.name) if hasattr(node.node_type, "name") else str(node.node_type)
                color = node_colors.get(node_type_str, "lightblue")

                ax.plot(xs, ys, linewidth=8, color=color, alpha=0.6, solid_capstyle="round")
                ax.plot(xs, ys, linewidth=2, color="black", alpha=0.8)

                # Label node
                cx, cy = np.mean(xs), np.mean(ys)
                ax.text(cx, cy, f"N{node.id}", fontsize=8, ha="center", va="center")

    # Plot edges as arrows between node centers
    if show_edges and hasattr(graph, "edges") and hasattr(graph, "nodes"):
        node_centers = {}
        for node in graph.nodes:
            polyline = node.polyline
            if len(polyline) >= 1:
                xs = [p[0] for p in polyline]
                ys = [p[1] for p in polyline]
                node_centers[node.id] = (np.mean(xs), np.mean(ys))

        for edge in graph.edges:
            src = node_centers.get(edge.src_id)
            dst = node_centers.get(edge.dst_id)
            if src and dst:
                dx = dst[0] - src[0]
                dy = dst[1] - src[1]
                length = np.sqrt(dx**2 + dy**2)
                if length > 0.1:
                    # Shorten arrow to not overlap with nodes
                    shrink = 0.2
                    start_x = src[0] + dx * shrink
                    start_y = src[1] + dy * shrink
                    end_x = dst[0] - dx * shrink
                    end_y = dst[1] - dy * shrink

                    ax.annotate(
                        "",
                        xy=(end_x, end_y),
                        xytext=(start_x, start_y),
                        arrowprops=dict(arrowstyle="->", color="gray", alpha=0.5),
                    )

    # Plot objects
    if show_objects and hasattr(graph, "objects"):
        for obj in graph.objects:
            x, y = obj.x, obj.y
            obj_class = str(obj.class_id.name) if hasattr(obj.class_id, "name") else str(obj.class_id)
            color = object_colors.get(obj_class, "gray")

            # Draw object as circle
            circle = Circle((x, y), radius=0.5, color=color, alpha=0.7)
            ax.add_patch(circle)

            # Add heading indicator
            heading = obj.heading
            dx = 0.5 * np.cos(heading)
            dy = 0.5 * np.sin(heading)
            ax.arrow(x, y, dx, dy, head_width=0.2, head_length=0.1, fc=color, ec="black")

            # Label
            ax.text(x, y + 0.8, f"O{obj.id}", fontsize=6, ha="center")

    # Set axis properties
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.grid(True, alpha=0.3)

    # Auto-fit axis limits
    if hasattr(graph, "bounding_box"):
        bbox = graph.bounding_box()
        margin = 2.0
        ax.set_xlim(bbox[0] - margin, bbox[2] + margin)
        ax.set_ylim(bbox[1] - margin, bbox[3] + margin)

    return ax


def plot_voxel_slice(
    voxels,
    z_slice: int = 0,
    ax=None,
    title: str = "Voxel Grid (Slice)",
) -> Any:
    """Plot a horizontal slice of the voxel grid."""
    if not MATPLOTLIB_AVAILABLE:
        return None

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    data = voxels.data
    if z_slice < 0 or z_slice >= data.shape[2]:
        z_slice = data.shape[2] // 2

    slice_data = data[:, :, z_slice].T  # Transpose for correct orientation

    ax.imshow(slice_data, origin="lower", cmap="gray_r", aspect="equal")
    ax.set_title(f"{title} (z={z_slice})")
    ax.set_xlabel("X (voxels)")
    ax.set_ylabel("Y (voxels)")

    return ax


def plot_gaussian_stats(
    gaussian_scene,
    ax=None,
    title: str = "Gaussian Scene Statistics",
) -> Any:
    """Plot statistics about the Gaussian scene."""
    if not MATPLOTLIB_AVAILABLE:
        return None

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    means = gaussian_scene.means
    opacities = gaussian_scene.opacities

    # Create histogram of Gaussian positions (top-down)
    ax.hist2d(
        means[:, 0],
        means[:, 1],
        bins=50,
        cmap="viridis",
        weights=opacities.flatten(),
    )
    ax.set_title(title)
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.set_aspect("equal")

    return ax


def create_debug_visualization(
    config_path: Optional[str] = None,
    output_dir: str = "outputs/viz",
    show: bool = True,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Create debug visualization for an LSD Vector Scene.

    Returns:
        Dict with info about generated visualizations
    """
    from src.envs.lsd_vector_scene_env import (
        LSDVectorSceneEnv,
        LSDVectorSceneEnvConfig,
        SceneGraphConfig,
        VisualStyleConfig,
        BehaviourConfig,
        create_env_from_config_path,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create environment
    if config_path and Path(config_path).exists():
        print(f"Loading config from: {config_path}")
        env = create_env_from_config_path(config_path)
    else:
        print("Using default configuration")
        config = LSDVectorSceneEnvConfig(
            scene_graph_config=SceneGraphConfig(
                topology_type="WAREHOUSE_AISLES",
                num_nodes=8,
                num_objects=10,
                density=0.5,
                route_length=25.0,
                seed=seed,
            ),
            visual_style_config=VisualStyleConfig(
                lighting="BRIGHT_INDOOR",
                clutter_level="MEDIUM",
                voxel_size=0.2,
            ),
            behaviour_config=BehaviourConfig(
                num_humans=2,
                num_robots=1,
                num_forklifts=1,
                use_simple_policy=True,
            ),
            max_steps=50,
        )
        env = LSDVectorSceneEnv(config)

    # Reset to generate scene
    obs = env.reset()
    scene_id = env.scene_id

    print(f"Scene ID: {scene_id}")
    print(f"Num nodes: {len(env.graph.nodes) if env.graph else 0}")
    print(f"Num objects: {len(env.graph.objects) if env.graph else 0}")
    if env.voxels:
        print(f"Voxel shape: {env.voxels.shape}")
    if env.gaussian_scene:
        print(f"Num Gaussians: {env.gaussian_scene.num_gaussians}")

    results = {
        "scene_id": scene_id,
        "output_dir": str(output_path),
        "files": [],
    }

    if not MATPLOTLIB_AVAILABLE:
        print("Skipping visualization: matplotlib not available")
        return results

    # Create multi-panel figure
    fig = plt.figure(figsize=(20, 10))

    # Panel 1: Scene graph top-down
    ax1 = fig.add_subplot(2, 3, 1)
    if env.graph:
        plot_scene_graph_2d(env.graph, ax=ax1, title=f"Scene Graph: {scene_id[:8]}")

    # Panel 2: Voxel slice (ground level)
    ax2 = fig.add_subplot(2, 3, 2)
    if env.voxels:
        plot_voxel_slice(env.voxels, z_slice=0, ax=ax2, title="Voxel Grid (z=0)")

    # Panel 3: Voxel slice (mid level)
    ax3 = fig.add_subplot(2, 3, 3)
    if env.voxels:
        mid_z = env.voxels.shape[2] // 2
        plot_voxel_slice(env.voxels, z_slice=mid_z, ax=ax3, title=f"Voxel Grid (z={mid_z})")

    # Panel 4: Gaussian density
    ax4 = fig.add_subplot(2, 3, 4)
    if env.gaussian_scene:
        plot_gaussian_stats(env.gaussian_scene, ax=ax4, title="Gaussian Density (Top-Down)")

    # Panel 5: Difficulty features
    ax5 = fig.add_subplot(2, 3, 5)
    action = np.array([0.5, 0.5])
    _, info, _ = env.step(action)
    difficulty = info.get("difficulty_features", {})

    feature_names = list(difficulty.keys())
    feature_values = [difficulty[k] for k in feature_names]

    if feature_names:
        bars = ax5.barh(feature_names, feature_values, color="steelblue")
        ax5.set_xlabel("Value")
        ax5.set_title("Difficulty Features")
        for bar, val in zip(bars, feature_values):
            ax5.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                     f"{val:.2f}", va="center", fontsize=9)

    # Panel 6: Scene info text
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis("off")

    info_text = [
        f"Scene ID: {scene_id}",
        f"Topology: {env.config.scene_graph_config.topology_type}",
        f"Nodes: {len(env.graph.nodes) if env.graph else 0}",
        f"Objects: {len(env.graph.objects) if env.graph else 0}",
        f"Voxel Shape: {env.voxels.shape if env.voxels else 'N/A'}",
        f"Gaussians: {env.gaussian_scene.num_gaussians if env.gaussian_scene else 0}",
        "",
        "Difficulty Features:",
    ]
    for k, v in difficulty.items():
        info_text.append(f"  {k}: {v:.3f}")

    ax6.text(0.1, 0.9, "\n".join(info_text), transform=ax6.transAxes,
             fontsize=10, verticalalignment="top", fontfamily="monospace")

    plt.tight_layout()

    # Save figure
    fig_path = output_path / f"scene_{scene_id[:8]}_debug.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    results["files"].append(str(fig_path))
    print(f"Saved: {fig_path}")

    # Save scene graph only
    fig2, ax = plt.subplots(1, 1, figsize=(12, 10))
    if env.graph:
        plot_scene_graph_2d(env.graph, ax=ax, title=f"Scene Graph: {scene_id}")
    graph_path = output_path / f"scene_{scene_id[:8]}_graph.png"
    plt.savefig(graph_path, dpi=150, bbox_inches="tight")
    results["files"].append(str(graph_path))
    plt.close(fig2)
    print(f"Saved: {graph_path}")

    # Save metadata
    metadata = {
        "scene_id": scene_id,
        "config": env.config.to_dict(),
        "difficulty_features": difficulty,
        "num_nodes": len(env.graph.nodes) if env.graph else 0,
        "num_objects": len(env.graph.objects) if env.graph else 0,
        "voxel_shape": list(env.voxels.shape) if env.voxels else None,
        "num_gaussians": env.gaussian_scene.num_gaussians if env.gaussian_scene else 0,
    }
    metadata_path = output_path / f"scene_{scene_id[:8]}_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    results["files"].append(str(metadata_path))
    print(f"Saved: {metadata_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Debug visualization for LSD Vector Scene"
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Path to scene config YAML",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/viz",
        help="Output directory for images",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't display plots (headless mode)",
    )

    args = parser.parse_args()

    try:
        results = create_debug_visualization(
            config_path=args.config_path,
            output_dir=args.output_dir,
            show=not args.no_show,
            seed=args.seed,
        )

        print()
        print("=" * 40)
        print("Visualization Complete")
        print("=" * 40)
        print(f"Scene ID: {results['scene_id']}")
        print(f"Output: {results['output_dir']}")
        print(f"Files: {len(results['files'])}")
        for f in results["files"]:
            print(f"  - {f}")

        return 0

    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
