"""
Configuration for Map-First Pseudo-Supervision.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class MapFirstSupervisionConfig:
    """Configuration for Map-First pseudo-supervision.

    Attributes:
        voxel_size: Voxel grid size in world units.
        map_max_points_per_voxel: Max accumulated samples per voxel.
        static_update_policy: Policy for updating static map.
        residual_method: Method for residual computation.
        dynamic_threshold: Threshold for dynamic evidence masking.
        dynamic_use_zscore: If True, use z-scored residuals for evidence.
        densify_enabled: Whether densification is enabled.
        densify_mode: "depth_map" or "world_points".
        occlusion_culling: Culling strategy for depth targets.
        semantics_enabled: Whether semantic stabilization is enabled.
        semantics_num_classes: Number of semantic classes if enabled.
        export_float16: Use float16 for large arrays in artifacts.
    """

    voxel_size: float = 0.25
    map_max_points_per_voxel: int = 50
    static_update_policy: Literal["visible_only", "visible_and_low_residual"] = "visible_only"
    residual_method: Literal["voxel_centroid", "knn"] = "voxel_centroid"
    dynamic_threshold: float = 0.3
    dynamic_use_zscore: bool = False
    densify_enabled: bool = True
    densify_mode: Literal["depth_map", "world_points"] = "depth_map"
    occlusion_culling: Literal["spherical_bins", "zbuffer"] = "spherical_bins"
    semantics_enabled: bool = False
    semantics_num_classes: int = 0
    export_float16: bool = True
