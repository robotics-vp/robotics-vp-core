"""
Configuration schema for manufacturing workcell environments.

Provides a frozen dataclass with serialization helpers and presets.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional


@dataclass(frozen=True)
class WorkcellEnvConfig:
    """
    Configuration for workcell environments.

    Covers topology, workcell composition, part mix, episode limits,
    physics mode, and difficulty parameters.
    """

    topology_type: Literal[
        "ASSEMBLY_BENCH",
        "CONVEYOR_LINE",
        "INSPECTION_STATION",
        "TOOL_CABINET",
        "MIXED_WORKCELL",
    ] = "ASSEMBLY_BENCH"

    # Workcell parameters
    num_stations: int = 2
    num_fixtures: int = 2
    num_bins: int = 4
    conveyor_enabled: bool = False

    # Parts parameters
    num_parts: int = 12
    part_types: tuple[str, ...] = ("bolt", "plate")

    # Episode limits
    max_steps: int = 200
    time_step_s: float = 1.0

    # Physics mode
    physics_mode: Literal["SIMPLE", "MUJOCO", "ISAAC"] = "SIMPLE"

    # Optional SceneTracks wiring
    enable_scene_tracks: bool = False

    # Rendering for datapack frames
    capture_rgb_frames: bool = False
    render_width: int = 128
    render_height: int = 128
    render_fps: int = 10
    render_max_frames: int = 50

    # Sensor bundle export
    capture_sensor_bundle: bool = False
    sensor_cameras: tuple[str, ...] = ("front",)
    sensor_depth_unit: Literal["meters"] = "meters"
    sensor_noise: Dict[str, Any] = field(default_factory=dict)
    sensor_noise_seed: Optional[int] = None

    # Difficulty parameters
    tolerance_mm: float = 2.0
    occlusion_level: float = 0.1
    tool_changes_required: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "topology_type": self.topology_type,
            "num_stations": self.num_stations,
            "num_fixtures": self.num_fixtures,
            "num_bins": self.num_bins,
            "conveyor_enabled": self.conveyor_enabled,
            "num_parts": self.num_parts,
            "part_types": list(self.part_types),
            "max_steps": self.max_steps,
            "time_step_s": self.time_step_s,
            "physics_mode": self.physics_mode,
            "enable_scene_tracks": self.enable_scene_tracks,
            "capture_rgb_frames": self.capture_rgb_frames,
            "render_width": self.render_width,
            "render_height": self.render_height,
            "render_fps": self.render_fps,
            "render_max_frames": self.render_max_frames,
            "capture_sensor_bundle": self.capture_sensor_bundle,
            "sensor_cameras": list(self.sensor_cameras),
            "sensor_depth_unit": self.sensor_depth_unit,
            "sensor_noise": dict(self.sensor_noise),
            "sensor_noise_seed": self.sensor_noise_seed,
            "tolerance_mm": self.tolerance_mm,
            "occlusion_level": self.occlusion_level,
            "tool_changes_required": self.tool_changes_required,
        }

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkcellEnvConfig":
        """Deserialize from dictionary."""
        part_types = data.get("part_types", ("bolt", "plate"))
        if isinstance(part_types, list):
            part_types = tuple(part_types)

        return cls(
            topology_type=data.get("topology_type", "ASSEMBLY_BENCH"),
            num_stations=data.get("num_stations", 2),
            num_fixtures=data.get("num_fixtures", 2),
            num_bins=data.get("num_bins", 4),
            conveyor_enabled=data.get("conveyor_enabled", False),
            num_parts=data.get("num_parts", 12),
            part_types=part_types,
            max_steps=data.get("max_steps", 200),
            time_step_s=data.get("time_step_s", 1.0),
            physics_mode=data.get("physics_mode", "SIMPLE"),
            enable_scene_tracks=data.get("enable_scene_tracks", False),
            capture_rgb_frames=data.get("capture_rgb_frames", False),
            render_width=data.get("render_width", 128),
            render_height=data.get("render_height", 128),
            render_fps=data.get("render_fps", 10),
            render_max_frames=data.get("render_max_frames", 50),
            capture_sensor_bundle=data.get("capture_sensor_bundle", False),
            sensor_cameras=tuple(data.get("sensor_cameras", ("front",))),
            sensor_depth_unit=data.get("sensor_depth_unit", "meters"),
            sensor_noise=data.get("sensor_noise", {}) or {},
            sensor_noise_seed=data.get("sensor_noise_seed"),
            tolerance_mm=data.get("tolerance_mm", 2.0),
            occlusion_level=data.get("occlusion_level", 0.1),
            tool_changes_required=data.get("tool_changes_required", 0),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "WorkcellEnvConfig":
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def from_yaml_path(cls, path: str) -> "WorkcellEnvConfig":
        """Load from YAML file."""
        import yaml

        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data.get("workcell_env", data))


# Presets for common scenarios
PRESETS: Dict[str, WorkcellEnvConfig] = {
    "assembly_bench_simple": WorkcellEnvConfig(
        topology_type="ASSEMBLY_BENCH",
        num_stations=1,
        num_fixtures=1,
        num_bins=2,
        conveyor_enabled=False,
        num_parts=6,
        part_types=("bolt", "plate"),
        max_steps=120,
        time_step_s=1.0,
        physics_mode="SIMPLE",
        tolerance_mm=2.0,
        occlusion_level=0.1,
        tool_changes_required=0,
    ),
    "conveyor_sorting": WorkcellEnvConfig(
        topology_type="CONVEYOR_LINE",
        num_stations=2,
        num_fixtures=1,
        num_bins=4,
        conveyor_enabled=True,
        num_parts=20,
        part_types=("widget_a", "widget_b", "widget_c"),
        max_steps=200,
        time_step_s=0.5,
        physics_mode="SIMPLE",
        tolerance_mm=3.0,
        occlusion_level=0.2,
        tool_changes_required=0,
    ),
    "inspection_simple": WorkcellEnvConfig(
        topology_type="INSPECTION_STATION",
        num_stations=1,
        num_fixtures=1,
        num_bins=2,
        conveyor_enabled=False,
        num_parts=8,
        part_types=("housing", "bracket"),
        max_steps=150,
        time_step_s=1.0,
        physics_mode="SIMPLE",
        tolerance_mm=1.0,
        occlusion_level=0.05,
        tool_changes_required=1,
    ),
}
