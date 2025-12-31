"""
Configuration schema for LSD Vector Scene environments.

Provides a Pydantic-compatible frozen dataclass for configuring
vector scene generation, visual style, and behaviour simulation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional
import json


@dataclass(frozen=True)
class LSDVectorSceneConfig:
    """
    Unified configuration for LSD Vector Scene environments.

    This config covers:
    - Scene topology (graph structure)
    - Visual style (lighting, clutter, materials)
    - Dynamic agents and behaviour
    - Randomization parameters
    """

    # Scene topology
    topology_type: Literal[
        "WAREHOUSE_AISLES",
        "KITCHEN_LAYOUT",
        "RESIDENTIAL_GARAGE",
        "OPEN_FLOOR",
        "OFFICE_CUBICLES",
    ] = "WAREHOUSE_AISLES"
    num_nodes: int = 10
    num_objects: int = 15
    density: float = 0.5  # Graph density (edges / possible edges)
    route_length: float = 30.0  # Typical path length in meters

    # Visual style
    lighting: Literal["BRIGHT_INDOOR", "DIM_INDOOR", "NIGHT", "MIXED"] = "BRIGHT_INDOOR"
    clutter_level: Literal["LOW", "MEDIUM", "HIGH"] = "MEDIUM"
    material_mix: tuple[str, ...] = ("metal", "plastic", "wood")
    voxel_size: float = 0.1  # Meters per voxel
    default_height: float = 3.0  # Meters

    # Dynamic agents
    num_humans: int = 2
    num_robots: int = 1
    num_forklifts: int = 0
    tilt: float = -1.0  # Behaviour adversarialness: -1 cooperative, +1 adversarial

    # Behaviour model
    use_trained_behaviour: bool = False
    behaviour_checkpoint: Optional[str] = None
    use_simple_policy: bool = True

    # Randomization
    random_seed: int = 0

    # Episode limits (will merge with EconParams)
    max_steps: int = 500

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "topology_type": self.topology_type,
            "num_nodes": self.num_nodes,
            "num_objects": self.num_objects,
            "density": self.density,
            "route_length": self.route_length,
            "lighting": self.lighting,
            "clutter_level": self.clutter_level,
            "material_mix": list(self.material_mix),
            "voxel_size": self.voxel_size,
            "default_height": self.default_height,
            "num_humans": self.num_humans,
            "num_robots": self.num_robots,
            "num_forklifts": self.num_forklifts,
            "tilt": self.tilt,
            "use_trained_behaviour": self.use_trained_behaviour,
            "behaviour_checkpoint": self.behaviour_checkpoint,
            "use_simple_policy": self.use_simple_policy,
            "random_seed": self.random_seed,
            "max_steps": self.max_steps,
        }

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LSDVectorSceneConfig":
        """Deserialize from dictionary."""
        # Handle material_mix as list or tuple
        material_mix = data.get("material_mix", ("metal", "plastic", "wood"))
        if isinstance(material_mix, list):
            material_mix = tuple(material_mix)

        return cls(
            topology_type=data.get("topology_type", "WAREHOUSE_AISLES"),
            num_nodes=data.get("num_nodes", 10),
            num_objects=data.get("num_objects", 15),
            density=data.get("density", 0.5),
            route_length=data.get("route_length", 30.0),
            lighting=data.get("lighting", "BRIGHT_INDOOR"),
            clutter_level=data.get("clutter_level", "MEDIUM"),
            material_mix=material_mix,
            voxel_size=data.get("voxel_size", 0.1),
            default_height=data.get("default_height", 3.0),
            num_humans=data.get("num_humans", 2),
            num_robots=data.get("num_robots", 1),
            num_forklifts=data.get("num_forklifts", 0),
            tilt=data.get("tilt", -1.0),
            use_trained_behaviour=data.get("use_trained_behaviour", False),
            behaviour_checkpoint=data.get("behaviour_checkpoint"),
            use_simple_policy=data.get("use_simple_policy", True),
            random_seed=data.get("random_seed", 0),
            max_steps=data.get("max_steps", 500),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "LSDVectorSceneConfig":
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def from_yaml_path(cls, path: str) -> "LSDVectorSceneConfig":
        """Load from YAML file."""
        import yaml
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data.get("lsd_vector_scene", data))


# Presets for common scenarios
PRESETS: Dict[str, LSDVectorSceneConfig] = {
    "warehouse_easy": LSDVectorSceneConfig(
        topology_type="WAREHOUSE_AISLES",
        num_nodes=5,
        num_objects=8,
        density=0.3,
        route_length=20.0,
        lighting="BRIGHT_INDOOR",
        clutter_level="LOW",
        num_humans=1,
        num_robots=0,
        num_forklifts=0,
        tilt=-1.0,
    ),
    "warehouse_medium": LSDVectorSceneConfig(
        topology_type="WAREHOUSE_AISLES",
        num_nodes=10,
        num_objects=15,
        density=0.5,
        route_length=30.0,
        lighting="DIM_INDOOR",
        clutter_level="MEDIUM",
        num_humans=2,
        num_robots=1,
        num_forklifts=1,
        tilt=0.0,
    ),
    "warehouse_hard": LSDVectorSceneConfig(
        topology_type="WAREHOUSE_AISLES",
        num_nodes=20,
        num_objects=30,
        density=0.7,
        route_length=50.0,
        lighting="NIGHT",
        clutter_level="HIGH",
        num_humans=4,
        num_robots=2,
        num_forklifts=2,
        tilt=0.5,
    ),
    "kitchen_simple": LSDVectorSceneConfig(
        topology_type="KITCHEN_LAYOUT",
        num_nodes=4,
        num_objects=10,
        density=0.6,
        route_length=8.0,
        lighting="BRIGHT_INDOOR",
        clutter_level="MEDIUM",
        num_humans=1,
        num_robots=0,
        num_forklifts=0,
        tilt=-0.5,
    ),
    "garage_cluttered": LSDVectorSceneConfig(
        topology_type="RESIDENTIAL_GARAGE",
        num_nodes=6,
        num_objects=20,
        density=0.8,
        route_length=10.0,
        lighting="DIM_INDOOR",
        clutter_level="HIGH",
        num_humans=0,
        num_robots=0,
        num_forklifts=0,
        tilt=0.0,
    ),
}


def load_lsd_vector_scene_config(
    data: Optional[Dict[str, Any]] = None,
    preset: Optional[str] = None,
) -> LSDVectorSceneConfig:
    """
    Load LSD Vector Scene config from dict and/or preset.

    Args:
        data: Optional dict with config values (overrides preset)
        preset: Optional preset name ("warehouse_easy", "warehouse_medium", etc.)

    Returns:
        LSDVectorSceneConfig instance
    """
    if preset and preset in PRESETS:
        base = PRESETS[preset]
        if not data:
            return base
        # Merge data onto preset
        merged = base.to_dict()
        merged.update(data)
        return LSDVectorSceneConfig.from_dict(merged)

    if data:
        return LSDVectorSceneConfig.from_dict(data)

    return LSDVectorSceneConfig()
