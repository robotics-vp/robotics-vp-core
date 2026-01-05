"""
Parametric part primitives for workcell scenes.
"""
from __future__ import annotations

import random
from typing import Dict, Optional, Tuple

from src.envs.workcell_env.scene.scene_spec import PartSpec

PartDimensions = Tuple[float, float, float]

PART_LIBRARY: Dict[str, Dict[str, object]] = {
    "bolt": {
        "dimensions_mm": (8.0, 8.0, 40.0),
        "mass_kg": 0.02,
        "material": "steel",
        "variation": 0.25,
    },
    "plate": {
        "dimensions_mm": (80.0, 60.0, 6.0),
        "mass_kg": 0.15,
        "material": "aluminum",
        "variation": 0.2,
    },
    "bracket": {
        "dimensions_mm": (60.0, 40.0, 30.0),
        "mass_kg": 0.25,
        "material": "steel",
        "variation": 0.2,
    },
    "housing": {
        "dimensions_mm": (120.0, 90.0, 60.0),
        "mass_kg": 0.8,
        "material": "polymer",
        "variation": 0.15,
    },
    "peg": {
        "dimensions_mm": (12.0, 12.0, 60.0),
        "mass_kg": 0.05,
        "material": "steel",
        "variation": 0.2,
    },
    "gear": {
        "dimensions_mm": (50.0, 50.0, 12.0),
        "mass_kg": 0.2,
        "material": "steel",
        "variation": 0.15,
    },
    "housing_cap": {
        "dimensions_mm": (90.0, 70.0, 20.0),
        "mass_kg": 0.3,
        "material": "polymer",
        "variation": 0.2,
    },
}


def sample_part_dimensions_mm(
    part_type: str, rng: Optional[random.Random] = None
) -> PartDimensions:
    """Sample dimensions for a part type with variation."""
    rng = rng or random.Random()
    entry = PART_LIBRARY.get(part_type, PART_LIBRARY["bolt"])
    base = entry["dimensions_mm"]
    variation = float(entry.get("variation", 0.15))
    return tuple(dim * (1.0 + rng.uniform(-variation, variation)) for dim in base)


def generate_part_spec(
    *,
    part_id: str,
    part_type: Optional[str] = None,
    position: Tuple[float, float, float],
    orientation: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    rng: Optional[random.Random] = None,
) -> PartSpec:
    """Generate a part spec with randomized dimensions."""
    rng = rng or random.Random()
    if part_type is None:
        part_type = rng.choice(list(PART_LIBRARY.keys()))
    entry = PART_LIBRARY.get(part_type, PART_LIBRARY["bolt"])
    base_dims = entry["dimensions_mm"]
    dims = sample_part_dimensions_mm(part_type, rng)
    volume_ratio = (dims[0] * dims[1] * dims[2]) / max(base_dims[0] * base_dims[1] * base_dims[2], 1e-6)
    mass = float(entry.get("mass_kg", 0.1)) * volume_ratio
    material = str(entry.get("material", "steel"))

    return PartSpec(
        id=part_id,
        position=position,
        orientation=orientation,
        part_type=str(part_type),
        mass_kg=float(mass),
        dimensions_mm=(float(dims[0]), float(dims[1]), float(dims[2])),
        material=material,
    )


__all__ = [
    "PART_LIBRARY",
    "sample_part_dimensions_mm",
    "generate_part_spec",
]
