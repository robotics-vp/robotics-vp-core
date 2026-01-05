"""
Parametric fixture definitions for workcell scenes.
"""
from __future__ import annotations

import random
from typing import Dict, Optional, Sequence, Tuple

from src.envs.workcell_env.scene.scene_spec import ConveyorSpec, FixtureSpec

FixtureDimensions = Tuple[float, float, float]

DEFAULT_FIXTURE_DIMENSIONS_MM: Dict[str, FixtureDimensions] = {
    "vise": (220.0, 160.0, 120.0),
    "jig": (350.0, 260.0, 140.0),
    "pallet": (600.0, 600.0, 120.0),
}

FIXTURE_VARIATION_FRACTION: Dict[str, float] = {
    "vise": 0.15,
    "jig": 0.2,
    "pallet": 0.1,
}

CLAMP_FORCE_RANGE_N: Dict[str, Tuple[float, float]] = {
    "vise": (120.0, 320.0),
    "jig": (80.0, 220.0),
    "pallet": (50.0, 160.0),
}

DEFAULT_CONVEYOR_DIMENSIONS_M: Tuple[float, float] = (1.6, 0.5)
CONVEYOR_DIMENSION_VARIATION: float = 0.2
CONVEYOR_SPEED_RANGE_M_S: Tuple[float, float] = (0.1, 0.4)


def sample_fixture_dimensions_mm(
    fixture_type: str, rng: Optional[random.Random] = None
) -> FixtureDimensions:
    """Sample fixture dimensions with variation."""
    rng = rng or random.Random()
    base = DEFAULT_FIXTURE_DIMENSIONS_MM.get(fixture_type, (300.0, 200.0, 120.0))
    variation = FIXTURE_VARIATION_FRACTION.get(fixture_type, 0.1)
    return tuple(dim * (1.0 + rng.uniform(-variation, variation)) for dim in base)


def sample_clamp_force_n(
    fixture_type: str, rng: Optional[random.Random] = None
) -> float:
    """Sample clamp force for a fixture type."""
    rng = rng or random.Random()
    force_range = CLAMP_FORCE_RANGE_N.get(fixture_type, (80.0, 200.0))
    return rng.uniform(force_range[0], force_range[1])


def generate_fixture_spec(
    *,
    fixture_type: str,
    fixture_id: str,
    position: Tuple[float, float, float],
    orientation: Tuple[float, float, float, float],
    rng: Optional[random.Random] = None,
    compatible_part_types: Optional[Sequence[str]] = None,
) -> FixtureSpec:
    """Generate a fixture spec for the requested fixture type."""
    clamp_force = sample_clamp_force_n(fixture_type, rng)
    compatible = tuple(compatible_part_types or ())
    return FixtureSpec(
        id=fixture_id,
        position=position,
        orientation=orientation,
        fixture_type=fixture_type,
        clamp_force_n=clamp_force,
        compatible_part_types=compatible,
    )


def generate_vise_fixture_spec(
    *,
    fixture_id: str,
    position: Tuple[float, float, float],
    orientation: Tuple[float, float, float, float],
    rng: Optional[random.Random] = None,
    compatible_part_types: Optional[Sequence[str]] = None,
) -> FixtureSpec:
    """Generate a vise fixture spec."""
    return generate_fixture_spec(
        fixture_type="vise",
        fixture_id=fixture_id,
        position=position,
        orientation=orientation,
        rng=rng,
        compatible_part_types=compatible_part_types,
    )


def generate_jig_fixture_spec(
    *,
    fixture_id: str,
    position: Tuple[float, float, float],
    orientation: Tuple[float, float, float, float],
    rng: Optional[random.Random] = None,
    compatible_part_types: Optional[Sequence[str]] = None,
) -> FixtureSpec:
    """Generate a jig fixture spec."""
    return generate_fixture_spec(
        fixture_type="jig",
        fixture_id=fixture_id,
        position=position,
        orientation=orientation,
        rng=rng,
        compatible_part_types=compatible_part_types,
    )


def generate_pallet_fixture_spec(
    *,
    fixture_id: str,
    position: Tuple[float, float, float],
    orientation: Tuple[float, float, float, float],
    rng: Optional[random.Random] = None,
    compatible_part_types: Optional[Sequence[str]] = None,
) -> FixtureSpec:
    """Generate a pallet fixture spec."""
    return generate_fixture_spec(
        fixture_type="pallet",
        fixture_id=fixture_id,
        position=position,
        orientation=orientation,
        rng=rng,
        compatible_part_types=compatible_part_types,
    )


def sample_conveyor_dimensions_m(rng: Optional[random.Random] = None) -> Tuple[float, float]:
    """Sample conveyor length/width dimensions in meters."""
    rng = rng or random.Random()
    base_length, base_width = DEFAULT_CONVEYOR_DIMENSIONS_M
    variation = CONVEYOR_DIMENSION_VARIATION
    length = base_length * (1.0 + rng.uniform(-variation, variation))
    width = base_width * (1.0 + rng.uniform(-variation, variation))
    return length, width


def generate_conveyor_segment_spec(
    *,
    segment_id: str,
    position: Tuple[float, float, float],
    orientation: Tuple[float, float, float, float],
    rng: Optional[random.Random] = None,
    length_m: Optional[float] = None,
    width_m: Optional[float] = None,
    speed_m_s: Optional[float] = None,
    bidirectional: bool = False,
) -> ConveyorSpec:
    """Generate a conveyor segment spec."""
    rng = rng or random.Random()
    sampled_length, sampled_width = sample_conveyor_dimensions_m(rng)
    length = float(length_m) if length_m is not None else float(sampled_length)
    width = float(width_m) if width_m is not None else float(sampled_width)
    speed = float(speed_m_s) if speed_m_s is not None else rng.uniform(*CONVEYOR_SPEED_RANGE_M_S)
    return ConveyorSpec(
        id=segment_id,
        position=position,
        orientation=orientation,
        length_m=float(length),
        speed_m_s=float(speed),
        width_m=float(width),
        bidirectional=bool(bidirectional),
    )


__all__ = [
    "sample_fixture_dimensions_mm",
    "sample_clamp_force_n",
    "generate_fixture_spec",
    "generate_vise_fixture_spec",
    "generate_jig_fixture_spec",
    "generate_pallet_fixture_spec",
    "sample_conveyor_dimensions_m",
    "generate_conveyor_segment_spec",
]
