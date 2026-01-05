"""
Procedural scene generator for workcell environments.
"""
from __future__ import annotations

import random
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from src.envs.workcell_env.config import WorkcellEnvConfig
from src.envs.workcell_env.scene.fixtures import (
    generate_conveyor_segment_spec,
    generate_fixture_spec,
    sample_conveyor_dimensions_m,
    sample_fixture_dimensions_mm,
)
from src.envs.workcell_env.scene.parts_library import generate_part_spec, sample_part_dimensions_mm
from src.envs.workcell_env.scene.scene_spec import (
    ContainerSpec,
    ConveyorSpec,
    FixtureSpec,
    PartSpec,
    StationSpec,
    WorkcellSceneSpec,
)


@dataclass
class _Placement:
    position: Tuple[float, float]
    radius_m: float


class WorkcellSceneGenerator:
    """
    Procedural generator for workcell scene specs.
    """

    def __init__(self, *, padding_m: float = 0.05, max_attempts: int = 200) -> None:
        self.padding_m = float(padding_m)
        self.max_attempts = int(max_attempts)

    def generate(self, config: WorkcellEnvConfig, seed: Optional[int] = None) -> WorkcellSceneSpec:
        """Generate a WorkcellSceneSpec based on the env config."""
        rng = random.Random(seed)
        bounds = self._bounds_for_topology(config.topology_type)

        placements: List[_Placement] = []

        stations = self._generate_stations(config, rng, bounds, placements)
        fixtures = self._generate_fixtures(config, rng, bounds, placements, stations)
        conveyors = self._generate_conveyors(config, rng, bounds, placements)
        containers, container_regions = self._generate_containers(config, rng, bounds, placements)
        parts = self._generate_parts(
            config,
            rng,
            bounds,
            container_regions,
            conveyors,
        )

        # Use rng for deterministic ID generation
        workcell_id = f"workcell_{rng.getrandbits(40):010x}"

        return WorkcellSceneSpec(
            workcell_id=workcell_id,
            stations=stations,
            fixtures=fixtures,
            parts=parts,
            tools=[],
            conveyors=conveyors,
            containers=containers,
            spatial_bounds=bounds,
        )

    def _bounds_for_topology(self, topology_type: str) -> Tuple[float, float, float]:
        defaults = {
            "ASSEMBLY_BENCH": (5.0, 4.0, 3.0),
            "CONVEYOR_LINE": (8.0, 4.0, 3.0),
            "INSPECTION_STATION": (4.0, 3.0, 3.0),
            "TOOL_CABINET": (4.5, 4.0, 3.0),
            "MIXED_WORKCELL": (6.0, 4.5, 3.0),
        }
        return defaults.get(topology_type, (5.0, 4.0, 3.0))

    def _generate_stations(
        self,
        config: WorkcellEnvConfig,
        rng: random.Random,
        bounds: Tuple[float, float, float],
        placements: List[_Placement],
    ) -> List[StationSpec]:
        stations: List[StationSpec] = []
        anchors = self._station_anchors(config.num_stations, config.topology_type, bounds)
        for idx, anchor in enumerate(anchors):
            position = self._place_near(anchor, rng, bounds, placements, radius_m=0.5, jitter_m=(0.2, 0.2))
            station_type, capabilities = self._station_profile(config.topology_type)
            stations.append(
                StationSpec(
                    id=f"station_{idx}",
                    position=(position[0], position[1], 0.0),
                    orientation=(1.0, 0.0, 0.0, 0.0),
                    station_type=station_type,
                    capabilities=capabilities,
                    payload_limit_kg=5.0,
                )
            )
        return stations

    def _station_profile(self, topology_type: str) -> Tuple[str, Tuple[str, ...]]:
        if topology_type == "INSPECTION_STATION":
            return "inspection", ("PICK", "PLACE", "INSPECT")
        if topology_type == "CONVEYOR_LINE":
            return "sorting", ("PICK", "PLACE", "ROUTE")
        if topology_type == "TOOL_CABINET":
            return "tooling", ("PICK", "PLACE")
        return "assembly", ("PICK", "PLACE", "FASTEN")

    def _station_anchors(
        self,
        num_stations: int,
        topology_type: str,
        bounds: Tuple[float, float, float],
    ) -> List[Tuple[float, float]]:
        if num_stations <= 0:
            return []
        y_offset = 0.6
        if topology_type == "CONVEYOR_LINE":
            y_offset = 0.9
        elif topology_type == "INSPECTION_STATION":
            y_offset = 0.0
        elif topology_type == "TOOL_CABINET":
            y_offset = 0.7

        span = bounds[0] * 0.7
        if num_stations == 1:
            xs = [0.0]
        else:
            start = -span / 2.0
            step = span / float(num_stations - 1)
            xs = [start + step * idx for idx in range(num_stations)]
        return [(x, y_offset) for x in xs]

    def _generate_fixtures(
        self,
        config: WorkcellEnvConfig,
        rng: random.Random,
        bounds: Tuple[float, float, float],
        placements: List[_Placement],
        stations: Sequence[StationSpec],
    ) -> List[FixtureSpec]:
        fixtures: List[FixtureSpec] = []
        fixture_types = self._fixture_types_for_topology(config.topology_type)
        for idx in range(config.num_fixtures):
            fixture_type = rng.choice(fixture_types)
            anchor = self._fixture_anchor_for_station(stations, idx)
            dims_mm = sample_fixture_dimensions_mm(fixture_type, rng)
            radius_m = max(dims_mm[0], dims_mm[1]) / 2000.0
            position = self._place_near(anchor, rng, bounds, placements, radius_m=radius_m, jitter_m=(0.3, 0.3))
            fixtures.append(
                generate_fixture_spec(
                    fixture_type=fixture_type,
                    fixture_id=f"fixture_{idx}",
                    position=(position[0], position[1], 0.0),
                    orientation=(1.0, 0.0, 0.0, 0.0),
                    rng=rng,
                    compatible_part_types=config.part_types,
                )
            )
        return fixtures

    def _fixture_types_for_topology(self, topology_type: str) -> List[str]:
        if topology_type == "INSPECTION_STATION":
            return ["jig"]
        if topology_type == "CONVEYOR_LINE":
            return ["pallet", "jig"]
        if topology_type == "TOOL_CABINET":
            return ["vise", "pallet"]
        return ["vise", "jig", "pallet"]

    def _fixture_anchor_for_station(
        self, stations: Sequence[StationSpec], idx: int
    ) -> Tuple[float, float]:
        if not stations:
            return 0.0, 0.0
        station = stations[idx % len(stations)]
        return station.position[0], station.position[1] - 0.4

    def _generate_conveyors(
        self,
        config: WorkcellEnvConfig,
        rng: random.Random,
        bounds: Tuple[float, float, float],
        placements: List[_Placement],
    ) -> List[ConveyorSpec]:
        conveyors: List[ConveyorSpec] = []
        if not config.conveyor_enabled and config.topology_type != "CONVEYOR_LINE":
            return conveyors

        num_segments = 1
        base_length, base_width = sample_conveyor_dimensions_m(rng)
        length_m = min(base_length, bounds[0] * 0.8)
        width_m = min(base_width, bounds[1] * 0.4)
        radius_m = ((length_m / 2.0) ** 2 + (width_m / 2.0) ** 2) ** 0.5
        for idx in range(num_segments):
            x_offset = (idx - (num_segments - 1) / 2.0) * (length_m + 0.3)
            position = self._place_near(
                (x_offset, 0.0),
                rng,
                bounds,
                placements,
                radius_m=radius_m,
                jitter_m=(0.1, 0.1),
            )
            conveyors.append(
                generate_conveyor_segment_spec(
                    segment_id=f"conveyor_{idx}",
                    position=(position[0], position[1], 0.0),
                    orientation=(1.0, 0.0, 0.0, 0.0),
                    rng=rng,
                    length_m=length_m,
                    width_m=width_m,
                    speed_m_s=rng.uniform(0.1, 0.4),
                )
            )
        return conveyors

    def _generate_containers(
        self,
        config: WorkcellEnvConfig,
        rng: random.Random,
        bounds: Tuple[float, float, float],
        placements: List[_Placement],
    ) -> Tuple[List[ContainerSpec], List[Dict[str, float]]]:
        containers: List[ContainerSpec] = []
        regions: List[Dict[str, float]] = []
        anchors = self._container_anchors(config.num_bins, config.topology_type, bounds)
        for idx, anchor in enumerate(anchors):
            slot_size_mm = (
                rng.uniform(250.0, 350.0),
                rng.uniform(180.0, 260.0),
                rng.uniform(120.0, 180.0),
            )
            radius_m = max(slot_size_mm[0], slot_size_mm[1]) / 2000.0
            position = self._place_near(anchor, rng, bounds, placements, radius_m=radius_m, jitter_m=(0.2, 0.2))
            containers.append(
                ContainerSpec(
                    id=f"bin_{idx}",
                    position=(position[0], position[1], 0.0),
                    orientation=(1.0, 0.0, 0.0, 0.0),
                    container_type="bin",
                    capacity=10,
                    slot_size_mm=slot_size_mm,
                )
            )
            regions.append(
                {
                    "cx": position[0],
                    "cy": position[1],
                    "half_x": slot_size_mm[0] / 2000.0,
                    "half_y": slot_size_mm[1] / 2000.0,
                    "z": 0.0,
                }
            )
        return containers, regions

    def _container_anchors(
        self,
        num_bins: int,
        topology_type: str,
        bounds: Tuple[float, float, float],
    ) -> List[Tuple[float, float]]:
        if num_bins <= 0:
            return []
        y_offset = -0.8
        if topology_type == "CONVEYOR_LINE":
            y_offset = -1.0
        elif topology_type == "INSPECTION_STATION":
            y_offset = -0.6
        elif topology_type == "TOOL_CABINET":
            y_offset = -1.0

        span = bounds[0] * 0.8
        if num_bins == 1:
            xs = [0.0]
        else:
            start = -span / 2.0
            step = span / float(num_bins - 1)
            xs = [start + step * idx for idx in range(num_bins)]
        return [(x, y_offset) for x in xs]

    def _generate_parts(
        self,
        config: WorkcellEnvConfig,
        rng: random.Random,
        bounds: Tuple[float, float, float],
        container_regions: List[Dict[str, float]],
        conveyors: Sequence[ConveyorSpec],
    ) -> List[PartSpec]:
        parts: List[PartSpec] = []
        part_positions: List[_Placement] = []

        conveyor_regions = []
        for conveyor in conveyors:
            conveyor_regions.append(
                {
                    "cx": conveyor.position[0],
                    "cy": conveyor.position[1],
                    "half_x": conveyor.length_m / 2.0,
                    "half_y": conveyor.width_m / 2.0,
                    "z": conveyor.position[2],
                }
            )

        num_conveyor_parts = min(len(conveyor_regions) * 3, config.num_parts // 3) if conveyor_regions else 0

        for idx in range(config.num_parts):
            part_type = rng.choice(list(config.part_types)) if config.part_types else None
            dims_mm = sample_part_dimensions_mm(part_type or "bolt", rng)
            radius_m = max(dims_mm[0], dims_mm[1]) / 2000.0

            if idx < num_conveyor_parts and conveyor_regions:
                region = conveyor_regions[idx % len(conveyor_regions)]
            elif container_regions:
                region = container_regions[idx % len(container_regions)]
            else:
                region = {"cx": 0.0, "cy": 0.0, "half_x": bounds[0] / 3.0, "half_y": bounds[1] / 3.0, "z": 0.0}

            position = self._place_in_region(region, rng, bounds, part_positions, radius_m)
            parts.append(
                generate_part_spec(
                    part_id=f"part_{idx}",
                    part_type=part_type,
                    position=(position[0], position[1], position[2]),
                    orientation=(1.0, 0.0, 0.0, 0.0),
                    rng=rng,
                )
            )
        return parts

    def _place_in_region(
        self,
        region: Dict[str, float],
        rng: random.Random,
        bounds: Tuple[float, float, float],
        part_positions: List[_Placement],
        radius_m: float,
    ) -> Tuple[float, float, float]:
        for _ in range(self.max_attempts):
            x = region["cx"] + rng.uniform(-region["half_x"], region["half_x"])
            y = region["cy"] + rng.uniform(-region["half_y"], region["half_y"])
            x = self._clamp(x, bounds[0] / 2.0 - radius_m)
            y = self._clamp(y, bounds[1] / 2.0 - radius_m)
            if self._is_collision_free(x, y, radius_m, part_positions):
                part_positions.append(_Placement((x, y), radius_m))
                return x, y, region.get("z", 0.0) + radius_m

        scanned = self._scan_for_free_position(
            x_min=region["cx"] - region["half_x"],
            x_max=region["cx"] + region["half_x"],
            y_min=region["cy"] - region["half_y"],
            y_max=region["cy"] + region["half_y"],
            bounds=bounds,
            placements=part_positions,
            radius_m=radius_m,
        )
        if scanned is not None:
            part_positions.append(_Placement(scanned, radius_m))
            return scanned[0], scanned[1], region.get("z", 0.0) + radius_m

        fallback = (region["cx"], region["cy"], region.get("z", 0.0) + radius_m)
        part_positions.append(_Placement((fallback[0], fallback[1]), radius_m))
        return fallback

    def _place_near(
        self,
        anchor: Tuple[float, float],
        rng: random.Random,
        bounds: Tuple[float, float, float],
        placements: List[_Placement],
        radius_m: float,
        jitter_m: Tuple[float, float],
    ) -> Tuple[float, float]:
        for _ in range(self.max_attempts):
            x = anchor[0] + rng.uniform(-jitter_m[0], jitter_m[0])
            y = anchor[1] + rng.uniform(-jitter_m[1], jitter_m[1])
            if not self._within_bounds(x, y, radius_m, bounds):
                continue
            if self._is_collision_free(x, y, radius_m, placements):
                placements.append(_Placement((x, y), radius_m))
                return x, y

        scanned = self._scan_for_free_position(
            x_min=-bounds[0] / 2.0 + radius_m,
            x_max=bounds[0] / 2.0 - radius_m,
            y_min=-bounds[1] / 2.0 + radius_m,
            y_max=bounds[1] / 2.0 - radius_m,
            bounds=bounds,
            placements=placements,
            radius_m=radius_m,
        )
        if scanned is not None:
            placements.append(_Placement(scanned, radius_m))
            return scanned[0], scanned[1]

        x = self._clamp(anchor[0], bounds[0] / 2.0 - radius_m)
        y = self._clamp(anchor[1], bounds[1] / 2.0 - radius_m)
        placements.append(_Placement((x, y), radius_m))
        return x, y

    def _within_bounds(
        self, x: float, y: float, radius_m: float, bounds: Tuple[float, float, float]
    ) -> bool:
        return abs(x) + radius_m <= bounds[0] / 2.0 and abs(y) + radius_m <= bounds[1] / 2.0

    def _is_collision_free(
        self, x: float, y: float, radius_m: float, placements: Sequence[_Placement]
    ) -> bool:
        for placement in placements:
            dx = x - placement.position[0]
            dy = y - placement.position[1]
            min_dist = radius_m + placement.radius_m + self.padding_m
            if dx * dx + dy * dy < min_dist * min_dist:
                return False
        return True

    def _clamp(self, value: float, limit: float) -> float:
        return max(min(value, limit), -limit)

    def _scan_for_free_position(
        self,
        *,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        bounds: Tuple[float, float, float],
        placements: Sequence[_Placement],
        radius_m: float,
    ) -> Optional[Tuple[float, float]]:
        step = max(radius_m * 2.0 + self.padding_m, 1e-3)
        x = x_min
        while x <= x_max:
            y = y_min
            while y <= y_max:
                if self._within_bounds(x, y, radius_m, bounds) and self._is_collision_free(
                    x, y, radius_m, placements
                ):
                    return x, y
                y += step
            x += step
        return None


__all__ = ["WorkcellSceneGenerator"]
