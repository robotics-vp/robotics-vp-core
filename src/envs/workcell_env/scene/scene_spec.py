"""
Scene specifications for manufacturing workcell environments.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List


class _JsonMixin:
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str):
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))


def _as_tuple(value: Any, default: tuple) -> tuple:
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return default


def _coerce_specs(items: Any, spec_cls):
    specs = []
    for item in items or []:
        if isinstance(item, spec_cls):
            specs.append(item)
        elif isinstance(item, dict):
            specs.append(spec_cls.from_dict(item))
    return specs


@dataclass(frozen=True)
class StationSpec(_JsonMixin):
    """Specification for a workcell station."""
    id: str
    position: tuple[float, float, float]
    orientation: tuple[float, float, float, float]
    station_type: str = "assembly"
    capabilities: tuple[str, ...] = ("PICK", "PLACE")
    payload_limit_kg: float = 5.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "position": list(self.position),
            "orientation": list(self.orientation),
            "station_type": self.station_type,
            "capabilities": list(self.capabilities),
            "payload_limit_kg": self.payload_limit_kg,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StationSpec":
        """Deserialize from dictionary."""
        capabilities = data.get("capabilities", ("PICK", "PLACE"))
        if isinstance(capabilities, list):
            capabilities = tuple(capabilities)
        return cls(
            id=data["id"],
            position=_as_tuple(data.get("position"), (0.0, 0.0, 0.0)),
            orientation=_as_tuple(data.get("orientation"), (1.0, 0.0, 0.0, 0.0)),
            station_type=data.get("station_type", "assembly"),
            capabilities=capabilities,
            payload_limit_kg=data.get("payload_limit_kg", 5.0),
        )


@dataclass(frozen=True)
class FixtureSpec(_JsonMixin):
    """Specification for fixtures in the workcell."""
    id: str
    position: tuple[float, float, float]
    orientation: tuple[float, float, float, float]
    fixture_type: str = "vise"
    clamp_force_n: float = 100.0
    compatible_part_types: tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "position": list(self.position),
            "orientation": list(self.orientation),
            "fixture_type": self.fixture_type,
            "clamp_force_n": self.clamp_force_n,
            "compatible_part_types": list(self.compatible_part_types),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FixtureSpec":
        """Deserialize from dictionary."""
        compatible = data.get("compatible_part_types", ())
        if isinstance(compatible, list):
            compatible = tuple(compatible)
        return cls(
            id=data["id"],
            position=_as_tuple(data.get("position"), (0.0, 0.0, 0.0)),
            orientation=_as_tuple(data.get("orientation"), (1.0, 0.0, 0.0, 0.0)),
            fixture_type=data.get("fixture_type", "vise"),
            clamp_force_n=data.get("clamp_force_n", 100.0),
            compatible_part_types=compatible,
        )


@dataclass(frozen=True)
class PartSpec(_JsonMixin):
    """Specification for a part handled in the workcell."""
    id: str
    position: tuple[float, float, float]
    orientation: tuple[float, float, float, float]
    part_type: str = "generic_part"
    mass_kg: float = 0.25
    dimensions_mm: tuple[float, float, float] = (50.0, 50.0, 20.0)
    material: str = "steel"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "position": list(self.position),
            "orientation": list(self.orientation),
            "part_type": self.part_type,
            "mass_kg": self.mass_kg,
            "dimensions_mm": list(self.dimensions_mm),
            "material": self.material,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PartSpec":
        """Deserialize from dictionary."""
        return cls(
            id=data["id"],
            position=_as_tuple(data.get("position"), (0.0, 0.0, 0.0)),
            orientation=_as_tuple(data.get("orientation"), (1.0, 0.0, 0.0, 0.0)),
            part_type=data.get("part_type", "generic_part"),
            mass_kg=data.get("mass_kg", 0.25),
            dimensions_mm=_as_tuple(data.get("dimensions_mm"), (50.0, 50.0, 20.0)),
            material=data.get("material", "steel"),
        )


@dataclass(frozen=True)
class ToolSpec(_JsonMixin):
    """Specification for a tool used in the workcell."""
    id: str
    position: tuple[float, float, float]
    orientation: tuple[float, float, float, float]
    tool_type: str = "gripper"
    precision_mm: float = 1.0
    compatible_part_types: tuple[str, ...] = ()
    cycle_time_s: float = 2.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "position": list(self.position),
            "orientation": list(self.orientation),
            "tool_type": self.tool_type,
            "precision_mm": self.precision_mm,
            "compatible_part_types": list(self.compatible_part_types),
            "cycle_time_s": self.cycle_time_s,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolSpec":
        """Deserialize from dictionary."""
        compatible = data.get("compatible_part_types", ())
        if isinstance(compatible, list):
            compatible = tuple(compatible)
        return cls(
            id=data["id"],
            position=_as_tuple(data.get("position"), (0.0, 0.0, 0.0)),
            orientation=_as_tuple(data.get("orientation"), (1.0, 0.0, 0.0, 0.0)),
            tool_type=data.get("tool_type", "gripper"),
            precision_mm=data.get("precision_mm", 1.0),
            compatible_part_types=compatible,
            cycle_time_s=data.get("cycle_time_s", 2.0),
        )


@dataclass(frozen=True)
class ConveyorSpec(_JsonMixin):
    """Specification for conveyor segments."""
    id: str
    position: tuple[float, float, float]
    orientation: tuple[float, float, float, float]
    length_m: float = 2.0
    speed_m_s: float = 0.2
    width_m: float = 0.5
    bidirectional: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "position": list(self.position),
            "orientation": list(self.orientation),
            "length_m": self.length_m,
            "speed_m_s": self.speed_m_s,
            "width_m": self.width_m,
            "bidirectional": self.bidirectional,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConveyorSpec":
        """Deserialize from dictionary."""
        return cls(
            id=data["id"],
            position=_as_tuple(data.get("position"), (0.0, 0.0, 0.0)),
            orientation=_as_tuple(data.get("orientation"), (1.0, 0.0, 0.0, 0.0)),
            length_m=data.get("length_m", 2.0),
            speed_m_s=data.get("speed_m_s", 0.2),
            width_m=data.get("width_m", 0.5),
            bidirectional=data.get("bidirectional", False),
        )


@dataclass(frozen=True)
class ContainerSpec(_JsonMixin):
    """Specification for bins, trays, and pallets."""
    id: str
    position: tuple[float, float, float]
    orientation: tuple[float, float, float, float]
    container_type: str = "bin"
    capacity: int = 10
    slot_size_mm: tuple[float, float, float] = (100.0, 100.0, 50.0)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "position": list(self.position),
            "orientation": list(self.orientation),
            "container_type": self.container_type,
            "capacity": self.capacity,
            "slot_size_mm": list(self.slot_size_mm),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContainerSpec":
        """Deserialize from dictionary."""
        return cls(
            id=data["id"],
            position=_as_tuple(data.get("position"), (0.0, 0.0, 0.0)),
            orientation=_as_tuple(data.get("orientation"), (1.0, 0.0, 0.0, 0.0)),
            container_type=data.get("container_type", "bin"),
            capacity=data.get("capacity", 10),
            slot_size_mm=_as_tuple(data.get("slot_size_mm"), (100.0, 100.0, 50.0)),
        )


@dataclass(frozen=True)
class WorkcellSceneSpec(_JsonMixin):
    """Specification for a full workcell scene layout."""
    workcell_id: str
    stations: List[StationSpec] = field(default_factory=list)
    fixtures: List[FixtureSpec] = field(default_factory=list)
    parts: List[PartSpec] = field(default_factory=list)
    tools: List[ToolSpec] = field(default_factory=list)
    conveyors: List[ConveyorSpec] = field(default_factory=list)
    containers: List[ContainerSpec] = field(default_factory=list)
    spatial_bounds: tuple[float, float, float] = (5.0, 5.0, 3.0)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "workcell_id": self.workcell_id,
            "stations": [spec.to_dict() for spec in self.stations],
            "fixtures": [spec.to_dict() for spec in self.fixtures],
            "parts": [spec.to_dict() for spec in self.parts],
            "tools": [spec.to_dict() for spec in self.tools],
            "conveyors": [spec.to_dict() for spec in self.conveyors],
            "containers": [spec.to_dict() for spec in self.containers],
            "spatial_bounds": list(self.spatial_bounds),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkcellSceneSpec":
        """Deserialize from dictionary."""
        return cls(
            workcell_id=data["workcell_id"],
            stations=_coerce_specs(data.get("stations", []), StationSpec),
            fixtures=_coerce_specs(data.get("fixtures", []), FixtureSpec),
            parts=_coerce_specs(data.get("parts", []), PartSpec),
            tools=_coerce_specs(data.get("tools", []), ToolSpec),
            conveyors=_coerce_specs(data.get("conveyors", []), ConveyorSpec),
            containers=_coerce_specs(data.get("containers", []), ContainerSpec),
            spatial_bounds=_as_tuple(data.get("spatial_bounds"), (5.0, 5.0, 3.0)),
        )
