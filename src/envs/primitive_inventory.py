"""Per-environment primitive inventory for semantic coverage graph.

Each environment/backend exports a typed inventory of manipulation, navigation,
contact, risk, recovery, and observation primitives.  The coverage graph
consumes these to determine which task–skill–env-primitive edges exist, are
missing, or are under-covered.

All additions are purely additive — no existing env code is modified.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional


# ---------------------------------------------------------------------------
# Core types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EnvPrimitive:
    """Single environment primitive capability."""

    primitive_id: str
    category: str  # manipulation | navigation | contact | risk | recovery | observation
    label: str
    description: str = ""
    backend_constraints: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnvPrimitiveInventory:
    """Complete primitive inventory for one environment family."""

    env_id: str
    env_family: str
    primitives: List[EnvPrimitive] = field(default_factory=list)
    object_families: List[str] = field(default_factory=list)
    risk_primitives: List[str] = field(default_factory=list)
    recovery_primitives: List[str] = field(default_factory=list)
    observation_limitations: List[str] = field(default_factory=list)
    backend_constraints: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # -- queries --
    def primitives_by_category(self, category: str) -> List[EnvPrimitive]:
        return [p for p in self.primitives if p.category == category]

    def primitive_ids(self) -> List[str]:
        return [p.primitive_id for p in self.primitives]

    def has_primitive(self, primitive_id: str) -> bool:
        return any(p.primitive_id == primitive_id for p in self.primitives)

    # -- serialisation --
    def to_dict(self) -> Dict[str, Any]:
        return {
            "env_id": self.env_id,
            "env_family": self.env_family,
            "primitives": [
                {
                    "primitive_id": p.primitive_id,
                    "category": p.category,
                    "label": p.label,
                    "description": p.description,
                    "backend_constraints": list(p.backend_constraints),
                    "metadata": dict(p.metadata),
                }
                for p in self.primitives
            ],
            "object_families": list(self.object_families),
            "risk_primitives": list(self.risk_primitives),
            "recovery_primitives": list(self.recovery_primitives),
            "observation_limitations": list(self.observation_limitations),
            "backend_constraints": list(self.backend_constraints),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EnvPrimitiveInventory":
        prims = [
            EnvPrimitive(
                primitive_id=p["primitive_id"],
                category=p["category"],
                label=p["label"],
                description=p.get("description", ""),
                backend_constraints=list(p.get("backend_constraints", [])),
                metadata=dict(p.get("metadata", {})),
            )
            for p in payload.get("primitives", [])
        ]
        return cls(
            env_id=str(payload["env_id"]),
            env_family=str(payload["env_family"]),
            primitives=prims,
            object_families=list(payload.get("object_families", [])),
            risk_primitives=list(payload.get("risk_primitives", [])),
            recovery_primitives=list(payload.get("recovery_primitives", [])),
            observation_limitations=list(payload.get("observation_limitations", [])),
            backend_constraints=list(payload.get("backend_constraints", [])),
            metadata=dict(payload.get("metadata", {})),
        )


# ---------------------------------------------------------------------------
# Factory + built-in inventories
# ---------------------------------------------------------------------------

_REGISTRY: Dict[str, EnvPrimitiveInventory] = {}


def register_inventory(inventory: EnvPrimitiveInventory) -> None:
    """Register an inventory in the global registry."""
    _REGISTRY[inventory.env_id] = inventory


def get_inventory(env_id: str) -> Optional[EnvPrimitiveInventory]:
    """Lookup inventory by env_id.  Returns None if not registered."""
    return _REGISTRY.get(env_id)


def list_registered_env_ids() -> List[str]:
    """Return all registered env IDs."""
    return sorted(_REGISTRY.keys())


def for_env(env_id: str) -> EnvPrimitiveInventory:
    """Get inventory for *env_id*, raising KeyError if unregistered."""
    inv = _REGISTRY.get(env_id)
    if inv is None:
        raise KeyError(
            f"No primitive inventory registered for env_id={env_id!r}.  "
            f"Registered: {list_registered_env_ids()}"
        )
    return inv


# ---------------------------------------------------------------------------
# Built-in: drawer-vase
# ---------------------------------------------------------------------------

_DRAWER_VASE = EnvPrimitiveInventory(
    env_id="drawer_vase",
    env_family="drawer_vase",
    primitives=[
        EnvPrimitive("locate_handle", "manipulation", "Locate drawer handle",
                     description="Visual localisation of the drawer handle pose"),
        EnvPrimitive("detect_fragile_obstacle", "observation", "Detect fragile obstacle",
                     description="Identify vase or other fragile objects near workspace"),
        EnvPrimitive("plan_safe_approach", "navigation", "Plan safe approach path",
                     description="Collision-free motion plan that avoids fragile objects"),
        EnvPrimitive("grasp_handle", "manipulation", "Grasp drawer handle",
                     description="Move to and close gripper on handle"),
        EnvPrimitive("open_with_clearance", "manipulation", "Open drawer with clearance",
                     description="Pull drawer open while maintaining vase clearance"),
        EnvPrimitive("retract_safe", "navigation", "Retract to safe position",
                     description="Return end-effector to home without contacts"),
        EnvPrimitive("collision_avoidance", "risk", "Collision avoidance",
                     description="Real-time obstacle avoidance during motion"),
        EnvPrimitive("fragile_proximity_recovery", "recovery", "Fragile proximity recovery",
                     description="Recovery behaviour when too close to fragile object"),
    ],
    object_families=["drawer", "vase", "handle", "gripper"],
    risk_primitives=["collision_avoidance", "fragile_proximity_recovery"],
    recovery_primitives=["fragile_proximity_recovery"],
    observation_limitations=["partial_occlusion", "depth_noise"],
    backend_constraints=["isaac_gym", "pybullet"],
)

# ---------------------------------------------------------------------------
# Built-in: dishwashing
# ---------------------------------------------------------------------------

_DISHWASHING = EnvPrimitiveInventory(
    env_id="dishwashing",
    env_family="dishwashing",
    primitives=[
        EnvPrimitive("locate_dish", "observation", "Locate dish/utensil",
                     description="Detect and localise dish, cup, or utensil"),
        EnvPrimitive("grasp_object", "manipulation", "Grasp dish/utensil",
                     description="Pick up dishware with appropriate grip"),
        EnvPrimitive("contact_surface", "contact", "Contact scrubbing surface",
                     description="Controlled contact between tool and dish surface"),
        EnvPrimitive("scrub", "manipulation", "Scrub motion",
                     description="Cyclic scrubbing motion with controlled force"),
        EnvPrimitive("rinse", "manipulation", "Rinse under water",
                     description="Hold object under water stream for rinsing"),
        EnvPrimitive("place_stow", "manipulation", "Place / stow dish",
                     description="Place cleaned dish in rack or stowage"),
        EnvPrimitive("wet_surface_slip", "risk", "Wet surface slip risk",
                     description="Grip instability on wet surfaces"),
        EnvPrimitive("breakage_recovery", "recovery", "Breakage recovery",
                     description="Recovery after dropping or chipping dishware"),
    ],
    object_families=["dish", "cup", "utensil", "sponge", "rack"],
    risk_primitives=["wet_surface_slip"],
    recovery_primitives=["breakage_recovery"],
    observation_limitations=["water_splash_occlusion", "specular_reflection"],
    backend_constraints=["pybullet"],
)

# ---------------------------------------------------------------------------
# Built-in: workcell
# ---------------------------------------------------------------------------

_WORKCELL = EnvPrimitiveInventory(
    env_id="workcell",
    env_family="workcell",
    primitives=[
        EnvPrimitive("pick", "manipulation", "Pick part",
                     description="Grasp and lift a part from fixture/conveyor"),
        EnvPrimitive("place", "manipulation", "Place part",
                     description="Position and release part into target fixture"),
        EnvPrimitive("insert", "manipulation", "Insert part",
                     description="Precision insertion into receptacle"),
        EnvPrimitive("align", "manipulation", "Align part",
                     description="Fine alignment before insertion or placement"),
        EnvPrimitive("avoid_collision", "risk", "Avoid collision",
                     description="Real-time collision avoidance with fixtures"),
        EnvPrimitive("recover_from_occlusion", "recovery", "Recover from occlusion",
                     description="Re-localise part after visual occlusion event"),
        EnvPrimitive("tool_change", "manipulation", "Tool change",
                     description="Switch gripper/tool at tool station"),
        EnvPrimitive("conveyor_track", "navigation", "Conveyor tracking",
                     description="Synchronise with moving conveyor"),
    ],
    object_families=["part", "fixture", "tool", "container", "conveyor"],
    risk_primitives=["avoid_collision"],
    recovery_primitives=["recover_from_occlusion"],
    observation_limitations=["fixture_occlusion", "reflective_parts"],
    backend_constraints=["isaac_gym", "pybullet"],
)

# Register built-ins
for _inv in (_DRAWER_VASE, _DISHWASHING, _WORKCELL):
    register_inventory(_inv)


__all__ = [
    "EnvPrimitive",
    "EnvPrimitiveInventory",
    "register_inventory",
    "get_inventory",
    "list_registered_env_ids",
    "for_env",
]
