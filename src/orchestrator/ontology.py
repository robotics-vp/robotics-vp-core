"""
Ontology: Object and affordance knowledge base.

Provides structured knowledge about objects, their properties, and affordances
that the orchestrator can use for reasoning about task execution.

This is additive infrastructure - no changes to Phase B math or RL training loops.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
from enum import Enum
import json


class ObjectCategory(Enum):
    """High-level object categories."""
    CONTAINER = "container"  # Things that hold other things (drawers, boxes)
    MANIPULANDA = "manipulanda"  # Things to be manipulated (objects to grasp)
    SURFACE = "surface"  # Flat surfaces (tables, shelves)
    TOOL = "tool"  # End-effector attachments or tools
    OBSTACLE = "obstacle"  # Things to avoid
    TARGET = "target"  # Goal locations or markers
    FRAGILE = "fragile"  # Breakable objects requiring care


class MaterialType(Enum):
    """Material properties affecting interaction."""
    RIGID = "rigid"  # Solid, non-deformable
    SOFT = "soft"  # Deformable
    FRAGILE = "fragile"  # Breakable
    SLIPPERY = "slippery"  # Low friction
    ROUGH = "rough"  # High friction
    LIQUID = "liquid"  # Fluid
    GRANULAR = "granular"  # Powder, sand


class AffordanceType(Enum):
    """Types of affordances objects can provide."""
    GRASPABLE = "graspable"  # Can be grasped
    PUSHABLE = "pushable"  # Can be pushed
    PULLABLE = "pullable"  # Can be pulled
    LIFTABLE = "liftable"  # Can be lifted
    OPENABLE = "openable"  # Can be opened (doors, drawers)
    CLOSABLE = "closable"  # Can be closed
    PLACEABLE = "placeable"  # Can have things placed on it
    INSERTABLE = "insertable"  # Can be inserted into something
    CONTAINABLE = "containable"  # Can contain other objects
    STACKABLE = "stackable"  # Can be stacked on or stacked upon
    ROLLABLE = "rollable"  # Can roll
    SLIDABLE = "slidable"  # Can slide


@dataclass
class AffordanceSpec:
    """
    Specification of an affordance for an object.

    Describes how an affordance can be activated and any constraints.
    """
    affordance_type: AffordanceType
    confidence: float = 1.0  # How certain we are about this affordance
    constraints: Dict[str, Any] = field(default_factory=dict)
    # e.g., {"max_force": 10.0, "requires_tool": False}
    activation_skill_ids: List[int] = field(default_factory=list)
    # Which skills can activate this affordance
    preconditions: List[str] = field(default_factory=list)
    # e.g., ["gripper_empty", "object_visible"]
    postconditions: List[str] = field(default_factory=list)
    # e.g., ["object_grasped", "object_in_gripper"]
    energy_cost_estimate: float = 0.0  # Estimated energy in Wh
    risk_level: float = 0.0  # Risk of damage/failure (0-1)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "affordance_type": self.affordance_type.value,
            "confidence": self.confidence,
            "constraints": self.constraints,
            "activation_skill_ids": self.activation_skill_ids,
            "preconditions": self.preconditions,
            "postconditions": self.postconditions,
            "energy_cost_estimate": self.energy_cost_estimate,
            "risk_level": self.risk_level,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AffordanceSpec":
        """Create from dictionary."""
        return cls(
            affordance_type=AffordanceType(d["affordance_type"]),
            confidence=d.get("confidence", 1.0),
            constraints=d.get("constraints", {}),
            activation_skill_ids=d.get("activation_skill_ids", []),
            preconditions=d.get("preconditions", []),
            postconditions=d.get("postconditions", []),
            energy_cost_estimate=d.get("energy_cost_estimate", 0.0),
            risk_level=d.get("risk_level", 0.0),
        )


@dataclass
class ObjectSpec:
    """
    Specification of an object in the environment.

    Captures physical properties, affordances, and semantic information.
    """
    object_id: str
    name: str
    description: str = ""
    category: ObjectCategory = ObjectCategory.MANIPULANDA
    material: MaterialType = MaterialType.RIGID

    # Physical properties
    mass_kg: float = 0.1
    dimensions_m: List[float] = field(default_factory=lambda: [0.1, 0.1, 0.1])  # [x, y, z]
    fragility: float = 0.0  # 0 = indestructible, 1 = extremely fragile
    friction_coefficient: float = 0.5

    # Affordances
    affordances: List[AffordanceSpec] = field(default_factory=list)

    # Semantic properties
    tags: List[str] = field(default_factory=list)
    # e.g., ["household", "kitchenware", "valuable"]

    # Relationships to other objects
    contains: List[str] = field(default_factory=list)  # Object IDs of contained objects
    supported_by: Optional[str] = None  # Object ID of supporting object
    attached_to: Optional[str] = None  # Object ID of attachment

    # Economic properties (for cost estimation)
    value_usd: float = 0.0  # Replacement cost
    damage_cost_usd: float = 0.0  # Cost if damaged

    # State
    is_movable: bool = True
    is_visible: bool = True
    current_state: Dict[str, Any] = field(default_factory=dict)
    # e.g., {"open": False, "position": [0, 0, 0], "orientation": [0, 0, 0, 1]}

    def get_affordance(self, aff_type: AffordanceType) -> Optional[AffordanceSpec]:
        """Get affordance by type if available."""
        for aff in self.affordances:
            if aff.affordance_type == aff_type:
                return aff
        return None

    def has_affordance(self, aff_type: AffordanceType) -> bool:
        """Check if object has a specific affordance."""
        return self.get_affordance(aff_type) is not None

    def list_affordances(self) -> List[str]:
        """Get list of affordance type names."""
        return [aff.affordance_type.value for aff in self.affordances]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "object_id": self.object_id,
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "material": self.material.value,
            "mass_kg": self.mass_kg,
            "dimensions_m": self.dimensions_m,
            "fragility": self.fragility,
            "friction_coefficient": self.friction_coefficient,
            "affordances": [aff.to_dict() for aff in self.affordances],
            "tags": self.tags,
            "contains": self.contains,
            "supported_by": self.supported_by,
            "attached_to": self.attached_to,
            "value_usd": self.value_usd,
            "damage_cost_usd": self.damage_cost_usd,
            "is_movable": self.is_movable,
            "is_visible": self.is_visible,
            "current_state": self.current_state,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ObjectSpec":
        """Create from dictionary."""
        affordances = [AffordanceSpec.from_dict(a) for a in d.get("affordances", [])]
        return cls(
            object_id=d["object_id"],
            name=d["name"],
            description=d.get("description", ""),
            category=ObjectCategory(d.get("category", "manipulanda")),
            material=MaterialType(d.get("material", "rigid")),
            mass_kg=d.get("mass_kg", 0.1),
            dimensions_m=d.get("dimensions_m", [0.1, 0.1, 0.1]),
            fragility=d.get("fragility", 0.0),
            friction_coefficient=d.get("friction_coefficient", 0.5),
            affordances=affordances,
            tags=d.get("tags", []),
            contains=d.get("contains", []),
            supported_by=d.get("supported_by"),
            attached_to=d.get("attached_to"),
            value_usd=d.get("value_usd", 0.0),
            damage_cost_usd=d.get("damage_cost_usd", 0.0),
            is_movable=d.get("is_movable", True),
            is_visible=d.get("is_visible", True),
            current_state=d.get("current_state", {}),
        )


@dataclass
class EnvironmentOntology:
    """
    Complete ontology for an environment.

    Contains all objects and their relationships, affordances, and constraints.
    """
    ontology_id: str
    name: str
    description: str = ""
    objects: Dict[str, ObjectSpec] = field(default_factory=dict)  # object_id -> ObjectSpec
    global_constraints: List[Dict[str, Any]] = field(default_factory=list)
    # e.g., [{"type": "collision_free", "objects": ["gripper", "vase"]}]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_object(self, obj: ObjectSpec):
        """Add an object to the ontology."""
        self.objects[obj.object_id] = obj

    def get_object(self, object_id: str) -> Optional[ObjectSpec]:
        """Get object by ID."""
        return self.objects.get(object_id)

    def get_objects_by_category(self, category: ObjectCategory) -> List[ObjectSpec]:
        """Get all objects of a specific category."""
        return [obj for obj in self.objects.values() if obj.category == category]

    def get_objects_with_affordance(self, aff_type: AffordanceType) -> List[ObjectSpec]:
        """Get all objects that have a specific affordance."""
        return [obj for obj in self.objects.values() if obj.has_affordance(aff_type)]

    def get_fragile_objects(self, threshold: float = 0.5) -> List[ObjectSpec]:
        """Get objects with fragility above threshold."""
        return [obj for obj in self.objects.values() if obj.fragility >= threshold]

    def compute_total_damage_risk(self) -> float:
        """Compute total potential damage cost."""
        return sum(obj.damage_cost_usd for obj in self.objects.values())

    def get_object_graph(self) -> Dict[str, List[str]]:
        """Get containment/support graph as adjacency list."""
        graph = {obj_id: [] for obj_id in self.objects}
        for obj_id, obj in self.objects.items():
            graph[obj_id].extend(obj.contains)
            if obj.supported_by and obj.supported_by in graph:
                graph[obj.supported_by].append(obj_id)
        return graph

    def summary(self) -> Dict[str, Any]:
        """Get ontology summary."""
        return {
            "total_objects": len(self.objects),
            "objects_by_category": {
                cat.value: len(self.get_objects_by_category(cat))
                for cat in ObjectCategory
            },
            "total_affordances": sum(
                len(obj.affordances) for obj in self.objects.values()
            ),
            "fragile_objects": len(self.get_fragile_objects()),
            "total_damage_risk": self.compute_total_damage_risk(),
            "constraints": len(self.global_constraints),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "ontology_id": self.ontology_id,
            "name": self.name,
            "description": self.description,
            "objects": {oid: obj.to_dict() for oid, obj in self.objects.items()},
            "global_constraints": self.global_constraints,
            "metadata": self.metadata,
            "summary": self.summary(),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EnvironmentOntology":
        """Create from dictionary."""
        objects = {
            oid: ObjectSpec.from_dict(odict)
            for oid, odict in d.get("objects", {}).items()
        }
        return cls(
            ontology_id=d["ontology_id"],
            name=d["name"],
            description=d.get("description", ""),
            objects=objects,
            global_constraints=d.get("global_constraints", []),
            metadata=d.get("metadata", {}),
        )

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "EnvironmentOntology":
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))


def build_drawer_vase_ontology() -> EnvironmentOntology:
    """
    Build ontology for drawer_vase environment.

    Contains drawer, vase, and table objects with appropriate affordances.
    """
    # Drawer
    drawer = ObjectSpec(
        object_id="drawer_01",
        name="Drawer",
        description="Wooden drawer with metal handle",
        category=ObjectCategory.CONTAINER,
        material=MaterialType.RIGID,
        mass_kg=2.0,
        dimensions_m=[0.4, 0.3, 0.15],
        fragility=0.1,
        friction_coefficient=0.6,
        affordances=[
            AffordanceSpec(
                affordance_type=AffordanceType.OPENABLE,
                constraints={"max_pull_force": 20.0, "direction": "forward"},
                activation_skill_ids=[2],  # PULL skill
                preconditions=["handle_grasped"],
                postconditions=["drawer_open"],
                energy_cost_estimate=0.05,
                risk_level=0.1,
            ),
            AffordanceSpec(
                affordance_type=AffordanceType.CLOSABLE,
                constraints={"max_push_force": 15.0},
                activation_skill_ids=[7],  # PUSH skill
                preconditions=["drawer_open"],
                postconditions=["drawer_closed"],
                energy_cost_estimate=0.03,
                risk_level=0.05,
            ),
            AffordanceSpec(
                affordance_type=AffordanceType.GRASPABLE,
                constraints={"grasp_point": "handle"},
                activation_skill_ids=[1],  # GRASP skill
                preconditions=["gripper_near_handle"],
                postconditions=["handle_grasped"],
                energy_cost_estimate=0.01,
                risk_level=0.05,
            ),
        ],
        tags=["furniture", "storage"],
        value_usd=50.0,
        damage_cost_usd=30.0,
        is_movable=False,
        current_state={"open": False, "extension": 0.0},
    )

    # Vase (fragile!)
    vase = ObjectSpec(
        object_id="vase_01",
        name="Glass Vase",
        description="Delicate glass vase that can break if knocked over",
        category=ObjectCategory.FRAGILE,
        material=MaterialType.FRAGILE,
        mass_kg=0.5,
        dimensions_m=[0.1, 0.1, 0.25],
        fragility=0.9,  # Very fragile!
        friction_coefficient=0.3,
        affordances=[
            AffordanceSpec(
                affordance_type=AffordanceType.GRASPABLE,
                constraints={"grip_force_max": 5.0, "requires_care": True},
                activation_skill_ids=[1],
                preconditions=["gripper_empty", "vase_visible"],
                postconditions=["vase_grasped"],
                energy_cost_estimate=0.02,
                risk_level=0.7,  # High risk due to fragility
            ),
            AffordanceSpec(
                affordance_type=AffordanceType.LIFTABLE,
                constraints={"lift_speed_max": 0.1},
                activation_skill_ids=[4],
                preconditions=["vase_grasped"],
                postconditions=["vase_lifted"],
                energy_cost_estimate=0.01,
                risk_level=0.6,
            ),
            AffordanceSpec(
                affordance_type=AffordanceType.PLACEABLE,
                constraints={"requires_stable_surface": True},
                activation_skill_ids=[6],
                preconditions=["vase_lifted"],
                postconditions=["vase_placed"],
                energy_cost_estimate=0.01,
                risk_level=0.5,
            ),
        ],
        tags=["decorative", "fragile", "glass", "valuable"],
        value_usd=100.0,
        damage_cost_usd=100.0,  # Total loss if broken
        is_movable=True,
        current_state={"upright": True, "position": [0.3, 0.0, 0.0]},
    )

    # Table/surface
    table = ObjectSpec(
        object_id="table_01",
        name="Work Table",
        description="Stable wooden table surface",
        category=ObjectCategory.SURFACE,
        material=MaterialType.RIGID,
        mass_kg=20.0,
        dimensions_m=[1.0, 0.6, 0.75],
        fragility=0.0,
        friction_coefficient=0.5,
        affordances=[
            AffordanceSpec(
                affordance_type=AffordanceType.PLACEABLE,
                constraints={"max_load_kg": 50.0},
                activation_skill_ids=[6],
                preconditions=["object_lifted"],
                postconditions=["object_on_table"],
                energy_cost_estimate=0.0,
                risk_level=0.1,
            ),
        ],
        tags=["furniture", "workspace"],
        value_usd=150.0,
        damage_cost_usd=50.0,
        is_movable=False,
        current_state={"height": 0.75},
    )

    # Build ontology
    ontology = EnvironmentOntology(
        ontology_id="drawer_vase_v1",
        name="Drawer-Vase Environment",
        description="Environment with drawer, fragile vase, and table",
        global_constraints=[
            {
                "type": "collision_avoidance",
                "priority": "high",
                "objects": ["vase_01"],
                "description": "Avoid colliding with fragile vase",
            },
            {
                "type": "fragility_awareness",
                "priority": "critical",
                "objects": ["vase_01"],
                "description": "Handle vase with extreme care or avoid entirely",
            },
        ],
        metadata={
            "difficulty": "medium",
            "primary_risk": "vase_breakage",
            "task_family": "drawer_manipulation",
        },
    )

    ontology.add_object(drawer)
    ontology.add_object(vase)
    ontology.add_object(table)

    # Set up relationships
    vase.supported_by = "table_01"
    drawer.supported_by = "table_01"

    return ontology


def build_simple_grasp_ontology() -> EnvironmentOntology:
    """Build simple ontology for basic grasp task (no fragile objects)."""
    cube = ObjectSpec(
        object_id="cube_01",
        name="Wooden Cube",
        description="Solid wooden cube for manipulation",
        category=ObjectCategory.MANIPULANDA,
        material=MaterialType.RIGID,
        mass_kg=0.2,
        dimensions_m=[0.05, 0.05, 0.05],
        fragility=0.0,
        friction_coefficient=0.6,
        affordances=[
            AffordanceSpec(
                affordance_type=AffordanceType.GRASPABLE,
                activation_skill_ids=[1],
                postconditions=["cube_grasped"],
                risk_level=0.1,
            ),
            AffordanceSpec(
                affordance_type=AffordanceType.LIFTABLE,
                activation_skill_ids=[4],
                preconditions=["cube_grasped"],
                postconditions=["cube_lifted"],
                risk_level=0.1,
            ),
            AffordanceSpec(
                affordance_type=AffordanceType.STACKABLE,
                activation_skill_ids=[6],
                preconditions=["cube_lifted"],
                postconditions=["cube_stacked"],
                risk_level=0.2,
            ),
        ],
        tags=["toy", "block"],
        value_usd=5.0,
        damage_cost_usd=0.0,
    )

    target = ObjectSpec(
        object_id="target_01",
        name="Target Location",
        description="Goal location marker",
        category=ObjectCategory.TARGET,
        material=MaterialType.RIGID,
        affordances=[],
        tags=["marker", "goal"],
        is_movable=False,
    )

    ontology = EnvironmentOntology(
        ontology_id="simple_grasp_v1",
        name="Simple Grasp Environment",
        description="Basic manipulation task with single object",
        metadata={"difficulty": "easy", "fragile_objects": False},
    )

    ontology.add_object(cube)
    ontology.add_object(target)

    return ontology
