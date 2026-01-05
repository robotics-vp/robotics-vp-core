# Workcell Environment Ontology

This document describes the entity schema for the manufacturing/workcell environment suite and how it maps to the vector scene representation.

## Entity Types

### Workcell (Root Container)

The top-level container for a manufacturing scene.

```
WorkcellSceneSpec:
  workcell_id: str
  stations: List[StationSpec]
  fixtures: List[FixtureSpec]
  parts: List[PartSpec]
  tools: List[ToolSpec]
  conveyors: List[ConveyorSpec]
  containers: List[ContainerSpec]
  spatial_bounds: Tuple[float, float, float]
```

### Station

A work station or bench where operations occur.

```
StationSpec:
  id: str
  position: Tuple[float, float, float]
  orientation: Tuple[float, float, float, float]  # quaternion
  station_type: str  # "bench", "conveyor_segment", "inspection_cell"
  work_area: Tuple[float, float, float]  # dimensions
```

### Fixture

A jig, vise, clamp, or holder that constrains parts.

```
FixtureSpec:
  id: str
  position: Tuple[float, float, float]
  orientation: Tuple[float, float, float, float]
  fixture_type: str  # "jig", "vise", "clamp", "bracket_holder"
  mounted_on: Optional[str]  # station_id
```

### Container

A bin, tray, pallet, or box for holding parts.

```
ContainerSpec:
  id: str
  position: Tuple[float, float, float]
  orientation: Tuple[float, float, float, float]
  container_type: str  # "bin", "tray", "pallet", "box"
  capacity: int  # max items
  current_count: int
```

### Conveyor

A moving belt or line segment.

```
ConveyorSpec:
  id: str
  start_position: Tuple[float, float, float]
  end_position: Tuple[float, float, float]
  width: float
  speed: float  # m/s
  direction: Tuple[float, float, float]  # unit vector
```

### Tool

An end-effector tool like wrench, screwdriver, gripper.

```
ToolSpec:
  id: str
  position: Tuple[float, float, float]
  orientation: Tuple[float, float, float, float]
  tool_type: str  # "wrench", "screwdriver", "gripper", "probe"
  in_use: bool
```

### Part

A manipulable object (bolt, screw, plate, bracket, etc.).

```
PartSpec:
  id: str
  position: Tuple[float, float, float]
  orientation: Tuple[float, float, float, float]
  part_type: str  # "bolt", "screw", "plate", "bracket", "housing", "peg"
  dimensions: Tuple[float, float, float]
  mass: float
  graspable: bool
```

## Relations / Affordances

### Spatial Relations

| Relation | Description | Example |
|----------|-------------|---------|
| `contains` | Container holds part(s) | `bin_0 contains [part_0, part_1]` |
| `mounted_on` | Fixture attached to station | `fixture_0 mounted_on station_0` |
| `at_position` | Entity at spatial location | `part_0 at_position (1.0, 0.5, 0.8)` |

### Constraint Relations

| Relation | Description | Example |
|----------|-------------|---------|
| `constrained_by` | Part held by fixture | `part_a constrained_by fixture_0` |
| `mates_with` | Parts assembled together | `bracket mates_with base` |
| `fastened_by` | Parts joined by fastener | `[part_a, part_b] fastened_by screw_0` |

### Affordances

| Affordance | Applicable To | Description |
|------------|---------------|-------------|
| `graspable` | Part | Can be picked by gripper |
| `insertable` | Part | Can be inserted into hole/slot |
| `fastenable` | Part, Fastener | Can be screwed/bolted |
| `inspectable` | Part | Can be visually inspected |
| `placeable` | Container, Fixture | Can receive parts |

## Task Graph Mapping

### Action Types

```
ActionType = Literal[
  "PICK",      # Grasp an object
  "PLACE",     # Release at location
  "INSERT",    # Insert into hole/slot
  "FASTEN",    # Screw/bolt
  "ROUTE",     # Cable routing
  "INSPECT",   # Visual check
  "PACKAGE",   # Box/wrap
]
```

### Task Graph Structure

```
TaskGraphSpec:
  nodes: List[ActionStepSpec]  # Action steps
  edges: List[Tuple[str, str]]  # Dependencies (src, dst)
  entry_node: str              # Starting node
  exit_nodes: List[str]        # Terminal nodes
```

### Example: Kitting Task Graph

```
Nodes:
  - pick_0: PICK part_0
  - place_0: PLACE part_0 -> tray_0
  - pick_1: PICK part_1
  - place_1: PLACE part_1 -> tray_0

Edges:
  pick_0 -> place_0
  place_0 -> pick_1
  pick_1 -> place_1

Entry: pick_0
Exit: [place_1]
```

### Example: Peg-in-Hole Task Graph

```
Nodes:
  - pick_peg: PICK peg_0
  - insert_peg: INSERT peg_0 -> hole_0 [tolerance: 1mm]

Edges:
  pick_peg -> insert_peg

Entry: pick_peg
Exit: [insert_peg]
```

## Vector Scene Mapping

The workcell ontology maps to `src/scene/vector_scene/graph.py`:

| Workcell Entity | Vector Scene Node Type |
|-----------------|------------------------|
| Station | `structure_node` |
| Fixture | `structure_node` (with `fixture` tag) |
| Container | `container_node` |
| Conveyor | `dynamic_structure_node` |
| Part | `object_node` |
| Tool | `tool_node` |

### Scene Graph Encoding

Each entity becomes a node in the scene graph with:
- **Position**: 3D coordinates
- **Orientation**: Quaternion
- **Type embedding**: One-hot or learned
- **State features**: Task-relevant state (grasped, placed, etc.)

### Integration with Scene IR Tracker

The workcell scene can be reconstructed from `SceneTracks_v1`:
1. Track trajectories provide object positions over time
2. Semantic labels (from map_first) provide entity types
3. `SceneTracksAdapter` maps tracks to `WorkcellSceneSpec`

## Orchestrator Integration

The orchestrator can query the ontology via `src/ontology/store.py`:

```python
# Query objects by type
parts = store.query_entities(entity_type="Part")

# Query spatial relations
contained = store.query_relations(relation="contains", subject="bin_0")

# Query affordances
graspable = store.query_affordances(affordance="graspable")
```

## Process Reward Integration

Difficulty features are computed from the ontology:

| Feature | Computation |
|---------|-------------|
| `part_count` | Number of Part entities |
| `occlusion_proxy` | Container fill rate |
| `tolerance_factor` | 1 / tolerance_mm |
| `horizon_length` | Task graph depth |
| `tool_changes` | Unique tool types used |
| `assembly_depth` | Max `mates_with` chain length |
