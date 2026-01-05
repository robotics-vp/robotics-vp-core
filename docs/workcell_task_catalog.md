# Workcell Task Catalog

This catalog defines the canonical manufacturing task families and difficulty progression for the workcell environment suite.

## Task Family Overview

| Family | Description | Use Case |
|--------|-------------|----------|
| **Logistics** | Material handling, kitting, sorting, bin picking | Warehouse ops, fulfillment, parts staging |
| **Assembly** | Component joining, fastening, insertion | Production lines, sub-assembly stations |
| **Inspection** | Visual QC, defect detection, measurement | In-line QA, final inspection |
| **Maintenance** | Tool changes, fixture setup, calibration | Cell changeover, preventive maintenance |

---

## Difficulty Ladder

Each task follows a 5-level difficulty progression (L0–L4):

| Level | Name | Characteristics |
|-------|------|-----------------|
| **L0** | Sandbox | Single object, no constraints, infinite time |
| **L1** | Tutorial | 2-4 objects, loose tolerances (5mm), simple layout |
| **L2** | Standard | 6-10 objects, moderate tolerances (2mm), obstacles present |
| **L3** | Production | 10+ objects, tight tolerances (0.5mm), time pressure, occlusions |
| **L4** | Expert | Real-world noise, multi-step dependencies, failure recovery |

---

## Task Catalog

### 1. Logistics Family

#### TASK-L001: Kitting (Tray Packing)

**Description:** Pick parts from source bins and place into kit tray slots.

| Level | Config | Success Criteria |
|-------|--------|------------------|
| L0 | 1 part, 1 slot, no time limit | Part in slot |
| L1 | 4 parts, 4 slots, tolerance=5mm | All parts placed |
| L2 | 8 parts, mixed sizes, tolerance=2mm | ≥90% placed correctly |
| L3 | 12 parts, nested trays, tolerance=1mm, 60s limit | ≥95% placed, under time |
| L4 | Variable parts, damaged tray recovery, occlusions | ≥98% placed, handle failures |

**Preset:** `PRESETS["kitting_l2"]`
**Compiler prompt:** `"Pack {n} {part_type} into tray with {tol}mm tolerance"`

---

#### TASK-L002: Bin Picking

**Description:** Extract target parts from cluttered bin with unknown poses.

| Level | Config | Success Criteria |
|-------|--------|------------------|
| L0 | 3 parts, no occlusion | All parts extracted |
| L1 | 6 parts, 20% occlusion | ≥80% extracted |
| L2 | 10 parts, 50% occlusion, mixed types | ≥85% extracted |
| L3 | 15 parts, heavy clutter, partial views | ≥90% extracted |
| L4 | Unknown part types, dynamic bin refill | ≥92% extracted, adapt to changes |

**Preset:** `PRESETS["bin_picking_l2"]`
**Compiler prompt:** `"Pick {n} parts from cluttered bin with {occlusion}% occlusion"`

---

#### TASK-L003: Conveyor Sorting

**Description:** Identify and divert items on moving belt into correct output bins.

| Level | Config | Success Criteria |
|-------|--------|------------------|
| L0 | 2 types, slow belt (0.1m/s) | All sorted |
| L1 | 3 types, belt=0.2m/s | ≥90% sorted |
| L2 | 4 types, belt=0.3m/s, variable spacing | ≥92% sorted |
| L3 | 5 types, belt=0.4m/s, defect detection | ≥95% sorted + defects flagged |
| L4 | Unknown types, belt jams, multi-lane | ≥97% sorted, handle exceptions |

**Preset:** `PRESETS["conveyor_sorting_l2"]`
**Compiler prompt:** `"Sort {n} item types on conveyor at {speed}m/s into {bins} bins"`

---

### 2. Assembly Family

#### TASK-A001: Peg-in-Hole Insertion

**Description:** Insert peg (cylindrical/prismatic) into matching hole with tight clearance.

| Level | Config | Success Criteria |
|-------|--------|------------------|
| L0 | 10mm peg, 12mm hole, no alignment | Peg seated |
| L1 | 8mm peg, 8.5mm hole, coarse alignment | Peg seated |
| L2 | 6mm peg, 6.2mm hole, force feedback | Peg seated, no jamming |
| L3 | 4mm peg, 4.1mm hole, chamferless | Peg seated, spiral search |
| L4 | Multiple pegs, variable holes, blind insertion | All pegs seated |

**Preset:** `PRESETS["peg_in_hole_l2"]`
**Compiler prompt:** `"Insert {peg_dia}mm peg into hole with {clearance}mm clearance"`

---

#### TASK-A002: Fastener Installation

**Description:** Pick fastener, align, and drive to torque spec.

| Level | Config | Success Criteria |
|-------|--------|------------------|
| L0 | Pre-aligned bolt, hand-tight | Fastener started |
| L1 | Single bolt, auto-align, 5Nm | Torque achieved |
| L2 | 4 bolts, pattern sequence, 10Nm | All bolts to spec |
| L3 | 8 bolts, mixed types, torque sequence | All bolts, angle verification |
| L4 | Cross-threaded recovery, stripped detection | All bolts, handle failures |

**Preset:** `PRESETS["fastener_l2"]`
**Compiler prompt:** `"Install {n} {fastener_type} to {torque}Nm"`

---

#### TASK-A003: Multi-Part Assembly

**Description:** Combine multiple sub-components into final assembly.

| Level | Config | Success Criteria |
|-------|--------|------------------|
| L0 | 2 parts, snap-fit | Assembly complete |
| L1 | 4 parts, ordered sequence | Assembly complete |
| L2 | 6 parts, fixture required, press-fit | Assembly + quality check |
| L3 | 10 parts, parallel sub-assemblies | Assembly, no sequence errors |
| L4 | Variable BOM, missing part handling | Assembly + exception recovery |

**Preset:** `PRESETS["assembly_l2"]`
**Compiler prompt:** `"Assemble {n} parts following {sequence} with {fixture}"`

---

### 3. Inspection Family

#### TASK-I001: Visual Defect Detection

**Description:** Inspect parts under camera for surface defects.

| Level | Config | Success Criteria |
|-------|--------|------------------|
| L0 | Binary (pass/fail), obvious defects | 100% detection |
| L1 | 3 defect types, good lighting | ≥95% detection |
| L2 | 5 defect types, variable lighting | ≥92% detection, ≤5% false positive |
| L3 | 10 defect types, subtle defects | ≥90% detection, ≤3% false positive |
| L4 | Unknown defect types, adaptive threshold | ≥88% detection, flagging novel defects |

**Preset:** `PRESETS["inspection_l2"]`
**Compiler prompt:** `"Inspect parts for {defect_types} under {lighting} conditions"`

---

#### TASK-I002: Dimensional Verification

**Description:** Measure part dimensions against specification.

| Level | Config | Success Criteria |
|-------|--------|------------------|
| L0 | 1 dimension, ±1mm | Measurement within spec |
| L1 | 3 dimensions, ±0.5mm | All dimensions pass |
| L2 | 6 dimensions, ±0.2mm, datum alignment | All dimensions, GD&T basic |
| L3 | 10 dimensions, ±0.1mm, statistical tracking | Cpk reporting |
| L4 | Full CMM-style inspection, unknown features | All features, root cause flagging |

**Preset:** `PRESETS["measurement_l2"]`
**Compiler prompt:** `"Verify {n} dimensions to ±{tolerance}mm"`

---

### 4. Maintenance Family

#### TASK-M001: Tool Change

**Description:** Swap end-effector tools using tool changer.

| Level | Config | Success Criteria |
|-------|--------|------------------|
| L0 | Manual tool dock, single tool | Tool attached |
| L1 | Auto tool changer, 2 tools | Tool swap complete |
| L2 | 4 tools, tool rack sequence | All swaps correct |
| L3 | 6 tools, tool wear tracking | Swaps + wear logging |
| L4 | Unknown tools, calibration after swap | Swaps + auto-calibration |

**Preset:** `PRESETS["tool_change_l2"]`
**Compiler prompt:** `"Change from {tool_a} to {tool_b} using {changer_type}"`

---

#### TASK-M002: Fixture Setup

**Description:** Configure workholding fixtures for new part variant.

| Level | Config | Success Criteria |
|-------|--------|------------------|
| L0 | Fixed fixture, part placement only | Part fixtured |
| L1 | Adjustable clamps, 2 positions | Part secured correctly |
| L2 | Modular fixture, 4 configurations | Config matches part variant |
| L3 | Flexible fixturing, auto-adjust | Config + force verification |
| L4 | Unknown part geometry, fixture synthesis | Config generated + verified |

**Preset:** `PRESETS["fixture_setup_l2"]`
**Compiler prompt:** `"Setup fixture for {part_variant} with {clamp_config}"`

---

## Task ID Reference

| Task ID | Family | Name | Default Level |
|---------|--------|------|---------------|
| TASK-L001 | Logistics | Kitting | L2 |
| TASK-L002 | Logistics | Bin Picking | L2 |
| TASK-L003 | Logistics | Conveyor Sorting | L2 |
| TASK-A001 | Assembly | Peg-in-Hole | L2 |
| TASK-A002 | Assembly | Fastener Installation | L2 |
| TASK-A003 | Assembly | Multi-Part Assembly | L2 |
| TASK-I001 | Inspection | Visual Defect Detection | L2 |
| TASK-I002 | Inspection | Dimensional Verification | L2 |
| TASK-M001 | Maintenance | Tool Change | L2 |
| TASK-M002 | Maintenance | Fixture Setup | L2 |

---

## Usage

### Programmatic Access

```python
from src.envs.workcell_env import WorkcellEnv
from src.envs.workcell_env.config import PRESETS

# Load a preset task
env = WorkcellEnv(config=PRESETS["kitting_l2"], seed=42)
obs = env.reset()

# Or via task compiler
from src.envs.workcell_env.compiler import WorkcellTaskCompiler
compiler = WorkcellTaskCompiler()
result = compiler.compile_from_prompt("Pack 8 bolts into tray with 2mm tolerance")
env = WorkcellEnv(config=result.config, seed=42)
```

### Difficulty Progression

```python
# Train curriculum: L0 → L4
levels = ["kitting_l0", "kitting_l1", "kitting_l2", "kitting_l3", "kitting_l4"]
for level_preset in levels:
    env = WorkcellEnv(config=PRESETS[level_preset])
    # Train until success threshold
    # Advance to next level
```

---

## Extending the Catalog

To add a new task:

1. Create task handler in `src/envs/workcell_env/tasks/`
2. Register in `tasks/__init__.py`
3. Add presets in `src/envs/workcell_env/config.py`
4. Add entry to this catalog with difficulty levels
5. Update compiler patterns in `compiler.py`
