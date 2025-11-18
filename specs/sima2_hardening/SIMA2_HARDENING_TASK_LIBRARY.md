# SIMA-2 Hardening: Task Library & Rollout Regime

**Status**: Draft
**Owner**: SIMA-2 Hardening Team
**Context**: Stage 2 Pipeline Hardening

## 1. Task Family Taxonomy

This taxonomy defines the canonical set of tasks for SIMA-2. Each family represents a distinct semantic cluster that the Ontology and Econ stack must recognize.

### 1.1. Core Manipulation Families

| Task Family | Description | Required Objects | Affordances | Typical Failure Modes | Recovery Strategy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `drawer_open` | Grasping a handle and pulling a prismatic joint to a limit. | `drawer`, `handle` | `graspable`, `pullable` | `grasp_slip`, `locked_joint`, `collision_with_frame` | `re-grasp`, `wiggle_pull` |
| `dish_place` | Picking a dish and placing it into a target receptacle (e.g., rack, cabinet). | `dish`, `target_surface` | `graspable`, `placeable`, `support_surface` | `drop`, `misalignment`, `unstable_place` | `pick_from_drop`, `nudge` |
| `wipe_surface` | Applying force while following a trajectory on a 2D manifold. | `cloth/sponge`, `surface` | `wipeable`, `compressible` | `insufficient_force`, `missed_patch`, `tool_loss` | `re-trace`, `re-grasp_tool` |
| `pick_and_place` | General transport of an object A to location B. | `object`, `target` | `graspable`, `transportable` | `drop_en_route`, `target_obstructed` | `clear_obstruction`, `retry_pick` |
| `sort_objects` | Iterative pick-and-place of multiple objects into categories. | `objects[]`, `bins[]` | `sortable`, `container` | `wrong_bin`, `bin_full`, `clutter_collision` | `remove_from_wrong_bin`, `shake_bin` |

### 1.2. Meta-Tasks (Recovery & Maintenance)

| Task Family | Description | Triggers | Success Criteria |
| :--- | :--- | :--- | :--- |
| `reset_after_failure` | Returning the robot/scene to a neutral state after a critical failure. | `drop`, `collision_lock` | Robot at home, workspace safe. |
| `inspect_scene` | Active perception scan to update world model. | `occlusion`, `lost_object` | Object pose confidence > threshold. |
| `tool_change` | Swapping end-effector or held tool. | `wrong_tool_for_task` | New tool secured. |

## 2. Rollout Templates

To ensure the Stage 2 pipeline sees a rich distribution of data, we define four canonical rollout templates. The SIMA-2 client must be able to generate these deterministically.

### 2.1. Normal Success (`template_success`)
*   **Description**: The "happy path". Minimal noise, optimal or near-optimal trajectory.
*   **Structure**: `Approach` -> `Interact` -> `Complete`.
*   **Signals**: High success probability, low risk, standard energy usage.
*   **Use Case**: Establishing baseline economic value (MPL, Wh).

### 2.2. Pure Failure (`template_failure`)
*   **Description**: A rollout that fails and terminates without recovery.
*   **Structure**: `Approach` -> `Interact` -> `FailureEvent` -> `Abort`.
*   **Signals**: High risk, fragility discovery.
*   **Use Case**: Training the `RiskTag` predictor and `FragilityNode` insertion in Ontology.

### 2.3. Failure → Recovery → Success (`template_recovery`)
*   **Description**: The "gold mine" for economic value. The agent fails, detects it, corrects, and succeeds.
*   **Structure**: `Approach` -> `Interact` -> `FailureEvent` -> `Pause` -> `RecoveryAction` -> `Interact` -> `Complete`.
*   **Signals**: `RecoveryTag`, high complexity, high time-cost but successful outcome.
*   **Use Case**: Validating `RecoveryNode` proposals and `OODTag` handling.

### 2.4. Long-Horizon Mixed (`template_mixed`)
*   **Description**: Chained subtasks with varying outcomes.
*   **Structure**: `TaskA(Success)` -> `TaskB(Fail)` -> `TaskB(Retry->Success)` -> `TaskC(Success)`.
*   **Use Case**: Stress testing the `SegmentationEngine` and `TaskGraphRefiner`'s ability to handle continuous streams.

## 3. Coverage Spec

For Stage 2.1–2.4 to function correctly, the input data must meet these coverage requirements per "hardening batch".

### 3.1. Distribution per Task Family
For each task family (e.g., `drawer_open`), a standard batch (N=1000 rollouts) should contain:

*   **60% Normal Success**: To solidify baseline statistics.
*   **20% Pure Failure**: To ensure fragility boundaries are mapped.
*   **15% Recovery**: To provide sufficient positive examples for robustness policies.
*   **5% Long-Horizon**: To test temporal segmentation and context retention.

### 3.2. Combinatorial Requirements
*   **Object Variance**: At least 3 distinct object geometries per task family (e.g., `drawer_handle_A`, `drawer_handle_B`, `drawer_knob_C`).
*   **Physics Variance**:
    *   Friction: Low, Medium, High.
    *   Mass: Light, Heavy.
*   **Semantic Signals**:
    *   Must include cases where `RiskTag` is high but outcome is Success (Lucky).
    *   Must include cases where `RiskTag` is low but outcome is Failure (Black Swan).

### 3.3. Required Semantic Signals
The SIMA-2 client (or the physics backend it wraps) must emit these ground-truth signals for validation:
1.  **Risk**: Instantaneous probability of failure.
2.  **Fragility**: Sensitivity of outcome to small perturbations.
3.  **Recovery**: Boolean flag indicating a correction maneuver is active.
4.  **Novelty**: Distance from training distribution (for `OODTag`).
5.  **OOD**: Boolean flag for Out-Of-Distribution states.
