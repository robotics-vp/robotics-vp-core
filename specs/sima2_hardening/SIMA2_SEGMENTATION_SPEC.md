# SIMA-2 Hardening: Episodic Segmentation & Subtask Labeling

**Status**: Draft
**Owner**: SIMA-2 Hardening Team
**Context**: Stage 2 Pipeline Hardening

## 1. Segmentation Engine Overview

The `SegmentationEngine` is a new module responsible for breaking down continuous, long-horizon SIMA-2 rollouts into discrete, semantically meaningful sub-episodes. This bridges the gap between raw continuous logs (DYNA-style) and the discrete event logic required by the Ontology and Task Graph.

### 1.1. Core Responsibilities
1.  **Temporal Boundary Detection**: Identifying start `t_start` and end `t_end` of distinct actions.
2.  **Semantic Labeling**: Assigning a high-level label (e.g., `grasp`, `approach`) to each segment.
3.  **Outcome Classification**: Determining if the segment ended in `success`, `failure`, or `retry`.

## 2. API Contract

### 2.1. Inputs
The engine consumes a stream of raw SIMA-2 states.

```json
{
  "stream_id": "rollout_123",
  "frames": [
    {
      "timestamp": 1001,
      "robot_state": { "joint_positions": [...], "ee_pose": [...] },
      "object_states": { "drawer": { "pose": [...] }, "handle": { "pose": [...] } },
      "primitives_active": ["move_arm"],  // Low-level controller signals
      "vision_embedding": [...]           // Optional: CLIP/ResNet embedding
    },
    ...
  ]
}
```

### 2.2. Outputs
The engine produces a list of `Segment` objects.

```json
[
  {
    "segment_id": "seg_001",
    "label": "approach_handle",
    "start_t": 1001,
    "end_t": 1045,
    "outcome": "success",
    "confidence": 0.98,
    "metadata": {
      "max_velocity": 0.5,
      "distance_covered": 0.2
    }
  },
  {
    "segment_id": "seg_002",
    "label": "grasp_handle",
    "start_t": 1046,
    "end_t": 1060,
    "outcome": "failure",
    "failure_reason": "gripper_slip",
    "confidence": 0.85
  }
]
```

## 3. Segment Taxonomy

### 3.1. Standard Segments
*   `approach`: Moving EE towards an object of interest.
*   `grasp`: Closing gripper on an object.
*   `pull/push`: Applying force to articulate a joint.
*   `lift`: Vertical transport against gravity.
*   `place`: Releasing object on a surface.
*   `transport`: Moving object while grasped.

### 3.2. Anomaly Segments
*   `collision`: Unintended contact with high force.
*   `slip`: Object moving relative to gripper while grasped.
*   `idle`: No significant motion for threshold time (may indicate planning or stuck).
*   `recovery_maneuver`: Explicit corrective motion (e.g., backing up after collision).

## 4. Integration with Stage 2 Pipeline

### 4.1. Feed into `SemanticPrimitiveExtractor` (Stage 2.1)
The `SemanticPrimitiveExtractor` currently relies on pre-labeled primitives. It will be upgraded to consume `SegmentationEngine` outputs.
*   **Mapping**: `Segment` -> `SemanticPrimitive`.
*   **Logic**: Each `Segment` becomes a candidate primitive. If `confidence` < threshold, it is flagged as `ambiguous`.

### 4.2. Feed into `OntologyUpdateEngine` (Stage 2.2)
*   **Affordance Discovery**: If a `pull` segment succeeds on a `drawer` object, the engine proposes adding `pullable` affordance to `drawer` in the Ontology.
*   **Fragility Mapping**: If `grasp` segments fail frequently on `handle_type_B`, the engine proposes a `FragilityNode` for that object-action pair.

### 4.3. Feed into `TaskGraphRefiner` (Stage 2.3)
*   **Structure Learning**: The sequence of segments (e.g., `approach` -> `grasp` -> `pull`) validates or refines the `TaskGraph` edges.
*   **Recovery Edges**: A sequence like `grasp(fail)` -> `regroup` -> `grasp(success)` triggers the insertion of a "Recovery Edge" in the Task Graph, formally codifying the recovery strategy.

### 4.4. Feed into `SemanticTagPropagator` (Stage 2.4)
*   **OODTag**: If a segment is labeled `unknown` or has very low confidence despite high motion, generate an `OODTag`.
*   **RecoveryTag**: Explicitly linked to `recovery_maneuver` segments.

## 5. Example Scenarios

### 5.1. Drawer Opening (Success)
*   `t=0-50`: `approach` (handle) -> Success
*   `t=51-60`: `grasp` (handle) -> Success
*   `t=61-100`: `pull` (drawer) -> Success

### 5.2. Drawer Opening (Slip & Retry)
*   `t=0-50`: `approach` (handle) -> Success
*   `t=51-60`: `grasp` (handle) -> Failure (Slip)
*   `t=61-70`: `recovery_maneuver` (open gripper, back off) -> Success
*   `t=71-80`: `approach` (handle) -> Success
*   `t=81-90`: `grasp` (handle) -> Success
*   **Result**: Generates `RecoveryTag` and potentially `FragilityNode` for the first grasp pose.
