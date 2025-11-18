# SIMA-2 Hardening: OOD & Recovery Semantics

**Status**: Draft
**Owner**: SIMA-2 Hardening Team
**Context**: Stage 2 Pipeline Hardening

## 1. Tag Vocabulary Extension

We extend the Stage 2.4 semantic tag vocabulary to explicitly handle Out-Of-Distribution (OOD) events and Recovery behaviors. These tags are critical for the Econ stack to price risk and resilience correctly.

### 1.1. `OODTag` (Out-Of-Distribution)
*   **Definition**: Marks a segment or state as significantly deviating from the training distribution or expected physics.
*   **Criteria**:
    *   **Visual**: CLIP embedding distance > `threshold_visual` (e.g., new object, weird lighting).
    *   **Kinematic**: Joint velocities or torques exceeding 99th percentile of training data.
    *   **Temporal**: Segment duration > 3x mean duration for that primitive type.
*   **Payload**:
    ```json
    {
      "tag_type": "OODTag",
      "severity": 0.8,
      "source": "visual_anomaly_detector",
      "details": { "feature": "lighting", "deviation": 4.5 }
    }
    ```

### 1.2. `RecoveryTag`
*   **Definition**: Marks a sequence where the agent successfully recovers from a failure or high-risk state.
*   **Criteria**:
    *   Sequence pattern: `Primitive(Fail)` -> `CorrectionPrimitive` -> `Primitive(Success)`.
    *   State restoration: Object returns to `safe_pose` after `unstable_pose`.
*   **Payload**:
    ```json
    {
      "tag_type": "RecoveryTag",
      "value_add": "high",
      "correction_type": "re-grasp",
      "cost_wh": 12.5
    }
    ```

## 2. Robustness Invariants

These invariants must be enforced by the `TaskGraphRefiner` and `SemanticTagPropagator`.

### 2.1. The "Recovery Imperative"
> **Invariant**: If `RiskTag.level > threshold` AND `Outcome == Success` AND NO `RecoveryTag` exists, the system must flag this as "Lucky" (False Positive Safety).
>
> *Action*: `TaskGraphRefiner` proposes a `VerificationNode` to check if this success is reproducible.

### 2.2. The "OOD Containment"
> **Invariant**: Any `OODTag` with `severity > critical` must trigger a `SafetyStop` proposal in the Task Graph unless a `RecoveryTag` immediately follows it.
>
> *Rationale*: We cannot trust the policy in high-OOD regimes without proven recovery capabilities.

### 2.3. The "Fragility-Recovery Duality"
> **Invariant**: A `FragilityNode` (identifying a weak point) must eventually be paired with a `RecoveryNode` (the fix).
>
> *Action*: If `FragilityNode` exists for > 100 episodes without a paired `RecoveryNode`, `OntologyUpdateEngine` raises a `MissingSkill` alert.

## 3. Smoke & Stress Tests

### 3.1. Scenario A: The "Stumble and Recover" (Recovery Validation)
*   **Setup**: Robot attempts `pick_cup`.
*   **Injection**: Force a gripper slip at `t=10` (simulated failure).
*   **Expected Behavior**:
    1.  Client emits `slip` event.
    2.  Policy executes `regroup`.
    3.  Policy retries `pick_cup` -> Success.
    4.  **Pipeline Assertions**:
        *   `SegmentationEngine` outputs: `grasp(fail)`, `regroup`, `grasp(success)`.
        *   `SemanticTagPropagator` emits `RecoveryTag`.
        *   `EconSignals` computes positive `ResilienceScore`.

### 3.2. Scenario B: The "Alien Object" (OOD Validation)
*   **Setup**: Robot attempts `wipe_surface`.
*   **Injection**: Replace `sponge` with `unknown_mesh_v9` (visually distinct).
*   **Expected Behavior**:
    1.  Vision system signals high embedding distance.
    2.  **Pipeline Assertions**:
        *   `SemanticTagPropagator` emits `OODTag` with `source="visual"`.
        *   `TaskGraphRefiner` proposes `InspectionNode` (look closer).

### 3.3. Scenario C: The "Doom Loop" (Failure Cluster)
*   **Setup**: Robot attempts `drawer_open`.
*   **Injection**: Lock the drawer joint (infinite resistance).
*   **Expected Behavior**:
    1.  Repeated `pull` failures.
    2.  **Pipeline Assertions**:
        *   `SegmentationEngine` outputs sequence of `pull(fail)`.
        *   `OntologyUpdateEngine` creates `FragilityNode` with `reason="locked"`.
        *   After N failures, `TaskGraphRefiner` proposes `AbortNode`.
