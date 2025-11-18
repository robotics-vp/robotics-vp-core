# SIMA-2 Hardening: Implementation Handoff for Codex

This document outlines the implementation order for the SIMA-2 Hardening Design Pack. Follow this sequence to transform the current stubbed integration into a trusted semantic data pump.

## Phase 1: The Data Pump (Client & Task Library)
**Spec**: [`SIMA2_HARDENING_TASK_LIBRARY.md`](./SIMA2_HARDENING_TASK_LIBRARY.md)

*   **Context**: Currently, `src/sima2/client.py` is a deterministic stub that returns a single, repetitive "drawer open" trace.
*   **Why**: The downstream pipeline (Ontology, Task Graph) needs rich, varied input to be tested properly. We need to see failures, recoveries, and different task types to validate that our semantic logic actually works.
*   **To-Do**:
    1.  Update `Sima2Client` to support the full **Task Taxonomy** (drawer, dish, wipe, etc.).
    2.  Implement the **Rollout Templates** (success, failure, recovery, mixed).
    3.  Ensure the client emits the required ground-truth signals (risk, fragility) for us to validate against later.

## Phase 2: The Segmenter (Segmentation Engine)
**Spec**: [`SIMA2_SEGMENTATION_SPEC.md`](./SIMA2_SEGMENTATION_SPEC.md)

*   **Context**: The current pipeline assumes "1 rollout = 1 primitive". In reality, a single rollout contains a sequence of actions (approach, grasp, pull).
*   **Why**: To detect complex behaviors like "Recovery" (fixing a mistake), we must be able to see the sequence `Action(Fail) -> Correction -> Action(Success)` within a single episode. The Segmentation Engine provides this temporal resolution.
*   **To-Do**:
    1.  Create `src/sima2/segmentation_engine.py`.
    2.  Implement the logic to slice continuous streams into `Segment` objects based on the spec.
    3.  Wire this into `SemanticPrimitiveExtractor` so it consumes segments instead of raw rollouts.

## Phase 3: The Semantic Tagger (OOD & Recovery)
**Spec**: [`SIMA2_OOD_AND_RECOVERY_SPEC.md`](./SIMA2_OOD_AND_RECOVERY_SPEC.md)

*   **Context**: We have basic tags, but we lack the specific semantics for "Weirdness" (OOD) and "Resilience" (Recovery).
*   **Why**: These are the critical signals for the Economics stack.
    *   **OODTag**: Tells us when the model is flying blind (High Risk).
    *   **RecoveryTag**: Tells us when the model saved a failing task (High Value).
    *   Without these, the Econ vectors (MPL, Damage) are just uncontextualized numbers.
*   **To-Do**:
    1.  Extend `SemanticTagPropagator` to generate `OODTag` and `RecoveryTag`.
    2.  Implement the **Robustness Invariants** (e.g., "Recovery Imperative") in `TaskGraphRefiner`.
    3.  Add the `OOD` and `Recovery` smoke scenarios to the test suite.

## Phase 4: The Trust Loop (Econ Correlation)
**Spec**: [`SIMA2_ECON_CORRELATION_SPEC.md`](./SIMA2_ECON_CORRELATION_SPEC.md)

*   **Context**: Currently, a "High Risk" tag is just a label. We don't know if it actually predicts real-world damage or cost.
*   **Why**: We need to *calibrate* our trust. If `RiskTag` correlates with high `Damage` cost, we trust it. If it doesn't, we ignore it. This feedback loop is what makes the system "Econ-aligned".
*   **To-Do**:
    1.  Create `src/analytics/econ_correlator.py`.
    2.  Implement the logic to compute conditional expectations ($E[\text{Cost} | \text{Tag}]$).
    3.  Generate the `TrustMatrix` artifact.
    4.  Update `DataPackRLSampler` to read the `TrustMatrix` and bias sampling accordingly.

## Phase 5: Scale & Verify (Stress Tests)
**Spec**: [`SIMA2_SCALING_AND_STRESS_TESTS.md`](./SIMA2_SCALING_AND_STRESS_TESTS.md)

*   **Context**: Our current tests run 1-10 episodes. Production needs to handle thousands.
*   **Why**: We need to ensure that a flood of data doesn't:
    *   OOM the worker nodes.
    *   Create millions of duplicate nodes in the Ontology.
    *   Cause the Task Graph to explode in depth.
*   **To-Do**:
    1.  Create `scripts/stress_test_sima2_pipeline.py`.
    2.  Implement the "10k Rollout" benchmark.
    3.  Add assertions for memory footprint and ontology write amplification.
