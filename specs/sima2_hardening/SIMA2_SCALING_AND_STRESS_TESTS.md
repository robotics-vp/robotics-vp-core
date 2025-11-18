# SIMA-2 Hardening: Scaling & Stress Smokes

**Status**: Draft
**Owner**: SIMA-2 Hardening Team
**Context**: Stage 2 Pipeline Hardening

## 1. Overview

Current smoke tests (`smoke_test_sima2_client.py`) only validate basic connectivity with N=1 or N=10 episodes. To harden the pipeline for production, we need large-scale synthetic stress tests that push the system to its limits in terms of volume, noise, and complexity.

## 2. Large-Scale Synthetic Stress Tests

We define a new test suite `stress_test_sima2_pipeline.py` that runs in a CI/CD "nightly" mode.

### 2.1. The "10k Rollout" Benchmark
*   **Goal**: Process 10,000 synthetic rollouts in < 1 hour.
*   **Composition**:
    *   40% `drawer_open` (High friction, frequent failures).
    *   40% `dish_place` (Precision tasks).
    *   20% `random_noise` (Garbage inputs to test robustness).
*   **Backend**: Uses `Sima2Client` in "fast-forward" mode (no sleep, pure CPU generation).

### 2.2. Worst-Case Sequences
*   **Rapid Failure Storm**: 1000 consecutive failures.
    *   *Assert*: Pipeline does not crash; `FragilityNode` creation is throttled (deduplicated); Memory usage remains stable.
*   **Tag Explosion**: Every primitive emits every possible tag.
    *   *Assert*: `SemanticTagPropagator` handles high cardinality without OOM.
*   **Deep Recursion**: A task graph depth of 50 levels (e.g., `Task` -> `Subtask` -> ...).
    *   *Assert*: `TaskGraphRefiner` handles deep nesting gracefully.

## 3. Metrics & Assertions

The stress test suite must assert the following quantitative metrics.

### 3.1. Performance
| Metric | Threshold | Rationale |
| :--- | :--- | :--- |
| **Throughput** | > 100 Hz (rollouts/sec) | Pipeline must keep up with accelerated sim. |
| **Latency (P99)** | < 50ms per primitive | Real-time tagging requirement. |
| **Memory Footprint** | < 2GB RSS | Must fit on standard worker nodes. |

### 3.2. Ontology Health
| Metric | Threshold | Rationale |
| :--- | :--- | :--- |
| **Write Amplification** | < 1.5x | 1 rollout shouldn't trigger 100 DB writes. |
| **Orphan Nodes** | 0 | Every node must be connected to the graph. |
| **Proposal Acceptance** | > 80% (for valid data) | Valid proposals shouldn't be dropped. |

### 3.3. Tag Distribution Sanity
| Metric | Threshold | Rationale |
| :--- | :--- | :--- |
| **RiskTag Saturation** | < 30% of total | If everything is risky, nothing is. |
| **RecoveryTag Precision** | > 90% | Recovery must imply prior failure. |
| **OODTag Rate** | < 5% (baseline) | System shouldn't be perpetually confused. |

## 4. Dataset Recipes

To support these tests, we define standard dataset generation recipes.

### 4.1. `dataset_stress_mix_v1`
```python
{
  "total_episodes": 10000,
  "distribution": {
    "drawer_open": 0.4,
    "dish_place": 0.4,
    "noise": 0.2
  },
  "params": {
    "failure_rate": 0.3,
    "recovery_rate": 0.1,
    "ood_rate": 0.05
  }
}
```

### 4.2. `dataset_edge_cases_v1`
*   Empty rollouts.
*   Rollouts with 1M steps.
*   Rollouts with invalid UTF-8 strings.
*   Rollouts with future timestamps.
