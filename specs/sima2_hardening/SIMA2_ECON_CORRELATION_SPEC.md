# SIMA-2 Hardening: Econ Correlation & Trust Calibration

**Status**: Draft
**Owner**: SIMA-2 Hardening Team
**Context**: Stage 2 Pipeline Hardening

## 1. Overview

SIMA-2 semantics (Risk, Fragility, OOD) are only valuable if they predict economic outcomes. This spec defines the "Trust Calibration" loop: correlating Stage 2 tags with Stage 3 EconVectors to determine which semantic signals are "real" and which are noise.

## 2. Correlation Analysis Pipeline

We introduce a new offline analysis module, the `EconCorrelator`, which runs periodically on the `Datapack` store.

### 2.1. Inputs
1.  **Stage 2 Tags**: `RiskTag`, `FragilityTag`, `RecoveryTag`, `OODTag`.
2.  **Stage 3 EconVectors**:
    *   `MPL` (Marginal Product of Labor): Task success / Time.
    *   `Wh` (Watt-hours): Energy consumed.
    *   `Damage`: Cost of hardware wear/tear.
    *   `WageParity`: Economic viability vs human labor.

### 2.2. Computed Statistics
The pipeline computes the conditional expectation of EconVectors given Tags.

*   **Risk Premium**: $E[\text{Damage} | \text{RiskTag=High}] / E[\text{Damage} | \text{RiskTag=Low}]$
    *   *Target*: > 2.0 (High risk should mean 2x damage/failure cost).
*   **Recovery Value**: $E[\text{MPL} | \text{RecoveryTag=True}] - E[\text{MPL} | \text{RecoveryTag=False}]$
    *   *Target*: Positive (Recoveries should save the episode's value).
*   **OOD Penalty**: $E[\text{SuccessRate} | \text{OODTag=High}]$
    *   *Target*: Low (OOD should correlate with lower reliability).

### 2.3. Output Artifact: `TrustMatrix`
A JSON document defining the "Trust Level" (0.0 to 1.0) for each semantic signal.

```json
{
  "RiskTag": {
    "trust_level": 0.85,
    "correlation_strength": "strong",
    "economic_impact": "high_damage_predictor"
  },
  "OODTag": {
    "trust_level": 0.4,
    "correlation_strength": "weak",
    "note": "OOD currently flagging too many benign lighting changes"
  }
}
```

## 3. Trust Thresholds & Policy Flow-Back

The `TrustMatrix` dictates how policies consume SIMA-2 data. We define three trust tiers.

### 3.1. Tier 1: Trusted (Trust > 0.8)
*   **Definition**: The tag is a proven predictor of economic reality.
*   **Policy Action**:
    *   **Sampler**: `DataPackRLSampler` *heavily* biases sampling based on this tag (e.g., oversample `RecoveryTag` episodes by 5x).
    *   **Orchestrator**: `SemanticOrchestrator` can trigger `SafetyStop` based solely on this tag.

### 3.2. Tier 2: Provisional (0.5 < Trust < 0.8)
*   **Definition**: The tag shows promise but has noise or weak correlation.
*   **Policy Action**:
    *   **Sampler**: Moderate sampling bias (1.5x).
    *   **Orchestrator**: Advisory only. "Warning: High Risk detected" but no auto-stop.
    *   **Datapack Auditor**: Flags these datapacks for human review.

### 3.3. Tier 3: Untrusted (Trust < 0.5)
*   **Definition**: The tag is uncorrelated or noisy.
*   **Policy Action**:
    *   **Ignored**: The system behaves as if the tag doesn't exist.
    *   **Debug**: `EconCorrelator` emits a `CalibrationAlert` to the engineering team to fix the detector.

## 4. Integration Points

### 4.1. `DatapackAuditor`
*   **Current**: Checks basic schema validity.
*   **New**: Checks `TrustMatrix`. If a Datapack relies heavily on "Untrusted" tags for its `QualityScore`, the Auditor downgrades the score.

### 4.2. `SemanticOrchestrator`
*   **Current**: Blindly accepts proposals.
*   **New**: Filters proposals by Trust.
    *   *Rule*: "Only accept `RefinementProposal` if source tag has Trust > 0.6."

### 4.3. `DataPackRLSampler`
*   **Current**: Random or FIFO.
*   **New**: `TrustWeightedSampling`.
    *   $P(sample) \propto \sum (Tag_i \times Trust_i \times EconImpact_i)$
