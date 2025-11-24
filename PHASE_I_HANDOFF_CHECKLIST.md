# Phase I Handoff Checklist

This checklist defines the **Entry Criteria** for Phase I Neuralization. Do not begin training neural components until all items are checked and verified by the Stage 5 Audit.

## 1. Core Infrastructure Readiness
- [ ] **Ingestion Pipeline:** `ros_bridge` and `isaac_adapter` are successfully producing `RawRollout` objects with valid timestamps and frame IDs.
- [ ] **Conditioning:** `ConditionVector` is correctly propagated through the system and logged in every rollout.
- [ ] **Economics:** `EconDomainAdapter` is calibrated for:
    - [ ] `sim_isaac`
    - [ ] `real_ros` (if applicable)
- [ ] **Audit Status:** `ready_for_neuralization == true` in the latest Audit Report.

## 2. Data Availability & Quality
- [ ] **Volume:** At least **100k** frames of raw vision data available in `results/policy_datasets_raw/`.
- [ ] **Diversity:** Dataset includes significant variance in lighting, texture, and object pose (verified by `compare_modalities.py`).
- [ ] **Stress Tests:** SIMA-2 Stress Tests run successfully with **≥ 10k** rollouts (`results/sima2_stress/`).
- [ ] **Tags:** `OODTag` and `RecoveryTag` are present in the stress dataset (non-zero count).

## 3. Semantic Truth
- [ ] **Heuristic Baseline:** `HeuristicSegmenter` achieves > 90% success rate on "easy" standard tasks.
- [ ] **TrustMatrix:**
    - [ ] Matrix is present and non-degenerate (not all zeros or ones).
    - [ ] High confidence scores correlate with successful task completion.
- [ ] **Ontology:** Object class definitions in `data/ontology/` match the assets used in Isaac Sim.

## 4. Safety & Bounds
- [ ] **Phase H:** `EconomicLearner` bounds ($\pm 20\%$ ROI EMA) are active and enforced.
- [ ] **Exploration:** Global exploration cap is set to 20%.
- [ ] **Advisory:** All neural flags are set to `advisory_only=True` in `config/neural_config.yaml`.

## 5. "Do Not Start" Gates
*   **STOP IF:** `EconDomainAdapter` shows > 50% discrepancy between Sim and Real rewards.
*   **STOP IF:** `TrustMatrix` is empty or uncalibrated.
*   **STOP IF:** `HeuristicSegmenter` is failing on basic approach/grasp primitives.
