# Phase I Neuralization Overview

## Executive Summary
Phase I Neuralization marks the transition of the robotics economics model from a heuristic, stub-based architecture (Stage 5 entry) to a fully learned, neural-based system. This phase focuses on replacing deterministic "mock" components with performant neural networks while maintaining the strict economic and semantic invariants established in Phase H.

The core objective is to enable the robot to *learn* the representations and policies that were previously hardcoded, unlocking generalization to new tasks and environments.

## Core Components

### 1. Vision Neuralization (RegNet + BiFPN)
*   **Goal:** Replace hash-based/random vision stubs with a robust perception backbone.
*   **Architecture:** RegNet (Y/Z variants) backbone with BiFPN feature fusion.
*   **Training:** Self-supervised contrastive learning (SimCLR/MoCo style) + reconstruction auxiliary losses.
*   **Output:** A rich, multi-scale feature pyramid `P3, P4, P5` for downstream consumers.

### 2. Spatial Memory (Spatial RNN)
*   **Goal:** Provide temporal context and object permanence.
*   **Architecture:** Comparison of S4D, ConvGRU, and ConvLSTM.
*   **Function:** Takes vision features $z_t$ and predicts latent dynamics $z_{t+1} | z_t, a_t$.
*   **Integration:** Feeds into the Policy Trunk and SIMA-2 Segmenter.

### 3. SIMA-2 Neural Segmenter
*   **Goal:** Replace `HeuristicSegmenter` with a learned boundary detector.
*   **Input:** Robot state + Spatial RNN features.
*   **Output:** Probability of segment boundary (start/end of primitives).
*   **Training:** Bootstrapped from heuristic labels, refined via human-in-the-loop or weak supervision.

### 4. Hydra Policy Neuralization
*   **Goal:** Train the multi-head Actor/Critic system.
*   **Architecture:** Shared Trunk (Vision + State + Condition) -> Skill-Specific Heads.
*   **Mechanism:** `ConditionVector.skill_mode` deterministically selects the active head.
*   **Loss:** Multi-objective RL (SAC/PPO) optimizing for Economic Value (EV) and Task Success.

## Integration Strategy
The neuralization process follows a strict curriculum to ensure stability:
1.  **Vision First:** Freeze the rest, train the eyes.
2.  **Memory Second:** Train the RNN on frozen vision features.
3.  **Segmentation Third:** Train the segmenter on vision+memory.
4.  **Policy Last:** Train the Hydra heads using the stable upstream representations.

## Stage 5 Dependencies & Integration
Phase I builds directly upon the Stage 5 Chunk 1 architecture. It assumes the following components are active and calibrated:

### Core Infrastructure
*   **Ingestion:** `ros_bridge.py` (Real) and `isaac_adapter.py` (Sim) are the canonical sources of `RawRollout` data.
*   **Conditioning:** `ConditionVector` is the unified channel for all policy modulation (Econ, SIMA-2, TFD).
*   **Economics:** `EconDomainAdapter` must be calibrated for each domain (Sim/Real) to normalize rewards.

### Semantic Truth
*   **SIMA-2:** The `HeuristicSegmenter` + `TrustMatrix` provide the "Silver Labels" for training.
*   **Tags:** `OODTag` and `RecoveryTag` are critical for filtering training data and supervising the failure heads.

### Constraints
*   **Phase H Bounds:** The `EconomicLearner` continues to enforce ROI EMA bounds ($\pm 20\%$) and exploration caps.
*   **Advisory Flag:** Neural outputs remain advisory-only until Phase II.

## Constraints & Invariants
*   **Determinism:** All forward passes must remain deterministic for a given seed/state.
*   **JSON-Safety:** All outputs must be serializable (or have adapters) for the Econ/Audit loop.
*   **Advisory Nature:** Neural outputs guide the policy but do not override safety/economic constraints directly.
