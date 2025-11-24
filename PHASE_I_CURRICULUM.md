# Phase I Full Training Curriculum

## Philosophy: Layer-wise Stability
We follow a strict "bottom-up" freezing schedule. We do **not** train everything end-to-end initially. Instability in the vision backbone would catastrophically destabilize the RL policy.

## Stage 1: Vision Foundation
*   **Component:** RegNet + BiFPN.
*   **Data Sources:**
    *   `results/policy_datasets_raw/` (Stage 1 Random Policy).
    *   `results/sima2_stress/` (Hard negatives from Stage 5).
    *   `data/ontology/` (Object class definitions).
*   **Task:** Contrastive Learning + Reconstruction.
*   **Outcome:** Frozen `vision_backbone.pth`.
*   **Verification:** Linear probe accuracy > 90% on object ID.

## Stage 2: Temporal Dynamics
*   **Component:** Spatial RNN (ConvGRU).
*   **Data Sources:**
    *   `results/policy_datasets_segmented/` (Stage 2 Heuristic Policy).
    *   `ConditionVector` + `TrustMatrix` (from `isaac_adapter`/`ros_bridge`).
*   **Task:** Forward Dynamics Prediction ($z_{t+1} | z_t, a_t$).
*   **Input:** Frozen features from Stage 1.
*   **Outcome:** Frozen `spatial_rnn.pth`.
*   **Verification:** MSE of predicted latent < threshold.

## Stage 3: Semantic Segmentation
*   **Component:** SIMA-2 Neural Segmenter.
*   **Data Sources:**
    *   `results/policy_datasets_segmented/` (Silver labels from Heuristic Segmenter).
    *   `results/sima2_stress/` (Failure/Recovery examples).
    *   `TrustMatrix` (Weighting signal).
*   **Task:** Supervised learning (Boundary + Classification).
*   **Input:** Frozen features from Stage 1 & 2.
*   **Outcome:** `neural_segmenter.pth`.
*   **Verification:** F1 Score > 0.85 against Heuristic labels.

## Stage 4: Policy Warm-Start (Behavior Cloning)
*   **Component:** Hydra Policy (Trunk + Heads).
*   **Data Sources:**
    *   `results/policy_datasets_segmented/` (Successful trajectories only).
    *   Filtered by `EconomicLearner` (positive ROI only).
*   **Task:** Behavior Cloning (BC).
*   **Input:** Frozen features from Stage 1 & 2.
*   **Outcome:** Pre-trained Policy weights.

## Stage 5: Policy Refinement (RL)
*   **Component:** Hydra Policy.
*   **Data:** Online interaction (Isaac Sim).
*   **Task:** SAC with Multi-Objective Loss.
*   **Input:** Frozen features from Stage 1 & 2.
*   **Outcome:** Final Phase I Policy.

## Joint-Finetuning Constraints
*   **Forbidden:** Backpropagating RL loss into the Vision Backbone during Stage 5. (Causes "catastrophic forgetting" of visual features not relevant to the immediate reward).
*   **Allowed:** Fine-tuning the Spatial RNN with a very low learning rate ($10^{-5}$) *only* after Policy performance plateaus.

## Isaac Rollout Requirements
To support this curriculum, we need a massive data generation campaign:
1.  **Random Policy Rollouts:** 50k eps (For Vision/Dynamics). Diversity is key.
2.  **Heuristic Policy Rollouts:** 50k eps (For Segmentation/BC). Success is key.
3.  **Corner Case Rollouts:** 10k eps (Recoveries, failures) to train the Segmenter's failure head.
