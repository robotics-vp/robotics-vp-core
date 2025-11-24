# Vision Neuralization Design

## Architecture Specification

### Backbone: RegNet
We will employ a **RegNetY** backbone (specifically `RegNetY-3.2GF` or `RegNetY-800MF` depending on compute constraints) for its favorable trade-off between accuracy and inference latency.
*   **Input:** RGB Images $(B, C, H, W)$, typically $224 \times 224$ or $256 \times 256$.
*   **Stages:** 4 stages with increasing channel widths.
*   **Output:** Feature maps $\{C3, C4, C5\}$ corresponding to strides $\{8, 16, 32\}$.

### Feature Fusion: BiFPN
To aggregate multi-scale features, we replace the simple averaging stub with a **Bi-directional Feature Pyramid Network (BiFPN)**.
*   **Inputs:** $\{C3, C4, C5\}$ from RegNet.
*   **Operations:** Top-down and bottom-up pathways with learnable weights.
*   **Fusion:** Fast normalized fusion: $O = \frac{\sum w_i \cdot I_i}{\epsilon + \sum w_i}$.
*   **Output:** Unified feature pyramid $\{P3, P4, P5\}$ with fixed channel depth (e.g., $D=128$).

### Projection Head
For contrastive training, a lightweight MLP projection head maps the flattened or pooled $P5$ features to a latent space $z_{proj}$.
*   **Structure:** `Linear -> BN -> ReLU -> Linear`.
*   **Dim:** 128 or 256.

## Training Procedure

### Hybrid Objective
We utilize a hybrid loss combining self-supervised contrastive learning with pixel-level reconstruction to encourage both semantic separability and spatial awareness.

$$ L_{total} = \lambda_{cont} L_{contrastive} + \lambda_{rec} L_{reconstruction} $$

#### 1. Contrastive Loss (SimCLR/MoCo)
*   **Data:** Pairs of augmented views $(x_i, x_j)$ of the same scene.
*   **Loss:** InfoNCE.
    $$ L_{i,j} = -\log \frac{\exp(sim(z_i, z_j)/\tau)}{\sum_{k \neq i} \exp(sim(z_i, z_k)/\tau)} $$
*   **Goal:** Invariant representation under lighting, noise, and viewpoint changes.

#### 2. Reconstruction Loss
*   **Decoder:** A lightweight convolutional decoder mirroring the backbone.
*   **Loss:** MSE or L1 between input image and reconstructed image.
*   **Goal:** Ensure features retain spatial details necessary for manipulation (e.g., object pose).

## Isaac Real-to-Sim Augmentation
To bridge the gap between Isaac Sim and the real world, we apply aggressive domain randomization during training:

1.  **Photometric:**
    *   Random brightness/contrast/saturation.
    *   Gaussian noise and blur.
    *   Color jitter.
2.  **Geometric:**
    *   Random crop and resize.
    *   Horizontal flip (if symmetric task).
    *   Small rotations ($\pm 15^\circ$).
3.  **Isaac-Specific:**
    *   **Texture Randomization:** Swap object/background textures in Sim.
    *   **Lighting Randomization:** Vary light position, intensity, and color.
    *   **Camera Noise:** Simulate sensor noise and lens distortion.

## Stage 5 Dependencies
*   **Ingestion:** `isaac_adapter.py` provides the canonical `RawRollout` structure for Sim data.
*   **Config:** `vision.use_neural_backbone=False` (initially) while training.
*   **Artifacts:**
    *   `results/policy_datasets_sima2_stress/` (Source of hard negatives/stress cases).
    *   `data/ontology/` (For object class definitions in auxiliary tasks).

## Implementation Plan
1.  **Dataset Collection:**
    *   **Source A:** Stage 1 Raw Datapacks (Random Policy).
    *   **Source B:** Stage 2 Segmented Datapacks (Heuristic Policy).
    *   **Source C:** SIMA-2 Stress Outputs (`results/sima2_stress/*.jsonl`) - Critical for robustness.
    *   **Ingestion:** All data must pass through `ros_bridge` or `isaac_adapter` to ensure `RawRollout` format.
2.  **Pre-training:** Train RegNet+BiFPN on this dataset using the Hybrid Objective.
3.  **Evaluation:** Linear probe on a small labeled subset (object classification/pose estimation) to verify feature quality.
4.  **Integration:** Replace `regnet_backbone.py` stub with the pre-trained weights.
