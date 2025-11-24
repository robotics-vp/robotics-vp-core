# Phase I Implementation Playbook

This playbook outlines the concrete steps, scripts, and metrics required to implement the Phase I Neuralization designs.

## 1. Training Scripts
Codex should implement the following scripts in `scripts/`:

### `scripts/train_vision_backbone.py`
*   **Purpose:** Train RegNet+BiFPN via contrastive learning.
*   **Inputs:**
    *   `--dataset_path`: Path to `results/policy_datasets_raw/`
    *   `--config`: `config/vision_training.yaml`
*   **Outputs:**
    *   `checkpoints/vision/regnet_bifpn_v1.pt`
    *   `results/vision_metrics.json` (Linear probe accuracy)

### `scripts/train_spatial_rnn.py`
*   **Purpose:** Train ConvGRU for forward dynamics prediction.
*   **Inputs:**
    *   `--vision_ckpt`: Path to frozen vision backbone.
    *   `--dataset_path`: `results/policy_datasets_segmented/`
*   **Outputs:**
    *   `checkpoints/rnn/spatial_rnn_v1.pt`
    *   `results/rnn_metrics.json` (Latent prediction MSE)

### `scripts/train_sima2_segmenter.py`
*   **Purpose:** Train Neural Segmenter on heuristic labels + tags.
*   **Inputs:**
    *   `--vision_ckpt`: Frozen vision backbone.
    *   `--rnn_ckpt`: Frozen RNN.
    *   `--dataset_path`: `results/policy_datasets_segmented/` + `results/sima2_stress/`
    *   `--trust_matrix`: Path to `trust_matrix.json`
*   **Outputs:**
    *   `checkpoints/segmenter/neural_segmenter_v1.pt`
    *   `results/segmenter_metrics.json` (F1 Score, Boundary Accuracy)

### `scripts/train_hydra_policy.py`
*   **Purpose:** Train Hydra Actor/Critic heads.
*   **Inputs:**
    *   `--vision_ckpt`: Frozen vision backbone.
    *   `--rnn_ckpt`: Frozen RNN.
    *   `--segmenter_ckpt`: Frozen Segmenter (optional, for auxiliary signals).
    *   `--config`: `config/hydra_training.yaml`
*   **Outputs:**
    *   `checkpoints/policy/hydra_policy_v1.pt`

## 2. Metrics & Tracking
All training runs must log the following to `wandb` or local JSON:

*   **Vision:**
    *   `contrastive_loss`
    *   `reconstruction_loss`
    *   `linear_probe_acc` (Object ID)
*   **RNN:**
    *   `dynamics_loss` (MSE)
    *   `temporal_consistency`
*   **Segmenter:**
    *   `boundary_f1`
    *   `classification_acc`
    *   `failure_prediction_auc`
*   **Policy:**
    *   `task_success_rate`
    *   `economic_value` (ROI)
    *   `energy_consumption` (Wh)
    *   `ood_recovery_rate`

## 3. Ablations
Run these comparisons to validate design choices:
1.  **Vision:** RegNet vs. ResNet50 (Baseline).
2.  **Memory:** ConvGRU vs. No-Memory (Frame-stacking).
3.  **Segmentation:** Neural vs. Heuristic (Latency & Consistency check).

## 4. Safety Conditions (The "Never" List)
*   **NEVER** change the reward math in `EconomicLearner` during neural training.
*   **NEVER** bypass `EconDomainAdapter` when calculating value.
*   **NEVER** disable Phase H bounds (ROI EMA, Budget) to boost exploration.
*   **ALWAYS** keep neural outputs as `advisory` until Phase II validation is complete.
