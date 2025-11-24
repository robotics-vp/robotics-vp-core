# RunPod Readiness Report

**Status**: ✅ READY FOR DEPLOYMENT
**Date**: 2025-11-24

## 1. Mandatory RunPod Settings
When launching a pod, ensure the following:

*   **GPU**: NVIDIA A100 (80GB) recommended, or RTX 3090/4090 (24GB) minimum.
*   **Image**: `nvidia/cuda:12.1.0-runtime-ubuntu22.04` (or similar CUDA 12.x base).
*   **Volume**: Mount `/workspace` (min 50GB) for persistent storage.
*   **Env Vars**: `PYTHONUNBUFFERED=1` (crucial for seeing logs in real-time).

## 2. Hardware Feasibility
*   **RTX 3090/4090 (24GB)**:
    *   **Training**: Feasible with `batch_size=16` (Vision) or `batch_size=32` (others) + AMP enabled.
    *   **Inference**: Feasible (DemoPolicy uses < 4GB VRAM).
*   **A100 (80GB)**:
    *   **Training**: Can scale to `batch_size=128`+ for Vision Backbone.

## 3. Configuration & Safety
*   **AMP**: Enabled by default in all scripts via `--use-mixed-precision`.
*   **Checkpointing**: Enabled in `RegNet`, `BiFPN`, `SpatialRNN` to save VRAM.
*   **OOM Recovery**: All training loops wrapped in `run_with_oom_recovery`.
*   **Neural Flags**:
    *   `use_neural_vision`: Respects flag (loads RegNet or Stub).
    *   `use_neural_spatial_rnn`: Respects flag (loads ConvGRU or Stub).
    *   `use_neural_sima2`: Respects flag (loads Segmenter or Stub).
    *   `use_neural_hydra`: Respects flag (loads HydraPolicy or Stub).

## 4. Launch Checklist
Once the pod is running and you have SSH'd in:

1.  **Setup Repo**:
    ```bash
    cd /workspace
    git clone <repo_url> robotics-vp-core
    cd robotics-vp-core
    pip install -r requirements.txt
    ```

2.  **Verify GPU**:
    ```bash
    python3 scripts/check_gpu_env.py
    # Expect: CUDA Available: True, Device Count >= 1
    ```

3.  **Run Smoke Test**:
    ```bash
    python3 scripts/smoke_test_stage6_end_to_end.py
    # Expect: "All components passed smoke test!"
    ```

4.  **Start Monitoring (Background)**:
    ```bash
    nohup python3 scripts/monitor_gpu_usage.py --interval 5 > gpu_monitor.log 2>&1 &
    ```

5.  **Run Full Training**:
    ```bash
    python3 scripts/run_stage6_train_all.py --seed 0
    ```

6.  **Visualize Results**:
    ```bash
    python3 scripts/visualize_training_and_demo_logs.py \
        --training-glob "results/training_logs/*.jsonl" \
        --output-dir "results/plots"
    ```

## 5. Troubleshooting
*   **CUDA OOM**:
    *   Check `gpu_monitor.log` to see peak usage.
    *   Ensure `--use-mixed-precision` is passed (it is by default in `run_stage6_train_all.py`).
    *   Reduce batch size in individual scripts if needed.
*   **Slow Training**:
    *   Check if `num_workers` in DataLoaders is too high/low (default is usually 0 or 4).
    *   Ensure `pin_memory=True` (enabled in datasets).

## 6. Artifacts
*   **Checkpoints**: Saved to `results/checkpoints/`.
*   **Logs**: Saved to `results/training_logs/`.
