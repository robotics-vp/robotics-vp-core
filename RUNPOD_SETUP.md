# RunPod Setup & Deployment Guide

This guide details how to deploy the Robotics V-P Economics Model on RunPod for Stage 6 training and demos.

## 1. Instance Selection
For Phase I training and demos, we recommend the following GPU instances:

*   **Recommended**: NVIDIA A100 (80GB) or A6000 (48GB)
    *   *Why*: Sufficient VRAM for large batch sizes and unoptimized datasets.
*   **Minimum**: NVIDIA RTX 3090 / 4090 (24GB)
    *   *Why*: Capable of running training with AMP and smaller batch sizes (e.g., 8-16).
*   **Image**: `nvidia/cuda:12.1.0-runtime-ubuntu22.04` (Base) or use our Dockerfile.

## 2. Docker Setup
We provide a dedicated Dockerfile for RunPod: `Dockerfile.runpod`.

### Building the Image
```bash
docker build -t robotics-vp-core:latest -f Dockerfile.runpod .
```

### Pushing to Container Registry (Optional)
If you use a custom registry:
```bash
docker tag robotics-vp-core:latest your-registry/robotics-vp-core:latest
docker push your-registry/robotics-vp-core:latest
```

## 3. Launching on RunPod
1.  **Select Pod**: Choose a GPU instance (e.g., 1x A100).
2.  **Template**: Select "RunPod Pytorch 2.1" or use your custom image.
3.  **Volume**: Mount a volume (e.g., `/workspace`) to persist data.
    *   *Size*: At least 50GB for datasets and checkpoints.
4.  **Environment Variables**:
    *   `WANDB_API_KEY`: If using Weights & Biases.
    *   `PYTHONUNBUFFERED=1`: For real-time logs.

## 4. Running Stage 6 Training
Once inside the pod (via SSH or Jupyter Terminal):

1.  **Clone/Update Repo** (if not using custom image):
    ```bash
    cd /workspace
    git clone https://github.com/robotics-vp/robotics-vp-core.git
    cd robotics-vp-core
    pip install -r requirements.txt
    ```

2.  **Verify GPU**:
    ```bash
    python3 scripts/check_gpu_env.py
    ```

3.  **Start Monitoring (Background)**:
    ```bash
    nohup python3 scripts/monitor_gpu_usage.py --interval 5 > gpu_monitor.log 2>&1 &
    ```

4.  **Run Training**:
    ```bash
    # Example: Train Vision Backbone with AMP
    python3 scripts/train_vision_backbone_real.py \
        --use-mixed-precision \
        --batch-size 32 \
        --epochs 50
    ```

    *Note*: Use `--use-mixed-precision` to enable AMP and reduce VRAM usage.

## 5. Running Demos
To run the demo policy in simulation:

```bash
python3 scripts/run_demo_in_sim.py \
    --task-id drawer_open \
    --use-mixed-precision \
    --num-episodes 10
```

## 6. Monitoring & Debugging
*   **Logs**: Check `results/training_logs/` for JSONL logs.
*   **GPU Usage**: Check `results/monitoring/gpu_usage.jsonl`.
*   **OOM Errors**: If you encounter CUDA OOM:
    1.  Reduce `--batch-size`.
    2.  Ensure `--use-mixed-precision` is on.
    3.  Check `gpu_monitor.log` for peak usage.

## 7. Exfiltrating Data
To download checkpoints or logs:
```bash
# Zip results
zip -r results.zip results/

# Download via scp (if SSH enabled)
scp root@<pod-ip>:/workspace/robotics-vp-core/results.zip ./
```
