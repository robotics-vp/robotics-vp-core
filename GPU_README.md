# GPU Training Setup Guide

This guide covers setting up GPU-accelerated training for the video-to-policy pipeline.

---

## Why GPU?

**CPU-friendly encoders** (Simple2DCNN, Simple3DCNN):
- ✅ Good for development and testing
- ✅ Run on laptop/desktop
- ⚠️ Slow for long training runs (1000+ episodes)

**GPU-accelerated encoders** (R3D18, TimeSformer):
- ✅ 10-50x faster training
- ✅ Pretrained weights available (Kinetics, ImageNet)
- ✅ Better video representations
- ⚠️ Requires CUDA-capable GPU

**When to switch**: Once video pipeline is validated on CPU, move to GPU for longer training runs.

---

## Cloud Providers

### Option 1: RunPod (Recommended for Quick Start)

**Pros**:
- Pay-per-second billing
- Pre-configured PyTorch templates
- Easy setup
- Affordable ($0.30-0.80/hr for RTX 4090)

**Setup**:
1. Go to https://runpod.io
2. Select "PyTorch" template (already has CUDA + PyTorch)
3. Choose GPU: RTX 4090, A5000, or A6000
4. Storage: 50GB minimum
5. Deploy pod

**After deployment**:
```bash
# Clone repo
git clone <your-repo-url>
cd robotics-v-p-economics-model

# Install dependencies
pip install -r requirements-gpu.txt

# Verify GPU
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python3 -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"

# Run training
python3 train_sac.py --config configs/dishwashing_video.yaml
```

### Option 2: AWS EC2 (For Production)

**Pros**:
- Stable, reliable
- Scalable (can launch many instances)
- Integration with S3, CloudWatch, etc.

**Cons**:
- More complex setup
- Hourly billing (more expensive than RunPod for short runs)

**Recommended Instance**: `g5.xlarge` (NVIDIA A10G, ~$1/hr)

**Setup**:
```bash
# Launch Ubuntu 22.04 instance with Deep Learning AMI
# (Already has CUDA + PyTorch installed)

ssh -i your-key.pem ubuntu@<instance-ip>

# Clone repo
git clone <your-repo-url>
cd robotics-v-p-economics-model

# Install project dependencies
pip install -r requirements-gpu.txt

# Run training
python3 train_sac.py --config configs/dishwashing_video.yaml
```

### Option 3: Lambda Labs (Budget Option)

**Pros**:
- Cheapest GPUs (~$0.50/hr for RTX 3090)
- Simple interface

**Cons**:
- Less reliable availability
- Fewer GPUs in stock

**Setup**: Similar to RunPod

### Option 4: Google Colab (Free Tier)

**Pros**:
- Free (with limits)
- Good for experimentation

**Cons**:
- Session timeouts (12 hours max)
- Can't run overnight
- Shared GPUs (slower)

**Setup**:
```python
# In Colab notebook:
!git clone <your-repo-url>
%cd robotics-v-p-economics-model
!pip install -r requirements-gpu.txt
!python3 train_sac.py --config configs/dishwashing_video.yaml
```

---

## Configuration for GPU

### 1. Enable R3D18 Encoder

Edit `configs/dishwashing_video.yaml`:
```yaml
encoder:
  type: "video"
  video:
    arch: "r3d18"          # Switch from simple2dcnn
    pretrained: true       # Use Kinetics pretrained weights
```

### 2. Increase Batch Size (Optional)

With GPU, you can use larger batches:
```yaml
sac:
  batch_size: 512         # Up from 256
  buffer_size: 200000     # Up from 100000
```

### 3. Enable TensorBoard (Optional)

```yaml
logging:
  tensorboard: true
  tensorboard_dir: "runs"
```

Then monitor with:
```bash
tensorboard --logdir runs --host 0.0.0.0 --port 6006
```

---

## Monitoring GPU Usage

### Check GPU Utilization

```bash
# Real-time monitoring
watch -n 1 nvidia-smi

# Or install gpustat
pip install gpustat
gpustat -i 1
```

### Expected GPU Usage

- **R3D18 encoder**: 40-60% GPU utilization
- **Simple3DCNN encoder**: 20-30% GPU utilization
- **Batch size 256**: ~4-6GB VRAM
- **Batch size 512**: ~8-12GB VRAM

If GPU utilization is low (<20%), likely CPU-bound. Check:
- Data loading (increase num_workers)
- Environment step time (optimize env)

---

## Training Speed Estimates

| Setup | Episodes/Hour | 1000 Episodes |
|-------|---------------|---------------|
| CPU (Simple2DCNN) | ~50 | 20 hours |
| GPU (Simple2DCNN) | ~200 | 5 hours |
| GPU (R3D18) | ~150 | 6.7 hours |

**Note**: Actual speeds depend on GPU, batch size, and environment complexity.

---

## Checkpointing & Resuming

Training automatically saves checkpoints every 100 episodes:
```
checkpoints/
├── sac_video_ep100.pt
├── sac_video_ep200.pt
└── sac_video_latest.pt
```

To resume:
```bash
python3 train_sac.py --config configs/dishwashing_video.yaml --resume checkpoints/sac_video_latest.pt
```

---

## Common Issues

### 1. CUDA Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Solution**:
```yaml
# Reduce batch size
sac:
  batch_size: 128  # Down from 256

# Or reduce buffer size
sac:
  buffer_size: 50000  # Down from 100000
```

### 2. Slow Data Loading

**Error**: GPU utilization <20%, but should be higher

**Solution**:
```python
# In train_sac.py, add num_workers to data loading
# (if using DataLoader)
train_loader = DataLoader(..., num_workers=4, pin_memory=True)
```

### 3. Connection Timeouts (RunPod/AWS)

**Error**: SSH connection dropped, training stopped

**Solution**:
```bash
# Use tmux or screen to persist sessions
tmux new -s training
python3 train_sac.py --config configs/dishwashing_video.yaml

# Detach: Ctrl+B, then D
# Reattach: tmux attach -t training
```

---

## Cost Estimates

### 1000 Episodes Training

| Provider | GPU | Cost/Hour | Time | Total Cost |
|----------|-----|-----------|------|------------|
| RunPod | RTX 4090 | $0.79 | 5 hrs | $3.95 |
| AWS EC2 | A10G | $1.01 | 5 hrs | $5.05 |
| Lambda Labs | RTX 3090 | $0.50 | 6 hrs | $3.00 |
| Colab | Free T4 | Free | 8 hrs | $0 (with limits) |

**Recommendation**: Start with RunPod for experimentation, move to AWS for production.

---

## Switching Back to CPU

To switch back to CPU mode (for debugging):

```yaml
# configs/dishwashing_video.yaml
encoder:
  video:
    arch: "simple2dcnn"  # Back to CPU-friendly
    pretrained: false
```

Or use state mode:
```bash
python3 train_sac.py --config configs/dishwashing_feasible.yaml
```

---

## Next Steps After GPU Training

Once you have GPU training working:

1. **Longer runs**: Train for 5000-10000 episodes
2. **Hyperparameter sweep**: Try different learning rates, batch sizes
3. **Real demonstrations**: Collect human video demos, fine-tune encoder
4. **Physics sim**: Add Isaac Gym/PyBullet for realistic rendering
5. **Transfer learning**: Sim→real transfer via vision

---

## Support

- RunPod Discord: https://discord.gg/runpod
- AWS Support: https://aws.amazon.com/support/
- Lambda Labs: support@lambdalabs.com

---

**Status**: GPU infrastructure ready, awaiting integration into train_sac.py.

**ETA**: GPU training fully working once video mode is integrated.
