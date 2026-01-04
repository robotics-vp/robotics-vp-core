# Scene IR Tracker

A unified perception+tracking subsystem combining SAM3D-Body, SAM3D-Objects as upstream priors and INRTracker-style inverse rendering refinement with Kalman tracking.

## Overview

The Scene IR Tracker provides:

1. **Object reconstruction** via SAM3D-Objects adapter - extracts shape/texture latents and geometry from RGB + instance masks
2. **Body reconstruction** via SAM3D-Body adapter - extracts SMPL-like body mesh, joints, and camera parameters
3. **Inverse rendering refinement** - optimizes entity parameters (texture, pose, shape) to match target images using differentiable rendering
4. **Kalman tracking** - maintains stable track IDs across frames using position, IoU, and latent similarity

## Architecture

```
RGB Frames + Instance Masks
         │
         ▼
┌─────────────────────────────────┐
│    SAM3D Adapters               │
│  ┌───────────┐  ┌───────────┐  │
│  │Objects    │  │Body       │  │
│  │Adapter    │  │Adapter    │  │
│  └───────────┘  └───────────┘  │
└─────────────────────────────────┘
         │
         ▼ SceneEntity3D list
┌─────────────────────────────────┐
│    IR Refiner                   │
│  - Texture optimization         │
│  - Pose/scale optimization      │
│  - Shape refinement             │
└─────────────────────────────────┘
         │
         ▼ Refined entities
┌─────────────────────────────────┐
│    Kalman Track Manager         │
│  - Association (dist+IoU+latent)│
│  - Birth/death logic            │
│  - EMA latent updates           │
└─────────────────────────────────┘
         │
         ▼
    SceneTracks
         │
         ▼
    Motion Hierarchy Node (optional)
```

## Configuration

### SceneIRTrackerConfig

```python
from src.vision.scene_ir_tracker import SceneIRTrackerConfig

config = SceneIRTrackerConfig(
    device="cuda",                    # or "cpu"
    precision="float32",              # or "float16"
    use_stub_adapters=True,           # Use stub SAM3D implementations
    body_joints_for_mhn=["pelvis", "left_hand", "right_hand"],
    
    ir_refiner_config=IRRefinerConfig(
        num_texture_iters=50,
        num_pose_iters=30,
        num_shape_iters=20,
        rgb_loss_weight=1.0,
        lpips_weight=0.1,
    ),
    
    tracking_config=TrackingConfig(
        association_distance_threshold=2.0,
        max_age=5,
        ema_alpha=0.9,
    ),
)
```

### Enabling in LSD Backend

```python
from src.config.lsd_vector_scene_config import LSDVectorSceneConfig

lsd_config = LSDVectorSceneConfig(
    enable_scene_ir_tracker=True,
    scene_ir_tracker_config=SceneIRTrackerConfig(use_stub_adapters=True),
    enable_motion_hierarchy=True,  # Also run MHN on tracked positions
)
```

## Usage

### Basic Usage

```python
from src.vision.scene_ir_tracker import SceneIRTracker, SceneIRTrackerConfig
from src.vision.nag.types import CameraParams

# Create tracker
tracker = SceneIRTracker(SceneIRTrackerConfig(device="cpu"))

# Create camera
camera = CameraParams.from_single_pose(
    position=(0, 0, -5),
    look_at=(0, 0, 0),
    up=(0, 1, 0),
    fov_deg=60,
    width=256,
    height=256,
)

# Process episode
scene_tracks = tracker.process_episode(
    frames=list_of_rgb_arrays,           # List of (H, W, 3) uint8
    instance_masks=list_of_mask_dicts,   # List of {id: (H, W) bool}
    camera=camera,
    class_labels=list_of_label_dicts,    # Optional
)

# Get results
print(f"Tracks: {len(scene_tracks.tracks)}")
print(f"Mean IR loss: {scene_tracks.metrics.mean_ir_loss}")

# Get positions for MHN
positions = scene_tracks.get_positions_for_mhn()  # (T, N, 3)
```

### Running the Script

```bash
python scripts/run_scene_ir_tracker_on_lsd.py \
    --num-episodes 5 \
    --num-frames 50 \
    --output tmp/tracker_out \
    --device cpu
```

## Outputs

### SceneTracks

Contains:
- `frames`: Per-frame list of `SceneEntity3D` objects
- `tracks`: Dict of `track_id` → entity history
- `metrics`: `SceneTrackerMetrics` with IR losses, ID switches, etc.

### SceneTrackerMetrics

- `ir_loss_per_frame`: IR loss at each frame
- `id_switch_count`: Total ID switches detected
- `occlusion_rate`: Fraction of occluded entity-frames
- `mean_ir_loss`: Mean IR loss across episode
- `converged_count`: Frames where IR converged

## Integration with MHN and Econ Reports

When enabled in LSD backend:

1. Scene tracker runs on each episode
2. Tracked entity positions are extracted via `get_positions_for_mhn()`
3. MHN runs on combined body+object positions
4. Results attached to `episode.scene_tracks` and `episode.motion_hierarchy`
5. Econ reports can compute scene tracker quality metrics

## Testing

```bash
# Run all scene IR tracker tests
python -m pytest tests/vision/scene_ir_tracker/ -v

# Specific tests
python -m pytest tests/vision/scene_ir_tracker/test_occlusion_composition.py -v
python -m pytest tests/vision/scene_ir_tracker/test_id_stability.py -v
```

## Notes

- **Stub Mode**: By default, SAM3D adapters use stub implementations that generate synthetic predictions. Set `use_stub_adapters=False` and provide model paths for real inference.
- **GPU**: IR refinement benefits from GPU. Tests use CPU fallback.
- **Backward Compatibility**: When `enable_scene_ir_tracker=False` (default), no scene tracker code runs and episode results have `scene_tracks=None`.

## Third-Party Installation

The Scene IR Tracker can use real upstream models for production quality:

### 1. Install Submodules

```bash
# Clone upstream repositories
git clone https://github.com/facebookresearch/sam-3d-objects.git third_party/sam3d_objects
git clone https://github.com/facebookresearch/sam-3d-body.git third_party/sam3d_body
git clone https://github.com/princeton-computational-imaging/INRTracker.git third_party/inrtracker

# Install each
pip install -e third_party/sam3d_objects
pip install -e third_party/sam3d_body
pip install -e third_party/inrtracker
```

### 2. Download Model Weights

```bash
# SAM3D-Objects weights
mkdir -p checkpoints/sam3d_objects
# Download from https://github.com/facebookresearch/sam-3d-objects/releases

# SAM3D-Body weights
mkdir -p checkpoints/sam3d_body
# Download from https://github.com/facebookresearch/sam-3d-body/releases
```

### 3. Verify Installation

```bash
python -m third_party.smoke
```

## Demo Scripts

### Basic Demo

```bash
python scripts/run_scene_ir_tracker_on_lsd.py \
    --num-episodes 5 \
    --num-frames 50 \
    --output tmp/tracker_out \
    --device cpu
```

### With Overlays and Loss Curves

```bash
python scripts/run_scene_ir_tracker_on_lsd.py \
    --num-episodes 2 \
    --num-frames 30 \
    --save-overlays \
    --save-loss-curves \
    --output tmp/tracker_demo
```

Output files:
- `metrics.json` - Aggregate metrics
- `loss_curves.json` - Per-frame IR losses
- `episode_XXXX/overlays/` - Overlay images (if `--save-overlays`)
