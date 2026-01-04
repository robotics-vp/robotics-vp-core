# Third-Party Dependencies

This directory contains wrappers and integration code for third-party research models.

## Required Repositories

The Scene IR Tracker uses the following external projects:

| Name | Repository | Purpose |
|------|-----------|---------|
| SAM3D-Objects | `facebookresearch/sam-3d-objects` | Object 3D reconstruction |
| SAM3D-Body | `facebookresearch/sam-3d-body` | Human mesh recovery |
| INRTracker | `princeton-computational-imaging/INRTracker` | Inverse neural radiance tracking |

## Installation

### Option 1: Git Submodules (Recommended)

```bash
# Initialize submodules
git submodule add https://github.com/facebookresearch/sam-3d-objects.git third_party/sam3d_objects
git submodule add https://github.com/facebookresearch/sam-3d-body.git third_party/sam3d_body
git submodule add https://github.com/princeton-computational-imaging/INRTracker.git third_party/inrtracker

# Install dependencies for each
pip install -e third_party/sam3d_objects
pip install -e third_party/sam3d_body
pip install -e third_party/inrtracker
```

### Option 2: Manual Installation

Clone each repository into `third_party/` and install:

```bash
cd third_party
git clone https://github.com/facebookresearch/sam-3d-objects.git sam3d_objects
git clone https://github.com/facebookresearch/sam-3d-body.git sam3d_body
git clone https://github.com/princeton-computational-imaging/INRTracker.git inrtracker
```

## Model Weights

Download pretrained weights:

```bash
# SAM3D-Objects
python -m third_party.sam3d_objects_wrapper --download-weights

# SAM3D-Body  
python -m third_party.sam3d_body_wrapper --download-weights

# INRTracker (optional - for advanced rendering)
# See INRTracker README for weight download
```

## Smoke Test

Verify installation:

```bash
python -m third_party.smoke
```

This checks:
1. All required modules are importable
2. Wrapper classes can be instantiated
3. CPU inference works on synthetic data (if weights present)

## Wrapper Modules

The wrappers provide a clean interface:

| Module | Class | Purpose |
|--------|-------|---------|
| `sam3d_objects_wrapper.py` | `SAM3DObjectsInference` | Run SAM3D-Objects inference |
| `sam3d_body_wrapper.py` | `SAM3DBodyInference` | Run SAM3D-Body inference |
| `lpips_wrapper.py` | `LPIPSLoss` | LPIPS perceptual loss |
| `nvdiffrast_wrapper.py` | `NVDiffrastRenderer` | Differentiable rendering |

All wrappers:
- Support CPU and GPU
- Provide graceful fallback if weights missing
- Have consistent error messages
