# External Data Sources for Phase 2 Perception Seam Training

This document analyzes external data sources for reducing Phase 2 seams' dependence on synthetic/testing scaffolding.

## Executive Summary

### GPU-Honest Classification

External data closes the synthetic-comfort-zone gap **structurally now** (adapter/integration/schema work), while promotion-credible seam training remains largely a **GPU-era step**.

| Level | Meaning | GPU Required? | Achievable Now? |
|-------|---------|---------------|-----------------|
| **Adapter-usable** | Can wire into stack, schema/intake work | No | ✅ Yes |
| **Prototype-trainable** | Small dev proof-of-life on CPU/dev subset | Minimal | ⚠️ Limited |
| **Promotion-credible** | Meaningful training for promotion decisions | Yes | 🔶 GPU-era |

### Source Classification

| Source | Scale | Adapter-usable | Prototype-trainable | Promotion-credible |
|--------|-------|----------------|---------------------|-------------------|
| **DROID** | 76K traj, 25.5M frames | ✅ Now | ⚠️ `droid_100` only | 🔶 GPU required |
| **Bridge V2** | 60K traj, 53K eps | ✅ Now | ⚠️ Small subset | 🔶 GPU required |
| **ALOHA suite** | ~20 datasets | ✅ Now | ⚠️ Single task | 🔶 GPU required |
| **KITTI** | Driving + LiDAR GT | ⚠️ Adapter needed | ⚠️ Small subset | 🔶 GPU required |

**Not worth prioritizing now:**
- Egocentric datasets (Ego4D, HOI4D, EgoDex) — wrong viewpoint for manipulation, no robot state
- SA-1B for SAM calibration — no manipulation context, quality signals don't transfer

### What's Achievable Now (No GPU)

- ✅ Adapter/schema work: wire LeRobot datasets into `MultiProviderSample` records
- ✅ Integration verification: load real data, verify schema correctness
- ✅ Evaluation design: establish what metrics matter before GPU training
- ⚠️ CPU proof-of-life: tiny subset training to verify loss functions work
- ❌ Promotion-credible training: requires GPU at meaningful scale

**Integration path:**
- Existing `lerobot_bridge.py` and `rlds_bridge.py` already provide `ReplayEpisodeRecord` rehydration
- Multi-camera observations map directly to `MultiProviderSample` records
- Temporal sequences map directly to V-JEPA temporal training data

---

## Detailed Analysis by Data Source

### 1. DROID (Distributed Robot Interaction Dataset)

**Source:** [lerobot/droid_1.0.1](https://huggingface.co/datasets/lerobot/droid_100) / [Full dataset](https://droid-dataset.github.io/)

**Scale:** 76K trajectories, 350h interaction, 564 scenes, 86 tasks, 25.5M frames

**Observations available:**
- `observation.images.exterior_image_1_left` (180×320 RGB)
- `observation.images.exterior_image_2_left` (180×320 RGB)
- `observation.images.wrist_image_left` (180×320 RGB)
- `observation.state` (7 motor states)
- `language_instruction` (task description)

**Which seams it helps:**

| Seam | Usefulness | How |
|------|------------|-----|
| **EvidenceFusionSeam** | ✅ STRONG | 3 cameras = 3 "providers" for held-out reconstruction |
| **VisionBackboneProjectionSeam** | ✅ STRONG | Dense RGB frames + task labels for object identity supervision |
| **VJEPATemporalAlignmentSeam** | ✅ STRONG | Long episodes (avg ~400 frames) with temporal structure |
| **SAMCalibrationSeam** | ⚠️ REQUIRES ADAPTER | No mask annotations; would need SAM inference pass to generate masks + quality heuristics |
| **DepthMetricCalibrationSeam** | ❌ NO GT | No depth ground truth; monocular-only |

**Integration surface:**
- Intake: `replay_episode_from_lerobot()` → `ReplayEpisodeRecord`
- Perception: Multi-camera obs → `MultiProviderSample` with `provider_kind="vision_backbone"`
- Training: Episodes → `VJEPATemporalSample` for temporal alignment

**GPU-Honest Usability:**
- ✅ **Adapter-usable now**: LeRobot v3 format, `lerobot_bridge.py` pathway exists
- ⚠️ **Prototype-trainable**: `droid_100` subset (100 episodes) fits in memory for CPU proof-of-life
- 🔶 **Promotion-credible**: Full 76K trajectories requires GPU; 400GB+ data footprint

---

### 2. BridgeData V2

**Source:** [IPEC-COMMUNITY/bridge_orig_lerobot](https://huggingface.co/datasets/IPEC-COMMUNITY/bridge_orig_lerobot) / [Original](https://rail-berkeley.github.io/bridgedata/)

**Scale:** 60K trajectories, 24 environments, WidowX robot

**Observations available:**
- `observation.images.image_0` through `image_3` (256×256 RGB, 4 cameras)
- `observation.state` (8 values: x, y, z, roll, pitch, yaw, pad, gripper)
- Natural language task labels

**Which seams it helps:**

| Seam | Usefulness | How |
|------|------------|-----|
| **EvidenceFusionSeam** | ✅ STRONG | 4 cameras > DROID's 3 for held-out reconstruction |
| **VisionBackboneProjectionSeam** | ✅ STRONG | Pick-place actions provide implicit object identity |
| **VJEPATemporalAlignmentSeam** | ✅ MODERATE | 5 FPS, shorter episodes than DROID |
| **SAMCalibrationSeam** | ⚠️ REQUIRES ADAPTER | Same limitation as DROID |
| **DepthMetricCalibrationSeam** | ❌ NO GT | No depth ground truth |

**Integration surface:**
- Same as DROID: `replay_episode_from_lerobot()` pathway
- 4 cameras provide richer multi-provider signal than DROID

**GPU-Honest Usability:**
- ✅ **Adapter-usable now**: LeRobot format, same pathway as DROID
- ⚠️ **Prototype-trainable**: Small subset possible on CPU; 4 cameras = richer fusion signal
- 🔶 **Promotion-credible**: 53K episodes requires GPU; lower FPS (5) than DROID (15)

---

### 3. Open-X ALOHA Suite

**Source:** [lerobot Open X-Embodiment collection](https://huggingface.co/collections/lerobot/open-x-embodiment)

**Scale:** ~20 datasets, bimanual manipulation, multi-camera arrays

**Key datasets:**
- `lerobot/aloha_mobile_cabinet` (128K frames)
- `lerobot/aloha_static_*` suite (cups, coffee, tape, etc.)
- Mobile and static variants

**Which seams it helps:**

| Seam | Usefulness | How |
|------|------------|-----|
| **EvidenceFusionSeam** | ✅ STRONG | Multi-camera bimanual setup |
| **VisionBackboneProjectionSeam** | ✅ MODERATE | Less diverse than DROID/Bridge |
| **VJEPATemporalAlignmentSeam** | ✅ MODERATE | Good for bimanual coordination temporal patterns |
| **Embodiment bridge (later)** | ✅ STRONG | Bimanual affordances relevant for G1 humanoid |

**GPU-Honest Usability:**
- ✅ **Adapter-usable now**: LeRobot format, individual tasks are small
- ⚠️ **Prototype-trainable**: Single task datasets fit on CPU for verification
- 🔶 **Promotion-credible**: Suite aggregation requires GPU; most valuable later for G1 bimanual

---

### 4. KITTI Depth Subset

**Source:** [KITTI Vision Benchmark](https://www.cvlibs.net/datasets/kitti/eval_odometry.php)

**Scale:** Driving dataset with LiDAR ground truth

**Observations available:**
- Stereo RGB cameras
- Velodyne LiDAR point clouds (ground truth depth)
- GPS/IMU

**Which seams it helps:**

| Seam | Usefulness | How |
|------|------------|-----|
| **DepthMetricCalibrationSeam** | ✅ STRONG | LiDAR provides metric depth ground truth |
| **Others** | ❌ NOT RELEVANT | Not manipulation-focused |

**Integration surface:**
- Requires custom adapter (not LeRobot-native)
- Maps to `DepthCalibrationSample` with `ground_truth_depth` from LiDAR

**GPU-Honest Usability:**
- ⚠️ **Adapter-usable**: Requires custom adapter (not LeRobot-native); no GPU for adapter work
- ⚠️ **Prototype-trainable**: Small subset possible on CPU for depth seam verification
- 🔶 **Promotion-credible**: Full KITTI requires GPU; domain mismatch (driving vs manipulation)

**Alternative:** NYU Depth V2 (indoor, Kinect depth) — closer domain but lower quality GT

---

### 5. Sources NOT Worth Prioritizing

**Egocentric datasets (Ego4D, EgoDex, HOI4D, OpenEgo):**
- ❌ Wrong viewpoint (first-person vs workspace cameras)
- ❌ No robot state/action supervision
- ❌ Human hand ≠ robot gripper affordances
- Verdict: Only useful much later for understanding human manipulation intent, not current seam training

**SA-1B for SAM calibration:**
- ❌ Generic images, not manipulation scenes
- ❌ Quality signals (predicted_iou, stability_score) don't transfer to manipulation context
- ❌ Would need manipulation-specific mask quality labeling
- Verdict: SAM calibration should be trained on manipulation-scene masks with downstream task correlation

---

## Mapping to Typed Intake Surfaces

### Current intake path

```
External LeRobot dataset
    ↓ HuggingFace datasets.load_dataset()
    ↓ replay_episode_from_lerobot()
ReplayEpisodeRecord + ReplayStepRecord
    ↓ (NEW ADAPTER NEEDED)
MultiProviderSample / seam-specific datasets
    ↓
Perception seam training infrastructure
```

### Proposed adapter additions

1. **`MultiProviderSample` from multi-camera LeRobot episode:**
   ```python
   def multi_provider_sample_from_lerobot_step(
       step: ReplayStepRecord,
       camera_keys: List[str],
   ) -> MultiProviderSample:
       """Convert LeRobot step with multi-camera obs to MultiProviderSample."""
       providers = []
       for cam_key in camera_keys:
           obs = step.obs.get(f"images.{cam_key}")
           if obs is not None:
               providers.append(ProviderObservation(
                   provider_id=cam_key,
                   provider_kind="vision_backbone",
                   availability_status="available",
                   truth_class="provider_backed",
                   features=encode_image_to_features(obs),  # via frozen backbone
               ))
       return MultiProviderSample(
           sample_id=step.record_id,
           scene_id=step.episode_id,
           frame_idx=step.step_idx,
           providers=providers,
           downstream_task_success=step.reward,  # proxy
       )
   ```

2. **`VJEPATemporalSample` from episode temporal window:**
   ```python
   def vjepa_temporal_sample_from_episode(
       steps: List[ReplayStepRecord],
       window_start: int,
       window_size: int,
   ) -> VJEPATemporalSample:
       """Extract temporal training sample from episode window."""
       # V-JEPA tokens would come from frozen V-JEPA inference
       # Future object states from subsequent frames
   ```

---

## Doctrine Updates

### Add to `neuralization_bridge_doctrine.md`:

```markdown
## External Data Requirements for Seam Promotion

### Synthetic scaffolding status

Synthetic generators (`generate_synthetic_*_samples`) are **bootstrapping scaffolds only**.
They enable:
- Unit testing of loss functions and data loaders
- Initial training loop verification
- Smoke testing of benchmark gate evaluation

They do NOT establish:
- Promotion credibility for production use
- Downstream task correlation validity
- Cross-WM usefulness evidence

### Promotion credibility levels

1. **Synthetic-only** (current state): seam works on synthetic data
   - Promotion: NOT CREDIBLE
   - Status: development/testing only

2. **External-data-backed**: seam trained/evaluated on real manipulation data (DROID, Bridge, etc.)
   - Promotion: CONDITIONALLY CREDIBLE
   - Requires: held-out reconstruction on external data, temporal prediction accuracy

3. **Cross-WM-validated**: seam outputs improve downstream WM performance
   - Promotion: FULLY CREDIBLE
   - Requires: Embodiment consumer improvement, Sim-Synth branch quality improvement

### Progression rule

A seam SHOULD NOT be promoted to `auto` or `required` posture until it has at least
**external-data-backed** credibility.  Synthetic-only training is insufficient for
production promotion decisions.
```

### Add to `multi_wm_architecture_plan.md` Phase 2 section:

```markdown
### External Data Integration Targets

Phase 2 seams have specific external data requirements:

| Seam | Primary Source | Secondary Source | Blocking on |
|------|---------------|------------------|-------------|
| EvidenceFusionSeam | DROID (3-cam) | Bridge V2 (4-cam) | LeRobot adapter |
| VisionBackboneProjectionSeam | DROID + Bridge | ALOHA suite | Frozen backbone inference |
| VJEPATemporalAlignmentSeam | DROID (15 FPS) | Bridge V2 | V-JEPA model availability |
| DepthMetricCalibrationSeam | KITTI | NYU Depth V2 | Custom adapter |
| SAMCalibrationSeam | — | — | Manipulation mask annotation |

**SAMCalibrationSeam is currently blocked** on manipulation-specific mask quality ground truth.
Options:
1. Generate SAM masks on DROID/Bridge, use downstream grasp success as quality proxy
2. Small-scale manual annotation of mask quality on manipulation scenes
3. Cross-provider agreement as calibration target (if multiple segmentation providers available)
```

---

## Implementation Priorities

### Question: Embodiment-facing consumer wiring vs external-data-backed training first?

**Recommendation: External-data adapter work first**, for these reasons:

1. **Adapter work requires no GPU**: Wire the data intake now, verify schemas work,
   establish evaluation metrics. This is achievable immediately.

2. **Foundation before consumption**: If perception seams are trained only on synthetic data,
   wiring them to Embodiment consumers tests the interface but not the usefulness. The
   Embodiment WM would consume low-credibility perception outputs.

3. **Prototype-trainable for verification**: Small CPU proof-of-life on `droid_100`
   (100 episodes) verifies the full pipeline works end-to-end before GPU investment.

4. **Promotion-credible training is GPU-era**: Meaningful training at scale to support
   actual promotion decisions will wait for GPU availability. This is honest.

5. **Depth seam remains blocked differently**: DepthMetricCalibrationSeam needs KITTI
   adapter (no GPU) plus domain-appropriate training (GPU).

### Concrete next steps (GPU-honest)

**Phase A: Adapter-usable now (no GPU)**

1. Add `MultiProviderSample` adapter from LeRobot multi-camera episodes
2. Add `VJEPATemporalSample` adapter for temporal window extraction
3. Verify schema correctness: load `droid_100`, construct samples, verify shapes
4. Add KITTI adapter for depth seam data intake (separate from training)

**Phase B: Prototype-trainable (CPU / dev GPU)**

1. Run EvidenceFusionSeam on `droid_100` (100 episodes) — verify loss decreases
2. Run VJEPATemporalAlignmentSeam on `droid_100` temporal windows — verify loss decreases
3. Establish baseline metrics for later comparison at scale
4. Document what scale/data is needed for promotion-credible evaluation

**Phase C: Promotion-credible (GPU required)**

1. Scale to full DROID + Bridge datasets (GPU)
2. Train to convergence with proper validation splits
3. Run benchmark gates on held-out external data
4. Establish cross-WM evaluation harness (perception → embodiment delta)

### What NOT to claim

- Do NOT claim seams are "ready" after adapter work alone
- Do NOT promote seams based on `droid_100` CPU proof-of-life
- Do NOT skip GPU-era training and go directly to embodiment wiring

---

## Sources

- [LeRobot Open X-Embodiment Collection](https://huggingface.co/collections/lerobot/open-x-embodiment)
- [LeRobotDataset v3.0 Blog](https://huggingface.co/blog/lerobot-datasets-v3)
- [DROID Dataset](https://droid-dataset.github.io/)
- [lerobot/droid_100](https://huggingface.co/datasets/lerobot/droid_100)
- [BridgeData V2](https://rail-berkeley.github.io/bridgedata/)
- [IPEC-COMMUNITY/bridge_orig_lerobot](https://huggingface.co/datasets/IPEC-COMMUNITY/bridge_orig_lerobot)
- [KITTI Vision Benchmark](https://www.cvlibs.net/datasets/kitti/eval_odometry.php)
- [V-JEPA 2 Paper](https://arxiv.org/abs/2506.09985)
- [Open X-Embodiment Paper](https://arxiv.org/abs/2310.08864)
