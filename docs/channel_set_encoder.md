# Channel-Set Encoder

## Rationale
Robotics deployments mix heterogeneous signals (vision, geometry, embodiment, etc.) that vary by site and hardware. The channel-set encoder canonicalizes these channels into a single, permutation-invariant token stream so downstream modules can rely on a consistent representation even when optional channels are missing.

This is additive: existing reward and econ modules remain unchanged. Outputs are diagnostics, representation embeddings, and optional sampling signals.

## Channel Roles
Required channels (train/eval):
- `vision_rgb`: RGB tokenization is mandatory. If a camera is absent, the pipeline must emit a deterministic render or proxy render derived from geometry.
- `embodiment`: Embodiment artifacts are required once the embodiment pipeline runs. Missing embodiment at runtime is a pipeline bug (synthetic unit-test mode is the exception).
- `geometry_scene_graph`: SceneGraphEncoder tokens provide the geometry substrate.

Optional channels:
- `geometry_gaussian_scene`
- `action_stream`, `joint_saturation`, `task_constraints`, `backend_tags`, `failure_taxonomy`

Channel definitions live in `configs/channel_groups_robotics.json`.

## Pipeline Overview
1) Token providers emit time-aligned tokens per channel.
2) `ChannelSetEncoder` projects each channel to `D_model` and adds a channel-id embedding.
3) Per-timestep set pooling (Set Transformer PMA style) yields canonical tokens.
4) Optional LOO-CL aligns each channel embedding to the aggregate of the remaining channels.

## vision_rgb Guarantee
When the environment has no camera, `vision_rgb` must still exist:
- Preferred: deterministic sim render.
- Fallback: deterministic proxy render from geometry (still output as `vision_rgb` tokens).

## Config
`config/pipeline.yaml` contains optional toggles:
- `use_channel_set_encoder`
- `use_loo_cl_pretrain`
- `aux_loss_loo_cl_weight`

## Troubleshooting
- Missing required channels: ensure required providers are enabled or run in synthetic test mode.
- Shape mismatch: providers must output `[B, T, D]` tokens (or `[T, D]` per episode).
- Determinism: use fixed seeds for token providers if you see drift across runs.

## Demo
Run the demo script:

```bash
python -m scripts.demo_channel_set_encoder
```

The script prints channel token shapes, canonical token shape, permutation-invariance diff, and LOO-CL metrics.

## Determinism
For strict determinism in demos and diagnostics, set:

```bash
VPE_DETERMINISTIC=1 VPE_DETERMINISTIC_SEED=0 python -m scripts.demo_channel_set_encoder
```
