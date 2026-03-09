# Quality Debt Follow-Ups

Snapshot from the 2026-03-09 sweep:

- `mypy src/`: `186` errors in `97` files
- `python3 -m ruff check .`: `338` findings
- `python3 -m ruff format --check .`: `874` files would be reformatted

## Next Typed Tranches

Recommended order for the next mypy sweeps:

1. RL/runtime leafs:
   `src/rl/episode_sampling.py`, `src/rl/hydra_losses.py`, `src/inference/demo_policy.py`
2. Vector-scene / env plumbing:
   `src/scene/vector_scene/graph.py`, `src/scene/vector_scene/autoencoder.py`, `src/envs/lsd_vector_scene_env.py`, `src/motor_backend/lsd_vector_scene_backend.py`
3. Orchestrator/econ control:
   `src/orchestrator/economic_controller.py`, `src/orchestrator/orchestration_transformer.py`, `src/regal/data_value.py`
4. Vision heads and training wrappers:
   `src/vision/motion_hierarchy/motion_hierarchy_node.py`, `src/vision/fragility_prior_head.py`, `src/vision/encoder_with_heads.py`, `src/encoders/video_encoder.py`

## Ruff Hotspots

Current dominant `ruff` buckets:

- `F401` unused imports: `220`
- `F841` unused locals: `71`
- `E402` import-order/bootstrap exceptions outside the current allowlist: `12`
- `F821` undefined names: `11`
- `F541` empty f-strings: `10`

Best next ruff targets by file concentration:

- `src/envs/lsd3d_env/proxy_geometry.py`
- `src/inference/demo_policy.py`
- `src/motor_backend/datapacks.py`
- `src/vision/nag/fitter.py`
- `src/vla/vla_trainer.py`

## What Slowed This Sweep Down

- Heterogeneous `Dict[str, object]` payloads created avoidable `mypy` churn. Prefer `TypedDict`, dataclasses, or small typed helper objects for stable payload shapes.
- Repeated `payload.get(...)` calls defeat narrowing. Pull into a local once, type-check it once, then use the narrowed variable.
- Optional-path fields (`Optional[str]`, `Optional[Path]`) often drift into call sites that treat them as required. If a field is mandatory for a code path, assert or normalize it at the boundary.
- NPZ sidecars are easier to type when they use one stable convention. The current safest pattern is a 0-D object array containing a dict payload for structured blobs.
- Optional third-party runtime dependencies (`clip`, `torchvision`, `imageio`) should either have stubs installed in dev requirements or carry explicit `import-not-found` ignores at the import site.
- Builder/factory functions should annotate the produced variable with the common supertype (`nn.Module`, protocol, interface) before branching, otherwise mypy locks onto the first branch.

## Build Ergonomics To Preserve

- Keep ratchets descending; do not raise baselines to absorb unrelated dirt.
- Favor filtered local `mypy` checks on touched files before full-repo runs to separate direct regressions from import-chain debt.
- When adding new advisory/sidecar artifacts, make the serialization shape explicit and keep loader/writer pairs symmetric in the same tranche.
- For script/bootstrap files, keep repo-root path setup patterns isolated so `E402` exceptions do not spread into `src/`.
