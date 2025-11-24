# Phase I – Implementation TODO (Hand-off to Claude)

Deterministic, JSON-safe, defaults-off. Keep econ/reward untouched. All logging as JSON lines with floats cast to Python float. Checkpoints = `{model_state_dict, config, trained_steps, seed, metrics?}`.

## Block A – Vision Neuralization (RegNet+BiFPN)
1) RegNet backbone (`src/vision/regnet_backbone.py`)
   - Implement deterministic conv stack; strides {8,16,32} → outputs {P3/P4/P5} (or {C3/C4/C5}); per-level dim from config.
   - Seedable init, no dropout; clamp/normalize to avoid NaNs; deterministic fallback when torch missing (hash-based).
   - Output: ordered dict level->tensor/ndarray; flatten utility preserved.
2) BiFPN fusion (`src/vision/bifpn_fusion.py`)
   - Real BiFPN: normalized positive weights with epsilon; top-down + bottom-up; deterministic level order.
   - Edge cases: single level, mismatched dims (align to min dim), empty returns {}.
3) Config/schema (`config/vision.yaml`, `src/vision/config.py`)
   - Add backbone option `regnet_bifpn`; params: `regnet_feature_dim`, `regnet_levels`, `bifpn_layers`, `bifpn_epsilon`; defaults keep stub/off.
4) PolicyObservationBuilder (`src/vision/policy_observation_builder.py`)
   - Add `backbone == "regnet_bifpn"`: frames → RegNet pyramid → optional BiFPN → flatten to z_v; honor `use_spatial_rnn`; JSON-safe exports; do not overwrite base latent.
5) ConditionedVisionAdapter (`src/vision/conditioned_adapter.py`)
   - Reuse RegNet+BiFPN path; FiLM modulation leaves base `z_v` invariant; fusion weights normalized; clamp scales to [0.1, 10].
6) Smokes
   - Determinism: fixed seed/frame → identical pyramids/BiFPN/flattened z_v.
   - Forward consistency: shapes match config dims; BiFPN normalized weights; no NaNs.
   - ConditionedVisionAdapter on RegNet+BiFPN: `z_v` unchanged, risk_map changes; flag gating respected.
   - Checkpoint round-trip: serialize/deserialize preserves outputs (JSON-safe).

## Block B – Spatial RNN (ConvGRU/S4D)
Files: `src/vision/spatial_rnn.py`, `scripts/train_spatial_rnn.py`, `tests/smoke_tests/test_spatial_rnn.py`
1) Model
   - Class `SpatialRNN(hidden_dim, feature_dim, levels, mode="convgru"|"s4d", seed=0)`.
   - `forward(sequence: List[np.ndarray|Tensor], initial_state=None) -> {level: Tensor, "summary": Tensor}`; deterministic seeding; clamp finite.
2) Dataset expectations
   - Use/extend `SpatialRNNDataset` to yield sequences of pyramids; shapes: list length T, each level → [C] or [C,H,W]; include task/episode/timestep metadata JSON-safe.
3) Training (`scripts/train_spatial_rnn.py`)
   - Losses: next-step prediction MSE; temporal smoothness (L2 on deltas); optional InfoNCE across time.
   - Deterministic checkpoint; log loss components per step (JSON lines).
4) Smokes
   - Determinism: fixed seed+inputs → identical outputs.
   - Shape: outputs match input levels; summary length predictable.
   - One grad step reduces loss > 0; no NaNs.
5) Integration
   - PolicyObservationBuilder: if `use_spatial_rnn`, pass flattened RegNet/BiFPN sequence through SpatialRNN; store JSON-safe summaries.
   - ConditionVector/TrustMatrix: optional conditioning/gating hook; default off.

## Block C – Neural SIMA-2 Segmenter
File: `src/segmenter/neural_sima2.py`; entry `scripts/eval_neural_sima2.py`; smokes in `tests/smoke_tests/test_neural_sima2.py`
1) Model
   - Backbone: lightweight U-Net/FPN with frozen encoder option.
   - Heads: boundary mask (sigmoid), optional primitive classification (softmax primitives).
   - Deterministic init; no dropout by default.
2) Losses
   - Boundary: focal loss (α, γ configurable) + optional dice.
   - Primitive: cross-entropy.
   - Total = λ_boundary*(focal+dice) + λ_primitive*CE.
3) Dataset expectations
   - Consume heuristic segmenter outputs: fields `image`, `boundary_mask`, `primitive_id`, `tags`, `task_id`, `episode_id`, `timestep`.
   - Shapes: image [C,H,W], mask [1,H,W], primitive int; metadata JSON-safe.
4) Accuracy target
   - ≥85% F1 on boundary mask (held-out). Log precision/recall/F1 per epoch.
5) Integration
   - Flag `use_neural_segmenter` (default false) in segmentation engine; fallback to heuristic if disabled/unavailable.
   - Checkpoint: deterministic state + config + metrics + seed.
6) Smokes/Eval
   - Deterministic forward (fixed seed/input → same mask/logits).
   - Serialization round-trip preserves outputs.
   - Eval harness computes F1/precision/recall and asserts F1 >= 0.85 on test split.

## Cross-Cutting
- Determinism: set seeds (np, torch CPU/CUDA) at entry; avoid nondeterministic ops; use epsilon guards/clamps.
- JSON-safe: configs/logs/checkpoints via `to_json_safe`; no non-serializable objects.
- Defaults off: new backbones/segmenter/conditioning gated; keep current behavior unless flag set.
- Logging schema: JSON lines with `{event, step, loss, loss_components, metrics, seed, config_digest, checkpoint_path}`.
