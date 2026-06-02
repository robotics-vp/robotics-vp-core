# WM Subsystem Debt Sweep - 2026-06-01

## Scope

This sweep records post-G1-primary repo debt across the multi-WM stack. It is a
next-session worklist, not a claim that the listed items were fixed here.

Baseline commit for this sweep:

- `9243359 feat: make g1 the primary humanoid target`

## Current Readiness Inputs

Commands run:

```bash
python3 scripts/economic_world_model/nightly_audit.py \
  --output-json /tmp/nightly_audit_post_g1.json \
  --output-markdown /tmp/nightly_audit_post_g1.md

python3 scripts/economic_world_model/check_wm_surface_hygiene.py \
  --output-dir /tmp/wm_surface_hygiene_debt_doc

python3 scripts/economic_world_model/check_g1_primary_env_hygiene.py \
  --output-dir /tmp/g1_primary_env_hygiene_debt_doc

python3 scripts/economic_world_model/check_gpu_run_hygiene.py \
  --manifest-dir configs/runpod/examples \
  --output-dir /tmp/gpu_run_hygiene_g1

python3 -m ruff check . --output-format=json > /tmp/ruff_post_g1.json
python3 -m mypy src/ --show-error-codes --no-error-summary > /tmp/mypy_post_g1.txt
./scripts/runpod/ensure_cli.sh
```

Results:

| Check | Result |
| --- | --- |
| Nightly audit | `status=ok`; no safe automatic additive task detected |
| WM surface hygiene | `status=ok_wm_surface_hygiene_passed`; `scanned_file_count=330`; `blocking_issue_count=0`; `risky_true_claim_count=0` |
| G1 primary hygiene | `status=ok_g1_primary_env_hygiene_passed`; `scanned_file_count=1650`; `legacy_primary_claim_count=0` |
| GPU run hygiene | `status=ok_gpu_run_hygiene_passed`; `manifest_count=3`; `safe_to_queue_count=3` |
| RunPod local prerequisites | blocked: `runpodctl` missing, `RUNPOD_API_KEY` unset, `RUNPOD_VOLUME_ID` unset |
| Broad ruff | 289 issues |
| Broad mypy | 413 errors |

Broad ruff by WM bucket:

| Bucket | Count |
| --- | ---: |
| General / other | 149 |
| Perception / Grounding | 51 |
| Sim / Synth / Physics | 35 |
| Embodiment / Actuation + humanoid readiness | 30 |
| Transport / Meta / Semantic orchestration | 21 |
| Economic WM | 3 |

Broad mypy by WM bucket:

| Bucket | Count |
| --- | ---: |
| Transport / Meta / Semantic orchestration | 100 |
| General / other | 86 |
| Perception / Grounding | 76 |
| Sim / Synth / Physics | 66 |
| Embodiment / Actuation + humanoid readiness | 65 |
| Economic WM | 20 |

## Post Local WM-Surface Pass Update

The first debt-burn pass cleared the narrowed local WM-surface gate and wired
the local RunPod provider ledger plus bio/neuro receipt joins. The following
checks now pass:

```bash
python3 -m ruff check \
  src/world_model \
  src/training/perception_seam_data.py \
  scripts/economic_world_model \
  scripts/runpod \
  tests/test_bio_neuro_substrate.py \
  tests/test_humanoid_phase7_signal_adapters.py \
  tests/test_humanoid_phase7_shadow_runtime_wiring.py

python3 -m mypy --follow-imports=silent \
  src/world_model \
  src/training/perception_seam_data.py \
  src/runpod \
  scripts/economic_world_model \
  scripts/runpod
```

The second local debt-burn pass cleared the full `src/vision` static surface.
This was perception/grounding seam hygiene only: optional PyTorch fallbacks,
NAG/SceneIR metadata narrowing, NumPy scalar casts, and unused import/local
cleanup. It did not run providers, train, write weights, or promote any visual
surface.

The following vision checks now pass:

```bash
python3 -m mypy --follow-imports=silent src/vision
python3 -m ruff check src/vision
python3 -m compileall src/vision -q
python3 -m pytest \
  tests/vision \
  tests/test_nag_core.py \
  tests/test_nag_lsd_integration.py \
  tests/test_vision_backbone_projection_proof_of_life_smoke.py \
  -q
```

The third local debt-burn pass cleared the full `src/vla` static surface. This
was provider-adapter and advisory-scaffold hygiene only: optional torch fallback
bases, Python 3.9-compatible annotations, MetaDINO optional model narrowing,
RECAP optional feature handling, and teacher-runtime payload widening. It did
not run OpenVLA, train, write weights, or promote teacher/runtime outputs.

The following VLA checks now pass:

```bash
python3 -m ruff check src/vla
python3 -m compileall src/vla -q
python3 -m mypy --follow-imports=silent src/vla \
  --show-error-codes --no-error-summary
python3 -m pytest -q \
  tests/test_vla_backend_policy.py \
  tests/test_teacher_runtime.py \
  tests/test_rollout_labeler.py \
  tests/test_train_vla_recap_offline.py \
  tests/test_vla_semantic_evidence.py
```

The fourth local debt-burn pass cleared the full `src/envs` static surface.
This was environment/curriculum hygiene only: fixed-base dishwashing/workcell
and LSD envs remain curriculum/regression producers, not G1 hardware or bipedal
proof. The pass added missing annotations, honest PyBullet missing-stub ignores,
default econ params for a demo path, and unused import/local cleanup.

The following env checks now pass:

```bash
python3 -m ruff check src/envs
python3 -m compileall src/envs -q
python3 -m mypy --follow-imports=silent src/envs \
  --show-error-codes --no-error-summary
python3 -m pytest -q \
  tests/envs \
  tests/test_lsd3d_geometry.py \
  tests/test_lsd_vector_scene_env.py \
  tests/test_lsd_integration.py \
  tests/test_workcell_paramount.py \
  tests/test_env_regality_compliance.py \
  tests/test_g1_primary_environment.py
```

The fifth local debt-burn pass cleared the full `src/rl` static surface. This
was sampler/Hydra/PPO typing and lint hygiene only. It preserved reward math,
loss semantics, and bounded advisory authority.

The following RL checks now pass:

```bash
python3 -m ruff check src/rl
python3 -m compileall src/rl -q
python3 -m mypy --follow-imports=silent src/rl \
  --show-error-codes --no-error-summary
python3 -m pytest -q \
  tests/test_weights.py \
  tests/test_sampler_policy.py \
  tests/test_train_sampler_policy.py \
  tests/test_sampling_determinism_seeded.py \
  tests/test_queue_dispatch_integration.py \
  tests/test_online_queue_dispatch_integration.py \
  tests/test_shadow_offline_rl.py \
  tests/test_shadow_replay_policy.py
```

The sixth local debt-burn pass cleared the full `src/scene` static surface.
This was vector-scene support hygiene only: mixed tensor/id payload typing,
NumPy scalar narrowing, tiled-scene list annotations, enum-index casts, and
minor unused import/local cleanup. `src/scene` remains lower-WM scene substrate
for Perception/Grounding, Sim/Synth, curriculum/regression, and future
trainer/runtime lanes. It is not trained/provider-backed truth.

The following scene checks now pass:

```bash
python3 -m ruff check src/scene
python3 -m mypy --follow-imports=silent src/scene \
  --show-error-codes --no-error-summary
python3 -m compileall src/scene -q
python3 -m pytest -q \
  tests/test_vector_scene_graph.py \
  tests/test_lsd_vector_scene_env.py \
  tests/test_lsd_integration.py \
  tests/test_lsd3d_geometry.py
```

The seventh local debt-burn pass cleared the full `src/motor_backend` static
surface. This was provider/hardware adapter hygiene only: optional Holosoma
provider probes remain fail-closed, provider-bound config replacement is typed
at the boundary, mixed metric/receipt metadata is represented honestly, and
LSD vector-scene scene-tracker containers are annotated. It did not run
Holosoma, ROS2, SDK2, Unitree, hardware, providers, GPU training, or policy
promotion.

The following motor-backend checks now pass:

```bash
python3 -m ruff check src/motor_backend
python3 -m mypy --follow-imports=silent src/motor_backend \
  --show-error-codes --no-error-summary
python3 -m compileall src/motor_backend -q
python3 -m pytest -q \
  tests/test_local_backend_factory_adapter.py \
  tests/test_backend_health.py \
  tests/test_holosoma_backend_interface.py \
  tests/test_holosoma_adapter_execution.py \
  tests/test_holosoma_runtime_binding.py \
  tests/test_holosoma_runtime_pack.py \
  tests/test_holosoma_adapter_realization.py \
  tests/test_synthetic_backend.py
```

The eighth local debt-burn pass cleared the full `src/replay` static surface.
This was replay/evidence substrate hygiene only: nested Economic WM window
grouping, governed-video and semantic-degraded importer metadata typing,
replay dataset precondition grouping, and receipt-label loop narrowing. It did
not download LeRobot data, run providers, train, write weights, execute
hardware, or promote replay rows.

The following replay checks now pass:

```bash
python3 -m ruff check src/replay
python3 -m mypy --follow-imports=silent src/replay \
  --show-error-codes --no-error-summary
python3 -m compileall src/replay -q
python3 -m pytest -q \
  tests/test_replay_schema.py \
  tests/test_replay_dataset.py \
  tests/test_receipt_ingest.py \
  tests/test_training_run_receipt_ingest.py \
  tests/test_dataset_bridges.py \
  tests/test_lerobot_perception_adapter.py
```

The ninth local debt-burn pass cleared the full `src/representation` static
surface. This was trainer/runtime token-substrate hygiene only: optional YAML
typing, contrastive-loss tensor narrowing, geometry-token guards,
Gaussian-scene projection typing, vector-scene device conversion, and unused
import cleanup. It did not train representation models, write weights, run
providers, download datasets, or promote representation outputs.

The following representation checks now pass:

```bash
python3 -m ruff check src/representation
python3 -m mypy --follow-imports=silent src/representation \
  --show-error-codes --no-error-summary
python3 -m compileall src/representation -q
python3 -m pytest -q \
  tests/representation \
  tests/epiplexity/test_curated_slices_token_only.py
```

The tenth local debt-burn pass cleared the full `src/process_reward` static
surface. This was reward-adjacent trainer/runtime hygiene only: mixed PBRS
diagnostic payload typing, orchestrator adjustment payload typing, feature
array narrowing, source-count annotations, and unused import/local cleanup. It
did not change PBRS math, fusion behavior, reward equations, controller math,
Phase B math, or promotion posture.

The following process-reward checks now pass:

```bash
python3 -m ruff check src/process_reward
python3 -m mypy --follow-imports=silent src/process_reward \
  --show-error-codes --no-error-summary
python3 -m compileall src/process_reward -q
python3 -m pytest -q tests/process_reward
```

The eleventh local debt-burn pass cleared the full `src/hrl` static surface
and fixed the direct Phase C HRL/VLA smoke entrypoint. This was HRL trainer,
controller, scripted-policy, and unified skill-graph hygiene only: optional
torch fallback bases are explicit, workcell skill specs are narrowed before
dict indexing, unused imports/locals were removed, and
`scripts/smoke_test_phase_c_hrl_vla.py` now bootstraps the repo root for direct
AGENTS.md-style invocation. It did not train HRL/VLA models, write datapack
truth beyond the script's existing ignored local smoke output, change reward
math, alter skill semantics, or promote any policy.

The following HRL checks now pass:

```bash
python3 -m ruff check src/hrl scripts/smoke_test_phase_c_hrl_vla.py
python3 -m mypy --follow-imports=silent src/hrl \
  --show-error-codes --no-error-summary
python3 -m compileall src/hrl scripts/smoke_test_phase_c_hrl_vla.py -q
python3 -m pytest -q \
  tests/test_skill_graph.py \
  tests/test_semantic_coverage_graph.py \
  tests/test_semantic_gap_closure.py \
  tests/test_coverage_evidence_harvester.py
python3 scripts/smoke_test_phase_c_hrl_vla.py --episodes 3
```

The aggregate focused suite for all touched debt-burn families also passes:

```bash
python3 -m pytest -q \
  tests/test_vector_scene_graph.py \
  tests/test_lsd_vector_scene_env.py \
  tests/test_lsd_integration.py \
  tests/test_lsd3d_geometry.py \
  tests/test_local_backend_factory_adapter.py \
  tests/test_backend_health.py \
  tests/test_holosoma_backend_interface.py \
  tests/test_holosoma_adapter_execution.py \
  tests/test_holosoma_runtime_binding.py \
  tests/test_holosoma_runtime_pack.py \
  tests/test_holosoma_adapter_realization.py \
  tests/test_synthetic_backend.py \
  tests/test_replay_schema.py \
  tests/test_replay_dataset.py \
  tests/test_receipt_ingest.py \
  tests/test_training_run_receipt_ingest.py \
  tests/test_dataset_bridges.py \
  tests/test_lerobot_perception_adapter.py \
  tests/representation \
  tests/epiplexity/test_curated_slices_token_only.py \
  tests/process_reward \
  tests/test_skill_graph.py \
  tests/test_semantic_coverage_graph.py \
  tests/test_semantic_gap_closure.py \
  tests/test_coverage_evidence_harvester.py
```

Result: `292 passed, 22 warnings`.

The twelfth local debt-burn pass cleared the next support-surface tier:
`src/regal`, `src/analytics`, `src/encoders`, `src/evidence`, and
`src/policies`. This was governance, economics-reporting, encoder,
precondition/evidence, and policy-registry hygiene only: optional torch aliases
are explicit, generic context values are narrowed before float conversion,
summary/report containers are typed, direct encoder module fields are typed as
`nn.Module`, and safe unused import/local cleanup was applied. It did not run
providers, train models, write weights, change promotion gates, mutate reward
math, or grant policy/Phase 7 authority.

The following support-tier checks now pass:

```bash
python3 -m ruff check src/regal
python3 -m mypy --follow-imports=silent src/regal \
  --show-error-codes --no-error-summary
python3 -m compileall src/regal -q
python3 -m pytest -q \
  tests/test_regal_uses_econ_tensor.py \
  tests/test_regal_promotion_policy.py \
  tests/test_regal_gates.py \
  tests/test_regal_reports_provenance.py \
  tests/test_regal_objective_integrity_blocks_early_scalarization.py \
  tests/test_econ_data_regal.py \
  tests/test_regal_training_runner.py \
  tests/test_econ_regal_sampling.py \
  tests/test_regal_phases.py \
  tests/test_regal_gates_patience.py \
  tests/test_governance_assessment.py \
  tests/test_bio_neuro_substrate.py

python3 -m ruff check src/analytics
python3 -m mypy --follow-imports=silent src/analytics \
  --show-error-codes --no-error-summary
python3 -m compileall src/analytics -q
python3 -m pytest -q \
  tests/analytics \
  tests/test_econ_reports.py \
  tests/smoke_tests/test_pricing_report_cli.py

python3 -m ruff check src/encoders
python3 -m mypy --follow-imports=silent src/encoders \
  --show-error-codes --no-error-summary
python3 -m compileall src/encoders -q
python3 - <<'PY'
import torch
from src.encoders.video_encoder import VideoEncoder
from src.encoders.student_video_encoder import AlignedVideoEncoder
for arch in ["simple2dcnn", "simple3dcnn"]:
    enc = VideoEncoder(latent_dim=16, arch=arch, input_channels=3)
    assert tuple(enc(torch.randn(2, 4, 3, 32, 32)).shape) == (2, 16)
student = AlignedVideoEncoder(latent_dim=16, arch="simple2dcnn", projection_dim=8)
z, zp = student.forward_with_projection(torch.randn(2, 4, 3, 32, 32))
assert tuple(z.shape) == (2, 16)
assert tuple(zp.shape) == (2, 16)
PY

python3 -m ruff check src/evidence
python3 -m mypy --follow-imports=silent src/evidence \
  --show-error-codes --no-error-summary
python3 -m compileall src/evidence -q
python3 -m pytest -q \
  tests/test_evidence_bus.py \
  tests/test_gen2sim_validity.py \
  tests/test_train_gen2sim_validity.py \
  tests/test_benchmark_gating.py \
  tests/test_economic_wm_evidence_hygiene.py \
  tests/test_perception_benchmark_evidence_emitter.py \
  tests/test_provider_adapter_benchmark_evidence_emitter.py \
  tests/test_vla_semantic_evidence.py

python3 -m ruff check src/policies
python3 -m mypy --follow-imports=silent src/policies \
  --show-error-codes --no-error-summary
python3 -m compileall src/policies -q
python3 -m pytest -q \
  tests/test_sampler_policy.py \
  tests/test_train_sampler_policy.py \
  tests/test_unified_quality_policy_backward_compat.py \
  tests/test_shadow_replay_policy.py \
  tests/test_vla_backend_policy.py \
  tests/test_plan_policy.py \
  tests/test_semantic_policy.py \
  tests/test_pricing_sentinel.py \
  tests/test_orchestrator_shell_policy.py \
  tests/test_queue_dispatch_policy.py \
  tests/test_fill_path_policy.py \
  tests/test_pipeline_stage_policy.py \
  tests/test_datapack_value_node_integration.py
```

The aggregate focused suite for this support-tier pass also passes:

```bash
python3 -m pytest -q \
  tests/test_regal_uses_econ_tensor.py \
  tests/test_regal_promotion_policy.py \
  tests/test_regal_gates.py \
  tests/test_regal_reports_provenance.py \
  tests/test_regal_objective_integrity_blocks_early_scalarization.py \
  tests/test_econ_data_regal.py \
  tests/test_regal_training_runner.py \
  tests/test_econ_regal_sampling.py \
  tests/test_regal_phases.py \
  tests/test_regal_gates_patience.py \
  tests/test_governance_assessment.py \
  tests/test_bio_neuro_substrate.py \
  tests/analytics \
  tests/test_econ_reports.py \
  tests/smoke_tests/test_pricing_report_cli.py \
  tests/test_evidence_bus.py \
  tests/test_gen2sim_validity.py \
  tests/test_train_gen2sim_validity.py \
  tests/test_benchmark_gating.py \
  tests/test_economic_wm_evidence_hygiene.py \
  tests/test_perception_benchmark_evidence_emitter.py \
  tests/test_provider_adapter_benchmark_evidence_emitter.py \
  tests/test_vla_semantic_evidence.py \
  tests/test_sampler_policy.py \
  tests/test_train_sampler_policy.py \
  tests/test_unified_quality_policy_backward_compat.py \
  tests/test_shadow_replay_policy.py \
  tests/test_vla_backend_policy.py \
  tests/test_plan_policy.py \
  tests/test_semantic_policy.py \
  tests/test_pricing_sentinel.py \
  tests/test_orchestrator_shell_policy.py \
  tests/test_queue_dispatch_policy.py \
  tests/test_fill_path_policy.py \
  tests/test_pipeline_stage_policy.py \
  tests/test_datapack_value_node_integration.py
```

Result: `193 passed`.

The thirteenth local debt-burn pass cleared the full `src/sima2` static
surface. This was semantic/perception support hygiene only: SIMA-2 config
provenance payloads are typed, semantic primitive risk helpers accept set/list
tag collections, and unused imports/locals were removed from advisory ontology,
task-graph, segmenter, and tag propagation code. It did not run SIMA-2
providers, train models, write weights, execute hardware, change task semantics,
or promote semantic/VLA evidence.

The following SIMA2 checks now pass:

```bash
python3 -m ruff check src/sima2
python3 -m mypy --follow-imports=silent src/sima2 \
  --show-error-codes --no-error-summary
python3 -m compileall src/sima2 -q
```

The fourteenth local debt-burn pass cleared the next seven support-surface
families: `third_party`, `src/epiplexity`, `src/diffusion`, `src/inference`,
`src/embodiment`, `src/utils`, and `src/physics`. This was provider-adapter,
trainer/eval-lane, governed diffusion/runtime, demo inference, embodiment
receipt, local utility, and fixed-base curriculum/backend hygiene only. It kept
optional SAM3D/LPIPS/diffusers paths fail-closed, preserved fallback smoke
behavior, widened mixed JSON payloads where they already carry strings, and
narrowed optional values before float/int conversion. It did not run providers,
download weights, train models, execute GPU/hardware, publish ROS2, write SDK2
commands, change physical constants, or promote fixed-base curriculum outputs.

The following support-surface checks now pass:

```bash
python3 -m ruff check third_party src/epiplexity src/diffusion src/inference src/embodiment src/utils src/physics
python3 -m mypy --follow-imports=silent third_party src/epiplexity src/diffusion src/inference src/embodiment src/utils src/physics \
  --show-error-codes --no-error-summary
python3 -m compileall third_party src/epiplexity src/diffusion src/inference src/embodiment src/utils src/physics -q
python3 -m third_party.smoke
python3 -m pytest -q tests/epiplexity
python3 -m pytest -q tests/test_video_diffusion_runtime.py tests/test_video_diffusion_stub_routing.py tests/test_diffusion_prompt_includes_constraints.py tests/test_governed_video_supervision.py tests/test_governed_video_world_model.py
python3 -m pytest -q tests/embodiment tests/test_embodiment_actuation_world_model.py tests/test_embodiment_actuation_phase34.py tests/test_embodiment_shadow_consumer.py
python3 -m pytest -q tests/test_backend_health.py tests/test_local_backend_factory_adapter.py tests/test_isaac_backend_shadow_contract.py tests/test_synthetic_backend.py tests/test_sim_synth_physics_world_model.py tests/test_sim_synth_physics_scripts.py
```

The fifteenth local debt-burn pass made full-repo `mypy` and `ruff` clean.
This was static/typing/script hygiene only across the remaining support
surfaces, legacy/dev scripts, and regression tests. It cleared the residual
`src/datasets`, `src/phase_h`, one-error mypy surfaces, ruff `F821`, safe
`F401`/`F841`, and small `E`/`F` rules while keeping old scripts as substrate
instead of deleting them. It added a missing checkpoint helper to
`scripts/train_spatial_rnn.py` so the script remains executable, but no
training command was run and no weights were written.

The following full-repo static checks now pass:

```bash
python3 -m mypy src/ --show-error-codes --no-error-summary
python3 -m ruff check .
python3 -m compileall src scripts tests -q
```

Focused non-training receipts from this pass:

```bash
python3 -m pytest -q \
  tests/process_reward \
  tests/analytics/test_combined_curriculum.py \
  tests/representation/test_homeostasis.py \
  tests/test_causal_replay_integration.py \
  tests/test_lsd_vector_scene_env.py \
  tests/vision/scene_ir_tracker/test_upstream_integration.py \
  tests/test_dataset_bridges.py

python3 scripts/smoke_test_condition_vector_end_to_end.py
python3 scripts/smoke_test_tfd_vision_chain.py
python3 scripts/smoke_test_econ_correlator_impl.py
python3 scripts/smoke_test_vision_interfaces.py
python3 scripts/run_scene_ir_eval.py --help
python3 scripts/train_spatial_rnn.py --help
```

Result: focused pytest `113 passed, 20 warnings`; smoke/help checks passed.
The warning set is the existing process-reward hindsight warning.

The residual local debt is no longer static hygiene. It is local wiring and
audit work that can be advanced without GPU/provider/hardware execution.

The sixteenth local debt-burn pass closed the Unitree rosbag2/MCAP unavailable
receipt posture. `TraceImportAdapterReceipt` now records dependency modules,
missing optional modules, input-path existence, fixture-shape-only posture, and
`real_import_claimed`. The bridge no longer treats a present rosbag2/MCAP path
as import execution. Current local runtime-bridge receipts report
`trace_import_unavailable_receipt_count=2`,
`trace_fixture_shape_only_count=0`, `rosbag2_real_import_claimed=false`, and
`mcap_real_import_claimed=false`.

Remaining local debt after this pass: provider bring-up readiness ledger,
LeRobot video-to-replay-to-perception receipts, Unitree event spines into Phase
6.4 advisory eval receipts, bio/neuro receipt joins, and neural trainability
audit artifacts.

Current residual broad ruff:

| Area | Count |
| --- | ---: |
| **Total** | **0** |

Current residual broad ruff by code:

| Code | Meaning | Count | Disposition |
| --- | --- | ---: | --- |
| all enabled rules | full repo static hygiene | 0 | keep clean as a guardrail before provider/GPU/hardware sessions |

Current residual full-repo mypy:

| Area | Count |
| --- | ---: |
| **Total actual `error:` records** | **0** |

Current residual full-repo mypy by kind:

| Kind | Count | Meaning |
| --- | ---: | --- |
| all enabled checks | 0 | full `src/` mypy surface is clean |

Legacy/support-surface disposition:

| Surface family | Not superseded because | What should happen |
| --- | --- | --- |
| `src/envs/`, `src/physics/`, `src/scenarios/`, `src/datasets/`, `src/replay/` | curriculum, regression, replay, and data-generation substrate for WMs | keep, type, and posture-tag as fixed-base curriculum or G1-relevant producer; do not treat fixed-base success as humanoid proof |
| `src/vision/`, `src/scene/`, `src/sima2/`, `src/vla/` | provider-facing perception/semantic algorithms and VLA scaffolds | wrap as Perception/Grounding producers or advisory provider adapters; real provider outputs remain external proof |
| `src/rl/`, `src/hrl/`, `src/policies/`, `src/process_reward/`, `src/encoders/`, `src/representation/` | trainer/runtime lanes for future lower-WM or policy components | keep but gate with manifests, receipts, no weight writes in local cleanup, and no promotion claims |
| `src/motor_backend/`, `src/embodiment/`, `src/ingestion/`, `src/runtime/` | hardware/provider/runtime adapter layer | keep as honest unavailable/proof-emitting adapters; do not collapse stubs into hardware truth |
| `src/economics/`, `src/valuation/`, `src/ontology/`, `src/evidence/`, `src/contracts/` | cross-cutting economic, receipt, and evidence contracts | keep; avoid mutating frozen Phase B math or controller equations |
| `scripts/`, `third_party/`, old demos/trainers | operational glue and historical smoke/prototype entrypoints | fix undefined names and safe lint; then either document as legacy/dev-only or migrate into receipt-emitting scripts |

Full-repo mypy/ruff disposition:

- Yes, it makes sense to clean up full-repo `mypy` and `ruff` now. The residual
  files are not outside the WM architecture so much as support surfaces the WMs
  consume: curriculum/replay producers, provider adapters, trainer/runtime
  lanes, smoke scripts, legacy demos, and cross-cutting contracts.
- The cleanup should stay additive and family-scoped. Type cleanup should
  preserve behavior unless a real bug is exposed. Ruff cleanup should fix
  `F821` undefined names first, then safe unused imports/locals, then the
  remaining style/exception issues.
- Legacy functionality should not be deleted just because it predates the
  multi-WM structure. Keep it when it produces curriculum, tests adapters,
  exercises provider pathways, or emits receipts. Reclassify it as dev-only or
  migrate it into receipt-emitting scripts when it is still useful but not a WM
  proof surface.
- The only surfaces that should be considered candidates for retirement are
  unreachable demos/scripts that neither pass smoke checks nor map to a WM
  producer, adapter, trainer lane, receipt contract, or historical regression.
  Retirement should be explicit and documented, not bundled into lint cleanup.

## Multi-WM Unwired Local Debt Fold-In

The companion unwired audit still has local work that is not provider/GPU or
hardware blocked. Those items are now part of this debt sweep instead of a
separate next-action queue.

| Local item | Source audit status | Why it is local | Next Action | Verify |
| --- | --- | --- | --- | --- |
| Static debt burn-down | ranked next local action | It is code/docs/test cleanup over checked-in support surfaces. | Continue mypy by family, then ruff by bug-first bucket. | `python3 -m mypy src/`; `python3 -m ruff check .` |
| Provider bring-up readiness ledger | missing local ledger | It maps provider families to commands, receipts, unavailable posture, RunPod profile, and owner WM without downloading weights or running providers. | Add a typed ledger/checker for SAM/SAM3D, DINO/SigLIP, V-JEPA2, OpenVLA, Isaac/Unitree, and Holosoma. | ledger lint plus `python3 scripts/runpod/prepare_launch_manifest.py --profile provider_bringup --volume-id "$RUNPOD_VOLUME_ID"` when prerequisites exist |
| Bio/neuro substrate receipt joins | substrate wired but isolated | Existing local receipts should become queryable lower-WM/Economic evidence rows without becoming promotion proof. | Add optional joins from `check_bio_neuro_substrate.py` output into lower-WM/economic consumption rows. | `python3 scripts/economic_world_model/check_bio_neuro_substrate.py --output-dir /tmp/bio_neuro` plus focused lower-WM consumption tests |
| Phase 7 bounded consumption | shadow adapters exist | Existing Phase 7 signal adapters can consume better lower-WM receipts once joins exist. | Wire only through existing adapters; do not add abstract Phase 7 vocabulary unless lower-WM receipts force it. | `python3 -m pytest -q tests/test_humanoid_phase7_signal_adapters.py tests/test_humanoid_phase7_shadow_runtime_wiring.py` |
| Script/smoke entrypoint hygiene | old operational glue | Direct script commands are local and should not require undocumented import paths. | Continue fixing direct-entry scripts where `ruff F821` or smoke runs show broken imports; classify unrecoverable old demos as dev-only. | relevant script smoke plus `python3 -m ruff check .` |

Externally blocked items remain explicitly blocked, not local debt: real
provider execution, Isaac/Unitree/Holosoma runtime proof, ROS2 publish, SDK2
write, Unitree hardware, GPU training, promotion-grade benchmarks, and Phase 8
weekly operations.

## G1 / Humanoid Neuralization Posture

The direct body/control neural scaffolds should be read as humanoid-first:

- `unitree_g1` / `bipedal_whole_body_unitree_g1` is now the repo primary target.
- Body/control neural scaffold work is G1/R1-class: whole-body state encoders,
  support/contact/balance predictors, loco-manipulation action heads,
  inverse-dynamics/retargeting lanes, fallback selectors, and latency/watchdog
  resource predictors.
- Perception, Sim/Synth/Physics, Economic, Transport, and Meta WMs are not
  exclusively humanoid modules, but their downstream posture now has to preserve
  G1/humanoid receipts and cannot silently reinterpret fixed-base curriculum as
  bipedal evidence.
- SAC was the main training-loop outlier: it still executes a CPU-capable
  dishwashing source loop, but now emits G1 primary metadata and marks
  dishwashing as fixed-base curriculum only.

## Cross-Cutting Debt

| Rank | Debt | Type | Why It Matters | Next Action | Verify |
| ---: | --- | --- | --- | --- | --- |
| 1 | Broad static hygiene is not clean | structural | Future GPU/provider sessions should not start with noisy lint/type failures unrelated to the run target. | Continue mypy by support-surface family, then burn down ruff by bug-first bucket. Keep commits small and avoid behavior changes unless a real bug is exposed. | `python3 -m mypy src/`; `python3 -m ruff check .` |
| 2 | RunPod is manifest-ready but locally launch-blocked | external/config | Provider bring-up, loop runs, and training cannot launch from this machine until CLI/auth/volume exist. | Install `runpodctl`, set `RUNPOD_API_KEY`, set `RUNPOD_VOLUME_ID` before first loop/train pod. | `./scripts/runpod/ensure_cli.sh` |
| 3 | Broad full-suite proof was not rerun in this sweep | verification | Focused tests pass, nightly audit passes, but the full suite may expose unrelated failures. | Run full pytest once static smoke debt is lower or in CI/GPU-capable lane. | `python3 -m pytest tests/ -v` |
| 4 | Some legacy naming remains in fixed-base curriculum modules | hygiene | G1 primary hygiene passes, but humans can still misread `workcell_isaaclab`-style names as target posture. | Add aliases/docs where module names cannot be safely renamed; prefer `curriculum_*` labels in new surfaces. | `python3 scripts/economic_world_model/check_g1_primary_env_hygiene.py --output-dir /tmp/g1_check` |

## Sim / Synth / Physics WM

| Subsystem | Current State | Debt / Bug | Next Action | Verify |
| --- | --- | --- | --- | --- |
| Backend/provider routing | Rich Phase 1.x surface exists under `src/world_model/sim_synth_physics/`; G1/Unitree target refs exist. | Real Isaac/Unitree/Holosoma runtime proof remains external. Static debt clusters in runtime targets, render materialization, outcome parsers, and synthetic branches. | Fix type hygiene in `runtime_targets.py`, `render_materialization.py`, `synthetic_branches.py`, then rerun Phase 1.x focused tests. | `python3 -m pytest -q tests/test_sim_synth_runtime_targets.py tests/test_sim_synth_runtime_bundles.py tests/test_sim_synth_phase1x_subsystems.py` |
| Asset and calibration truth | Unitree asset contracts and target hardware classes exist. | Default local runs still do not prove real asset parse, real sim execution, or calibrated transforms. | Keep receipts unavailable until assets/runtime are present; next local work is naming and type cleanup only. | `python3 scripts/economic_world_model/probe_phase4_unitree_blockers.py --output-dir /tmp/unitree_probes` |
| Synthetic branch/admission | Gen2Sim and branch-admission contracts exist. | `synthetic_branches.py` has mypy object/indexing errors; branch utility is still mostly structural. | Type row payloads and counters before GPU branch runs. | `python3 -m mypy --follow-imports=silent src/world_model/sim_synth_physics/synthetic_branches.py` |
| Sim-to-embodiment transfer | Boundary is documented and receipt-shaped. | No policy-controlled Unitree sim trace or sim-real transfer evidence yet. | Defer proof to provider/loop pod; keep local receipts honest. | `python3 scripts/runpod/prepare_launch_manifest.py --profile g1_loop_run --volume-id "$RUNPOD_VOLUME_ID"` |

## Perception / Grounding WM

| Subsystem | Current State | Debt / Bug | Next Action | Verify |
| --- | --- | --- | --- | --- |
| Provider contracts | SAM, DINO/SigLIP, V-JEPA, depth provider contracts and real-or-unavailable posture exist. | No real provider execution; promotion remains provisional. | Provider bring-up should start with `provider_bringup` manifest and write receipts before any training claim. | `python3 scripts/runpod/prepare_launch_manifest.py --profile provider_bringup` |
| Neural seams | Evidence fusion, annotation bridge, V-JEPA temporal, and vision-backbone projection seams have local CPU proof lanes. | Static debt clusters in `src/training/perception_seam_trainer.py`, `src/training/perception_seam_losses.py`, and `src/world_model/perception_grounding/neural_seams.py`. | Fix ruff/mypy without changing seam semantics. | `python3 -m ruff check src/training/perception_seam_trainer.py src/training/perception_seam_losses.py src/world_model/perception_grounding/neural_seams.py` |
| Semantic successor | `SemanticVLA` is explicitly scaffolding-only; distributed semantic bridge successor exists structurally. | Real provider-backed semantic-analysis successor is not brought up/trained. | Keep `SemanticVLA` demoted; identify provider family only when provider window is real. | `python3 -m pytest -q tests/test_perception_grounding_world_model.py tests/test_vla_semantic_evidence.py` |
| Benchmark gates | Promotion gates and provisional evidence paths exist. | Non-provisional metric reports need real provider outputs. | Do not promote receipt-only evidence; prepare provider-specific benchmark runs later. | `python3 -m pytest -q tests/test_perception_benchmark_evidence_emitter.py tests/test_perception_seam_training.py` |

## Embodiment / Actuation WM And Humanoid Readiness

| Subsystem | Current State | Debt / Bug | Next Action | Verify |
| --- | --- | --- | --- | --- |
| G1 primary doctrine | `unitree_g1` and `bipedal_whole_body_unitree_g1` are canonical primary target. | No real G1 hardware/sim proof. | Keep hygiene gate in CI and preserve fixed-base curriculum boundaries. | `python3 scripts/economic_world_model/check_g1_primary_env_hygiene.py --output-dir /tmp/g1_check` |
| Morphology / bipedal chassis | G1 morphology, 29-DoF chassis, support/balance schemas, and readiness receipts exist. | Mypy clusters in `bipedal_chassis.py`; hardware-calibrated limits remain unavailable. | Type optional float handling and list annotations; do not alter physical constants without evidence. | `python3 -m mypy --follow-imports=silent src/world_model/embodiment_actuation/bipedal_chassis.py` |
| Phase 4 Unitree local harnesses | ROS2/SDK2-shaped dry-run, trace, watchdog, safety, recovery, MuJoCo probe surfaces exist. | Real ROS2/colcon, SDK2 Linux runtime, live streams, command echo, calibration, operator drills remain unavailable locally. | Keep unavailable receipts; run real proof only on configured host/pod/hardware. | `python3 -m pytest -q tests/test_humanoid_phase4_unitree_local_harnesses.py tests/test_humanoid_phase4_unitree_runtime_evidence_bridge.py` |
| Neural architecture scaffolds | JEPA/ACT/Diffusion/topology-contrastive and body-control scaffolds exist. | No trained whole-body policy, no real multi-joint demonstration corpus. | Next local work is type/static cleanup; training waits for GPU/corpus. | `python3 -m pytest -q tests/test_embodiment_actuation_phase34.py tests/test_humanoid_phase35_bipedal_chassis.py` |
| SAC curriculum loop | SAC now emits G1 target metadata while executing dishwashing curriculum. | Still uses fixed-base dishwashing source; not a G1 control policy. | Keep it as plumbing/curriculum only; use `g1_sac_training` manifest for future proof-of-life training. | `python3 -m pytest -q tests/test_runpod_launch_profiles.py` |

## Economic WM

| Subsystem | Current State | Debt / Bug | Next Action | Verify |
| --- | --- | --- | --- | --- |
| Scaffold/state/allocation | EconomicState, allocation envelopes, lower-WM rows, resource surfaces, shadow work orders, and supervision records exist. | No trained estimator/dynamics/allocator/governance components. | Preserve scaffold-only posture until provider/GPU/corpus receipts exist. | `python3 -m pytest -q tests/test_economic_world_model_scaffold.py tests/test_economic_wm_phase5_local_prep.py` |
| Lower-WM ingestion | Canonical lower-WM refs are preserved and maturity sweep is explicit. | Production-ready lower-WM refs are still zero in local evidence. | Keep maturity sweep as preflight before any training. | `python3 -m pytest -q tests/test_economic_wm_lower_wm_consumption.py tests/test_economic_wm_lower_wm_maturity_sweep.py` |
| Neural manifest/trainer scaffold | Six learned components and non-training trainer scaffold exist. | GPU training required for five components; static mypy debt in `resource_surfaces.py` and `provider_runbook_validation.py`. | Fix Economic WM mypy cluster as a small isolated pass. | `python3 -m mypy --follow-imports=silent src/world_model/economic_world_model` |
| Run manifests | Example manifests and launch profiles pass hygiene. | Actual launch prerequisites are missing locally. | Install/auth RunPod before provider/loop/train execution. | `./scripts/runpod/ensure_cli.sh` |

## WM Transport, Semantic Runtime, And Meta-Regal WM

| Subsystem | Current State | Debt / Bug | Next Action | Verify |
| --- | --- | --- | --- | --- |
| Phase 6 transport | Contracts, exporters/receivers, rows, topology, uncertainty, neural manifest, losses, and advisory runtime exist. | No bridge/receiver training, no latency/topology benchmarks, no live authority. | Static cleanup in transport/orchestration before future bridge training. | `python3 -m pytest -q tests/test_wm_transport_phase6_scaffold.py tests/test_wm_transport_phase63_neural_scaffold.py tests/test_wm_transport_phase64_runtime_eval.py` |
| Semantic runtime/orchestration | Runtime scorers, semantic policies, queue selection, and learned helper seams exist. | Highest mypy bucket is orchestration/semantic: `pipeline_manager.py`, `orchestration_transformer.py`, `semantic_feedback_packets.py`, `semantic_world_model.py`. | Burn down type errors in orchestration first because it is shared by many WMs. | `python3 -m mypy --follow-imports=silent src/orchestrator src/world_model/semantic_world_model.py src/world_model/semantic_feedback_packets.py` |
| Phase 6.5 local meta-node | MetaNodeState, trainer/loss scaffold, robustness reports, and denied gates exist. | No trained meta-node models. | Keep as training-contract scaffold until lower-WM receipts and GPU exist. | `python3 -m pytest -q tests/test_humanoid_phase65_meta_node_trainer_scaffold.py` |
| Phase 7 meta-regal control | Stage-A typed surfaces, shadow runtime, eval harness, signal adapters, and hypernetwork scaffold exist. | Mypy debt in `phase7_eval.py`; no authority, no trained composition, no lower-WM live proof. | Fix type errors and keep authority denied. | `python3 -m mypy --follow-imports=silent src/world_model/humanoid_readiness/phase7_eval.py src/world_model/humanoid_readiness/phase7_hypernetwork.py` |

## Bio / Neuro Inspiration Implementation Status

The bio/neuro items are not uniformly "implemented." The current honest state:

| Inspiration | Status | Evidence / Gap |
| --- | --- | --- |
| Efference copy / corollary discharge | Local substrate wired after this sweep | `SelfMotionExpectation` and `SelfDisturbanceReceipt` now exist as typed local surfaces. Still missing: trained predictor, real observed-motion corpus, and automatic runtime-loop emission. |
| Active sensing | Local proposal/receipt substrate wired after this sweep | `ActiveSensingProposal` and `ActiveSensingReceipt` now exist. Still missing: executed active-sensing actions, measured information gain, and full Economic WM value-of-information shaping. |
| Neuromodulation / allostasis | Local broadcast/ack substrate wired after this sweep | `RegimeBroadcast` and `RegimeAcknowledgmentReceipt` now exist as low-bandwidth advisory surfaces. Still missing: trained Economic regime estimator, real downstream adaptation, and meta-regal composition training. |
| Plasticity gating | Partially implemented | Perception promotion gates, benchmark evidence, provisional-vs-promotion logic, and `promotion_eligible=false` discipline are real. Full training eligibility/consolidation receipt family is not complete. |
| Motor synergies + interoception | Local heuristic substrate wired after this sweep | `SynergyCodebookEntry` and `InteroceptiveState` now exist. Still missing: learned codebook, real interoceptive telemetry, and hardware-calibrated activation patterns. |
| Immune-style anomaly governance | Local anomaly/escalation substrate wired after this sweep | `AnomalySuspicionReceipt` and `GovernanceEscalationEvent` now exist with abstention. Still missing: trained anomaly critics and meta-regal immune-style composition. |

## Multi-WM Roadmap Status Answer

Implemented locally:

- Lower-WM structural scaffolds and receipts across Sim/Synth/Physics,
  Perception/Grounding, Embodiment/Actuation, Economic WM, Transport, Phase 6.5,
  and Phase 7.
- Bounded neural seams and non-training neural manifests for major WMs.
- G1 primary doctrine and posture hygiene.
- RunPod manifest prep for provider bring-up, loop runs, and training.
- Honest unavailable/provider/GPU/hardware boundaries.

Not implemented as proof:

- Real provider execution for SAM/DINO/SigLIP/V-JEPA/depth/OpenVLA-style lanes.
- Real Isaac/Unitree/Holosoma GPU/runtime proof.
- Trained lower-WM, Economic WM, transport, meta-node, or Phase 7 models.
- Real G1 sim/hardware dispatch, ROS2 publish, SDK2 writes, command echo, or
  operator recovery traces.
- Promotion-grade benchmarks.
- Phase 8 weekly production loop operations.

## Ranked Next-Session Work

1. **Keep full-repo static hygiene green**
   - What: use `python3 -m mypy src/` and `python3 -m ruff check .` as
     guardrails before and after local wiring work.
   - Why now: the repo is static-clean; letting it regress would hide real
     provider/GPU/hardware blockers behind local noise.
   - Verify: `python3 -m mypy src/`; `python3 -m ruff check .`.
   - Do not: broaden static cleanup into reward math, weight writes, or
     behavior-changing refactors.

2. **Provider bring-up readiness ledger**
   - What: create a typed local provider ledger that maps SAM/SAM3D,
     DINO/SigLIP, V-JEPA2, OpenVLA, Isaac/Unitree, and Holosoma to commands,
     expected receipts, unavailable posture, RunPod profile, and owner WM.
   - Why now: the provider backlog is spread across roadmap and JSON backlog
     files; the next provider day should start from an executable ledger.
   - Verify: ledger lint/checker plus provider manifest generation when
     RunPod prerequisites exist.
   - Do not: download weights, run providers, or claim provider execution
     locally.

3. **LeRobot video to replay to perception receipts**
   - What: normalize video/camera receipts into replay rows and perception
     samples while preserving ids, frame/step/timestamp/camera keys, sidecars,
     runtime refs, provenance, and unavailable posture.
   - Why now: this advances lower-WM evidence plumbing without downloads,
     provider execution, or GPU truth claims.
   - Verify: video receipt -> replay rows -> perception sample tests.
   - Do not: treat placeholder/flattened CPU features as promotion-grade
     provider features.

4. **Unitree event spines into Phase 6.4 advisory eval**
   - What: use existing Unitree event-spine producers/refs and wire fresh
     `event_spine_ref` values into Phase 6.4 advisory runtime/eval receipts.
   - Why now: this gives transport eval better local lower-WM labels without
     inventing a parallel event model.
   - Verify: refs -> receipts/evals tests with blocker/unavailable gates.
   - Do not: bypass receivers, grant authority, or claim hardware/provider
     proof.

5. **Neural trainability audit**
   - What: emit additive JSON/JSONL/doc artifacts over neural/seam/encoder/
     policy/head/bridge/receiver/trainer surfaces with executable follow-up
     rows and plane routing.
   - Why now: the static-clean repo can now distinguish code gaps from GPU,
     provider, hardware, data, and benchmark blockers.
   - Verify: audit checker plus static checks.
   - Do not: train, write weights, or mark blocked components promotion
     eligible.

6. **Bio/neuro receipt join wiring**
   - What: join the already-wired local substrate receipts into normal
     lower-WM/Economic consumption rows.
   - Why now: the substrate should become queryable evidence without
     pretending it is trained or promotion-grade.
   - Verify: substrate checker plus focused lower-WM consumption tests.
   - Do not: treat the joins as active sensing execution, interoceptive
     hardware telemetry, trained anomaly critics, or Phase 7 authority.

## Updated Goal Message For Next Session

Boot in `/Users/amarmurray/robotics-vp-core` on `main`.

Read `AGENTS.md`,
`codex_skills/economic-world-model-roadmap/SKILL.md`,
`codex_skills/roadmap-execution-companion/SKILL.md`,
`docs/economic_world_model/wm_subsystem_debt_sweep_2026_06_01.md`, and
`docs/economic_world_model/multi_wm_unwired_surface_audit_2026_06_01.md`.

Burn down all remaining local subsystem debt in
`wm_subsystem_debt_sweep_2026_06_01.md` continuously and robustly. Full-repo
`mypy src/` and `ruff check .` are currently clean; keep them green as
guardrails after every tranche.

Further, burn down and wire all remaining local items from
`multi_wm_unwired_surface_audit_2026_06_01.md`: provider bring-up readiness
ledger, LeRobot video-to-replay-to-perception receipt plumbing, Unitree
event-spine refs into Phase 6.4 advisory runtime/eval receipts, neural
trainability audit artifacts, bio/neuro receipt joins into lower-WM/Economic
consumption rows, and bounded Phase 7 receipt consumption through existing
adapters only. Keep the now-hardened Unitree rosbag2/MCAP unavailable receipts
green; do not claim real rosbag2/MCAP imports without real files, installed
dependencies, and parser execution.

Keep G1/bipedal whole-body primary. Treat stable-base mobile manipulation as
fallback/degraded mode and fixed-base tabletop/workcell/dishwashing as
curriculum/regression only. Keep all work additive, typed, receipt-emitting,
and honest about unavailable GPU/provider/hardware proof. Do not mutate frozen
Phase B math, stable checkpoints, reward/controller equations, trust/w_econ
lambda math, or policy authority. Do not write weights, do not claim
promotion, and do not expand Phase 7 abstractions unless lower-WM receipts
force it. Run focused verification after each tranche, refresh broad
`python3 -m mypy src/` and `python3 -m ruff check .` counts, and update
`docs/economic_world_model/progress_log.md` plus
`docs/economic_world_model/implementation_notes.md`.
