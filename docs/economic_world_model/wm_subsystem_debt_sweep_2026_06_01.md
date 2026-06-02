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

The residual debt is now full-repo static hygiene outside that narrowed gate.
It should still be burned down, because these modules are lower-WM producers,
trainer/runtime lanes, curriculum sources, or receipt consumers. They are not
automatically obsolete just because the WM architecture now governs them.

Current residual broad ruff:

| Area | Count |
| --- | ---: |
| `scripts/` | 38 |
| `src/utils/` | 10 |
| `src/analytics/` | 9 |
| `tests/` | 9 |
| `src/orchestrator/` | 8 |
| `src/learning/` | 7 |
| `src/sima2/` | 7 |
| `third_party/` | 7 |
| `src/inference/` | 6 |
| `src/sima/` | 6 |
| `src/config/` | 4 |
| `src/contracts/` | 4 |
| `src/tfd/` | 4 |
| other checked-in support surfaces | 29 |
| **Total** | **148** |

Current residual broad ruff by code:

| Code | Meaning | Count | Disposition |
| --- | --- | ---: | --- |
| `F401` | unused imports | 85 | mostly safe mechanical cleanup |
| `F841` | unused locals | 43 | mostly safe, but inspect demos/trainers where variables imply missing receipts |
| `F821` | undefined names | 6 | treat as bugs before mechanical cleanup |
| other `E`/`F` rules | style/ambiguous names/bare except | 14 | mechanical except where exceptions hide provider/runtime failures |

Current residual full-repo mypy:

| Area | Count |
| --- | ---: |
| `src/regal/` | 6 |
| `src/analytics/` | 5 |
| `src/encoders/` | 5 |
| `src/evidence/` | 5 |
| `src/policies/` | 5 |
| `src/sima2/` | 4 |
| `third_party/` | 4 |
| `src/epiplexity/` | 4 |
| `src/diffusion/` | 4 |
| `src/inference/` | 4 |
| `src/embodiment/` | 3 |
| `src/utils/` | 3 |
| `src/physics/` | 3 |
| `src/datasets/` | 3 |
| `src/phase_h/` | 2 |
| other checked-in support surfaces | 10 |
| **Total actual `error:` records** | **70** |

Current residual full-repo mypy by kind:

| Kind | Count | Meaning |
| --- | ---: | --- |
| `arg-type` | 28 | interface drift and weak payload narrowing |
| `assignment` | 16 | optional dependency/module typing, tensor/list reuse, schema mismatch |
| `var-annotated` | 5 | untyped mutable containers |
| `attr-defined` | 5 | object payloads not narrowed before attribute access |
| `dict-item` | 5 | dicts typed too narrowly for receipt/config payloads |
| `call-arg` | 3 | stale call signatures and constructor drift |
| `return-value` | 2 | declared receipt/runtime outputs too narrow |
| `import-untyped` | 1 | installed dependencies without stubs |
| `operator` | 1 | narrowed tensor/optional arithmetic gaps |
| `name-defined` | 1 | missing or stale definitions |
| `override` | 1 | interface override drift |
| `import-not-found` | 1 | missing optional dependency |
| `index` | 1 | weakly typed enum/list indices |

Legacy/support-surface disposition:

| Surface family | Not superseded because | What should happen |
| --- | --- | --- |
| `src/envs/`, `src/physics/`, `src/scenarios/`, `src/datasets/`, `src/replay/` | curriculum, regression, replay, and data-generation substrate for WMs | keep, type, and posture-tag as fixed-base curriculum or G1-relevant producer; do not treat fixed-base success as humanoid proof |
| `src/vision/`, `src/scene/`, `src/sima2/`, `src/vla/` | provider-facing perception/semantic algorithms and VLA scaffolds | wrap as Perception/Grounding producers or advisory provider adapters; real provider outputs remain external proof |
| `src/rl/`, `src/hrl/`, `src/policies/`, `src/process_reward/`, `src/encoders/`, `src/representation/` | trainer/runtime lanes for future lower-WM or policy components | keep but gate with manifests, receipts, no weight writes in local cleanup, and no promotion claims |
| `src/motor_backend/`, `src/embodiment/`, `src/ingestion/`, `src/runtime/` | hardware/provider/runtime adapter layer | keep as honest unavailable/proof-emitting adapters; do not collapse stubs into hardware truth |
| `src/economics/`, `src/valuation/`, `src/ontology/`, `src/evidence/`, `src/contracts/` | cross-cutting economic, receipt, and evidence contracts | keep; avoid mutating frozen Phase B math or controller equations |
| `scripts/`, `third_party/`, old demos/trainers | operational glue and historical smoke/prototype entrypoints | fix undefined names and safe lint; then either document as legacy/dev-only or migrate into receipt-emitting scripts |

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

1. **Full-repo mypy burn-down by support-surface family**
   - What: burn down the residual full-repo mypy debt in this order:
     `src/regal/`, `src/analytics/`, `src/encoders/`, `src/evidence/`,
     `src/policies/`, then `src/sima2/`, `third_party/`, `src/epiplexity/`,
     `src/diffusion/`, `src/inference/`, and the remaining lower-count
     support surfaces.
   - Why now: these are the lower-WM producers, provider adapters,
     curriculum/replay surfaces, and trainer/runtime lanes that the WM stack
     consumes. Leaving them noisy makes future provider/GPU proof harder to
     trust.
   - Verify: `python3 -m mypy src/`.
   - Do not: change reward math, write weights, promote local scaffolds, or
     convert fixed-base curriculum into G1 proof while typing.

2. **Full-repo ruff burn-down with bug-first handling**
   - What: fix `F821` undefined names first, then safe `F401`/`F841`
     mechanical cleanup, then the remaining small `E`/`F` rules.
   - Why now: most residual ruff is mechanical, but undefined names and bare
     exceptions can hide broken scripts or provider/runtime blockers.
   - Verify: `python3 -m ruff check .`.
   - Do not: delete historical scripts blindly; either keep them working,
     mark them dev-only, or migrate them into receipt-emitting paths.

3. **Provider bring-up readiness ledger**
   - What: create a typed local provider ledger that maps SAM/SAM3D,
     DINO/SigLIP, V-JEPA2, OpenVLA, Isaac/Unitree, and Holosoma to commands,
     expected receipts, unavailable posture, RunPod profile, and owner WM.
   - Why now: the provider backlog is spread across roadmap and JSON backlog
     files; the next provider day should start from an executable ledger.
   - Verify: ledger lint/checker plus provider manifest generation when
     RunPod prerequisites exist.
   - Do not: download weights, run providers, or claim provider execution
     locally.

4. **Bio/neuro receipt join wiring**
   - What: join the already-wired local substrate receipts into normal
     lower-WM/Economic consumption rows.
   - Why now: the substrate should become queryable evidence without
     pretending it is trained or promotion-grade.
   - Verify: substrate checker plus focused lower-WM consumption tests.
   - Do not: treat the joins as active sensing execution, interoceptive
     hardware telemetry, trained anomaly critics, or Phase 7 authority.

5. **Legacy/support-surface disposition**
   - What: for each now-typed support family, decide whether it is a
     lower-WM producer, trainer/runtime lane, curriculum/regression source,
     provider/hardware adapter, or legacy/dev-only tool.
   - Why now: the repo is mostly encompassed by the WM architecture, but not
     every directory belongs under `src/world_model/`; some should remain as
     substrate with explicit contracts.
   - Verify: doc updates plus focused tests for the family touched.
   - Do not: move modules or rename public APIs unless the tests and docs prove
     the migration boundary.

6. **RunPod prerequisite closeout**
   - What: install/auth `runpodctl`, set `RUNPOD_API_KEY`, and set
     `RUNPOD_VOLUME_ID`.
   - Why now: provider/loop/train profiles and the new provider readiness
     ledger are still local planning surfaces until these prerequisites exist.
   - Verify: `./scripts/runpod/ensure_cli.sh`.
   - Do not: call manifest preparation or ledger generation a remote run.

5. **External/provider/hardware proof lanes**
   - What: after local static debt is quiet, run the provider bring-up,
     G1 loop, and training profiles only on configured RunPod/provider/hardware
     planes with manifests under `.agent/runs/<run_id>/manifest.json`.
   - Why now: this is the remaining gap between local typed receipts and real
     roadmap proof.
   - Verify: provider/runtime receipts, run manifests, focused smoke tests, and
     nightly audit updates.
   - Do not: claim provider, GPU, Isaac/Unitree, Holosoma, or promotion proof
     without real execution artifacts.
