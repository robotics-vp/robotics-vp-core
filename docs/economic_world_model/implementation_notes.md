# Economic World Model Implementation Notes

## 2026-05-21 - Economic WM canonical consumption and neural manifest

### What changed

- Added a lower-WM canonical consumption preflight for Economic WM rows.
- Added local canonical reference-pack compilation when rows do not yet carry direct lower-WM state refs.
- Added `economic_wm_canonical_consumption_row_v1` rows with `source_refs.canonical_lower_wm_refs` for Perception / Grounding, Sim / Synth / Physics, and Embodiment / Actuation.
- Added an Economic WM neural architecture manifest covering:
  - datapack composition network
  - economic state estimator
  - economic dynamics model
  - distributional Pareto allocator
  - discrete receding-horizon allocator
  - governance reciprocity compiler
- Each neural component now carries explicit input surfaces, output surfaces, loss families, training signals, promotion gates, runtime plane, blockers, `training_ready=false`, and `promotion_eligible=false`.

### Boundary

This is canonical-consumption and neural-topology preparation only. The current reference packs are compiled locally from existing row identity when source rows lack direct lower-WM refs; that is structural scaffolding, not proof that lower-WM producers emitted them natively. No GPU training, provider bring-up, promotion-grade benchmark, external teacher invocation, or reward/trust/`w_econ`/lambda-controller mutation is claimed.

### Verification

- `python3 -m ruff check src/world_model/economic_world_model scripts/economic_world_model/prepare_economic_wm_lower_wm_consumption_preflight.py scripts/economic_world_model/build_economic_wm_neural_architecture_manifest.py tests/test_economic_wm_lower_wm_consumption.py tests/test_economic_wm_neural_architecture_manifest.py` (`All checks passed!`)
- `python3 -m compileall src/world_model/economic_world_model scripts/economic_world_model/prepare_economic_wm_lower_wm_consumption_preflight.py scripts/economic_world_model/build_economic_wm_neural_architecture_manifest.py -q`
- `python3 -m pytest -q tests/test_economic_wm_lower_wm_consumption.py tests/test_economic_wm_neural_architecture_manifest.py` (`5 passed`)
- `python3 scripts/economic_world_model/prepare_economic_wm_lower_wm_consumption_preflight.py --output-dir artifacts/economic_world_model/economic_wm_lower_wm_consumption_preflight --corpus-manifest artifacts/economic_world_model/economic_wm_training_rows/economic_wm_training_corpus_manifest_v1.json --rows artifacts/economic_world_model/economic_wm_training_rows/economic_wm_replay_feature_rows_v1.jsonl` (`status=ok`, `compiled_reference_count=15`, `promotion_eligible=false`)
- `python3 scripts/economic_world_model/build_economic_wm_neural_architecture_manifest.py --output-dir artifacts/economic_world_model/economic_wm_neural_architecture_manifest --lower-wm-preflight artifacts/economic_world_model/economic_wm_lower_wm_consumption_preflight/economic_wm_lower_wm_consumption_preflight_v1.json` (`component_count=6`, `gpu_train_required_count=5`, `promotion_eligible=false`)
- `python3 scripts/economic_world_model/nightly_audit.py --output-json artifacts/economic_world_model/nightly_audit_summary.json --output-markdown artifacts/economic_world_model/nightly_audit_summary.md` (`status=ok`)
- `python3 -m compileall src/ -q && python3 -m pytest tests/ -q` (`1691 passed, 2 skipped, 32 warnings`)

## 2026-05-21 - Economic WM provider runbook validation

### What changed

- Added a validation report layer for Economic WM provider runbooks.
- Added checks for template-only authority, launch denial, pending manifest status, empty runtime fields, external/provider/GPU guard commands, required template-key coverage, and no promotion posture.
- Added a CLI validator that writes JSON and Markdown validation artifacts.

### Boundary

This validates storage safety only. It is explicitly `safe_for_launch=false` and does not turn any template into provider, GPU, benchmark, or promotion evidence.

### Verification

- `python3 -m ruff check src/world_model/economic_world_model scripts/economic_world_model/validate_economic_wm_provider_runbook.py tests/test_economic_wm_provider_runbook_validation.py` (`All checks passed!`)
- `python3 -m compileall src/world_model/economic_world_model scripts/economic_world_model/validate_economic_wm_provider_runbook.py -q`
- `python3 -m pytest -q tests/test_economic_wm_provider_runbook_validation.py` (`3 passed`)
- `python3 scripts/economic_world_model/validate_economic_wm_provider_runbook.py --output-dir artifacts/economic_world_model/economic_wm_provider_runbook_validation --runbook artifacts/economic_world_model/economic_wm_provider_runbook/economic_wm_provider_runbook_v1.json --manifest-template-dir artifacts/economic_world_model/economic_wm_provider_runbook/manifest_templates` (`status=ok`, `safe_for_template_storage=true`, `safe_for_launch=false`)
- `python3 scripts/economic_world_model/nightly_audit.py --output-json artifacts/economic_world_model/nightly_audit_summary.json --output-markdown artifacts/economic_world_model/nightly_audit_summary.md` (`status: ok`)

## 2026-05-21 - Economic WM provider runbook templates

### What changed

- Added a template-only Economic WM provider runbook layer downstream of the teacher/provider evidence contract.
- Added manifest-shaped templates for non-stub teacher runtime invocation, provider truth receipts, GPU training runtime receipts, promotion benchmark evidence, and local replay-row linkage checks.
- Added a CLI compiler that writes JSON, Markdown, and per-template manifest stubs under `artifacts/economic_world_model/economic_wm_provider_runbook/`.

### Boundary

This is runbook prep only. It does not launch RunPod, run a provider, execute GPU training, promote a model, or mutate reward math. External/provider/GPU templates include guard commands and remain `launch_allowed=false` until a real execution path replaces them and records receipts.

### Verification

- `python3 -m ruff check src/world_model/economic_world_model scripts/economic_world_model/compile_economic_wm_provider_runbook.py tests/test_economic_wm_provider_runbook.py` (`All checks passed!`)
- `python3 -m compileall src/world_model/economic_world_model scripts/economic_world_model/compile_economic_wm_provider_runbook.py -q`
- `python3 -m pytest -q tests/test_economic_wm_provider_runbook.py` (`2 passed`)
- `python3 scripts/economic_world_model/compile_economic_wm_provider_runbook.py --output-dir artifacts/economic_world_model/economic_wm_provider_runbook --contract artifacts/economic_world_model/economic_wm_teacher_provider_contracts/economic_wm_teacher_provider_contract_v1.json` (`authority_class=runbook_template_only`, `template_count=5`, `promotion_eligible=false`)
- `python3 scripts/economic_world_model/nightly_audit.py --output-json artifacts/economic_world_model/nightly_audit_summary.json --output-markdown artifacts/economic_world_model/nightly_audit_summary.md` (`status: ok`)

## 2026-05-21 - Economic WM teacher/provider evidence contracts

### What changed

- Added typed Economic WM evidence requirements for non-stub teacher runtime, external provider truth, promotion-grade benchmark evidence, GPU runtime receipts, and replay-row linkage integrity.
- Added `EconomicWMTeacherProviderContract` as a contract-prep artifact downstream of the shadow allocation eval.
- Added a CLI that emits JSON and Markdown contract-pack artifacts.

### Boundary

This is evidence-contract prep only. It does not run a provider, run GPU training, promote a model, or mutate reward math. The current output explicitly reports `provider_bringup_ready=false`, `gpu_training_ready=false`, and `promotion_eligible=false`.

### Verification

- `python3 -m ruff check src/world_model/economic_world_model/evidence_contracts.py src/world_model/economic_world_model/__init__.py scripts/economic_world_model/prepare_economic_wm_teacher_provider_contracts.py tests/test_economic_wm_teacher_provider_contracts.py` -> `All checks passed!`
- `python3 -m compileall src/world_model/economic_world_model scripts/economic_world_model/prepare_economic_wm_teacher_provider_contracts.py -q`
- `python3 -m pytest -q tests/test_economic_wm_teacher_provider_contracts.py` -> `2 passed`
- `python3 scripts/economic_world_model/prepare_economic_wm_teacher_provider_contracts.py --output-dir artifacts/economic_world_model/economic_wm_teacher_provider_contracts --scaffold-report artifacts/economic_world_model/economic_wm_scaffold/economic_wm_scaffold_report_v1.json --allocation-eval artifacts/economic_world_model/economic_wm_shadow_allocation_eval/economic_wm_shadow_allocation_eval_v1.json --corpus-manifest artifacts/economic_world_model/economic_wm_training_rows/economic_wm_training_corpus_manifest_v1.json --rows artifacts/economic_world_model/economic_wm_training_rows/economic_wm_replay_feature_rows_v1.jsonl` -> `authority_class=evidence_contract_only`, `promotion_eligible=false`
- `python3 scripts/economic_world_model/nightly_audit.py --output-json artifacts/economic_world_model/nightly_audit_summary.json --output-markdown artifacts/economic_world_model/nightly_audit_summary.md` -> `status: ok`

## 2026-05-21 - Economic WM shadow allocation eval

### What changed

- Added a shadow-only Economic WM allocation evaluator over local scaffold and row-corpus artifacts.
- Added candidate scoring for benchmark replay curation, shadow-gap replay closure, teacher/provider evidence-contract preparation, and denied GPU training.
- Added a CLI that emits `economic_wm_shadow_allocation_eval_v1` JSON and Markdown reports.

### Boundary

This is advisory only. It ranks local allocation candidates but does not execute allocation, mutate rewards, train a model, promote a model, or imply provider/GPU bring-up.

### Verification

- `python3 -m ruff check src/world_model/economic_world_model scripts/economic_world_model/evaluate_economic_wm_shadow_allocations.py tests/test_economic_wm_shadow_allocation_eval.py` -> `All checks passed!`
- `python3 -m compileall src/world_model/economic_world_model scripts/economic_world_model/evaluate_economic_wm_shadow_allocations.py -q`
- `python3 -m pytest -q tests/test_economic_wm_shadow_allocation_eval.py` -> `2 passed`
- `python3 scripts/economic_world_model/evaluate_economic_wm_shadow_allocations.py --output-dir artifacts/economic_world_model/economic_wm_shadow_allocation_eval --scaffold-report artifacts/economic_world_model/economic_wm_scaffold/economic_wm_scaffold_report_v1.json --corpus-manifest artifacts/economic_world_model/economic_wm_training_rows/economic_wm_training_corpus_manifest_v1.json --rows artifacts/economic_world_model/economic_wm_training_rows/economic_wm_replay_feature_rows_v1.jsonl` -> `recommended_candidate=prepare_teacher_provider_evidence_contracts`, `promotion_eligible=false`
- `python3 scripts/economic_world_model/nightly_audit.py --output-json artifacts/economic_world_model/nightly_audit_summary.json --output-markdown artifacts/economic_world_model/nightly_audit_summary.md` -> `status: ok`

## 2026-05-21 - Economic WM training-row materialization

### What changed

- Added a local Economic WM row-corpus layer under `src/world_model/economic_world_model/training_rows.py`.
- Added `economic_wm_replay_feature_row_v1` rows with benchmark/shadow truth, deterministic feature vectors, deterministic target vectors, sidecar refs, and denied-promotion reasons.
- Added `economic_wm_training_corpus_manifest_v1` to summarize row counts, scaffold linkage, blocker posture, and artifact refs.
- Added a CLI materializer for Stage-1 proposal admissions.

### Boundary

This materializes rows only. It does not execute training, bring up providers, promote a model, or grant Economic WM outputs authority over frozen reward, trust-net, `w_econ`, or lambda-controller math.

### Verification

- `python3 -m ruff check src/world_model/economic_world_model scripts/economic_world_model/materialize_economic_wm_training_rows.py tests/test_economic_wm_training_rows.py` -> `All checks passed!`
- `python3 -m compileall src/world_model/economic_world_model scripts/economic_world_model/materialize_economic_wm_training_rows.py -q`
- `python3 -m pytest -q tests/test_economic_wm_training_rows.py` -> `2 passed`
- `python3 scripts/economic_world_model/materialize_economic_wm_training_rows.py --output-dir artifacts/economic_world_model/economic_wm_training_rows --scaffold-report artifacts/economic_world_model/economic_wm_scaffold/economic_wm_scaffold_report_v1.json` -> `row_count=5`, `benchmark_ready_count=2`, `shadow_only_count=3`, `promotion_eligible=false`
- `python3 scripts/economic_world_model/nightly_audit.py --output-json artifacts/economic_world_model/nightly_audit_summary.json --output-markdown artifacts/economic_world_model/nightly_audit_summary.md` -> `status: ok`

## 2026-05-21 - Economic WM scaffold artifacts

### What changed

- Added the first native Economic WM scaffold package under `src/world_model/economic_world_model/`.
- Added deterministic `economic_state_v1`, `allocation_envelope_v1`, and `economic_wm_scaffold_report_v1` artifacts.
- Added a CLI builder that runs or consumes the Economic WM entry preflight and writes JSON plus a Markdown summary under `artifacts/`.
- Added tests for state/envelope derivation and report round-trip behavior.

### Boundary

This is a scaffold-only Economic WM entry. It exposes resource reservoirs, flow fields, dissipation fields, bottlenecks, opportunity fields, and a denied-action envelope. It does not train a model, promote a model, run GPU/provider bring-up, or mutate frozen Phase B reward, trust-net, `w_econ`, or lambda-controller math.

### Verification

- `python3 -m ruff check src/world_model/economic_world_model scripts/economic_world_model/build_economic_wm_scaffold.py tests/test_economic_world_model_scaffold.py` -> `All checks passed!`
- `python3 -m compileall src/world_model/economic_world_model scripts/economic_world_model/build_economic_wm_scaffold.py -q`
- `python3 -m pytest -q tests/test_economic_world_model_scaffold.py` -> `2 passed`
- `python3 scripts/economic_world_model/build_economic_wm_scaffold.py --output-dir artifacts/economic_world_model/economic_wm_scaffold` -> `readiness_class=scaffold_ready_training_blocked`, `promotion_eligible=false`
- `python3 scripts/economic_world_model/nightly_audit.py --output-json artifacts/economic_world_model/nightly_audit_summary.json --output-markdown artifacts/economic_world_model/nightly_audit_summary.md` -> `status: ok`

## 2026-05-21 - Economic WM entry preflight

### What changed

- Added a typed Economic WM entry preflight report.
- Added a CLI preflight that runs or consumes the Stage-1 bridge-readiness sweep.
- The preflight emits separate scaffold and training readiness booleans.
- Refreshed the training backlog around governed-video and future Economic WM training lanes.

### Boundary

This authorizes scaffold work only. It does not train, promote, or claim GPU/provider-backed Economic WM functionality.

### Verification

- `python3 -m ruff check src/economics/economic_wm_entry.py src/economics/__init__.py scripts/economic_world_model/economic_wm_entry_preflight.py tests/test_economic_wm_entry_preflight.py`
- `python3 -m compileall src/economics scripts/economic_world_model/economic_wm_entry_preflight.py -q`
- `python3 -m pytest -q tests/test_economic_wm_entry_preflight.py` -> `3 passed, 1 warning`
- `python3 scripts/economic_world_model/economic_wm_entry_preflight.py --output-dir artifacts/economic_world_model/economic_wm_entry_preflight` -> `readiness_class=scaffold_ready_training_blocked`
- `python3 scripts/economic_world_model/nightly_audit.py --output-json artifacts/economic_world_model/nightly_audit_summary.json --output-markdown artifacts/economic_world_model/nightly_audit_summary.md` -> `status: ok`

## 2026-05-20 - Stage-1 bridge readiness sweep

### What changed

- Added a local sweep script for Stage-1 governed-video readiness.
- The sweep creates five manifest variants and runs Stage-1, replay import, RLDS export, and LeRobot export.
- The report validates benchmark readiness, calibration class, reconstruction grounding class, reconstruction-training eligibility, blocking preconditions, and bridge truth preservation.

### Boundary

This is a structural replay/export sweep. It does not run GPU training, bring up providers, execute non-stub SceneTracks, or promote any model. Artifacts remain local under `artifacts/` and are intentionally not tracked.

### Verification

- `python3 -m ruff check scripts/economic_world_model/sweep_stage1_bridge_readiness.py tests/test_stage1_bridge_readiness_sweep.py`
- `python3 -m compileall scripts/economic_world_model/sweep_stage1_bridge_readiness.py -q`
- `python3 -m pytest -q tests/test_stage1_bridge_readiness_sweep.py` -> `1 passed, 1 warning`
- `python3 scripts/economic_world_model/sweep_stage1_bridge_readiness.py --output-dir artifacts/economic_world_model/stage1_bridge_readiness_sweep` -> `status: ok`, `scenario_count=5`, `benchmark_ready_count=2`, `shadow_only_count=3`
- `python3 scripts/economic_world_model/nightly_audit.py --output-json artifacts/economic_world_model/nightly_audit_summary.json --output-markdown artifacts/economic_world_model/nightly_audit_summary.md` -> `status: ok`

## 2026-05-20 - Benchmark truth through replay bridges

### What changed

- `ingest_governed_video_admission_log` now stores the source benchmark gate in canonical replay metadata.
- RLDS and LeRobot bridge exports include benchmark gates and future-training signals in row metadata.
- RLDS and LeRobot rehydration restores those payloads back into canonical replay rows.

### Boundary

This preserves truth metadata only. It does not make public bridge rows lossless, train an Economic WM, or change benchmark-readiness criteria.

### Verification

- `python3 -m ruff check src/replay/importers.py src/dataset_bridges/rlds_bridge.py src/dataset_bridges/lerobot_bridge.py tests/test_dataset_bridges.py tests/test_replay_dataset.py`
- `python3 -m compileall src/replay src/dataset_bridges -q`
- `python3 -m pytest -q tests/test_dataset_bridges.py tests/test_replay_dataset.py` -> `12 passed, 2 warnings`
- `python3 scripts/economic_world_model/nightly_audit.py --output-json artifacts/economic_world_model/nightly_audit_summary.json --output-markdown artifacts/economic_world_model/nightly_audit_summary.md` -> `status: ok`

## 2026-05-20 - Camera-calibrated benchmark gating

### What changed

- Added optional camera-calibration requirements to the benchmark-gating helper.
- Stage-1 now enables that requirement for governed-video benchmark readiness.
- Stage-1 benchmark metadata derives calibration class from the same normalized sensor bundle used to build reconstruction sidecars.
- Stage-1 SceneTracks truth payload assembly preserves metadata-level SceneTracks refs before falling back to top-level refs.

### Boundary

This is a readiness gate only. It does not calibrate cameras, run SceneTracks, train a model, or promote any provider. It prevents benchmark-ready claims when calibration evidence is absent.

### Verification

- `python3 -m ruff check src/evidence/benchmark_gating.py scripts/run_stage1_pipeline.py tests/test_benchmark_gating.py tests/test_stage1_pipeline_governed.py`
- `python3 -m compileall src/evidence scripts/run_stage1_pipeline.py -q`
- `python3 -m pytest -q tests/test_benchmark_gating.py tests/test_stage1_pipeline_governed.py` -> `10 passed, 4 warnings`
- `python3 scripts/economic_world_model/nightly_audit.py --output-json artifacts/economic_world_model/nightly_audit_summary.json --output-markdown artifacts/economic_world_model/nightly_audit_summary.md` -> `status: ok`

## 2026-05-20 - Reconstruction grounding eligibility reports

### What changed

- Added `ReconstructionGroundingReport` to the 4D reconstruction module.
- Stage-1 writes `reconstruction_grounding_report_v1` sidecars for every governed-video episode.
- The report records calibration class, grounding class, training eligibility, benchmark readiness, missing refs, quality scores, and source artifact refs.
- Stage-1 future-training signals now expose reconstruction calibration and training eligibility directly.
- Replay discovery and bridge coverage preserve the reconstruction grounding report ref.

### Boundary

This is a truth/eligibility sidecar. It does not run non-stub SceneTracks, does not calibrate cameras, does not train a predictor, and does not promote reconstruction or video-state modeling. It only prevents downstream consumers from treating reconstruction sidecar presence as calibrated real grounding.

### Verification

- `python3 -m ruff format src/vision/reconstruction/four_d_reconstruction.py src/vision/reconstruction/__init__.py scripts/run_stage1_pipeline.py src/replay/ingest.py tests/test_four_d_reconstruction.py tests/test_stage1_pipeline_governed.py tests/test_replay_dataset.py`
- `python3 -m ruff check src/vision/reconstruction/four_d_reconstruction.py src/vision/reconstruction/__init__.py scripts/run_stage1_pipeline.py src/replay/ingest.py tests/test_four_d_reconstruction.py tests/test_stage1_pipeline_governed.py tests/test_replay_dataset.py`
- `python3 -m compileall src/vision/reconstruction scripts/run_stage1_pipeline.py src/replay/ingest.py -q`
- `python3 -m pytest -q tests/test_four_d_reconstruction.py tests/test_stage1_pipeline_governed.py tests/test_replay_dataset.py` -> `14 passed, 5 warnings`

## 2026-05-20 - Pre-Economic-WM Stage-1 replay/export integration

### What changed

- Added `docs/economic_world_model/pre_economic_wm_integration_stage.md`.
- Added `scripts/export_governed_video_stage1_bridges.py` for governed-video admission-log export into canonical replay, RLDS-shaped JSONL, and LeRobot-shaped JSONL.
- Stage-1 now writes teacher contract, teacher action-envelope, and teacher-trace sidecars for every governed-video episode, including explicit unavailable fallback semantics.
- Stage-1 reconstruction sidecars now avoid synthetic calibration refs when no calibration exists in the manifest.
- Replay discovery/import and replay preconditions now preserve and recognize reconstruction, branch-evaluation, teacher-contract, teacher-action, and teacher-trace refs.
- Dataset bridge sidecar extraction now preserves `_path` / `_paths` fields as internal sidecar pointers in addition to refs and ids.

### Boundary

This is integration scaffolding, not Economic WM training. It does not run GPU/provider training, does not promote a video predictor, does not claim non-stub OpenVLA/SAM3D execution, and does not alter frozen Phase B reward, trust-net, `w_econ`, or lambda-controller math.

### Verification

- `python3 -m ruff format scripts/run_stage1_pipeline.py scripts/export_governed_video_stage1_bridges.py src/replay/importers.py src/replay/preconditions.py src/replay/ingest.py src/dataset_bridges/sidecar_refs.py tests/test_stage1_pipeline_governed.py tests/test_replay_dataset.py`
- `python3 -m ruff check scripts/run_stage1_pipeline.py scripts/export_governed_video_stage1_bridges.py src/replay/importers.py src/replay/preconditions.py src/replay/ingest.py src/dataset_bridges/sidecar_refs.py tests/test_stage1_pipeline_governed.py tests/test_replay_dataset.py`
- `python3 -m compileall scripts/run_stage1_pipeline.py scripts/export_governed_video_stage1_bridges.py src/replay/importers.py src/replay/preconditions.py src/replay/ingest.py src/dataset_bridges/sidecar_refs.py -q`
- `python3 -m pytest -q tests/test_stage1_pipeline_governed.py tests/test_replay_dataset.py tests/test_dataset_bridges.py tests/test_teacher_runtime.py tests/test_four_d_reconstruction.py` -> `19 passed, 5 warnings`
- `python3 scripts/economic_world_model/nightly_audit.py --output-json artifacts/economic_world_model/nightly_audit_summary.json --output-markdown artifacts/economic_world_model/nightly_audit_summary.md` -> `status: ok`
- `python3 -m pytest tests/ -q` -> `1663 passed, 2 skipped, 29 warnings`

## 2026-05-20 - Phase 3 local-loop sidecars and neural architecture scaffolds

### What changed

- Added `sidecars.py` under `src/world_model/embodiment_actuation/`.
- The local embodiment runner now writes canonical Phase 3 state/receipt/consumer JSON sidecars, Phase 3.4 JSONL rows, non-promotional training manifests, morphology receipts, and a neural architecture manifest per episode.
- Added `neural_architectures.py` with four CPU-forward architecture skeletons:
  - temporal JEPA/action-conditioned latent predictor
  - ACT-style chunked transformer head
  - Diffusion Policy-style action denoiser
  - topology-contrastive morphology consistency head
- Threaded Phase 3 sidecar refs through `EmbodimentProfileSummary`, datapack validation, and representation-token payloads.
- Added `train_embodiment_phase34_neural_architectures.py` to `scripts/TRAINING_MIGRATION_BACKLOG.json` as the future GPU/provider-gated trainer for these scaffolds.
- Extended `scripts/smoke_test_embodiment_phase34.py` to write and verify the neural architecture manifest locally.
- Added `docs/economic_world_model/phase3_closure_assessment.md` to separate local structural closure from external GPU/provider/hardware evidence gates.

### Boundary

This is neural scaffolding, not training. The pass intentionally makes future GPU work concrete by fixing shapes, objectives, sidecar manifests, and promotion blockers. It does not run GPU training, does not import GR00T/V-JEPA/Diffusion Policy/ACT code, does not validate Unitree/Isaac/Holosoma runtime execution, and does not grant runtime authority.

### Verification

- `git diff --check && python3 -m ruff check src/world_model/embodiment_actuation src/embodiment/runner.py src/embodiment/datapack_adapter.py src/valuation/datapack_schema.py src/valuation/datapack_validators.py src/representation/token_providers.py tests/test_embodiment_actuation_phase34.py tests/embodiment/test_embodiment_module.py && python3 -m compileall src/world_model/embodiment_actuation src/embodiment src/valuation src/representation tests/test_embodiment_actuation_phase34.py tests/embodiment/test_embodiment_module.py -q && python3 -m pytest tests/test_embodiment_actuation_phase34.py tests/embodiment/test_embodiment_module.py tests/test_embodiment_actuation_world_model.py -q` -> `20 passed`
- `python3 scripts/smoke_test_embodiment_phase34.py --out-dir artifacts/embodiment_phase34 --variant g1_29dof` -> `status: ok`, neural architecture manifest `promotion_eligible=false`
- `python3 -m pytest tests/ -q` -> `1662 passed, 2 skipped, 28 warnings`

## 2026-05-20 - Phase 3 morphology and 3.4 learned-seam scaffolding

### What changed

- Added G1 morphology/evidence contracts in `morphology.py`.
- Added Phase 3.4 neural seams in `neural_seams.py`.
- Added training-row and manifest builders in `training_corpus.py`.
- Added local smoke script `scripts/smoke_test_embodiment_phase34.py`.
- Added external pattern absorption note at `docs/economic_world_model/phase3_external_pattern_absorption.md`.

### External pattern absorption

The pass translates Unitree/Isaac/GR00T-style public patterns into native WM
contracts: morphology/action-space profiles, sim2sim/sim2real evidence posture,
co-training/training-row discipline, and explicit gap receipts. It does not
import those projects as ontology or claim provider execution.

### Local smoke result

`python3 scripts/smoke_test_embodiment_phase34.py --out-dir artifacts/embodiment_phase34 --scan-root /Users/amarmurray/code/unitree_rl_gym --scan-root /Users/amarmurray/code/unitree_sim_isaaclab --scan-root /Users/amarmurray/code/unitree_models --variant g1_29dof` produced `status: ok`, `variant: g1_29dof`, `joint_count: 29`, `obs_dim: 47`, `priv_obs_dim: 50`, four Phase 3.4 rows, finite seam outputs, and `promotion_eligible=false`.

### Verification

- `git diff --check && python3 -m ruff check src/world_model/embodiment_actuation scripts/smoke_test_embodiment_phase34.py tests/test_embodiment_actuation_phase34.py tests/test_embodiment_actuation_world_model.py && python3 -m compileall src/world_model/embodiment_actuation scripts/smoke_test_embodiment_phase34.py tests/test_embodiment_actuation_phase34.py tests/test_embodiment_actuation_world_model.py -q && python3 -m pytest tests/test_embodiment_actuation_world_model.py tests/test_embodiment_actuation_phase34.py -q` -> `13 passed`
- `python3 scripts/smoke_test_embodiment_phase34.py --out-dir artifacts/embodiment_phase34 --scan-root /Users/amarmurray/code/unitree_rl_gym --scan-root /Users/amarmurray/code/unitree_sim_isaaclab --scan-root /Users/amarmurray/code/unitree_models --variant g1_29dof` -> `status: ok`
- `python3 -m pytest tests/embodiment/test_embodiment_module.py tests/test_embodiment_shadow_consumer.py tests/test_sim_synth_phase1x_subsystems.py -q` -> `29 passed`
- `python3 scripts/economic_world_model/nightly_audit.py --output-json artifacts/economic_world_model/nightly_audit_summary.json --output-markdown artifacts/economic_world_model/nightly_audit_summary.md` -> `status: ok`
- `python3 -m pytest tests/ -q` -> `1660 passed, 2 skipped, 24 warnings`


## 2026-05-20 - Phase 3.1-3.3 Embodiment / Actuation implementation

### What changed

- Added the additive `src/world_model/embodiment_actuation/` package.
- Added canonical state contracts in `state.py` for capability, embodiment profile, actuator configuration, joint state, contact state, safety envelope, action space, observation interface, contact/affordance graph, local dynamics forecast, inverse retarget trace, action proposal bundle, drift, cost, and calibration targets.
- Added receipt contracts in `receipts.py` for every Phase 3.1 minimum receipt plus Sim↔Embodiment transfer.
- Added provider/runtime contracts in `provider_contracts.py` for Unitree G1, Holosoma, Isaac, and generic Embodiment providers without requiring those providers to be executable.
- Added promotion posture in `promotion.py`; learned seams remain blocked unless posture, provider availability, and benchmark signals justify execution.
- Added a shadow compiler in `compiler.py` and first downstream consumers in `consumers.py`.
- Extended `src/world_model/sim_synth_physics/adapters/embodiment_inputs.py` to preserve Phase 3 transfer scores and authority posture.

### Phase coverage

- **3.1**: canonical state and receipts are live.
- **3.2**: shadow compiler consumes existing advisory embodiment artifacts, registry/adapters, Perception shadow surfaces, Sim/provider refs, and optional joint state.
- **3.3**: first shadow downstream consumers are live for Sim/Synth, Perception, Runtime validation, and Economic receipt ingest.

### Guardrails preserved

- No native GR00T import.
- No hardware deployment.
- No provider/GPU claims.
- No policy promotion.
- No changes to frozen Phase B baseline math, trust-net, `w_econ`, or lambda controller paths.
- All runtime-facing outputs remain `authority_level=none` unless a later explicit promotion tranche changes that.

### Verification

- `git diff --check && python3 -m ruff check src/world_model/embodiment_actuation src/world_model/sim_synth_physics/adapters/embodiment_inputs.py tests/test_embodiment_actuation_world_model.py && python3 -m compileall src/ tests/test_embodiment_actuation_world_model.py -q && python3 -m pytest tests/test_embodiment_actuation_world_model.py tests/embodiment/test_embodiment_module.py tests/test_embodiment_shadow_consumer.py tests/test_sim_synth_phase1x_subsystems.py -q` -> `36 passed`
- `python3 scripts/economic_world_model/nightly_audit.py --output-json artifacts/economic_world_model/nightly_audit_summary.json --output-markdown artifacts/economic_world_model/nightly_audit_summary.md` -> `status: ok`
- `python3 -m pytest tests/ -q` -> `1654 passed, 2 skipped, 24 warnings`


## 2026-05-19 - Phase 1.x closure sheet and Phase 3 prep artifacts

### What changed

- Added `docs/economic_world_model/phase1x_closure_assessment.md`.
- Added `docs/economic_world_model/groot_inspired_functionality_status.md`.
- Added `docs/economic_world_model/phase3_embodiment_actuation_spec_prep.md`.
- Updated roadmap and actuation doctrine links so the new artifacts are not isolated notes.

### Assessment

Phase 1.x is documented as locally structural closure-ready: Category A = `0`,
unresolved Category C = `0`, and remaining blockers are external evidence gates.
The meaningful remaining debt is provider/GPU/hardware/benchmark debt, not a
known local code-structure gap. In particular, Isaac/Unitree latency and
safety-watchdog profiles should not be filled with placeholder YAML; they need
truthful runtime or hardware evidence.

### GR00T-inspired boundary

The end state borrows GR00T/Isaac discipline for teacher/student lanes,
deploy-shaped observations, domain randomization, reset curricula,
action-space hygiene, export gates, and sim-to-real transfer receipts. It does
not adopt GR00T as ontology. Current wiring is limited to typed Perception
evidence, Phase 1.x runtime truth/training admissibility, local Holosoma ONNX
action smoke, local Isaac/Unitree scan truth, and early Embodiment/Runtime
adapter substrates.

### Phase 3 prep

The first Phase 3 implementation should be additive canonical-state work under
`src/world_model/embodiment_actuation/`: state contracts, receipts, a shadow
compiler from existing advisory artifacts, and tests proving no runtime
authority by default. Provider bring-up and hardware claims remain later.

### Verification

- `git diff --check && python3 -m compileall src/`
- `python3 -m pytest tests/test_scan_phase1_runtime_layouts.py tests/test_setup_holosoma_local_smoke.py tests/test_local_holosoma_smoke.py tests/test_sim_synth_phase1x_subsystems.py -q` -> `12 passed`
- `python3 scripts/economic_world_model/nightly_audit.py --output-json artifacts/economic_world_model/nightly_audit_summary.json --output-markdown artifacts/economic_world_model/nightly_audit_summary.md` -> `status: ok`
- `python3 -m pytest tests/ -q` -> `1647 passed, 2 skipped, 24 warnings`


## 2026-05-19 - Reproducible local Holosoma smoke bootstrap

### What changed

- Added `scripts/setup_holosoma_local_smoke.py`.
- The script writes the user-site `.pth` shim for the existing local Holosoma
  checkout and intentionally does not run pip.
- It supports `--dry-run`, `--remove`, `--holosoma-root`, `--site-packages-dir`,
  and `--pth-name`.
- It prints machine-readable JSON with the shim path, path entries, missing
  paths, the minimal smoke dependency install hint, and the post-install smoke
  command.
- Added `tests/test_setup_holosoma_local_smoke.py`.

### Current local read

The local shim is now reproducible with:

```bash
python3 scripts/setup_holosoma_local_smoke.py
```

That recreates
`/Users/amarmurray/Library/Python/3.9/lib/python/site-packages/robotics_vp_holosoma_local.pth`
with these entries:

- `/Users/amarmurray/code/holosoma/src/holosoma`
- `/Users/amarmurray/code/holosoma/src/holosoma_inference`
- `/Users/amarmurray/code/holosoma/src/holosoma_retargeting`

This remains a local deploy-smoke bootstrap only. Full Holosoma runtime execution
still requires explicit runtime enablement and real provider/runtime evidence.

### Verification

- `python3 -m ruff check scripts/setup_holosoma_local_smoke.py tests/test_setup_holosoma_local_smoke.py`
- `python3 -m compileall scripts/setup_holosoma_local_smoke.py tests/test_setup_holosoma_local_smoke.py`
- `python3 -m pytest tests/test_setup_holosoma_local_smoke.py -q` -> `3 passed`
- `python3 scripts/setup_holosoma_local_smoke.py` -> `status: installed`
- `python3 scripts/local_holosoma_smoke.py --preflight-only --out-dir artifacts/holosoma_local_probe` -> `ready: true`, `policy_kind: onnx_deploy`
- `python3 scripts/local_holosoma_smoke.py --episodes 1 --out-dir artifacts/holosoma_local_probe` -> finite ONNX action output
- `python3 -m ruff check scripts/setup_holosoma_local_smoke.py tests/test_setup_holosoma_local_smoke.py && git diff --check && python3 -m compileall src/ scripts/setup_holosoma_local_smoke.py tests/test_setup_holosoma_local_smoke.py && python3 -m pytest tests/ -q` -> `1647 passed, 2 skipped, 24 warnings`
- `python3 scripts/economic_world_model/nightly_audit.py --output-json artifacts/economic_world_model/nightly_audit_summary.json --output-markdown artifacts/economic_world_model/nightly_audit_summary.md` -> `status: ok`, no drift

## 2026-05-19 - Local Holosoma ONNX deploy smoke

### What changed

- Used a user-site `.pth` shim pointing at the existing local Holosoma checkout
  instead of running `pip install -r requirements-holosoma.txt`. This avoids the
  full heavy dependency set while making `holosoma`, `holosoma_inference`, and
  `holosoma_retargeting` importable.
- Installed only the narrow smoke-path dependencies with `--no-cache-dir`:
  `tyro`, `loguru`, `omegaconf`, `tqdm`, `tensordict`, `tensorboard`,
  `trimesh`, and `onnxruntime`; captured the same set in `requirements-holosoma-smoke.txt`.
- Split local smoke behavior by policy kind:
  - `.onnx` policies use ONNX deploy/action smoke
  - non-ONNX policies continue to use the native Holosoma eval path
- Updated Holosoma backend entrypoint imports to support the local upstream
  checkout layout.
- Added an explicit `ROBOTICS_VP_ENABLE_HOLOSOMA_RUNTIME=1` gate for WM runtime
  execution claims, keeping importable local packages separate from concrete
  simulator/runtime availability.

### Current local read

The selected local policy is
`/Users/amarmurray/code/holosoma/src/holosoma_inference/holosoma_inference/models/loco/g1_29dof/fastsac_g1_29dof.onnx`.
A native Holosoma eval loop is the wrong path for that artifact because the eval
entrypoint expects a serialized training checkpoint with embedded
`experiment_config`. The local non-GPU proof is now an ONNX deploy/action smoke:
`actor_obs [1, 100] -> action [1, 29]` with finite `float32` output.

This does not promote Holosoma runtime evidence. It proves local policy loading
and deploy-path inference only; simulated episode evidence, motion quality, and
provider runtime receipts remain future provider/GPU work. WM runtime routing
therefore remains shadow/fallback by default unless
`ROBOTICS_VP_ENABLE_HOLOSOMA_RUNTIME=1` is set.

### Verification

- `python3 -m ruff check scripts/local_holosoma_smoke.py tests/test_local_holosoma_smoke.py src/motor_backend/holosoma_backend.py`
- `python3 -m compileall scripts/local_holosoma_smoke.py tests/test_local_holosoma_smoke.py src/motor_backend/holosoma_backend.py`
- `python3 -m pytest tests/test_local_holosoma_smoke.py tests/test_holosoma_backend_interface.py -q` -> `4 passed`
- `python3 scripts/local_holosoma_smoke.py --preflight-only --out-dir artifacts/holosoma_local_probe` -> `ready: true`, `policy_kind: onnx_deploy`
- `python3 scripts/local_holosoma_smoke.py --episodes 1 --out-dir artifacts/holosoma_local_probe` -> wrote `holosoma_onnx_deploy_smoke.json` with finite action output
- `python3 -m ruff check scripts/local_holosoma_smoke.py tests/test_local_holosoma_smoke.py tests/test_sim_synth_physics_world_model.py src/motor_backend/holosoma_backend.py src/world_model/sim_synth_physics/holosoma_runtime_gate.py src/world_model/sim_synth_physics/backend_adapters.py src/world_model/sim_synth_physics/adapters/backend_holosoma.py src/world_model/sim_synth_physics/runtime_targets.py src/world_model/sim_synth_physics/shadow_execution.py src/world_model/sim_synth_physics/backend_runtime_execution.py src/world_model/sim_synth_physics/adapters/holosoma_adapter_execution.py && git diff --check && python3 -m compileall src/ scripts/local_holosoma_smoke.py tests/test_local_holosoma_smoke.py tests/test_sim_synth_physics_world_model.py && python3 -m pytest tests/ -q` -> `1644 passed, 2 skipped, 24 warnings`

## 2026-05-19 - Holosoma local preflight separates provider install from GPU debt

### What changed

- `scripts/local_holosoma_smoke.py` no longer requires `--policy-id` when a
  local Holosoma policy contract can provide a selected checkpoint.
- Added `--preflight-only`.
- The script now writes `holosoma_smoke_preflight.json` with:
  - Holosoma Python-module availability
  - selected policy ref and source
  - policy checkpoint existence
  - readiness
  - missing preconditions
- Added `tests/test_local_holosoma_smoke.py`.

### Current local read

`python3 scripts/scan_phase1_runtime_layouts.py --output-path artifacts/sim_synth_runtime_layout_scan.json`
finds a local Holosoma policy checkpoint and ready local Holosoma surfaces on
this host. A direct smoke cannot yet run because the optional `holosoma` Python
module is not importable. The new preflight records that as
`missing_preconditions: ["holosoma_python_module"]` while confirming the policy
checkpoint exists.

This means the remaining Phase 1 work is not perfectly summarized as
“GPU-only.” More precisely:

- Isaac/Unitree concrete execution and GGDS/LDM materialization remain
  GPU/runtime/asset blocked.
- Holosoma has a local non-GPU provider-install blocker before concrete smoke
  execution can run.

### Verification

- `python3 -m ruff check scripts/local_holosoma_smoke.py tests/test_local_holosoma_smoke.py`
- `git diff --check && python3 -m compileall scripts/local_holosoma_smoke.py tests/test_local_holosoma_smoke.py -q && python3 -m pytest tests/test_local_holosoma_smoke.py -q` ->
  `2 passed`
- `python3 scripts/local_holosoma_smoke.py --preflight-only --out-dir artifacts/holosoma_local_probe` ->
  `ready: false`, missing `holosoma_python_module`, policy checkpoint exists
- `python3 -m ruff check scripts/local_holosoma_smoke.py tests/test_local_holosoma_smoke.py && git diff --check && python3 -m compileall src/ scripts/local_holosoma_smoke.py tests/test_local_holosoma_smoke.py && python3 -m pytest tests/ -q` -> `1642 passed, 3 skipped, 24 warnings`

## 2026-05-19 - Phase 1.x subsystem index trainer-row propagation

### What changed

- `build_backend_selector_rows_from_receipts(...)` and
  `build_branch_planner_rows_from_receipts(...)` now carry the compiled
  `phase1x_subsystem_index_v1` into row metadata.
- Trainer-facing rows preserve:
  - subsystem index ID and schema version
  - structural status
  - subsystem count and subsystem IDs
  - coverage summary
  - provider ownership rule
  - honest blocker class
- Added regression coverage that compiles a world state, projects receipt
  bundles into both trainer row families, and verifies the subsystem index
  survives the projection.

### Why this was the right next local step

The previous pass made the subsystem index a compiled world-state artifact.
This pass makes it survive into the trainer-facing corpus boundary, where
promotion and benchmark preparation will actually inspect rows. Without this,
the index could still decay into state-only documentation once rows are
exported.

### Verification

- `python3 -m ruff check src/world_model/sim_synth_physics/training_corpus.py tests/test_sim_synth_phase1x_subsystems.py`
- `git diff --check && python3 -m compileall src/world_model/sim_synth_physics tests/test_sim_synth_phase1x_subsystems.py -q && python3 -m pytest tests/test_sim_synth_phase1x_subsystems.py tests/test_sim_synth_training_corpus.py -q` ->
  `7 passed`
- `git diff --check && python3 -m compileall src/ && python3 -m pytest tests/ -q` ->
  `1640 passed, 3 skipped, 24 warnings`

## 2026-05-19 - Phase 1.x subsystem index in compiled WM state

### What changed

- Added `src/world_model/sim_synth_physics/subsystems.py`.
- The new `phase1x_subsystem_index_v1` maps all 10 Sim / Synth / Physics
  Phase 1.x subsystems to:
  - owned modules
  - typed state surfaces
  - receipt surfaces
  - learned or reserved seams
  - promotion gates
  - provider families
  - external blockers
  - runtime artifact-ref keys
- `compile_sim_synth_physics_world_state(...)` now embeds the subsystem index
  in world-state metadata using the compiled artifact refs and receipt
  inventory.
- The package exports `PHASE1X_SUBSYSTEM_SPECS`,
  `Phase1xSubsystemSpec`, and `build_phase1x_subsystem_index(...)`.

### Why this was the right next local step

The Phase 1.x decomposition had been specified in doctrine but was not yet a
machine-readable runtime surface. That made it easy for future audits to drift
back into prose interpretation. This pass turns subsystem ownership into a
compiled WM artifact while staying honest about provider/GPU blockers.

This does not claim the package has been physically refactored into 10
directories or that all provider lanes are executable. It gives future
provider bring-up, training, and closure audits a stable index for deciding
which subsystem owns each surface and which blockers are external.

### Verification

- `python3 -m ruff check src/world_model/sim_synth_physics/subsystems.py src/world_model/sim_synth_physics/compiler.py src/world_model/sim_synth_physics/__init__.py tests/test_sim_synth_phase1x_subsystems.py`
- `git diff --check && python3 -m compileall src/world_model/sim_synth_physics tests/test_sim_synth_phase1x_subsystems.py -q && python3 -m pytest tests/test_sim_synth_phase1x_subsystems.py tests/test_sim_synth_phase1x_surfaces.py tests/test_sim_synth_physics_world_model.py -q` ->
  `36 passed`
- `git diff --check && python3 -m compileall src/ && python3 -m pytest tests/ -q` ->
  `1639 passed, 3 skipped, 24 warnings`

## 2026-05-19 - Phase 1.x training-gate promotion preconditions

### What changed

- Added `build_phase1x_training_gate(...)` with
  `phase1x_training_gate_v1` output.
- The gate is now emitted through both Sim / Synth / Physics helper trainers:
  - dataset summary
  - training summary
  - runtime package
  - training job result
  - Regal result/runtime metadata
  - execution preconditions
- The runtime package `promotion_stage` now requires:
  - existing benchmark-density gate readiness
  - Phase 1.x training-gate readiness
- The gate blocks promotion if selected rows do not match the admissibility
  summary, diagnostic rows are present, runtime manifest validation is not
  clean, or negative-supervision rows exist without reject-head training.

### Why this was the right next local step

The previous tranche trained bounded reject heads from negative-supervision
sidecars. This tranche closes the surrounding promotion gap: a helper package
can no longer look promotion-ready from benchmark density alone if its training
corpus violates the Phase 1.x admissibility/reject-head contract.

This is still local structural readiness. It does not claim that the helper
models are good, that provider truth exists, or that GPU-backed benchmarks have
been run. It makes the later provider/GPU season stricter by giving those runs
a concrete precondition surface to satisfy.

### Verification

- `python3 -m ruff check src/world_model/sim_synth_physics/training_corpus.py src/world_model/sim_synth_physics/__init__.py scripts/train_sim_synth_backend_selector.py scripts/train_sim_synth_branch_planner.py tests/test_train_sim_synth_backend_selector.py tests/test_train_sim_synth_branch_planner.py tests/test_sim_synth_training_corpus.py`
- `git diff --check && python3 -m compileall src/world_model/sim_synth_physics scripts/train_sim_synth_backend_selector.py scripts/train_sim_synth_branch_planner.py tests/test_train_sim_synth_backend_selector.py tests/test_train_sim_synth_branch_planner.py tests/test_sim_synth_training_corpus.py -q && python3 -m pytest tests/test_train_sim_synth_backend_selector.py tests/test_train_sim_synth_branch_planner.py tests/test_sim_synth_physics_world_model.py tests/test_sim_synth_training_corpus.py -q` ->
  `43 passed`
- `git diff --check && python3 -m compileall src/ && python3 -m pytest tests/ -q` ->
  `1637 passed, 3 skipped, 24 warnings`

## 2026-05-19 - Phase 1.x reject-head training over negative supervision

### What changed

- Added `reject_head` / `reject_probability` output to:
  - `LearnedBackendSelector`
  - `LearnedBranchPlanner`
- `train_backend_selector(...)` and `train_branch_planner(...)` now accept
  `negative_rows` and train a BCE reject loss against positive vs negative
  supervision rows.
- Existing backend/fidelity/randomization and branch mode/yield losses still use
  selected positive rows only.
- Trainer scripts pass negative-supervision sidecars into those reject losses and
  report `reject_accuracy`.
- Runtime package metadata now advertises
  `phase1x_reject_probability_head_v1`.
- Promoted helper payloads with `reject_recommended` or high
  `reject_probability` stay trace-visible but are not applied to runtime backend
  or branch decisions.
- Legacy checkpoints without a reject head load safely with a low-reject default.

### Why this was the right next local step

This is the first pass where negative-supervision rows become actual learning
signal. It deliberately does not let negative rows train the positive target
heads. The model learns a bounded reject surface while the existing helper
semantics stay intact. Runtime also respects that reject surface by refusing to
apply a promoted learned payload that recommends rejection.

This reduces the Phase 1.x local debt from “negative evidence is preserved but
unused” to “negative evidence trains a bounded reject head, pending provider
truth and benchmark validation.”

### Verification

- `python3 -m ruff check src/world_model/sim_synth_physics/backend_selector.py src/world_model/sim_synth_physics/branch_planner.py src/world_model/sim_synth_physics/compiler.py src/world_model/sim_synth_physics/synthetic_branches.py src/world_model/sim_synth_physics/promotion.py scripts/train_sim_synth_backend_selector.py scripts/train_sim_synth_branch_planner.py tests/test_train_sim_synth_backend_selector.py tests/test_train_sim_synth_branch_planner.py tests/test_sim_synth_physics_world_model.py tests/test_sim_synth_training_corpus.py`
- `git diff --check && python3 -m compileall src/world_model/sim_synth_physics scripts/train_sim_synth_backend_selector.py scripts/train_sim_synth_branch_planner.py tests/test_train_sim_synth_backend_selector.py tests/test_train_sim_synth_branch_planner.py tests/test_sim_synth_physics_world_model.py tests/test_sim_synth_training_corpus.py -q && python3 -m pytest tests/test_train_sim_synth_backend_selector.py tests/test_train_sim_synth_branch_planner.py tests/test_sim_synth_physics_world_model.py tests/test_sim_synth_training_corpus.py -q` ->
  `43 passed`
- `git diff --check && python3 -m compileall src/ && python3 -m pytest tests/ -q` ->
  `1637 passed, 3 skipped, 24 warnings`

## 2026-05-19 - Phase 1.x excluded-row sidecars for trainer inputs

### What changed

- Added `split_phase1x_training_rows(...)` with
  `phase1x_training_row_split_v1` output.
- The split preserves four groups:
  - selected positive / legacy rows for current helper training
  - `negative_supervision` rows
  - `diagnostic_only` rows
  - other excluded rows, if a future status appears
- Backend-selector and branch-planner training scripts now write:
  - `*_rows.jsonl` for selected positive training rows
  - `*_negative_supervision_rows.jsonl`
  - `*_diagnostic_rows.jsonl`
- The negative and diagnostic sidecars are included in training summaries, job
  results, and Regal artifact manifests.

### Why this was the right next local step

The previous pass correctly excluded negative and diagnostic rows from positive
helper losses. This pass prevents a different failure mode: losing the excluded
rows as actionable artifacts. Negative supervision remains deferred as a model
loss, but it is now preserved as a first-class local dataset sidecar.

### Verification

- `python3 -m ruff check src/world_model/sim_synth_physics/training_corpus.py src/world_model/sim_synth_physics/__init__.py scripts/train_sim_synth_backend_selector.py scripts/train_sim_synth_branch_planner.py tests/test_sim_synth_training_corpus.py tests/test_train_sim_synth_backend_selector.py tests/test_train_sim_synth_branch_planner.py`
- `git diff --check && python3 -m compileall src/world_model/sim_synth_physics scripts/train_sim_synth_backend_selector.py scripts/train_sim_synth_branch_planner.py tests/test_sim_synth_training_corpus.py tests/test_train_sim_synth_backend_selector.py tests/test_train_sim_synth_branch_planner.py -q && python3 -m pytest tests/test_sim_synth_training_corpus.py tests/test_train_sim_synth_backend_selector.py tests/test_train_sim_synth_branch_planner.py -q` ->
  `10 passed`
- `git diff --check && python3 -m compileall src/ && python3 -m pytest tests/ -q` ->
  `1634 passed, 3 skipped, 24 warnings`

## 2026-05-19 - Phase 1.x trainer-side admissibility enforcement

### What changed

- Added `select_phase1x_positive_training_rows(...)` and
  `phase1x_positive_training_row_selection_v1` summaries.
- Exported the selector through the Sim / Synth / Physics package surface.
- `scripts/train_sim_synth_backend_selector.py` and
  `scripts/train_sim_synth_branch_planner.py` now split source rows before
  training:
  - `positive_training` rows are selected
  - `legacy_dataset_row` rows remain selected for explicit historical datasets
  - `negative_supervision` rows are counted and excluded until the helper losses
    support negative examples directly
  - `diagnostic_only` rows are counted and excluded
- Dataset summaries, training summaries, job results, and Regal receipt-label
  coverage now record source row counts, selected row counts, excluded row
  counts, status counts, reason counts, and bounded excluded row refs.

### Why this was the right next local step

The prior tranche made each harvested row's admissibility state visible. Leaving
the trainer entrypoints unchanged would still allow invalid or negative rows to
be flattened into positive supervised labels. The trainer losses are currently
positive-label helper losses, so this pass keeps negative supervision available
as counted evidence without pretending the losses know how to use it yet.

The remaining debt is explicit and smaller: add negative-example losses or a
separate reject/utility head later, then promote selected negative supervision
from counted evidence into trainable signal.

### Verification

- `python3 -m ruff check src/world_model/sim_synth_physics/training_corpus.py src/world_model/sim_synth_physics/__init__.py scripts/train_sim_synth_backend_selector.py scripts/train_sim_synth_branch_planner.py tests/test_sim_synth_training_corpus.py tests/test_train_sim_synth_backend_selector.py tests/test_train_sim_synth_branch_planner.py`
- `git diff --check && python3 -m compileall src/world_model/sim_synth_physics scripts/train_sim_synth_backend_selector.py scripts/train_sim_synth_branch_planner.py tests/test_sim_synth_training_corpus.py tests/test_train_sim_synth_backend_selector.py tests/test_train_sim_synth_branch_planner.py -q && python3 -m pytest tests/test_sim_synth_training_corpus.py tests/test_train_sim_synth_backend_selector.py tests/test_train_sim_synth_branch_planner.py -q` ->
  `10 passed`
- `git diff --check && python3 -m compileall src/ && python3 -m pytest tests/ -q` ->
  `1634 passed, 3 skipped, 24 warnings`

## 2026-05-19 - Phase 1.x training-admissibility gating

### What changed

- Added a shared local classifier that emits
  `phase1x_training_admissibility_v1` records for training-row builders.
- Backend-selector rows now carry top-level and metadata-level admissibility
  posture.
- Branch-planner rows now carry the same posture using per-branch replay and
  branch-validity receipts.
- The classifier separates:
  - `positive_training`
  - `negative_supervision`
  - `diagnostic_only`
- Reasons include manifest validation failures, planning-only target source,
  missing outcomes, missing validity receipts, branch rejection, and replay
  rejection.

### Why this was the right next local step

Receipt emission and manifest validation made the evidence visible. This pass
makes it actionable for training-row consumers. Rows that are structurally
healthy and unfiltered can be used as positive targets; rows with explicit
replay/branch rejects can become negative supervision; rows with missing or
manifest-inconsistent evidence stay diagnostic-only.

This avoids a common failure mode in receipt-heavy systems: collecting honest
evidence and then accidentally flattening it back into undifferentiated
training rows.

### Verification

- `python3 -m ruff check src/world_model/sim_synth_physics/training_corpus.py tests/test_sim_synth_training_corpus.py`
- `python3 -m pytest tests/test_sim_synth_training_corpus.py -q` ->
  `4 passed`
- `python3 -m pytest tests/test_sim_synth_training_corpus.py tests/test_sim_synth_physics_world_model.py tests/test_sim_synth_phase1x_surfaces.py tests/test_sim_synth_camera_geometry.py tests/test_sim_synth_vectorized_runtime.py -q` ->
  `40 passed`
- `git diff --check && python3 -m compileall src/ && python3 -m pytest tests/ -q` ->
  `1634 passed, 3 skipped, 24 warnings`

## 2026-05-19 - Phase 1.x runtime receipt manifest validation

### What changed

- Added `validate_runtime_receipt_manifest(...)` to `training_corpus.py`.
- The validator compares manifest `receipt_family_counts` against the actual
  harvested bundle representation and reports:
  - `validation_status`
  - mismatched families
  - missing required families
  - actual receipt-family counts
- Live-directory harvest now expands runtime-emitted receipt bundle wrappers
  into their constituent receipts before grouping.
- Backend-selector and branch-planner rows now carry manifest validation status
  and mismatched-family diagnostics.

### Why this was the right next local step

The runtime manifest made receipt emission auditable. This pass makes the audit
checkable after harvest. Without it, a downstream row could carry a manifest id
while silently missing bundled receipts because the harvester only saw wrapper
files.

The validator is intentionally narrow: it verifies receipt accounting and
required-family completeness. It does not certify provider execution, benchmark
readiness, calibration quality, or promotion eligibility.

### Verification

- `python3 -m ruff check src/world_model/sim_synth_physics tests/test_sim_synth_training_corpus.py tests/test_sim_synth_physics_world_model.py`
- `python3 -m pytest tests/test_sim_synth_training_corpus.py tests/test_sim_synth_physics_world_model.py tests/test_sim_synth_phase1x_surfaces.py tests/test_sim_synth_camera_geometry.py tests/test_sim_synth_vectorized_runtime.py -q` ->
  `40 passed`
- `python3 -m compileall src/ && python3 -m pytest tests/ -q` ->
  `1634 passed, 3 skipped, 24 warnings`

## 2026-05-19 - Phase 1.x runtime receipt manifest consolidation

### What changed

- `SimSynthPhysicsRuntime` now builds and emits
  `sim_synth_runtime_receipt_manifest_v1` as `runtime_receipt_manifest.json`.
- The manifest captures:
  - emitted receipt families and ids
  - artifact keys and output paths
  - required vs optional receipt posture
  - missing required families
  - optional provider/runtime families that were not emitted
  - receipt-family counts and training-feedback row count
- `SimSynthPhysicsLoopResult` and `to_dict()` now include the manifest.
- `sim_synth_training_feedback_v1` now links back to the manifest id and
  manifest status.
- `training_corpus.py` now harvests manifest artifacts from live dirs / mixed
  files and projects manifest metadata into backend-selector and branch-planner
  rows.

### Why this was the right next local step

The last several Phase 1.x passes made useful receipts real. This pass prevents
those receipts from becoming another loose artifact pile. The runtime now emits
one audit surface that says what was produced, what was optional and absent, and
whether any required local receipt family is missing.

This is still local-only. It does not certify provider execution, benchmark
readiness, or calibration quality. It makes those future claims easier to audit
by forcing every run to enumerate its receipt surface explicitly.

### Verification

- `python3 -m ruff check src/world_model/sim_synth_physics tests/test_sim_synth_training_corpus.py tests/test_sim_synth_physics_world_model.py`
- `python3 -m pytest tests/test_sim_synth_training_corpus.py tests/test_sim_synth_physics_world_model.py tests/test_sim_synth_phase1x_surfaces.py tests/test_sim_synth_camera_geometry.py tests/test_sim_synth_vectorized_runtime.py -q` ->
  `40 passed`
- `python3 -m compileall src/ && python3 -m pytest tests/ -q` ->
  `1634 passed, 3 skipped, 24 warnings`

## 2026-05-18 - Phase 1.x replay-validity and task-consistency receipts

### What changed

- Added `ReplayValidityReceipt` to the Sim / Synth / Physics receipt family.
- Added `build_replay_validity_receipts(...)`, combining:
  - outcome receipt status / realized yield
  - branch-validity admissibility
  - task-measurement consistency
  - sim-real transfer consistency
  - sensor-alignment score
- `SimSynthPhysicsRuntime` now emits `replay_validity_receipts.json` and embeds
  per-branch replay validity in loop results and training-feedback rows.
- `training_corpus.py` now harvests replay-validity receipts and exposes
  aggregate reject reasons to backend-selector rows plus per-branch validity and
  consistency values to branch-planner rows.

### Why this was the right next local step

Branch validity decides whether a generated branch is admissible before or
during execution. Replay validity decides whether the resulting branch outcome
should be allowed into later training/evaluation surfaces. That second decision
needed its own receipt; otherwise invalid, blocked, or weakly grounded outcomes
could still look like ordinary training rows downstream.

The receipt is deliberately conservative. It records local estimates and reject
reasons now, then gives future provider replay, benchmark gates, and real
sim-real evidence a typed place to land later.

### Verification

- `python3 -m ruff check src/world_model/sim_synth_physics tests/test_sim_synth_camera_geometry.py tests/test_sim_synth_phase1x_surfaces.py tests/test_sim_synth_training_corpus.py tests/test_sim_synth_physics_world_model.py`
- `python3 -m pytest tests/test_sim_synth_camera_geometry.py tests/test_sim_synth_phase1x_surfaces.py tests/test_sim_synth_training_corpus.py tests/test_sim_synth_physics_world_model.py tests/test_sim_synth_vectorized_runtime.py -q` ->
  `40 passed`
- `python3 -m compileall src/ && python3 -m pytest tests/ -q` ->
  `1634 passed, 3 skipped, 24 warnings`

## 2026-05-18 - Phase 1.x sensor-alignment receipts

### What changed

- Extended `camera_geometry.py` with:
  - `camera_intrinsics_from_mapping(...)`
  - `transform_from_mapping(...)`
  - `camera_round_trip_error(...)`
- Added `SensorAlignmentReceipt` for CPU-local camera/sensor geometry checks.
- Added `build_sensor_alignment_receipt(...)`, deriving status from scene
  hierarchy semantic context, camera intrinsics, camera extrinsics / pose, and a
  local projection round-trip check.
- `SimSynthPhysicsRuntime` now emits `sensor_alignment_receipt.json` and embeds
  the receipt in loop results plus training-feedback rows.
- `training_corpus.py` now harvests the receipt and exposes alignment status,
  score, checks, and metrics to backend-selector and branch-planner rows.

### Why this was the right next local step

After branch-validity receipts, the next cheap Phase 1.x gap was sensor truth.
We cannot validate provider observation quality without GPU/runtime bring-up, but
we can make the geometry contract visible and testable now: intrinsics,
extrinsics, and projection round-trip posture are no longer implicit inside
semantic context or asset notes.

The receipt preserves honest semantics. `geometry_contract_validated` means the
local metadata is internally consistent; it does not mean a real camera, Isaac
render, UE5 render, or Unitree sensor path has been calibrated. Missing or
invalid metadata remains `alignment_contract_missing` or
`alignment_contract_invalid`.

### Verification

- `python3 -m ruff check src/world_model/sim_synth_physics tests/test_sim_synth_camera_geometry.py tests/test_sim_synth_phase1x_surfaces.py tests/test_sim_synth_training_corpus.py tests/test_sim_synth_physics_world_model.py`
- `python3 -m pytest tests/test_sim_synth_camera_geometry.py tests/test_sim_synth_phase1x_surfaces.py tests/test_sim_synth_training_corpus.py tests/test_sim_synth_physics_world_model.py tests/test_sim_synth_vectorized_runtime.py -q` ->
  `40 passed`
- `python3 -m compileall src/ && python3 -m pytest tests/ -q` ->
  `1634 passed, 3 skipped, 24 warnings`

## 2026-05-18 - Phase 1.x branch-validity and reject-filter receipts

### What changed

- Added `BranchValidityReceipt` to the Sim / Synth / Physics receipt family.
- Added `build_branch_validity_receipts(...)`, deriving per-branch validity
  from `Gen2SimAdmissionState`, branch admission preconditions, benchmark gate
  status, semantic-grounding posture, and scene-materialization status.
- `SimSynthPhysicsRuntime` now emits the receipt bundle as
  `branch_validity_receipts.json` and embeds the same evidence in loop results
  and training-feedback rows.
- `training_corpus.py` now harvests branch-validity receipts from live dirs or
  mixed receipt files.
- Backend-selector rows receive aggregate admission/reject counts and reject
  reasons; branch-planner rows receive the per-branch receipt id, validity
  score, admission score, admissibility flag, evidence status, and reject
  reasons.

### Why this was the right next local step

The previous Phase 1.x pass made scene/transfer evidence training-visible. The
next load-bearing gap was branch admissibility: without a durable validity
receipt, the system could plan branches and emit outcomes while losing the
reason a branch was admitted or rejected.

This pass keeps that decision replayable and trainable without pretending that
GPU/provider evidence exists. When benchmark gates are unavailable, validity is
marked as `local_estimate`; provider-season runs can later replace that with
benchmark-supported evidence through the same artifact shape.

### Verification

- `python3 -m ruff check src/world_model/sim_synth_physics tests/test_sim_synth_training_corpus.py tests/test_sim_synth_physics_world_model.py`
- `python3 -m pytest tests/test_sim_synth_training_corpus.py tests/test_sim_synth_physics_world_model.py tests/test_sim_synth_phase1x_surfaces.py tests/test_sim_synth_vectorized_runtime.py -q` ->
  `36 passed`
- `python3 -m compileall src/ && python3 -m pytest tests/ -q` ->
  `1633 passed, 3 skipped, 24 warnings`

## 2026-05-18 - Phase 1.x consumers: scene and transfer evidence made consequential

### What changed

- `compile_synthetic_branch_plans(...)` now consumes `SceneHierarchyState`.
- Each `SyntheticBranchPlan` now carries a `scene_hierarchy_ref` in both
  `gap_target_refs` and metadata.
- `compile_branch_render_provider_state(...)` now receives the scene hierarchy
  and records it in provider config / metadata.
- Render materialization source context and manifests now preserve the scene
  hierarchy reference.
- `sim_synth_training_feedback_v1` now includes a transfer-evidence summary
  over:
  - task measurement posture
  - sim-real gap score / realism confidence
  - backend mismatch / calibration staleness
  - surrogate forecast and calibration posture
- `training_corpus.py` now harvests the new receipt family and exposes it in
  backend-selector / branch-planner rows.

### Why this matters

The earlier Phase 1.x pass made a clean state/receipt family. This pass makes
that family harder to ignore: scene hierarchy now shapes branch/materialization
planning, and transfer evidence becomes visible to training-row consumers.

This is still not provider bring-up. The value is that future GPU/runtime runs
will arrive into a training/evidence pipeline that already knows how to carry
scene structure, sim-real risk, backend mismatch, and surrogate posture.

### Verification

- `python3 -m ruff check src/world_model/sim_synth_physics tests/test_sim_synth_phase1x_surfaces.py tests/test_sim_synth_training_corpus.py tests/test_sim_synth_physics_world_model.py`
- `python3 -m pytest tests/test_sim_synth_phase1x_surfaces.py tests/test_sim_synth_training_corpus.py tests/test_sim_synth_physics_world_model.py tests/test_sim_synth_vectorized_runtime.py -q`

## 2026-05-18 - Phase 1.x re-entry: first shared Sim / Synth / Physics surface family

### What changed

- Added live Phase 1.x typed state under `src/world_model/sim_synth_physics/`:
  - `TaskMeasurementSurface`
  - `SimulatorBackendContractState`
  - `TaskDefinitionContractState`
  - `SceneHierarchyState`
  - `DifferentiablePhysicsProviderState`
  - `SurrogatePhysicsProviderState`
- Added the first paired transfer / surrogate receipt family:
  - `TaskMeasurementReceipt`
  - `SimRealGapReceipt`
  - `BackendMismatchReceipt`
  - `SurrogatePhysicsReceipt`
  - `SurrogateCalibrationReceipt`
- The compiler now embeds those surfaces in `SimSynthPhysicsWorldState`, stable
  artifact refs, and the compiled receipt inventory.
- The runtime now emits the paired receipts into loop results, serialized
  artifact files, and training-feedback manifests.
- The Habitat-style sim→task→measurement protocol is now an explicit contract
  pair instead of an inference over loosely adjacent fields.
- Added CPU-local geometry helpers:
  - `camera_intrinsics_from_fov(...)`
  - `compose_transforms(...)`
  - `invert_transform(...)`
  - `unproject_depth(...)`
  - `project_points(...)`
- Added `VectorizedSimRunner` / `VectorizedSimBatchResult` as the first local
  batch-execution facade. It is intentionally `sequential_batch` today: the
  shape is now explicit, but there is no false claim of parallel GPU sim.

### Why this was the right Phase 1.x re-entry move

The roadmap had already named these surfaces as the cleanest CPU-local Phase
1.x opening, but they were still doctrine only. Making them real first gives
the reopenable Sim / Synth / Physics lane shared joints before we spend cycles
on provider-specific lanes that cannot yet be brought up locally.

The implementation stays epistemically honest:

- differentiable and surrogate providers default to `contract_reserved`
- surrogate calibration defaults to `not_calibrated`
- sim-real gap remains `estimated` until runtime evidence becomes real
- vectorized execution is a batch facade, not a speedup claim

That is the right posture while RunPod is unavailable: structure the future
evidence channels now, preserve provider sovereignty boundaries, and leave the
truth labels conservative.

### Verification

- `python3 -m compileall src/world_model/sim_synth_physics`
- `python3 -m pytest tests/test_sim_synth_phase1x_surfaces.py tests/test_sim_synth_camera_geometry.py tests/test_sim_synth_physics_world_model.py tests/test_sim_synth_vectorized_runtime.py -q`

## 2026-05-18 - Phase 2 final local pocket: LeRobot projection adapter parity

### What changed

- Finished the documented but previously missing LeRobot projection adapter:
  - `vision_backbone_projection_sample_from_lerobot_step(...)`
  - `vision_backbone_projection_samples_from_episode(...)`
  - `adapt_lerobot_episodes_for_vision_backbone_projection(...)`
- The adapter is intentionally honest:
  - camera slots become stable proxy identity labels
  - features are CPU-safe placeholder features unless a future provider season
    supplies real frozen-backbone outputs
  - cross-provider targets are proxy targets for plumbing verification only
- `scripts/smoke_test_vision_backbone_projection_seam.py` now supports:
  - `--data-source synthetic`
  - `--data-source mock_lerobot_droid`
  - `--data-source local_lerobot_rows`
- Added tests that verify both the adapter path and the local row-bundle proof
  path.

### Why this was the right last no-GPU Phase 2 move

There is no actual local real-data row bundle in the workspace today, so a real
external-data proof would have been theater. The useful local move was to make
the future real proof frictionless instead.

This closes a claim-vs-code mismatch: the adapter module already named a
LeRobot → projection path, but only evidence-fusion and temporal adaptation were
implemented. With this pass, the first promotion-chain seam now has the same
cheap intake grammar as the later seams.

The labels remain `camera_slot_proxy`, not real object identities. That is
deliberate: this is the last structural hardening pass before returning to
Phase 1.x, not a promotion claim.

### Local run

- `python3 scripts/smoke_test_vision_backbone_projection_seam.py --steps 30 --data-source mock_lerobot_droid --artifact-dir artifacts/phase2_local_proof_of_life/vision_backbone_projection_mock_lerobot_30 --require-loss-decrease`
- result: initial validation loss `5.8005`, best validation loss `5.7418`,
  `loss_decreased: true`, `3` training receipts, `3` validation receipts, `1`
  benchmark receipt

### Verification

- `python3 -m ruff check src/dataset_bridges/lerobot_perception_adapter.py scripts/smoke_test_vision_backbone_projection_seam.py tests/test_lerobot_perception_adapter.py tests/test_vision_backbone_projection_proof_of_life_smoke.py` ->
  pass
- `python3 -m pytest -q tests/test_lerobot_perception_adapter.py tests/test_vision_backbone_projection_proof_of_life_smoke.py` ->
  `49 passed`
- `python3 -m compileall src/ && python3 -m pytest tests/ -q` ->
  `1628 passed, 3 skipped, 24 warnings`

## 2026-05-18 - Phase 2 closure audit and live semantic-bridge receipts

### What changed

- Added `docs/economic_world_model/phase2_closure_assessment.md`.
- `src/world_model/perception_grounding/compiler.py` now emits live
  `SemanticBridgeReceipt` objects for:
  - `sim_synth`
  - `embodiment`
  - `annotation`
  - `economic`
- The serialized bridge receipts are stored in Perception WM metadata and are
  reconstructed into `compile_perception_grounding_with_receipts(...)` output.
- `tests/test_embodiment_shadow_consumer.py` now verifies:
  - `SemanticBridgeReceipt` is part of the live compilation family
  - all four active bridge kinds emit receipts
  - their quality / usefulness scores remain bounded

### Why this was the right closure move

The May 18 audit turned up one last genuine internal seam: the WM-native bridge
family was part of canonical state, but its receipt contract was still only a
type, not live compiler evidence. That is exactly the kind of quiet gap the
closure standard is meant to flush out.

Once the live bridge receipts landed, the audited structural sheet reached:

- Category A: `0`
- Category B: provider / GPU / real-data / calibration / held-out-evidence only
- Category C: `0`

So Phase 2 is now best described as **structurally closure-ready**, not
provider-ready and not promotion-ready. While GPU access is absent, we can keep
doing cheap local hardening opportunistically, but those tasks are no longer the
reason the phase itself must stay structurally open.

### Verification

- `python3 -m ruff check src/world_model/perception_grounding/compiler.py tests/test_embodiment_shadow_consumer.py` ->
  pass
- `python3 -m pytest -q tests/test_embodiment_shadow_consumer.py tests/test_perception_grounding_compiler.py` ->
  `39 passed`
- `python3 -m compileall src/ && python3 -m pytest tests/ -q` ->
  `1623 passed, 3 skipped, 24 warnings`

## 2026-05-18 - Phase 2 local vision-backbone projection proof artifacts

### What changed

- Added `scripts/smoke_test_vision_backbone_projection_seam.py`.
- The script runs `VisionBackboneProjectionSeam` through the real local trainer
  path and emits:
  - `vision_backbone_projection_seam_proof_of_life.json`
  - `vision_backbone_projection_metric_report.json`
  - `vision_backbone_projection_benchmark_evidence.json`
  - `training_runtime_manifest.json`
  - persistent checkpoint, registry summary, and full receipt JSON
- Added `tests/test_vision_backbone_projection_proof_of_life_smoke.py` to keep
  the artifact contract from drifting.

### Why this was the right next local Phase 2 step

The prior pass made `vision_backbone_projection` locally trainable and
benchmarkable. This pass finishes the near-term local job: the first seam in
the promotion dependency chain now has the same manifest / receipt / evidence
shape as EvidenceFusion and V-JEPA temporal alignment.

Because provider bring-up is intentionally delayed, this is the strongest
honest move available locally. It makes the eventual DINOv2/SigLIP season
consume a prepared lane rather than reopening mundane artifact plumbing under
scarcer GPU time.

The proof remains synthetic and provisional. It is a statement about
structural readiness, not provider readiness.

### Local run

- `python3 scripts/smoke_test_vision_backbone_projection_seam.py --steps 40 --artifact-dir artifacts/phase2_local_proof_of_life/vision_backbone_projection_synth_40 --require-loss-decrease`
- result: initial validation loss `5.7407`, best validation loss `5.2823`,
  `loss_decreased: true`, `4` training receipts, `4` validation receipts, `1`
  benchmark receipt

### Verification

- `python3 -m ruff check scripts/smoke_test_vision_backbone_projection_seam.py tests/test_vision_backbone_projection_proof_of_life_smoke.py src/training/perception_seam_data.py src/training/perception_seam_benchmarks.py tests/test_perception_seam_training.py` ->
  pass
- `python3 -m pytest -q tests/test_vision_backbone_projection_proof_of_life_smoke.py tests/test_perception_seam_training.py` ->
  `32 passed`

## 2026-05-18 - Phase 2 local vision-backbone projection training lane

### What changed

- `src/training/perception_seam_data.py` now has a real typed lane for
  `vision_backbone_projection`:
  - `VisionBackboneProjectionSample`
  - `VisionBackboneProjectionBatch`
  - `VisionBackboneProjectionDataset`
  - `generate_synthetic_vision_backbone_projection_samples(...)`
  - `create_vision_backbone_projection_loader(...)`
- `src/training/perception_seam_benchmarks.py` now includes
  `VisionBackboneProjectionBenchmark` and registers it in
  `BENCHMARK_REGISTRY`.
- The benchmark lane measures:
  - object-identity retrieval accuracy
  - scene-retrieval accuracy
  - cross-provider alignment score
- `tests/test_perception_seam_training.py` now covers:
  - projection loss behavior
  - projection sample generation
  - projection collation / loader creation
  - projection benchmark evaluation

### Why this was the right next local Phase 2 step

Provider bring-up is intentionally damped until RunPod or equivalent GPU
capacity is available. The remaining useful local work is therefore to make
the eventual provider season structurally cheap.

`vision_backbone_projection` is the first dependency in the current Phase 2
promotion chain, but before this pass it had no first-class trainer data lane
or benchmark evaluator even though the seam and loss already existed. That
would have forced future DINOv2/SigLIP bring-up work to solve basic local batch
plumbing during a much scarcer GPU window.

This pass does **not** claim provider readiness, benchmark credibility, or
promotion proximity. It only removes a local structural gap so later real
provider evidence has somewhere honest to land.

### Verification

- `python3 -m ruff check src/training/perception_seam_data.py src/training/perception_seam_benchmarks.py tests/test_perception_seam_training.py` ->
  pass
- `python3 -m pytest -q tests/test_perception_seam_training.py` ->
  `31 passed`
- `python3 -m compileall src/training/perception_seam_data.py src/training/perception_seam_benchmarks.py tests/test_perception_seam_training.py -q` ->
  pass
- `python3 -m compileall src/ && python3 -m pytest tests/ -q` ->
  `1621 passed, 3 skipped, 24 warnings`

## 2026-05-11 - Phase 2 local perception proof-of-life artifacts

### What changed

- `benchmark_evidence_emitter.py` no longer imports
  `evaluate_seam_on_annotations` at module import time. The import is now local
  to `emit_annotation_benchmark_evidence(...)`, which keeps fresh-process
  `src.training.perception_seam_data` imports from cycling through package
  `__init__` before the training-data module is initialized.
- `scripts/smoke_test_perception_seam_training.py` now emits typed local proof
  artifacts:
  - `perception_seam_proof_of_life_v2`
  - `perception_seam_metric_report_v1`
  - provisional `perception_benchmark_evidence_v1`
  - `training_runtime_manifest_v1`
  - persistent checkpoint and registry summary under the artifact directory
  - full training / validation / benchmark receipts
- `scripts/perception_proof_of_life_utils.py` now provides deterministic
  DROID-shaped mock LeRobot replay episodes so local adapter verification does
  not have to duplicate mock episode construction across proof scripts.
- the script records initial, final, and best validation loss and can enforce
  improvement with `--require-loss-decrease`.
- `--data-source mock_lerobot_droid` now creates DROID-shaped mock LeRobot
  episodes, converts them through `adapt_lerobot_episodes_for_evidence_fusion`,
  and trains the same EvidenceFusion seam. This is adapter-path proof only; it
  is still mock data, not external `droid_100`.
- `scripts/smoke_test_vjepa_temporal_seam.py` now provides the same local proof
  pattern for `VJEPATemporalAlignmentSeam`, including synthetic and
  mock-LeRobot temporal windows, typed artifact emission, and provisional
  benchmark evidence.
- both proof scripts now accept `--data-source local_lerobot_rows` plus a
  local JSON/JSONL LeRobot-like row bundle, so a tiny real-data export can
  drive the same proof path without introducing a HuggingFace dependency
  requirement into the local environment first.
- tests now cover both the original fresh-process import failure and the typed
  artifact bundles produced by the local EvidenceFusion and V-JEPA proof
  scripts.

### Why this was the right local Phase 2 step

GPU/provider bring-up is intentionally paused. The useful local move was to
prove that two different Phase 2 seams can produce durable evidence and
manifest artifacts without creating a promotion claim. This gives the later
real-data / GPU runs a contract-shaped landing zone while staying honest about
the current evidence class.

The produced evidence is synthetic and provisional. It is useful as plumbing
proof, not benchmark proof.

The new `local_lerobot_rows` path narrows the next gap: once a tiny DROID or
Bridge row export exists locally, the proof scripts can consume it directly.
That still does not make the result promotion-grade, but it turns the
external-data prototype step into an execution problem instead of a missing
contract problem.

### Local run

- `python3 scripts/smoke_test_perception_seam_training.py --steps 80 --artifact-dir artifacts/phase2_local_proof_of_life/evidence_fusion_80 --require-loss-decrease`
- result: initial validation loss `1.1481`, best validation loss `1.0016`,
  `loss_decreased: true`, `16` training receipts, `8` validation receipts, `1`
  benchmark receipt
- output bundle: ignored under `artifacts/phase2_local_proof_of_life/evidence_fusion_80/`
- `python3 scripts/smoke_test_perception_seam_training.py --steps 40 --data-source mock_lerobot_droid --artifact-dir artifacts/phase2_local_proof_of_life/mock_lerobot_droid_40 --require-loss-decrease`
- result: initial validation loss `1.1966`, best validation loss `1.1252`,
  `loss_decreased: true`, `8` training receipts, `4` validation receipts, `1`
  benchmark receipt
- output bundle: ignored under `artifacts/phase2_local_proof_of_life/mock_lerobot_droid_40/`
- `python3 scripts/smoke_test_vjepa_temporal_seam.py --steps 40 --data-source synthetic --artifact-dir artifacts/phase2_local_proof_of_life/vjepa_temporal_synth_40 --require-loss-decrease`
- result: initial validation loss `114.1529`, best validation loss `72.8604`,
  `loss_decreased: true`, `4` training receipts, `4` validation receipts, `1`
  benchmark receipt
- output bundle: ignored under `artifacts/phase2_local_proof_of_life/vjepa_temporal_synth_40/`
- `python3 scripts/smoke_test_vjepa_temporal_seam.py --steps 30 --data-source mock_lerobot_droid --artifact-dir artifacts/phase2_local_proof_of_life/vjepa_temporal_mock_lerobot_30 --require-loss-decrease`
- result: initial validation loss `169.5367`, best validation loss `128.8715`,
  `loss_decreased: true`, `3` training receipts, `3` validation receipts, `1`
  benchmark receipt
- output bundle: ignored under `artifacts/phase2_local_proof_of_life/vjepa_temporal_mock_lerobot_30/`
- focused tests now also exercise `--data-source local_lerobot_rows` for both
  proof scripts using a temporary LeRobot-like JSONL bundle

### Verification

- `python3 -m ruff check scripts/perception_proof_of_life_utils.py scripts/smoke_test_perception_seam_training.py scripts/smoke_test_vjepa_temporal_seam.py src/world_model/perception_grounding/benchmark_evidence_emitter.py tests/test_perception_seam_proof_of_life_smoke.py tests/test_vjepa_temporal_proof_of_life_smoke.py` ->
  pass
- `python3 -m ruff format --check scripts/perception_proof_of_life_utils.py scripts/smoke_test_perception_seam_training.py scripts/smoke_test_vjepa_temporal_seam.py src/world_model/perception_grounding/benchmark_evidence_emitter.py tests/test_perception_seam_proof_of_life_smoke.py tests/test_vjepa_temporal_proof_of_life_smoke.py` ->
  pass
- `python3 -m pytest tests/test_perception_seam_proof_of_life_smoke.py tests/test_vjepa_temporal_proof_of_life_smoke.py tests/test_lerobot_perception_adapter.py tests/test_perception_seam_training.py tests/test_perception_benchmark_evidence_emitter.py tests/test_provider_adapter_benchmark_evidence_emitter.py -q` ->
  `81 passed`

## 2026-05-11 - Phase 2 provider-adapter benchmark evidence emitter

### What changed

- `benchmark_evidence_emitter.py` now has a provider-adapter evidence path for:
  - `vision_backbone_projection`
  - `sam_calibration`
  - `depth_metric_calibration`
  - `vjepa_temporal_alignment`
- `scripts/emit_perception_provider_adapter_benchmark_evidence.py` exposes the
  path as a CLI. It accepts a JSON payload containing one provider invocation
  receipt, a list of receipts, or a Perception state metadata payload with
  `provider_adapter_receipts`.
- the emitter builds `perception_benchmark_evidence_v1` from receipt aggregates:
  success count, fallback count, output quality, output-token presence, latency
  budget posture, and receipt consistency.
- optional inputs are linked in evidence metadata:
  - provider-adapter checkpoint ref/status
  - `training_runtime_manifest_v1` path, digest, run id, training kind, and
    artifact keys
  - external metric-report path/digest for held-out, non-provisional benchmark
    scores
- receipt-only evidence remains provisional by default. Non-provisional evidence
  must come from an explicit override or metric report, so provider invocation
  success cannot silently become promotion-grade benchmark proof.

### Why this was the right next Phase 2 step

The compiler can now consume receipt-backed provider tokens, and annotation /
graph benchmark evidence can be emitted from persisted annotation exports. The
remaining missing piece was a provider-specific artifact lane for the adapters
themselves. This change gives each provider adapter a repeatable
evidence-emission path while preserving the important distinction between
runtime receipts and held-out benchmark evidence.

This is still not GPU provider bring-up. It creates the local artifact and
manifest-linking discipline needed for future DINO/SigLIP, SAM, depth, and
V-JEPA runs to become promotion inputs without turning receipt success into
sovereign truth.

### Verification

- `python3 -m ruff check src/world_model/perception_grounding/benchmark_evidence_emitter.py src/world_model/perception_grounding/__init__.py scripts/emit_perception_provider_adapter_benchmark_evidence.py tests/test_provider_adapter_benchmark_evidence_emitter.py` ->
  pass
- `python3 -m pytest tests/test_provider_adapter_benchmark_evidence_emitter.py -q` ->
  `3 passed`
- `python3 -m compileall src/world_model/perception_grounding scripts/emit_perception_provider_adapter_benchmark_evidence.py tests/test_provider_adapter_benchmark_evidence_emitter.py -q` ->
  pass
- `python3 -m compileall src/ && python3 -m pytest tests/ -v` ->
  `1610 passed, 3 skipped, 24 warnings`

## 2026-05-11 - Phase 2 runtime provider-token path

### What changed

- `compile_perception_grounding_world_state(...)` now lets successful runtime
  provider adapter outputs feed benchmark object tokens:
  - `vision_backbone_projection` output from `dinov2_vit_l_14`
  - `vjepa_temporal_alignment` output from `vjepa2`, reduced over temporal steps
- the token source becomes `provider_backed` only when the corresponding
  `ProviderInvocationReceipt` reports `invocation_status: success` and
  `fallback_used: false`
- failed, skipped, missing-receipt, or shape-incompatible provider outputs fall
  back to `heuristic_scene_graph` tokens with provisional evidence
- the provider surface now distinguishes live runtime provider inputs from the
  `vision_backbone_stub` posture and records `runtime_provider_inputs` metadata
  for SAM calibration, DINO/SigLIP projection, depth calibration, and V-JEPA
  temporal alignment
- default V-JEPA WM object tokens are padded to the seam's declared
  `d_wm_token`, so the temporal alignment seam can run from compiled scene graph
  tokens without an explicit `wm_object_tokens` argument

### Why this was the right next Phase 2 step

The previous tranche made persisted annotation-export benchmark evidence
routine. The next bottleneck was token provenance: benchmark object tokens still
mostly entered through explicit compile-time injection. This change makes the
live compiler path capable of using provider/runtime tensors, but only with a
successful invocation receipt attached.

This is not provider bring-up and not a promotion claim. Real DINOv2/SigLIP,
SAM, depth, and V-JEPA execution still needs GPU/provider work and benchmark
artifact density. The structural gain is that when those outputs are available,
annotation export and benchmark evidence can carry receipt-backed runtime truth.

### Verification

- `python3 -m ruff check src/world_model/perception_grounding/compiler.py tests/test_perception_grounding_compiler.py` ->
  pass
- `python3 -m pytest tests/test_perception_grounding_compiler.py -q` ->
  `18 passed`
- `python3 -m compileall src/ && python3 -m pytest tests/ -v` ->
  `1607 passed, 3 skipped, 24 warnings`

## 2026-05-11 - Phase 2 benchmark evidence emitter

### What changed

- added a routine annotation-export benchmark-evidence emitter:
  - `src/world_model/perception_grounding/benchmark_evidence_emitter.py`
  - `scripts/emit_perception_annotation_benchmark_evidence.py`
- the emitter loads persisted `annotation_export_v2` artifacts, evaluates
  `scene_graph_transformer` or `annotation_bridge_projection`, and writes a
  `perception_benchmark_evidence_v1` artifact
- the emitted evidence preserves:
  - source annotation export path
  - token provenance inferred from the export records
  - checkpoint reference status (`not_supplied`, `present`, or
    `missing_fresh_init`)
  - seam descriptor metadata
  - explicit `promotion_claim: not_implied_by_emitter`
- tests now cover provider-backed evidence, heuristic/provisional evidence, and
  the CLI emission path

### Why this was the right next Phase 2 step

The Phase 2 bottleneck was no longer "can the benchmark evidence type exist?"
or "can annotation records be evaluated in memory?" The missing step was a
repeatable persisted artifact producer over real annotation-export files. This
keeps the current priority stack focused on Perception / Grounding while making
promotion inputs easier to produce, inspect, and feed back into the compiler.

The next implementation target should be runtime provider-backed token
production: benchmark object tokens should come from live provider/runtime
outputs instead of mostly entering through explicit compile-time injection.

### Verification

- `python3 -m compileall src/world_model/perception_grounding scripts/emit_perception_annotation_benchmark_evidence.py tests/test_perception_benchmark_evidence_emitter.py -q` -> pass
- `python3 -m ruff check src/world_model/perception_grounding/benchmark_evidence_emitter.py scripts/emit_perception_annotation_benchmark_evidence.py tests/test_perception_benchmark_evidence_emitter.py src/world_model/perception_grounding/__init__.py` -> pass
- `python3 -m pytest -q tests/test_perception_benchmark_evidence_emitter.py tests/test_annotation_bridge_projection.py tests/test_perception_grounding_compiler.py tests/test_perception_grounding_neural_seams.py tests/test_embodiment_shadow_consumer.py` -> `113 passed`
- `python3 -m compileall src/ && python3 -m pytest tests/ -v` -> `1604 passed, 3 skipped, 24 warnings`

## 2026-05-11 - GR00T / VIRAL / DoorMan borrowing doctrine

### What changed

- added a standalone GR00T-VisualSim2Real borrowing note that treats GR00T /
  VIRAL / DoorMan as a concrete sim-to-real training/eval/config discipline,
  not as Ixion topology
- mapped portable patterns across WMs:
  - composable experiment specs
  - privileged teacher to deployable student lanes
  - domain-randomization provenance
  - dataset-reset curricula
  - eval/checkpoint/export gates
  - callback/measurement/receipt emitters
  - run-ledger discipline
- mapped the same patterns across the six Embodiment / Actuation subsystems
- updated roadmap and multi-WM docs to keep the later Phase 1.x
  Sim/Synth/Physics return explicit after Phase 2
- updated Perception / Grounding docs so Phase 2 can borrow deployable
  observation discipline now without shifting implementation priority
- added a light run-manifest cross-reference for future teacher/student and
  sim-to-real profile refs

### Why this was the right shape

The external repo is operationally useful because it shows a working plant:
Hydra-composed experiments, privileged-state teacher PPO, vision DAgger
students, RGB-delay stress, camera extrinsics randomization,
demonstration-seeded resets, checkpoint/eval/export flow, and W&B/callback
measurement loops.

The safe extraction is to encode those as typed contracts and receipts under
the existing WM topology. The unsafe extraction would be to make GR00T, Isaac,
PPO, DAgger, ResNet, ONNX, or G1 task primitives into stack ontology.

### Current sequencing

- Phase 2 Perception / Grounding remains the active implementation center.
- Phase 2 can borrow camera-bundle, egocentric-sensor,
  extrinsics-randomization, degraded-observation, and augmentation-provenance
  discipline now.
- After Phase 2, the roadmap returns to Sim / Synth / Physics Phase 1.x
  because additional provider-family, transfer-boundary,
  runtime/materialization, and run-manifest obligations were added after Phase
  1 structural closure.
- Phase 3 Embodiment should then consume the same teacher/student,
  dataset-reset, eval/export, and G1-facing config discipline through its six
  subsystems.

### Verification

- `python3 -m compileall src/` -> pass
- `python3 -m pytest tests/ -v` -> `1601 passed, 3 skipped, 24 warnings`

## 2026-04-11 — Nightly pass: audit-only execution, no safe scaffold delta

### What changed

- re-ran the nightly audit and refreshed:
  - `artifacts/economic_world_model/nightly_audit_summary.json`
  - `artifacts/economic_world_model/nightly_audit_summary.md`
- confirmed the audit-selected next task is still:
  - `id: audit_only`
  - `classification: docs_only`
  - `execute_now: false`

### Why this was the highest-value additive step

- the roadmap skill explicitly requires refreshing audit truth before taking new
  execution actions.
- this run's audit did not surface a higher-priority missing additive scaffold
  that is safe to execute automatically.
- preserving that skip discipline avoids speculative churn and keeps frozen
  zones untouched while verification remains green.

### Verification

- `python3 scripts/economic_world_model/nightly_audit.py --output-json artifacts/economic_world_model/nightly_audit_summary.json --output-markdown artifacts/economic_world_model/nightly_audit_summary.md` → pass
- audit-embedded checks:
  - `./scripts/agent/verify.sh` → pass
  - `python3 -m compileall src scripts/economic_world_model -q` → pass
  - `python3 -m pytest -q tests/test_runtime_packets.py tests/embodiment/test_registry.py tests/test_objective_runtime_builder.py tests/test_constraint_set.py tests/test_pricing_sentinel.py tests/test_value_ledger.py` → pass

### Current status and next task

- current nightly posture remains `docs_only` until the audit reports a safe,
  concrete additive scaffold.
- next recommended task remains unchanged: prioritize live-path sidecar wiring
  and governed-video preconditions before any training-lane expansion.

## 2026-04-08 — Phase 2 Perception / Grounding: benchmark-evidence discipline is now load-bearing

### What changed

The Perception / Grounding WM no longer treats benchmark evidence as an
untyped side dict around the seam path.

This pass made three concrete corrections:

- the annotation-export lane now has a real bounded neural successor:
  - `AnnotationBridgeProjectionSeam`
  - `annotation_bridge_projection_loss`
  - trainer dispatch / registry wiring
- annotation-export evaluation now preserves evidence provenance and refuses
  promotion when the object-token source is still heuristic
- graph transformer, annotation bridge, and provider-adapter promotion paths
  now accept typed persisted benchmark-evidence artifacts rather than relying
  on ambient in-memory mappings

### Why this matters

The important correction was not just “one more seam.”

It was making the promotion contract honest along the whole lower-WM lane:

- provider-backed object tokens are now preferred when benchmark/evaluation
  logic needs token matrices
- heuristic scene-graph tokens remain allowed only as explicit provisional
  fallback
- receipts and promotion resolvers now preserve that distinction instead of
  silently treating heuristic evidence as promotion-grade
- annotation export is now part of the promotion-evidence path, not only a
  downstream labeling convenience

This keeps the current Phase 2 posture aligned with the repo’s anti-fake-
promotion rule: structural wiring is allowed to land before GPU-era training,
but promotion cannot silently advance ahead of honest evidence.

### Current Phase 2 subsystem posture

- provider surface and canonical scene substrate are real and loop-facing
- shadow consumers now exist for sim/synth, annotation/export, and embodiment
- evidence routing / fusion has a first bounded neural seam
- annotation bridge now has its own bounded projection seam and training lane
- benchmark-evidence artifacts are now typed and persisted
- promotion governance is stricter for:
  - `scene_graph_transformer`
  - `annotation_bridge_projection`
  - provider-adapter seams

### Honest remaining work

The remaining bottleneck is no longer “missing seam scaffolding.”

It is the absence of routine non-provisional artifact production upstream of
promotion:

1. graph-transformer benchmark evidence still needs a routine artifact
   producer over the persisted annotation-export path
2. benchmark object tokens should be produced by actual provider/runtime
   outputs on the live path, not mainly by explicit compile-time inputs
3. SAM / depth / V-JEPA provider calibrators still need their own benchmark
   artifact producers and trainer-manifest linkage before promotion claims
   become operational rather than structural

## 2026-04-03 — Phase 2 Reconciliation: semantic successor topology made explicit

### What changed

The locally created Phase 2 package had a real topology gap: semantic bridge
types and doctrine existed, but they were not actually carried by the
top-level `PerceptionGroundingWorldState`. The branch also lacked the explicit
provider/dataset/task/deployment-resource surface family we discussed using
Habitat-style patterns for.

This pass fixed that by making the schema itself more operational:

- `PerceptionGroundingWorldState` now carries:
  - `provider_surface`
  - `dataset_surface`
  - `task_measurements`
  - `deployment_resource_surface`
  - `semantic_bridge_registry`
- `state.py` now defines:
  - `ProviderSurfaceState`
  - `DatasetSurfaceState`
  - `TaskMeasurementSurface`
  - `DeploymentResourceSurface`
  - `ComputeEnvelopeState`
  - `InferenceCapacityState`
  - `BatteryState`
  - `ThermalState`
- `receipts.py` now defines:
  - `ProviderAvailabilityReceipt`
  - `InferenceHeadroomReceipt`
  - `DeploymentResourceReceipt`

### Why this matters

This turns the branch away from “Perception owns semantics” as a slogan and
toward a more implementation-shaped subsystem:

- semantic bridges are now structurally part of the top-level state
- provider/runtime inventory is explicit
- dataset/world inventory is explicit
- task/measurement surfaces are explicit
- deployment/resource posture is explicit

That gives later compiler/runtime work a clear typed target without inventing a
new top-level WM or collapsing everything into one environment object.

### SemanticVLA posture

`src/vla/semantic_vla.py` remains importable for backward compatibility, but
it is now explicitly tested as scaffolding with successor metadata pointing to
the distributed semantic bridge family.

This is the right transitional posture:

- not promoted as the future semantic-analysis owner
- not deleted and forgotten
- explicitly replaced by:
  - canonical perception substrate
  - WM-native bridge family
  - provider-backed / fusion-backed evidence

### Semantic-bridge verification status

The semantic-bridge refinement is now explicit branch truth rather than a
half-landed local pass:

- semantic bridge registry/state is carried by the top-level Perception WM state
- semantic bridge promotion logic is covered by tests
- bridge serialization coverage now includes the embodiment bridge as well as
  the sim-synth, annotation, and economic bridge surfaces
- `SemanticVLA` scaffolding/successor metadata is covered by tests

### What still remains internal in Phase 2

- compiler/runtime build path for `PerceptionGroundingWorldState`
- evidence-fusion implementation
- temporal-grounding implementation
- provider-adapter wiring, including `backbone_stub.py` posture
- downstream Sim / Synth / Physics hook
- downstream annotation/evidence hook
- replay/training export wiring

## 2026-04-02 — Phase 2 Tranche 2.0: Perception / Grounding WM Schema

### What was built

New package `src/world_model/perception_grounding/` following the exact pattern established by `sim_synth_physics/`:

**State types** (`state.py`):
- `ObjectTrackState`: frozen dataclass with track_id, 3D pose (16-element homogeneous matrix), feature_token (d=128 vector matching Graph Transformer node dim), provider_sources, temporal persistence metadata, affordance/risk hints. Confidence and uncertainty are clipped to [0,1].
- `SceneEdge`: typed edge with explicit edge_type vocabulary (spatial_adjacency, contact, containment, occlusion, temporal_co_occurrence, affordance_relation) matching the neuralization doctrine's Graph Transformer edge types with d=64 edge features.
- `SceneGraphState`: scene graph aggregating object_tracks + edges + scene_summary_token (d=256). This is the primary output of the Graph Transformer and the canonical scene representation consumed by downstream bridges.
- `TemporalGroundingState`: temporal persistence state tracking visible/occluded/lost/recovered tracks, coherence scores, memory token count, and helper posture. This is the output of the causal transformer temporal grounding module.
- `EvidenceRoutingState`: evidence fusion ownership with per-provider contribution weights, fusion method, confidence/disagreement, and helper posture. This is the output of the set transformer (Perceiver-style) fusion module.
- `PerceptionGroundingWorldState`: top-level state composing scene_graph + temporal_grounding + evidence_routing + input_context + maturity_stage. Starts at `schema_only`, targets `shadow_runtime`.

**Receipt types** (`receipts.py`):
- `ProviderInvocationReceipt`: per-provider invocation/skip with quality, latency, fallback reason.
- `GroundingCalibrationReceipt`: calibration evidence with grounding accuracy, spatial accuracy, provider agreement, downstream task correlation.
- `EvidenceFusionReceipt`: per-fusion pass with provider IDs/weights, output counts, helper posture.
- `TemporalGroundingReceipt`: per-frame temporal tracking with maintained/lost/recovered/id-switch counts.
- `PerceptionContributionReceipt`: per-episode contribution receipt for Economic WM consumption.

**Provider contracts** (`provider_contracts.py`):
- `PerceptionProviderContract`: base contract with availability, provider_truth_class, loading_posture, learned_adapter_posture, calibration_status, fallback semantics.
- `SAMProviderContract`: SAM 3/3.1 specific — image/video predictor availability, memory/multiplex mode, calibration_head_posture, mask_to_token_projector_posture. Default unavailable with `scene_tracks_only` fallback.
- `VisionBackboneProviderContract`: DINOv2/SigLIP — backbone_dim=1024, projection_output_dim=128, projection_head_posture. Default unavailable with `deterministic_stub` fallback (existing backbone_stub.py).
- `VJEPAProviderContract`: V-JEPA 2 — dual-homing contract, upstream_repo=`facebookresearch/vjepa2`, projection_posture, temporal_alignment_head_posture. Default unavailable with `planning_only` fallback.
- `DepthProviderContract`: DepthAnythingV2/UniDepth — metric_calibration_head_posture, camera_intrinsics_required. Default unavailable with `scene_tracks_geometry_only` fallback.
- `PerceptionProviderRegistry`: registry composing all provider contracts.

**Promotion machinery** (`promotion.py`):
- `resolve_graph_transformer_helper()`: disabled|auto|required for the Graph Transformer
- `resolve_temporal_grounding_helper()`: disabled|auto|required for temporal grounding
- `resolve_evidence_fusion_helper()`: disabled|auto|required for evidence fusion
- Shared `_check_demotion()` with same three demotion triggers as sim_synth_physics (benchmark_gate_revoked, evidence_failure, failure_rate exceeding threshold)

### Design decisions

1. Feature token dimensionality (d=128 for objects, d=64 for edges, d=256 for scene summary) follows the neuralization bridge doctrine exactly.
2. Provider contracts default to `unavailable` — anti-stub compliant. No silent provider masquerade.
3. All learned components carry explicit `helper_posture` and `promotion_stage` fields in their state types, not just in promotion resolvers.
4. `PerceptionContributionReceipt` is explicitly designed for Economic WM consumption — carries grounding_quality, semantic_yield, calibration_confidence, action_relevance_prior, novelty_score, temporal_stability.
5. Maturity stage on the top-level state (`schema_only` → `logging_only` → `shadow_runtime` → ...) matches the WM maturity ladder from multi_wm_architecture_plan.md.

## 2026-04-02

- Finished a late-Phase-1 closure pass over the remaining local/runtime/install honesty seams:
  - `src/world_model/sim_synth_physics/runtime_launch.py` now blocks launch readiness on `asset::...` host-preflight gaps instead of letting those asset blockers sit only in binding/work-order metadata
  - `src/world_model/sim_synth_physics/runtime_work_orders.py` now preserves:
    - runtime-layout install-ready / install-partial / install-blocked profile groups
    - host-preflight ready / verified component sets
    - launch missing preconditions and notes
  - `src/world_model/sim_synth_physics/training_corpus.py` now preserves the same stronger local truth in backend-selector and branch-planner rows
- Why this matters:
  - before this tranche, launch/work-order/training surfaces were not perfectly aligned; a lane could still look cleaner in launch or trainer exports than the runtime binding actually said
  - after this tranche, blocked local runtime/install/asset truth is consistent across the late Phase-1 path
  - this was the last meaningful internal pseudo-readiness seam found in the audited closure pass

- Captured the current host reality explicitly:
  - the repo-root scan says both Isaac/Unitree and Holosoma have zero usable profiles on this host
  - no relevant runtime env vars are set
  - no external `isaaclab`, `unitree_sdk2py`, or `holosoma` Python modules are importable
  - no external Isaac/Unitree/Holosoma runtime roots were found in the common local clone directories the branch audits
- Why this matters:
  - the branch can now justify “remaining blockers are external” with an actual host report instead of architectural optimism
  - the next meaningful move is to install or point at real external runtimes/assets/checkpoints or bring up the GPU-backed materialization lane, not to add another Phase-1 abstraction

- Made `scripts/scan_phase1_runtime_layouts.py` a real repo-root CLI and host-reality summary surface:
  - inserted repo root into `sys.path` before `src.*` imports so the script now runs directly from the workspace root
  - added `scan_summary` compression for both backend lanes
  - the summary now preserves:
    - `usable_profiles`
    - `install_ready_profiles`
    - `install_partial_profiles`
    - `install_blocked_profiles`
    - selected policy / deploy / runtime-report refs and sources
    - selected verified / partial target ids
    - host-preflight blockers
- Why this matters:
  - Phase 1 is now at the point where the remaining blocker set is mostly external-runtime/asset/GPU reality, so the scan itself has to express that local truth clearly
  - on the current host, the script now reports both Isaac/Unitree and Holosoma as blocked with zero usable profiles, which is the right honest local read rather than a silent failure to inspect those lanes
  - this is another additive closure step that improves practical Category B observability without adding a new runtime ladder


- Made selected-ref validation operational in the downstream Phase-1 path:
  - `src/world_model/sim_synth_physics/runtime_work_orders.py` now treats mismatched/missing selected-runtime outputs as explicit runtime preconditions instead of still allowing `satisfied_by_external_runtime_outcomes`
  - `src/world_model/sim_synth_physics/training_corpus.py` now only prefers `external_runtime_outcome_receipt` as the backend-selector target source when selected-ref validation says the harvested outputs are actually acceptable
- Why this matters:
  - before this tranche, the branch could preserve mismatch truth in metadata but still act as if the outcome was good enough
  - after this tranche, the mismatch truth is load-bearing in completion posture and training-source selection
  - this is another Phase-1-local removal of pseudo-readiness rather than a new runtime abstraction

- Explicitly re-audited and closed the lingering Tier 3.4 / 3.5 verification ambiguity:
  - added `tests/test_sim_synth_phase1_verification.py`
  - Tier 3.4 now has direct assertions around:
    - inferential frontier gain / epiplexity / transfer behavior
    - provenance-quality influence on confidence
    - agenda score uplift from inferential priors
    - branch-contract confidence reaction to backend provenance flags
  - Tier 3.5 now has direct assertions around:
    - humanoid-target domain-randomization and system-ID policy compilation
    - unitree-target randomization axes and calibration targets
    - adaptation/calibration receipt reaction to route status and runtime evidence
- Why this matters:
  - these items were functioning, but they were still sitting in the closure sheet as “not yet explicitly re-audited”
  - this tranche turns them into explicitly verified surfaces instead of informal confidence
  - the closure conversation can now focus more narrowly on the remaining external-runtime/GPU blocker set

- Tightened Phase-1 runtime-outcome truth against selected runtime refs:
  - `src/world_model/sim_synth_physics/runtime_bundles.py` now ensures output-contract construction can see the already-selected runtime binding
  - `src/world_model/sim_synth_physics/runtime_outcomes.py` now carries expected selected policy / deploy-config / runtime-report refs inside the output contract, includes them as exact refs when locally present, and emits `selected_ref_validation` in the output summary / outcome receipt
  - `src/world_model/sim_synth_physics/runtime_work_orders.py` and `training_corpus.py` now preserve that validation status in execution-facing and trainer-facing artifacts
- Why this matters:
  - before this tranche, “runtime outputs harvested” did not say whether the harvested artifacts actually matched the selected runtime refs that the lane intended to use
  - after this tranche, selected-runtime mismatch or missing-output truth is explicit and replayable on the audited path
  - this keeps collapsing repo-local ambiguity without introducing a new ladder rung

- Tightened Phase-1 concrete ref selection against real local runtime artifacts:
  - `src/world_model/sim_synth_physics/ref_evidence.py` now exposes candidate-selection and candidate-summary helpers, so ref choice can be driven by verification status instead of list order
  - `src/world_model/sim_synth_physics/adapters/isaac_unitree_runtime_pack.py` and `src/world_model/sim_synth_physics/adapters/holosoma_runtime_pack.py` now choose:
    - `primary_policy_ref`
    - `primary_deploy_config_ref`
    - `primary_runtime_report_ref`
    from the best verified local candidate when available, and preserve the chosen source plus candidate-evidence summaries
  - `src/world_model/sim_synth_physics/adapters/isaac_unitree_runtime_binding.py` and `src/world_model/sim_synth_physics/adapters/holosoma_runtime_binding.py` now preserve `selected_*_source` on the binding path instead of quietly inheriting first-candidate ordering
  - `src/world_model/sim_synth_physics/runtime_work_orders.py` and `training_corpus.py` now carry that source/evidence truth into execution-facing and trainer-facing artifacts
- Why this matters:
  - before this tranche, the branch could still choose a worse local artifact simply because it appeared earlier in a candidate list
  - after this tranche, real verified local checkpoint/report/deploy artifacts outrank missing earlier candidates on the audited path
  - this keeps pushing the remaining ambiguity outward toward real external runtime/install/GPU reality instead of repo-local candidate ordering

- Promoted usable-profile truth into the runtime-layout contract itself:
  - `src/world_model/sim_synth_physics/runtime_layouts.py` now emits:
    - `usable_profiles`
    - `install_ready_profiles`
    - `install_partial_profiles`
    - `install_blocked_profiles`
  - this keeps `ready_profiles` available for the broader “root exists” view while making the stronger profile truth first-class and replayable
- Threaded that usable-profile truth through the downstream Phase-1 path:
  - `src/world_model/sim_synth_physics/runtime_bundles.py` now uses `usable_profiles` when choosing/ordering profiles
  - `src/world_model/sim_synth_physics/runtime_bridge.py` now emits `runtime_layout_usable_profiles`
  - `src/world_model/sim_synth_physics/runtime_work_orders.py`, `compiler.py`, and `training_corpus.py` now preserve that field in execution-facing and trainer-facing artifacts
- Why this matters:
  - after the previous tranche, deployment/runtime-pack logic already knew the stronger truth, but downstream consumers still had to reconstruct it or silently assume `ready_profiles` meant “usable”
  - this tranche closes that internal mismatch on the audited path
  - the remaining ambiguity is pushed further out toward actual host/runtime/install/assets/checkpoints/GPU reality

- Tightened the Phase-1 profile/policy selection seam against real local runtime reality:
  - `src/world_model/sim_synth_physics/runtime_layouts.py` now evaluates multiple candidate policy roots and chooses the root that actually carries checkpoint evidence instead of letting an explicit-but-empty policy root win by position alone
  - the policy contracts now preserve `policy_root_source` and the candidate-root rows used for that decision
- Tightened deployment/runtime-pack readiness without adding a new ladder rung:
  - `src/world_model/sim_synth_physics/adapters/isaac_unitree_deployment.py` and `src/world_model/sim_synth_physics/adapters/holosoma_deployment.py` now treat install-blocked profiles as unusable and use verified targets instead of raw target existence
  - `src/world_model/sim_synth_physics/adapters/isaac_unitree_runtime_pack.py` and `src/world_model/sim_synth_physics/adapters/holosoma_runtime_pack.py` now prefer usable profiles and preserve `runtime_target_preflight_status` plus verified/unverified target truth
- Why this matters:
  - before this tranche, the branch could still overstate readiness in two ways:
    - an explicit but empty policy root could mask a discovered runtime root with real checkpoints
    - an install-blocked repo root could still count as a usable runtime profile downstream
  - this tranche removes both pseudo-readiness seams on the audited path
  - the remaining blocker is more honestly “real install/assets/checkpoints/GPU existence” rather than internal selection optimism

- Tightened the Phase-1 target-preflight path without adding a new ladder rung:
  - `src/world_model/sim_synth_physics/runtime_targets.py` now computes install-shape verification for runtime targets instead of stopping at `Path.exists()`
  - the verification is additive and marker-based:
    - exact markers for repo/install shapes
    - glob markers for assets, checkpoints, motion clips, and retargeting bundles
    - `verification_status` reflects `missing`, `local_path_exists`, `install_shape_ready`, `install_shape_partial`, or `install_shape_missing`
  - `ready_target_ids` semantics were intentionally left unchanged to avoid broad churn; the stronger truth is consumed at the binding/preflight/export layer instead
- Consumed that stronger truth where it matters:
  - `src/world_model/sim_synth_physics/ref_evidence.py` now treats selected evidence with `ready == False` as unready/missing for host-preflight summarization
  - `src/world_model/sim_synth_physics/adapters/isaac_unitree_runtime_binding.py` and `src/world_model/sim_synth_physics/adapters/holosoma_runtime_binding.py` now build selected-target evidence from runtime-target rows instead of plain `describe_ref_evidence(ref)`
  - those bindings now emit:
    - `selected_verified_target_ids`
    - `selected_partial_target_ids`
  - `src/world_model/sim_synth_physics/runtime_work_orders.py` and `src/world_model/sim_synth_physics/training_corpus.py` now preserve those fields plus selected-target evidence in executor-facing and trainer-facing artifacts
- Why this matters:
  - the branch previously had a quiet mismatch between:
    - richer pack/layout/install truth
    - weaker selected-target truth at the binding layer
  - this tranche closes that mismatch on the audited path
  - empty SDK/assets/motion/retargeting directories no longer look equivalent to install-shaped targets once a binding is actually selected
  - this is another Phase-1-local removal of fake readiness while keeping the current runtime ladder stable

- Closed the active Tier 3 shadow/fallback honesty gap without adding a new runtime rung:
  - `src/world_model/sim_synth_physics/shadow_execution.py` now derives binding-aware Isaac env-config and Holosoma work-order fields from the already-emitted `BackendRuntimeExecutionReceipt` metadata instead of only carrying the deeper runtime ladder as receipt-side metadata
  - the shadow layer now consumes:
    - selected profile
    - selected policy ref
    - selected launch root
    - selected target refs
    - selected motion sources / retargeting root (Holosoma)
    - host-preflight and selected-profile install status
  - Holosoma shadow preconditions now merge selected binding preflight/install blockers into the work-order truth instead of relying only on missing-asset lists
  - shadow receipt metadata now includes `shadow_runtime_binding_consumed`
- Tightened branch-planner fallback honesty:
  - `src/world_model/sim_synth_physics/synthetic_branches.py` now computes:
    - `branch_helper_resolution`
    - `branch_helper_resolution_reason`
    - `branch_helper_payload_applied`
  - these fields explicitly distinguish:
    - learned payload applied
    - heuristic retained because helper is shadow-candidate
    - heuristic retained because helper was demoted
    - heuristic retained because helper is unavailable
  - `src/world_model/sim_synth_physics/training_corpus.py` now preserves those fields plus learned-trace generation/yield hints in branch-planner training rows
- Why this matters:
  - previously the branch could carry a learned helper trace while still forcing downstream consumers to infer whether that trace actually controlled the plan
  - now the control authority split between heuristic and learned helper is explicit in both canonical planning metadata and trainer exports
  - combined with the shadow binding change, this closes another Phase-1-local honesty gap while keeping the current ladder structure intact

- Implemented Tier 3.2 promotion/demotion machinery (Claude-authored):
  - Added `_check_demotion(benchmark_gate, evidence_signals)` to `promotion.py`
  - Three demotion triggers:
    - `benchmark_gate_revoked`: explicit boolean signal
    - `evidence_failure`: explicit boolean signal
    - `recent_failure_rate > demotion_failure_threshold` (threshold from benchmark_gate, default 0.5)
  - Demoted stage is `demoted_to_shadow` (weight 0.25, same as shadow_candidate)
  - Status dict carries `demotion_reason` and `evidence_signals` when demoted
  - `_check_demotion` is a shared function imported by both `backend_selector_runtime.py` and `branch_planner_runtime.py`
  - All three resolvers (`resolve_helper`, `resolve_backend_selector_helper`, `resolve_branch_planner_helper`) accept optional `evidence_signals` param
  - Backward compatible: no `evidence_signals` → no demotion check → existing behavior unchanged
  - Consumer impact: `compiler.py:312`, `calibration.py:139`, `synthetic_branches.py:314` check `promotion_stage == "promoted"` — a demoted helper will correctly NOT match this check, causing fallback to heuristic behavior
  - Fixed `test_holosoma_binding_records_runtime_target_contract` stale assertion: `pack_status` now accepts `pack_partial` since install-hardened pack correctly reports partial readiness for an empty test directory

- Pushed the Phase-1 Category B edge further into actual local host/runtime consumption:
  - added `src/world_model/sim_synth_physics/local_runtime_discovery.py`
  - targeted autodiscovery now checks common local roots such as:
    - `$HOME/code`
    - `$HOME/src`
    - `$HOME/repos`
    - workspace-adjacent roots
  - this is deliberately narrow and exact-name based; it does not mark a lane ready unless the discovered path actually exists
- Preserved that discovery inside the existing runtime path rather than inventing a new layer:
  - `runtime_targets.py` now exposes autodiscovered upstream roots as normal runtime-target refs with `source=autodiscovery`
  - `runtime_layouts.py` now uses the same discovery path for runtime layouts and policy-contract fallback
  - Isaac/Holosoma policy contracts can now consume real checkpoints/configs/reports from discovered upstream runtime roots when no explicit policy root is wired
- Why this matters:
  - this closes another real internal gap between “host has a real clone/checkpoint locally” and “Phase 1 can actually see and use that evidence”
  - the remaining blocker is increasingly whether those real installs/assets/checkpoints exist at all, not whether the WM can notice them once they do

- Consumed the richer upstream runtime evidence against more concrete local host/runtime truth:
  - added `src/world_model/sim_synth_physics/ref_evidence.py` for selected-ref verification
  - `src/world_model/sim_synth_physics/adapters/isaac_unitree_runtime_binding.py` now emits:
    - `selected_ref_evidence`
    - `selected_target_ref_evidence`
    - `selected_asset_ref_evidence`
    - `host_preflight_status`
  - `src/world_model/sim_synth_physics/adapters/holosoma_runtime_binding.py` now emits:
    - selected policy / launch / retargeting evidence
    - selected existing motion-source evidence
    - `missing_motion_sources`
    - `host_preflight_status`
- Preserved that truth downstream:
  - `src/world_model/sim_synth_physics/runtime_launch.py` now consumes non-asset host-preflight blockers and exposes host-preflight status in launch plans/receipts
  - `src/world_model/sim_synth_physics/runtime_work_orders.py` now preserves host-preflight state in work-order metadata
  - `src/world_model/sim_synth_physics/training_corpus.py` now preserves host-preflight fields in backend-selector and branch-planner rows
- Why this matters:
  - upstream runtime packs already knew “declared vs verified” and “existing vs missing”
  - this tranche made the selected runtime-binding surfaces consume that truth instead of flattening it during mode-specific selection
  - Phase 1 can now be more explicit about the distinction between contract-readiness and locally verified launch/runtime readiness

- Normalized the Claude handoff artifact discipline:
  - `docs/economic_world_model/claude_to_comment_on.md` is now reset to a single current-state artifact
  - historical tranche detail should stay in:
    - `docs/economic_world_model/progress_log.md`
    - `docs/economic_world_model/implementation_notes.md`
- Why this matters:
  - the old accretive pattern made it harder to tell current branch truth from stacked historical context
  - the new pattern is better aligned with the repo doctrine that Codex should leave a clean handoff after meaningful implementation steps

- Closed the last explicit compiler-side incompleteness items from the active Phase-1 verification tranche:
  - `src/world_model/sim_synth_physics/state.py` now treats `PhysicsExecutionContract` as canonical world-state, not only a runtime byproduct
  - `src/world_model/sim_synth_physics/compiler.py` now compiles that contract with the active fallback-backend posture and emits `compiled_receipt_inventory` plus `runtime_depth_projection` metadata
  - `src/world_model/sim_synth_physics/runtime.py` now trusts the compiled execution contract on the normal path
- Preserved that closure into downstream data surfaces:
  - `src/world_model/sim_synth_physics/training_corpus.py` now harvests `physics_execution_contract`
  - backend-selector and branch-planner rows now preserve:
    - `physics_execution_contract_id`
    - compiled route status
    - requested/resolved backend
    - compiler-owned receipt inventory id
    - projected binding / bridge / upstream-pack status
- Test coverage now checks the new compiler-owned closure directly:
  - `tests/test_sim_synth_physics_world_model.py` verifies:
    - compiled execution-contract presence
    - artifact refs carry the execution-contract id
    - compiled receipt inventory round-trips through `to_dict()`
  - `tests/test_sim_synth_training_corpus.py` verifies:
    - harvested bundles preserve `physics_execution_contract`
    - trainer rows preserve the new compiler-side metadata
- Why this matters:
  - the active Phase-1 closure argument is now much stronger
  - the compiler/runtime boundary is no longer hiding one of its most load-bearing backend-routing truths
  - downstream training/export paths now carry both runtime receipts and the pre-runtime compiler closure that produced them
- Current judgment:
  - audited Category A cluster: closed
  - remaining work is increasingly external-runtime / asset / GPU constrained, though Phase 1 should still stay active until those lanes are exercised more concretely

- Nightly audit selector now treats failed verification as first-class scheduling input:
  - `scripts/economic_world_model/nightly_audit.py` adds `_verification_repair_task(verification)` and routes `_next_task(...)` through it before scanning additive scaffolding candidates
  - explicit `agent_verify` handling now emits:
    - `id`: `agent_verify_regression`
    - `classification`: `verification_hardening`
    - rationale and target files aligned with current failure posture
  - non-agent verification failures now emit a generic `verification_regression` task
- Why this matters:
  - prior behavior could claim “No missing additive step detected” even with failing baseline checks, which undermined nightly autonomy quality
  - the updated behavior keeps nightly execution aligned with “green baseline first, additive roadmap second”
- Test coverage:
  - `tests/test_economic_world_model_nightly_audit.py` now asserts:
    - `agent_verify` failure takes precedence over scaffold candidates
    - generic verification failure also takes precedence
  - existing `_next_task()` tests were updated to pass explicit verification context
- Validation run:
  - `python3 -m pytest -q tests/test_economic_world_model_nightly_audit.py`
  - `python3 scripts/economic_world_model/nightly_audit.py --output-json artifacts/economic_world_model/nightly_audit_summary.json --output-markdown artifacts/economic_world_model/nightly_audit_summary.md`
  - `python3 -m compileall src/`
  - `python3 -m pytest tests/ -v` (1329 passed, 3 skipped)

- Ran the active Tier 1 / Tier 3 Phase-1 verification tranche and closed three real internal incompleteness items:
  - `src/world_model/sim_synth_physics/gen2sim_admission.py` now builds a typed `Gen2SimAdmissionReceipt`
  - `src/world_model/sim_synth_physics/runtime.py` now emits and writes that receipt beside the existing adaptation / binding / bridge / runtime / shadow / render / outcome artifacts
  - `src/world_model/sim_synth_physics/training_corpus.py` now harvests the gen2sim receipt and preserves it in backend-selector and branch-planner rows
- Deepened shadow-execution honesty without adding a new ladder rung:
  - `src/world_model/sim_synth_physics/shadow_execution.py` now threads the already-existing runtime-ladder truth into `BackendShadowExecutionReceipt` metadata:
    - runtime execution receipt id / status
    - adapter receipt id / status / execution path
    - adapter realization posture
    - launch receipt id / status
    - outcome receipt id / status
    - runtime-binding selected profile / policy / launch root
    - `shadow_harvest_mode`
  - this closes the earlier mismatch where Tier 2 runtime bring-up surfaces existed but the Tier 3 shadow lane could still jump around them
- Tightened trainer/export completeness on the branch-planner lane:
  - branch-planner rows now preserve:
    - adaptation receipt id
    - calibration receipt id / score
    - shadow execution receipt id / status
    - gen2sim receipt id / admissible/blocked counts
  - backend-selector rows now also preserve `backend_shadow_harvest_mode`
- Added focused coverage:
  - `tests/test_sim_synth_branch_helpers.py` now covers typed gen2sim receipt generation
  - `tests/test_sim_synth_physics_world_model.py` now checks:
    - gen2sim receipt emission
    - shadow receipt runtime-ladder threading
    - `shadow_harvest_mode`
    - world-state `to_dict()` round-trip for core Phase-1 state
  - `tests/test_sim_synth_training_corpus.py` now checks that harvested bundles preserve gen2sim / adaptation / calibration / shadow truth in trainer rows
- Current closure judgment after this tranche:
  - resolved internal gaps:
    - gen2sim state-only termination
    - shadow path bypassing deeper runtime truth
    - branch-planner receipt-chain flattening
  - remaining Category A cluster:
    - `PhysicsExecutionContract` still lives at runtime rather than as canonical compiled state
    - compiler-side state still does not carry the deeper runtime-binding depth as explicitly as the runtime artifact chain does
  - honestly externalized remainder:
    - real Isaac / Unitree runtime, assets, checkpoints
    - real Holosoma host/runtime, motion/retargeting assets, policies
    - real GPU-backed GGDS / LDM materialization
- Focused validation run:
  - `python3 -m compileall src/world_model/sim_synth_physics tests/test_sim_synth_branch_helpers.py tests/test_sim_synth_training_corpus.py tests/test_sim_synth_physics_world_model.py -q`
  - `python3 -m ruff check src/world_model/sim_synth_physics tests/test_sim_synth_branch_helpers.py tests/test_sim_synth_training_corpus.py tests/test_sim_synth_physics_world_model.py`
  - `python3 -m pytest -q tests/test_sim_synth_branch_helpers.py tests/test_sim_synth_training_corpus.py tests/test_sim_synth_physics_world_model.py`
  - `python3 -m pytest -q tests/test_sim_synth_physics_scripts.py tests/test_sim_synth_runtime_launch.py`
  - `git diff --check`

## 2026-03-27

- Added canonical external-launch receipt handling to the Phase-1 backend runtime seam:
  - `src/world_model/sim_synth_physics/receipts.py` now defines `backend_runtime_launch_receipt_v1`
  - `src/world_model/sim_synth_physics/runtime_launch.py` now maps prepared or executed launch results into that receipt
  - `src/world_model/sim_synth_physics/backend_runtime_execution.py` now attaches the launch receipt to runtime execution metadata and writes launch report / launch receipt artifacts whenever the lane stops at external runtime bring-up
  - the runtime can now optionally execute that external launch path directly through `SimSynthPhysicsRuntime.execute_world_state(..., execute_external_runtime_launch=True)` / `run_planning_window(...)`
- Preserved the new receipt end to end:
  - `src/world_model/sim_synth_physics/runtime.py` now carries it through runtime evidence, feedback manifests, loop summaries, and artifact emission
  - `scripts/run_phase1_runtime_launch.py` now exposes the receipt directly so standalone launch/preflight runs can still produce a canonical artifact
  - `src/world_model/sim_synth_physics/training_corpus.py` now harvests the launch receipt and exposes launch status in backend-selector and branch-planner rows
- Why this matters:
  - Phase 1 is no longer limited to “runtime launch prepared” as an opaque side effect
  - it can now record whether an upstream runtime launch was blocked, prepared, executed, or failed, which is much closer to the mechanics-first stopping condition we want before calling the remainder external-runtime/GPU constrained

- Refined the roadmap doctrine for compute and battery as first-class allocatable resources:
  - `docs/economic_world_model/multi_wm_architecture_plan.md` now places inferential compute capacity / availability and concrete battery state earlier than the economic WM:
    - Phase 3 owns canonical embodiment/deployment-adjacent resource state
    - Phase 3.5 audits G1/R1-capacity realism for those contracts
    - Phase 4A / 4E make their runtime and communication consequences real
    - Phase 5 later turns them into allocatable economic objects
    - transport and meta-node layers only learn over those receipts later
  - the plan now explicitly names the kinds of state and behavior to instantiate:
    - compute envelope
    - battery / reserve / thermal posture
    - placement class
    - allocatable headroom
    - QoS / degraded-mode receipts
    - resource-aware backend / fidelity / inference decisions
- Updated the related planning artifacts:
  - `docs/economic_world_model/roadmap.md` now includes an explicit sequencing and staged-RL rule for compute/battery resource handling
  - `docs/economic_world_model/humanoid_target_readiness.md` now includes compute-envelope / placement readiness, concrete battery-state readiness, and compute-pressure degradation as explicit humanoid-target checklist items
- Why this matters:
  - it keeps the roadmap aligned with the branch’s core doctrine that lower WMs own replayable typed state first, while the economic WM later governs over those contracts instead of inventing them
  - it also makes “energy” concrete in a way that will matter later for Unitree-class deployment, offload decisions, and bounded inferential spend

- Added OSS-shaped runtime-layout and policy contracts inside Phase 1:
  - `src/world_model/sim_synth_physics/runtime_layouts.py` now scans for backend runtime layouts and policy banks rather than only generic root existence
  - for Isaac/Unitree it recognizes `IsaacLab`, `unitree_sim_isaaclab`, `unitree_rl_gym`, `HumanoidVerse`, `xr_teleoperate`, and Unitree asset/policy roots
  - for Holosoma it recognizes repo, motion-bank, policy-bank, and retargeting-bundle posture
  - those contracts now flow through backend bindings, runtime bridges, runtime work orders, and `src/orchestrator/loop_run_backlog.py`
- Why this matters:
  - the repo can now say which upstream-style runtime surface is actually present on a host
  - that is more actionable than a flat “runtime target exists” bit and better aligned with the honest Phase-1 remainder around roots, assets, and policies

- Added WM-owned backend runtime bundles and launch specs:
  - `src/world_model/sim_synth_physics/runtime_bundles.py` now emits:
    - `backend_runtime_bundle_v1`
    - `backend_launch_spec_v1`
  - `src/world_model/sim_synth_physics/runtime_launch.py` and `scripts/run_phase1_runtime_launch.py` now provide the launcher/preflight layer over those artifacts
  - `src/world_model/sim_synth_physics/backend_runtime_execution.py` writes those artifacts whenever it materializes an Isaac/Unitree or Holosoma runtime request
  - the preferred launch command is derived from the discovered layout/profile/policy posture instead of being left as a human-only note
  - `src/world_model/sim_synth_physics/runtime_work_orders.py` now threads that launch command into work-order `command_hints`
  - if the external host is otherwise ready but there is still no in-process backend adapter, the WM now emits `runtime_launch_prepared` as an honest intermediate status
- Why this matters:
  - the Phase-1 runtime lane now has a canonical “what to launch next” artifact
  - this is exactly the kind of mechanics-first, scalable runtime plumbing the roadmap wants before claiming the honest remainder is external runtime/GPU/asset access

- Added explicit backend runtime work orders on top of the new bridge contract:
  - `src/world_model/sim_synth_physics/runtime_work_orders.py` now emits typed work orders for the Isaac/Unitree and Holosoma runtime lanes
  - those work orders preserve:
    - linked non-training GPU backlog ids
    - command hints
    - missing runtime targets
    - missing assets
    - runtime preconditions
    - concrete-vs-shadow execution satisfaction
  - `src/world_model/sim_synth_physics/runtime.py` now writes `backend_runtime_work_orders.json` and threads work-order status into loop summaries and training-feedback manifests
- Why this matters:
  - the Phase-1 backend lane now produces an executor-facing bring-up artifact, not just a planning receipt
  - this is the right posture for “data/GPU/runtime availability is the blocker” because the loop now says exactly what still needs to be run when the GPU/runtime window opens

- Added a typed backend runtime bridge inside Phase 1:
  - `src/world_model/sim_synth_physics/runtime_bridge.py` now compiles `BackendRuntimeBridgeState`
  - the state captures:
    - transport profile and runtime stack
    - planner / control / observation rates
    - action decimation and latency budget
    - observation/action/telemetry contracts
    - safety channels
    - runtime-target readiness and missing-target truth
  - `src/world_model/sim_synth_physics/runtime.py` now emits `backend_runtime_bridge_receipt_v1` beside the binding, runtime, shadow, calibration, render, and outcome receipts
- Why this matters:
  - the WM no longer treats “binding exists” as enough to describe the backend lane
  - it now owns the actual slow-loop-to-runtime contract surface the rest of the stack will have to trust later for Isaac/Unitree/Holosoma execution
  - replay/training no longer need to reverse-engineer planner-vs-servo timing, transport posture, or missing runtime targets from scattered metadata

- Preserved the bridge contract into downstream receipt harvesters:
  - `src/world_model/sim_synth_physics/training_corpus.py` now harvests `backend_runtime_bridge_receipt_v1`
  - backend-selector and branch-planner rows now keep:
    - bridge receipt id
    - bridge status
    - execution authority
    - transport profile
    - bridge readiness score
    - missing runtime targets
- This is an important complete-subsystem step:
  - the new bridge contract is not runtime-only bookkeeping
  - it already affects the trainer/export truth path
  - that is the right standard for Phase 1 if we want the honest stopping condition to become runtime/assets/GPU limits rather than more missing WM plumbing

- Added a concrete backend-runtime receipt path inside Phase 1:
  - `src/world_model/sim_synth_physics/backend_runtime_execution.py` now binds requested Isaac/Holosoma backends into explicit runtime requests and optional concrete `evaluate_policy(...)` execution
  - `src/world_model/sim_synth_physics/runtime.py` emits `backend_runtime_execution_receipt_v1` beside the existing shadow receipt
  - the loop therefore now has three honest backend postures:
    - request/materialization only when policy/runtime are missing
    - shadow execution/work-order materialization
    - concrete backend evaluation when the runtime module and policy are actually present
- Why this matters:
  - Phase 1 is no longer blocked on inventing a backend-runtime receipt contract
  - the remaining backend gap is now much more honestly “real runtime module + real policy/assets/calibration” rather than “missing WM plumbing”
  - downstream trainer/export code can now distinguish shadow-runtime from concrete-runtime evidence

- Added conditional concrete render execution under the Phase-1 provider seam:
  - `src/world_model/sim_synth_physics/render_materialization.py` now executes NAG counterfactual generation when a real source LSD episode exists and the provider is genuinely non-stub
  - it now executes GGDS scene optimization when a real source Gaussian scene exists and the optimizer is concretely initialized
  - when those preconditions are absent, the loop stays on explicit work-order receipts with named missing requirements
- Why this matters:
  - the provider seam is now closer to the target “real-or-unavailable” posture
  - the remaining gap is the actual renderer/LDM/runtime stack, not missing WM-owned materialization logic
  - this is exactly the direction Phase 1 should be moving before we claim it is externally blocked

- Preserved the new robot-asset contract in trainer-facing export:
  - the sim/synth training-corpus path now harvests `robot_asset_contract_receipt_v1`
  - backend-selector and branch-planner rows now keep asset-contract refs, readiness score, and missing-asset context
- This matters because:
  - hardware-target readiness is now part of downstream training/export truth, not only runtime truth
  - Phase 1 is therefore moving closer to the desired state where replay/training consume WM receipts directly rather than reconstructing backend readiness from loose metadata

- Made the robot-asset contract operative inside backend materialization:
  - backend shadow/work-order paths now emit backend-local sidecars for:
    - robot asset contract
    - calibration contract
    - IO contract
  - Isaac/Holosoma shadow receipts now include those refs directly
  - runtime evidence and calibration scoring now react to missing backend-side asset obligations
- This is a useful Phase-1 shift:
  - the stack is no longer only saying “assets missing”
  - it is emitting the exact backend-local contract artifacts the future concrete runtime should satisfy
  - the honest remainder is therefore more concretely an execution/assets problem

- Added a canonical robot-asset contract inside Phase 1:
  - `src/world_model/sim_synth_physics/asset_contracts.py` now compiles a typed asset/calibration/action/observation contract from backend binding plus embodiment context
  - the WM state now carries `RobotAssetContractState`
  - the runtime now emits `robot_asset_contract_receipt_v1`
- Why this matters:
  - the loop can now say exactly which Unitree-target robot description, mapping, calibration, and IO contracts are missing
  - that moves another backend/hardware seam out of vague metadata and into a canonical receipt path
  - the remaining blocker is increasingly actual assets/calibration/runtime integration, not missing schema/plumbing

- Pushed the Phase-1 adaptation/calibration seam past mostly plan-time scoring:
  - `src/world_model/sim_synth_physics/runtime_evidence.py` now summarizes backend shadow execution, render materialization, and branch-outcome evidence
  - `src/world_model/sim_synth_physics/calibration.py` now uses that evidence when computing readiness/quality and stores the evidence summary directly in receipt metadata
  - `src/world_model/sim_synth_physics/runtime.py` now rebuilds adaptation/calibration receipts after materialization and outcome compilation so the emitted receipts reflect loop evidence rather than only pre-execution intent
- This is the right direction for Phase 1:
  - keep typed adaptation/calibration receipts
  - make them react to real WM loop evidence as soon as that evidence exists
  - keep the honest remaining gaps focused on concrete runtimes/assets/GPU/data, not missing feedback plumbing

- Pushed Phase 1 further into real backend/provider materialization instead of stopping at typed planning state:
  - `src/world_model/sim_synth_physics/shadow_execution.py` now emits Holosoma shadow work-order receipts/artifacts alongside Isaac shadow execution receipts
  - `src/world_model/sim_synth_physics/render_materialization.py` now materializes LSD scene-config artifacts and NAG/GGDS work orders under the WM runtime
  - `src/world_model/sim_synth_physics/runtime.py` now propagates those materialization results into `RenderProviderReceipt`, `SimulationOutcomeReceipt`, the training-feedback manifest, and loop summaries
- The important doctrine point is unchanged:
  - this is not permission to pretend Holosoma or GGDS/NAG are fully concrete runtimes yet
  - it does mean the default posture is now WM-owned materialization with explicit preconditions and artifact refs, not provider selection that terminates at compile time
  - the honest remaining blockers are concrete runtime/data/GPU/asset gaps, which is where Phase 1 should end up

- Pushed the Phase-1 backend-execution seam past a literal Isaac stub:
  - `src/envs/physics/isaac_backend.py` now exposes an explicit shadow-contract backend with deterministic reset/step/media/summary/state behavior rather than only raising `NotImplementedError`
  - this is not being treated as "real Isaac runtime" or as permission to declare the Unitree path done
  - the honest remaining gap is now narrower:
    - concrete Isaac Sim / Isaac Gym / Unitree asset execution
    - robot descriptions, latency/calibration sidecars, and hardware-grade runtime evidence
- The WM now uses that seam directly:
  - `src/world_model/sim_synth_physics/runtime.py` emits a `backend_shadow_execution_receipt_v1` whenever an Isaac-target planning window can at least exercise the explicit shadow contract
  - `src/world_model/sim_synth_physics/training_corpus.py` now harvests that receipt so backend-selector training can tell the difference between planning-only bundles and shadow-runtime bundles
  - this is the right current posture for Phase 1:
    - no literal stub default
    - real shadow execution/materialization where possible
    - explicit remaining gap where concrete Isaac/Unitree execution is still absent

- Tightened the doctrine against "stood up in name only" WMs:
  - the multi-WM plan now has an explicit mechanics-first WM readiness rule
  - the doctrine now says neuralization is part of scalable mechanics, not a deprioritized separate layer
  - each WM should be judged on a bounded closed loop, not on whether it can emit logs or canonical-looking state objects
  - future-state completion now means all relevant downstream consumers for the hardware-ready loop are wired and changed by that WM, not merely one downstream demo consumer
- The practical execution rule is:
  - keep the scalable mechanics substrate ahead of non-load-bearing learned claims
  - learned control, prediction, adaptation, routing, and refinement should become part of the real subsystem as soon as they can honestly carry loop load
  - if a phase is still missing executors, adapters, safety/precondition gates, replay/training exports, or live downstream consumers, that phase is still structurally incomplete
  - the maturity ladder should be read as:
    - `schema_only`
    - `logging_only`
    - `shadow_runtime`
    - `bounded_runtime_authority`
    - `benchmark_gated_primary`
    - `production_recurrent`

- Added an explicit Phase 8 after the WM and meta-node phases:
  - the multi-WM plan now treats the post-Phase-7 period as a production-loop runtime / weekly GPU operations phase rather than leaving that operating model implicit
  - the roadmap now spells out the intended order inside that phase:
    - external dataset aggregation and loop runs
    - receipt / corpus export
    - training
    - fine-tuning where the receipts justify it
    - benchmarking and promotion / redeployment
    - only then latency / inference / cost hardening
- The practical doctrine change is:
  - backlog exhaustion is now part of the architecture plan, not just an operational preference
  - the stack should keep burning down uncalled runs, trainers, fine-tuning lanes, and provider bring-up items until the honest blockers are mostly compute, data density, calibration, benchmark evidence, latency, and deployment cost

- Tightened the long-range Unitree program target:
  - July 2027 remains the purchase / initial integration milestone
  - September 30, 2027 is now the explicit stronger target for sustainably autonomous G1 operation
- This materially changes how the roadmap should be read:
  - it is not enough for the stack to be merely "purchase-ready" by mid-2027
  - by September 2027 the control loop should be able to run repeatedly on G1, emit replay/telemetry/calibration/safety/governance receipts, export those artifacts into recurring training cycles, and improve without recurring architecture rewrites
- The practical prioritization effect is:
  - lower-WM plumbing and deployment-enabler phases are even more central now
  - Phase 4A/4B/4C/4E-style work is no longer just predeployment hygiene; it is part of the path to sustainable autonomy
  - if the program reaches summer 2027 with missing on-robot replay capture, missing degraded-mode truth, missing recovery/teleop tracing, or missing recurring export/retrain plumbing, then it is behind even if the purchase/integration moment itself succeeds

- Tightened the post-September 2026 execution model:
  - the intended cadence is now weekly A100-backed work, not occasional broad training sweeps
  - the unit of progress is a WM sub-module, not a vague whole-WM training claim
  - each weekly tranche should move in order from loop runs to receipt/corpus export to training to fine-tuning
  - sim/synth/physics is still the first weekly WM focus, followed by perception/grounding, then embodiment/actuation, then economic-WM consolidation, then local meta-node neuralization and later meta-node superposition/control
- This is the important operational constraint:
  - after September 1, 2026, if the weekly A100 budget is spent mostly on fine-tuning before loop/provider truth is real, the program will look busy while still being structurally behind
  - the weekly ladder is meant to prevent that failure mode

- Added a dated program assumption for the pre-G1 push:
  - serious multi-WM training is assumed to start on September 1, 2026
  - the current multi-WM architecture should have its plumbing laid by August 31, 2026
  - July 2027 is treated as a Unitree G1 pre-purchase readiness window, not as a promise of deployment readiness
- This changes the interpretation of roadmap completeness:
  - through August 31, 2026, the job is to finish canonical lower-WM and economic-WM plumbing
  - after September 1, 2026, the emphasis should shift toward training runs, provider bring-up, calibration, benchmarks, and Unitree-specific integration
  - if missing work after September 2026 is still mainly contract/plumbing debt, the roadmap is behind
  - if missing work is mainly data, GPUs, calibration, assets, and benchmark evidence, the roadmap is on the right shape

- Clarified the V-JEPA 2 posture across lower WMs and backlogs:
  - it should not live only as a future sim/synth/physics note
  - it now belongs explicitly in both the sim/synth/physics WM and the later perception/grounding WM
  - the preferred bring-up path is upstream `facebookresearch/vjepa2` integration when that is faster and more honest than local reimplementation
  - the provider must still sit behind typed runtime/provider contracts, provider-truth receipts, calibration traces, and benchmark gates
- The backlog split is now explicit rather than implied:
  - `scripts/FOUNDATION_MODEL_BRINGUP_BACKLOG.json` tracks V-JEPA 2 provider bring-up for both WM lanes
  - `scripts/LOOP_RUN_BACKLOG.json` tracks the corresponding manual bring-up runs
  - `scripts/TRAINING_MIGRATION_BACKLOG.json` and `docs/economic_world_model/full_stack_training_backlog.md` now track the two fine-tuning lanes separately
  - this preserves the intended sequencing: bring up lower-WM runtime/provider plumbing first, then let fine-tuning follow once the real state/receipt surfaces exist

- Replaced the actively used Stage-1 diffusion stub posture with an explicit runtime/provider contract:
  - added `src/diffusion/video_diffusion_runtime.py`
  - the new runtime wraps the governed proposal planner but resolves honest provider truth for the materialization backend
  - `backend_policy` now follows `auto|real|disabled|stub`
  - `real` is strict real-or-unavailable
  - `auto` now means "go ahead with governed planning, but record `heuristic_fallback` and `plan_only` if no local/cached diffusers checkpoint exists"
  - every `DiffusionProposal` now carries:
    - `diffusion_provider_truth`
    - `diffusion_backend_selected`
    - `diffusion_backend_policy`
    - `diffusion_model_ref`
    - `diffusion_materialization_mode`
- Wired that contract through the active Stage-1 and orchestration paths:
  - `scripts/run_stage1_pipeline.py` now instantiates `VideoDiffusionRuntime` instead of using `VideoDiffusionStub` directly
  - admission logs, datapack metrics, agent profile metadata, and pipeline stats now preserve diffusion backend/materialization truth
  - `src/orchestrator/diffusion_requests.py` now instantiates the runtime by default for prompt-driven proposal generation
- Tightened the GGDS/LDM seam so the repo stops silently normalizing dummy-LDM behavior:
  - `scripts/train_ggds_on_lsd_vector_scenes.py` now accepts `--backend-policy auto|real|disabled|stub`
  - `auto` no longer silently returns `create_dummy_ldm()`
  - smoke/scaffolding still works, but only when the caller explicitly requests `stub`
  - the training summary now includes `ldm_provider_truth`
- This is the doctrine shift in practical terms:
  - stubs are still allowed as smoke aids
  - they are no longer acceptable as silent defaults for live WM/runtime paths once a real provider contract can exist
  - the desired failure mode is now:
    - real backend if locally available
    - otherwise honest `unavailable` / `heuristic_fallback`
    - not "pretend the stub is the backend"
- Added `scripts/FOUNDATION_MODEL_BRINGUP_BACKLOG.json` to keep the remaining model-shaped gaps explicit with OSS targets and prerequisites:
  - governed video diffusion
  - GGDS/LDM renderer stack
  - vision backbone stub replacement
  - semantic VLA placeholder replacement
  - Isaac/Unitree backend execution
- Updated `scripts/TRAINING_MIGRATION_BACKLOG.json` and `scripts/LOOP_RUN_BACKLOG.json` so the existing training/run backlogs also describe these surfaces honestly instead of implying the migrated wrappers alone are enough.

- Clarified the relationship between the old heuristic purge and new multi-WM work:
  - the earlier heuristic/advisory sweep should be treated as the first repo-wide high-impact pass
  - it should not be treated as proof that every later WM module has already had its deterministic priors reviewed
  - each WM boundary should explicitly rerun that audit for its own canonical state, runtime seams, replay/training exports, benchmark surfaces, and adapters
- The architecture docs now say this directly so the rule is operational:
  - a WM is not structurally "done" just because an earlier global purge happened
  - the remaining heuristics inside that WM must be explicitly classified as fallback priors or lifted into learned/runtime-package seams
  - the stopping condition should be data/GPU/assets/calibration/benchmark limits, not inherited confidence from the old sweep

- Clarified the ontology architecture into two repo-native layers instead of one vague future "ontology" bucket:
  - operational / module-level ontology is the in-stack typed operational state substrate and digital-twin layer
  - WM-transport ontology is the later typed interoperability contract between adjacent WMs
- The docs now make the complement explicit:
  - ontology defines the semantic / governance contract
  - the isomorphic tensor / transport bridge is the compiled differentiable realization that should respect that contract
  - neither should be used as an excuse to invent a giant symbolic mother-WM
- The RL/training split is now explicit:
  - operational ontology training should improve module encoding/decoding fidelity, temporal/event prediction, uncertainty calibration, provenance quality, and governance satisfaction
  - WM-transport ontology training should improve WM-to-ontology-to-WM translation quality, preserve topology/causal structure/actionability, and decompose bridge-only vs downstream-only vs joint effects
  - both are trained from completed-loop/postmortem quality, governance satisfaction, counterfactual improvement, and downstream yield, while frozen core reward math remains untouched for now
- Current-state honesty is now written down in the architecture docs:
  - the repo mostly has operational ontology substrate/plumbing today
  - it does not yet have a fully neural ontology layer
  - it does not yet have a full WM-transport ontology implementation
  - lower WMs still come first, then economic-WM consolidation, then ontology-mediated transport

## 2026-03-26

- Landed the real runtime-package path for the first two learned sim/synth helper seams:
  - backend selector training/runtime package
  - branch planner training/runtime package
  - live wrappers in `semantic_simulation`, `diffusion_requests`, and `coverage_loop` now accept those packages instead of forcing direct in-memory helper objects only
- The package lane is now more production-shaped than "checkpoint on disk":
  - runtime packages resolve package-relative checkpoints
  - trainer/export outputs now stamp explicit target hardware and subsystem posture metadata
  - the emitted package artifact is intended to be the real runtime contract, not a sidecar note for later cleanup
- The next concrete step toward a complete subsystem also landed:
  - backend-selector and branch-planner trainers can now ingest canonical WM runtime receipt bundles instead of requiring only hand-shaped row datasets
  - `src/world_model/sim_synth_physics/training_corpus.py` is the shared contract for that projection
  - the trainer scripts now emit compiled dataset artifacts so receipt-derived rows become a stable training corpus artifact, not a transient loader path
- Important implementation detail:
  - the branch-planner feature contract expected `heuristic_generation_mode`
  - the runtime compiler was not passing it
  - this is now fixed, so trained branch-planner packages see the same core context fields at inference time that they saw at training time
- This is also where the complete-subsystem rule starts to matter for the new lower WM:
  - backend-selector and branch-planner outputs are no longer just "advice around the WM"
  - they are bounded-authority helper seams inside the WM’s canonical agenda / physics / branch planning path
  - the main remaining gaps are increasingly honest external blockers rather than missing neural/runtime/package scaffolding
  - for the Unitree G1/R1-oriented target, those blockers are:
    - Unitree-class sim adapters and robot-description assets
    - whole-body branch and replay corpora
    - calibration receipts for contact, balance, and latency-sensitive behavior
    - larger GPU-backed helper training/eval
    - humanoid benchmark receipts strong enough to justify `required` posture

- Landed the next `sim_synth_physics` consolidation step:
  - WM-owned simulation jobs now carry `inferential_learnability_contract`
  - WM-owned synthetic branch plans now carry their own inferential learnability contracts rather than borrowing admission heuristics only
  - `Gen2SimAdmissionState` now uses bounded inferential thresholds and summary density in addition to benchmark/grounding truth
  - `DiffusionConditioningState` now carries admissible vs blocked branch splits and inferential summaries so render budgeting and diffusion ordering are WM-owned instead of implicit orchestration behavior
- This is the important posture shift in practice:
  - epiplexity/inferential learnability is no longer just a replay/training concern
  - the new lower WM now uses it directly inside simulation and synth agenda selection
  - blocked synthetic branches remain visible, but they no longer drive diffusion priority or render budget as if they were equally admissible
- Updated `docs/economic_world_model/multi_wm_architecture_plan.md` to make the downstream-WM rule explicit:
  - once a WM affects replay, admission, simulation, diffusion, or training selection, it should carry epiplexity-based inferential learnability as canonical typed metadata rather than leaving it as an external overlay

- Landed the first code tranche from the advisory-purge plan:
  - added `src/economics/inferential_contract.py` as the shared canonical learnability/admission contract
  - replay datasets now attach `inferential_learnability_contract` per episode and summarize learnability-class density at the manifest level
  - `src/orchestrator/shadow_advisory.py` now emits inferential learnability summaries plus canonical inferential work orders instead of leaving epiplexity/inferential evidence as overlay-only context
  - `src/orchestrator/adaptation_budgeting.py` now builds inferential execution work orders through one shared helper instead of reassembling the same contract ad hoc
  - `src/training/training_manifest.py` and `src/training/regal_training_runner.py` now persist inferential learnability and inferential work-order summaries in canonical runtime manifests
  - the shadow/offline training entrypoints now register explicit `inferential_learnability_summary` and `inferential_work_orders` artifacts beside the existing advisory and scorer artifacts
- This is the concrete posture shift for epiplexity/inferential evidence:
  - still additive and outside frozen reward math
  - no longer just an overlay for replay and training consumers
  - replay descriptors and promotion reporting can now consume a replay-native inferential contract instead of recomputing only from scattered summary fields

- Started the first real implementation tranche under the new multi-WM plan instead of leaving it as architecture-only:
  - added `src/world_model/sim_synth_physics/` as the initial canonical package for the next lower WM
  - the package now defines typed state for:
    - agenda ownership
    - backend/fidelity context
    - diffusion conditioning
    - synthetic branch plans
    - gen2sim admission
    - outcome/calibration receipts
  - `src/orchestrator/semantic_simulation.py` now delegates agenda compilation into that WM runtime/compiler boundary instead of owning the agenda contract itself
- The new package is deliberately neuralization-ready from the start rather than heuristic-first:
  - agenda ranking still uses the existing bounded learned gap-ranker path
  - backend/fidelity selection now has its own benchmark-gated helper seam
  - synthetic branch planning now has its own benchmark-gated helper seam
  - both seams use explicit helper status / promotion-stage traces while heuristics remain only the prior/fallback path
- Diffusion ownership is now also inside that WM boundary rather than in a separate orchestration helper:
  - added `src/world_model/sim_synth_physics/diffusion_contracts.py` as the WM-owned gap-driven diffusion-plan layer
  - `src/orchestrator/diffusion_requests.py` now adapts `SimSynthPhysicsWorldState` / `DiffusionConditioningState` into downstream prompt specs instead of recomputing ranked gaps itself
  - `src/orchestrator/coverage_loop.py` now compiles one shared `SimSynthPhysicsWorldState` and derives both agenda and diffusion prompts from it, so those two planning surfaces stop drifting
- The multi-WM plan and roadmap now state the rule explicitly for later phases:
  - future WMs and enabler phases should launch with bounded learned seams and typed `disabled|auto|required` runtime posture from their first tranche
  - do not create fresh heuristic-only control islands and plan to purge them later
- This is the correct current posture for the new WM boundary:
  - canonical state ownership has moved out of a scattered orchestrator helper
  - downstream callers still keep compatibility via the legacy agenda view
  - the learned seams are present now even though promoted helper packages for backend/branch policy are still future work
  - agenda and diffusion conditioning now share one canonical WM-owned planning state
  - the next consolidation step should be package-loading runtime shims and receipt wiring for backend/branch helper packages, not another standalone planner

- Added `docs/economic_world_model/advisory_purge_wiring_plan.md` as the advisory-doctrine counterpart to the earlier heuristic sweep:
  - it separates surfaces that should remain advisory from surfaces that should become:
    - canonical metadata
    - preconditions
    - work orders
    - bounded authority
    - benchmark-gated successors
  - it ranks the current advisory gaps and identifies epiplexity / inferential signal-yield as the top remaining tranche because those signals already shape replay weighting and adaptation budgeting while still behaving too much like overlays
  - it also records the architectural posture shift that frozen Phase B math should stay the rollback anchor now without being treated as sacred forever once benchmark-gated successor evidence eventually exists
- Updated `docs/epiplexity.md` to reflect the same posture:
  - epiplexity remains bounded and non-reward-changing today
  - but its portable summaries are now explicitly framed as a future canonical learnability class rather than a permanently advisory metric
- Updated `docs/economic_world_model/multi_wm_architecture_plan.md` and `docs/economic_world_model/roadmap.md` so the new multi-WM topology and roadmap rules explicitly distinguish:
  - external advisory providers
  - preview/report layers
  - internal typed WM-to-WM receipts that should graduate out of the advisory bucket once they shape runtime or training

- Added `docs/economic_world_model/humanoid_target_readiness.md` as the concrete G1/R1-facing readiness artifact:
  - it converts the high-level hardware-target discussion into an explicit checklist
  - it includes a benchmark matrix for balance, locomotion-manipulation, recovery, dexterity, degraded sensing/comms, and related humanoid-specific promotion classes
  - it records a repo-grounded gap map against current files such as:
    - `src/embodiment/core.py`
    - `src/embodiment/registry.py`
    - `src/envs/physics/isaac_backend.py`
    - `src/ingestion/x_humanoid_adapter.py`
    - `src/vision/scene_ir_tracker/io/scene_tracks_runner.py`
    - `src/motor_backend/workcell_env_backend.py`
  - it also makes Unitree sim-env integration, robot asset/calibration handling, companion-compute middleware, and teleop/recovery fallback explicit future requirements rather than implied concerns
- `docs/economic_world_model/multi_wm_architecture_plan.md` now links to that readiness artifact from the humanoid-target and Phase 3.5 sections so the planning stack has one concrete checklist for later embodied-target work.

- Added `docs/economic_world_model/multi_wm_architecture_plan.md` to make the next architecture expansion explicit instead of leaving it as conversational intent:
  - the stack should grow into:
    - perception / grounding WM
    - embodiment / actuation WM
    - sim / synth / physics WM
    - economic WM over those lower WMs
    - meta-node superposition / control WM above the economic WM
  - this only makes sense if each lower WM becomes a canonical typed state owner rather than another advisory sidecar lane
- The plan resolves the sequencing question explicitly:
  - the economic WM should keep hardening now, but it should not be treated as the final neuralized top layer before lower WMs emit canonical state
  - the next WM to build should be sim / synth / physics, because that is where the production flywheel still remains most distributed across orchestrator, diffusion, branch generation, and physics backends
  - the future cross-WM “isomorphic tensor” idea should be implemented as typed middleware between adjacent WMs, not as an early giant shared latent
  - an overarching meta-node superposition WM should also stay deferred until the existing local meta-node objects themselves are neuralized and robust
- The plan now makes the local meta-node prerequisite explicit:
  - current local meta-nodes are real routing/control objects
  - but they are still mostly named bounded-control surfaces with learned layers around them
  - they are not yet fully learned geometric/cybernetic objects in their own right
  - so a dedicated local meta-node neuralization / robustness phase should land before any higher-order meta-node mother-WM
- The plan now also makes the hardware target implication explicit:
  - if the actual target is Unitree G1/R1-class readiness, several current assumptions are only provisional
  - current workcell/tabletop envs should be treated as partial manipulation domains, not full humanoid-readiness proxies
  - a future Unitree G1/R1 sim-env integration lane should be treated as a named roadmap item, not left implicit under generic “humanoid envs”
  - a dedicated later phase should audit which lower-WM and submodule models are large enough for 21+ DoF whole-body control and which can remain compact because they operate over typed summaries
  - the future embodiment/perception/sim contracts will need richer:
    - proprioception
    - IMU
    - force/torque
    - latency / control-rate
    - whole-body kinematic state
    - spatial state
  - the plan now also names additional humanoid-target requirements that were previously only implicit:
    - companion-compute / communication middleware
    - operator / teleop / recovery fallback contracts
    - robot asset and calibration management
    - a humanoid-specific benchmark taxonomy for promotion
  - this is why the plan now includes a separate humanoid target capacity and environment-refit phase before claiming serious hardware-readiness
- Phase 1 in that plan is intentionally concrete and near-implementation-shaped:
  - proposed additive package: `src/world_model/sim_synth_physics/`
  - proposed ownership:
    - simulation agenda
    - diffusion conditioning
    - synthetic branch plans
    - backend / fidelity selection
    - gen2sim admission context
    - outcome / calibration receipts
  - proposed absorption targets:
    - `src/orchestrator/semantic_simulation.py`
    - `src/orchestrator/diffusion_requests.py`
    - `src/evidence/gen2sim_validity.py`
    - `scripts/collect_local_synthetic_branches.py`
    - `src/envs/physics/*`
    - `src/motor_backend/holosoma_backend.py`
- Later phases are now explicitly named with preconditions rather than remaining hand-wavy:
  - perception / grounding WM
  - embodiment / actuation WM
  - real-time servo vs governance control-loop separation
  - sensor-fusion shim
  - physical safety layer
  - spatial state / SLAM integration
  - economic-WM consolidation over lower canonical WMs
  - cross-WM transport bridges
  - the later meta-node superposition / control WM

- `PipelineManager` no longer strands stage activation above the learned-helper contract:
  - `src/orchestrator/pipeline_stage_policy.py` now defines the explicit feature contract over:
    - iteration history
    - per-stage outcome history
    - progress trends
    - execution-precondition truth
    - shell activation readiness
    - objective preset and config-flag state
  - it also defines the explicit heuristic priors for stage-priority distribution and next-iteration config flags, so the bootstrap logic is auditable instead of buried in one method
  - `src/orchestrator/pipeline_stage_policy_training.py` now provides the real bounded helper training path over `PipelineManager` state receipts
  - `scripts/train_pipeline_stage_policy.py` now emits:
    - pipeline-stage dataset summary
    - model config
    - execution-precondition artifact
    - training summary
    - `pipeline_stage_policy_package.json`
    - runtime manifest / checkpoint registry outputs under `RegalTrainingRunner`
- The pipeline-shell runtime seam is now honest and neurally scalable:
  - `src/orchestrator/pipeline_stage_policy_runtime.py` loads helper packages with `disabled|auto|required` semantics
  - `src/orchestrator/pipeline_manager.py` now preserves:
    - `policy_source`
    - `promotion_stage`
    - `stage_policy_trace`
    while letting bounded helpers materially affect stage priority ordering and config suggestions
  - shell activation itself remains hard-gated by execution readiness, so the helper cannot fake activation on an unready substrate
- The main remaining control-plane heuristic core is now queue/curriculum weighting:
  - `SemanticOrchestratorV2` is wired
  - `PipelineManager` is wired
  - `queue_selection.py` / `episode_sampling.py` are the next live seam where learned signals exist but the lane still largely computes its own bounded action from heuristics

- `SemanticOrchestratorV2` no longer strands the shell-policy layer outside the learned-helper contract:
  - `src/orchestrator/orchestrator_shell_policy.py` now defines the explicit feature contract over semantic snapshot truth, recap/execution readiness, segmentation/OOD pressure, semantic-WM meta state, meta expected deltas, and preset availability
  - `src/orchestrator/orchestrator_shell_policy_training.py` now provides the real bounded helper training path over snapshot-plus-advisory receipts
  - `scripts/train_orchestrator_shell_policy.py` now emits:
    - orchestrator shell dataset summary
    - model config
    - execution-precondition artifact
    - training summary
    - `orchestrator_shell_policy_package.json`
    - runtime manifest / checkpoint registry outputs under `RegalTrainingRunner`
- The runtime shell seam is now honest and neurally scalable:
  - `src/orchestrator/orchestrator_shell_policy_runtime.py` loads helper packages with `disabled|auto|required` semantics
  - `src/orchestrator/semantic_orchestrator_v2.py` now blends learned preset/strategy/safety/activation outputs against the explicit heuristic prior instead of claiming a learned shell with no package/runtime contract
  - runtime receipts now preserve:
    - `policy_source`
    - `promotion_stage`
    - `helper_trace`
    so later economic-WM/meta-node-WM conditioning can learn on why the shell chose what it chose
- The remaining higher-order orchestration gap is now specifically `PipelineManager`, not a vague “orchestrator” bucket:
  - `SemanticOrchestratorV2` is wired
  - `PipelineManager` still assembles stage activation and pipeline-shell choices mostly deterministically
  - that should be the next control-plane neuralization tranche before broader queue/curriculum policy work

- Gen2sim validity/value admission now shares the same honest helper contract as agenda ranking and fill routing:
  - `scripts/collect_local_synthetic_branches.py` now emits `*_gen2sim_validity.json`, so each local synthetic branch carries an explicit admission assessment instead of only trust/gap proxy metadata
  - `src/training/synthetic_branch_corpus.py` now loads those assessments, summarizes admission/promotion state, and changes synth-share caps plus branch-priority scaling when gen2sim validity is missing or weak
  - `scripts/train_offline_with_local_synth.py` now records gen2sim admission artifacts in the canonical runtime output instead of treating synth validity as an internal weighting detail
- The learned substrate for gen2sim admission is now real:
  - `src/evidence/gen2sim_validity.py` now exposes:
    - an explicit feature contract
    - bounded helper-trace blending
    - conditioning-feature recording for later meta-choice learning
  - `src/evidence/gen2sim_validity_training.py` now provides the actual helper model/training path
  - `src/evidence/gen2sim_validity_runtime.py` now loads helper packages with `disabled|auto|required` semantics
  - `scripts/train_gen2sim_validity.py` now emits:
    - gen2sim dataset summary
    - model config
    - execution-precondition artifact
    - training summary
    - `gen2sim_validity_package.json`
    - runtime manifest / checkpoint registry outputs under `RegalTrainingRunner`
- `src/regal/data_value.py` now consumes that contract correctly:
  - generated/synthetic datapack admission no longer depends on a loose `gen2sim_validity_score` scalar
  - the explicit assessment remains the source-of-truth prior
  - the learned helper can only apply bounded deltas unless it is benchmark-gated promoted
  - helper status and conditioning traces are preserved in the returned report so later economic-WM/meta-node-WM trainers can learn on “why this branch/datapack was admitted”
- This is the right current posture for gen2sim neuralization:
  - the learned substrate exists and affects runtime honestly
  - benchmark-unready packages remain bounded `shadow_candidate` helpers
  - promotion still requires empirical receipt density, so local distillation does not get mistaken for production-grade gen2sim truth

- Fill-path routing now shares the same bounded helper contract as the rest of the coverage loop:
  - `scripts/train_fill_path_policy.py` now emits:
    - fill-path dataset summary
    - model config
    - execution-precondition artifact
    - training summary
    - `fill_path_policy_package.json`
    - runtime manifest / checkpoint registry outputs under `RegalTrainingRunner`
  - `src/world_model/fill_path_runtime.py` now resolves that package into a runtime helper with explicit benchmark-gate status
  - `src/orchestrator/fill_path_routing.py` now blends:
    - heuristic fill-method priors
    - learned fill-path probabilities
    through a bounded helper weight
  - `src/orchestrator/coverage_loop.py` now consumes that routing helper and records:
    - `routing_policy`
    - helper promotion stage
    - heuristic vs learned score traces
    on each emitted fill decision
- This is the right current posture for fill-path neuralization:
  - governance/readiness hard gates remain explicit
  - the learned helper is real and runtime-active
  - benchmark-unready packages remain bounded `shadow_candidate` helpers
  - fill-outcome records now preserve routing traces so later economic-WM/orchestrator layers can learn on “why this path was chosen,” not just the winning method
  - the next consistency upgrade is later gen2sim validity/value admission, not more hidden fill-path heuristics

- Sim/gen2sim agenda ranking now uses the learned gap-ranker substrate instead of leaving it stranded:
  - `scripts/train_gap_ranker.py` now emits:
    - gap-ranker dataset summary
    - model config
    - execution-precondition artifact
    - training summary
    - `gap_ranker_package.json`
    - runtime manifest / checkpoint registry outputs under `RegalTrainingRunner`
  - `src/world_model/gap_ranker_runtime.py` now resolves that package into a runtime helper with explicit benchmark-gate status
  - `src/orchestrator/gap_agenda_ranking.py` now blends heuristic and learned ranking with bounded helper weights:
    - `shadow_candidate` packages are bounded
    - `promoted` packages get stronger influence
    - `required` mode fails if no benchmark-gated package is present
- `src/orchestrator/semantic_simulation.py` and `src/orchestrator/diffusion_requests.py` now share that same ranking contract:
  - simulation agenda and diffusion-gap prompts no longer diverge on “why this gap was chosen”
  - each agenda item/prompt now records:
    - `ranking_policy`
    - helper promotion stage
    - score trace (heuristic vs learned contribution)
  - `src/orchestrator/coverage_loop.py` now threads the helper into both agenda and diffusion compilation rather than reserving learned ranking only for later stages
- This is the right current posture for sim-agenda neuralization:
  - the heuristic gap score remains the explicit prior
  - the learned model is real and runtime-active
  - promotion semantics are explicit instead of implied by checkpoint existence
  - the next consistency upgrade is to align fill-path routing and later gen2sim validity scoring to the same helper contract

- Meta-transformer runtime/training are now actually connected:
  - `scripts/train_meta_transformer_synthetic.py` now emits `meta_transformer_package.json` beside the checkpoint/model-config/precondition artifacts
  - `src/orchestrator/meta_transformer_runtime.py` loads that package and reconstructs `MetaTransformerNet` for CPU inference
  - `src/orchestrator/meta_transformer.py` now accepts `helper_package_path` plus `helper_mode=disabled|auto|required`
  - `src/policies/meta_advisor.py` now threads the same package path/mode into the live policy facade
- The learned package now covers the real planning seam instead of only latent embeddings:
  - `src/orchestrator/meta_transformer_planning.py` defines a shared planning-context vector over:
    - semantic-WM features
    - econ signals
    - datapack signals
    - selector meta-choice receipts
  - `src/orchestrator/semantic_runtime_learning.py` now exports that same context plus explicit:
    - `objective_preset`
    - `chosen_backend`
    - `energy_profile_weights`
    - `data_mix_weights`
    - `expected_deltas`
    into `MetaTransformerSample`
  - `src/orchestrator/meta_transformer_training.py` now trains planning heads directly on the real `MetaTransformerNet` substrate, so the helper learns the same meta-choice surface that runtime previously derived purely by hand
  - `src/orchestrator/meta_transformer_runtime.py` now decodes those heads and records planning traces, and `src/orchestrator/meta_transformer.py` now applies them with bounded shadow/promoted blending plus an explicit `planning_application` receipt
- Promotion/readiness for the meta-transformer is now materially stricter and sequential:
  - sample count alone no longer promotes the lane
  - benchmark readiness now also requires enough:
    - `bounded_ready_count`
    - `semantic_grounded_count`
    - `route_success_count`
    - `authority_success_count`
  - `auto` uses benchmark-unready packages only as `shadow_candidate` helpers
  - `required` refuses those packages outright
- This is the right current posture for meta neuralization:
  - the trained architecture, dataset substrate, and runtime helper are now real
  - the heuristic `MetaTransformer` outputs remain the explicit prior, but they are now a bounded prior rather than the only planner:
    - learned objective/backend candidates can override when confidence and promotion stage justify it
    - learned energy/data-mix/expected-delta heads now blend against the prior even in `shadow_candidate`
    - `orchestration_plan` remains a deterministic bounded projection downstream of those chosen planning fields
  - the learned package now materially influences:
    - authority selection
    - shared policy state
    - diffusion conditioning
    - ontology-token predictions
    - objective preset / backend / energy-profile / data-mix / expected-delta choice
  - the next layer above this is not another fake package; it is later economic-WM and meta-node-WM conditioning over the same helper contract

- Orchestration transformer training/eval now use one honest instruction/runtime contract:
  - `src/orchestrator/training_dataset.py` now persists `instruction_text` in saved samples, reconstructs typed samples from JSON, and derives deterministic instruction tokens from runtime/context metadata
  - `src/orchestrator/semantic_runtime_learning.py` now preserves runtime instruction / execution-mode metadata when exporting orchestration samples from the semantic runtime corpus
  - `scripts/eval_orchestration_transformer.py` now uses the same deterministic tokenization path instead of random placeholder tokens
- `scripts/train_orchestration_transformer.py` is no longer just “wrapped but still fake inside”:
  - prefers `orchestration_runtime_dataset.json` exports from the semantic runtime corpus
  - falls back to synthetic/mixed corpora only explicitly and keeps those benchmark-unready
  - emits:
    - orchestration dataset + summary
    - model config
    - execution-precondition artifact
    - subset metrics
    - training summary
    - training job result
    - runtime manifest / checkpoint registry outputs under `RegalTrainingRunner`
  - now trains the bounded sequence contract honestly as `tool_prediction_contract: bounded_tool_sequence_v2`
  - uses an explicit PAD/stop label in target tool sequences instead of overloading the first tool id as padding
  - tracks:
    - active-token accuracy
    - first-tool accuracy
    - full-sequence accuracy
    - stop-token accuracy
- This is the right current neuralization posture for orchestration:
  - instruction, semantic, and selector meta-choice conditioning are now stable and trainable
  - runtime-corpus receipts are the preferred supervision source
  - synthetic fallback remains available for bring-up, but no longer masquerades as benchmark-ready
  - the next upgrade is no longer sequence supervision; it is the higher-order objective/backend/data-mix planner above the sequence head

- Semantic datapack-selection is now a real learned-helper lane rather than a runtime-only seam:
  - `src/orchestrator/semantic_policy.py` now defines:
    - `DatapackSelectionContext`
    - context-conditioned helper adjustment caps
    - richer `scorer_trace.context_trace` receipts
    - a bounded neural helper package contract with:
      - `model_kind`
      - ordered feature inputs
      - one-hidden-layer MLP weights/biases
      - interpretable local contributor traces from the active network path
  - the old literal feature coefficients remain only as the bootstrap prior
  - the new learned/helper part is still bounded, but it is no longer a flat bump on top of the prior
- `src/orchestrator/semantic_simulation.py` now enforces an honest sequential promotion path for selector helpers:
  - `disabled`
  - `auto` with explicit `shadow_candidate` vs `promoted` helper stages
  - `required`, which now fails unless the scorer package is benchmark-gated ready
  - this matters because “package exists” is no longer treated as equivalent to “package is production-ready”
- Added `src/orchestrator/datapack_selection_training.py` plus `scripts/train_datapack_selection_scorers.py`:
  - the trainer consumes `selection_summary` run-log receipts
  - builds selected-outcome and positive-pairwise supervision examples
  - learns both:
    - a bounded feature-MLP reranker over `DatapackSelectionFeatures`
    - context weights over `DatapackSelectionContext` for adjustment-cap conditioning
  - emits:
    - `datapack_selection_training_dataset.json`
    - `datapack_selection_dataset_summary.json`
    - `datapack_selection_model_config.json`
    - `datapack_selection_execution_preconditions.json`
    - `datapack_selection_scorer_package.json`
    - `datapack_selection_training_summary.json`
    - `training_job_result.json`
    - canonical runtime manifest/checkpoint-registry outputs under `RegalTrainingRunner`
- Selector receipts now persist across the real runtime bridge:
  - `src/orchestrator/semantic_simulation.py` writes per-episode `*_selection_summary_v1.json`
  - `src/replay/ingest.py` carries `selection_summary` into replay episodes and provenance refs
  - `src/orchestrator/semantic_runtime_learning.py` preserves those receipts into runtime rows and orchestration samples
- The observation/conditioning path now reacts to selector meta-choice:
  - `src/orchestrator/semantic_transformer_bridge.py` encodes selection-feedback features
  - `src/orchestrator/orchestration_transformer.py` appends those features into `_encode_ctx(...)`
  - orchestration activation plans and metadata now keep:
    - `selection_policy`
    - selected datapack ids
    - the distilled `selection_meta_choice` summary
- This is the right current posture for selector neuralization:
  - bootstrap prior stays explicit and auditable
  - helper reranking and helper strength are now learnable from receipts
  - benchmark-unready packages still influence runtime only through a bounded shadow-stage clamp
  - future conditioning can move upward into the economic WM and then the later meta-node WM without changing the current runtime contract again
  - full counterfactual datapack-choice supervision is still a later density problem, not something this pass should fake

- Observation/conditioning now reacts to semantic-runtime truth instead of merely carrying it:
  - `src/semantic/runtime_backbone.py` now derives a compact `semantic_runtime_truth` block from the semantic world model:
    - scene-track backend truth
    - teacher-runtime truth
    - vision-backbone truth
    - benchmark signals
    - execution-precondition summary
  - those summaries are now written into `SemanticSnapshot.metadata`
- `src/observation/adapter.py` now threads those runtime-truth fields into the condition-builder inputs even when they originated from semantic snapshots rather than raw datapack metadata.
- `src/observation/condition_vector_builder.py` now uses those signals materially:
  - benchmark-unready grounding
  - failed execution preconditions
  - blocked/mixed semantic fusion
  now contribute bounded OOD and recovery signals instead of remaining sidecar-style facts with no effect on the condition vector
- This closes the remaining runtime honesty gap for these modules:
  - runtime truth no longer disappears between semantic snapshot construction and policy/diffusion conditioning
  - the next missing pieces are training/export lanes that learn from these richer receipts

- Rollout-labeler semantics now survive into the real datapack contract instead of dying as local sidecars:
  - `src/motor_backend/datapacks.py` now treats `quality_score`, `novelty_score`, and arbitrary `metadata` as first-class datapack-config fields
  - `src/ontology/datapack_registry.py` now upserts those richer configs into ontology records instead of skipping existing datapacks and preserving stale truth
  - this matters because reruns of labeled datapacks now refresh readiness/provenance instead of pinning old fallback states forever
- `src/vla/rollout_labeler.py` now aggregates a bounded but materially useful labeled-datapack truth contract:
  - teacher-runtime backend truth
  - vision-backbone truth
  - SceneTracks grounding truth
  - artifact refs for teacher contract / action / trace / VLA semantic evidence
  - explicit execution preconditions for promotion-ready labeled datapacks
  - bounded quality / novelty proxy scores with the proxy kind recorded in metadata
  - the teacher outputs remain external/advisory; what changed is that downstream routing can now see the truth about them instead of only seeing tags
- `src/orchestrator/semantic_simulation.py` now performs a second enrichment pass after semantic fusion:
  - labeled datapacks pick up fusion/world-model/snapshot/advisory artifact refs
  - execution preconditions are recomputed with semantic-fusion readiness included
  - this closes the prior gap where the vision lane was truthful at sidecar emission time but thin again by the time selection/replay/readiness looked at the datapack object
- The remaining vision-side runtime audit is now narrower:
  - rollout-labeler and labeled-datapack truth are wired
  - the next sweep should focus on observation-adapter/runtime-backbone bridges and on the training/export lane that eventually learns from these richer labeled-datapack receipts

- Shadow-advisory scorer fallback is now externally visible instead of only behaviorally visible:
  - `src/orchestrator/shadow_advisory.py` now emits:
    - `semantic_runtime_scorer_preconditions`
    - `semantic_runtime_scorer_work_orders`
  - when a scorer package is missing, the advisory payload now carries a blocking work order pointing at `scripts/train_semantic_runtime_scorers.py` rather than quietly degrading to heuristic scoring with no artifact-level trace
- The main consumers now preserve that state into runtime artifacts:
  - `scripts/run_shadow_advisory_pass.py`
  - `scripts/train_shadow_replay_policy.py`
  - `scripts/train_shadow_offline_rl.py`
  - `scripts/train_shadow_pricing_models.py`
  - `scripts/train_sac_with_ontology_logging.py`
  now all write/register scorer-precondition and scorer-work-order artifacts
- This matters because “scored shadow advisory” vs “heuristic fallback advisory” is now a real observable distinction in manifests and backlog scans, not a detail hidden inside `build_shadow_advisory_output(...)`.

- `scripts/train_meta_transformer_synthetic.py` now uses the actual meta-transformer training substrate instead of random placeholder tensors:
  - it accepts:
    - `meta_transformer_runtime_dataset.json` exports
    - saved dataset JSON inputs
    - explicit synthetic generation only when requested
  - it now instantiates the real `MetaTransformerNet`, uses the existing batching/loss/eval helpers from `src/orchestrator/meta_transformer_training.py`, and emits:
    - dataset summary
    - model config
    - execution preconditions
    - training history
    - training summary
    - training job result
    - canonical runtime manifest and checkpoint registry when run under `RegalTrainingRunner`
  - synthetic fallback samples now carry the same planning-target contract as runtime-export samples instead of only authority/token labels, so the lightweight path is materially closer to the heavyweight trainer
- This is the correct posture for the meta-transformer lane:
  - the script is no longer fake
  - synthetic data is no longer the implicit truth source
  - benchmark readiness still depends on runtime-corpus density, not on the mere existence of a migrated script

- Semantic datapack/scenario selection now has an explicit promotion path instead of a forever-hardcoded score:
  - `src/orchestrator/semantic_policy.py` now defines:
    - `DatapackSelectionFeatures`
    - `DatapackSelectionScorerPackage`
    - a bounded learned-helper adjustment on top of the explicit prior score
  - the old hand-written coefficients are still present, but only as the bootstrap prior over a first-class feature contract
  - this is important because those terms are now:
    - auditable as runtime features
    - reusable as training targets/features
    - replaceable later by a learned helper without changing the rest of the runtime contract
- `src/orchestrator/semantic_simulation.py` now makes helper promotion state explicit:
  - `selection_scorer_mode="disabled"` for bootstrap bring-up
  - `selection_scorer_mode="auto"` for shadow/helper-if-present rollout
  - `selection_scorer_mode="required"` when the learned helper must exist or the run should fail honestly
  - the resulting helper state is written into `selection_summary`, so downstream replay/runtime analysis can distinguish:
    - learned-helper-backed selection
    - heuristic fallback
    - explicitly disabled helper use
- The remaining honest gap in this lane is no longer runtime wiring; it is training:
  - we still need the corpus/export/training job that produces the datapack-selection scorer package
  - until that exists, `required` is a supported promotion target, not the default operating mode

- `scripts/train_vla_recap_offline.py` is no longer a lightweight side lane outside the runtime contract:
  - the direct `train_offline(...)` entrypoint is preserved for existing smoke/inference consumers
  - but the trainer now always emits:
    - recap dataset summary
    - recap feature-config artifact
    - recap training preconditions / benchmark gate
    - recap training summary
    - training-job result
    - latest and best checkpoints under the same schema that `src/vla/recap_inference.py` expects
  - the CLI path now wraps the same logic under `RegalTrainingRunner`, so RECAP head training produces:
    - `training_runtime_manifest.json`
    - `checkpoint_registry.json`
    - canonical runtime artifact registration
    - recap-row trajectory audits labeled honestly as `recap_row_projection`
- This is the right production posture for the RECAP lane:
  - keep tiny local recap corpora runnable for regression tests
  - keep the recap checkpoint contract stable for inference
  - do not let small local recap corpora silently masquerade as promotion-ready training
  - keep the benchmark gate explicit until a materially real recap corpus exists

- Semantic datapack/scenario selection is no longer just a tag-overlap sort:
  - `src/orchestrator/semantic_policy.py` now exposes:
    - `DatapackSelectionDecision`
    - `rank_datapacks_for_intent(...)`
    - `summarize_datapack_selection(...)`
  - the ranking stays bounded and deterministic for now, but it now uses materially more real loop state:
    - ARH-adjusted historical scenario outcomes per datapack
    - candidate quality and novelty
    - benchmark/readiness support from datapack metadata when present
    - explicit gap-fill pressure for tags that the current scenario history has not covered
  - this means datapack choice in `semantic_simulation` is now affected by actual historical and readiness evidence, not just set overlap and an ARH subtraction term.
- `src/orchestrator/semantic_simulation.py` now wires that ranking into the live selection path instead of throwing it away:
  - ontology datapacks and local-YAML fallback datapacks are both ranked on the same contract and merged by score
  - missing-gap fallback no longer means “replace the ontology choice wholesale”; it now means “surface additional gap-fill candidates in the same ranked pool”
  - the chosen subset is emitted as `selection_summary` on `SemanticSimulationResult`
  - the same summary is persisted into the semantic run log so later replay/training analysis can see what the runtime actually chose and why
- This is the correct current production posture for semantic policy selection:
  - deterministic and auditable now
  - materially shaped by runtime/economic/readiness evidence now
  - explicitly left on the later neuralization path once the replay corpus is dense enough to support learned routing honestly

- SceneTracks truth semantics now have one shared consumer-facing normalization layer instead of a replay/bootstrap-only fix:
  - `src/evidence/scene_tracks_truth.py` now does two separate jobs:
    - `resolve_scene_tracks_backend(...)` reads nested runner metadata like `runner.run_config.backend_selected`, passthrough flags, stub flags, and adapter status before falling back to any looser artifact hints
    - `scene_tracks_truth_from_metadata(...)` then re-applies the canonical rule that only `real` keeps `scene_tracks_non_stub`, `semantic_grounding_non_heuristic`, and training eligibility
  - this matters because some newer Stage-1 and synthetic-corpus paths were still treating any sidecar presence or passthrough backend as strong enough evidence, which reopened the same honesty gap we had already closed in replay/bootstrap.
- The remaining SceneTracks truth leaks are now closed in the higher-value downstream consumers:
  - `scripts/run_stage1_pipeline.py`
    - no longer infers `scene_tracks_backend=real` from `scene_tracks_path` or `scene_tracks_v1` alone
    - filters explicit `future_training_signals` so stale `scene_tracks_non_stub=true` cannot override known passthrough/stub metadata
  - `scripts/collect_local_synthetic_branches.py`
    - now writes normalized scene-track truth into branch metadata instead of marking passthrough corpora as non-stub
  - `src/training/synthetic_branch_corpus.py`
    - now re-normalizes branch source metadata before benchmark-gating the corpus, so local synth cannot look benchmark-ready on passthrough grounding
  - `src/orchestrator/semantic_runtime_scorers.py`
    - live runtime scoring no longer treats `scene_tracks_backend in {real,passthrough,auto}` as non-stub support
  - `src/orchestrator/semantic_fusion_runner.py`
    - degraded-fusion artifacts now preserve the same normalized truth instead of copying any stale `scene_tracks_non_stub` bit through failure paths
- The SAM3D host requirement is now a first-class runtime artifact:
  - `src/evidence/grounded_data_host.py` collects:
    - GPU/CUDA availability
    - OpenCV availability
    - SAM3D repo presence
    - SAM3D checkpoint presence
    - one derived `real_sam3d_grounding_ready` boolean
  - `src/vision/scene_ir_tracker/io/scene_tracks_runner.py` now emits:
    - `grounded_data_host_capabilities`
    - `grounded_data_host_preconditions`
  - `src/orchestrator/loop_run_backlog.py` now consumes that same host-capability scan, so the loop backlog and the live SceneTracks runner agree on what “real grounded data is actually possible on this host” means
  - this directly encodes the operational truth from the recent bootstrap pass: local passthrough runs are useful for plumbing, but they do not substitute for a Linux/NVIDIA + SAM3D host when grounded data is the contract.

- Local synthetic branch corpora now have an explicit runtime/training contract instead of a loose NPZ-only shape:
  - `src/training/synthetic_branch_corpus.py` loads the branch corpus plus optional metadata/gap-label sidecars, summarizes source provenance and gap labels, emits an execution-precondition artifact, emits a benchmark-gate artifact, and compiles a bounded training policy from that truth.
  - this policy is intentionally conservative:
    - missing corpus metadata clamps synthetic share harder
    - missing semantic-gap labels clamps synthetic share harder
    - heuristic or benchmark-ineligible grounding clamps synthetic share harder
  - the goal is not to ban synthetic training, but to stop under-described corpora from exerting full-weight influence on offline training.
- `scripts/collect_local_synthetic_branches.py` now writes the missing provenance fields the trainer needs:
  - `scene_tracks_backend`
  - `teacher_runtime_backend_selected`
  - `vision_backbone_selected`
  - `semantic_grounding_mode`
  - `semantic_memory_grounded`
  - plus `future_training_signals` / `future_training_artifacts`
  - this makes the branch corpus explicit about whether it came from real grounded seeds or a more heuristic/fallback lane.
- `scripts/train_offline_with_local_synth.py` is no longer an isolated lightweight script:
  - it now loads the explicit branch corpus contract
  - it rescales branch influence by branch value / coverage-gap metadata / corpus readiness
  - it caps effective synth share with the policy compiled from corpus truth
  - it emits:
    - `synthetic_branch_summary.json`
    - `synthetic_branch_metrics.json`
    - `synthetic_branch_execution_preconditions.json`
    - `synthetic_branch_benchmark_gate.json`
    - canonical actor checkpoints under the run output dir
    - `training_job_result.json`
    - full `RegalTrainingRunner` runtime artifacts when not run with `--skip-regal-runner`
  - the script still uses proxy delta metrics for `w_econ` inputs because the corpus does not yet carry true realized branch outcome deltas; this is now explicit and bounded instead of implicit.
- Added `docs/economic_world_model/heuristic_advisory_sidecar_inventory.md`:
  - it ranks the remaining heuristic/advisory/sidecar surfaces by loop impact and distortion
  - it marks the local-synth lane as the tranche landed in this pass
  - it documents the remaining top runtime gaps honestly
- Added `scripts/RUNTIME_WIRING_BACKLOG.json`:
  - separates non-training runtime gaps from `scripts/TRAINING_MIGRATION_BACKLOG.json`
  - now keeps a `completed` section for runtime modules already wired
  - currently keeps the remaining active backlog focused on:
    - SceneTracks passthrough truthiness
    - shadow advisory sampling learning
    - semantic policy datapack selection
- Updated `scripts/TRAINING_MIGRATION_BACKLOG.json`:
  - moved `train_offline_with_local_synth.py` from pending to migrated
  - left `train_vla_recap_offline.py` and `train_meta_transformer_synthetic.py` pending because they still lack the same runtime/receipt/parity treatment
- Stage-1 semantic/diffusion routing now carries structured runtime truth instead of collapsing back to flat prompt heuristics:
  - `src/orchestrator/diffusion_requests.py` now attaches:
    - governed hypotheses
    - routing context
    - benchmark signals
    - coverage-gap / trust / economic priority fields
  - this applies to both datapack-guidance prompts and coverage-gap prompts, so the diffusion contract now preserves actual routing intent from the orchestrator instead of reducing everything to `semantic_tags + objective_preset`.
- `src/diffusion/real_video_diffusion_stub.py` is still a stub renderer, but it is no longer a keyword-first selector:
  - when governed hypotheses are present, it reranks them with routing context before rendering proposals
  - when no governed hypotheses are present, fallback proposals are explicit and bounded
  - heuristic or benchmark-unready routing now clamps proposal confidence/novelty instead of letting fallback proposals look equally trustworthy
- `scripts/run_stage1_pipeline.py` now treats benchmark readiness as a real admission boundary:
  - emits a per-video `benchmark_gate_v1.json` sidecar
  - propagates benchmark and routing fields into admission rows, datapack metrics, and regal annotations
  - downgrades non-benchmark-ready proposals into `shadow_stage1_datapack` work orders
  - caps shadow datapacks to tier 0 with lower effective trust so downstream replay/training sampling sees the difference
  - keeps a benchmark-ready path for manifests that carry real SceneTracks and real vision-backbone declarations
- The remaining Stage-1 limitation is honest and bounded:
  - seed-tag extraction is still deterministic bootstrap logic
  - the governed hypotheses, routing scores, and benchmark gate now dominate admission and proposal shaping, so the heuristic seed no longer silently controls the whole diffusion lane
- Replay/bootstrap SceneTracks truth now has a shared normalization helper:
  - `src/evidence/scene_tracks_truth.py` defines the canonical rule:
    - `real` can count as non-stub / non-heuristic / training-eligible
    - `passthrough`, `stub`, and `auto` remain explicit fallback lanes
  - explicit old flags can still rescue unknown older bundles when backend identity is missing, but they can no longer override known passthrough/stub/auto backends into looking real.
- `src/replay/ingest.py` now uses that normalization before writing replay episode metadata:
  - passthrough rollouts can still carry semantic density and grounded-world-model summaries
  - but replay metadata no longer sets:
    - `scene_tracks_non_stub=true`
    - `semantic_grounding_ready=true`
    - `semantic_grounding_non_heuristic=true`
    solely because passthrough or auto was present
- `scripts/bootstrap_semantic_workcell_loop.py` now writes the same truth semantics into the workcell bootstrap lane:
  - `metadata.json` preserves the selected backend and training-eligibility status
  - only real SceneTracks remain eligible to set `scene_tracks_non_stub` / `semantic_grounding_non_heuristic`
  - passthrough bootstrap runs remain useful for corpus/debugging, but they stop overstating readiness in downstream replay/runtime consumers
- The bootstrap workcell lane now emits canonical runtime traces instead of stopping at semantic sidecars:
  - each episode now writes:
    - `*_runtime_packet_v1.json`
    - `*_event_spine_v1.json`
    - `*_decision_ledger_v1.json`
  - `metadata.json` now carries:
    - `runtime_packet_id`
    - `event_refs`
    - `decision_refs`
    - `grounded_data_ready`
    - `grounded_data_mode`
    - explicit SAM3D/GPU requirements
  - this fixes the prior gap where bootstrap replay/runtime corpora could be operationally stable yet still guarantee `bounded_ready_count=0` because the canonical trace substrate never existed.
- The bootstrap lane now separates grounded-data truth from benchmark readiness:
  - real SAM3D plus a GPU-backed non-fallback backend is now the explicit requirement for `grounded_data_ready`
  - that truth is recorded in trace-sidecar decisions and summary metadata instead of remaining doc-only
  - benchmark eligibility still remains false in this lane by default because trace completeness and real grounding are not the same as full teacher/vision/runtime promotion readiness
- Workcell coverage mapping is now aligned to the actual graph contract:
  - `src/world_model/coverage_evidence_harvester.py` now canonicalizes env ids such as `workcell_env` into the registered `workcell` inventory
  - harvested task→skill and skill→primitive evidence now uses the same canonical skill ids as the graph (`hrl:*`, `workcell:*`, etc.) instead of the old mismatched `skill:*` envelope
  - `src/hrl/skill_graph.py` now includes a built-in `peg_in_hole` workcell skill chain, and `src/orchestrator/coverage_loop.py` enables it automatically for workcell envs
  - this fixes the earlier failure mode where many workcell rows could still leave the coverage loop effectively blind because the evidence keys did not line up with the graph topology
- Shadow advisory replay selection now has a learned runtime-scorer seam instead of staying purely rule-weighted:
  - `src/orchestrator/shadow_advisory.py` now auto-loads a semantic runtime scorer package when one is colocated with the replay dataset (or when one is passed explicitly) and scores replay-native semantic runtime rows before building per-episode advisory output
  - `src/rl/econ_regal_sampling.py` now accepts bounded learned inputs for:
    - route success probability
    - authority confidence
    - counterfactual value
    - predicted regret
    - authority-switch recommendation
  - those learned signals can now change sampling priority and queue tags without bypassing the existing bounded queue caps or reward math
- Queue metadata now preserves the learned evidence instead of dropping it:
  - `src/orchestrator/queue_selection.py` carries `semantic_runtime_score` through `build_live_queue_selection(...)`
  - the live queue lane therefore keeps the learned-vs-fallback provenance visible when training-time bounded reweighting happens
  - if no scorer package is present, the queue path stays functional on the explicit heuristic fallback

- Added a canonical full-stack training backlog document at `docs/economic_world_model/full_stack_training_backlog.md`:
  - it records the current workspace truth that replay, coverage, and semantic-runtime corpora are still tiny
  - it ranks the real learned lanes by production importance and dependency instead of by script existence
  - it explicitly recommends `workcell_data_refresh` as the first recurring remote bundle before heavier scorer/refiner/shadow jobs
  - it now also distinguishes local passthrough refreshes from the canonical recurring lane: benchmark-grade workcell refresh/replay assumes real SAM3D on a Linux/NVIDIA A100-class host
- Added Runpod training-bundle scaffolding under `scripts/runpod/`:
  - `FULL_STACK_TRAINING_BUNDLES.json` is the checked-in bundle source of truth
  - `assess_full_stack_training.py` scans the workspace and emits honest readiness/blocker state
  - `execute_training_bundle.py` runs the selected bundle locally or inside a pod and writes receipts
  - `launch_training_bundle.py` wraps `runpodctl create pod` so recurring runs can be launched from one checked-in entrypoint
- The recurring workcell bundle is now stricter:
  - `workcell_data_refresh` assumes `--backend-policy real` rather than `auto`
  - readiness for that bundle now expects local SAM3D repos plus checkpoints instead of treating passthrough as enough
  - semantic runtime training now tracks real-grounded replay counts separately from raw replay counts so passthrough corpus growth does not masquerade as canonical promotion-ready data
- The Runpod path is deliberately conservative:
  - it refuses to auto-run bundles whose data thresholds are not met unless `--force` is passed
  - it keeps frozen-baseline lanes out of the recurring automation path
  - it switches pod teardown behavior based on storage mode, preferring `remove` when a network volume is attached and `stop` otherwise
- The backlog doc also captures the adjacent architectural recommendation that the next large WM tranche should likely be sim/synth/physics, but only after the repo wires high-impact heuristic/advisory/sidecar surfaces into the actual runtime/training/reward loops.
- Training-run receipt ingestion now carries real backend-truth fields into replay precondition evaluation:
  - `src/replay/receipt_ingest.py` now extracts per-episode signal overrides from observed online receipt rows (`scene_tracks_non_stub`, `scene_tracks_backend`, teacher backend selectors, and grounding flags) and merges them into enriched replay episode metadata before `build_replay_execution_preconditions(...)`.
  - this directly fixes the prior gap where `scene_tracks_non_stub` and `teacher_runtime_real` stayed false in training-run readiness summaries despite receipt-side evidence.
- The readiness probe now validates the intended state:
  - `scripts/economic_world_model/run_receipt_readiness_probe.py` emits explicit real backend metadata in the observed receipt row and writes refreshed summary artifacts.
  - latest probe output shows target predicates all true:
    - `signal_bool::budget_settlement_live`: 1
    - `signal_bool::scene_tracks_non_stub`: 1
    - `signal_bool::teacher_runtime_real`: 1
- Regression coverage tightened:
  - `tests/test_training_run_receipt_ingest.py` now asserts both backend-truth predicates are satisfied in the execution-precondition summary when real backend fields are present in receipt rows.
- Nightly audit freshness logic is now robust to newest-first progress-log ordering:
  - `_progress_latest_date()` in `scripts/economic_world_model/nightly_audit.py` now selects `max(...)` over all `## YYYY-MM-DD` headings instead of taking the final heading in file order.
  - `tests/test_economic_world_model_nightly_audit.py::test_progress_latest_date_uses_most_recent_heading` now uses reverse-chronological headings and verifies the newest date is selected.
- Added a repeatable real-run readiness probe at `scripts/economic_world_model/run_receipt_readiness_probe.py`:
  - executes a minimal real `RegalTrainingRunner` run that emits a training runtime manifest, promotion ledger, budget settlement report, and receipt artifacts
  - immediately rehydrates that run through `build_training_run_receipt_label_bundle(...)`
  - writes a stable summary to `artifacts/economic_world_model/readiness_probe/readiness_probe_summary.json` and `.md`
- Current probe findings for the targeted predicates:
  - `signal_bool::budget_settlement_live`: true (count=1)
  - `signal_bool::scene_tracks_non_stub`: false (count=0)
  - `signal_bool::teacher_runtime_real`: false (count=0)
  - this confirms the receipt/readiness plumbing is live while grounding/teacher-real evidence still does not arrive in this minimal training lane
- Nightly execution still remained in audit-only mode for this pass:
  - `python3 scripts/economic_world_model/nightly_audit.py --output-json artifacts/economic_world_model/nightly_audit_summary.json --output-markdown artifacts/economic_world_model/nightly_audit_summary.md` refreshed both summary artifacts.
  - selector result stayed `next_task.id=audit_only` with `execute_now=false`, so no new safe additive scaffold was missing according to the current roadmap/doc/code scan.
- Verification for this pass:
  - `./scripts/agent/verify.sh`
  - `PYTHONPYCACHEPREFIX=/tmp/pycache python3 -m compileall src scripts/economic_world_model -q`
  - `python3 -m pytest -q tests/test_runtime_packets.py tests/embodiment/test_registry.py tests/test_objective_runtime_builder.py tests/test_constraint_set.py tests/test_pricing_sentinel.py tests/test_value_ledger.py tests/test_economic_world_model_nightly_audit.py`
  - `python3 scripts/economic_world_model/run_receipt_readiness_probe.py --output-root artifacts/economic_world_model/readiness_probe --seed 17`
- This keeps the lane honest: no synthetic "work landed" claim when the selector says no safe additive gap is currently missing.
- Recommended next move remains evidence accumulation, not another schema rewrite:
  - feed real SceneTracks backend status and teacher-runtime real/unavailable status from non-stub ingestion paths into the training-run replay/receipt summaries so the two remaining false predicates can be measured as true in practice.

## 2026-03-24

- OpenVLA and MetaDINO are now explicit backend-policy surfaces rather than soft-fail scaffolds:
  - `src/vla/openvla_controller.py` now accepts `backend_policy` and `vision_backbone_policy` with `auto|real|disabled|stub`
  - `auto` means “try real local model, otherwise unavailable”; it no longer silently degrades to a fake action/embedding path
  - `stub` is still allowed, but only by explicit caller choice
  - this is the correct long-term shape because benchmark/promotion code can now reject fake capability without special-casing individual scripts
- The teacher lane now preserves backend truth rather than flattening it away:
  - `src/vla/teacher_runtime.py` now records controller backend status in the contract and action-envelope metadata
  - `src/vla/rollout_labeler.py` now defaults to `disabled` unless OpenVLA is explicitly enabled or a policy override is provided, and it treats import/load/inference failure as `unavailable`, not “good enough stub labels”
  - that makes later replay/import/benchmark logic materially more trustworthy
- Benchmark gating now exists as a first-class evidence primitive:
  - `src/evidence/benchmark_gating.py` compiles metadata into explicit signals for:
    - real SceneTracks grounding
    - real teacher runtime
    - real vision backbone
    - non-heuristic semantic grounding
  - the gate blocks passthrough/stub/heuristic cases, so “ready for smoke/dev” is no longer confused with “ready for benchmark/promotion”
- Replay/readiness now carries those stronger signals:
  - `src/replay/preconditions.py`, `src/replay/importers.py`, and `src/replay/ingest.py` now surface:
    - `teacher_runtime_real`
    - `vision_backbone_real`
    - `semantic_grounding_non_heuristic`
    - `benchmark_eligible`
  - that keeps benchmark/promotion gating aligned with the same precondition summary machinery already used for shell activation
- There is now a loop-run backlog separate from the training backlog:
  - `scripts/LOOP_RUN_BACKLOG.json` is the operational queue for semantic/control loop runs
  - each entry records:
    - the exact command
    - host/model/data preconditions
    - internal vs external data requirements
    - whether it is safe for auto-trigger
    - optional benchmark-gate requirements
  - `src/orchestrator/loop_run_backlog.py` evaluates those preconditions against the host
  - `scripts/scan_loop_run_backlog.py` turns that into a JSON summary and can execute ready `auto_trigger=true` runs
- Current intended division of labor:
  - training backlog = heavyweight learning jobs and migration tracking
  - loop-run backlog = concrete runtime/self-improvement exercises and data-collection/coverage-validation runs
  - benchmark gate = strict promotion barrier that rejects stub/passthrough/heuristic success masquerading as real semantic capability

- The semantic runtime scorer layer now exists in both lightweight and heavyweight forms:
  - `src/orchestrator/semantic_runtime_scorers.py` trains lightweight local models over the runtime corpus for:
    - meta-route success
    - orchestration-route success
    - authority calibration
    - counterfactual value
    - route regret
  - it also scores live semantic-world-model plus transformer packets in shadow mode so the runtime can emit reranking/calibration evidence before any learned controller authority is granted
- The heavyweight training plumbing is no longer implicit:
  - `src/orchestrator/semantic_runtime_scorer_training.py` builds an explicit scorer-training dataset over the same semantic runtime row schema
  - when torch is available, it trains a multitask scorer net and saves a checkpoint instead of forcing the repo to stop at deterministic local models
  - `scripts/train_semantic_runtime_scorers.py` is now the canonical scorer-training/export entrypoint for this lane
- The live runtime boundary now exposes both transformer callouts plus shared scorer feedback:
  - `run_pipeline_step_with_causal_order(...)` can now emit:
    - `meta_transformer_execution`
    - `orchestration_transformer_execution`
    - `semantic_runtime_scoring`
  - this makes the broader loop concrete:
    - semantic WM feeds both transformer lanes
    - both lanes produce bounded execution packets
    - scorer outputs turn those packets back into shadow route/calibration/regret evidence
- The training backlog is now aligned with the implementation:
  - `scripts/TRAINING_MIGRATION_BACKLOG.json` now includes `train_semantic_runtime_scorers.py`
  - that keeps the heavyweight learned scorer path explicit inside the repo's future-training envelope rather than leaving it as an unwritten follow-up

- The pre-training semantic learning loop now exists as code, not just as a future-training idea:
  - `src/orchestrator/semantic_runtime_learning.py` builds canonical replay-backed rows that join:
    - semantic-world-model summaries
    - OpenVLA / teacher semantic evidence
    - DINO / SceneTracks / Map-First proxy evidence
    - fusion summaries
    - transformer targets
    - outcome labels
    - shadow counterfactuals / regret targets
  - this means the repo can start learning bounded semantic-routing behavior from replay before any full controller-training run is turned on
- The export path is explicit:
  - `scripts/export_semantic_runtime_learning_corpus.py` loads a canonical replay dataset and writes:
    - `semantic_runtime_learning_rows.jsonl`
    - `semantic_runtime_learning_summary.json`
    - `meta_transformer_runtime_dataset.json`
    - `orchestration_runtime_dataset.json`
- The semantic runtime corpus now makes the broader loop concrete:
  - OpenVLA/teacher evidence feeds the semantic world model through teacher traces and VLA semantic sidecars
  - DINO/SceneTracks/Map-First proxy evidence feeds the same world model through grounding summaries and confidence metadata
  - semantic-world-model state then feeds both transformer shells
  - replay outcomes and shadow counterfactuals feed back into future training and inferential labels
- This is the correct intermediate production posture:
  - keep execution bounded
  - accumulate runtime data
  - train scorers/calibrators first
  - only then promote to learned reranking/control over the same packet shape
- The broader architecture is documented in `docs/economic_world_model/semantic_runtime_learning_loop.md`, which separates:
  - the learning pipeline (corpus -> runtime datasets -> future training)
  - the inferential pipeline (counterfactuals -> regret -> reranking/calibration evidence)

- Transformer promotion is now explicit rather than deferred:
  - `src/orchestrator/semantic_transformer_bridge.py` is the shared semantic-world-model featurization layer for transformer shells
  - it normalizes a real `SemanticWorldModelState` into bounded numeric features, top object/meta-node summaries, semantic tokens, tool biases, and deterministic routing heuristics
- `MetaTransformer` is no longer only a feature-fusion helper:
  - it now exposes `propose_plan(...)` as a real pipeline interface
  - that method consumes econ signals, datapack signals, and semantic-world-model state
  - it emits semantic-aware objective/backend/energy/data-mix choices, bounded orchestration steps, execution preconditions, and a typed execution work order
  - `MetaTransformerOutputs` now carries `execution_mode`, `bounded_actions`, `execution_preconditions`, `execution_work_order`, and execution metadata so the packet can later be promoted without another schema break
- The pipeline callout is no longer a silent stub:
  - `run_pipeline_step_with_causal_order(...)` now passes semantic-world-model inputs into the meta-transformer path
  - the returned `run_specs` now include a `meta_transformer_execution` packet beside the existing soft suggestions
- `OrchestratorContext` and `OrchestratorResult` now carry execution-oriented semantic transformer state:
  - context can carry `semantic_world_model`, `semantic_snapshot`, and `semantic_metadata`
  - results can carry `execution_mode`, `activation_plan`, `activation_work_order`, and metadata
- The orchestration transformer now consumes the semantic world model materially:
  - `_encode_ctx(...)` appends a semantic-WM feature vector to the existing econ/customer/profile context
  - `propose_orchestrated_plan(...)` now derives objective preset, energy profile, data mix, backend preference, tool biases, execution preconditions, and a bounded activation/work-order packet from semantic/econ/data state
  - the transformer is still additive and bounded; it does not bypass readiness checks or the frozen Phase B baseline
- The long-term direction is now documented in `docs/economic_world_model/semantic_authority_promotion.md`:
  - advisory packet remains the starting point
  - preconditioned execution is the next layer
  - bounded meta-node authority is the correct promotion target
  - learned transformer control should arrive later over the same packet shape, not via a separate rewrite

- SceneTracks backend selection is now precondition-driven instead of defaulting to stubs:
  - `run_scene_tracks(...)` defaults to `backend_policy="auto"`
  - `auto` first attempts a real local SAM3D tracker with `allow_fallbacks=False`, so installed packages + local checkpoints immediately activate on-device inference without extra caller wiring
  - if real SAM3D is not locally ready and segmentation masks are available, `auto` falls back to `zero_inference_passthrough`
  - stubs are now opt-in via explicit `backend_policy="stub"` or `use_stub_adapters=True`; they are no longer the silent default
- Backend choice is now first-class runtime metadata:
  - runner metadata records `backend_policy`, `backend_selected`, and `real_backend_failure`
  - auto-selected real runs and passthrough runs can therefore be distinguished by replay/semantic consumers without re-deriving the decision from logs
- Caller surface:
  - `scripts/run_scene_tracks.py` now exposes `--backend-policy auto|real|passthrough|stub`
  - `src/ingestion/x_humanoid_adapter.py` now exposes `scene_tracks_backend_policy`
- SceneTracks now has an explicit no-spend backend alongside real/stub SAM3D:
  - `SceneIRTrackerConfig.zero_inference_passthrough` can be enabled directly or via `run_scene_tracks(...)` / `scripts/run_scene_tracks.py --zero-inference-passthrough`
  - when enabled, `SceneIRTracker` skips SAM3D adapter loading and inverse-rendering refinement, then reconstructs objects/bodies deterministically from segmentation masks plus depth/camera geometry
  - when depth is absent, the tracker still emits coarse world-frame entities from mask centroid plus camera rays so the rest of the semantic/orchestrator stack continues to run
- The zero-inference backend is intentionally honest rather than sovereign:
  - `adapter_status()` reports `zero_inference_passthrough` / `overall_mode=passthrough`
  - execution-precondition metadata records `scene_ir_backend_passthrough=True`
  - training eligibility still requires real SAM3D backends, so passthrough keeps plumbing live without pretending the perception stack is fully grounded
- The same opt-in backend is now available to ingestion callers:
  - `src/ingestion/x_humanoid_adapter.py` exposes `scene_tracks_zero_inference_passthrough`
  - CLI callers can use `scripts/run_scene_tracks.py --zero-inference-passthrough`
- Upstream SceneTracks production is no longer geometry-only:
  - `DatapackFramesContract` now carries `class_labels` and `object_refs` alongside instance masks.
  - `src/vision/scene_ir_tracker/io/datapack_frame_reader.py` derives those labels from datapack `scene_spec` / metadata when available, then publishes a `semantic_context` bundle with object catalog, segmentation-label metadata, semantic tags, and label-source provenance.
  - `src/vision/scene_ir_tracker/io/scene_tracks_runner.py` now passes those labels into `SceneIRTracker.process_episode(...)` instead of reconstructing unlabeled tracks whenever segmentation IDs are present.
- `SceneTracks_v1` artifacts are now semantically self-describing enough for downstream grounding:
  - added per-track arrays for label source, category, label confidence, motion score, hint object ID, semantic tags, and affordances
  - merged those fields into `scene_tracks_v1/summary_json` plus a dedicated `scene_tracks_v1/semantic_summary_json`
  - mirrored semantic density / grounding readiness / semantic tags back into datapack metadata and execution-precondition signals so replay/admission code can reason about semantic quality without reopening the NPZ
- The tracking stack now preserves exact upstream identity instead of forcing downstream heuristics:
  - `SceneEntity3D` now carries `source_instance_id`, `source_object_id`, and `label_source`
  - `SceneIRTracker` accepts per-frame `object_refs`
  - `KalmanTrackManager` now preserves those refs onto stable tracks, and `SceneTracks_v1` persists `track_source_instance_id` / `track_source_object_id`
- Sensor bundles can now carry explicit segmentation/object identity joins:
  - `SensorBundleData` and `write_sensor_bundle(...)` now support `segmentation_label_map` plus `scene_object_catalog`
  - `src/motor_backend/workcell_env_backend.py` populates those fields for workcell sensor bundles using explicit object IDs and seg IDs
  - `datapack_frame_reader.py` can now consume that explicit bundle metadata instead of relying only on scene-spec ordering
- Teacher-runtime payloads now carry structured semantics instead of only action floats:
  - `infer_teacher_semantics(...)` in `src/evidence/teacher_trace.py` extracts object refs, affordances, and risk hints from instructions plus teacher action payloads
  - `TeacherTrace.from_vla_action(...)` stores those hints in both trace metadata and per-step metadata
  - `TeacherActionEnvelope` in `src/vla/teacher_runtime.py` now persists `semantic_tags`, `object_refs`, `affordance_hints`, and `risk_hints`, and `to_vla_payload()` forwards them into the governed VLA evidence sidecar
- Stub/fallback dependence is now explicit:
  - the SAM3D adapter wrappers report backend modes (`real`, `wrapper_fallback`, `import_failure_stub`, `load_failure_stub`, `stub_requested`)
  - requesting real models with `allow_fallbacks=False` now fails explicitly instead of silently degrading to stubs
  - `run_scene_tracks(...)` records adapter status and requires stub/fallback-free backends before marking the run training-ready
- The rollout-labeler path no longer drops the richer teacher semantics:
  - `src/vla/rollout_labeler.py` now copies `object_refs`, `affordance_hints`, and `risk_hints` from `TeacherActionEnvelope` into the teacher trace and governed VLA semantic-evidence sidecars
- `SemanticWorldModelBuilder` now reads those richer producer-side fields:
  - track label confidence now contributes to semantic confidence
  - track label/source/category metadata and explicit track-source object IDs are preserved in grounded object metadata
  - producer-side affordance/risk tags and teacher object refs now influence grounded object affordances/risk tags instead of being dropped
- This pass keeps all changes additive. It raises semantic quality and semantic density upstream, but it does not change frozen Phase B dynamics/reward math and it does not grant planner/controller sovereignty to the semantic layer.

## 2026-03-07

- Introduced `RuntimePacket` and `ContractPacket` scaffolding in `src/runtime/packets.py` so objective/econ/constraint/evidence contracts can converge without touching default runtime behavior.
- Introduced `EmbodimentRegistry` and `CapabilityProfile` scaffolding in `src/embodiment/registry.py` so robot identity can be normalized before adapter rewrites begin.
- Added a repo-local nightly audit path plus real Codex execution wrappers for local CLI and GitHub/cloud runners.
- Re-centered the docs around Codex app automation plus the repo-local skill as the preferred autonomous path, with CLI and GitHub/cloud as fallbacks.
- Verified the new substrate with agent-ergonomics checks, compileall, targeted pytest, audit generation, and nightly runner shell syntax.
- Kept all new code additive and outside the stable frozen Phase B baseline zones.

## 2026-03-08

- Added `runtime_packet_sidecar_payload(...)` in `src/runtime/packets.py` so runtime packets can be emitted as a deterministic run-level sidecar without forcing replay schema changes.
- Wired `run_shadow_control_plane(...)` to emit `runtime_packets.json`, derive shadow-workcell observation/action schema refs from the existing episode log, and attach packet context into the episode artifact bundle.
- Wired `ingest_shadow_run(...)` to load `runtime_packets.json` when present and thread packet IDs, contract IDs, and sidecar refs into replay metadata/provenance while remaining backward-compatible with older runs that do not have packet sidecars.
- Extended targeted tests to cover sidecar payload serialization, shadow artifact emission, replay metadata round-tripping, and replay-dataset ingestion of packet refs.
- Kept the change additive: no replay dataclass shape changes, no stable Phase B baseline math changes, and no broad adapter refactor.

- Added `src/runtime/event_spine.py` with `RuntimeEvent`, `DecisionLedgerEntry`, and deterministic EventSpine / DecisionLedger sidecar payload builders so ordered runtime events and governance/economic decisions can be persisted without touching the replay schema.
- Wired `src/shadow_runtime/control_plane.py` to emit `event_spine.json` and `decision_ledger.json` with stable event kinds including `queue_reweight`, `pricing_tick_published`, `pricing_tick_suppressed`, `regal_warn`, `regal_veto`, `adaptation_admitted`, `adaptation_denied`, `collect_more_data`, `datapack_credit_assigned`, `promotion_hold`, and `promotion_recommend_promote` when applicable.
- Threaded stable `event_refs` and `decision_refs` through replay episode/step/window `metadata`, with sidecar file refs stored in `provenance`, so downstream consumers can join against the sidecars without requiring replay dataclass changes.
- Bound each emitted event and decision to the new runtime packet layer via `runtime_packet_id`, `contract_id`, objective/econ/pricing/regal artifact refs, and actor/critic/advisor provenance; receipt label refs are present as empty placeholders for future downstream linkage.
- Verified the new layer with targeted sidecar round-trip tests, shadow runner artifact tests, replay schema/dataset tests, receipt-ingest coverage, and `python3 -m compileall src -q`.

## 2026-03-09

- Added `src/runtime/action_adapter_v2.py` and `src/runtime/observation_adapter_v2.py`, plus packet-builder support for schema-producing adapter objects, so runtime contracts can carry explicit timing/provenance instead of relying on ad hoc `SchemaRef` construction at every call site.
- Added `src/evidence/bus.py`, `src/evidence/belief_state.py`, and `src/evidence/teacher_trace.py` to create a common evidence publication layer with validity, disagreement, artifact refs, and advisory teacher-trace semantics.
- Wired `src/vla/rollout_labeler.py` to persist `teacher_trace_v1.json` sidecars and upgraded `src/vla/semantic_evidence.py` so VLA semantic evidence carries governed provenance including teacher-trace refs and fallback mode.
- Wired `src/orchestrator/semantic_fusion_runner.py` to emit `*_evidence_bus_v1.json` and `*_belief_state_v1.json` beside semantic-fusion artifacts, so semantic evidence is no longer trapped in component-local files.
- Reopened `src/world_model/` with `src/world_model/governed_video_world_model.py`, which builds belief-state-driven video-state snapshots and ranked geometry-first hypotheses without touching the stable Phase B checkpoint.
- Upgraded `scripts/run_stage1_pipeline.py` to support manifest-backed real video references, deterministic semantic extraction, governed video-state sidecars/hypotheses, and hypothesis-conditioned diffusion proposals; the script is now directly runnable via `python3 scripts/run_stage1_pipeline.py ...` without `ModuleNotFoundError`.
- Made `src/vision/scene_ir_tracker/io/scene_tracks_runner.py` expose `use_stub_adapters` as a runner option instead of hardwiring it, and exposed richer OpenVLA fallback provenance in `src/vla/openvla_controller.py`.
- Updated the roadmap, automation spec, repo guidance, and training backlog to treat Phase B as a frozen stable baseline plus an additive successor track. Learned video-state modeling is now documented as a backlog item gated on real-video grounding, teacher-runtime hardening, and governed supervision bundles.
- Tightened the roadmap and nightly-selection rules so autonomous passes know to consume Week 6.5 and Week 6.75 in order instead of skipping ahead to learned video-state training.
- Verified the tranche with compileall, focused pytest coverage around evidence/runtime/video-state integration, and a direct Stage-1 CLI smoke run.

- Wired `src/vision/reconstruction/four_d_reconstruction.py` into the live Stage-1 loop so every governed video episode now emits a reconstruction sidecar with calibration completeness, frame windows, geometry refs, and evidence joins.
- Wired `src/world_model/governed_video_supervision.py`, `src/economics/counterfactual_eval.py`, `src/economics/value_targets.py`, and `src/governance/trace.py` into the live Stage-1 loop so candidate futures now emit runtime packets, branch evaluations, event spine rows, decision-ledger rows, governance traces, counterfactual evals, value-target packs, and value-ledger receipts.
- Tightened `src/vla/teacher_runtime.py` plus `src/vla/rollout_labeler.py` so the live rollout-labeling path emits explicit teacher adapter contracts and teacher action envelopes even when OpenVLA is disabled or unavailable; fallback is now replayable instead of implicit.
- Expanded targeted coverage with `tests/test_four_d_reconstruction.py`, `tests/test_teacher_runtime.py`, and `tests/test_governed_video_supervision.py`, and extended Stage-1 / rollout-labeler tests to assert the new live-loop sidecars exist.

## 2026-03-19

- Updated `scripts/economic_world_model/nightly_audit.py` to fix stale roadmap drift evaluation:
  - `_progress_latest_date()` now returns the last dated `## YYYY-MM-DD` heading in `docs/economic_world_model/progress_log.md`, not the first.
  - Added `_event_spine_spec_pending()` and `_contains_phrase(...)` so EventSpine/GovernanceTrace recommendation state is derived from additive code/doc presence rather than a hardcoded `pending=True`.
  - Updated the audit compile command to `PYTHONPYCACHEPREFIX=/tmp/pycache python3 -m compileall src scripts/economic_world_model -q` to avoid sandbox cache-permission failures.
- Added `tests/test_economic_world_model_nightly_audit.py` covering:
  - most-recent progress date extraction
  - EventSpine spec pending=false when code/docs are present
  - fallback to `audit_only` when all candidate tasks are complete
- Regenerated audit artifacts with the updated logic:
  - `artifacts/economic_world_model/nightly_audit_summary.json`
  - `artifacts/economic_world_model/nightly_audit_summary.md`
  - current result: `status=ok`, `roadmap_drift.signals=[]`, `next_task.id=audit_only`.

## 2026-03-21

- Added `src/dataset_bridges/rlds_bridge.py` and `src/dataset_bridges/lerobot_bridge.py` as additive, lossy adapters from canonical replay records to standard interchange formats.
- Kept internal schema richness explicit by preserving objective/econ/pricing/ledger/event/decision/runtime/governance references in bridge metadata rather than flattening them away.
- Added `src/dataset_bridges/__init__.py` exports so bridge adapters can be imported as a package-level surface.
- Added `tests/test_dataset_bridges.py` to validate ordering, terminal-step flags, and sidecar-ref preservation for both RLDS and LeRobot adapter outputs.
- Extended `scripts/economic_world_model/nightly_audit.py` with `_dataset_bridge_scaffold_pending()` and a corresponding `dataset_bridge_scaffold` candidate so roadmap selection includes Week 7+ bridge scaffolding status.
- Extended `tests/test_economic_world_model_nightly_audit.py` with coverage that asserts dataset-bridge candidate selection when the new scaffold is pending.

## 2026-03-22

- Added `src/world_model/semantic_world_model.py` as an additive object-centric semantic memory layer:
  - `SemanticWorldModelState` now carries objects, relations, capability scores, topology, risk register, and meta-node routing state.
  - `SemanticWorldModelBuilder` derives that state from Stage 1 governed video evidence or from rollout semantic-fusion evidence without touching frozen Phase B dynamics math.
- `SemanticWorldModelBuilder` no longer relies only on flat tags when richer grounding exists:
  - it now accepts `SceneTracks_v1`, teacher traces, and VLA semantic evidence as direct inputs
  - it emits track-scoped objects (`track:<track_id>`) with confidence/salience derived from visibility, occlusion, convergence, IR loss, motion, and semantic confidence
  - it derives grounded spatial relations such as `inside`, `near`, `moves_with`, and `rests_on` from real `poses_t` geometry before layering canonical priors on top
- Added `src/semantic/runtime_backbone.py` so runtime producers can emit the same semantic packet family every time:
  - world model sidecar
  - semantic snapshot sidecar
  - orchestrator advisory sidecar
- Stage 1 is no longer only a keyword-tag pipeline:
  - `scripts/run_stage1_pipeline.py` now seeds a richer semantic vocabulary, materializes semantic world-model/snapshot/advisory sidecars, and threads capability/meta-node context into datapack signal bundles and regal annotations.
- Runtime semantic fusion now stops at a shared packet instead of dying at evidence fusion:
  - `src/orchestrator/semantic_fusion_runner.py` now emits semantic world-model, snapshot, and orchestrator sidecars beside belief/evidence sidecars when fusion succeeds.
- Runtime semantic fusion now passes the real grounding artifacts into that packet:
  - `scene_payload` is used as the SceneTracks source
  - `teacher_trace_v1` and VLA semantic evidence sidecars are passed into semantic world-model construction instead of being reduced to only flat semantic tags
- Snapshot/orchestrator/observation/sampler wiring is now materially stronger:
  - `src/semantic/models.py` gives `SemanticSnapshot` a first-class `semantic_world_model` field.
  - `src/semantic/aggregator.py` can now carry that field through Stage 2 aggregation.
  - `src/orchestrator/semantic_orchestrator_v2.py` now translates semantic topology/capabilities into meta-node weights rather than only shallow sampler tags.
  - `src/observation/adapter.py` and `src/observation/condition_vector_builder.py` now expose capability/topology/meta-node signals in the observation/condition path.
  - `src/rl/episode_sampling.py` now uses advisory meta-node weights as bounded replay-priority multipliers.
- Added `docs/economic_world_model/semantic_gap_matrix.md` as the written semantic sweep and translation layer for the repo.
- Verification for this semantic-world-model pass:
  - `python3 -m compileall src scripts/run_stage1_pipeline.py tests -q`
  - `python3 -m ruff check scripts/run_stage1_pipeline.py src/world_model/semantic_world_model.py src/semantic/models.py src/semantic/aggregator.py src/semantic/runtime_backbone.py src/orchestrator/semantic_orchestrator_v2.py src/orchestrator/semantic_fusion_runner.py src/observation/adapter.py src/observation/condition_vector_builder.py src/rl/episode_sampling.py tests/test_stage1_pipeline_governed.py tests/test_semantic_world_model_backbone.py`
  - `python3 -m pytest -q tests/test_stage1_pipeline_governed.py tests/test_semantic_world_model_backbone.py tests/test_governed_video_world_model.py tests/test_semantic_fusion_emit_flag.py tests/test_semantic_fusion_orchestrator_smoke.py tests/test_semantic_policy.py`
- Remaining blocker:
  - grounded semantic memory now exists, but only when upstream producers emit usable SceneTracks/class labels/teacher evidence. The next upgrade is to improve those upstream producers so the semantic world model sees dense grounded artifacts more often and fewer stub/fallback cases.

- Added `src/dataset_bridges/sidecar_refs.py` with `extract_sidecar_refs(...)` to centralize replay sidecar extraction for bridge exports.
- The extractor keeps bridge exports additive and forward-compatible by harvesting references from replay record fields and `metadata`/`provenance` keys that end in `*_ref`, `*_refs`, `*_id`, or `*_ids`.
- Switched `src/dataset_bridges/rlds_bridge.py` and `src/dataset_bridges/lerobot_bridge.py` to use the shared extractor instead of hardcoded per-key sidecar mappings, reducing future maintenance when new governed-supervision refs are introduced.
- Extended `tests/test_dataset_bridges.py` so RLDS/LeRobot bridge outputs assert preservation of representative Week 6.75/7+ sidecar refs (`counterfactual_eval_ref`, `value_target_refs`, `belief_state_ref`, `teacher_trace_ref`, and `governed_supervision_refs`).
- Added `scripts/economic_world_model/publish_codex_change.sh` to publish automation commits to `origin/main` when the local change is a safe fast-forward, while falling back to a timestamped `codex/ewm-nightly-*` branch when direct main pushes are rejected.
- Updated `scripts/economic_world_model/run_nightly_codex_task.sh` so the generated Codex task now requires publication via the helper and reports either the published ref or the exact push blocker before the run is considered complete.
- Updated `docs/economic_world_model/AUTOMATION_SPEC.md`, `codex_skills/economic-world-model-roadmap/SKILL.md`, and the live app automation prompt to treat unpublished local commits as incomplete automation output.

- Added `src/economics/inferential_reward.py` as a shared successor-layer compiler for `InferentialSignalYield` and `InferentialRewardBreakdown`, keeping signal-yield math additive and outside frozen Phase B reward/dynamics code.
- Extended `InferentialTrainingCandidate` and `InferentialTrainingGate` to carry frontier gain, epiplexity, transfer, governance, and optional signal-yield overrides, then compile a canonical inferential reward breakdown before making budget decisions.
- Wired advisory consumers to use the compiled signal-yield path:
  - `src/orchestrator/shadow_advisory.py` now computes signal yield from replay frontier gain plus any available epiplexity fields.
  - `src/rl/econ_regal_sampling.py` now admits signal yield as a bounded replay-priority input.
  - `src/rl/episode_sampling.py` and `src/policies/sampler_weights.py` now emit/consume `signal_yield_score` and `inferential_replay_weight`, including a new `inferential_yield` weighting strategy.
  - `src/orchestrator/queue_selection.py` now preserves inferential reward evidence in queue metadata.
- Refactored the epiplexity core so tracker cache entries are baseline-independent absolute runs with estimator provenance and `flops_estimate`, while baseline-relative `delta_epi_vs_baseline` is derived only when consumers compare a candidate against a baseline.
- Promoted `RequentialEstimator` from a zero-return stub into an online evaluate-then-update estimator, so the second estimator path now produces nontrivial learnability scores instead of placeholder zeros.
- Added canonical epiplexity overlay helpers and automatic repo merging:
  - `src/epiplexity/metadata.py` now writes/loads `epiplexity_overlays.jsonl`, manages default selectors, and lets consumers recover the best available repr/budget even when `_default` is absent.
  - `src/valuation/datapack_repo.py` now auto-merges epiplexity overlays during `load_all(...)` and invalidates cached task loads when the overlay sidecar changes.
- Wired `scripts/run_epiplexity_curated_slices.py` to persist canonical overlays in both full and token-only modes, so portable fallback runs now emit the same summary shape consumed by samplers and replay/inferential advisory code.
- Corrected downstream consumers that had been reading the wrong epiplexity slot:
  - `src/orchestrator/datapack_engine.py` now uses `epi_repr_id` or the datapack default selector rather than incorrectly reading the baseline repr’s delta.
  - `src/orchestrator/homeostatic_plan_writer.py` and `src/representation/homeostasis.py` now understand canonical nested epiplexity summaries instead of only legacy `mean_variance`/`variance` placeholders.
  - `src/evaluation/probe_harness.py` now reports real baseline/after means rather than recycling the delta into those fields.
- Kept the change additive: no edits to the stable Phase B checkpoint, no legacy world-model math rewrite, and no baseline reward-path mutation.
- Added `docs/economic_world_model/ewm-nightly.automation.toml` as a checked-in mirror of the live Codex app automation config, omitting only local timestamp fields so the active prompt/schedule/environment are versioned with the repo.
- Updated `docs/economic_world_model/AUTOMATION_SPEC.md` to point at the checked-in automation snapshot as the Git-tracked source of truth for the live app automation state.

- Added `docs/economic_world_model/self_improvement_preconditions_sweep.md` to capture where the repo should stop at advisory sidecars versus where it now has enough substrate to promote those sidecars into self-improvement preconditions.
- The sweep treats queue dispatch as the positive template because it already gives advisory outputs bounded influence in live training paths, and it argues that the next promotions should be work orders, promotion evidence joins, replay roundtrip/rehydration, governed-video admission contracts, and explicit degraded-evidence artifacts rather than broader controller sovereignty.
- The same sweep also marks modules that should stay advisory for now, especially `src/orchestrator/semantic_orchestrator_v2.py`, `src/orchestrator/pipeline_manager.py`, `src/orchestrator/economic_controller.py`, and `src/hrl/high_level_controller.py`, because they still sit above insufficiently strict packet/event/evidence ingestion layers.

- Added `src/evidence/preconditions.py` as the shared execution-readiness/work-order vocabulary:
  - `ExecutionPreconditionsReport` now normalizes artifact presence, signal thresholds, boolean requirements, and explicit blockers into one JSON-safe artifact.
  - `ExecutionWorkOrder` now gives downstream training/review/data-collection consumers a stable executable-vs-blocked order surface instead of requiring each module to reinterpret advisory summaries independently.
- Added `src/replay/preconditions.py` plus `ReplayDatasetBuilder` wiring so canonical replay datasets now persist trace completeness as data, not just as latent sidecar refs:
  - episodes now carry `metadata.execution_preconditions`
  - manifests now carry `metadata.execution_precondition_summary`
  - RLDS/LeRobot exports can now roundtrip back into canonical replay rows with sidecar refs rehydrated instead of dropped
- Extended `src/orchestrator/adaptation_budgeting.py` and `src/orchestrator/shadow_advisory.py` so inferential budget decisions become actual work-order artifacts gated by replay trace completeness; this is the first place the repo now creates explicit preconditions for self-improvement rather than only reporting desirability.
- Hardened governed-video admission and fusion failure handling:
  - `scripts/run_stage1_pipeline.py` now emits `governed_video/proposal_admission_v1.jsonl` with proposal-level execution preconditions and admission work orders, and admitted datapacks carry those artifacts into `episode_metrics` / `regal_annotations`
  - `src/world_model/governed_video_supervision.py` now accepts a stable ledger path instead of hardcoding `/tmp`
  - `src/orchestrator/semantic_fusion_runner.py` now writes per-episode degraded-evidence artifacts/work orders on mismatch or missing-input failures instead of silently skipping them
- Hardened weak execution substrates before widening planner authority:
  - `src/vision/scene_ir_tracker/io/scene_tracks_runner.py` now classifies stub-adapter / low-quality outputs as not training-eligible in explicit precondition metadata
  - `src/vla/teacher_runtime.py` now emits execution-precondition metadata for both contract availability and per-prediction failures
  - `src/policies/unified_quality.py` and `src/rl/episode_sampling.py` now let execution preconditions block datapack/replay eligibility, making the new substrate materially affect who can train
- Kept top-level shells advisory but no longer blind:
  - `src/orchestrator/semantic_orchestrator_v2.py`, `src/orchestrator/pipeline_manager.py`, `src/phase_h/controller.py`, and `src/phase_h/economic_learner.py` now surface precondition summaries as advisory routing/repair context instead of pretending blocked substrates are ready.

- Added `scripts/SHELL_ACTIVATION_BACKLOG.json` as the machine-readable higher-shell promotion backlog and `src/orchestrator/shell_activation.py` as the evaluator:
  - backlog entries define per-shell activation thresholds (`min_report_count`, `min_ready_count`, `max_blocked_count`, `min_mean_readiness_score`)
  - backlog entries can also carry future-training-only satisfied-precondition requirements such as `signal_bool::teacher_runtime_live` or `artifact::training_runtime_manifest`
  - the evaluator normalizes either aggregated summaries or single readiness reports into one activation assessment shape
- Promoted the higher shells from “advisory only” to “conditionally executable” where the substrate is already good enough:
  - `src/orchestrator/semantic_orchestrator_v2.py` now emits `execution_mode`, a bounded routing `activation_plan`, and a typed shell work order when the current semantic-routing backlog item is activated
  - `src/orchestrator/pipeline_manager.py` now builds a bounded next-iteration stage-activation plan and work order instead of only returning advisory preview text when readiness is green
  - `src/phase_h/advisory_integration.py` now exposes `build_phase_h_activation_plan(...)`, letting Phase H routing become a bounded activation artifact while preserving the existing ±20% caps
  - `src/phase_h/controller.py` and `src/phase_h/economic_learner.py` now surface shell activation state, activation work orders, and future-training backlog readiness in their cycle summaries
- Added `docs/economic_world_model/shell_activation_backlog.md` to explain which shell promotions are auto-activating today versus which ones stay in the future-training backlog until stronger grounding and runtime evidence become explicit readiness checks.

## 2026-03-24

- Added `src/replay/importers.py` as the importer-side bridge for governed-video supervision and degraded semantic-fusion artifacts:
  - `ingest_governed_video_admission_log(...)` converts `governed_video/proposal_admission_v1.jsonl` rows into canonical `ReplayEpisodeRecord`, `ReplayStepRecord`, and `ReplayWindowRecord` entries while preserving source execution preconditions/work orders.
  - `ingest_semantic_degraded_artifacts(...)` converts `*_semantic_degraded_v1.json` outputs into canonical negative-supervision replay episodes instead of leaving them as dead-end failure sidecars.
  - importer normalization now aliases `*_path`/`*_paths` keys into `*_ref`/`*_refs` forms so downstream replay code sees one stable reference vocabulary.
- Extended replay readiness synthesis so future-training checks become explicit data:
  - `src/evidence/preconditions.py` now supports soft artifact/signal checks (`soft_required_artifact_refs`, `soft_min_signal_thresholds`, `soft_max_signal_thresholds`, `soft_boolean_signals`) that appear in `satisfied_preconditions` summaries without blocking current execution readiness.
  - `src/replay/preconditions.py` now derives/imports `future_training_signals` and `future_training_artifacts`, then emits them as soft readiness checks for replay episodes.
  - `collect_replay_artifact_refs(...)` now harvests `*_path` and `*_paths` keys in addition to `*_ref`/`*_id` families so importer-produced replay rows can roundtrip through the same precondition summarizer.
- Tightened the future-training signal merge policy on replay import:
  - replay-owned facts such as `replay_roundtrip_complete` are recomputed by the importer and no longer inherit stale `false` values from Stage 1 source artifacts.
  - source-computed grounding signals such as `scene_tracks_non_stub`, `semantic_memory_grounded`, and `budget_settlement_live` are merged monotonically so importer paths preserve positive upstream evidence when they cannot re-derive it from filenames alone.
- Wired the new importer/readiness path back into producers:
  - `scripts/run_stage1_pipeline.py` now writes explicit `future_training_signals` and `future_training_artifacts` into governed-video admission records and feeds them into `build_execution_preconditions(...)` as soft checks.
  - `src/orchestrator/semantic_fusion_runner.py` now writes the same explicit future-training fields into degraded semantic-fusion artifacts so importer-side replay can recover them without heuristics.
  - `src/replay/dataset.py` now exposes `.add_governed_video_admission_log(...)` and `.add_semantic_degraded_artifacts(...)` so these producer outputs become first-class canonical replay inputs.
- Updated the nightly audit selector to track the new state of the backlog:
  - `scripts/economic_world_model/nightly_audit.py` now emits `future_training_evidence_wiring` as the next additive task when shell activation backlog entries exist but replay/training paths still lack explicit promotion-ledger or budget-settlement evidence.
  - `tests/test_economic_world_model_nightly_audit.py` now covers that candidate selection path so automation does not regress back to `audit_only`.
- Closed that future-training evidence gap with canonical runner + receipt-ingest wiring:
  - `src/training/regal_training_runner.py` now synthesizes `promotion_ledger_v1.json` from the current promotion evidence artifact and `budget_settlement_v1.json` from observed receipt/coverage evidence, then registers both in the unified runtime artifact map before writing the manifest.
  - `src/training/training_manifest.py` now has explicit `promotion_ledger_path`, `promotion_ledger_digest`, `budget_settlement_path`, `budget_settlement_digest`, and `budget_settlement_live` fields so downstream code no longer has to infer those from generic `artifact_paths`.
  - `src/replay/receipt_ingest.py` now enriches training-run replay bundles in memory with those manifest-derived artifacts/signals and recomputes replay execution preconditions, so receipt-label ingestion becomes a real bridge back into shell-activation readiness instead of only a labeling utility.
- Added focused regression coverage for the new bridge:
  - `tests/test_regal_training_runner.py` now asserts that canonical training runs emit promotion-ledger and budget-settlement artifacts and that the runtime manifest records them explicitly.
  - `tests/test_training_run_receipt_ingest.py` now asserts that training-run receipt ingestion surfaces `artifact::training_runtime_manifest`, `artifact::promotion_ledger_ref`, and `signal_bool::budget_settlement_live` in its execution-precondition summary metadata.
- The nightly audit now returns `audit_only` after these changes, which is the intended result:
  - the missing additive substrate is gone
  - future-training shell backlog items are now waiting on real run evidence, not missing code paths

- Added a first-class semantic coverage substrate beside the semantic WM:
  - `src/hrl/skill_graph.py` defines the repo-level skill graph spanning HRL, SIMA, VLA, and Stage 2 hints.
  - `src/envs/primitive_inventory.py` defines typed env primitive inventories for `drawer_vase`, `dishwashing`, and `workcell`.
  - `src/world_model/semantic_coverage_graph.py` compiles those plus runtime evidence into a typed task × skill × env-primitive graph.
  - `src/world_model/coverage_evidence_harvester.py` harvests real evidence counts and priority scalars from replay/runtime rows instead of relying only on hand-authored coverage priors.
- Added the first cybernetic learning surfaces for the coverage loop:
  - `src/world_model/fill_outcome_store.py` persists append-only fill outcomes as supervised training data.
  - `src/world_model/gap_ranker.py` trains a learned marginal-value model for missing-edge ranking.
  - `src/world_model/fill_path_policy.py` trains a learned fill-method policy over `real_sim | diffusion | synthetic_branch | blocked`.
  - `src/world_model/semantic_state_encoder.py` provides both a deterministic flat encoder and a torch-backed set encoder so semantic WM state can condition learned downstream modules without collapsing the packet schema again.
- Wired the coverage graph into the broader synth loop:
  - `src/orchestrator/coverage_loop.py` now runs the evidence-harvest → graph-build → gap-rank → sim-agenda → diffusion-prompt → fill-decision cycle.
  - `src/orchestrator/diffusion_requests.py` now supports gap-driven prompt compilation.
  - `src/orchestrator/semantic_simulation.py` now supports ranked simulation agendas compiled from coverage deficits.
  - `src/orchestrator/pipeline_manager.py` can now emit `semantic_coverage` artifacts when explicitly configured.
  - `scripts/collect_local_synthetic_branches.py` and `scripts/train_latent_diffusion.py` can now consume gap labels / semantic conditioning so synthetic branch collection and latent diffusion training stop being purely trust/econ driven.
- Added additive routing guards around the new loop:
  - `src/process_reward/evidence_adapter.py` turns process-reward outputs into evidence/precondition packets.
  - `src/evidence/backend_health.py` turns perception/runtime backend degradation into explicit readiness metadata.
  - `src/governance/assessment.py` turns governance traces into coverage and veto summaries that can later be routed back into graph weights and meta-node decisions.
- Closed the first packetized return path for that loop:
  - `src/world_model/semantic_feedback_packets.py` now defines typed `CoverageOutcomePacket`, `WMValidationPacket`, and `GraphMutationProposal` surfaces plus compiled trust/econ/readiness overlays.
  - `src/orchestrator/coverage_loop.py` now consumes those packets, merges them with harvested edge priorities, marks governance-blocked edges directly on the graph, and emits `feedback_summary`, `wm_validation_summary`, `trust_calibration_overlay`, `econ_calibration_overlay`, and `graph_mutation_proposals`.
  - `src/orchestrator/pipeline_manager.py` now runs coverage compilation before transformer routing when configured and injects the resulting summaries into `OrchestratorContext.semantic_metadata`, so both transformer shells can react to coverage outcomes, WM validation pressure, and graph-expansion pressure in the same pass.
  - `src/orchestrator/semantic_transformer_bridge.py`, `src/orchestrator/meta_transformer.py`, and `src/orchestrator/orchestration_transformer.py` now encode feedback fields such as `missing_edge_fraction`, `wm_validation_error_rate`, `trust_overlay_mean`, `econ_overlay_mean`, and `graph_mutation_pressure` into the semantic feature vector and bounded action/work-order plans.
- Upgraded the synth-facing training consumers to actually use semantic conditioning:
  - `scripts/train_latent_diffusion.py` now threads semantic conditioning into the latent MLP and transformer models during real training rather than only loading the sidecar.
  - `scripts/train_trust_aware_world_model.py` now carries the same conditioning through trust-aware reconstruction and rollout losses.
  - `scripts/train_world_model_from_datapacks.py` now appends semantic-gap/process-reward/coverage features and additive semantic-gap weighting when building datapack world-model datasets, so latent/synthetic branching paths stop being purely trust × `w_econ` weighted.
  - `scripts/sample_zv_rollouts.py`, `scripts/eval_world_model_rollouts.py`, and `scripts/train_horizon_agnostic_world_model.py` now reopen semantic-conditioned latent checkpoints compatibly.
- Important remaining limitation:
  - That limitation is now closed additively for the semantic runtime loop itself:
    - `src/world_model/semantic_wm_correction.py` compiles WM-validation packets into an explicit correction overlay and applies it to a copy of the semantic WM for downstream routing.
    - `src/world_model/graph_mutation_executor.py` applies bounded runtime graph mutations under governance/confidence thresholds before coverage-graph construction, so topology is no longer fixed for the duration of the loop.
    - `src/world_model/feedback_topology_adapters.py` provides the learned trust/econ/readiness/correction overlay package plus shadow-fit training from real coverage edges, and `scripts/train_semantic_feedback_adapters.py` provides the heavyweight persisted training path.
    - `src/world_model/semantic_wm_refiner.py` now sits beside the deterministic builder as a learned successor layer. It learns bounded object/relation/capability correction deltas plus graph-mutation proposal scoring, and `scripts/train_semantic_wm_refiner.py` provides the heavyweight persisted training path from coverage-loop artifacts.
    - `src/orchestrator/coverage_loop.py` now emits both `input_semantic_world_model.json` and `semantic_wm_refiner_summary.json`, shadow-fits or loads a persisted refiner package, merges learned deltas with the heuristic correction overlay, and keeps the resulting outputs inside the governed correction/mutation route rather than writing directly into the base builder.
- The remaining limitation is no longer missing code-paths. It is whether enough real coverage-loop artifacts exist to promote those learned overlay adapters and governed graph mutations from shadow/provisional use into stronger production authority.

- D4 knob calibration is now a real helper lane rather than a fake learned label:
  - `src/regal/knob_model.py` keeps the heuristic provider as the explicit prior and delegates learned resolution to `src/regal/knob_model_runtime.py`.
  - `resolve_knob_model(...)` enforces the same honest bounded helper semantics used elsewhere in this pass:
    - no package -> heuristic fallback unless `required=True`
    - non-benchmark-gated package -> bounded `shadow_candidate` helper only
    - benchmark-gated package -> stronger but still bounded influence
- `scripts/train_knob_model.py` is the canonical trainer for that lane. It accepts either:
  - explicit knob training dataset JSON
  - runtime/exported `knob_policy_receipt_v1` JSON or JSONL
  - optional heuristic-bootstrap synthetic rows as an additive fallback source
  and emits the full runtime contract:
  - `knob_model_dataset.json`
  - dataset summary
  - execution-preconditions report
  - training summary
  - `knob_model_package.json`
  - canonical runtime manifest / checkpoint registry artifacts
- `src/orchestrator/homeostatic_plan_writer.py` now preserves enough context to make future training honest:
  - `GateStatus.knob_policy`
  - `GateStatus.knob_policy_used`
  - `GateStatus.knob_regime_features`
  - `GateStatus.knob_base_config`
  This means the knob lane’s meta-choice no longer disappears once the plan is written.
- `scripts/run_closed_loop_smoke.py` now writes `knob_policy_receipt.json` so the repo has an immediate regression/smoke substrate for knob-model training without inventing a fake online learner.
- The next mandate-level runtime gaps are above this helper, not inside it:
  - higher-order orchestrator shell/stage/meta-choice policy is still largely deterministic
  - queue/curriculum weighting is still mostly heuristic
  - real grounded-data promotion is still blocked on GPU + SAM3D availability
  - several remaining heavyweight trainers are still data-limited rather than plumbing-limited, as recorded in `docs/economic_world_model/full_stack_training_backlog.md`

- Queue dispatch is no longer the fake boundary in that lane:
  - `src/orchestrator/queue_dispatch_policy.py` defines a stable feature/target contract over live queue entries, preserving advisory priority, replay action, queue tags, semantic-runtime scorer outputs, execution-precondition state, and receipt-feedback outcomes.
  - `scripts/train_queue_dispatch_policy.py` is now the canonical trainer/runtime-package path for that helper under `RegalTrainingRunner`.
  - `src/orchestrator/queue_selection.py` now blends helper output against the explicit heuristic multiplier prior with bounded `disabled|auto|required` semantics, and `src/rl/episode_sampling.py` plus the main shadow/online trainer entrypoints thread that helper into the real sampling loop.
  - Honest remainder at that point: the deeper sampler base-weight / curriculum-strategy logic in `src/rl/episode_sampling.py` still used frontier/econ/curriculum heuristics underneath the now-real queue-dispatch layer.

- That deeper sampler substrate is now also real and bounded:
  - `src/rl/sampler_policy.py` defines stable pool-level and episode-level feature contracts for strategy choice, frontier/econ plan parameters, and strategy-conditioned base-weight adjustment.
  - `scripts/train_sampler_policy.py` is now the canonical trainer/runtime-package path for that helper under `RegalTrainingRunner`.
  - `src/rl/episode_sampling.py` now blends helper outputs against explicit heuristic priors for:
    - strategy selection
    - per-episode base weights
    - frontier/econ threshold and focus ratios
  - `DataPackRLSampler` now emits `sampler_policy_receipt_v1` artifacts, and the main shadow/online training entrypoints persist those receipts into runtime outputs so later training can move off heuristic bootstrap targets.
  - Honest remainder: this lane is no longer missing wiring. It is blocked on receipt density. The helper should remain benchmark-gated until queue outcome receipts and replay counterfactual labels are dense enough to promote it beyond `shadow_candidate`.

- The semantic runtime scorer lane now has the same honest runtime contract as the other helper modules:
  - `scripts/train_semantic_runtime_scorers.py` now emits:
    - runtime training dataset
    - dataset summary
    - execution-precondition artifact
    - model config
    - training summary
    - the legacy linear scorer package for compatibility
    - `semantic_runtime_scorer_runtime_package.json`
    - canonical runtime manifest / checkpoint registry artifacts under `RegalTrainingRunner`
  - `src/orchestrator/semantic_runtime_scorer_runtime.py` gives the scorer a stable runtime-package loader, and `src/orchestrator/shadow_advisory.py` now prefers that package while preserving contract type, promotion stage, benchmark gate, and legacy-fallback truth in the advisory output.
  - Honest remainder: this lane is no longer missing production wiring. It is blocked on execution-ready / semantic-grounded replay density for promotion.

- The semantic coverage helper lanes are now canonical runtime helpers rather than raw-checkpoint sidecars:
  - `scripts/train_semantic_feedback_adapters.py` and `scripts/train_semantic_wm_refiner.py` now emit canonical dataset/precondition/model/training/runtime-package artifacts under `RegalTrainingRunner`.
  - `src/world_model/feedback_topology_runtime.py` and `src/world_model/semantic_wm_refiner_runtime.py` now provide bounded `disabled|auto|required` helper loading consistent with the other learned helper lanes.
  - `src/orchestrator/coverage_loop.py` now consumes those runtime packages directly, applies explicit shadow/promoted blend weights for feedback overlays, applies explicit shadow/promoted scales for learned correction overlays and graph-mutation proposal confidences, and records helper status in the coverage summaries instead of silently treating persisted packages, raw checkpoints, and shadow-fit fallbacks as equivalent.
  - `src/orchestrator/pipeline_manager.py` now forwards runtime-package refs and helper modes directly into the coverage loop instead of open-coding checkpoint loads.
  - Honest remainder: these lanes are no longer missing runtime packaging. They remain benchmark-gated until enough repeated coverage-loop artifacts accumulate to promote them beyond `shadow_candidate`.

- The sim/synth helper runtime packages are now relocatable and package-faithful instead of training-directory-bound:
  - `scripts/train_sim_synth_backend_selector.py` and `scripts/train_sim_synth_branch_planner.py` now write runtime packages with artifact refs relative to the package root when possible, which makes the emitted package portable across training/output directories.
  - `src/world_model/sim_synth_physics/backend_selector_runtime.py` and `src/world_model/sim_synth_physics/branch_planner_runtime.py` now resolve relative `checkpoint_path` values against `package_path` and preserve package metadata (`package_id`, `package_path`, `promotion_stage`, `metadata`) on the loaded helper object.
  - `tests/test_sim_synth_physics_world_model.py` now exercises the end-to-end reload path using package JSONs with relative checkpoint refs, which is the real contract the downstream WM runtime will consume.
  - Honest remainder: the helper seam is no longer brittle, but helper promotion is still limited by real branch/backend receipt density rather than packaging.

- The queue/sampler lane has started the advisory-doctrine cleanup:
  - `src/orchestrator/queue_selection.py` now emits explicit `receipt_kind`, `authority_class`, `decision_scope`, and `reward_math_mutation` fields on both live queue-selection inputs and queue-dispatch receipts.
  - `src/rl/episode_sampling.py` now carries those fields into `sampler_policy_receipt_v1` artifacts and also emits the sampler receipt from `dispatch_queue(...)`, so the bounded authority exercised during training-distribution selection is typed instead of implicitly inferred.
  - `src/rl/sac.py` now preserves the same bounded-authority classification in online replay sampling artifacts, which closes the last obvious runtime hole where queue influence could still look like anonymous advisory metadata.
  - Honest remainder: this is a contract/doctrine correction, not yet the final orchestration-level advisory purge. Higher-shell and orchestration control surfaces still need the same cleanup treatment.

- Inferential admission is now a first-class contract instead of an interpretation of summaries:
  - `src/economics/inferential_contract.py` now defines `inferential_admission_contract_v1`, which packages per-episode decisions, learnability-class carry-through, and work-order summaries into one canonical artifact.
  - `src/economics/inferential_training_gate.py` now emits typed work-order-class decisions with explicit `receipt_kind`, `authority_class`, `decision_scope`, and `reward_math_mutation` fields.
  - `src/orchestrator/adaptation_budgeting.py` and `src/orchestrator/shadow_advisory.py` now propagate that admission contract into live advisory outputs and per-episode rows, so downstream consumers stop reverse-engineering admission truth from `adaptation_budget.summary`.

- The canonical training/runtime path now preserves inferential admission directly:
  - `src/training/regal_training_runner.py` and `src/training/training_manifest.py` now carry `inferential_admission_summary` in the unified runtime manifest.
  - `scripts/train_shadow_replay_policy.py`, `scripts/train_shadow_offline_rl.py`, `scripts/train_shadow_pricing_models.py`, `scripts/train_sac_with_ontology_logging.py`, and `scripts/run_shadow_advisory_pass.py` now emit/register `inferential_admission_contract.json` beside the existing learnability and work-order artifacts.
  - This matters because trainer/runtime reporting can now distinguish:
    - learnability class density
    - admission decision density
    - executable work-order density
    instead of flattening them into one advisory budget summary.

- Epiplexity-based learnability is now promoted into datapack-owned canonical metadata:
  - `src/valuation/datapack_schema.py` now includes `inferential_learnability_contract`.
  - `src/valuation/datapack_repo.py` now attaches or preserves that contract when datapacks are loaded and epiplexity overlays are applied, while preserving richer receipt-backed contracts if they already exist.
  - `src/rl/episode_sampling.py` now preserves the datapack’s canonical learnability contract in RL descriptors instead of always reconstructing signal-yield state from local epiplexity fields.
  - This closes the doctrine gap where epiplexity was “visible” but not yet treated as canonical selection metadata across datapacks, replay descriptors, and training artifacts.

- Honest next advisory cleanup after this pass:
  - internal orchestration sidecars still need a companion `control_plane_context` / receipt path
  - especially `semantic_fusion_runner`, `runtime_backbone`, Stage-1 pipeline emissions, and replay ingest
  - that is now the next place where bounded internal selectors still look softer than they really are

- That semantic-runtime companion receipt now exists:
  - `src/semantic/runtime_backbone.py` now emits `orchestrator_control_plane_context_v1` beside the semantic WM, semantic snapshot, and orchestrator advisory.
  - The new context artifact carries typed authority metadata (`receipt_kind`, `authority_class`, `decision_scope`, `reward_math_mutation`) plus the actual control-plane fields that matter downstream:
    - meta-node weights
    - focus objective presets
    - sampler strategy overrides
    - benchmark signals
    - execution preconditions
    - semantic-runtime truth
    - semantic-WM summary
  - This is important because the repo now has a canonical place to preserve bounded internal selector state without pretending that the original `orchestrator_advisory` JSON is itself the whole runtime truth surface.

- The main semantic-runtime producers now preserve that context:
  - `scripts/run_stage1_pipeline.py` writes `*_control_plane_context_v1.json` into `governed_video`.
  - `src/orchestrator/semantic_fusion_runner.py` writes and records the same artifact for runtime-fusion episodes.
  - `scripts/bootstrap_semantic_workcell_loop.py` now carries `control_plane_context_path` through bootstrap metadata and summary outputs.

- Replay ingest now treats the control-plane companion as canonical metadata instead of ignoring it:
  - `src/replay/ingest.py` discovers `control_plane_context_path` sidecars, preserves `control_plane_context_ref` in provenance, and hydrates the parsed context into replay episode metadata.
  - This gives downstream training/runtime consumers a stable typed place to recover internal selector/meta-node state without scraping free-form sidecars.

- Honest remainder after this control-plane-context pass:
  - external-provider doctrine cleanup still remains
  - higher-shell orchestration and Phase H surfaces still need the same reclassification treatment
  - but the lower semantic-runtime boundary is no longer one of the major advisory-truth gaps

- External-provider doctrine is now explicit in code instead of being a convention:
  - `src/evidence/provider_truth.py` defines the shared `external_provider_truth_v1` contract so teacher/VLA and SceneTracks lanes can publish canonical provider availability / fallback / calibration / grounding metadata without promoting their predictions to truth.
  - `src/vla/teacher_runtime.py` and `src/evidence/teacher_trace.py` now preserve `provider_truth` on teacher adapter contracts, teacher action envelopes, and teacher traces; this means replay and downstream runtime-learning can see whether the teacher was real, disabled, unavailable, or degraded without inferring that from the prediction payload itself.
  - `src/vision/scene_ir_tracker/io/scene_tracks_runner.py` now emits `scene_tracks_provider_truth`, and `src/evidence/scene_tracks_truth.py` now treats that explicit grounding-class metadata as canonical when it exists.
  - `src/vla/rollout_labeler.py` and `src/replay/ingest.py` now preserve both teacher and SceneTracks provider truth into datapack/replay metadata, which is the important doctrine correction: provider outputs remain advisory, provider status does not.

- The remaining top-shell and curriculum doctrine lag is now also closed:
  - `src/phase_h/advisory_integration.py`, `src/phase_h/controller.py`, and `src/phase_h/economic_learner.py` now emit typed shell receipt fields and preserve `input_receipt_context` instead of flattening consumed execution/precondition/work-order inputs into generic advisory summaries.
  - `src/orchestrator/pipeline_manager.py` now carries the same typed input receipt context through build/preview/report flows and into shell activation work-order metadata.
  - `src/rl/curriculum.py` is no longer described as “purely advisory”; it now emits `curriculum_dispatch_receipt_v1` as bounded training-distribution authority, which matches what the module has really been doing.

- Honest remainder after this pass:
  - no major non-GPU advisory contract gap remains in the already-wired internal control-plane stack
  - real grounded-data promotion still depends on GPU + SAM3D and better receipt density
  - future advisory cleanup, if any, should mostly be doctrine cleanup in newly added modules rather than another large retroactive purge of the existing loop

- Anti-regression contract hygiene is now explicit:
  - `scripts/check_canonical_receipt_contracts.py` statically scans the main internal control-plane packages and fails when internal receipt emitters expose `receipt_kind` without the rest of the canonical authority tuple (`authority_class`, `decision_scope`, `reward_math_mutation`)
  - the same checker now treats provider-truth surfaces as canonical metadata rather than soft conventions and fails when those surfaces are used without the shared contract path
  - `scripts/run_full_repo_verification.py` runs this checker by default so the advisory purge turns into a maintained invariant rather than a one-time cleanup

- The sim/synth corpus lane now defaults closer to live production receipts:
  - `src/world_model/sim_synth_physics/training_corpus.py` can harvest `sim_synth_physics_world_state_v1`, `physics_calibration_receipt_v1`, and `simulation_outcome_receipt_v1` files directly from receipt directories and reassemble canonical bundles
  - the backend-selector and branch-planner trainers can now auto-build datasets from `--receipt-dir` inputs or from nearby runtime output roots when no explicit dataset or receipt bundle is provided
  - dataset summaries now preserve receipt provenance (`receipt_source_kind`, `receipt_dirs`, `receipt_bundle_count`) so downstream promotion decisions can tell whether a helper was fit on live harvested runtime receipts or on manually assembled bundles
  - while landing this, the WM package surface was trimmed to avoid eager compiler/runtime imports from `src/world_model/sim_synth_physics/__init__.py`, which fixed a real circular-import failure between the package and `src/orchestrator/diffusion_requests.py`

- Promotion/readiness reporting now surfaces the canonical classes that replaced advisory-only summaries:
  - `src/regality/promotion_reporting.py` now summarizes control-plane context density and provider-truth density, not just inferential learnability density
  - reports now carry:
    - `work_order_ready_count`
    - `control_plane_context_summary`
    - `teacher_provider_truth_summary`
    - `scene_tracks_provider_truth_summary`
  - per-node coverage also now records control-plane / provider-truth episode counts so readiness reviews can see whether canonical metadata is actually present across the loop rather than only in spot artifacts

- Practical consequence:
  - the remaining within-mandate work is not another broad advisory purge
  - it is maintaining these invariants as new helpers, new WMs, and new training/reporting lanes are added
  - the real blockers now are receipt density, grounded-data availability, and benchmark evidence, not missing contract scaffolding

- Phase 1 sim/synth/physics WM now has a more honest run-time boundary:
  - `src/world_model/sim_synth_physics/runtime.py` is no longer only a compiler wrapper; it now owns:
    - compile-time legacy agenda compatibility
    - compile-time diffusion-plan compatibility
    - planning-window execution into canonical backend-routing, calibration, outcome, and training-feedback artifacts
  - the new runtime emits:
    - `physics_execution_contract_v1`
    - `physics_calibration_receipt_v1`
    - `simulation_outcome_receipt_v1`
    - `sim_synth_training_feedback_v1`
  - this matters because the sim/synth WM is now closer to the Phase 1 requirement that it own branch-to-training feedback and receipt emission in the live loop, not just gap-ranking and branch proposal state

- Backend routing is now explicit and honest:
  - `src/world_model/sim_synth_physics/backend_router.py` preserves the difference between:
    - requested backend
    - resolved backend
    - route status (`ready`, `fallback`, `blocked`)
  - `isaac` is now treated as an explicit adapter gap with typed fallback metadata rather than an implicit generic backend name
  - `holosoma` is treated as an external execution provider whose local availability is checked honestly at runtime
  - this is still not "full Isaac/Unitree functionality", but it is the right in-phase posture: explicit typed ownership of the gap, not silent fallback

- Phase 1 input normalization is less ad hoc now:
  - `src/world_model/sim_synth_physics/adapters/economic_inputs.py` and `src/world_model/sim_synth_physics/adapters/embodiment_inputs.py` now normalize urgency, value-target, capability, control-constraint, and latency/contact signals before they enter WM canonical state
  - this begins the actual module-boundary separation the architecture doc called for, instead of leaving economic and embodiment context as raw dict passthroughs

- Orchestrator ownership is thinner now, which is the intended direction:
  - `src/orchestrator/semantic_simulation.py` now asks the WM runtime for the legacy agenda view instead of building it itself
  - `src/orchestrator/diffusion_requests.py` now goes through the WM runtime boundary to obtain world state and diffusion plans before adapting them into prompt specs
  - this is not the end of Phase 1 ownership transfer, but it is the correct direction: orchestrator files become adapters/clients, not alternate owners

- New WM-owned scripts now exist for the next tranches:
  - `scripts/compile_sim_synth_physics_plan.py`
  - `scripts/run_sim_synth_physics_loop.py`
  - these scripts provide a canonical plan/loop entrypoint for Phase 1 instead of relying only on older orchestrator-side or ad hoc script flows

- Architecture doctrine tightened:
  - `docs/economic_world_model/multi_wm_architecture_plan.md` now includes a cross-phase "phase exit rule":
    - do not move to the next phase while the current phase still has implementable ownership/runtime/adapter/package gaps
    - only move when the remaining blockers are primarily data/GPU/asset/calibration/benchmark constraints
  - Phase 1 now explicitly targets real Isaac Sim / Isaac Gym / Unitree-class adapter functionality behind typed backend routing rather than letting PyBullet fallback become the de facto end state

- Synthetic branch generation is less script-owned now:
  - `src/world_model/sim_synth_physics/synthetic_branches.py` now owns:
    - local branch rollout/gating helpers over the stable world model
    - coverage-gap labeling for collected branches
    - compilation of WM synthetic branch plans
    - canonical branch corpus metadata construction
  - `scripts/collect_local_synthetic_branches.py` still loads the world model, trust-net, and source datasets, but the script is now primarily a worker over WM-owned branch logic rather than the owner of those rules
  - this is a meaningful Phase 1 shift because the architecture plan explicitly called out local synthetic branch generation as something that should stop being script-owned

- Gen2sim admission is also more honestly inside the WM boundary now:
  - `src/world_model/sim_synth_physics/gen2sim_admission.py` now owns:
    - compilation of `Gen2SimAdmissionState` for live WM planning
    - local synthetic branch corpus gen2sim assessment rows and summaries
  - `src/world_model/sim_synth_physics/compiler.py` now delegates admission-state compilation into that module instead of open-coding the logic inline
  - the lower-level evidence helper in `src/evidence/gen2sim_validity.py` still remains the reusable scoring engine, but it is no longer acting as the architectural owner of the branch-admission flow

- Practical consequence for Phase 1 sequencing:
  - there is still no excuse to move to Phase 2 yet
  - the explicit remaining Phase 1 work is still implementable and still in-bounds:
    - real Isaac Sim / Isaac Gym backend adapter work
    - Unitree-class sim-env integration
    - richer Holosoma execution integration
    - domain-randomization / system-ID policy and receipts
    - NAG / LSD / GGDS productionization
  - only after those are pushed as far as the repo can honestly push them should the phase be considered blocked mainly by data, GPU, asset, or benchmark limits

- Phase 1 backend ownership is now structurally better aligned with the stated target posture:
  - `src/world_model/sim_synth_physics/backend_adapters.py` turns backend names into explicit adapter descriptors with simulator family, target hardware class, execution envelope, and fallback truth
  - the Isaac path is still honestly non-executable, but it now carries explicit Unitree-target metadata and required-asset expectations instead of hiding as a generic backend token
  - Holosoma is now described as a Unitree-class external execution adapter behind the WM boundary, not just as an adjacent backend module

- Phase 1 backend execution binding is now a real state surface, not just metadata:
  - `src/world_model/sim_synth_physics/adapters/backend_pybullet.py`, `backend_holosoma.py`, and `backend_isaac.py` encode the actual runtime entrypoints and asset expectations for each backend family
  - `src/world_model/sim_synth_physics/backend_bindings.py` compiles those into `BackendExecutionBindingState`
  - the loop now emits `backend_execution_binding_receipt_v1`, so runtime stack, observation adapter, asset profile, and missing-asset truth are canonical artifacts rather than inference from backend names
  - this is especially important for the Unitree-target Isaac path, because the repo can now say “binding exists but assets are missing” instead of collapsing that state into one generic fallback string

- Phase 1 now has a typed physics-adaptation layer rather than ad hoc randomization metadata:
  - `PhysicsAdaptationPolicyState` carries domain-randomization profile, system-identification profile, robot-asset profile, randomization axes, and calibration targets
  - `physics_adaptation_receipt_v1` is emitted in the live WM loop so downstream training and readiness logic can see whether the loop is still tabletop-oriented, humanoid-shadow-oriented, or closer to benchmark-ready adaptation posture
  - this closes one of the explicit Phase 1 gaps from the plan: domain randomization and system identification are no longer just an aspirational note

- NAG / LSD / GGDS are now wrapped by a WM-owned provider seam:
  - each `SyntheticBranchPlan` now carries a `BranchRenderProviderState`
  - the WM emits `render_provider_receipt_v1` artifacts and threads provider kind/status into diffusion routing and training-corpus extraction
  - the provider seam is now richer than selection metadata: it also carries materialization entrypoints and provider config payloads for NAG counterfactual generation and GGDS scene texturing
  - this still does not mean GGDS is “done”; the concrete optimizer is still stub-only without a real LDM + renderer stack
  - the important architectural change is that branch/render ownership now sits inside the WM boundary, so the remaining work is about making those providers concrete rather than first creating a canonical contract

- Honest remaining Phase 1 blocker statement after this pass:
  - we still should not leave Phase 1
  - the remainder is now dominated by concrete execution assets and GPU/provider reality:
    - Isaac Sim / Isaac Gym execution
    - Unitree robot assets and sim-env bindings
    - richer Holosoma runtime binding
    - concrete GGDS/LDM materialization
    - grounded GPU-backed perception-conditioned sim

- Phase 1 backend-runtime execution is now less fake around whole-body training:
  - `src/world_model/sim_synth_physics/backend_runtime_execution.py` can now use Holosoma in two honest modes:
    - `evaluate_policy(...)` when a runtime policy id exists
    - `train_policy(...)` when the runtime exists and the WM has motion datapacks / direct motion clips but no policy id yet
  - the important doctrine change is that we no longer pretend “missing policy id” blocks the lane when a trainable motion-source contract is actually present
  - `runtime_training_completed` now propagates into runtime evidence and calibration/adaptation receipts, so the train path counts as concrete loop evidence rather than a side note

- Unitree-target asset contracts are now more canonical and less manifest-shaped:
  - `src/world_model/sim_synth_physics/asset_manifest.py` normalizes humanoid asset aliases into canonical requirements such as:
    - `unitree_robot_description`
    - `whole_body_joint_map`
    - `camera_extrinsics`
    - `imu_extrinsics`
    - `force_torque_calibration`
    - `actuator_latency_profile`
    - `joint_limit_profile`
    - `safety_watchdog_profile`
  - `adapters/backend_isaac.py` now bases backend readiness on those canonical requirements instead of a thin four-key manifest
  - `asset_contracts.py` now unions backend-specific requirements with hardware-specific Unitree requirements, which makes the contract more honest for humanoid readiness and prevents “backend binding says one thing, hardware contract silently expects more” drift
  - `shadow_execution.py` and backend runtime bindings now preserve normalized asset-manifest state in backend-local sidecars so later Isaac/Unitree bring-up can use the same typed contract rather than inventing a parallel asset checklist

- Practical Phase 1 consequence after this tranche:
  - the remaining backend gap is increasingly not “the WM cannot express or route the runtime”
  - it is “the host/runtime/assets/policies are not present yet”
  - that is the right direction for this phase

- Backend-runtime target discovery is now explicit:
  - `src/world_model/sim_synth_physics/runtime_targets.py` describes the external runtime roots the WM is actually waiting on for each backend family
  - for Isaac/Unitree this now includes:
    - Isaac Lab / Isaac Sim / Unitree RL Gym / HumanoidVerse-style roots
    - Unitree SDK2 root
    - Unitree asset root
    - local Python bridge availability
  - for Holosoma this now includes:
    - Holosoma root
    - Holosoma motion root
    - Holosoma policy root
    - retargeting root
    - local Python bridge availability
  - these runtime-target contracts are now propagated into backend binding metadata and backend runtime-request sidecars, which makes the remaining Phase 1 blocker state much more operational

- GPU bring-up queues are now explicitly split:
  - `scripts/FOUNDATION_MODEL_BRINGUP_BACKLOG.json` remains the model/provider bring-up and fine-tune inventory
  - `scripts/TRAINING_MIGRATION_BACKLOG.json` remains the training migration queue
  - `scripts/NON_TRAINING_GPU_RUN_BACKLOG.json` is now the dedicated runtime/materialization smoke queue for:
    - Isaac/Unitree runtime smoke
    - Holosoma runtime eval smoke
    - Stage-1 video diffusion materialization
    - OpenVLA teacher runtime
    - non-stub vision backbone runtime
    - V-JEPA 2 runtime
    - real SAM3D workcell grounding refresh
  - `src/orchestrator/non_training_gpu_run_backlog.py` and `scripts/scan_non_training_gpu_run_backlog.py` make that queue evaluable with the same typed precondition logic as the broader loop-run backlog

- External runtime launch completion is no longer the terminal truth for Phase 1 backend bring-up:
  - `src/world_model/sim_synth_physics/runtime_outcomes.py` introduces two new canonical surfaces:
    - `backend_runtime_output_contract_v1`
    - `backend_runtime_outcome_receipt_v1`
  - the contract is built from the existing runtime bundle / launch spec and the output receipt is emitted after harvesting upstream outputs from the chosen runtime profile
  - current upstream-shaped output conventions are defined for:
    - `unitree_sim_isaaclab`
    - `unitree_rl_gym`
    - `HumanoidVerse`
    - `xr_teleoperate`
    - Holosoma repo / motion bank / policy bank / retargeting bundle
  - this is deliberately not overfit to one repo; it is a typed output contract that can absorb current upstream layouts while preserving WM ownership of the resulting truth

- The new output receipt now changes live WM behavior rather than sitting as another sidecar:
  - `backend_runtime_execution.py` threads output contract + output receipt into runtime execution metadata and artifact emission
  - `runtime.py` now exposes `backend_runtime_outcome_receipt` in the canonical loop result, loop summary, and artifact set
  - `runtime_evidence.py` now exposes:
    - `runtime_output_status`
    - `runtime_output_harvested`
    - `runtime_output_artifact_count`
    - `runtime_output_artifact_kinds`
  - `calibration.py` now gives bounded credit to harvested upstream runtime evidence even when execution happened out-of-process rather than in the local Python bridge
  - `runtime_work_orders.py` now allows `satisfied_by_external_runtime_outcomes` when the bridge is still shadow-authority but the upstream runtime produced harvestable outputs
  - `training_corpus.py` now preserves external-runtime outcome ids/status/counts for backend-selector and branch-planner corpora

- The runtime-root / asset / policy posture is now less flat and more upstream-realistic:
  - `runtime_layouts.py` now exposes deploy, policy, and data candidates per profile rather than only “root exists / root missing”
  - `runtime_targets.py` now names more optional Unitree-adjacent runtime surfaces:
    - `unitree_sdk2_python_root`
    - `teleimager_root`
    - `unitree_il_lerobot_root`
  - `runtime_bundles.py` now carries the output contract so the standalone launch path and full WM runtime share the same external-runtime expectations
  - `scripts/run_phase1_runtime_launch.py` can now harvest upstream outcomes and emit a standalone `backend_runtime_outcome_receipt_v1`

- Honest Phase 1 state after this tranche:
  - the WM can now represent:
    - launch prepared but not executed
    - launch executed with no outputs harvested
    - launch executed with outputs harvested
  - that means the remaining backend gap is increasingly about actual external runtime/assets/GPU availability rather than missing canonical WM receipt plumbing
## 2026-04-01

- The Isaac/Unitree executable-adapter lane now has an explicit consumer surface, not just a request surface:
  - added `src/world_model/sim_synth_physics/adapters/isaac_unitree_executable_consumer.py`
  - the consumer tells the WM whether the request is being handed to:
    - a local python bridge
    - an external sim launch consumer
    - an external teleop bridge
    - an external LeRobot eval consumer
  - it also preserves remaining preconditions instead of letting the request look “consumed” by default

- The important topology gain is:
  - request and consumer are now separate typed artifacts
  - that means the WM can say:
    - “here is the executable-adapter request”
    - “here is the consumer currently responsible for that request”
    - without pretending that either one is already the final real Unitree runtime adapter

- The runtime path now uses that distinction:
  - `runtime_bundles.py` emits both request and consumer
  - `runtime_launch.py` uses the consumer to drive launch mediation
  - `backend_runtime_execution.py` now writes the consumer into runtime artifacts and receipt metadata
  - `scripts/run_isaac_unitree_executable_adapter.py` exposes the pair end to end

- Why this matters:
  - it removes another place where execution mediation could hide behind generic launch semantics
  - it also makes the next Phase-1 cut clearer: the remaining missing piece is a real adapter implementation over this surface, not a lack of typed consumer structure

- The next concrete Isaac/Unitree executable-adapter surface is now inside the WM runtime path, not just implied by launch strings:
  - added `src/world_model/sim_synth_physics/adapters/isaac_unitree_executable_adapter.py`
  - `build_backend_runtime_bundle(...)` now emits `backend_executable_adapter_request_v1` for Isaac/Unitree runtime artifacts
  - the request includes:
    - deployment mode and adapter entrypoint
    - robot variant / placement class
    - required target ids and required asset ids
    - normalized asset refs
    - calibration / observation / action contracts
    - output expectations
    - environment overrides and remaining preconditions
  - this is a real Phase-1 improvement because the executable lane is now a typed subsystem surface rather than just a command template

- The launch layer now actually consumes that adapter request:
  - `runtime_launch.py` merges executable-adapter env overrides, preconditions, and notes into launch preparation
  - launch receipts now preserve `executable_adapter_request` in metadata
  - `scripts/run_isaac_unitree_executable_adapter.py` gives the lane a dedicated runnable surface over the WM artifacts without inventing a separate orchestration path

- Why this matters:
  - it removes another “looks real but only in strings” boundary
  - it keeps the remaining Unitree gap honest: the missing piece is increasingly the real upstream runtime/assets/GPU path, not a lack of typed executable-adapter structure in the repo
  - it also gives the future concrete adapter a clear contract to consume once the upstream runtime is present

- Branch-truth reconciliation for the multi-WM program is now explicit:
  - the master committed docs (`multi_wm_architecture_plan.md`, `roadmap.md`) were already the main source of truth
  - this pass landed the previously local-only supporting doctrine/spec/collaboration files so the branch now explicitly carries:
    - a neuralization/semantic-bridge doctrine
    - an active Sim/Synth/Physics closure tranche spec
    - a held Perception/Grounding schema tranche spec
    - a Codex/Claude collaboration doctrine
  - `CLAUDE.md` now references `.agent/claude_copilot.md`, which means the collaboration posture is no longer merely implied by local state
  - `docs/economic_world_model/claude_to_comment_on.md` should now be treated as a real per-tranche handoff artifact, not a dormant template

- The next concrete Isaac/Unitree Phase 1 improvement is now landed:
  - `src/world_model/sim_synth_physics/adapters/isaac_unitree_deployment.py` defines a Unitree-aware deployment contract over:
    - runtime profiles
    - sim/teleop/lerobot/physical deployment modes
    - policy readiness
    - robot-asset readiness
    - missing preconditions per mode
  - `backend_isaac.py` now emits this contract in binding metadata and uses it to distinguish:
    - `runtime_ready`
    - `external_launch_ready`
    - `external_launch_assets_missing`
    - legacy shadow/asset-gap states
  - this is materially better than the old posture where Isaac could only be “runtime-ready” or some generic shadow state; the WM can now express the intermediate but important truth that external launch is structurally ready even if in-process local execution is not

- The runtime/bundle/bridge path now consumes that deployment contract:
  - `runtime_targets.py` now accepts the `unitree_lerobot_root` alias and treats `unitree_il_lerobot_root` and `xr_teleoperate_root` as valid members of the external-runtime root family
  - `runtime_layouts.py` now understands a `unitree_lerobot` runtime profile
  - `runtime_bundles.py` now:
    - supports `unitree_lerobot` launch specs
    - uses deployment-contract preferred-profile ordering instead of relying only on generic runtime-layout ordering
    - preserves the deployment contract inside runtime bundles and launch specs
  - `runtime_bridge.py` now:
    - recognizes `unitree_xr_teleop_bridge` and `unitree_lerobot_eval_bridge`
    - preserves deployment-contract metadata in the bridge receipt path
    - treats `external_launch_ready` / `external_launch_assets_missing` as first-class binding states instead of collapsing them into the older shadow/runtime buckets
  - `runtime_launch.py` now exports the corresponding environment variables for `UNITREE_SDK2_PYTHON_ROOT`, `TELEIMAGER_ROOT`, and `UNITREE_IL_LEROBOT_ROOT`
  - `runtime_outcomes.py` now has an explicit `unitree_lerobot` upstream output family, so harvested outputs are no longer hard-coded to IsaacLab / RL Gym / HumanoidVerse / XR Teleoperate only
  - `backend_runtime_execution.py` now passes the deployment contract into runtime-bundle construction, making the runtime-launch path topology-consistent with the binding metadata

- A real bug was fixed while landing the deployment contract:
  - `unitree_policy_root` was incorrectly making the runtime layer treat `unitree_rl_gym` as “profile-ready” even when only a policy bank was present
  - this was removed from the runtime-root → profile mapping, so policy-bank availability no longer masquerades as runtime-root availability
  - this is exactly the sort of subtle truthiness bug we want these contracts to prevent

- New focused test coverage now exists for this tranche:
  - `tests/test_isaac_unitree_deployment.py`
  - additions in:
    - `tests/test_sim_synth_runtime_targets.py`
    - `tests/test_sim_synth_runtime_layouts.py`
    - `tests/test_sim_synth_runtime_bundles.py`
    - `tests/test_sim_synth_physics_world_model.py`
  - the tests now explicitly exercise:
    - sim launch readiness
    - teleop readiness
    - lerobot-eval readiness
    - physical deployment missing-precondition logic
    - deployment-driven profile preference
    - XR teleop / sdk2_python / teleimager transport-bridge posture

- The next concrete Isaac/Unitree runtime-execution layer is now in place:
  - `src/world_model/sim_synth_physics/adapters/isaac_unitree_adapter_execution.py` now sits between the executable-adapter consumer and the launch/outcome path
  - that layer deliberately does not pretend to be the final adapter; instead it makes the next maturity rung explicit by distinguishing:
    - request
    - consumer
    - adapter execution mediation
    - launch
    - harvested runtime outcome
  - the main statuses it names are:
    - `local_bridge_ready`
    - `local_bridge_missing`
    - `local_bridge_handed_off`
    - `external_launch_ready`
    - `external_launch_completed`
    - `external_launch_failed`

- The live Phase-1 runtime path now preserves that mediation explicitly:
  - `backend_runtime_execution.py` now writes:
    - `backend_runtime_adapter_execution.json`
    - `backend_runtime_adapter_receipt.json`
  - the runtime metadata now carries `executable_adapter_request`, `executable_adapter_consumer`, `adapter_execution`, and `adapter_receipt` together so the lane no longer jumps directly from consumer choice to generic launch status
  - `runtime.py` now surfaces `backend_runtime_adapter_receipt` as a first-class loop artifact and carries its id/status/execution-path into:
    - loop summaries
    - training feedback
    - outcome metadata

- The downstream training/export path now sees the new truth as well:
  - `training_corpus.py` now harvests `backend_runtime_adapter_receipt_v1`
  - backend-selector and branch-planner rows now preserve:
    - adapter receipt id
    - adapter status
    - adapter execution path
  - this matters because launch completion alone is not enough to tell whether the lane merely prepared an external launch, really executed one, or tried to hand off to a missing local bridge

- The standalone WM-facing runner now exposes the same maturity split:
  - `scripts/run_isaac_unitree_executable_adapter.py` now emits:
    - `executable_adapter_request`
    - `executable_adapter_consumer`
    - `adapter_execution`
    - `adapter_receipt`
    - the existing launch `result` and `receipt`
  - that keeps the standalone Isaac/Unitree lane topology-consistent with the live runtime path

- New focused tests for this sub-tranche:
  - `tests/test_isaac_unitree_adapter_execution.py`
  - additions in:
    - `tests/test_sim_synth_runtime_launch.py`
    - `tests/test_sim_synth_physics_world_model.py`
    - `tests/test_sim_synth_training_corpus.py`

- The next concrete local-runtime cut is now landed too:
  - `src/world_model/sim_synth_physics/adapters/isaac_unitree_adapter_realization.py` now defines how the current branch concretely realizes the Isaac/Unitree lane after execution mediation
  - it deliberately names the current real options instead of pretending a finished hardware adapter already exists:
    - `local_backend_factory`
    - `external_launch_delegate`
    - blocked realization
  - that means the branch can now say not just “the adapter is ready” but “the adapter is realized today through local backend-factory handoff” or “it is only realized through an external delegate path”

- The live runtime path now preserves that realization explicitly:
  - `backend_runtime_execution.py` rebuilds the realization after adapter-execution finalization so the realization follows the latest mediation truth
  - `runtime.py` writes `backend_runtime_adapter_realization.json` as a root-level loop artifact
  - the training-feedback and loop-summary path now carry adapter realization status/path in addition to adapter execution status/path

- Downstream trainer/export rows now preserve the new truth:
  - `training_corpus.py` now emits:
    - `backend_runtime_adapter_realization_path`
    - `backend_runtime_adapter_realization_status`
  - this matters because “external delegate” and “local backend factory” are both more concrete than generic launch readiness, but they are still meaningfully different readiness states

- The standalone WM-facing runner now emits the realization surface too:
  - `scripts/run_isaac_unitree_executable_adapter.py` now includes `adapter_realization` beside request / consumer / execution / receipt / launch result

- New focused tests for this realization tranche:
  - `tests/test_isaac_unitree_adapter_realization.py`
  - additions in:
    - `tests/test_sim_synth_runtime_launch.py`
    - `tests/test_sim_synth_physics_world_model.py`
    - `tests/test_sim_synth_training_corpus.py`

- The next Phase 1 runtime-materialization cut now closes one more fake seam:
  - `local_backend_factory_adapter.py` makes explicit local adapter materialization a typed invocation/result surface instead of letting it hide inside a direct backend-factory jump
  - this matters because “realized locally” and “still only contract-shaped” are no longer collapsed together once the executable-adapter ladder reaches realization

- Holosoma is now much closer to structural parity with Isaac/Unitree:
  - it now has request / consumer / adapter-execution / adapter-realization surfaces
  - those surfaces are emitted in runtime bundles and preserved into backend runtime execution receipts
  - local train-from-motion now correctly clears `policy_checkpoint` when no policy is the honest bounded mode

- The remaining Phase 1 gap is therefore narrower and more honest:
  - Isaac/Unitree still needs the actual upstream runtime/assets/policies behind the new materialization surfaces
  - Holosoma still needs the actual host/runtime/motion/policy/retargeting assets behind the same surfaces
  - the remaining blockers are increasingly external runtime, asset, GPU, and benchmark-density issues rather than missing typed loop plumbing

- The next backend-specific closure tranche is now landed:
  - Isaac/Unitree has an explicit `upstream_runtime_pack` surface over runtime profiles, ready targets, policy-bank surfaces, deploy surfaces, telemetry surfaces, and asset refs
  - Holosoma now has:
    - a real deployment contract (`sim_eval`, `motion_train`, `retarget_eval`)
    - an explicit upstream runtime pack over runtime roots, motion surfaces, retargeting surfaces, reward-overlay posture, and telemetry surfaces
- This matters because the branch can now distinguish:
  - backend binding readiness
  - deployment-mode readiness
  - upstream runtime-pack readiness
  - executable-adapter request / consumer / execution / realization / launch / outcome
  without collapsing those into one backend status bit
- Downstream Phase 1 consumers now preserve that truth:
  - runtime bundles carry `upstream_runtime_pack`
  - runtime-bridge receipts preserve it
  - runtime work orders now inherit pack status and missing components
  - backend-selector / branch-planner corpus rows now preserve pack status, ready surfaces, and missing components
- The `scan_phase1_runtime_layouts.py` script now emits the same deployment and runtime-pack view, so Phase 1 scanning no longer stops at “roots/layouts/policy contract”
- Honest remainder after this tranche:
  - the repo now knows how to describe backend-specific upstream runtime packs, but those packs are still provider-owned/external reality
  - the next concrete work is still real runtime/assets/policies/hosts, not another speculative abstraction layer

- The next concrete Phase-1 closure rung is now explicit:
  - `isaac_unitree_runtime_binding.py` and `holosoma_runtime_binding.py` sit between upstream runtime packs and executable-adapter requests
  - that layer deliberately chooses mode-relevant policy / motion / retargeting / launch / target surfaces instead of inheriting every pack-level missing component indiscriminately
  - the runtime lane can now say:
    - which selected surfaces are actually bound for this mode
    - which missing components are still relevant for this mode
    - whether the lane is `binding_ready`, `binding_partial`, or `binding_blocked`

- This matters topologically because the backend path is now:
  - backend binding
  - deployment contract
  - upstream runtime pack
  - runtime binding
  - executable-adapter request
  - executable-adapter consumer
  - adapter execution
  - adapter realization
  - local materialization / external launch
  - harvested runtime outcomes

- The most important bug fixed in this tranche was not cosmetic:
  - local Holosoma concrete execution was still inheriting pack-level `sim_eval` blockers even when the branch had enough to do honest local eval or train-from-motion
  - specifically, the `motion_train` patch path in `backend_runtime_execution.py` was mutating a stale `sim_eval` request in place, which left irrelevant `policy_surface` / `policy_checkpoint` blockers alive
  - the branch now rebuilds the Holosoma executable-adapter request from the patched runtime binding when `motion_train` is the honest local mode

- Consequences of the fix:
  - local Holosoma eval can proceed when the branch has:
    - a real local runtime bridge
    - an explicit policy ref
  - local Holosoma train-from-motion can proceed when the branch has:
    - a real local runtime bridge
    - motion datapacks and/or inline clips
  - absent external repo roots or launch surfaces no longer masquerade as local-runtime blockers in those two cases

- The runtime-binding layer is now preserved end to end:
  - `runtime_bundles.py` writes `backend_runtime_binding.json`
  - `runtime_launch.py` uses runtime-binding-selected launch/root/command surfaces and missing components
  - `runtime_work_orders.py` now reports `runtime_binding_status` plus binding-selected profile/policy/root state
  - `runtime.py` now carries runtime-binding refs/status into loop summaries and training feedback
  - `training_corpus.py` now exports runtime-binding status and selected surfaces into backend-selector / branch-planner rows
  - `scan_phase1_runtime_layouts.py` now emits runtime bindings for both Isaac/Unitree and Holosoma scans

- Focused verification for this tranche:
  - `python3 -m compileall src/world_model/sim_synth_physics scripts/scan_phase1_runtime_layouts.py scripts/run_isaac_unitree_executable_adapter.py tests/test_isaac_unitree_runtime_binding.py tests/test_holosoma_runtime_binding.py tests/test_scan_phase1_runtime_layouts.py tests/test_sim_synth_runtime_bundles.py tests/test_sim_synth_runtime_launch.py tests/test_sim_synth_physics_world_model.py tests/test_sim_synth_training_corpus.py tests/test_sim_synth_runtime_work_orders.py -q`
  - `python3 -m ruff check src/world_model/sim_synth_physics scripts/scan_phase1_runtime_layouts.py scripts/run_isaac_unitree_executable_adapter.py tests/test_isaac_unitree_runtime_binding.py tests/test_holosoma_runtime_binding.py tests/test_scan_phase1_runtime_layouts.py tests/test_sim_synth_runtime_bundles.py tests/test_sim_synth_runtime_launch.py tests/test_sim_synth_physics_world_model.py tests/test_sim_synth_training_corpus.py tests/test_sim_synth_runtime_work_orders.py`
  - `python3 -m pytest -q tests/test_isaac_unitree_runtime_binding.py tests/test_holosoma_runtime_binding.py tests/test_scan_phase1_runtime_layouts.py tests/test_sim_synth_runtime_bundles.py tests/test_sim_synth_runtime_launch.py tests/test_sim_synth_physics_world_model.py tests/test_sim_synth_training_corpus.py tests/test_sim_synth_runtime_work_orders.py`
  - result: `44 passed`

- Honest remainder after this tranche:
  - the new binding layer is real, but it is still binding against provider-owned/external runtime packs
  - the next high-leverage work remains:
    - real Isaac/Unitree runtime/assets/policies behind the ladder
    - real Holosoma host/runtime/motion/retargeting assets behind the same ladder
    - GPU-backed GGDS / video materialization

- The next audited Phase-1 closure fix was in the concrete-runtime evidence path, not in a new abstraction layer:
  - `runtime_outcomes.py` can now harvest explicit local runtime artifacts and emit `backend_runtime_outcome_receipt_v1` without depending on a launch receipt
  - the receipt metadata now carries `harvest_mode=local_runtime_execution` when the evidence came from a concrete local runtime rather than an external launch handoff
  - this matters because local concrete runtime success should not be flattened back into launch-shaped semantics once the branch already has direct policy / rollout / metrics outputs

- `runtime_outcome_parsers.py` was tightened for the same reason:
  - rollout `trajectory` artifacts and `episode_*` captures are now classified as dataset surfaces before generic motion-dataset fallbacks
  - this prevents local runtime rollout evidence from undercounting dataset-ready trainer/replay material when the path is concrete but still local

- `backend_runtime_execution.py` now makes that outcome truth load-bearing:
  - after successful concrete local runtime eval/train, it explicitly harvests policy / metrics / rollout outputs
  - it writes:
    - `backend_runtime_output_contract.json`
    - `backend_runtime_output_summary.json`
    - `backend_runtime_outcome_receipt.json`
  - it also threads the outcome receipt back into runtime-execution metadata so the loop can distinguish:
    - launch-shaped external execution
    - concrete local runtime execution with harvested outputs

- The Isaac/Unitree local bridge lane needed one more honesty fix:
  - `isaac_unitree_runtime_binding.py` now treats local `sim_eval` as a narrower local-binding mode, so stale upstream-pack placeholders like `preferred_runtime_profile`, `runtime_profile_surface`, and generic `runtime_profile` no longer block a lane that already has a real local bridge plus the actual local prerequisites it needs
  - `isaac_unitree_executable_adapter.py` now uses binding-selected missing components as the primary request truth and only supplements them with still-missing required assets or policy state
  - effect: a concretely executable local Isaac path is no longer marked blocked before it even reaches explicit local backend materialization

- The new tests are intentionally topology-specific:
  - `tests/test_sim_synth_runtime_outcomes.py` now verifies explicit local-runtime artifact harvest without a launch receipt
  - `tests/test_sim_synth_physics_world_model.py` now verifies:
    - concrete local Isaac runtime through the local bridge / backend-factory path
    - concrete local Holosoma eval/train paths preserving `backend_runtime_outcome_receipt`
    - local runtime outcome receipts marking policy / dataset / metrics surfaces ready when those artifacts were actually emitted

- The result is another honest narrowing of Phase 1:
  - internal gap removed: local concrete runtime no longer degrades into launch-shaped truth
  - internal gap removed: local Isaac bridge no longer inherits irrelevant external-pack blockers
  - remaining blockers are more decisively external:
    - real upstream runtime/assets/checkpoints
    - real host/runtime installs
    - GPU-backed model/materialization availability

- The next closure step stayed in that same Phase-1 lane and made the upstream runtime/assets/checkpoints more concrete without inventing a new layer:
  - `runtime_layouts.py` now exposes profile-level evidence:
    - candidate counts
    - primary refs
    - git metadata when the runtime root is a real local clone
  - Isaac and Holosoma policy contracts now expose:
    - `primary_checkpoint_ref`
    - `primary_deploy_config_ref`
    - `primary_runtime_report_ref`
    - candidate-record inventories and counts

- This matters because “root exists” and “there are some candidates somewhere under it” are not enough for Phase 1 closure:
  - the WM can now point at the specific checkpoint / deploy-config / runtime-report surface it intends to use
  - work orders and trainer rows can now say what the selected upstream evidence actually was
  - the branch no longer needs to rediscover those refs downstream from raw candidate lists

- Isaac also now carries a more honest asset posture:
  - `asset_manifest.py` still preserves declared manifest presence, but it now records local verification status for path-like asset refs
  - `isaac_unitree_runtime_pack.py` distinguishes:
    - declared asset ids
    - locally verified asset ids
    - declared-only asset ids
  - this is important because a manifest key alone should not be the only signal of hardware/sim readiness

- Holosoma now does the analogous thing for motion inputs:
  - the runtime pack distinguishes motion sources that actually exist locally from motion sources that are only named
  - that makes the motion-train lane more honest in the same way the Isaac asset lane is becoming more honest

- Those evidence surfaces are now load-bearing downstream:
  - `runtime_work_orders.py` preserves selected primary refs plus profile evidence
  - `training_corpus.py` preserves:
    - upstream profile root
    - upstream profile git metadata
    - candidate counts
    - selected primary policy/deploy/report refs
    - verified-vs-declared Isaac asset truth
    - existing Holosoma motion-source truth

- The practical effect is that the remaining blocker is more decisively external:
  - the branch now knows much more specifically which runtime roots/checkpoints/assets it would use
  - if the next step still fails, it is increasingly because those upstream repos/assets/checkpoints are not actually present or usable, not because Phase 1 lacked a typed way to name them

- Phase 1 install/preflight evidence now sits one level deeper in the same runtime lane rather than creating a new rung:
  - `runtime_layouts.py` now distinguishes root/candidate truth from install-shape truth on each runtime profile
  - that install-shape truth is profile-local and includes:
    - entrypoint-path expectations
    - primary entrypoint ref
    - install-preflight status
    - verified vs missing install components
- The important topology correction in this tranche was selected-profile resolution:
  - upstream runtime packs can still prefer one profile while a binding honestly selects another
  - because of that, install-preflight truth cannot stay only at the pack-preferred profile
  - both Isaac/Unitree and Holosoma packs now carry `profile_install_by_id`
  - bindings now resolve install-preflight against the actual selected profile before computing `runtime_profile_surface`, missing components, and host-preflight
- That fixed a real false-readiness / false-blocker pair:
  - partially discovered Isaac profiles can still be selected as the best local upstream profile even when deployment-level `preferred_profile` stays empty
  - Holosoma `motion_train` no longer inherits `holosoma_repo` install blockers when the selected local mode is `holosoma_motion_bank`
- Downstream preservation was kept aligned:
  - `runtime_work_orders.py` now carries profile install status, selected primary entrypoint refs, and selected install missing-components
  - `training_corpus.py` now exports the same install/preflight truth into backend-selector and branch-planner rows
- Closure interpretation after this tranche:
  - no new internal ladder was added
  - no new fake readiness was introduced
  - the remaining honest blocker stays external:
    - real local Isaac/Unitree installs/assets/checkpoints
    - real local Holosoma runtime/motion/policy/retargeting assets
    - real GPU-backed materialization

- Non-GPU host-consumption pass:
  - Public Isaac/Unitree and Holosoma repos are now present under `/Users/amarmurray/code`, and the existing Phase-1 scan path is consuming them as real local evidence instead of treating this host as empty.
  - The Holosoma lane had a real internal incompleteness: motion, policy, and retargeting surfaces existed inside a real local repo, but the branch only knew how to consume them if they were restated as separate top-level roots. That is now fixed by deriving those subroots directly from the repo when present.
  - Holosoma policy selection also had a real false-readiness seam: generic `**/*.pt` matching could select retargeting demo artifacts as runtime policy. The candidate patterns now prefer actual model/checkpoint surfaces instead.
  - The Isaac lane exposed a second internal issue once public local clones existed: autodiscovery could outrank an explicit caller-provided runtime profile. Deployment selection, runtime-bundle selection, and runtime-bridge transport selection now keep explicit context authoritative while still preserving autodiscovered evidence.
  - The host scan now demonstrates a useful non-GPU split:
    - Isaac/Unitree: real runtime roots, real target roots, real policy/runtime-report refs, still blocked on concrete asset-manifest/calibration/watchdog surfaces
    - Holosoma: repo-local runtime/model/motion/retargeting surfaces now materially visible and `host_preflight_ready`
  - This means Phase 1 can still progress without a GPU whenever real runtime roots/assets/checkpoints arrive; the GPU is now more clearly the bottleneck for actual execution/materialization, not for local evidence consumption.

- Additional late Phase-1 Unitree asset derivation:
  - `asset_manifest.py` now accepts the existing Isaac runtime-target contract so asset normalization can reuse already-selected/discovered roots instead of doing ambient host discovery.
  - That keeps the new behavior topology-conscious:
    - the Phase-1 scan benefits from public local roots immediately
    - backend binding / asset contracts / runtime materialization see the same truth
    - unrelated code paths do not silently become host-dependent
  - Derived asset selection currently prefers:
    - robot description:
      - `unitree_models` USD
      - then `unitree_rl_gym` / `HumanoidVerse` / `xr_teleoperate` robot descriptions
    - whole-body joint map:
      - `HumanoidVerse` `g1_29dof.yaml`
      - then `unitree_sim_isaaclab/robots/unitree.py`
      - then URDF fallback
    - joint limits:
      - `HumanoidVerse` `g1_29dof.yaml`
      - then public URDF limit surfaces
    - recommended contracts:
      - `control_frequency_profile` from `unitree_sim_isaaclab/sim_main.py`
      - `teleop_recovery_contract` from `xr_teleoperate` emergency-stop/damping surfaces
  - Explicit manifest entries still win when they are real, but missing local-path placeholders no longer outrank verified derived local files.
  - After rerunning the live host scan on this machine, the Isaac missing/preflight set dropped from five asset blockers to two:
    - remaining:
      - `asset::actuator_latency_profile`
      - `asset::safety_watchdog_profile`
  - I searched the local public repos specifically for latency/watchdog/safety artifacts after this pass:
    - there are public control-frequency and soft-emergency-stop signals
    - there is still no clean whole-body latency-contract or safety-watchdog artifact I would count as those required surfaces without overclaiming
  - That makes the remaining non-GPU Isaac asset gap much narrower and much more honestly external.

- Phase 2 implementation note: the first Perception / Grounding compiler tranche should reuse the existing semantic-world-model heuristic grounding as the initial heuristic backend rather than creating a second disconnected semantic heuristic stack. That is now the live posture in `src/world_model/perception_grounding/compiler.py`.

- Why this is the right first functional tranche:
  - it keeps lower-WM ownership inside the new Perception / Grounding WM
  - it compiles canonical state from real upstream sources already on disk
  - it keeps `SemanticVLA` transitional instead of pretending it is load-bearing
  - it makes the semantic bridge family start affecting real downstream consumers instead of remaining declaration-only

- Current functional shape:
  - upstream inputs:
    - scene tracks
    - belief state
    - VLA semantic evidence
  - canonical compiled outputs:
    - scene graph
    - temporal grounding
    - evidence routing
    - provider/dataset/task/deployment-resource surfaces
    - semantic bridge registry
  - downstream consumers:
    - Sim / Synth semantic context / inferential summary
    - rollout labeling / annotation metadata

- This is intentionally still `shadow_runtime`:
  - helper posture is typed and compiled
  - bridge outputs are functional and downstream-consumed
  - but there is no bounded runtime authority yet
  - provider/runtime truth still needs its own emitted receipt path

- Important residual internal work after this tranche:
  - emit live `ProviderAvailabilityReceipt`, `InferenceHeadroomReceipt`, and `DeploymentResourceReceipt` from the Perception compiler/runtime path instead of only carrying the typed state surfaces
  - compile provider/runtime inventory truth directly into Perception WM state rather than inferring only from payload presence
  - add the next consumer tranche so the bridge family expands beyond:
    - one Sim / Synth context consumer
    - one annotation consumer

- 2026-04-08 Sim / Synth / Physics doctrine note:
  - provider-family placement now lives in the existing owning docs rather than
    in new standalone notes:
    - `multi_wm_architecture_plan.md` owns Newton / UnrealRoboticsLab /
      WinDiNet / Habitat-style placement inside the 10-subsystem topology
    - `roadmap.md` owns the single integrated Phase 1.x follow-on
    - `actuation_embodiment_world_model.md` owns only the explicit
      Sim↔Embodiment transfer boundary
  - this intentionally avoids duplicating provider doctrine in Embodiment or
    transport docs

## Future sim-to-online stabilization tranche

This is future work, not the current repo bottleneck. The point of reserving it
now is to make later sim-to-online stabilization legible and typed without
rewriting the stack around real-robot finetuning prematurely.

- `src/training/sim_to_online/manifest.py`
  - ownership: training-manifest layer under the existing multi-WM topology
  - typed objects: `ReplayMixturePolicy`, `WarmStartPolicy`,
    `ActorCriticUpdateSchedule`, `SimOnlineTrainingWindow`
  - consumes: Sim / Synth / Physics transfer receipts and replay provenance,
    Embodiment deployment-side transfer receipts
  - emits: training-window manifests and provenance-bearing run metadata
  - why later: current priority is lower-WM structural and provider truth, not
    active real-robot online adaptation

- `src/training/sim_to_online/replay_mixture.py`
  - ownership: training-data composition policy, not WM truth ownership
  - typed objects: replay-mixture selectors and provenance-aware mixture specs
  - consumes: retained simulation data, retained prior real data, and online
    adaptation windows
  - emits: replay-mixture decisions / diagnostics tied to training windows
  - why later: only becomes honest once real online windows and transfer
    receipts exist

- `src/training/sim_to_online/checkpoint_receipts.py`
  - ownership: resume / restore integrity doctrine
  - typed objects: `CheckpointCompletenessReceipt`
  - consumes: checkpoint payloads, optimizer/target-network state, scheduler or
    entropy/temperature state where relevant
  - emits: completeness receipts and resume-risk summaries
  - why later: needed when there is an actual online adaptation loop to resume

- `src/training/sim_to_online/update_schedule.py`
  - ownership: future training-schedule policy for online adaptation
  - typed objects: `ActorCriticUpdateSchedule`
  - consumes: training-window context, transfer-stability evidence,
    embodiment-side deployment drift
  - emits: explicit update-schedule records and schedule-related diagnostics
  - why later: not a current repo-wide algorithm decision and should not become
    premature SAC doctrine

- `src/training/sim_to_online/transfer_stability.py`
  - ownership: transfer-stability evaluation layer
  - typed objects: `TransferStabilityReceipt`
  - consumes: `SimRealGapReceipt`, `PhysicsAdaptationReceipt`,
    `BackendMismatchReceipt`, `DeploymentTransferDriftReceipt`,
    `ActionFeasibilityDegradationReceipt`
  - emits: transfer-stability summaries for replay/training/economic consumers
  - why later: depends on both sim-side and embodiment-side transfer truth being
    real first

- `src/training/sim_to_online/window_runner.py`
  - ownership: asynchronous episodic real-hardware adaptation runner
  - typed objects: `SimOnlineTrainingWindow`, `OnlineAdaptationEpisodeReceipt`
  - consumes: bounded replay windows, checkpoint-completeness status, update
    schedules, transfer-stability state
  - emits: per-window and per-episode adaptation receipts
  - why later: this should come only after the stack has honest on-robot loop
    receipts and replay export discipline

## Future UE5 / Unreal provider-family tranche

This is future provider/runtime work, not a reason to pull implementation
priority away from the current lower-WM bottlenecks. The point of reserving
these targets now is to make UE5 legible as a bounded provider family inside
the Sim / Synth / Physics WM.

- `src/world_model/sim_synth_physics/providers/ue5_scene_provider.py`
  - ownership: Sim / Synth / Physics WM Scene / Asset / Materialization Layer
  - contracts: `UESceneMaterializationState`, `UEAssetContentContract`
  - consumes: `SceneHierarchyState`, asset manifests, branch scene intent,
    deployment-matched digital-twin refs
  - emits: scene-materialization state and asset-readiness receipts
  - honest blocker class: UE5 project/runtime layout, asset roots,
    photogrammetry/digital-twin inputs, GPU host when materialization becomes
    real

- `src/world_model/sim_synth_physics/providers/ue5_render_provider.py`
  - ownership: Render / Diffusion / Materialization Lane
  - contracts: `UEPhotorealRenderReceipt`, `UESimRealVisualAlignmentReceipt`
  - consumes: `BranchRenderProviderState`, branch plans, camera/sensor configs,
    realism targets
  - emits: photoreal render receipts, visual-alignment receipts, materialized
    artifact refs
  - honest blocker class: GPU host, UE5 runtime/headless render support,
    calibrated material/light profiles

- `src/world_model/sim_synth_physics/providers/ue5_sensor_provider.py`
  - ownership: Sim / Synth provider/runtime surface with downstream
    Perception/Embodiment consumers
  - contracts: `UESensorSimulationContract`, `UESensorSimulationReceipt`
  - consumes: scene/materialization state, sensor suite config, timing/noise
    policy, branch evaluation context
  - emits: simulated sensor bundles plus timing/noise/synchronization receipts
  - honest blocker class: plugin/runtime availability, calibrated sensor models,
    GPU host, middleware-specific sensor exporters

- `src/world_model/sim_synth_physics/providers/ue5_randomization_provider.py`
  - ownership: Fidelity / Randomization / Calibration Allocator
  - contracts: `UERandomizationPolicyState`, `UEPCGLayoutGenerationReceipt`
  - consumes: WM-owned randomization policy, coverage targets, scene/layout
    families, asset contracts
  - emits: randomized-scene receipts, PCG layout generation receipts, coverage
    metadata
  - honest blocker class: UE PCG/plugin availability, asset libraries,
    calibrated layout/randomization policies

- `src/world_model/sim_synth_physics/providers/ue5_digital_twin_ingest.py`
  - ownership: Scene / Asset / Materialization Layer with Task/Measurement
    consequences
  - contracts: `UEDigitalTwinIngestReceipt`,
    `UEDigitalTwinRegistrationContract`
  - consumes: photogrammetry, SLAM point clouds, LiDAR-supported
    reconstruction outputs, registration/pose exports
  - emits: digital-twin ingest receipts, deployment-matched regression
    environment refs, pose/registration provenance
  - honest blocker class: RealityScan/CLI or equivalent pipeline access, site
    captures, registration truth, Linux/remote-command runtime

- `src/world_model/sim_synth_physics/providers/ue5_middleware_bridge.py`
  - ownership: Backend / Runtime / Provider Surface
  - contracts: `UEMiddlewareBridgeContract`, `UEHybridBackendBindingState`
  - consumes: backend runtime bridge state, transport profile, ROS / ROS2 /
    gRPC posture, companion-compute assumptions
  - emits: bridge contracts, middleware readiness receipts, hybrid backend
    binding state
  - honest blocker class: ROS2/plugin availability, gRPC bridge/runtime
    support, packaged UE robotics project, host networking and deployment
    topology

- backlog hooks, not immediate pressure:
  - if a later GPU/runtime tranche wants UE work, the first honest targets are:
    headless render/materialization smoke, digital-twin ingest smoke, sensor
    simulation smoke, randomization/PCG smoke, middleware/hybrid-backend
    evaluation
  - these belong in non-training GPU/runtime backlogs only once there is a real
    wrapper script or runtime entrypoint to point at, not as fake active work

## 2026-04-09 — Nightly pass: agent verify regression closure

### What was built

- Introduced a canonical Claude shim template at
  `scripts/agent/claude_shim_template.md`.
- Switched both shim enforcement surfaces to this single source:
  - `scripts/agent/verify.sh`
  - `scripts/agent/bootstrap.sh`
- Added `tests/test_agent_shim_template.py` to prevent future drift between the
  template and `CLAUDE.md`, and to assert the copilot shim import remains
  explicit.

### Why this was the highest-value additive step

- The nightly audit had selected `agent_verify_regression` as the top safe task
  because baseline repository verification was failing.
- Restoring `agent_verify` correctness is a gating prerequisite before selecting
  additional roadmap scaffolding.
- This is additive hardening only: no frozen Phase B baseline math, checkpoint,
  trust net, `w_econ` lattice, or lambda controller logic was touched.

### Verification

- `./scripts/agent/verify.sh` → pass
- `python3 -m compileall src/` → pass
- `python3 -m pytest -q tests/test_agent_shim_template.py tests/test_economic_world_model_nightly_audit.py` → pass (`9 passed`)
- `python3 scripts/economic_world_model/nightly_audit.py --output-json artifacts/economic_world_model/nightly_audit_summary.json --output-markdown artifacts/economic_world_model/nightly_audit_summary.md` → pass

### Current status and next task

- Refreshed nightly audit now reports `status: ok` with no failing verification
  checks.
- Current next task posture is `docs_only` (no higher-priority missing additive
  scaffold detected on this scan).

## 2026-04-12 — Nightly audit parser hardening

### What was built

- Updated `scripts/economic_world_model/nightly_audit.py` so
  `_progress_latest_date()` accepts dated markdown headings with trailing
  descriptive text and heading levels `##` through `######`.
- Added/updated nightly-audit parser regression tests in
  `tests/test_economic_world_model_nightly_audit.py`:
  - H2 dated heading with suffix text
  - H3 dated heading with suffix text
- Refreshed nightly audit artifacts after the parser change:
  - `artifacts/economic_world_model/nightly_audit_summary.json`
  - `artifacts/economic_world_model/nightly_audit_summary.md`

### Why this was the highest-value additive step

- The nightly audit was under-reporting `progress_log_latest` because it only
  matched headings that were exactly `## YYYY-MM-DD`.
- This weakened drift/readiness signal quality and could incorrectly frame
  backlog freshness decisions.
- Hardening this parser is additive verification infrastructure: no frozen
  Phase B baseline math, checkpoint assets, trust-net, `w_econ`, or lambda
  controller logic was touched.

### Verification

- `python3 -m pytest -q tests/test_economic_world_model_nightly_audit.py`
  (`8 passed`)
- `PYTHONPYCACHEPREFIX=/tmp/pycache python3 -m compileall scripts/economic_world_model -q` (pass)
- `python3 scripts/economic_world_model/nightly_audit.py --output-json artifacts/economic_world_model/nightly_audit_summary.json --output-markdown artifacts/economic_world_model/nightly_audit_summary.md` (pass)

### Current status and next task

- Refreshed nightly audit remains `status: ok`.
- `progress_log_latest` now resolves to `2026-04-11`, matching the current top
  dated entry format.
- Next recommended task remains `audit_only` / `docs_only` until the audit
  surfaces a concrete safe additive scaffold.

## 2026-04-12 — Sim / Synth / Physics WM: SIM1 tactic note (docs-only)

### What was built

- Added a short SIM1-derived tactic note to
  `docs/economic_world_model/multi_wm_architecture_plan.md` under the Sim /
  Synth / Physics WM provider-family section.
- Added a matching roadmap reminder in
  `docs/economic_world_model/roadmap.md` under the reopenable Phase 1.x
  provider-and-boundary pass.
- Kept the note explicitly subordinate to repo doctrine: SIM1 is treated as a
  source of narrow provider-lane tactics, not as an architecture template or
  ontology for this stack.

### Why this matters

- The useful borrowing is practical and local to existing Sim / Synth /
  Physics ownership areas:
  - runnable provider/runtime bring-up discipline
  - metric-consistent, calibration-aware world instantiation for
    physics-sensitive lanes
  - staged `generate -> smooth -> replay -> filter` branch materialization
  - explicit reject filtering with typed reject receipts
  - replay-validity / task-consistency checks for mismatch evaluation
  - render/materialization as a downstream lane rather than sovereign center
  - replay/export discipline and training-worthiness gating
- This keeps our canonical typed state, receipts, replay/training exports,
  provider ownership, calibration, admission, and training feedback posture
  sovereign while still capturing a concrete external tactic source.

### Verification

- `git diff --check -- docs/economic_world_model/multi_wm_architecture_plan.md docs/economic_world_model/roadmap.md docs/economic_world_model/progress_log.md docs/economic_world_model/implementation_notes.md`

## 2026-04-12 — Doctrine: In-Place TTT / HALO admissible borrowings

### What was built

- Added a new doctrine subsection to
  `docs/economic_world_model/multi_wm_architecture_plan.md` covering:
  - In-Place TTT as a bounded subsystem-local fast-adaptation pattern
  - HALO as a local geometric confidence / abstention pattern
  - a combined synthesis paragraph
  - explicit anti-overfit guardrails
  - a far-future `WM-local shaping networks` note
- Added a short reinforcing note to
  `docs/economic_world_model/doctrine_economic_wm_future_architecture.md`
  tying those borrowings back to slow / meso / fast separation and bounded
  downstream shaping envelopes.
- Added a short local anomaly-head abstention note to
  `docs/economic_world_model/doctrine_bio_neuro_architecture_inspirations.md`.

### Why this matters

- This makes explicit that external neural methods can inform **subsystem
  shaping** without becoming top-level architecture templates for vpcore.
- The resulting doctrine is more concrete about where bounded adaptive memory
  is admissible, where calibrated abstention belongs, and why neither should
  bypass typed WM boundaries, typed receipts, or slow governance structure.
- The future shaping-network note also clarifies the sequencing:
  WM-local modulators and conditioning fields may become admissible only after
  subsystem ownership, receipts, and neural maturity are already real.

### Verification

- `git diff --check -- docs/economic_world_model/multi_wm_architecture_plan.md docs/economic_world_model/doctrine_economic_wm_future_architecture.md docs/economic_world_model/doctrine_bio_neuro_architecture_inspirations.md docs/economic_world_model/progress_log.md docs/economic_world_model/implementation_notes.md`
