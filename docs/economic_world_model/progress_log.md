# Economic World Model Progress Log

## 2026-05-20 - Phase 3 runner sidecars and neural architecture scaffolding

- **Changed**:
  - added `src/world_model/embodiment_actuation/sidecars.py`, which materializes canonical Phase 3 state, receipts, shadow consumer payloads, morphology receipts, Phase 3.4 rows, and neural-architecture manifests from the existing local embodiment runner
  - wired `src/embodiment/runner.py` so normal embodiment extraction now emits Phase 3 sidecars beside `EmbodimentProfile`, affordance, skill, cost, drift, and calibration artifacts
  - extended `EmbodimentProfileSummary`, datapack validation, and representation-token payloads so Phase 3 refs survive datapack/export paths instead of remaining ambient files
  - added `src/world_model/embodiment_actuation/neural_architectures.py` with CPU-runnable scaffolds for temporal JEPA-style latent prediction, ACT-style action chunking, Diffusion Policy-style action denoising, and topology-contrastive morphology consistency
  - extended `scripts/smoke_test_embodiment_phase34.py` so the local smoke writes a neural-architecture manifest and verifies finite CPU forwards for those architecture scaffolds
  - placed future Phase 3 neural training in `scripts/TRAINING_MIGRATION_BACKLOG.json` as `train_embodiment_phase34_neural_architectures.py`, explicitly blocked on GPU/provider/benchmark/latency evidence
  - extended runner and Phase 3.4 tests to prove sidecar emission, non-promotional posture, finite neural forward passes, and datapack ref preservation
- **Why this matters**:
  - Phase 3 is no longer only callable through isolated tests or smoke scripts; the regular local embodiment loop now leaves canonical WM artifacts and future-training manifests behind each episode
  - neural work that will eventually need GPU/provider evidence now has concrete local contracts, shapes, blockers, and proof-of-life without pretending training or promotion happened
  - runtime authority remains `none`; GPU/provider/native-runtime/hardware evidence remains explicitly blocked
- **Verification**:
  - `git diff --check && python3 -m ruff check src/world_model/embodiment_actuation src/embodiment/runner.py src/embodiment/datapack_adapter.py src/valuation/datapack_schema.py src/valuation/datapack_validators.py src/representation/token_providers.py tests/test_embodiment_actuation_phase34.py tests/embodiment/test_embodiment_module.py && python3 -m compileall src/world_model/embodiment_actuation src/embodiment src/valuation src/representation tests/test_embodiment_actuation_phase34.py tests/embodiment/test_embodiment_module.py -q && python3 -m pytest tests/test_embodiment_actuation_phase34.py tests/embodiment/test_embodiment_module.py tests/test_embodiment_actuation_world_model.py -q` (`20 passed`)
  - `python3 scripts/economic_world_model/nightly_audit.py --output-json artifacts/economic_world_model/nightly_audit_summary.json --output-markdown artifacts/economic_world_model/nightly_audit_summary.md` (`status: ok`)
  - `python3 scripts/smoke_test_embodiment_phase34.py --out-dir artifacts/embodiment_phase34 --variant g1_29dof` (`status: ok`, neural architecture manifest `promotion_eligible=false`)
  - `python3 -m pytest tests/ -q` (`1662 passed, 2 skipped, 28 warnings`)

## 2026-05-20 - Phase 3 morphology evidence and 3.4 neural scaffolding

- **Changed**:
  - added `src/world_model/embodiment_actuation/morphology.py` with G1 morphology profiles, joint specs, registry-entry conversion, and OSS/local evidence receipts
  - added `src/world_model/embodiment_actuation/neural_seams.py` with CPU-runnable local-contact-dynamics, inverse-retargeting, action-proposal, and drift/calibration seam modules
  - added `src/world_model/embodiment_actuation/training_corpus.py` with Phase 3.4 training rows, manifest generation, JSONL write/load, and explicit non-promotional blocker posture
  - added `scripts/smoke_test_embodiment_phase34.py` and `tests/test_embodiment_actuation_phase34.py`
  - added `docs/economic_world_model/phase3_external_pattern_absorption.md` documenting what was borrowed from Unitree/Isaac/GR00T-style public patterns and what remains external
- **Why this matters**:
  - Phase 3.4 is now locally code-real without requiring GPU training: seams execute forward on CPU and training rows are materialized
  - Unitree G1 morphology/action-space evidence is now a typed WM artifact rather than a prose-only future concern
  - promotion remains blocked until provider, GPU, benchmark, latency, watchdog, and hardware drift evidence exist
- **Verification**:
  - `git diff --check && python3 -m ruff check src/world_model/embodiment_actuation scripts/smoke_test_embodiment_phase34.py tests/test_embodiment_actuation_phase34.py tests/test_embodiment_actuation_world_model.py && python3 -m compileall src/world_model/embodiment_actuation scripts/smoke_test_embodiment_phase34.py tests/test_embodiment_actuation_phase34.py tests/test_embodiment_actuation_world_model.py -q && python3 -m pytest tests/test_embodiment_actuation_world_model.py tests/test_embodiment_actuation_phase34.py -q` (`13 passed`)
  - `python3 scripts/smoke_test_embodiment_phase34.py --out-dir artifacts/embodiment_phase34 --scan-root /Users/amarmurray/code/unitree_rl_gym --scan-root /Users/amarmurray/code/unitree_sim_isaaclab --scan-root /Users/amarmurray/code/unitree_models --variant g1_29dof` (`status: ok`)
  - `python3 -m pytest tests/embodiment/test_embodiment_module.py tests/test_embodiment_shadow_consumer.py tests/test_sim_synth_phase1x_subsystems.py -q` (`29 passed`)
  - `python3 scripts/economic_world_model/nightly_audit.py --output-json artifacts/economic_world_model/nightly_audit_summary.json --output-markdown artifacts/economic_world_model/nightly_audit_summary.md` (`status: ok`)
  - `python3 -m pytest tests/ -q` (`1660 passed, 2 skipped, 24 warnings`)


## 2026-05-20 - Phase 3.1-3.3 Embodiment / Actuation WM shadow substrate

- **Changed**:
  - added `src/world_model/embodiment_actuation/` with canonical Phase 3 state contracts, receipt contracts, provider/runtime contracts, promotion posture, a shadow compiler, and shadow downstream consumers
  - compiler now builds `EmbodimentActuationWorldState` from existing advisory embodiment artifacts, registry entries, `ActionAdapterV2`, `ObservationAdapterV2`, Perception embodiment-shadow surfaces, provider contracts, optional joint state, and source refs
  - receipt family now covers compilation, capability profile, action-space validation, observation interface, contact/affordance, local dynamics, inverse retargeting, action proposal, safety envelope, drift, calibration, cost, and Sim↔Embodiment transfer
  - downstream consumers now emit shadow-only Sim/Synth transfer context, Perception feedback, Runtime adapter validation, and Economic receipt bundles
  - extended Sim/Synth embodiment-input normalization to preserve Phase 3 action feasibility, retarget readiness, drift, safety status, and authority level
  - added `tests/test_embodiment_actuation_world_model.py` covering permissive missing-data posture, receipt completeness, provider-honesty, shadow consumers, and seam promotion gating
- **Why this matters**:
  - Phase 3 is now code-real through 3.3 without requiring GPUs, provider bring-up, GR00T import, or hardware claims
  - the GR00T-inspired lane now has a native socket for teacher/student, action-space hygiene, transfer receipts, and promotion gates instead of a foreign execution island
  - runtime authority remains explicitly `none`; safety/latency/watchdog evidence remains external when not provided
- **Verification**:
  - `git diff --check && python3 -m ruff check src/world_model/embodiment_actuation src/world_model/sim_synth_physics/adapters/embodiment_inputs.py tests/test_embodiment_actuation_world_model.py && python3 -m compileall src/ tests/test_embodiment_actuation_world_model.py -q && python3 -m pytest tests/test_embodiment_actuation_world_model.py tests/embodiment/test_embodiment_module.py tests/test_embodiment_shadow_consumer.py tests/test_sim_synth_phase1x_subsystems.py -q` (`36 passed`)
  - `python3 scripts/economic_world_model/nightly_audit.py --output-json artifacts/economic_world_model/nightly_audit_summary.json --output-markdown artifacts/economic_world_model/nightly_audit_summary.md` (`status: ok`)
  - `python3 -m pytest tests/ -q` (`1654 passed, 2 skipped, 24 warnings`)


## 2026-05-19 - Phase 1.x closure assessment and Phase 3 spec prep

- **Changed**:
  - added `docs/economic_world_model/phase1x_closure_assessment.md`, recording Category A = `0`, unresolved Category C = `0`, and remaining Phase 1.x blockers as external provider / GPU / asset / calibration / benchmark / native-runtime evidence
  - added `docs/economic_world_model/groot_inspired_functionality_status.md`, separating GR00T-inspired teacher/student, deploy-observation, randomization, transfer, and promotion-gate patterns from the repo-native multi-WM ontology
  - added `docs/economic_world_model/phase3_embodiment_actuation_spec_prep.md`, defining the first Phase 3 canonical state, receipt, shadow compiler, learned-seam, and provider-contract prep targets
  - cross-linked the new status/spec artifacts from `roadmap.md` and `actuation_embodiment_world_model.md`
- **Why this matters**:
  - the Phase 1.x return leg now has the same explicit closure sheet pattern used for Phase 2 rather than an implicit judgment call
  - the remaining Isaac/Unitree latency/watchdog and native-runtime gaps stay honest as external evidence gates, not local defaults to invent
  - Phase 3 can begin from typed Embodiment / Actuation state prep after owner/Claude acceptance instead of jumping straight to hardware/provider bring-up
- **Verification**:
  - `git diff --check && python3 -m compileall src/`
  - `python3 -m pytest tests/test_scan_phase1_runtime_layouts.py tests/test_setup_holosoma_local_smoke.py tests/test_local_holosoma_smoke.py tests/test_sim_synth_phase1x_subsystems.py -q` (`12 passed`)
  - `python3 scripts/economic_world_model/nightly_audit.py --output-json artifacts/economic_world_model/nightly_audit_summary.json --output-markdown artifacts/economic_world_model/nightly_audit_summary.md` (`status: ok`)
  - `python3 -m pytest tests/ -q` (`1647 passed, 2 skipped, 24 warnings`)


## 2026-05-19 - Local Holosoma smoke bootstrap is reproducible

- **Changed**:
  - added `scripts/setup_holosoma_local_smoke.py`, a no-pip bootstrap for the
    local Holosoma `.pth` path shim
  - the script installs, dry-runs, or removes
    `robotics_vp_holosoma_local.pth` and reports the exact path entries and
    follow-up smoke commands as JSON
  - added `tests/test_setup_holosoma_local_smoke.py` for install, missing-path,
    and remove behavior
- **Why this matters**:
  - the local ONNX deploy-smoke setup is now reproducible on a fresh host
    without accidentally pulling the full Holosoma provider dependency tree
  - this keeps the distinction clean: local deploy inference can be reproduced
    cheaply, while full simulated episode/runtime evidence remains gated
- **Verification**:
  - `python3 -m ruff check scripts/setup_holosoma_local_smoke.py tests/test_setup_holosoma_local_smoke.py`
  - `python3 -m compileall scripts/setup_holosoma_local_smoke.py tests/test_setup_holosoma_local_smoke.py`
  - `python3 -m pytest tests/test_setup_holosoma_local_smoke.py -q` (`3 passed`)
  - `python3 scripts/setup_holosoma_local_smoke.py` (`status: installed`)
  - `python3 scripts/local_holosoma_smoke.py --preflight-only --out-dir artifacts/holosoma_local_probe` (`ready: true`, `policy_kind: onnx_deploy`)
  - `python3 scripts/local_holosoma_smoke.py --episodes 1 --out-dir artifacts/holosoma_local_probe` (`actor_obs [1, 100] -> action [1, 29]`, finite `float32`)
  - `python3 -m ruff check scripts/setup_holosoma_local_smoke.py tests/test_setup_holosoma_local_smoke.py && git diff --check && python3 -m compileall src/ scripts/setup_holosoma_local_smoke.py tests/test_setup_holosoma_local_smoke.py && python3 -m pytest tests/ -q` (`1647 passed, 2 skipped, 24 warnings`)
  - `python3 scripts/economic_world_model/nightly_audit.py --output-json artifacts/economic_world_model/nightly_audit_summary.json --output-markdown artifacts/economic_world_model/nightly_audit_summary.md` (`status: ok`, no drift)

## 2026-05-19 - Local Holosoma ONNX deploy smoke runs without full provider install

- **Changed**:
  - exposed the existing local Holosoma checkout through a tiny user-site `.pth`
    path shim instead of running the full heavy dependency install
  - installed only narrow local runtime deps needed for the smoke path
    (`tyro`, `loguru`, `omegaconf`, `tqdm`, `tensordict`, `tensorboard`,
    `trimesh`, `onnxruntime`) with `--no-cache-dir`; captured the same set in `requirements-holosoma-smoke.txt`
  - `scripts/local_holosoma_smoke.py` now distinguishes ONNX deploy smoke from
    native Holosoma checkpoint evaluation
  - fixed the Holosoma backend entrypoint import shape so the local upstream
    checkout layout (`holosoma.eval_agent`, `holosoma.train_agent`) works
  - added an explicit `ROBOTICS_VP_ENABLE_HOLOSOMA_RUNTIME=1` gate so making
    Holosoma importable does not automatically promote full WM runtime execution
- **Why this matters**:
  - the local Holosoma lane is no longer blocked on provider visibility
  - the selected policy is an ONNX artifact, so the correct local proof is an
    ONNX deploy/action smoke, not a native Holosoma eval loop expecting a
    serialized training checkpoint with `experiment_config`
  - full simulated episode/runtime evidence remains future GPU/provider work;
    this is a local deploy-path proof, not benchmark promotion evidence, and WM
    runtime routing remains shadow/fallback unless explicitly enabled
- **Verification**:
  - `python3 -m ruff check scripts/local_holosoma_smoke.py tests/test_local_holosoma_smoke.py src/motor_backend/holosoma_backend.py`
  - `python3 -m compileall scripts/local_holosoma_smoke.py tests/test_local_holosoma_smoke.py src/motor_backend/holosoma_backend.py`
  - `python3 -m pytest tests/test_local_holosoma_smoke.py tests/test_holosoma_backend_interface.py -q` (`4 passed`)
  - `python3 scripts/local_holosoma_smoke.py --preflight-only --out-dir artifacts/holosoma_local_probe` (`ready: true`, `policy_kind: onnx_deploy`)
  - `python3 scripts/local_holosoma_smoke.py --episodes 1 --out-dir artifacts/holosoma_local_probe` wrote `holosoma_onnx_deploy_smoke.json` with `actor_obs [1, 100] -> action [1, 29]`, finite `float32` output
  - `python3 -m ruff check scripts/local_holosoma_smoke.py tests/test_local_holosoma_smoke.py tests/test_sim_synth_physics_world_model.py src/motor_backend/holosoma_backend.py src/world_model/sim_synth_physics/holosoma_runtime_gate.py src/world_model/sim_synth_physics/backend_adapters.py src/world_model/sim_synth_physics/adapters/backend_holosoma.py src/world_model/sim_synth_physics/runtime_targets.py src/world_model/sim_synth_physics/shadow_execution.py src/world_model/sim_synth_physics/backend_runtime_execution.py src/world_model/sim_synth_physics/adapters/holosoma_adapter_execution.py && git diff --check && python3 -m compileall src/ scripts/local_holosoma_smoke.py tests/test_local_holosoma_smoke.py tests/test_sim_synth_physics_world_model.py && python3 -m pytest tests/ -q` (`1644 passed, 2 skipped, 24 warnings`)

## 2026-05-19 - Holosoma local preflight separates provider install from GPU debt

- **Changed**:
  - `scripts/local_holosoma_smoke.py` can now auto-resolve the selected local
    Holosoma policy checkpoint from the runtime policy contract when
    `--policy-id` is omitted
  - added `--preflight-only`, which writes
    `holosoma_smoke_preflight.json` and reports Holosoma module availability,
    selected policy ref/source, policy existence, readiness, and missing
    preconditions without attempting runtime execution
  - added tests for missing-module and auto-policy behavior
- **Why this matters**:
  - the current host has local Holosoma roots/checkpoints, so Phase 1 is not
    literally “GPU-only” in the Holosoma lane
  - the live blocker is now explicit: `holosoma_python_module` is missing while
    the selected policy checkpoint exists
  - Isaac/Unitree and GGDS/LDM remain GPU/runtime/asset blocked, but Holosoma
    can be advanced by local provider installation before RunPod is available
- **Verification**:
  - `python3 -m ruff check scripts/local_holosoma_smoke.py tests/test_local_holosoma_smoke.py`
  - `git diff --check && python3 -m compileall scripts/local_holosoma_smoke.py tests/test_local_holosoma_smoke.py -q && python3 -m pytest tests/test_local_holosoma_smoke.py -q` (`2 passed`)
  - `python3 scripts/local_holosoma_smoke.py --preflight-only --out-dir artifacts/holosoma_local_probe` (`ready: false`, missing `holosoma_python_module`, policy checkpoint exists)
  - `python3 -m ruff check scripts/local_holosoma_smoke.py tests/test_local_holosoma_smoke.py && git diff --check && python3 -m compileall src/ scripts/local_holosoma_smoke.py tests/test_local_holosoma_smoke.py && python3 -m pytest tests/ -q` (`1642 passed, 3 skipped, 24 warnings`)

## 2026-05-19 - Phase 1.x subsystem index trainer-row propagation

- **Changed**:
  - backend-selector and branch-planner receipt-row builders now preserve
    `phase1x_subsystem_index_v1` identity, coverage summary, subsystem IDs,
    ownership rule, structural status, and honest blocker class in
    trainer-facing metadata
  - added regression coverage proving compiled subsystem indices survive from
    world-state metadata into both trainer row families
- **Why this matters**:
  - subsystem legibility now reaches the surfaces that training, promotion, and
    later benchmark audits actually inspect
  - this avoids reintroducing prose-only ownership once rows leave the compiled
    world-state artifact
- **Verification**:
  - `python3 -m ruff check src/world_model/sim_synth_physics/training_corpus.py tests/test_sim_synth_phase1x_subsystems.py`
  - `git diff --check && python3 -m compileall src/world_model/sim_synth_physics tests/test_sim_synth_phase1x_subsystems.py -q && python3 -m pytest tests/test_sim_synth_phase1x_subsystems.py tests/test_sim_synth_training_corpus.py -q` (`7 passed`)
  - `git diff --check && python3 -m compileall src/ && python3 -m pytest tests/ -q` (`1640 passed, 3 skipped, 24 warnings`)

## 2026-05-19 - Phase 1.x subsystem index in compiled WM state

- **Changed**:
  - added `phase1x_subsystem_index_v1`, a machine-readable mapping of the
    Sim / Synth / Physics WM's 10 Phase 1.x subsystems to owned modules, typed
    state surfaces, receipt families, learned/reserved seams, promotion gates,
    provider families, runtime artifact refs, and honest external blockers
  - compiled `SimSynthPhysicsWorldState` metadata now carries the subsystem
    index with runtime artifact refs and compiled receipt-family coverage
  - package exports expose `build_phase1x_subsystem_index(...)` and the static
    `PHASE1X_SUBSYSTEM_SPECS`
- **Why this matters**:
  - the 10-subsystem Phase 1.x decomposition is no longer doctrine-only
  - downstream audits can now distinguish local structural ownership from
    provider/GPU/asset/benchmark blockers without re-parsing roadmap prose
- **Verification**:
  - `python3 -m ruff check src/world_model/sim_synth_physics/subsystems.py src/world_model/sim_synth_physics/compiler.py src/world_model/sim_synth_physics/__init__.py tests/test_sim_synth_phase1x_subsystems.py`
  - `git diff --check && python3 -m compileall src/world_model/sim_synth_physics tests/test_sim_synth_phase1x_subsystems.py -q && python3 -m pytest tests/test_sim_synth_phase1x_subsystems.py tests/test_sim_synth_phase1x_surfaces.py tests/test_sim_synth_physics_world_model.py -q` (`36 passed`)
  - `git diff --check && python3 -m compileall src/ && python3 -m pytest tests/ -q` (`1639 passed, 3 skipped, 24 warnings`)

## 2026-05-19 - Phase 1.x training-gate promotion preconditions

- **Changed**:
  - added `phase1x_training_gate_v1` as the structural promotion gate for Sim /
    Synth / Physics trainer outputs
  - backend-selector and branch-planner dataset summaries, training summaries,
    runtime packages, job results, Regal metadata, and execution preconditions
    now carry the gate
  - runtime package promotion now requires both benchmark-density readiness and
    Phase 1.x training-gate readiness
  - the gate checks selected-row count consistency, absence of diagnostic rows,
    clean runtime receipt manifest validation, and reject-head coverage whenever
    negative-supervision sidecars exist
- **Why this matters**:
  - reject-head training made negative supervision usable, but promotion still
    needed a single auditable yes/no surface tying admissibility to package
    readiness
  - this keeps local trainer progress useful without implying provider truth,
    GPU-backed calibration, or benchmark credibility
- **Verification**:
  - `python3 -m ruff check src/world_model/sim_synth_physics/training_corpus.py src/world_model/sim_synth_physics/__init__.py scripts/train_sim_synth_backend_selector.py scripts/train_sim_synth_branch_planner.py tests/test_train_sim_synth_backend_selector.py tests/test_train_sim_synth_branch_planner.py tests/test_sim_synth_training_corpus.py`
  - `git diff --check && python3 -m compileall src/world_model/sim_synth_physics scripts/train_sim_synth_backend_selector.py scripts/train_sim_synth_branch_planner.py tests/test_train_sim_synth_backend_selector.py tests/test_train_sim_synth_branch_planner.py tests/test_sim_synth_training_corpus.py -q && python3 -m pytest tests/test_train_sim_synth_backend_selector.py tests/test_train_sim_synth_branch_planner.py tests/test_sim_synth_physics_world_model.py tests/test_sim_synth_training_corpus.py -q` (`43 passed`)
  - `git diff --check && python3 -m compileall src/ && python3 -m pytest tests/ -q` (`1637 passed, 3 skipped, 24 warnings`)

## 2026-05-19 - Phase 1.x reject-head training over negative supervision

- **Changed**:
  - added bounded `reject_probability` heads to the Sim / Synth / Physics
    backend-selector and branch-planner helper models
  - trainer entrypoints now pass preserved `negative_supervision` sidecar rows
    into reject-head losses while keeping backend/mode/yield heads trained only
    on positive/legacy rows
  - runtime packages, model configs, training summaries, Regal metadata, and
    checkpoint metadata now record the negative-supervision contract and reject
    accuracy
  - promoted learned helper payloads that recommend rejection now stay as traces
    and do not override heuristic backend or branch choices
- **Why this matters**:
  - the previous sidecars were evidence-preserving but not yet trainable
  - this converts filtered Phase 1.x outcomes into a first local learned signal
    without mixing them into positive labels
  - the reject head remains bounded and non-promotional until provider truth and
    benchmark evidence exist
- **Verification**:
  - `python3 -m ruff check src/world_model/sim_synth_physics/backend_selector.py src/world_model/sim_synth_physics/branch_planner.py src/world_model/sim_synth_physics/compiler.py src/world_model/sim_synth_physics/synthetic_branches.py src/world_model/sim_synth_physics/promotion.py scripts/train_sim_synth_backend_selector.py scripts/train_sim_synth_branch_planner.py tests/test_train_sim_synth_backend_selector.py tests/test_train_sim_synth_branch_planner.py tests/test_sim_synth_physics_world_model.py tests/test_sim_synth_training_corpus.py`
  - `git diff --check && python3 -m compileall src/world_model/sim_synth_physics scripts/train_sim_synth_backend_selector.py scripts/train_sim_synth_branch_planner.py tests/test_train_sim_synth_backend_selector.py tests/test_train_sim_synth_branch_planner.py tests/test_sim_synth_physics_world_model.py tests/test_sim_synth_training_corpus.py -q && python3 -m pytest tests/test_train_sim_synth_backend_selector.py tests/test_train_sim_synth_branch_planner.py tests/test_sim_synth_physics_world_model.py tests/test_sim_synth_training_corpus.py -q` (`43 passed`)
  - `git diff --check && python3 -m compileall src/ && python3 -m pytest tests/ -q` (`1637 passed, 3 skipped, 24 warnings`)

## 2026-05-19 - Phase 1.x excluded-row sidecars for trainer inputs

- **Changed**:
  - added `phase1x_training_row_split_v1` row splitting for Sim / Synth /
    Physics trainer inputs
  - backend-selector and branch-planner training scripts now write explicit
    negative-supervision and diagnostic JSONL sidecars beside the positive
    training dataset
  - Regal training manifests register the excluded-row sidecars as artifacts, so
    rejected rows are preserved for later negative-loss work instead of only
    counted in summaries
- **Why this matters**:
  - trainer-side admissibility enforcement should not erase negative evidence
  - the current helper losses remain positive-only, but future reject/utility
    heads now have a local artifact path to consume
  - this is still structural/data plumbing, not a claim that negative examples
    are already improving model quality
- **Verification**:
  - `python3 -m ruff check src/world_model/sim_synth_physics/training_corpus.py src/world_model/sim_synth_physics/__init__.py scripts/train_sim_synth_backend_selector.py scripts/train_sim_synth_branch_planner.py tests/test_sim_synth_training_corpus.py tests/test_train_sim_synth_backend_selector.py tests/test_train_sim_synth_branch_planner.py`
  - `git diff --check && python3 -m compileall src/world_model/sim_synth_physics scripts/train_sim_synth_backend_selector.py scripts/train_sim_synth_branch_planner.py tests/test_sim_synth_training_corpus.py tests/test_train_sim_synth_backend_selector.py tests/test_train_sim_synth_branch_planner.py -q && python3 -m pytest tests/test_sim_synth_training_corpus.py tests/test_train_sim_synth_backend_selector.py tests/test_train_sim_synth_branch_planner.py -q` (`10 passed`)
  - `git diff --check && python3 -m compileall src/ && python3 -m pytest tests/ -q` (`1634 passed, 3 skipped, 24 warnings`)

## 2026-05-19 - Phase 1.x trainer-side admissibility enforcement

- **Changed**:
  - added `phase1x_positive_training_row_selection_v1` selection summaries for
    Sim / Synth / Physics training rows
  - backend-selector and branch-planner trainer entrypoints now train only on
    `positive_training` rows plus explicit legacy dataset rows
  - `negative_supervision` and `diagnostic_only` rows are excluded from current
    positive-only helper losses while their counts, reasons, and row refs remain
    visible in dataset summaries, training summaries, job results, and Regal
    receipt-label coverage
- **Why this matters**:
  - the previous pass made admissibility legible; this pass makes it enforced
    at the local trainer boundary
  - negative supervision is no longer accidentally treated as positive labels
    before the helper models have explicit negative-example losses
  - this remains local-only trainer hygiene, not provider truth or promotion
    evidence
- **Verification**:
  - `python3 -m ruff check src/world_model/sim_synth_physics/training_corpus.py src/world_model/sim_synth_physics/__init__.py scripts/train_sim_synth_backend_selector.py scripts/train_sim_synth_branch_planner.py tests/test_sim_synth_training_corpus.py tests/test_train_sim_synth_backend_selector.py tests/test_train_sim_synth_branch_planner.py`
  - `git diff --check && python3 -m compileall src/world_model/sim_synth_physics scripts/train_sim_synth_backend_selector.py scripts/train_sim_synth_branch_planner.py tests/test_sim_synth_training_corpus.py tests/test_train_sim_synth_backend_selector.py tests/test_train_sim_synth_branch_planner.py -q && python3 -m pytest tests/test_sim_synth_training_corpus.py tests/test_train_sim_synth_backend_selector.py tests/test_train_sim_synth_branch_planner.py -q` (`10 passed`)
  - `git diff --check && python3 -m compileall src/ && python3 -m pytest tests/ -q` (`1634 passed, 3 skipped, 24 warnings`)

## 2026-05-19 - Phase 1.x training-admissibility gating for receipt rows

- **Changed**:
  - added `phase1x_training_admissibility_v1` classification for harvested
    backend-selector and branch-planner rows
  - rows now distinguish positive training rows, negative-supervision rows,
    and diagnostic-only rows using runtime manifest validation, target-source
    posture, branch-validity reasons, and replay-validity reasons
  - branch-planner rows now treat missing outcomes, missing branch/replay
    validity, manifest mismatches, and planning-only targets as diagnostic
    blockers rather than positive training data
- **Why this matters**:
  - the runtime can now emit rich receipts without every row being interpreted
    as equally trainable
  - negative/filtered outcomes remain useful, but they are labeled as negative
    supervision instead of being silently mixed into positive targets
  - this is still local-only structural gating; provider truth, benchmark
    credibility, and calibration quality remain future evidence gates
- **Verification**:
  - `python3 -m ruff check src/world_model/sim_synth_physics/training_corpus.py tests/test_sim_synth_training_corpus.py`
  - `python3 -m pytest tests/test_sim_synth_training_corpus.py -q` (`4 passed`)
  - `python3 -m pytest tests/test_sim_synth_training_corpus.py tests/test_sim_synth_physics_world_model.py tests/test_sim_synth_phase1x_surfaces.py tests/test_sim_synth_camera_geometry.py tests/test_sim_synth_vectorized_runtime.py -q` (`40 passed`)
  - `git diff --check && python3 -m compileall src/ && python3 -m pytest tests/ -q` (`1634 passed, 3 skipped, 24 warnings`)

## 2026-05-19 - Phase 1.x runtime receipt manifest validation

- **Changed**:
  - added `validate_runtime_receipt_manifest(...)` for harvested Sim / Synth /
    Physics receipt bundles
  - live-directory harvesting now expands runtime-emitted receipt-bundle files
    such as branch-validity, replay-validity, render-provider, simulation
    outcome, and backend work-order bundles into bundle rows
  - backend-selector and branch-planner training rows now expose runtime
    manifest validation status and mismatched-family diagnostics
- **Why this matters**:
  - manifest presence alone is not enough; training consumers need to know
    whether manifest receipt-family counts match the actual harvested bundle
  - this closes a local-only integrity gap before provider-era runs start
    emitting larger receipt sets
  - the validator remains structural: it checks internal receipt accounting, not
    provider truth, benchmark success, or calibration quality
- **Verification**:
  - `python3 -m ruff check src/world_model/sim_synth_physics tests/test_sim_synth_training_corpus.py tests/test_sim_synth_physics_world_model.py`
  - `python3 -m pytest tests/test_sim_synth_training_corpus.py tests/test_sim_synth_physics_world_model.py tests/test_sim_synth_phase1x_surfaces.py tests/test_sim_synth_camera_geometry.py tests/test_sim_synth_vectorized_runtime.py -q` (`40 passed`)
  - `python3 -m compileall src/ && python3 -m pytest tests/ -q` (`1634 passed, 3 skipped, 24 warnings`)

## 2026-05-19 - Phase 1.x runtime receipt manifest consolidation

- **Changed**:
  - added a `sim_synth_runtime_receipt_manifest_v1` artifact emitted as
    `runtime_receipt_manifest.json` from the Sim / Synth / Physics runtime
  - the manifest records emitted receipt families, receipt ids, artifact paths,
    required-vs-optional status, missing required families, route posture, and
    training-feedback row counts
  - `sim_synth_training_feedback_v1` now carries the runtime manifest id,
    manifest status, and missing-required-family list
  - receipt harvesting now recognizes runtime receipt manifests and exposes
    manifest id/status/counts in backend-selector and branch-planner rows
- **Why this matters**:
  - the Phase 1.x receipt family has grown enough that relying on loose files
    alone would invite drift
  - the manifest gives future provider-era runs a single audit surface for
    checking whether all required local receipts were emitted before training
    rows are trusted
  - optional provider/runtime receipts remain honest: not emitted is tracked
    without becoming a missing-required failure
- **Verification**:
  - `python3 -m ruff check src/world_model/sim_synth_physics tests/test_sim_synth_training_corpus.py tests/test_sim_synth_physics_world_model.py`
  - `python3 -m pytest tests/test_sim_synth_training_corpus.py tests/test_sim_synth_physics_world_model.py tests/test_sim_synth_phase1x_surfaces.py tests/test_sim_synth_camera_geometry.py tests/test_sim_synth_vectorized_runtime.py -q` (`40 passed`)
  - `python3 -m compileall src/ && python3 -m pytest tests/ -q` (`1634 passed, 3 skipped, 24 warnings`)

## 2026-05-18 - Phase 1.x replay-validity / task-consistency receipts

- **Changed**:
  - added `ReplayValidityReceipt` as the per-branch post-outcome training
    admissibility receipt tying outcome status to task, transfer, branch, and
    sensor evidence
  - runtime execution now emits `replay_validity_receipts.json`, includes the
    receipts in loop results, and threads per-row replay validity into
    `sim_synth_training_feedback_v1`
  - receipt harvesting now recognizes replay-validity receipts and projects
    aggregate reject reasons into backend-selector rows plus per-branch
    validity / consistency scores into branch-planner rows
- **Why this matters**:
  - the roadmap's replay-validity / task-consistency filter is now a concrete
    CPU-local artifact rather than a future provider-season note
  - branch outcomes can be filtered from training for explicit reasons
    (`outcome_blocked_by_admission`, `sensor_alignment_unready`, high sim-real
    gap, or benchmark-gate absence) instead of being silently treated as useful
  - this remains a local estimate until real provider replay and benchmark
    evidence exist
- **Verification**:
  - `python3 -m ruff check src/world_model/sim_synth_physics tests/test_sim_synth_camera_geometry.py tests/test_sim_synth_phase1x_surfaces.py tests/test_sim_synth_training_corpus.py tests/test_sim_synth_physics_world_model.py`
  - `python3 -m pytest tests/test_sim_synth_camera_geometry.py tests/test_sim_synth_phase1x_surfaces.py tests/test_sim_synth_training_corpus.py tests/test_sim_synth_physics_world_model.py tests/test_sim_synth_vectorized_runtime.py -q` (`40 passed`)
  - `python3 -m compileall src/ && python3 -m pytest tests/ -q` (`1634 passed, 3 skipped, 24 warnings`)

## 2026-05-18 - Phase 1.x geometry-backed sensor alignment receipts

- **Changed**:
  - extended CPU-local camera geometry helpers with metadata parsers for
    common intrinsics / extrinsics shapes plus round-trip reprojection checks
  - added `SensorAlignmentReceipt` as the typed camera/sensor geometry receipt
    for Sim / Synth / Physics scene materialization posture
  - runtime execution now emits `sensor_alignment_receipt.json`, includes the
    receipt in loop results, and threads alignment status / score into
    training-feedback rows
  - receipt harvesting now recognizes sensor-alignment receipts and projects
    their status, score, checks, and metrics into backend-selector and
    branch-planner training rows
- **Why this matters**:
  - this turns camera intrinsics/extrinsics alignment from a loose asset note
    into replayable CPU-local evidence before provider/GPU bring-up exists
  - the receipt is intentionally a geometry-contract check, not a calibration
    or benchmark claim; missing or invalid contracts remain explicit
  - future Isaac/UE5/Habitat-style providers can plug real observation bundles
    into the same receipt family instead of introducing bespoke sensor gates
- **Verification**:
  - `python3 -m ruff check src/world_model/sim_synth_physics tests/test_sim_synth_camera_geometry.py tests/test_sim_synth_phase1x_surfaces.py tests/test_sim_synth_training_corpus.py tests/test_sim_synth_physics_world_model.py`
  - `python3 -m pytest tests/test_sim_synth_camera_geometry.py tests/test_sim_synth_phase1x_surfaces.py tests/test_sim_synth_training_corpus.py tests/test_sim_synth_physics_world_model.py tests/test_sim_synth_vectorized_runtime.py -q` (`40 passed`)
  - `python3 -m compileall src/ && python3 -m pytest tests/ -q` (`1634 passed, 3 skipped, 24 warnings`)

## 2026-05-18 - Phase 1.x branch-validity / reject-filter receipts

- **Changed**:
  - added `BranchValidityReceipt` as the typed per-branch admission and
    reject-filter receipt for Sim / Synth / Physics branches
  - runtime execution now emits `branch_validity_receipts.json`, includes the
    receipts in `SimSynthPhysicsLoopResult`, and threads per-branch validity
    into `sim_synth_training_feedback_v1` rows
  - receipt harvesting now recognizes standalone and bundled branch-validity
    receipts
  - backend-selector rows now expose aggregate branch-validity admission /
    reject counts, while branch-planner rows expose per-branch validity score,
    admission score, evidence status, and reject reasons
- **Why this matters**:
  - the SIM1/Habitat-derived `generate -> smooth -> replay -> filter` posture
    now has a concrete CPU-local reject-filter artifact instead of remaining a
    doctrine-only reminder
  - branch admission becomes replayable training evidence, not a transient
    compiler choice hidden behind `Gen2SimAdmissionState`
  - current evidence remains deliberately conservative (`local_estimate` unless
    benchmark gates are ready), so this does not claim provider/GPU bring-up
- **Verification**:
  - `python3 -m ruff check src/world_model/sim_synth_physics tests/test_sim_synth_training_corpus.py tests/test_sim_synth_physics_world_model.py`
  - `python3 -m pytest tests/test_sim_synth_training_corpus.py tests/test_sim_synth_physics_world_model.py tests/test_sim_synth_phase1x_surfaces.py tests/test_sim_synth_vectorized_runtime.py -q` (`36 passed`)
  - `python3 -m compileall src/ && python3 -m pytest tests/ -q` (`1633 passed, 3 skipped, 24 warnings`)

## 2026-05-18 - Phase 1.x consumers: scene hierarchy and transfer evidence made training-visible

- **Changed**:
  - `SceneHierarchyState` now flows into:
    - `SyntheticBranchPlan.gap_target_refs`
    - branch-plan metadata
    - render-provider config / metadata
    - render materialization manifests and source context
  - `sim_synth_training_feedback_v1` rows now carry a transfer-evidence
    summary derived from:
    - `TaskMeasurementReceipt`
    - `SimRealGapReceipt`
    - `BackendMismatchReceipt`
    - `SurrogatePhysicsReceipt`
    - `SurrogateCalibrationReceipt`
  - `harvest_sim_synth_receipt_bundles(...)` now recognizes and bundles the
    new Phase 1.x receipt family
  - backend-selector and branch-planner training rows now expose task
    measurement values, sim-real gap score/status, backend mismatch score/status,
    surrogate posture, and scene hierarchy refs
- **Why this matters**:
  - the first Phase 1.x tranche created the joints; this pass makes them load
    bearing
  - future training/eval code can now condition on scene/materialization
    structure and transfer-risk evidence instead of rediscovering those facts
    from loose artifacts
  - the evidence remains honest: current local rows expose estimated and
    contract-reserved posture until provider/GPU execution replaces it
- **Verification**:
  - `python3 -m ruff check src/world_model/sim_synth_physics tests/test_sim_synth_phase1x_surfaces.py tests/test_sim_synth_training_corpus.py tests/test_sim_synth_physics_world_model.py`
  - `python3 -m pytest tests/test_sim_synth_phase1x_surfaces.py tests/test_sim_synth_training_corpus.py tests/test_sim_synth_physics_world_model.py tests/test_sim_synth_vectorized_runtime.py -q`

## 2026-05-18 - Phase 1.x re-entry tranche: shared surfaces, task protocol, transfer receipts, geometry, and batch runner

- **Changed**:
  - made the previously reserved Phase 1.x shared surface family concrete in
    `src/world_model/sim_synth_physics/`:
    - `TaskMeasurementSurface`
    - `SceneHierarchyState`
    - `DifferentiablePhysicsProviderState`
    - `SurrogatePhysicsProviderState`
  - added the paired bounded receipt family:
    - `TaskMeasurementReceipt`
    - `SimRealGapReceipt`
    - `BackendMismatchReceipt`
    - `SurrogatePhysicsReceipt`
    - `SurrogateCalibrationReceipt`
  - wired those surfaces into `SimSynthPhysicsWorldState`, compiler artifact
    refs, receipt inventory, runtime loop results, training-feedback manifests,
    and emitted runtime artifact files
  - made the Habitat-style simulator/task split explicit with:
    - `SimulatorBackendContractState`
    - `TaskDefinitionContractState`
  - added CPU-local camera geometry utilities under
    `src/world_model/sim_synth_physics/utils/camera_geometry.py`
  - added `VectorizedSimRunner` / `VectorizedSimBatchResult` as an honest
    sequential local batch facade for later provider-season vectorization work
- **Why this matters**:
  - the Phase 1.x return leg is no longer merely a docs reservation; the first
    cross-subsystem joints now exist in code and serialize through the live
    compiler/runtime path
  - the sim→task→measurement handoff is now explicit rather than inferred from
    agenda and backend fields after the fact
  - the transfer/surrogate receipts are deliberately bounded and honest
    (`estimated`, `contract_reserved`, `not_calibrated`) while RunPod and real
    provider evidence remain absent
  - the geometry and batch utilities pull two explicitly local backlog items
    forward without reopening any frozen Phase B math or pretending GPU bring-up
    happened
- **Verification**:
  - `python3 -m compileall src/world_model/sim_synth_physics`
  - `python3 -m pytest tests/test_sim_synth_phase1x_surfaces.py tests/test_sim_synth_camera_geometry.py tests/test_sim_synth_physics_world_model.py tests/test_sim_synth_vectorized_runtime.py -q`

## 2026-05-18 - Phase 2 final local pocket: LeRobot projection adapter parity

- **Changed**:
  - completed the previously documented but missing
    LeRobot → `VisionBackboneProjectionSample` adapter path in
    `src/dataset_bridges/lerobot_perception_adapter.py`
  - added:
    - `vision_backbone_projection_sample_from_lerobot_step(...)`
    - `vision_backbone_projection_samples_from_episode(...)`
    - `adapt_lerobot_episodes_for_vision_backbone_projection(...)`
  - extended `scripts/smoke_test_vision_backbone_projection_seam.py` so the
    first promotion-chain seam now accepts the same local intake grammar as the
    other proof lanes:
    - `synthetic`
    - `mock_lerobot_droid`
    - `local_lerobot_rows`
  - added adapter coverage plus local-row-bundle smoke coverage for the
    projection proof lane
- **Why this matters**:
  - the repo already claimed a LeRobot → projection adapter path in module
    docs, but only the evidence-fusion and temporal adapters were actually live
  - closing that gap gives the first promotion-chain seam the same cheap local
    intake path as the later seams before the implementation center returns to
    Phase 1.x
  - the projection labels remain explicit CPU-safe proxies
    (`camera_slot_proxy`), so this is still provisional plumbing proof rather
    than object-identity or provider-readiness evidence
- **Local run**:
  - `python3 scripts/smoke_test_vision_backbone_projection_seam.py --steps 30 --data-source mock_lerobot_droid --artifact-dir artifacts/phase2_local_proof_of_life/vision_backbone_projection_mock_lerobot_30 --require-loss-decrease`
    produced an ignored local artifact bundle with initial validation loss
    `5.8005`, best validation loss `5.7418`, `3` training receipts, `3`
    validation receipts, and `1` benchmark receipt
- **Phase handoff**:
  - no real local LeRobot row bundle is currently present in the workspace, so
    the remaining tiny-real-data proof is now opportunistic rather than the
    reason to keep Phase 2 as the active local center
  - after this final local pocket, the implementation center returns to the
    queued Phase 1.x Sim / Synth / Physics return leg
- **Verification**:
  - `python3 -m ruff check src/dataset_bridges/lerobot_perception_adapter.py scripts/smoke_test_vision_backbone_projection_seam.py tests/test_lerobot_perception_adapter.py tests/test_vision_backbone_projection_proof_of_life_smoke.py` (pass)
  - `python3 -m pytest -q tests/test_lerobot_perception_adapter.py tests/test_vision_backbone_projection_proof_of_life_smoke.py` (`49 passed`)
  - `python3 -m compileall src/ && python3 -m pytest tests/ -q` (`1628 passed, 3 skipped, 24 warnings`)

## 2026-05-18 - Phase 2 closure audit + live semantic-bridge receipts

- **Changed**:
  - added `docs/economic_world_model/phase2_closure_assessment.md`
  - updated `phase2_closure_standard.md` with the current branch read
  - added live compiler emission for `SemanticBridgeReceipt` across the active
    `sim_synth`, `embodiment`, `annotation`, and `economic` bridge family
  - `compile_perception_grounding_with_receipts(...)` now returns those bridge
    receipts alongside the rest of the live receipt family
  - extended tests to assert that all four active bridge kinds emit typed
    receipts with bounded quality/usefulness scores
- **Why this matters**:
  - the closure audit found one final narrow internal seam worth removing:
    semantic bridge receipts were typed but not yet live
  - with that landed, the audited Phase 2 structural sheet now reads as
    Category A `0`, Category C `0`; the remaining blockers are honest Category B
    provider / GPU / real-data / calibration / held-out-evidence items
  - this lets the roadmap distinguish optional local hardening from the work
    that actually has to keep the phase open
- **Verification**:
  - `python3 -m ruff check src/world_model/perception_grounding/compiler.py tests/test_embodiment_shadow_consumer.py` (pass)
  - `python3 -m pytest -q tests/test_embodiment_shadow_consumer.py tests/test_perception_grounding_compiler.py` (`39 passed`)
  - `python3 -m compileall src/ && python3 -m pytest tests/ -q` (`1623 passed, 3 skipped, 24 warnings`)

## 2026-05-18 - Phase 2: local vision-backbone projection proof artifacts

- **Changed**:
  - added `scripts/smoke_test_vision_backbone_projection_seam.py`, a local CPU
    proof-of-life lane for `VisionBackboneProjectionSeam`
  - the new lane emits the same durable artifact family used by the earlier
    EvidenceFusion and V-JEPA temporal proofs:
    - persistent seam checkpoint
    - `perception_seam_metric_report_v1`
    - provisional `perception_benchmark_evidence_v1`
    - `training_runtime_manifest_v1`
    - full training / validation / benchmark receipts
  - added `tests/test_vision_backbone_projection_proof_of_life_smoke.py`
    covering artifact emission, explicit promotion hold, and manifest posture
- **Why this matters**:
  - `vision_backbone_projection` is the first seam in the current Phase 2
    promotion chain, so it should not remain structurally less mature than the
    downstream local proof lanes while provider bring-up is deferred
  - this turns the future DINOv2/SigLIP bring-up window into an evidence-input
    problem rather than a local artifact-plumbing problem
  - the emitted evidence stays synthetic, provisional, and explicitly
    `promotion_eligible: false`
- **Local run**:
  - `python3 scripts/smoke_test_vision_backbone_projection_seam.py --steps 40 --artifact-dir artifacts/phase2_local_proof_of_life/vision_backbone_projection_synth_40 --require-loss-decrease`
    produced an ignored local artifact bundle with initial validation loss
    `5.7407`, best validation loss `5.2823`, `4` training receipts, `4`
    validation receipts, and `1` benchmark receipt
- **Verification**:
  - `python3 -m ruff check scripts/smoke_test_vision_backbone_projection_seam.py tests/test_vision_backbone_projection_proof_of_life_smoke.py src/training/perception_seam_data.py src/training/perception_seam_benchmarks.py tests/test_perception_seam_training.py` (pass)
  - `python3 -m pytest -q tests/test_vision_backbone_projection_proof_of_life_smoke.py tests/test_perception_seam_training.py` (`32 passed`)

## 2026-05-18 - Phase 2: local vision-backbone projection training lane

- **Changed**:
  - added first-class `VisionBackboneProjectionSample`,
    `VisionBackboneProjectionBatch`, and
    `VisionBackboneProjectionDataset` support to
    `src/training/perception_seam_data.py`
  - added synthetic projection-sample generation plus a dedicated loader
    factory, so `vision_backbone_projection` no longer depends on ad hoc batch
    objects just to exercise its trainer path locally
  - added `VisionBackboneProjectionBenchmark` with identity-retrieval,
    scene-retrieval, and cross-provider-alignment metrics, and registered it
    in the seam benchmark registry
  - extended `tests/test_perception_seam_training.py` with loss, dataset,
    loader, and benchmark coverage for the projection lane
- **Why this matters**:
  - the Phase 2 promotion dependency chain starts with
    `vision_backbone_projection`, but before this pass the repo had a seam and
    loss without a first-class local training-data / benchmark lane
  - with RunPod/provider bring-up deferred, the useful local move is to remove
    future structural friction now rather than pretend GPU-backed projection
    evidence is near
  - later DINOv2/SigLIP provider runs can now land on a typed trainer path
    instead of inventing one during the scarce GPU season
- **Verification**:
  - `python3 -m ruff check src/training/perception_seam_data.py src/training/perception_seam_benchmarks.py tests/test_perception_seam_training.py` (pass)
  - `python3 -m pytest -q tests/test_perception_seam_training.py` (`31 passed`)
  - `python3 -m compileall src/training/perception_seam_data.py src/training/perception_seam_benchmarks.py tests/test_perception_seam_training.py -q` (pass)
  - `python3 -m compileall src/ && python3 -m pytest tests/ -q` (`1621 passed, 3 skipped, 24 warnings`)

## 2026-05-11 - Phase 2: local perception proof-of-life artifacts

- **Changed**:
  - fixed a fresh-process import cycle between
    `src.training.perception_seam_data` and the Perception / Grounding package
    export path by making annotation benchmark evaluation import lazy inside
    `benchmark_evidence_emitter.py`
  - upgraded `scripts/smoke_test_perception_seam_training.py` from a loose JSON
    smoke into a local CPU EvidenceFusion proof-of-life producer that emits:
    - persistent seam checkpoint under the chosen artifact directory
    - `perception_seam_metric_report_v1`
    - provisional `perception_benchmark_evidence_v1`
    - `training_runtime_manifest_v1`
    - full training / validation / benchmark receipts
  - added `scripts/perception_proof_of_life_utils.py` to generate
    deterministic DROID-shaped mock LeRobot replay episodes for local adapter
    verification
  - added a `--require-loss-decrease` guard and explicit initial / best /
    final validation loss accounting
  - added a `--data-source mock_lerobot_droid` mode that generates DROID-shaped
    mock LeRobot episodes, passes them through the LeRobot perception adapter,
    and then trains EvidenceFusion locally without requiring external data
  - added `scripts/smoke_test_vjepa_temporal_seam.py`, a matching local CPU
    proof-of-life lane for `VJEPATemporalAlignmentSeam` that emits the same
    typed artifact family and supports synthetic or mock-LeRobot temporal
    windows
  - both proof scripts now accept `--data-source local_lerobot_rows` plus a
    local JSON/JSONL LeRobot-like row bundle path, so a tiny external-data
    proof can reuse the same adapter → seam → trainer → manifest path without
    requiring a new dependency stack first
  - added focused tests for the fresh import path and typed artifact emission
    for both EvidenceFusion and V-JEPA temporal proof scripts
- **Why this matters**:
  - this lands the cheap local Phase 2 prototype-train proof-of-life lane
    without spending GPU budget and without pretending promotion is near
  - the local runs now prove that both the EvidenceFusion and V-JEPA temporal
    seams can train through the real trainer path and emit manifest/evidence
    artifacts in the same vocabulary later GPU/provider runs will use
  - the new local row-bundle intake path makes the next cheap external-data
    proof executable from a local LeRobot export without turning HuggingFace
    or GPU bring-up into a prerequisite for Phase 2 local progress
  - the emitted evidence remains synthetic, provisional, and explicitly
    `promotion_eligible: false`; a real `droid_100` / provider-backed run is
    still future work
- **Local run**:
  - `python3 scripts/smoke_test_perception_seam_training.py --steps 80 --artifact-dir artifacts/phase2_local_proof_of_life/evidence_fusion_80 --require-loss-decrease`
    produced an ignored local artifact bundle with initial validation loss
    `1.1481`, best validation loss `1.0016`, `16` training receipts, `8`
    validation receipts, `1` benchmark receipt, and provisional benchmark
    evidence
  - `python3 scripts/smoke_test_perception_seam_training.py --steps 40 --data-source mock_lerobot_droid --artifact-dir artifacts/phase2_local_proof_of_life/mock_lerobot_droid_40 --require-loss-decrease`
    exercised the LeRobot adapter path with DROID-shaped mock data; initial
    validation loss was `1.1966`, best validation loss was `1.1252`, with `8`
    training receipts, `4` validation receipts, and `1` benchmark receipt
  - `python3 scripts/smoke_test_vjepa_temporal_seam.py --steps 40 --data-source synthetic --artifact-dir artifacts/phase2_local_proof_of_life/vjepa_temporal_synth_40 --require-loss-decrease`
    produced a local temporal proof bundle with initial validation loss
    `114.1529`, best validation loss `72.8604`, `4` training receipts, `4`
    validation receipts, and `1` benchmark receipt
  - `python3 scripts/smoke_test_vjepa_temporal_seam.py --steps 30 --data-source mock_lerobot_droid --artifact-dir artifacts/phase2_local_proof_of_life/vjepa_temporal_mock_lerobot_30 --require-loss-decrease`
    exercised the LeRobot adapter temporal path with DROID-shaped mock data;
    initial validation loss was `169.5367`, best validation loss was
    `128.8715`, with `3` training receipts, `3` validation receipts, and `1`
    benchmark receipt
- **Verification**:
  - `python3 -m ruff check scripts/perception_proof_of_life_utils.py scripts/smoke_test_perception_seam_training.py scripts/smoke_test_vjepa_temporal_seam.py src/world_model/perception_grounding/benchmark_evidence_emitter.py tests/test_perception_seam_proof_of_life_smoke.py tests/test_vjepa_temporal_proof_of_life_smoke.py` (pass)
  - `python3 -m ruff format --check scripts/perception_proof_of_life_utils.py scripts/smoke_test_perception_seam_training.py scripts/smoke_test_vjepa_temporal_seam.py src/world_model/perception_grounding/benchmark_evidence_emitter.py tests/test_perception_seam_proof_of_life_smoke.py tests/test_vjepa_temporal_proof_of_life_smoke.py` (pass)
  - `python3 -m pytest tests/test_perception_seam_proof_of_life_smoke.py tests/test_vjepa_temporal_proof_of_life_smoke.py tests/test_lerobot_perception_adapter.py tests/test_perception_seam_training.py tests/test_perception_benchmark_evidence_emitter.py tests/test_provider_adapter_benchmark_evidence_emitter.py -q`
    (`81 passed`)

## 2026-05-11 - Phase 2: provider-adapter benchmark evidence emitter

- **Changed**:
  - extended `src/world_model/perception_grounding/benchmark_evidence_emitter.py`
    with provider-adapter benchmark evidence emission for
    `vision_backbone_projection`, `sam_calibration`,
    `depth_metric_calibration`, and `vjepa_temporal_alignment`
  - added `scripts/emit_perception_provider_adapter_benchmark_evidence.py` so
    provider-adapter evidence can be emitted from persisted
    `ProviderInvocationReceipt` payloads
  - linked optional `training_runtime_manifest_v1` and external metric-report
    references into emitted evidence metadata instead of changing the manifest
    schema
  - exported the provider-adapter emitter from the Perception / Grounding
    package and added focused tests for receipt-only provisional evidence,
    non-provisional metric reports, and the CLI path
- **Why this matters**:
  - this closes the provider-specific benchmark artifact-producer gap without
    overstating readiness: receipt-only evidence stays provisional by default
    and records `promotion_claim: not_implied_by_emitter`
  - provider adapter promotion inputs are now inspectable and repeatable across
    DINO/SigLIP projection, SAM calibration, depth calibration, and V-JEPA
    temporal alignment
  - Phase 2 remains the active implementation center. This does not bring up
    real GPU providers and does not claim provider-adapter promotion; it creates
    the artifact lane those future provider/GPU runs can feed.
- **Verification**:
  - `python3 -m ruff check src/world_model/perception_grounding/benchmark_evidence_emitter.py src/world_model/perception_grounding/__init__.py scripts/emit_perception_provider_adapter_benchmark_evidence.py tests/test_provider_adapter_benchmark_evidence_emitter.py` (pass)
  - `python3 -m pytest tests/test_provider_adapter_benchmark_evidence_emitter.py -q`
    (`3 passed`)
  - `python3 -m compileall src/world_model/perception_grounding scripts/emit_perception_provider_adapter_benchmark_evidence.py tests/test_provider_adapter_benchmark_evidence_emitter.py -q` (pass)
  - `python3 -m compileall src/ && python3 -m pytest tests/ -v`
    (`1610 passed, 3 skipped, 24 warnings`)

## 2026-05-11 - Phase 2: receipt-backed runtime provider tokens

- **Changed**:
  - tightened the Perception / Grounding compiler's benchmark-token source
    selection so runtime `vision_backbone_projection` and
    `vjepa_temporal_alignment` outputs become `provider_backed` annotation/export
    evidence only when the matching `ProviderInvocationReceipt` reports
    `success` without fallback
  - expanded the provider surface to replace the vision stub with
    `dinov2_vit_l_14` when real backbone features are supplied, and to expose
    live SAM, depth, and V-JEPA input/seam posture in
    `runtime_provider_inputs`
  - padded default V-JEPA WM object tokens to the seam's declared
    `d_wm_token`, so temporal alignment can run from scene-graph object tokens
    without explicit caller-provided WM tokens
  - added focused compiler tests for successful DINO projection token export,
    successful V-JEPA temporal token export, and failed projection fallback
- **Why this matters**:
  - this closes the next named Phase 2 gap after benchmark-evidence emission:
    benchmark object tokens are no longer mainly an explicit compile-time
    injection path
  - provider-backed token provenance is now receipt-backed runtime truth, not a
    topology claim; failed/skipped seams still produce heuristic/provisional
    evidence and cannot promote annotation or graph evidence by accident
  - Phase 2 remains the active implementation center. This does not bring up
    real GPU DINOv2/V-JEPA providers or claim promotion readiness; it makes the
    compiler path honest when those provider tensors are supplied.
- **Verification**:
  - `python3 -m ruff check src/world_model/perception_grounding/compiler.py tests/test_perception_grounding_compiler.py` (pass)
  - `python3 -m pytest tests/test_perception_grounding_compiler.py -q`
    (`18 passed`)
  - `python3 -m compileall src/ && python3 -m pytest tests/ -v`
    (`1607 passed, 3 skipped, 24 warnings`)

## 2026-05-11 - Phase 2: annotation-export benchmark evidence emitter

- **Changed**:
  - added `src/world_model/perception_grounding/benchmark_evidence_emitter.py`
    to turn persisted `annotation_export_v2` files into typed
    `perception_benchmark_evidence_v1` artifacts
  - added `scripts/emit_perception_annotation_benchmark_evidence.py` so
    scene-graph transformer and annotation-bridge benchmark evidence can be
    emitted repeatably from the CLI
  - exported the new emitter from the Perception / Grounding package
  - added focused tests covering provider-backed evidence, provisional
    heuristic-token blocking, and the CLI path
  - updated the Phase 2 roadmap notes so the next priority is now runtime
    provider-backed token production, followed by provider-specific benchmark
    artifact producers
- **Why this matters**:
  - this closes the first named Phase 2 gap after the annotation-export and
    benchmark-evidence contracts: benchmark evidence is no longer only a data
    class or an in-memory evaluator, it has a routine persisted artifact path
  - the emitter preserves token provenance, checkpoint reference status, and an
    explicit `promotion_claim: not_implied_by_emitter` marker, so fresh or
    provisional runs cannot masquerade as promotion-grade evidence
  - Phase 2 remains the active implementation center; this is evidence
    production for Perception / Grounding, not a topology change or a Phase 1.x
    diversion
- **Verification**:
  - `python3 -m compileall src/world_model/perception_grounding scripts/emit_perception_annotation_benchmark_evidence.py tests/test_perception_benchmark_evidence_emitter.py -q` (pass)
  - `python3 -m ruff check src/world_model/perception_grounding/benchmark_evidence_emitter.py scripts/emit_perception_annotation_benchmark_evidence.py tests/test_perception_benchmark_evidence_emitter.py src/world_model/perception_grounding/__init__.py` (pass)
  - `python3 -m pytest -q tests/test_perception_benchmark_evidence_emitter.py tests/test_annotation_bridge_projection.py tests/test_perception_grounding_compiler.py tests/test_perception_grounding_neural_seams.py tests/test_embodiment_shadow_consumer.py` (`113 passed`)
  - `python3 -m compileall src/ && python3 -m pytest tests/ -v` (`1604 passed, 3 skipped, 24 warnings`)

## 2026-05-11 - Doctrine: GR00T / VIRAL / DoorMan borrowing pass

- **Changed**:
  - created
    `docs/economic_world_model/doctrine_groot_visualsim2real_borrowings.md`
    to map GR00T-VisualSim2Real patterns into Ixion without treating GR00T
    as topology, ontology, or Isaac sovereignty
  - updated `docs/economic_world_model/roadmap.md` with a GR00T borrowing
    track under the external architecture / sim-to-online / embodiment-prep
    area
  - updated `docs/economic_world_model/multi_wm_architecture_plan.md` with
    GR00T as an admissible external-pattern source for teacher/student seams,
    typed run manifests, domain-randomization receipts, dataset-reset
    profiles, eval/export gates, and Sim-to-Embodiment transfer receipts
  - updated `docs/economic_world_model/perception_external_data_roadmap.md`
    with Phase 2 deployable observation discipline: camera bundles,
    egocentric profiles, extrinsics randomization receipts,
    observation-delay/degraded-observation surfaces, and visual augmentation
    provenance
  - updated `docs/actuation_embodiment_world_model.md` with a GR00T / VIRAL /
    DoorMan subsection under external architecture borrowings
  - lightly cross-referenced future teacher/student sim-to-real fields in
    `docs/agent_ergonomics/run_manifest_schema.md`
- **Why this matters**:
  - GR00T is valuable as a concrete sim-to-real training/eval/config plant,
    but the repo needs those patterns routed through existing WMs, receipts,
    manifests, and promotion gates
  - the pass preserves Phase 2 Perception / Grounding as the active
    implementation center while making the later Phase 1.x Sim / Synth /
    Physics return legible after Phase 2
  - it keeps Isaac Lab / Isaac Sim as backend/provider lanes, not owners of
    truth, and keeps PPO/DAgger/ResNet/ONNX as examples rather than stack-wide
    mandates
- **Verification**:
  - `python3 -m compileall src/` (pass)
  - `python3 -m pytest tests/ -v` (`1601 passed, 3 skipped, 24 warnings`)

## 2026-04-12 — Doctrine: admissible borrowings from In-Place TTT and HALO

- **Changed**:
  - updated `docs/economic_world_model/multi_wm_architecture_plan.md` with a
    new doctrine subsection on admissible neural-shaping borrowings from
    In-Place TTT and HALO
  - added a future-facing `WM-local shaping networks` note in the same
    architecture doc, explicitly as a far-future admissible direction rather
    than an implementation commitment
  - updated
    `docs/economic_world_model/doctrine_economic_wm_future_architecture.md`
    with a short multi-timescale reinforcement covering bounded adaptive
    seams, abstention heads, and envelope-shaped downstream influence
  - updated
    `docs/economic_world_model/doctrine_bio_neuro_architecture_inspirations.md`
    with a local anomaly-head abstention note
- **Why this matters**:
  - this records what can be borrowed from external neural methods without
    letting them become architecture templates or new master ontologies
  - it sharpens the repo's doctrine around subsystem-local plasticity,
    calibrated abstention, slow-versus-fast separation, and bounded future
    shaping networks
  - it preserves typed ownership, typed receipts, topological separation, and
    promotion discipline while making later subsystem-native neural seams more
    concrete
- **Verification**:
  - targeted docs check via `git diff --check -- docs/economic_world_model/multi_wm_architecture_plan.md docs/economic_world_model/doctrine_economic_wm_future_architecture.md docs/economic_world_model/doctrine_bio_neuro_architecture_inspirations.md docs/economic_world_model/progress_log.md docs/economic_world_model/implementation_notes.md`

## 2026-04-12 — Sim / Synth / Physics WM: SIM1 tactic note (docs-only)

- **Changed**:
  - updated `docs/economic_world_model/multi_wm_architecture_plan.md` with a
    short doctrine-safe note on lane-specific tactics worth borrowing from
    SIM1 while keeping SIM1 explicitly subordinate to our architecture
  - updated `docs/economic_world_model/roadmap.md` with the matching
    Phase 1.x roadmap reminder
  - tactics called out:
    - real runnable provider/runtime lane discipline
    - physics-aligned world instantiation for physics-sensitive lanes
    - staged `generate -> smooth -> replay -> filter` branch production
    - explicit admission/reject filtering with typed reject receipts
    - replay-validity / task-consistency checks for drift and mismatch
    - render/materialization as downstream lane, not sovereign center
    - replay/export discipline and training-worthiness gating
- **Why this matters**:
  - this sharpens the Sim / Synth / Physics WM with concrete lane tactics
    without letting an external deformable-data engine define our ontology,
    ownership boundaries, or receipt structure
- **Verification**:
  - targeted docs check via `git diff --check -- docs/economic_world_model/multi_wm_architecture_plan.md docs/economic_world_model/roadmap.md docs/economic_world_model/progress_log.md docs/economic_world_model/implementation_notes.md`

## 2026-04-12 — Nightly audit date-parse hardening (additive verification scaffold)

- **Changed**:
  - hardened `scripts/economic_world_model/nightly_audit.py` progress-log date parsing so nightly selection reads dated headings that include trailing titles (for example `## YYYY-MM-DD — ...`) and level-3 dated headings used in historical notes.
  - added regression coverage in `tests/test_economic_world_model_nightly_audit.py` for:
    - dated H2 headings with suffix text
    - dated H3 headings with suffix text
  - refreshed nightly audit artifacts:
    - `artifacts/economic_world_model/nightly_audit_summary.json`
    - `artifacts/economic_world_model/nightly_audit_summary.md`
- **Verification**:
  - `python3 -m pytest -q tests/test_economic_world_model_nightly_audit.py` (`8 passed`)
  - `PYTHONPYCACHEPREFIX=/tmp/pycache python3 -m compileall scripts/economic_world_model -q` (pass)
  - `python3 scripts/economic_world_model/nightly_audit.py --output-json artifacts/economic_world_model/nightly_audit_summary.json --output-markdown artifacts/economic_world_model/nightly_audit_summary.md` (pass; status `ok`)
  - embedded verification in audit remained green: `./scripts/agent/verify.sh`, compileall, targeted runtime/econ pytest bundle
- **Next recommended task**:
  1. keep nightly docs + verification refresh cadence while `next_task.id` remains `audit_only`
  2. when a safe additive task appears, prioritize live-path sidecar/contract emission before detached helper additions

## 2026-04-11 — Nightly audit refresh (docs-only, no safe additive scaffold)

- **Changed**:
  - refreshed nightly audit artifacts:
    - `artifacts/economic_world_model/nightly_audit_summary.json`
    - `artifacts/economic_world_model/nightly_audit_summary.md`
  - recorded the current nightly posture as docs-only with no code-scaffold
    delta in this pass.
- **Verification**:
  - `python3 scripts/economic_world_model/nightly_audit.py --output-json artifacts/economic_world_model/nightly_audit_summary.json --output-markdown artifacts/economic_world_model/nightly_audit_summary.md`
  - audit result: `status: ok`, `safe for automatic execution: no`,
    `next task classification: docs_only`
  - embedded verification in audit: `./scripts/agent/verify.sh` (pass),
    `python3 -m compileall src scripts/economic_world_model -q` (pass),
    targeted runtime/econ pytest bundle (pass)
- **Next recommended task**:
  1. keep nightly docs + verification refresh cadence until the audit reports a
     concrete safe additive scaffold
  2. when a safe task appears, prioritize live-path sidecar/contract emission
     over detached helper-only additions
  3. keep governed video-world-model ordering intact: real-video grounding and
     teacher-runtime hardening before training-backlog expansion

## 2026-04-08 — Phase 2 Perception / Grounding WM: Annotation-Bridge Lane + Persistent Benchmark Evidence

- **Changed**: completed the first bounded annotation-export projection lane and tightened promotion discipline around it:
  - added `AnnotationBridgeProjectionSeam` in `src/world_model/perception_grounding/neural_seams.py`
  - added `annotation_bridge_projection_loss` and trainer dispatch support in `src/training/perception_seam_losses.py` and `src/training/perception_seam_trainer.py`
  - added annotation-export seam evaluation in `src/training/perception_seam_data.py`, including explicit provisional gating when evidence is derived from heuristic object tokens rather than provider-backed features
  - wired compiler shadow execution, receipt emission, and promotion resolution for the annotation bridge in `src/world_model/perception_grounding/compiler.py`, `receipts.py`, and `promotion.py`
- **Changed**: turned benchmark evidence from ad hoc dicts into a typed persisted artifact:
  - added `src/world_model/perception_grounding/benchmark_evidence.py`
  - annotation exports now preserve object-token provenance (`source_kind`, `truth_class`, `provider_id`, provisional flag) in `src/world_model/perception_grounding/annotation_export.py`
  - graph transformer, annotation bridge, and provider-adapter promotion logic now accepts persisted benchmark evidence and stays in shadow monitoring when evidence is missing or provisional
  - compiler benchmark-token selection now prefers provider-backed sources and only falls back to heuristic scene-graph tokens under an explicit non-promoting posture
- **Verification**:
  - `python3 -m compileall src`
  - `python3 -m ruff check src/world_model/perception_grounding/annotation_export.py src/world_model/perception_grounding/benchmark_evidence.py src/world_model/perception_grounding/compiler.py src/world_model/perception_grounding/promotion.py src/world_model/perception_grounding/receipts.py src/world_model/perception_grounding/__init__.py src/training/perception_seam_data.py tests/test_annotation_bridge_projection.py tests/test_perception_grounding_compiler.py tests/test_perception_grounding_neural_seams.py`
  - `python3 -m pytest tests/test_annotation_bridge_projection.py tests/test_perception_grounding_compiler.py tests/test_perception_grounding_neural_seams.py tests/test_perception_grounding_world_model.py -q`
  - result: `138 passed`
- **Next recommended task**:
  1. make graph-transformer benchmark evidence routine and non-provisional by generating it directly from persisted annotation-export artifacts rather than only supporting the artifact contract structurally
  2. turn benchmark object-token sourcing into a real runtime artifact path from vision-backbone / V-JEPA provider outputs instead of relying on explicit compile-time injection
  3. add provider-specific benchmark artifact producers and trainer-manifest linkage for `vision_backbone_projection`, `sam_calibration`, `depth_metric_calibration`, and `vjepa_temporal_alignment`, then promote in dependency order: vision backbone projection → scene graph transformer → annotation bridge projection → provider calibrators

## 2026-04-04 — Doctrine: Autoencoder / Codebook Posture (stack + Economic WM + Embodiment)

- **Updated** `docs/economic_world_model/neuralization_bridge_doctrine.md`: new § Autoencoder / Codebook Posture Across the Stack—layer taxonomy table; Perception, Semantic→Economic (Perceiver primary; optional bounded auxiliaries only), Embodiment, Sim (light), explicit non-role for transport/meta-governance
- **Updated** `docs/economic_world_model/doctrine_economic_wm_future_architecture.md`: § Autoencoder / Manifold-Compression Posture (bounded auxiliary yes, backbone no; DS3M/RED-SDS primary preserved); placements before slow projection, motifs, meso/slow summarization; explicit non-replacements for estimator/dynamics/allocator/governance/transport/meta-regal; staged neuralization + research-bucket notes that AE research is auxiliary to A/B
- **Updated** `docs/actuation_embodiment_world_model.md`: short § Autoencoder / Codebook Use Inside Embodiment—supports inverse/retargeting/ACT/diffusion lanes, does not replace them
- **Left** `multi_wm_architecture_plan.md` unchanged (avoid redundant topology restatement; bridge + Economic + Embodiment docs own the delta)

## 2026-04-04 — Doctrine: Bio/Neuro Architecture Inspirations + Economic WM PINN Posture

- **Created** `docs/economic_world_model/doctrine_bio_neuro_architecture_inspirations.md`:
  - Six bio/neuro organizational principles evaluated against the multi-WM topology: efference copy, active sensing, neuromodulation/allostasis, plasticity gating, motor synergies + interoception, immune-style anomaly governance
  - Each candidate: exact WM insert point, architecture family, layer classification (reuses `neuralization_bridge_doctrine.md` taxonomy), typed surfaces, phase timing, explicit boundary discipline
  - Cross-cutting neural architecture table consistent with existing bridge doctrine
  - Sequencing section: preserve now (efference copy, active sensing, plasticity gating); defer until lower-WM maturity (regime broadcast / synergies); post–Sep-2026 (immune composition, hypernetwork regime composition)
- **Added** PINN posture subsection to `doctrine_economic_wm_future_architecture.md`:
  - Core stance: constraint-informed Economic submodule yes, PINN-shaped Economic backbone no
  - Good placements: slow-manifold invariant residuals, meso-timescale reservoir-flow, battery/thermal/wear/queue evolution, counterfactual resource-transition rollouts, invariant regularization
  - Bad placements: allocator, governance transport, meta-regal composition, WM-to-WM transport, anything implying clean PDE economics
- **Added** minimal cross-references in `multi_wm_architecture_plan.md` and `docs/actuation_embodiment_world_model.md` (pointers only; no restatement of the bio note or subsystem catalogs)
- **Anti-redundancy**: canonical topology, bridge taxonomy, embodiment decomposition, meta-regal structure, and Economic multi-timescale design stay in their existing docs

## 2026-04-04 — Phase 2 Perception / Grounding WM: Embodiment Shadow Consumer + Full Receipt Family + Subsystem Discipline

- **Created** `src/world_model/perception_grounding/embodiment_shadow_consumer.py`:
  - `EmbodimentShadowSurface`: typed perception→embodiment shadow surface with per-object action relevance (reachability, obstruction, affordance feasibility, contact preconditions, misalignment risk), scene-level summaries, body-object engagement, resource/deployment readiness, provider truth posture, evidence quality for embodiment trust
  - `ObjectActionRelevance`: per-object body-relevant action summary
  - `EmbodimentShadowConsumptionReceipt`: receipt for each shadow consumption pass
  - `consume_perception_for_embodiment()`: typed consumer entry point
  - Shadow/advisory posture only — no control authority, no planner sovereignty
  - Reduced-quality but honest behavior when providers are unavailable
  - Shaped by `docs/actuation_embodiment_world_model.md` doctrine (6-subsystem Embodiment WM)
- **Deepened** receipt emission in `compiler.py`:
  - Compiler now emits **all 8 receipt types** live on every compilation pass:
    `ProviderAvailabilityReceipt`, `EvidenceFusionReceipt`, `ProviderInvocationReceipt`, `GroundingCalibrationReceipt`, `InferenceHeadroomReceipt`, `DeploymentResourceReceipt`, `TemporalGroundingReceipt`, `PerceptionContributionReceipt`
  - `compile_perception_grounding_with_receipts()` now extracts and returns full receipt family
  - Provider availability receipts cover all known providers with honest truth-class and install status
  - Deployment resource receipt identifies concrete bottlenecks (compute, battery, thermal, posture)
  - Grounding calibration receipt computes cross-provider agreement and spatial/temporal accuracy
  - Perception contribution receipt packages episode-level quality for future Economic WM consumption
- **Codified** internal subsystem decomposition in `__init__.py` module docstring:
  - 7 named subsystems: Object/Track Persistence, Scene Graph/Relation State, Temporal Grounding, Evidence Routing/Fusion, Affordance/Action-Relevance Bridge Surface, Provider/Runtime/Deployment-Resource Truth, Replay/Export/Bridge Registry Surfaces
  - Each subsystem documented with: canonical typed state owned, receipts emitted, neural successor path (architecture + capacity + training objective), downstream consumers, and explicit "NOT" boundary
  - Boundary rules enforced: no mother-latent, no provider-owned truth, no bridge-becomes-downstream-WM, no economic pre-collapse, no ungoverned fast→slow leakage
- **Created** `tests/test_embodiment_shadow_consumer.py` (20 tests):
  - Embodiment shadow consumer: typed output, per-object relevance, scene summaries, body-object engagement, resource readiness, provider truth, evidence quality, shadow posture, receipt completeness, serialization roundtrip, reduced-quality empty state, no sovereignty assertion
  - Full receipt family: all 8 receipt types present, provider availability covers all providers, grounding calibration metrics, deployment bottleneck identification, temporal persistence metrics, perception contribution for Economic WM, metadata completeness
  - Integration: compile → receipt family → embodiment shadow consumer end-to-end pipeline
- **No regressions**: all 97 existing perception/grounding tests pass, 20 new tests pass (117 total)
- **Phase 2 status**: 3 shadow consumers now wired (SimSynth, Annotation/VLA, Embodiment). Full receipt family live. Internal subsystem discipline codified. Maturity remains `shadow_runtime`.

## 2026-04-03 — WM Decomposition Standard Baseline + Sim/Synth/Physics GPU-Era Revisit Target + Execution Plane Standup

- **Updated** `docs/economic_world_model/multi_wm_architecture_plan.md`:
  - Expanded the Sim / Synth / Physics WM section (Recommended WM Set §3) from a sparse purpose/gaps summary to a full 10-subsystem decomposition with typed interfaces, neural structure candidates, timescale hierarchy, topological placement, and robostack/G1 contribution
  - Ten named subsystems: backend/runtime/provider surface, task/measurement/episode layer, scene/asset/materialization layer, branch planner/evaluator, sim-real gap/realism evaluator, fidelity/randomization/calibration allocator, render/diffusion/materialization lane, differentiable-physics provider lane, drift/calibration/backend mismatch evaluator, training-worthiness/synthetic-yield evaluator
  - Added Phase 1.x GPU-Era Subsystem Decomposition Revisit section between Phase 1 and Phase 2, explicitly marking the future standard without reopening Phase 1 implementation
  - Tied Habitat-derived adoption items to specific subsystems
- **Updated** `docs/economic_world_model/roadmap.md`:
  - Added GPU-era decomposition revisit callout to the Sim/Synth/Physics Habitat adoption track
  - Added WM Section Decomposition Standard section referencing the 9-point readiness template
- **Updated** `ROADMAP_STAGES_2_5.md`:
  - Added Multi-WM Decomposition Standard note explaining that all WM sections are now held to richer subsystem decomposition standards
- **Updated** `README.md`:
  - Expanded Sim/Synth/Physics WM description to list the 10 internal subsystems
  - Added paragraph on the multi-WM roadmap's movement toward canonical ownership, subsystem decomposition, typed receipts, and bounded neural seams
- **Created** RunPod execution plane: `codex_skills/runpod-gpu-execution/SKILL.md`, `docs/agent_ergonomics/runpod_execution_plane.md`, `scripts/runpod/` (ensure_cli, launch_pod, exec_remote, sync_up, sync_down, collect_billing, cleanup_idle)
- **Created** run manifest schema: `docs/agent_ergonomics/run_manifest_schema.md`, `configs/runpod/examples/`, `results/run_registry/README.md`
- **Created** roadmap execution companion: `codex_skills/roadmap-execution-companion/SKILL.md`, `docs/agent_ergonomics/roadmap_execution_companion.md`
- **Created** Feynman integration posture: `docs/agent_ergonomics/feynman_integration_posture.md`
- **Updated** `AGENTS.md` with execution plane guidance (local vs Codex cloud vs RunPod) and run manifest recording
- This re-baselines the standard for future WM roadmap sections and establishes the Sep 2026 execution model

## 2026-04-03 — Perception Seam Training Infrastructure + External Data Adapters

- **Created** `src/training/perception_seam_losses.py`:
  - Loss functions for all perception seam types: `evidence_fusion_loss`, `sam_calibration_loss`, `depth_metric_calibration_loss`, `vjepa_temporal_alignment_loss`, `vision_backbone_projection_loss`
  - `SeamLossResult` dataclass with total loss, component breakdown, and training metrics
  - Supervised/contrastive/predictive objectives (NOT direct RL)
- **Created** `src/training/perception_seam_data.py`:
  - Dataset classes: `ProviderAgreementDataset`, `EvidenceFusionDataset`, `SAMCalibrationDataset`, `DepthCalibrationDataset`, `VJEPATemporalDataset`
  - Typed sample dataclasses: `MultiProviderSample`, `ProviderObservation`, `VJEPATemporalSample`
  - Synthetic data generators for testing/verification
  - Data loader factory functions with proper collation
- **Created** `src/training/perception_seam_trainer.py`:
  - `PerceptionSeamTrainer`: training orchestrator with gradient accumulation, validation, checkpointing
  - Receipt emission: `SeamTrainingStepReceipt`, `SeamValidationReceipt`, `BenchmarkGateReceipt`
  - Early stopping, LR scheduling, benchmark gate integration
- **Created** `src/training/perception_seam_benchmarks.py`:
  - Per-seam benchmark evaluators: `EvidenceFusionBenchmark`, `SAMCalibrationBenchmark`, `DepthCalibrationBenchmark`, `VJEPATemporalBenchmark`
  - `BenchmarkGateResult` with promotion decision logic
- **Created** `docs/economic_world_model/perception_external_data_roadmap.md`:
  - GPU-honest classification of external data sources (DROID, Bridge V2, ALOHA, KITTI)
  - 3-level classification: adapter-usable (no GPU) → prototype-trainable (dev GPU) → promotion-credible (GPU required)
  - Doctrine updates for promotion credibility levels
- **Created** `src/dataset_bridges/lerobot_perception_adapter.py`:
  - `multi_provider_sample_from_lerobot_step`: LeRobot multi-camera step → `MultiProviderSample`
  - `vjepa_temporal_sample_from_episode_window`: episode window → `VJEPATemporalSample`
  - `FeatureExtractionConfig`: placeholder, flattened, or frozen_backbone strategies
  - `discover_camera_keys`: auto-discovers camera keys from DROID/Bridge/ALOHA observation formats
  - Dataset-level adapters for batch processing
- **Created** `tests/test_perception_seam_training.py` (26 tests):
  - Loss function correctness, data loader collation, benchmark evaluation
- **Created** `tests/test_lerobot_perception_adapter.py` (43 tests):
  - Camera key discovery for DROID/Bridge/ALOHA formats
  - Feature extraction strategies on CPU
  - Multi-provider sample conversion with realistic data shapes
  - V-JEPA temporal sample extraction with sliding windows
- This closes the "Seam Training Infrastructure" gap identified in Phase 2 planning
- Adapter work is adapter-usable now (no GPU); prototype-trainable requires `droid_100` subset; promotion-credible training requires GPU

## 2026-04-03 — Provider Adapter Neural Seams (Phase 2 Implementation)

- **Added** four provider adapter neural seams to `src/world_model/perception_grounding/neural_seams.py`:
  - `SAMCalibrationSeam` (~500K-2M params): calibrates SAM mask confidence, epistemic uncertainty, prompt satisfaction
  - `VisionBackboneProjectionSeam` (~1M params): 2-layer MLP projecting DINOv2/SigLIP features to WM token space
  - `DepthMetricCalibrationSeam` (~500K-1M params): learns scale/shift for metric depth + per-pixel uncertainty
  - `VJEPATemporalAlignmentSeam` (~2-5M params): cross-attention aligning V-JEPA temporal predictions to WM object tokens
- **Created** `src/world_model/perception_grounding/seam_registry.py`:
  - `PerceptionSeamRegistry` class: manages seam lifecycle (register, load, save, unload)
  - `SeamDescriptor` dataclass: tracks seam state (posture, checkpoint path, param count)
  - `create_default_registry()` factory: pre-registers all standard seam types
  - Checkpoint persistence and device placement support
- **Added** `resolve_provider_adapter_helper()` to `promotion.py`:
  - Resolver for per-provider adapter seams with `disabled|auto|required` posture
  - Demotion logic on evidence failure or benchmark gate revocation
- **Updated** `__init__.py`:
  - Exports all new seams, registry classes, and resolver function
- **Created** `tests/test_perception_grounding_neural_seams.py`:
  - 37 tests covering forward pass, batching, param counts, registry operations, promotion logic
- **Wired** seams into compiler in `compiler.py`:
  - Added `_invoke_provider_adapter_seam()` helper with receipt emission
  - Compiler accepts optional seam parameters (sam_calibration_seam, vision_backbone_projection_seam, etc.)
  - Compiler accepts optional provider inputs (sam_mask_features, backbone_features, depth_map, vjepa_tokens, etc.)
  - Seams invoked when promoted + inputs available; skipped otherwise
  - `ProviderInvocationReceipt` emitted for each seam invocation with status, latency, quality
  - `compile_perception_grounding_with_receipts()` returns all receipts including provider adapter receipts
- **Documented** training objectives for all seams in module docstring:
  - Each seam has primary, secondary, and auxiliary supervised objectives
  - Objectives are supervised/contrastive/predictive, NOT direct RL on task reward
  - Checkpoint governance via `PerceptionSeamRegistry`
- This completes the highest-leverage Phase 2 implementation work identified
- Next: seam training infrastructure, additional downstream consumers, benchmark gates

## 2026-04-03 — WM Section Readiness Standard + Scalable Imitation-Learning Pipelines

- **Added** WM Section Readiness Standard to `multi_wm_architecture_plan.md`:
  - 9-point template for all future WM sections (canonical mission, subsystem decomposition, typed surfaces, neural candidates, hyperparameter governance, topological placement, timescale hierarchy, robostack contribution, phase sequencing honesty)
  - Standard ensures all WM sections meet the decomposition rigor present in Economic WM and Embodiment/Actuation WM plans
  - Distinguishes inactive-but-structurally-rigorous vs vague "figure it out later" posture
- **Added** scalable imitation-learning pipelines to Embodiment/Actuation WM:
  - Ownership placement in Inverse-Dynamics/Retargeting Lane (Subsystem 4) and Joint Skill/Action Proposal Head (Subsystem 5)
  - Typed artifacts: DemonstrationIngestReceipt, RetargetingTraceBundle, ActionRecoveryReceipt, DatapackQualityReceipt, ImitationPriorSnapshot, ImitationDriftReceipt
  - Model families: ACT-style chunking, LeRobot interfaces, diffusion policy, inverse-dynamics heads, retargeting networks
  - Hyperparameter governance by WM constraints (DoF, contact richness, safety envelope, task family)
  - 5-stage promotion ladder: scripted fallback → imitation shadow → imitation advisory → benchmark-gated promotion → production recurrent
- **Updated** `docs/actuation_embodiment_world_model.md`:
  - Expanded Subsystem 4 (Inverse-Dynamics Lane) with imitation-learning pipeline functions and typed artifacts
  - Expanded Subsystem 5 (Action Proposal Head) with imitation integration and promotion ladder
  - New dedicated "Scalable Imitation-Learning Pipelines" section with full doctrine
  - Added UMI/Retargeting Patterns to "What We Borrow" section
- **Updated** `multi_wm_architecture_plan.md` Phase 3 section:
  - Added "Scalable Imitation-Learning Pipelines" subsection
  - Updated OSS dependency map with imitation learning deps (ACT, LeRobot, Diffusion Policy, UMI)
- **Updated** `.agent/claude_copilot.md`:
  - Changed implementation priority from Phase 1 (Sim/Synth/Physics) to Phase 2 (Perception/Grounding)
  - Phase 1 declared structurally closed on 2026-04-02; remaining blockers are external
  - Updated watch list and anti-patterns for Phase 2 focus
- All changes are doc-only and Phase 2-compatible (spec sharpening for Phase 3 target, not phase transition)

## 2026-04-03 — Embodiment / Actuation WM Specification (Doc-Only Pass)

- **Created** `docs/actuation_embodiment_world_model.md`:
  - Full canonical WM spec: mission, six core subsystems, typed interfaces, external-architecture borrowing logic, timescale hierarchy, real-robot-readiness mapping, phase sequencing, anti-patterns
  - Clarifies how our multi-WM topology differs from "single predictive model" framing
  - Specifies six subsystems: capability/embodiment state surface, contact/affordance graph builder, local contact dynamics model, inverse-dynamics/retargeting lane, joint skill/action proposal head, drift/calibration/cost evaluator
  - Proposes 8 typed interfaces: EmbodimentState, ContactAffordanceGraph, LocalDynamicsQuery/Forecast, InverseRetargetTrace, ActionProposalBundle, EmbodimentDriftSummary, CalibrationTargetSet, EmbodimentCostVector
  - Maps existing embryonic artifacts (EmbodimentProfile_v1 through CalibrationTargets_v1) to their producing WM subsystems
  - Documents borrowing logic for V-JEPA 2, LeRobot/ACT, Diffusion Policy, Isaac Lab, TD-MPC2 — all entering as bounded seams, not ontology replacements
  - Three-timescale hierarchy: fast inner loop (proprio/contact), mid-level (action chunks/dynamics), slow supervisory (selection/economics/governance)
  - Concrete readiness targets mapped to workcell task catalog (bin picking, peg-in-hole, fastener installation, kitting, tool change)
- **Modified** `README.md`: sharpened Embodiment WM entry with six-subsystem summary and doc reference
- **Modified** `ROADMAP_STAGES_2_5.md`: added Embodiment WM context block referencing full spec
- **Modified** `docs/embodiment_module.md`: added architectural context linking to Embodiment WM spec, explaining existing artifacts as embryonic WM outputs
- **Modified** `docs/motor_backends.md`: added architectural context connecting motor backends to Embodiment WM execution layer
- **Modified** `docs/isaac_integration_outline.md`: added architectural context for Isaac Lab as a bounded motor backend, not master ontology
- **Modified** `docs/economic_world_model/multi_wm_architecture_plan.md`: updated section 2 with six-subsystem summary, existing artifact mapping, gap list; updated Phase 3 section with typed interface targets and doc reference
- **Modified** `docs/economic_world_model/roadmap.md`: expanded Phase 3 preparatory section with full borrowing discipline and doc reference
- All changes are doc-only and Phase 2-compatible (spec sharpening for later phase target, not phase transition)

## 2026-04-03 — First Bounded Neural Seam + Receipt Emission (Claude Implementation Pass)

- **Created** `src/world_model/perception_grounding/neural_seams.py`:
  - `EvidenceFusionSeam(torch.nn.Module)` — real set-attention module (2-head MHA, d_model=32, ~10-50K params)
  - Replaces hardcoded 0.55/0.25/0.15/0.05 evidence weights at `promoted` promotion stage
  - `encode_provider_features()` — typed feature encoder for provider kind/availability/truth/belief signals
  - `heuristic_init()` classmethod, `describe()` metadata, `param_count()` introspection
- **Modified** `src/world_model/perception_grounding/compiler.py`:
  - `_evidence_routing()` now branches on `promotion_stage`:
    - `"heuristic_fallback"` → existing hardcoded weighted fusion
    - `"promoted"` + seam provided → neural seam forward pass
    - graceful fallback on neural seam error
  - `EvidenceFusionReceipt` emitted on every compilation (both paths)
  - `compile_perception_grounding_world_state()` accepts optional `evidence_fusion_seam=`
  - New `PerceptionCompilationResult(state, receipts)` dataclass
  - New `compile_perception_grounding_with_receipts()` function
- **Modified** `tests/test_perception_grounding_compiler.py`:
  - 9 new tests covering neural seam forward pass, batched input, masking, backward compatibility, promoted path, fallback behavior, receipt emission, compile_with_receipts, and seam introspection
- **Verification**: 12/12 compiler tests pass, 48/48 perception grounding tests pass, 1412/1412 full suite tests pass, compile clean
- **Significance**: This is the first time the Perception / Grounding WM has a real `torch.nn.Module` behind the promotion posture. The anti-heuristic-without-neuralization standard is now satisfied at the evidence fusion surface. Heuristic fusion is explicitly transitional with a real neural successor codepath.

## 2026-04-03 — Economic WM + Meta-Regal-Node + Embodiment Doctrine Pass

- **Created** `docs/economic_world_model/doctrine_economic_wm_future_architecture.md`:
  - Economic WM framed as typed allocator-governor for productive flow / dissipation / allocation
  - Not scalar reward head, dashboard, PnL tracker, or mother-latent
  - Multi-timescale design: fast/meso/slow-adiabatic variable split
  - Asymmetric upward/downward transport
  - Four-component internal decomposition: state estimator → dynamics → allocator → governance
  - Staged neuralization: typed ontology → neural estimation → dynamics → allocator → local compilers
  - Quant-inspired algorithmic imports: coherent risk, distributional Pareto, regime switching, risk budgeting, stress testing, execution-cost awareness
  - Superstatistics posture: keep multi-timescale and regime-mixing ideas, do not keep vague temperature metaphors
  - Research buckets: regime-aware state estimation, risk-aware Pareto allocation, superstatistical abstractions, adiabatic control, differentiable simulation coupling
  - Sovereignty clarification: Economic WM is first-class contributor, not sole governor
  - Intra-domain vs inter-domain Pareto distinction made explicit
- **Created** `docs/economic_world_model/doctrine_meta_regal_node_wm.md`:
  - Meta-regal-node WM as the governance-pluralism composition layer
  - Three governance levels: subsystem/local, domain governance, meta-governance
  - Two kinds of Pareto: intra-domain (within Economic WM) vs inter-domain (across governance nodes)
  - What the meta-layer must model: governance state, composition modes (Pareto/lexicographic/veto/advisory/confidence-weighted), transport (conflict/override/failure receipts)
  - Governance pluralism principle: no single domain ontology can silently redefine others
  - Staged neuralization: governance node neuralization before meta-composition learning
  - Anti-patterns: no governance collapse, no scalar governance score, no opaque meta-controller
- **Updated** `multi_wm_architecture_plan.md`:
  - Executive conclusion updated: Economic WM is first-class contributor, not sole sovereign
  - New "Future Economic WM Architecture" section with sovereignty clarification
  - Phase 7 rewritten as "Meta-Regal-Node Superposition / Control WM" with domain-governance composition, inter-domain Pareto, governance-pluralism principle
  - Meta-node section (#5) rewritten with three governance levels, two Pareto kinds, superposition rationale
  - Habitat adoption track refined to 3-tier classification (design-pattern / code candidate / GPU-blocked)
- **Updated** `roadmap.md`:
  - Future Economic WM section with sovereignty clarification
  - Future meta-regal-node section with governance pluralism, inter-domain Pareto
  - Anti-heuristic rule strengthened: "necessary but not sufficient" test added
  - New "Embodiment-Facing Subsystem Usefulness Rule" section
  - Habitat adoption track refined to 3-tier classification
- **Updated** `neuralization_bridge_doctrine.md`:
  - Level 5 rewritten from "Meta-Node Governance" to "Meta-Regal-Node Governance" with inter-domain composition, governance-pluralism, confidence-aware node weighting
- **Updated** `phase2_closure_standard.md`:
  - Anti-heuristic-without-neuralization section (from prior pass)
- **Updated** `doctrine_provider_dataset_resource_surfaces.md`:
  - Cross-WM resource surface scope (from prior pass)
- **Updated** `claude_to_comment_on.md`:
  - Embodiment-facing usefulness pressure added to robust-subsystem read
  - Economic WM and meta-regal-node doctrine summaries added
  - Clean current-state handoff artifact for next Codex tranche
- **Tests**: no code changes; 1422 passed, 0 failures (docs-only pass)

## 2026-04-03 — Architectural Review + Anti-Heuristic / Habitat Doctrine Updates

- **Reviewed** Codex's Tranche 2.1 (shadow compiler + first downstream consumers):
  - compiler is real: compiles canonical state from scene tracks, belief state, VLA semantic evidence
  - two downstream consumers wired: sim-synth semantic_inputs.py + rollout_labeler.py
  - semantic bridge family compiled and consumed for sim-synth and annotation bridges
  - assessment: genuine early `shadow_runtime`, not schema decoration
- **Identified** remaining Category A items:
  - receipts typed but not emitted by compiler (clearest remaining gap)
  - provider contracts disconnected from compiler (ad-hoc string inference, not contract registry)
  - evidence fusion always uses hardcoded weights regardless of promotion stage
  - embodiment bridge compiled but orphaned (no downstream consumer)
  - dimensional regime markers missing (heuristic d=8, not yet self-documenting)
- **Added** anti-heuristic-without-neuralization rule to:
  - `roadmap.md`: explicit rule that structural preparation is necessary but not sufficient; bounded neural seams should begin as real codepaths once substrate is honest enough
  - `phase2_closure_standard.md`: new section naming earliest neural seams (evidence fusion, annotation bridge, provider calibration heads)
- **Added** Habitat extraction posture to:
  - `roadmap.md`: per-WM Habitat absorption status; Sim/Synth/Physics identified as biggest remaining opportunity
  - `multi_wm_architecture_plan.md`: new "Habitat Extraction Posture" section with named adoption track items for Sim/Synth/Physics
  - `doctrine_provider_dataset_resource_surfaces.md`: cross-WM resource surface scope (not Perception-only)
- **Updated** `claude_to_comment_on.md` with corrected Tranche 2.2 framing:
  - Priority 1: receipt emission + promotion-gate wiring
  - Priority 2: provider contract → compiler connection
  - Priority 3: embodiment shadow consumer skeleton
  - Priority 4: dimensional regime + bridge input source markers
  - Priority 5: first bounded neural seam implementation (evidence fusion, annotation bridge, provider calibration)
  - Habitat reminder for Sim/Synth/Physics adoption track
- **Tests**: 48 passed, compile clean

## 2026-04-03 — Phase 2 Reconciliation: Semantic Bridges + Provider/Resource Surfaces

- **Reconciled** the locally created Phase 2 Perception / Grounding WM package into a coherent current-state tranche instead of leaving the branch in a half-landed state.
- **Integrated** `SemanticBridgeRegistry` into `PerceptionGroundingWorldState`, so the semantic successor stack is now part of the canonical top-level Perception WM state rather than an adjacent orphan module.
- **Added** Habitat-inspired but WM-native lower-WM surfaces in `src/world_model/perception_grounding/state.py`:
  - `ProviderSurfaceState`
  - `DatasetSurfaceState`
  - `TaskMeasurementSurface`
  - `DeploymentResourceSurface`
  - `ComputeEnvelopeState`
  - `InferenceCapacityState`
  - `BatteryState`
  - `ThermalState`
- **Added** typed receipts in `src/world_model/perception_grounding/receipts.py`:
  - `ProviderAvailabilityReceipt`
  - `InferenceHeadroomReceipt`
  - `DeploymentResourceReceipt`
- **Verified** the semantic successor posture:
  - `resolve_semantic_bridge_helper()` is now covered by tests
  - all four bridge families are now represented in registry serialization coverage
  - `src/vla/semantic_vla.py` scaffolding posture is now covered by tests, including successor metadata
  - focused compile + lint + perception-grounding regression now pass on the reconciled Phase 2 package
- **Added doctrine**:
  - `docs/economic_world_model/doctrine_provider_dataset_resource_surfaces.md`
  - refined `docs/economic_world_model/doctrine_semantic_bridge_successor.md`
  - refined Phase 2 wording in `multi_wm_architecture_plan.md` and `roadmap.md`
- **Current status**:
  - Phase 1 remains structurally closed enough and should not be reopened without new external runtime/assets or a direct contradiction
  - Phase 2 is now the active implementation center with a cleaner semantic successor posture
  - remaining Phase 2 blockers are still mostly compiler/runtime/adapters/downstream wiring, not schema/doctrine ambiguity

## 2026-04-02 — Phase 2 Kickoff: Perception / Grounding WM Tranche 2.0

- **Phase transition**: Phase 1 Sim/Synth/Physics WM declared structurally closed. Zero Category A items. Remaining blockers are external GPU/runtime/asset items recorded in `phase1_external_gpu_runtime_backlog.md`.
- **Created**: `docs/economic_world_model/phase2_closure_standard.md` — Category A/B/C closure framework for Phase 2
- **Created**: `docs/economic_world_model/phase1_external_gpu_runtime_backlog.md` — explicit pre-training prerequisites
- **Created**: `src/world_model/perception_grounding/` package with:
  - `state.py`: 6 canonical state types (ObjectTrackState, SceneEdge, SceneGraphState, TemporalGroundingState, EvidenceRoutingState, PerceptionGroundingWorldState)
  - `receipts.py`: 5 receipt types (ProviderInvocation, GroundingCalibration, EvidenceFusion, TemporalGrounding, PerceptionContribution)
  - `provider_contracts.py`: 6 provider contract types (base, SAM 3/3.1, DINOv2/SigLIP, V-JEPA 2, Depth, Registry)
  - `promotion.py`: 3 helper resolvers (graph_transformer, temporal_grounding, evidence_fusion) with shared demotion
- **Created**: `tests/test_perception_grounding_world_model.py` — 33 tests, all passing
- **Updated**: `claude_to_comment_on.md` with Phase 2 status, neuralization map, and next tranche recommendation
- **Neuralization specified**: Full subsystem map with neural structure, capacity bands, governing WM, promotion posture, and downstream consumers for all 7 perception subsystems
- **Maturity**: `schema_only` — state types exist and serialize; no compiler or runtime yet

## 2026-04-02

- Changed: finished a late-Phase-1 closure pass over the remaining local/runtime/install honesty seams:
  - `src/world_model/sim_synth_physics/runtime_launch.py` now treats `asset::...` host-preflight blockers as launch blockers instead of filtering them out
  - `src/world_model/sim_synth_physics/runtime_work_orders.py` now preserves:
    - runtime-layout install-ready / install-partial / install-blocked profiles
    - host-preflight ready / verified component sets
    - launch missing preconditions and notes
  - `src/world_model/sim_synth_physics/training_corpus.py` now preserves the same stronger local evidence in backend-selector and branch-planner rows
- Why this matters:
  - before this tranche, the branch could still preserve blocked truth in runtime bindings while letting the launch surface or trainer rows look cleaner than the real host state
  - now launch, work-order, and training surfaces agree about blocked local runtime/install/asset truth
  - this closes the last meaningful internal pseudo-readiness seam found in the late Phase-1 audit
- Host audit summary:
  - `scripts/scan_phase1_runtime_layouts.py` now reports both backend lanes as blocked on this host
  - no relevant Isaac/Unitree/Holosoma env vars are set
  - no external `isaaclab`, `unitree_sdk2py`, or `holosoma` Python modules are importable
  - no external Isaac/Unitree/Holosoma runtime roots were found in the common local clone directories the branch audits
- Verification: `python3 -m compileall src/world_model/sim_synth_physics/runtime_launch.py src/world_model/sim_synth_physics/runtime_work_orders.py src/world_model/sim_synth_physics/training_corpus.py tests/test_sim_synth_runtime_launch.py tests/test_sim_synth_runtime_work_orders.py tests/test_sim_synth_training_corpus.py -q`, `python3 -m ruff check src/world_model/sim_synth_physics/runtime_launch.py src/world_model/sim_synth_physics/runtime_work_orders.py src/world_model/sim_synth_physics/training_corpus.py tests/test_sim_synth_runtime_launch.py tests/test_sim_synth_runtime_work_orders.py tests/test_sim_synth_training_corpus.py`, `python3 -m pytest -q tests/test_sim_synth_runtime_launch.py tests/test_sim_synth_runtime_work_orders.py tests/test_sim_synth_training_corpus.py`, `python3 -m pytest -q tests/test_scan_phase1_runtime_layouts.py tests/test_sim_synth_physics_world_model.py tests/test_sim_synth_physics_scripts.py`, `python3 scripts/scan_phase1_runtime_layouts.py --output-path /tmp/phase1_runtime_scan_final.json`, and `git diff --check` passed (results: `17 passed`, `50 passed`).
- Status summary:
  - audited Phase-1 Category A count is now `0` across the closure surfaces touched in this pass
  - the remaining blocker set is now honestly external on this host:
    - real Isaac/Unitree installs/assets/checkpoints
    - real Holosoma runtime/motion/policy/retargeting assets
    - real GPU-backed GGDS / LDM / video materialization

- Changed: made `scripts/scan_phase1_runtime_layouts.py` a real repo-root Phase-1 host-reality probe instead of a scan that only worked cleanly under pytest import conditions:
  - the script now inserts repo root into `sys.path` before importing `src.*`
  - it now emits `scan_summary` for both Isaac/Unitree and Holosoma lanes, including:
    - usable / install-ready / install-partial / install-blocked profiles
    - selected policy / deploy / runtime-report refs and sources
    - selected verified / partial target ids
    - host-preflight blockers
- Why this matters:
  - Phase 1 is now close enough to the external-runtime frontier that the host-reality scan itself needs to be a trustworthy CLI surface, not just a test-import helper
  - this tranche converts another vague Category B statement into an explicit local report
  - on the current host, the scan now says both lanes are blocked with zero usable profiles rather than leaving that truth implicit across many receipts
- Verification: `python3 -m compileall scripts/scan_phase1_runtime_layouts.py tests/test_scan_phase1_runtime_layouts.py -q`, `python3 -m ruff check scripts/scan_phase1_runtime_layouts.py tests/test_scan_phase1_runtime_layouts.py`, `python3 -m pytest -q tests/test_scan_phase1_runtime_layouts.py`, `python3 scripts/scan_phase1_runtime_layouts.py --output-path /tmp/phase1_runtime_scan_20260402.json`, and `git diff --check` passed (result: `1 passed`).
- Status summary:
  - the Phase-1 host scan is now closed as an internal tooling honesty gap
  - Category B is now easier to read directly from a local host report instead of inferring it from dispersed runtime-pack/binding artifacts

- Changed: made selected-ref validation operational in Phase-1 downstream consumers instead of leaving it as receipt-only metadata:
  - `src/world_model/sim_synth_physics/runtime_work_orders.py` now refuses to mark `satisfied_by_external_runtime_outcomes` when `selected_ref_validation` reports mismatched or missing selected refs
  - mismatched/missing selected-runtime components now become explicit runtime preconditions on the work order path
  - `src/world_model/sim_synth_physics/training_corpus.py` now stops preferring `external_runtime_outcome_receipt` as the backend-selector target source when the harvested outputs fail selected-ref validation
- Why this matters:
  - before this tranche, the branch could correctly record a selected-ref mismatch and still operationally treat the outcome as satisfactory
  - now the mismatch truth actually changes completion posture and trainer-source selection
  - this removes another pseudo-readiness seam without adding a new ladder rung
- Verification: `python3 -m compileall src/world_model/sim_synth_physics/runtime_work_orders.py src/world_model/sim_synth_physics/training_corpus.py tests/test_sim_synth_runtime_work_orders.py tests/test_sim_synth_training_corpus.py -q`, `python3 -m ruff check src/world_model/sim_synth_physics/runtime_work_orders.py src/world_model/sim_synth_physics/training_corpus.py tests/test_sim_synth_runtime_work_orders.py tests/test_sim_synth_training_corpus.py`, `python3 -m pytest -q tests/test_sim_synth_runtime_work_orders.py tests/test_sim_synth_training_corpus.py`, and `git diff --check` passed (result: `8 passed`).
- Status summary:
  - the audited selected-ref validation consumer path has no new Category A gap
  - Category B is again narrowed toward actual external-runtime/install/GPU blockers rather than internal misuse of harvested outputs

- Changed: explicitly closed the lingering Tier 3.4 / 3.5 verification ambiguity on the audited Phase-1 path:
  - added `tests/test_sim_synth_phase1_verification.py`
  - Tier 3.4 coverage now directly checks:
    - `build_simulation_job_inferential_contract()`
    - `benchmark_provenance_quality()`
    - `agenda_score_with_inferential_prior()`
    - `build_branch_plan_inferential_contract()`
  - Tier 3.5 coverage now directly checks:
    - `compile_physics_adaptation_policy()`
    - humanoid randomization axes and calibration targets
    - `build_physics_adaptation_receipt()`
    - `build_physics_calibration_receipt()`
    - reaction to route status and runtime evidence
- Why this matters:
  - those items were no longer substantively mysterious, but they were still being carried as unresolved Category C because they had not been directly re-audited
  - the explicit audit now shows the current path is structurally sound there
  - the honest remainder is even more clearly external runtime/install/assets/GPU reality rather than unclassified Phase-1-local behavior
- Verification: `python3 -m compileall tests/test_sim_synth_phase1_verification.py -q`, `python3 -m ruff check tests/test_sim_synth_phase1_verification.py`, and `python3 -m pytest -q tests/test_sim_synth_phase1_verification.py` passed (result: `4 passed`).
- Status summary:
  - Tier 3.4 is closed on the audited path
  - Tier 3.5 is closed on the audited path
  - Category C unresolved count is now `0` on the current closure sheet

- Changed: tightened Phase-1 runtime-outcome honesty so harvested outputs are now checked against the selected runtime refs instead of only being counted/classified:
  - `src/world_model/sim_synth_physics/runtime_bundles.py` now passes runtime-binding truth into the output-contract build path
  - `src/world_model/sim_synth_physics/runtime_outcomes.py` now:
    - carries expected selected policy / deploy-config / runtime-report refs in the output contract
    - includes exact selected refs in harvest sources when those local artifacts exist
    - emits `selected_ref_validation` in the output summary / outcome receipt
  - `src/world_model/sim_synth_physics/runtime_work_orders.py` and `training_corpus.py` now preserve selected-ref validation status in execution-facing and trainer-facing artifacts
- Why this matters:
  - before this tranche, a runtime lane could harvest outputs successfully without saying whether those outputs matched the chosen runtime policy/report surfaces
  - now “runtime outputs harvested” and “selected runtime refs matched” are distinct but adjacent truths
  - this removes another pseudo-readiness seam and pushes the honest remainder further toward real external runtime/install/GPU reality
- Verification: `python3 -m compileall src/world_model/sim_synth_physics/runtime_bundles.py src/world_model/sim_synth_physics/runtime_outcomes.py src/world_model/sim_synth_physics/runtime_work_orders.py src/world_model/sim_synth_physics/training_corpus.py tests/test_sim_synth_runtime_outcomes.py tests/test_sim_synth_runtime_work_orders.py tests/test_sim_synth_training_corpus.py tests/test_sim_synth_runtime_bundles.py -q`, `python3 -m ruff check src/world_model/sim_synth_physics/runtime_bundles.py src/world_model/sim_synth_physics/runtime_outcomes.py src/world_model/sim_synth_physics/runtime_work_orders.py src/world_model/sim_synth_physics/training_corpus.py tests/test_sim_synth_runtime_outcomes.py tests/test_sim_synth_runtime_work_orders.py tests/test_sim_synth_training_corpus.py tests/test_sim_synth_runtime_bundles.py`, `python3 -m pytest -q tests/test_sim_synth_runtime_outcomes.py tests/test_sim_synth_runtime_work_orders.py tests/test_sim_synth_training_corpus.py tests/test_sim_synth_runtime_bundles.py`, and `git diff --check` passed (result: `12 passed`).
- Status summary:
  - the audited selected-output validation cluster has no new Category A gap
  - Category B is now more explicitly about whether real runtime artifacts exist at all, not whether harvested outputs can be matched back to the selected runtime surfaces once they do

- Changed: tightened Phase-1 checkpoint / deploy-config / runtime-report selection so verified local artifacts now outrank merely earlier candidates in runtime-pack and binding selection:
  - `src/world_model/sim_synth_physics/ref_evidence.py` now exposes reusable candidate-evidence selection/summarization helpers
  - `src/world_model/sim_synth_physics/adapters/isaac_unitree_runtime_pack.py` and `src/world_model/sim_synth_physics/adapters/holosoma_runtime_pack.py` now:
    - choose `primary_policy_ref`, `primary_deploy_config_ref`, and `primary_runtime_report_ref` from the best verified local candidate when available
    - preserve `*_ref_source`
    - preserve candidate-evidence summaries for policy / deploy / runtime-report surfaces
  - `src/world_model/sim_synth_physics/adapters/isaac_unitree_runtime_binding.py` and `src/world_model/sim_synth_physics/adapters/holosoma_runtime_binding.py` now preserve the selected ref source on the binding path instead of silently inheriting first-candidate ordering
  - `src/world_model/sim_synth_physics/runtime_work_orders.py` and `training_corpus.py` now carry that upstream/selected ref evidence into execution-facing and trainer-facing artifacts
- Why this matters:
  - the branch previously had stronger install/profile truth, but the concrete checkpoint/report/deploy ref could still quietly depend on candidate ordering
  - verified local runtime artifacts now win over earlier missing candidates without inventing a new ladder rung
  - this removes another repo-local ambiguity and pushes the honest remainder further toward actual external install/runtime/GPU reality
- Verification: `python3 -m compileall src/world_model/sim_synth_physics/ref_evidence.py src/world_model/sim_synth_physics/adapters/isaac_unitree_runtime_pack.py src/world_model/sim_synth_physics/adapters/holosoma_runtime_pack.py src/world_model/sim_synth_physics/adapters/isaac_unitree_runtime_binding.py src/world_model/sim_synth_physics/adapters/holosoma_runtime_binding.py src/world_model/sim_synth_physics/runtime_work_orders.py src/world_model/sim_synth_physics/training_corpus.py tests/test_isaac_unitree_runtime_pack.py tests/test_holosoma_runtime_pack.py tests/test_isaac_unitree_runtime_binding.py tests/test_holosoma_runtime_binding.py -q`, `python3 -m ruff check src/world_model/sim_synth_physics/ref_evidence.py src/world_model/sim_synth_physics/adapters/isaac_unitree_runtime_pack.py src/world_model/sim_synth_physics/adapters/holosoma_runtime_pack.py src/world_model/sim_synth_physics/adapters/isaac_unitree_runtime_binding.py src/world_model/sim_synth_physics/adapters/holosoma_runtime_binding.py src/world_model/sim_synth_physics/runtime_work_orders.py src/world_model/sim_synth_physics/training_corpus.py tests/test_isaac_unitree_runtime_pack.py tests/test_holosoma_runtime_pack.py tests/test_isaac_unitree_runtime_binding.py tests/test_holosoma_runtime_binding.py`, `python3 -m pytest -q tests/test_isaac_unitree_runtime_pack.py tests/test_holosoma_runtime_pack.py tests/test_isaac_unitree_runtime_binding.py tests/test_holosoma_runtime_binding.py tests/test_sim_synth_runtime_work_orders.py tests/test_sim_synth_training_corpus.py`, and `git diff --check` passed (result: `20 passed`).
- Status summary:
  - the audited ref-selection cluster has no new Category A gap
  - Category B is now even more clearly about whether the local runtime/install/checkpoint reality actually exists, not whether the WM picks the strongest local artifact once it does

- Changed: promoted `usable_profiles` into the Phase-1 runtime-layout contract and threaded that stronger profile truth through downstream artifacts:
  - `src/world_model/sim_synth_physics/runtime_layouts.py` now emits:
    - `usable_profiles`
    - `install_ready_profiles`
    - `install_partial_profiles`
    - `install_blocked_profiles`
  - `src/world_model/sim_synth_physics/runtime_bundles.py` now prefers `usable_profiles` for profile selection/ordering while still preserving the broader `ready_profiles` surface
  - `src/world_model/sim_synth_physics/runtime_bridge.py`, `runtime_work_orders.py`, `compiler.py`, and `training_corpus.py` now preserve `runtime_layout_usable_profiles` so downstream execution/training surfaces do not need to reconstruct “usable” from weaker root-exists semantics
- Why this matters:
  - the branch previously had the stronger profile truth, but only implicitly in deployment/runtime-pack logic
  - now the layout contract itself exposes that truth, and the rest of the Phase-1 runtime path can consume it honestly
  - this removes another pseudo-readiness seam without adding a new runtime rung
- Verification: `python3 -m compileall src/world_model/sim_synth_physics/runtime_layouts.py src/world_model/sim_synth_physics/runtime_bundles.py src/world_model/sim_synth_physics/runtime_bridge.py src/world_model/sim_synth_physics/runtime_work_orders.py src/world_model/sim_synth_physics/compiler.py src/world_model/sim_synth_physics/training_corpus.py tests/test_sim_synth_runtime_layouts.py tests/test_sim_synth_runtime_bundles.py tests/test_sim_synth_runtime_work_orders.py tests/test_sim_synth_runtime_launch.py tests/test_scan_phase1_runtime_layouts.py tests/test_sim_synth_training_corpus.py tests/test_sim_synth_physics_world_model.py -q`, `python3 -m ruff check src/world_model/sim_synth_physics/runtime_layouts.py src/world_model/sim_synth_physics/runtime_bundles.py src/world_model/sim_synth_physics/runtime_bridge.py src/world_model/sim_synth_physics/runtime_work_orders.py src/world_model/sim_synth_physics/compiler.py src/world_model/sim_synth_physics/training_corpus.py tests/test_sim_synth_runtime_layouts.py tests/test_sim_synth_runtime_bundles.py tests/test_sim_synth_runtime_work_orders.py tests/test_sim_synth_runtime_launch.py tests/test_scan_phase1_runtime_layouts.py tests/test_sim_synth_training_corpus.py tests/test_sim_synth_physics_world_model.py`, `python3 -m pytest -q tests/test_sim_synth_runtime_layouts.py tests/test_sim_synth_runtime_bundles.py tests/test_sim_synth_runtime_work_orders.py tests/test_sim_synth_runtime_launch.py tests/test_scan_phase1_runtime_layouts.py tests/test_sim_synth_training_corpus.py tests/test_sim_synth_physics_world_model.py`, and `git diff --check` passed (result: `56 passed`).
- Status summary:
  - the audited usable-profile propagation cluster has no new Category A gap
  - Category B is now even more clearly about real local installs/assets/checkpoints/GPU/provider reality rather than internal profile-truth reconstruction

- Changed: tightened Phase-1 profile/target/policy selection so deployment/runtime-pack readiness is now driven by usable profiles, verified targets, and real local checkpoint-bearing roots instead of raw existing roots:
  - `src/world_model/sim_synth_physics/runtime_layouts.py` now selects policy roots across multiple candidates more honestly, so an explicit-but-empty policy root no longer outranks a discovered runtime root that actually contains checkpoints
  - `src/world_model/sim_synth_physics/adapters/isaac_unitree_deployment.py` and `src/world_model/sim_synth_physics/adapters/holosoma_deployment.py` now use usable profiles plus verified targets rather than `ready_profiles`/`ready_target_ids` path-existence posture
  - `src/world_model/sim_synth_physics/adapters/isaac_unitree_runtime_pack.py` and `src/world_model/sim_synth_physics/adapters/holosoma_runtime_pack.py` now preserve `runtime_target_preflight_status`, verified target ids, and usable-profile preference instead of treating raw target existence as enough runtime-pack evidence
- Why this matters:
  - install-blocked runtime profiles no longer count as deployable just because the repo root exists
  - empty explicit policy roots no longer hide discovered local checkpoint banks
  - the remaining blocker is pushed further toward real local runtime/install/assets/checkpoints/GPU reality rather than internal profile-selection optimism
- Verification: `python3 -m compileall src/world_model/sim_synth_physics/runtime_layouts.py src/world_model/sim_synth_physics/adapters/isaac_unitree_deployment.py src/world_model/sim_synth_physics/adapters/holosoma_deployment.py src/world_model/sim_synth_physics/adapters/isaac_unitree_runtime_pack.py src/world_model/sim_synth_physics/adapters/holosoma_runtime_pack.py tests/test_sim_synth_runtime_layouts.py tests/test_isaac_unitree_deployment.py tests/test_holosoma_deployment.py tests/test_isaac_unitree_runtime_pack.py tests/test_holosoma_runtime_pack.py tests/test_sim_synth_runtime_bundles.py tests/test_isaac_unitree_runtime_binding.py tests/test_scan_phase1_runtime_layouts.py -q`, `python3 -m ruff check src/world_model/sim_synth_physics/runtime_layouts.py src/world_model/sim_synth_physics/adapters/isaac_unitree_deployment.py src/world_model/sim_synth_physics/adapters/holosoma_deployment.py src/world_model/sim_synth_physics/adapters/isaac_unitree_runtime_pack.py src/world_model/sim_synth_physics/adapters/holosoma_runtime_pack.py tests/test_sim_synth_runtime_layouts.py tests/test_isaac_unitree_deployment.py tests/test_holosoma_deployment.py tests/test_isaac_unitree_runtime_pack.py tests/test_holosoma_runtime_pack.py tests/test_sim_synth_runtime_bundles.py tests/test_isaac_unitree_runtime_binding.py tests/test_scan_phase1_runtime_layouts.py`, `python3 -m pytest -q tests/test_sim_synth_runtime_layouts.py tests/test_isaac_unitree_deployment.py tests/test_holosoma_deployment.py tests/test_isaac_unitree_runtime_pack.py tests/test_holosoma_runtime_pack.py tests/test_sim_synth_runtime_bundles.py tests/test_sim_synth_runtime_launch.py tests/test_isaac_unitree_runtime_binding.py tests/test_holosoma_runtime_binding.py tests/test_scan_phase1_runtime_layouts.py`, and `git diff --check` passed (result: `39 passed`).
- Status summary:
  - the audited profile/target/policy selection cluster has no new Category A gap
  - Category B is now narrower and more concrete: real local installs/assets/checkpoints and GPU-backed provider/runtime reality

- Changed: tightened Phase-1 target-preflight truth so runtime-target existence is no longer treated as enough on the selected-target binding path:
  - `src/world_model/sim_synth_physics/runtime_targets.py` now emits install-shape verification metadata for runtime targets:
    - `verification_status`
    - `verified`
    - `matched_markers`
    - `missing_markers`
    - `primary_marker_ref`
  - `src/world_model/sim_synth_physics/adapters/isaac_unitree_runtime_binding.py` and `src/world_model/sim_synth_physics/adapters/holosoma_runtime_binding.py` now consume that selected-target evidence directly and emit:
    - `selected_verified_target_ids`
    - `selected_partial_target_ids`
    - selected-target evidence that can block host preflight even when a target root exists
  - `src/world_model/sim_synth_physics/runtime_work_orders.py` and `src/world_model/sim_synth_physics/training_corpus.py` now preserve that selected-target truth instead of flattening it back into pack-level readiness
- Why this matters:
  - empty SDK, asset, motion, or retargeting roots no longer look launch-ready just because the path exists
  - the branch can now distinguish:
    - selected target exists and is install-shaped
    - selected target exists but is only partial
    - selected target is still missing
  - this removes another fake-readiness seam without changing the broader runtime ladder or forcing churn through `ready_target_ids`
- Verification: `python3 -m compileall src/world_model/sim_synth_physics/runtime_targets.py src/world_model/sim_synth_physics/ref_evidence.py src/world_model/sim_synth_physics/adapters/isaac_unitree_runtime_binding.py src/world_model/sim_synth_physics/adapters/holosoma_runtime_binding.py src/world_model/sim_synth_physics/runtime_work_orders.py src/world_model/sim_synth_physics/training_corpus.py tests/test_sim_synth_runtime_targets.py tests/test_isaac_unitree_runtime_binding.py tests/test_holosoma_runtime_binding.py tests/test_sim_synth_training_corpus.py tests/test_sim_synth_physics_world_model.py -q`, `python3 -m ruff check src/world_model/sim_synth_physics/runtime_targets.py src/world_model/sim_synth_physics/ref_evidence.py src/world_model/sim_synth_physics/adapters/isaac_unitree_runtime_binding.py src/world_model/sim_synth_physics/adapters/holosoma_runtime_binding.py src/world_model/sim_synth_physics/runtime_work_orders.py src/world_model/sim_synth_physics/training_corpus.py tests/test_sim_synth_runtime_targets.py tests/test_isaac_unitree_runtime_binding.py tests/test_holosoma_runtime_binding.py tests/test_sim_synth_training_corpus.py tests/test_sim_synth_physics_world_model.py`, `python3 -m pytest -q tests/test_sim_synth_runtime_targets.py tests/test_isaac_unitree_runtime_binding.py tests/test_holosoma_runtime_binding.py tests/test_sim_synth_training_corpus.py tests/test_sim_synth_physics_world_model.py`, `python3 -m pytest -q tests/test_sim_synth_runtime_layouts.py tests/test_isaac_unitree_runtime_pack.py tests/test_holosoma_runtime_pack.py tests/test_sim_synth_runtime_launch.py tests/test_sim_synth_runtime_work_orders.py tests/test_scan_phase1_runtime_layouts.py`, and `git diff --check` passed (results: `46 passed`, `22 passed`).
- Status summary:
  - the audited target-preflight cluster has no new Category A gap
  - the remaining blocker is even more clearly real local runtime/install/assets/checkpoints/GPU reality rather than missing internal verification surfaces

- Changed: closed the Tier 3.6 shadow-execution honesty gap and tightened Tier 3.3 branch-planner fallback truth on the active Phase-1 verification path:
  - `src/world_model/sim_synth_physics/shadow_execution.py` now consumes selected runtime-binding surfaces when deriving Isaac shadow env-configs and Holosoma shadow work orders
  - shadow receipts now explicitly record `shadow_runtime_binding_consumed` and preserve selected profile / launch root / policy ref / motion-source truth inside the materialized artifacts themselves rather than only sibling receipt metadata
  - Holosoma shadow preconditions now include selected binding host-preflight and selected-profile install gaps, deduped against existing missing-asset signals
  - `src/world_model/sim_synth_physics/synthetic_branches.py` now records whether the branch helper actually controlled the plan or only contributed a trace:
    - `branch_helper_resolution`
    - `branch_helper_resolution_reason`
    - `branch_helper_payload_applied`
  - `src/world_model/sim_synth_physics/training_corpus.py` now preserves that control truth in branch-planner trainer rows instead of forcing downstream consumers to infer fallback from mixed generation-mode and trace fields
- Why this matters:
  - the shadow lane no longer claims deeper runtime-ladder honesty while still deriving most of its execution inputs from generic context alone
  - a learned branch-planner trace no longer looks like active control when the heuristic path actually retained authority because the helper was shadow-candidate, demoted, or unavailable
  - this is another Phase-1-local reduction in fake readiness without adding a new ladder rung
- Verification: `python3 -m compileall src/world_model/sim_synth_physics/shadow_execution.py src/world_model/sim_synth_physics/synthetic_branches.py src/world_model/sim_synth_physics/training_corpus.py tests/test_sim_synth_physics_world_model.py tests/test_sim_synth_training_corpus.py -q`, `python3 -m ruff check src/world_model/sim_synth_physics/shadow_execution.py src/world_model/sim_synth_physics/synthetic_branches.py src/world_model/sim_synth_physics/training_corpus.py tests/test_sim_synth_physics_world_model.py tests/test_sim_synth_training_corpus.py`, `python3 -m pytest -q tests/test_sim_synth_physics_world_model.py tests/test_sim_synth_training_corpus.py`, and `git diff --check` passed (result: `33 passed`).
- Status summary:
  - Tier 3.6 is now closed on the audited path
  - Tier 3.3 fallback honesty is materially closed on the audited path
  - remaining explicit verification items still needing deliberate classification are Tier 3.4 and Tier 3.5

- Changed: closed Tier 3.2 (promotion/demotion machinery) Category A gap — Claude-authored implementation:
  - `src/world_model/sim_synth_physics/promotion.py`: added `_check_demotion()` and `evidence_signals` parameter to `resolve_helper()`
  - `src/world_model/sim_synth_physics/backend_selector_runtime.py`: threaded evidence-based demotion through both direct-loaded and package-loaded paths
  - `src/world_model/sim_synth_physics/branch_planner_runtime.py`: same demotion threading
  - Added 7 new tests covering demotion triggers (evidence_failure, benchmark_gate_revoked, failure_rate), no-demotion on healthy evidence, and all three resolver types
  - Fixed stale test expectation in `test_holosoma_binding_records_runtime_target_contract`: updated to accept `pack_partial` (honest result from install-hardened code)
- Why this matters:
  - previously, a promoted helper stayed promoted forever regardless of subsequent evidence — this was a structural completeness gap
  - demoted helpers get weight 0.25 (shadow_candidate level), so compiler/calibration/branch consumers correctly fall back to heuristic behavior
  - three demotion triggers: `benchmark_gate_revoked`, `evidence_failure`, `recent_failure_rate > threshold`
  - `demoted_to_shadow` is a fourth internal promotion stage, not a new mode
- Verification: `python3 -m compileall`, `python3 -m pytest tests/test_sim_synth_physics_world_model.py` and full Phase 1 suite: 61 passed, 0 failed
- Status summary:
  - Tier 3.2 is now structurally closed
  - remaining unverified Tier 3 items: 3.1, 3.3, 3.4, 3.5, 3.6
  - highest-risk next: 3.6 (shadow execution ladder threading), 3.1 (render provider receipts), 3.3 (branch planner fallback receipts)

- Changed: pushed the active Phase-1 Category B edge further toward real local host/runtime evidence:
  - added `src/world_model/sim_synth_physics/local_runtime_discovery.py`
  - `runtime_targets.py` now supports targeted autodiscovery of common local upstream repo roots for Isaac/Unitree and Holosoma lanes when embodiment/env roots are not explicitly wired
  - `runtime_layouts.py` now uses the same targeted autodiscovery and allows policy-contract fallback to discovered runtime roots when those roots contain real checkpoints/configs/reports
- Why this matters:
  - hosts with real local clones/checkpoints can now be consumed more honestly without requiring every relevant env var to be pre-wired first
  - the branch stays real-or-unavailable because missing roots still remain explicit; autodiscovery only closes the “real clone exists locally but the WM cannot see it yet” gap
- Verification: `python3 -m compileall src/world_model/sim_synth_physics tests/test_sim_synth_runtime_targets.py tests/test_sim_synth_runtime_layouts.py tests/test_isaac_unitree_runtime_pack.py tests/test_holosoma_runtime_pack.py tests/test_scan_phase1_runtime_layouts.py -q`, `python3 -m ruff check src/world_model/sim_synth_physics tests/test_sim_synth_runtime_targets.py tests/test_sim_synth_runtime_layouts.py tests/test_isaac_unitree_runtime_pack.py tests/test_holosoma_runtime_pack.py tests/test_scan_phase1_runtime_layouts.py`, `python3 -m pytest -q tests/test_sim_synth_runtime_targets.py tests/test_sim_synth_runtime_layouts.py tests/test_isaac_unitree_runtime_pack.py tests/test_holosoma_runtime_pack.py tests/test_scan_phase1_runtime_layouts.py tests/test_isaac_unitree_runtime_binding.py tests/test_holosoma_runtime_binding.py tests/test_sim_synth_runtime_launch.py`, and `git diff --check` passed (result: `30 passed`).
- Status summary:
  - the branch is better at consuming actual local clone/install/checkpoint reality
  - the dominant remainder is now even more clearly the presence or absence of real installs/assets/checkpoints/GPU, not a missing host-discovery seam

- Changed: consumed the richer Phase-1 upstream runtime evidence against more concrete local host/runtime reality without adding a new ladder rung:
  - added `src/world_model/sim_synth_physics/ref_evidence.py`
  - Isaac and Holosoma runtime bindings now emit selected-surface evidence plus `host_preflight_status`
  - Isaac host preflight now distinguishes declared-only asset refs from locally verified asset refs at the binding level rather than only inside the upstream runtime pack
  - Holosoma bindings now prefer locally existing motion sources when selecting motion-train/runtime surfaces instead of carrying missing motion refs forward as if they were equally selected
  - `runtime_launch.py` now consumes non-asset host-preflight gaps, while work orders and training rows preserve the fuller host-preflight truth
- Why this matters:
  - the branch can now distinguish:
    - contract-ready
    - locally verified enough to launch
    - still blocked by local host/runtime/install reality
  - this removes another pseudo-readiness seam without inventing a new runtime rung
- Verification: `python3 -m compileall src/world_model/sim_synth_physics tests/test_isaac_unitree_runtime_binding.py tests/test_holosoma_runtime_binding.py tests/test_sim_synth_runtime_launch.py tests/test_scan_phase1_runtime_layouts.py -q`, `python3 -m ruff check src/world_model/sim_synth_physics tests/test_isaac_unitree_runtime_binding.py tests/test_holosoma_runtime_binding.py tests/test_sim_synth_runtime_launch.py tests/test_scan_phase1_runtime_layouts.py`, `python3 -m pytest -q tests/test_isaac_unitree_runtime_binding.py tests/test_holosoma_runtime_binding.py tests/test_sim_synth_runtime_launch.py tests/test_scan_phase1_runtime_layouts.py tests/test_sim_synth_runtime_work_orders.py tests/test_sim_synth_training_corpus.py tests/test_sim_synth_physics_world_model.py`, and `git diff --check` passed (result: `46 passed`).
- Status summary:
  - another internal honesty gap is closed on the active Phase-1 runtime-binding cluster
  - the remainder is increasingly actual host/runtime/assets/checkpoints/GPU reality rather than missing local evidence classification

- Changed: normalized `docs/economic_world_model/claude_to_comment_on.md` back into a single current-state handoff artifact instead of an accreted stack of tranche notes.
- Why this matters:
  - the collaboration artifact is easier to read as current branch truth
  - historical tranche detail now stays where it belongs:
    - `docs/economic_world_model/progress_log.md`
    - `docs/economic_world_model/implementation_notes.md`
  - this makes it harder to confuse audited-cluster closure with total Phase-1 closure
- Verification: `git diff --check` passed. This was a docs-only cleanup pass.

- Changed: closed the remaining compiler-side Category A cluster from the active Phase-1 Tier 1 / Tier 3 verification pass:
  - `src/world_model/sim_synth_physics/state.py` now carries `physics_execution_contract` inside `SimSynthPhysicsWorldState`
  - `src/world_model/sim_synth_physics/compiler.py` now compiles that contract with the configured fallback backend and emits a compiler-owned receipt inventory / runtime-depth projection
  - `src/world_model/sim_synth_physics/runtime.py` now reuses the compiled execution contract instead of rebuilding it on the happy path
  - `src/world_model/sim_synth_physics/training_corpus.py` now harvests the compiled execution contract and compiler-owned receipt inventory into backend-selector and branch-planner rows
- Why this matters:
  - backend routing is now canonical compiled state, not only runtime reconstruction
  - the compiler now exposes what it already knows about the deeper runtime ladder instead of leaving that truth implicit until runtime artifacts appear
  - trainer/export rows no longer flatten away the new compiler-side closure
- Verification: `python3 -m compileall src/world_model/sim_synth_physics tests/test_sim_synth_physics_world_model.py tests/test_sim_synth_training_corpus.py -q`, `python3 -m ruff check src/world_model/sim_synth_physics tests/test_sim_synth_physics_world_model.py tests/test_sim_synth_training_corpus.py`, `python3 -m pytest -q tests/test_sim_synth_physics_world_model.py tests/test_sim_synth_training_corpus.py`, and `git diff --check` passed (result: `26 passed`).
- Status summary:
  - audited compiler-side Category A count for the active closure spec is now `0`
  - the dominant remainder is now increasingly honest Category B:
    - real Isaac / Unitree runtime, assets, checkpoints
    - real Holosoma host/runtime, motion/retargeting assets, policies
    - real GPU-backed GGDS / LDM materialization
- Next recommended task: keep Phase 1 as the implementation center and use the now-compiled closure surfaces to harden concrete Isaac/Unitree and Holosoma evidence lanes rather than pivoting upward prematurely.

- Changed: hardened `scripts/economic_world_model/nightly_audit.py` task selection so verification failures are prioritized over scaffold discovery:
  - added `_verification_repair_task(...)` and made `_next_task(...)` consume verification results before evaluating additive candidates
  - when `agent_verify` fails, the audit now recommends `agent_verify_regression` (targeting `CLAUDE.md`, `scripts/agent/verify.sh`, and `AGENTS.md`) instead of incorrectly reporting “No missing additive step detected”
  - for non-`agent_verify` failures, the audit now emits a generic `verification_regression` remediation task
- Changed: expanded `tests/test_economic_world_model_nightly_audit.py` with verification-priority coverage:
  - added tests proving `agent_verify` failure outranks scaffold tasks
  - added tests proving generic verification failure still outranks scaffold tasks
  - updated existing `_next_task(...)` tests for the new verification-aware signature
- Verification: `python3 -m pytest -q tests/test_economic_world_model_nightly_audit.py`, `python3 scripts/economic_world_model/nightly_audit.py --output-json artifacts/economic_world_model/nightly_audit_summary.json --output-markdown artifacts/economic_world_model/nightly_audit_summary.md`, `python3 -m compileall src/`, and `python3 -m pytest tests/ -v` (result: 1329 passed, 3 skipped).
- Status summary: nightly audit now correctly reports verification remediation as the top next step under failing gates, improving autonomous task selection honesty.
- Next recommended task: resolve the `agent_verify` failure (`CLAUDE.md` shim mismatch) before attempting additional roadmap wiring.

- Changed: ran the Phase 1 Sim / Synth / Physics Tier 1 / Tier 3 verification pass and closed the biggest remaining internal receipt-chain gaps without adding new runtime-ladder rungs:
  - added `gen2sim_admission_receipt_v1` and threaded it through the runtime result, artifact emission, and training-corpus harvest path
  - updated `shadow_execution.py` so backend shadow receipts now carry the deeper runtime-ladder truth already compiled in Tier 2:
    - runtime execution
    - adapter mediation
    - adapter realization
    - launch status
    - harvested outcome status
    - `shadow_harvest_mode`
  - updated `training_corpus.py` so branch-planner rows now preserve:
    - gen2sim receipt ids/counts
    - adaptation receipt ids
    - calibration receipt ids/scores
    - shadow execution ids/status
- Why this matters:
  - Phase 1 is more honest now: `gen2sim` is no longer state-only, shadow execution no longer bypasses the deeper runtime lane, and branch-planner export is no longer flatter than backend-selector export
  - the remaining open Phase 1 gaps are narrower and more clearly split between internal compiler work and honestly external runtime/assets/GPU blockers
- Verification: `python3 -m compileall src/world_model/sim_synth_physics tests/test_sim_synth_branch_helpers.py tests/test_sim_synth_training_corpus.py tests/test_sim_synth_physics_world_model.py -q`, `python3 -m ruff check src/world_model/sim_synth_physics tests/test_sim_synth_branch_helpers.py tests/test_sim_synth_training_corpus.py tests/test_sim_synth_physics_world_model.py`, `python3 -m pytest -q tests/test_sim_synth_branch_helpers.py tests/test_sim_synth_training_corpus.py tests/test_sim_synth_physics_world_model.py`, `python3 -m pytest -q tests/test_sim_synth_physics_scripts.py tests/test_sim_synth_runtime_launch.py`, and `git diff --check` passed (results: `30 passed`, `10 passed`).
- Status summary: Tier 1 / Tier 3 verification reduced Phase 1 Category A findings to a narrow compiler-side cluster:
  - `PhysicsExecutionContract` is still not a canonical compiled-state artifact
  - deeper runtime-binding truth is still clearer in runtime artifacts than in the compiled world state
- Next recommended task: keep Phase 1 as the implementation center and close that remaining compiler-side cluster before claiming structural closure; Perception prep is now allowed only as a secondary parallel activity.

## 2026-03-27

- Changed: pushed the Phase-1 backend lane past “launch spec only” and into canonical external-launch evidence:
  - `src/world_model/sim_synth_physics/runtime_launch.py` now builds `backend_runtime_launch_receipt_v1` artifacts over prepared or executed upstream runtime launches
  - `src/world_model/sim_synth_physics/backend_runtime_execution.py` now records those launch receipts whenever the Isaac/Unitree or Holosoma lane stops at an external launch path, and can optionally execute the launch command through the WM runtime instead of leaving that step entirely outside the loop
  - `src/world_model/sim_synth_physics/runtime.py` now surfaces the launch receipt in the loop result, loop summary, training feedback manifest, runtime evidence, and artifact set
  - `scripts/run_phase1_runtime_launch.py` now emits the launch receipt explicitly, including an optional pure receipt artifact
  - `src/world_model/sim_synth_physics/training_corpus.py` now harvests the launch receipt so backend-selector and branch-planner corpora can distinguish:
    - planning-only
    - external launch attempted
    - shadow runtime
    - concrete runtime
- Why this matters:
  - the Phase-1 backend lane can now remember that an upstream Isaac/Unitree or Holosoma runtime was actually launched, not just that a command string existed
  - that is another step toward the honest remainder we want: external roots/assets/policies/GPU become the blocker, not missing receipt wiring between WM planning and upstream runtime execution
- Verification: `python3 -m compileall src/world_model/sim_synth_physics scripts/run_phase1_runtime_launch.py scripts/run_sim_synth_physics_loop.py tests/test_sim_synth_runtime_launch.py tests/test_sim_synth_physics_world_model.py tests/test_sim_synth_physics_scripts.py tests/test_sim_synth_training_corpus.py -q`, `python3 -m ruff check src/world_model/sim_synth_physics scripts/run_phase1_runtime_launch.py scripts/run_sim_synth_physics_loop.py tests/test_sim_synth_runtime_launch.py tests/test_sim_synth_physics_world_model.py tests/test_sim_synth_physics_scripts.py tests/test_sim_synth_training_corpus.py`, `python3 -m pytest -q tests/test_sim_synth_runtime_launch.py tests/test_sim_synth_physics_world_model.py tests/test_sim_synth_physics_scripts.py tests/test_sim_synth_training_corpus.py`, and `git diff --check` passed.

- Changed: refined the multi-WM roadmap so inferential compute capacity and concrete battery state are now treated as early lower-WM resource contracts rather than late economic-only abstractions:
  - `docs/economic_world_model/multi_wm_architecture_plan.md` now says compute / battery should enter first as canonical embodiment/deployment state, then become allocatable economic-WM budget objects, then only later become transport/meta-node governance inputs
  - Phase 3 now explicitly calls for `ComputeEnvelopeState` / `BatteryState`-style canonical state, resource forecasting, placement/QoS receipts, and resource-aware learned seams
  - Phase 3.5 now explicitly requires onboard/companion compute and battery-budget assumptions in the G1/R1 capacity audit
  - Phase 4A / 4E now explicitly make control-rate, offload, placement, and degraded-mode consequences real instead of treating compute and battery as background commentary
  - Phase 5 now explicitly turns compute and battery into allocatable economic budget objects for inference, routing, simulation, diffusion, data collection, and conservation
- Changed: updated `docs/economic_world_model/roadmap.md` and `docs/economic_world_model/humanoid_target_readiness.md` to match:
  - the roadmap now carries a staged RL doctrine for these resources:
    - lower-WM prediction / calibration
    - bounded local allocation
    - economic cross-resource tradeoffs
    - only later meta-node Pareto policy
  - the humanoid-readiness checklist and benchmark matrix now include compute-envelope / placement budgeting, concrete battery-state contracts, and compute-pressure degradation as explicit readiness surfaces
- Why this matters:
  - it keeps compute and battery from showing up first as vague “energy” or “econ tensor” concerns
  - it also gives the roadmap a better anti-fake-standup rule: lower WMs and deployment layers must make these constraints real before higher layers are allowed to optimize over them
- Verification: `git diff --check` passed. This was a docs-only refinement.

- Changed: made the remaining Phase-1 runtime-root / policy gap more operational by teaching the WM to recognize OSS-shaped runtime layouts and policy banks:
  - added `src/world_model/sim_synth_physics/runtime_layouts.py` plus `scripts/scan_phase1_runtime_layouts.py`
  - Isaac/Unitree lanes now detect layout and policy posture for `IsaacLab`, `unitree_sim_isaaclab`, `unitree_rl_gym`, `HumanoidVerse`, `xr_teleoperate`, and Unitree asset/policy roots instead of flattening everything into a generic runtime-target bit
  - Holosoma lanes now detect repo, motion-bank, policy-bank, and retargeting-bundle posture as canonical backend metadata
  - backend bindings, bridge receipts, runtime work orders, and host-capability scans now preserve `runtime_layout_contract`, `policy_contract`, ready profiles, and policy-readiness truth
- Why this matters:
  - the honest remainder is now narrowed from “some runtime roots are missing” to “which concrete repo/profile/policy surface is missing on this host”
  - that is the right Phase-1 direction if we want the remaining blockers to become real roots/assets/policies/GPU instead of fuzzy adapter debt
- Verification: targeted `compileall`, targeted `ruff check`, `pytest -q tests/test_sim_synth_runtime_layouts.py tests/test_sim_synth_runtime_targets.py tests/test_sim_synth_physics_world_model.py tests/test_sim_synth_runtime_work_orders.py`, and `git diff --check` passed.

- Changed: turned those runtime-layout signals into WM-owned runtime bundles and launch specs instead of leaving them as discovery metadata only:
  - added `src/world_model/sim_synth_physics/runtime_bundles.py`
  - added `src/world_model/sim_synth_physics/runtime_launch.py` plus `scripts/run_phase1_runtime_launch.py`
  - `src/world_model/sim_synth_physics/backend_runtime_execution.py` now emits `backend_runtime_bundle_v1` and `backend_launch_spec_v1` artifacts for Isaac/Unitree and Holosoma lanes whenever the WM materializes a runtime request
  - runtime work orders now preserve those launch specs and append the preferred launch command to `command_hints`, so work orders point at an actual upstream-shaped bring-up path rather than only naming missing preconditions
  - when roots, assets, and policies are ready but no in-process backend bridge exists yet, the runtime can now emit `runtime_launch_prepared` instead of pretending the only blocker is a missing local module
  - the launch profiles are intentionally inspired by real OSS runtime shapes such as `unitree_sim_isaaclab`, `unitree_rl_gym`, `HumanoidVerse`, `IsaacLab`, and Holosoma, while staying inside the WM’s typed-contract posture
- Why this matters:
  - the Phase-1 backend lane now owns not just “what is missing” but “what should be launched next when the host is ready”
  - this is a material step toward the honest stopping condition the roadmap wants: concrete roots, assets, and policies become the blocker, not missing launch/bundle plumbing inside the repo
- Verification: targeted `compileall`, targeted `ruff check`, `pytest -q tests/test_sim_synth_runtime_layouts.py tests/test_sim_synth_runtime_targets.py tests/test_sim_synth_runtime_bundles.py tests/test_sim_synth_physics_world_model.py tests/test_sim_synth_runtime_work_orders.py`, and `git diff --check` passed.

- Changed: made the new backend bridge contract operational by emitting WM-owned backend runtime work orders:
  - `src/world_model/sim_synth_physics/runtime_work_orders.py` now compiles typed runtime bring-up work orders from the bridge receipt, runtime receipt, and robot-asset receipt
  - `src/world_model/sim_synth_physics/runtime.py` now writes `backend_runtime_work_orders.json` and threads work-order ids/statuses into loop summaries and training-feedback manifests
  - those work orders link directly to `scripts/NON_TRAINING_GPU_RUN_BACKLOG.json`, so the WM now names the concrete Isaac/Unitree or Holosoma runtime task to run later instead of only naming missing targets/assets abstractly
- Why this matters:
  - the Phase-1 backend lane now emits an executor-facing artifact, not just readiness descriptors
  - this narrows the remaining gap further toward actual host/runtime/GPU availability and away from missing planning-to-operations wiring
- Verification: targeted `compileall`, targeted `ruff check`, `pytest -q tests/test_sim_synth_physics_world_model.py tests/test_sim_synth_physics_scripts.py tests/test_sim_synth_runtime_work_orders.py`, and `git diff --check` passed.

- Changed: added a typed backend runtime bridge contract inside the Phase-1 sim/synth/physics WM:
  - `src/world_model/sim_synth_physics/runtime_bridge.py` now compiles `BackendRuntimeBridgeState` from backend binding, robot-asset contract, embodiment control constraints, and runtime-target readiness
  - the runtime now emits `backend_runtime_bridge_receipt_v1`, writes it into the loop artifact set, and threads its ids/status into outcome receipts, loop summaries, and training-feedback manifests
  - `src/world_model/sim_synth_physics/training_corpus.py` now harvests that receipt so backend-selector and branch-planner rows preserve bridge transport/readiness/authority truth instead of reconstructing it later
- Why this matters:
  - backend binding is no longer the last typed stop before runtime; the WM now explicitly owns the slow-loop to runtime bridge contract
  - Isaac/Unitree and Holosoma lanes can now name planner-vs-servo rates, transport profile, IO/telemetry contracts, safety channels, and missing runtime targets as canonical receipt truth
  - this is another Phase-1 shift from “described integration” to “owned integration contract”, which is the right direction before the remaining blockers become fully external runtime/assets/GPU constraints
- Verification: targeted `compileall`, targeted `ruff check`, `pytest -q tests/test_sim_synth_physics_world_model.py tests/test_sim_synth_training_corpus.py tests/test_sim_synth_physics_scripts.py`, and `git diff --check` passed.

- Changed: turned the Phase-1 backend runtime seam from request metadata into a WM-owned concrete runtime receipt path:
  - added `src/world_model/sim_synth_physics/backend_runtime_execution.py`
  - `src/world_model/sim_synth_physics/runtime.py` now emits `backend_runtime_execution_receipt_v1` for requested Isaac/Holosoma lanes, even when the main execution contract still falls back
  - when a real runtime module and policy id are present, the WM now prefers concrete `evaluate_policy(...)` execution through existing Isaac Lab / Holosoma backend seams and records rollout/metrics artifacts instead of stopping at shadow/work-order sidecars
  - `src/world_model/sim_synth_physics/training_corpus.py` now harvests that runtime receipt so backend-selector training can distinguish planning-only, shadow-runtime, and concrete-runtime bundles
- Changed: pushed the Phase-1 render-provider seam past pure work orders when real source artifacts and non-stub providers exist:
  - `src/world_model/sim_synth_physics/render_materialization.py` now materializes NAG counterfactual datapacks when a real source LSD episode and non-stub renderer path are available
  - it now materializes GGDS scene outputs when a real source Gaussian scene and concretely initialized optimizer are available
  - otherwise the WM stays on explicit work-order receipts with named preconditions instead of silently dropping into stub render defaults
- Verification: targeted `compileall`, targeted `ruff check`, `pytest -q tests/test_sim_synth_physics_world_model.py tests/test_sim_synth_training_corpus.py`, and `git diff --check` passed.
- Blocked: the remaining explicit Phase-1 gap is now even more clearly runtime/assets/policy/GPU constrained:
  - real Isaac Lab / Isaac Sim / Unitree runtime module plus assets and policies
  - real Holosoma host/runtime plus policy/data availability
  - real GGDS renderer/LDM initialization and source-scene corpus at scale

- Changed: preserved robot-asset readiness through the sim/synth training-corpus path:
  - `src/world_model/sim_synth_physics/training_corpus.py` now harvests `robot_asset_contract_receipt_v1`
  - backend-selector and branch-planner training rows now carry asset-contract refs, readiness score, and missing-asset signals instead of dropping that hardware-readiness truth at export time
- Verification: targeted `compileall`, targeted `ruff check`, `pytest -q tests/test_sim_synth_physics_world_model.py tests/test_sim_synth_physics_scripts.py tests/test_sim_synth_training_corpus.py tests/test_isaac_backend_shadow_contract.py`, and `git diff --check` passed.

- Changed: made the new robot-asset contract load-bearing inside backend materialization:
  - `src/world_model/sim_synth_physics/shadow_execution.py` now writes backend-local robot-asset, calibration, and IO sidecars for Isaac and Holosoma shadow materialization paths
  - backend shadow receipts now carry `robot_asset_contract_id`, sidecar refs, calibration contracts, observation contracts, and action contracts
  - `src/world_model/sim_synth_physics/runtime_evidence.py` and `src/world_model/sim_synth_physics/calibration.py` now react to missing-asset counts from those sidecars instead of treating asset readiness as a separate passive signal
- Verification: targeted `compileall`, targeted `ruff check`, `pytest -q tests/test_sim_synth_physics_world_model.py tests/test_sim_synth_physics_scripts.py tests/test_sim_synth_training_corpus.py tests/test_isaac_backend_shadow_contract.py`, and `git diff --check` passed.

- Changed: pulled Unitree/humanoid asset readiness into a canonical Phase-1 contract:
  - added `src/world_model/sim_synth_physics/asset_contracts.py`
  - `src/world_model/sim_synth_physics/compiler.py` now emits `RobotAssetContractState` on the WM state
  - `src/world_model/sim_synth_physics/runtime.py` now emits `robot_asset_contract_receipt_v1` and threads it into outcome/training/loop-summary artifacts
  - the sim/synth loop can now name concrete required assets, calibration contracts, observation contracts, and action contracts for hardware-target backends instead of only surfacing generic missing-asset strings
- Verification: targeted `compileall`, targeted `ruff check`, `pytest -q tests/test_sim_synth_physics_world_model.py tests/test_sim_synth_physics_scripts.py tests/test_sim_synth_training_corpus.py tests/test_isaac_backend_shadow_contract.py`, and `git diff --check` passed.

- Changed: pushed Phase-1 physics adaptation/calibration receipts closer to real runtime evidence:
  - `src/world_model/sim_synth_physics/runtime_evidence.py` now summarizes backend shadow execution, render materialization, and branch-outcome evidence
  - `src/world_model/sim_synth_physics/calibration.py` now folds that evidence into adaptation/calibration readiness metadata and scores instead of relying only on plan-time route/fidelity heuristics
  - `src/world_model/sim_synth_physics/runtime.py` now rebuilds those receipts after backend/render/outcome materialization so the emitted loop artifacts reflect actual WM loop evidence
- Verification: targeted `compileall`, targeted `ruff check`, `pytest -q tests/test_sim_synth_physics_world_model.py tests/test_sim_synth_physics_scripts.py tests/test_sim_synth_training_corpus.py tests/test_isaac_backend_shadow_contract.py`, and `git diff --check` passed.

- Changed: pushed the Phase-1 sim/synth backend/materialization loop further past compile-time-only ownership:
  - `src/world_model/sim_synth_physics/shadow_execution.py` now materializes explicit Holosoma shadow work orders in addition to Isaac shadow execution, so Holosoma-target planning windows emit WM-owned backend receipts and artifacts rather than stopping at binding metadata
  - `src/world_model/sim_synth_physics/render_materialization.py` now writes branch/provider artifacts for LSD scene configs and NAG/GGDS work orders, and `src/world_model/sim_synth_physics/runtime.py` now threads those artifacts into render-provider receipts, outcome receipts, and the training-feedback manifest
  - `src/world_model/sim_synth_physics/training_corpus.py` now carries render materialization status/mode/artifact refs into branch-planner training rows instead of flattening everything back to provider-kind only
  - the honest remaining Phase-1 gaps are now narrower: concrete Holosoma runtime execution, real Isaac/Unitree asset execution, and concrete NAG/GGDS renderer/LDM execution
- Verification: targeted `compileall`, targeted `ruff check`, `pytest -q tests/test_sim_synth_physics_world_model.py tests/test_sim_synth_physics_scripts.py tests/test_sim_synth_training_corpus.py tests/test_isaac_backend_shadow_contract.py`, and `git diff --check` passed.

- Changed: pushed the sim/synth/physics WM further into backend-runtime ownership instead of leaving Isaac as a dead class:
  - `src/envs/physics/isaac_backend.py` now provides an explicit shadow-contract backend with reset/step/media/summary/state APIs backed by `IsaacAdapter`
  - `src/world_model/sim_synth_physics/runtime.py` now emits `backend_shadow_execution_receipt_v1` for Isaac-target planning windows and writes shadow execution artifacts into the WM loop output
  - `src/world_model/sim_synth_physics/training_corpus.py` now harvests that receipt so backend-selector training rows can distinguish planning-only bundles from shadow-runtime bundles
  - the Phase-1 docs/backlog now describe the honest remaining gap as concrete Isaac Sim / Isaac Gym / Unitree asset execution rather than a literal backend stub
- Verification: targeted `compileall`, targeted `ruff check`, `pytest -q tests/test_isaac_backend_shadow_contract.py tests/test_sim_synth_physics_world_model.py tests/test_sim_synth_training_corpus.py tests/test_sim_synth_physics_scripts.py`, JSON validation for `scripts/FOUNDATION_MODEL_BRINGUP_BACKLOG.json`, and `git diff --check` passed.

- Changed: tightened the roadmap doctrine against "WM stands up as logging only":
  - `docs/economic_world_model/multi_wm_architecture_plan.md` now defines a mechanics-first WM readiness rule and a maturity ladder from `schema_only` through `production_recurrent`
  - the doctrine now explicitly treats neuralization as part of scalable mechanics rather than as a separate later luxury
  - the rule now explicitly requires all relevant downstream consumers for the future hardware-ready loop, not merely one consumer, before a WM can count as structurally real
  - `docs/economic_world_model/roadmap.md` and `docs/economic_world_model/humanoid_target_readiness.md` now mirror that rule so later phases cannot be declared complete on logging/demo-only behavior
- Verification: `git diff --check` passed. This was a docs-only roadmap tightening pass.

- Changed: added an explicit Phase 8 to the long-range planning docs:
  - `docs/economic_world_model/multi_wm_architecture_plan.md` now names a production-loop runtime / weekly GPU operations phase after the WM and meta-node phases
  - `docs/economic_world_model/roadmap.md` now describes the same endgame as Phase C: external dataset aggregation, loop runs, corpus export, training, fine-tuning, benchmarking, promotion/redeployment, then latency/inference focus
  - `docs/economic_world_model/humanoid_target_readiness.md` now connects the post-September-2027 posture to recurring GPU/Runpod execution and backlog exhaustion rather than leaving it as an implied next step
- Verification: `git diff --check` passed. This was a docs-only roadmap extension.

- Changed: tightened the long-range Unitree target in the planning docs:
  - July 2027 now remains the purchase / initial integration milestone
  - September 30, 2027 is now the explicit stronger target for sustainably autonomous G1 operation
  - the docs now say that by that date the control loop should be running repeatedly, collecting replay/telemetry/governance receipts, and feeding recurring improvement cycles rather than still behaving like a one-off bring-up effort
- Changed: aligned the roadmap consequences of that stronger target:
  - `docs/economic_world_model/roadmap.md` now extends the post-September 2026 execution program through September 2027
  - `docs/economic_world_model/humanoid_target_readiness.md` now adds the post-purchase conversion window from July through August 2027 and the September 2027 autonomy bar
  - `docs/economic_world_model/multi_wm_architecture_plan.md` now makes the same target explicit so lower-WM and deployment-enabler phases are judged against recurring on-robot loop readiness, not just pre-purchase structure
- Verification: `git diff --check` passed. This was a docs-only target-tightening pass.

- Changed: refined the dated pre-G1 roadmap into an explicit weekly operating model after training begins:
  - starting September 1, 2026, the docs now assume a weekly A100-backed program
  - work is explicitly described as sub-module by sub-module inside each WM
  - each weekly pass now follows the order: loop runs, receipt/corpus export, training runs, then fine-tuning only where the gates justify it
  - the initial order of attack is now written as sim/synth/physics first, then perception/grounding, then embodiment/actuation, then economic-WM consolidation, then local meta-node neuralization and later meta-node superposition/control
- Verification: `git diff --check` passed. This was a docs-only scheduling refinement.
- Next recommended task: turn that weekly A100 doctrine into a WM-by-WM execution table with named sub-modules, target loop runs, target trainers, and entry/exit gates for each week of the first September-to-December training season.

- Changed: turned the pre-G1 roadmap into a dated program assumption instead of an undated aspiration:
  - `docs/economic_world_model/multi_wm_architecture_plan.md` now assumes serious multi-WM training starts on September 1, 2026 and says the current architecture should have its plumbing laid by August 31, 2026
  - `docs/economic_world_model/roadmap.md` now splits the work into a plumbing-first window through August 31, 2026 and a training/calibration/Unitree-hardening window from September 1, 2026 through July 2027
  - `docs/economic_world_model/humanoid_target_readiness.md` now frames July 2027 as a pre-purchase readiness window where remaining blockers should be hardware/data/calibration/benchmark limits, not missing canonical plumbing
- Verification: `git diff --check` passed. This was a docs-only scheduling refinement.
- Next recommended task: translate the August 31, 2026 plumbing deadline into a per-WM checklist with explicit `must_be_real_by_sep_2026` items for sim/synth/physics, perception/grounding, embodiment/actuation, and economic-WM ingestion.

- Changed: updated the WM architecture docs and model-shaped backlogs so V-JEPA 2 is no longer treated as a vague future ingredient for only one stack slice:
  - `docs/economic_world_model/multi_wm_architecture_plan.md` now places V-JEPA 2 explicitly in both the Phase-1 sim/synth/physics WM and the later Phase-2 perception/grounding WM
  - `docs/economic_world_model/roadmap.md` now says to prefer upstream `facebookresearch/vjepa2` bring-up where that is faster and more honest than local reimplementation, while keeping it behind typed provider/runtime contracts and receipts
  - `scripts/FOUNDATION_MODEL_BRINGUP_BACKLOG.json`, `scripts/LOOP_RUN_BACKLOG.json`, and `scripts/TRAINING_MIGRATION_BACKLOG.json` now carry explicit V-JEPA 2 bring-up, loop-run, and fine-tuning backlog items for both WM lanes
  - `docs/economic_world_model/full_stack_training_backlog.md` now records the same split on the fine-tuning/training side instead of leaving it as architecture prose only
- Verification: `python3 -m json.tool scripts/FOUNDATION_MODEL_BRINGUP_BACKLOG.json >/dev/null`, `python3 -m json.tool scripts/LOOP_RUN_BACKLOG.json >/dev/null`, `python3 -m json.tool scripts/TRAINING_MIGRATION_BACKLOG.json >/dev/null`, and `git diff --check` passed. This was a docs/backlog refinement.
- Next recommended task: when the first V-JEPA 2 runtime wrapper lands, emit provider-truth, calibration, and benchmark-gate receipts directly into the sim/synth/physics and perception/grounding WM state surfaces rather than normalizing it as a generic latent helper.

- Changed: converted the live Stage-1 diffusion path from an actively used stub class into a real runtime/provider contract:
  - added `src/diffusion/video_diffusion_runtime.py`
  - `scripts/run_stage1_pipeline.py` and `src/orchestrator/diffusion_requests.py` now call `VideoDiffusionRuntime` rather than instantiating the stub directly
  - the runtime now records explicit provider truth (`real`, `heuristic_fallback`, `disabled`, `stub`) and materialization posture on every proposal/datapack/admission record instead of leaving diffusion status implicit
  - `auto` now means governed planning with honest planning-only fallback when no real diffusers checkpoint is locally available; `real` is strict real-or-unavailable
- Changed: tightened another model-stub seam in `scripts/train_ggds_on_lsd_vector_scenes.py`:
  - the GGDS training harness no longer silently defaults to a dummy LDM in `auto`
  - smoke use now requires explicit `--backend-policy stub`
  - the script now emits LDM provider truth into its summary and raises honestly when no real backend is configured
- Changed: added `scripts/FOUNDATION_MODEL_BRINGUP_BACKLOG.json` and updated `scripts/LOOP_RUN_BACKLOG.json` / `scripts/TRAINING_MIGRATION_BACKLOG.json` so remaining model-shaped gaps are tracked as real bring-up/fine-tune/training work with OSS targets rather than as vague future cleanup:
  - governed video diffusion
  - GGDS/LDM renderer stack
  - vision backbone stub replacement
  - semantic VLA placeholder replacement
  - Isaac/Unitree execution bring-up
- Changed: strengthened the cross-phase doctrine in `docs/economic_world_model/multi_wm_architecture_plan.md` and `docs/economic_world_model/roadmap.md`:
  - no new WM or enabling phase should default to literal stubs when a real-or-unavailable provider contract is possible
  - `stub` should be explicit-only for smoke/scaffolding
  - the normal target posture is real runtime/provider plumbing with GPU/weights/assets as the honest blocker
- Verification: `python3 -m compileall src/diffusion src/orchestrator/diffusion_requests.py scripts/run_stage1_pipeline.py scripts/train_ggds_on_lsd_vector_scenes.py tests/test_video_diffusion_runtime.py tests/test_video_diffusion_stub_routing.py tests/test_stage1_pipeline_governed.py tests/test_lsd_integration.py -q`, `python3 -m ruff check src/diffusion src/orchestrator/diffusion_requests.py scripts/run_stage1_pipeline.py scripts/train_ggds_on_lsd_vector_scenes.py tests/test_video_diffusion_runtime.py tests/test_video_diffusion_stub_routing.py tests/test_stage1_pipeline_governed.py tests/test_lsd_integration.py`, `python3 -m pytest -q tests/test_video_diffusion_runtime.py tests/test_video_diffusion_stub_routing.py tests/test_stage1_pipeline_governed.py`, `python3 -m pytest -q tests/test_lsd_integration.py::TestGGDSTraining::test_dummy_ldm tests/test_lsd_integration.py::TestGGDSTraining::test_load_ldm_auto_does_not_silently_stub`, `python3 -m json.tool scripts/FOUNDATION_MODEL_BRINGUP_BACKLOG.json >/dev/null`, `python3 -m json.tool scripts/LOOP_RUN_BACKLOG.json >/dev/null`, `python3 -m json.tool scripts/TRAINING_MIGRATION_BACKLOG.json >/dev/null`, and `git diff --check` passed.
- Blocked: `tests/test_lsd_integration.py::TestGGDSTraining::test_ggds_smoke` did not complete promptly after this change-set, so I did not claim a full-file pass for the older LSD integration harness.

- Changed: refined `docs/economic_world_model/multi_wm_architecture_plan.md` and `docs/economic_world_model/roadmap.md` so the repo now says explicitly that multi-WM work must rerun the deterministic-prior / heuristic audit inside each WM boundary rather than assuming the earlier heuristic purge finished the job globally.
- Changed: the docs now frame the earlier heuristic pass correctly:
  - it was a high-value repo-wide sweep over the live stack
  - it was not a substitute for per-WM review once new canonical WM boundaries are introduced
  - each WM tranche now needs its own explicit disposition of deterministic owners vs fallback priors vs learned/runtime-package seams
- Verification: `git diff --check` passed. This was a docs-only refinement.
- Next recommended task: as each new WM tranche lands, add a short per-WM heuristic-review checklist and receipt of what was kept as fallback prior vs neuralized seam so the doctrine stays operational.

- Changed: refined `docs/economic_world_model/multi_wm_architecture_plan.md` and `docs/economic_world_model/roadmap.md` so the ontology story is explicit and no longer easy to blur:
  - operational / module-level ontology is now named as the in-stack cybernetic digital-twin layer for entities, tasks, datapacks, events, provenance, governance hooks, and module/runtime state
  - WM-transport ontology is now named separately as the typed semantic/governance contract for adjacent-WM interoperability
  - the docs now say explicitly that the isomorphic tensor / transport bridge is the differentiable realization of the WM-transport contract, not a replacement for ontology and not a reason to collapse everything into one symbolic or latent mother-layer
- Changed: made the RL / training roles explicit for both ontology layers:
  - operational ontology training is framed around module-to-ontology fidelity, event/state prediction, temporal consistency, uncertainty calibration, provenance quality, and governance satisfaction
  - WM-transport ontology training is framed around WM-to-ontology-to-WM translation quality, topology/causal/dependency preservation, synchronized-loop success, and decomposed bridge-only vs downstream-only vs joint gains
  - both layers are tied to completed-loop/postmortem quality, governance satisfaction, counterfactual improvement, and downstream yield rather than to immediate takeover of frozen core reward math
- Changed: made current-state honesty explicit in the roadmap docs:
  - today the repo mostly has operational ontology substrate/plumbing
  - it does not yet have a fully neural ontology layer
  - it does not yet have a full WM-transport ontology implementation
  - sequencing remains lower WMs first, then economic-WM consolidation, then ontology-mediated WM transport
- Verification: `git diff --check` passed. This was a docs-only refinement.
- Next recommended task: when the next lower-WM tranche lands, thread the operational/module ontology language into its state contracts and receipts directly, while keeping the WM-transport ontology reserved for the later adjacent-WM bridge phase.

## 2026-03-26

- Changed: completed the follow-on `sim_synth_physics` helper-package tranche and started the advisory pivot for that subsystem:
  - added real trainer/export/runtime-package lanes for the WM backend selector and branch planner:
    - `scripts/train_sim_synth_backend_selector.py`
    - `scripts/train_sim_synth_branch_planner.py`
    - `src/world_model/sim_synth_physics/backend_selector.py`
    - `src/world_model/sim_synth_physics/backend_selector_runtime.py`
    - `src/world_model/sim_synth_physics/branch_planner.py`
    - `src/world_model/sim_synth_physics/branch_planner_runtime.py`
  - the live compiler/runtime wrappers now accept those helper packages through the real WM/runtime path instead of leaving them as detached training utilities:
    - `src/world_model/sim_synth_physics/compiler.py`
    - `src/orchestrator/semantic_simulation.py`
    - `src/orchestrator/diffusion_requests.py`
    - `src/orchestrator/coverage_loop.py`
  - helper packages now resolve package-relative checkpoints and carry explicit target-hardware/subsystem-posture metadata, so the emitted artifacts are closer to production-shaped packages than local-path training leftovers
  - the trainer/export lane now also accepts canonical WM runtime receipt bundles rather than only pre-shaped row datasets:
    - `src/world_model/sim_synth_physics/training_corpus.py` projects `SimSynthPhysicsWorldState`, calibration receipts, and simulation-outcome receipts into backend-selector and branch-planner rows
    - the training scripts now emit a compiled dataset artifact even when the source input was runtime receipts, which is the right direction for a real subsystem corpus lane
  - fixed a real contract mismatch while landing that path: the branch-planner runtime context now includes `heuristic_generation_mode`, so the trained feature contract matches live inference instead of silently degrading
- Changed: started the advisory follow-up for this subsystem by tightening the doctrine in `docs/economic_world_model/advisory_purge_wiring_plan.md`: sim/synth backend and branch helper lanes are now part of the bounded-authority bucket rather than another advisory-shaped planning seam.
- Changed: tightened the subsystem doctrine in the core planning docs so each WM is treated as a damn-near-production-ready subsystem target with honest remaining blockers:
  - `docs/economic_world_model/multi_wm_architecture_plan.md`
  - `docs/economic_world_model/roadmap.md`
  - `docs/economic_world_model/humanoid_target_readiness.md`
  - the explicit target remains Unitree G1/R1-class readiness, and the remaining blockers are named as data, GPUs, calibration truth, benchmark evidence, and Unitree-class assets rather than missing neural/package/runtime scaffolding
- Verification: `python3 -m compileall src/world_model/sim_synth_physics src/orchestrator/semantic_simulation.py src/orchestrator/diffusion_requests.py src/orchestrator/coverage_loop.py scripts/train_sim_synth_backend_selector.py scripts/train_sim_synth_branch_planner.py tests/test_sim_synth_physics_world_model.py tests/test_train_sim_synth_backend_selector.py tests/test_train_sim_synth_branch_planner.py -q`, `python3 -m ruff check src/world_model/sim_synth_physics src/orchestrator/semantic_simulation.py src/orchestrator/diffusion_requests.py src/orchestrator/coverage_loop.py scripts/train_sim_synth_backend_selector.py scripts/train_sim_synth_branch_planner.py tests/test_sim_synth_physics_world_model.py tests/test_train_sim_synth_backend_selector.py tests/test_train_sim_synth_branch_planner.py`, `python3 -m pytest -q tests/test_sim_synth_physics_world_model.py tests/test_coverage_compilation.py tests/test_gap_agenda_ranking.py tests/test_coverage_loop.py tests/test_train_sim_synth_backend_selector.py tests/test_train_sim_synth_branch_planner.py`, and `git diff --check` passed.
- Blocked: honest promotion blockers for this subsystem are now clearer:
  - Unitree-class sim adapters and robot assets
  - grounded whole-body replay and branch corpora
  - calibration receipts for contact-rich whole-body behavior
  - GPU budget for materially larger helper training/eval
  - G1/R1-class benchmark evidence for promotion beyond `auto`
- Next recommended task: wire the compiled receipt-derived datasets into real artifact harvesters and branch-execution paths so the backend-selector and branch-planner corpora are sourced from live sim receipts by default rather than only by manual receipt-bundle export.

- Changed: completed the next `sim_synth_physics` tranche by threading the canonical inferential learnability contract into WM-owned agenda ranking, synthetic-branch planning, gen2sim admission, and diffusion ordering:
  - `src/world_model/sim_synth_physics/compiler.py` now assigns inferential learnability contracts to simulation jobs and branch plans, uses them as bounded ranking priors, and includes job/branch inferential summaries in WM metadata
  - `src/world_model/sim_synth_physics/diffusion_contracts.py` now preserves inferential contracts and diffusion-priority scores on WM-owned diffusion plans instead of treating branch admission as a side note
  - `DiffusionConditioningState` and `Gen2SimAdmissionState` now carry admissible-vs-blocked branch splits plus inferential summaries so diffusion/render budgeting is sourced from the WM rather than implicit orchestration defaults
  - `src/orchestrator/coverage_loop.py` now surfaces the WM inferential summaries in its runtime summary so downstream readiness and replay consumers can see the new planning truth explicitly
- Changed: updated `docs/economic_world_model/multi_wm_architecture_plan.md` to make the cross-phase rule explicit that epiplexity-based inferential learnability should be carried by downstream WMs as canonical typed metadata once they affect replay, admission, simulation, diffusion, or training selection.
- Verification: `python3 -m compileall src/world_model/sim_synth_physics src/orchestrator/coverage_loop.py src/orchestrator/diffusion_requests.py tests/test_sim_synth_physics_world_model.py tests/test_gap_agenda_ranking.py tests/test_coverage_compilation.py -q`, `python3 -m ruff check src/world_model/sim_synth_physics src/orchestrator/coverage_loop.py src/orchestrator/diffusion_requests.py tests/test_sim_synth_physics_world_model.py tests/test_gap_agenda_ranking.py tests/test_coverage_compilation.py`, `python3 -m pytest -q tests/test_sim_synth_physics_world_model.py tests/test_gap_agenda_ranking.py tests/test_coverage_compilation.py tests/test_coverage_loop.py`, and `git diff --check` passed.
- Next recommended task: add the first real trainer/export/runtime-package contracts for WM backend-selector and branch-planner helpers, then feed branch-outcome receipts back into the new inferential summaries instead of leaving them as one-pass priors.

- Changed: completed the first code tranche from `docs/economic_world_model/advisory_purge_wiring_plan.md`. Added `src/economics/inferential_contract.py` as the shared canonical learnability/admission contract, then wired it through:
  - `src/replay/dataset.py` for per-episode `inferential_learnability_contract` plus manifest-level `inferential_learnability_summary`
  - `src/orchestrator/shadow_advisory.py` for inferential learnability summaries and canonical inferential work-order emission
  - `src/orchestrator/adaptation_budgeting.py` for shared inferential execution-work-order construction
  - `src/rl/episode_sampling.py` so replay descriptors can consume the canonical inferential contract instead of recomputing solely from scattered summary fields
  - `src/regality/promotion_reporting.py` so promotion evidence can now see learnability-class density directly
  - `src/training/training_manifest.py` and `src/training/regal_training_runner.py` so canonical training manifests persist inferential learnability and inferential work-order summaries
- Changed: the main shadow-training entrypoints now write and register explicit inferential artifacts instead of burying them inside `shadow_advisory.json` only:
  - `scripts/train_shadow_offline_rl.py`
  - `scripts/train_shadow_replay_policy.py`
  - `scripts/train_shadow_pricing_models.py`
  - `scripts/train_sac_with_ontology_logging.py`
  - `scripts/run_shadow_advisory_pass.py`
- Changed: updated `docs/epiplexity.md` so it reflects the newly landed current state; replay and training now have a canonical inferential contract layer even though broader learnability promotion is still incomplete.
- Verification: `python3 -m compileall src/economics/inferential_contract.py src/economics/inferential_training_gate.py src/orchestrator/adaptation_budgeting.py src/orchestrator/shadow_advisory.py src/orchestrator/queue_selection.py src/replay/dataset.py src/replay/receipt_ingest.py src/regality/promotion_reporting.py src/rl/episode_sampling.py src/training/training_manifest.py src/training/regal_training_runner.py scripts/train_shadow_offline_rl.py scripts/train_shadow_replay_policy.py scripts/train_shadow_pricing_models.py scripts/train_sac_with_ontology_logging.py scripts/run_shadow_advisory_pass.py tests/test_inferential_contract.py tests/test_replay_dataset.py tests/test_receipt_ingest.py tests/test_training_manifest.py tests/test_shadow_advisory_pass.py tests/test_promotion_reporting.py -q`, `python3 -m ruff check src/economics/inferential_contract.py src/economics/inferential_training_gate.py src/orchestrator/adaptation_budgeting.py src/orchestrator/shadow_advisory.py src/orchestrator/queue_selection.py src/replay/dataset.py src/replay/receipt_ingest.py src/regality/promotion_reporting.py src/rl/episode_sampling.py src/training/training_manifest.py src/training/regal_training_runner.py scripts/train_shadow_offline_rl.py scripts/train_shadow_replay_policy.py scripts/train_shadow_pricing_models.py scripts/train_sac_with_ontology_logging.py scripts/run_shadow_advisory_pass.py tests/test_inferential_contract.py tests/test_replay_dataset.py tests/test_receipt_ingest.py tests/test_training_manifest.py tests/test_shadow_advisory_pass.py tests/test_promotion_reporting.py`, `python3 -m pytest -q tests/test_inferential_contract.py tests/test_inferential_training_gate.py tests/test_replay_dataset.py tests/test_receipt_ingest.py tests/test_training_manifest.py tests/test_shadow_advisory_pass.py tests/test_promotion_reporting.py tests/test_queue_dispatch_policy.py tests/test_queue_dispatch_integration.py tests/test_online_queue_dispatch_integration.py tests/test_online_promotion_reporting.py`, and `git diff --check` passed.
- Next recommended task: take the second advisory tranche by reclassifying live queue/curriculum/orchestration outputs from "advisory-only" to explicit bounded-authority receipts, then thread the inferential learnability contract into synthetic-branch admission and sim/diffusion agenda ranking.

- Changed: started Phase 1A / 1B from `docs/economic_world_model/multi_wm_architecture_plan.md` by landing the first canonical `src/world_model/sim_synth_physics/` package. The new additive WM boundary now owns:
  - typed `SimSynthPhysicsWorldState`
  - WM-owned `SimulationAgenda` / `SimulationJobSpec`
  - `PhysicsContextState`
  - `DiffusionConditioningState`
  - `SyntheticBranchPlan`
  - `Gen2SimAdmissionState`
  - canonical receipt contracts for calibration and simulation outcomes
- Changed: moved live simulation-agenda compilation out of `src/orchestrator/semantic_simulation.py` and into the new WM compiler/runtime boundary. The orchestration surface now consumes the WM-owned agenda contract and returns the legacy agenda view only for compatibility.
- Changed: wired bounded learned seams into the new WM boundary from the start instead of leaving them as a later cleanup:
  - agenda ranking continues to use the existing promoted/shadow gap-ranker path
  - backend/fidelity selection now has a benchmark-gated learned-helper seam
  - synthetic branch planning now has a benchmark-gated learned-helper seam
  - both seams record helper status, promotion stage, and trace receipts while keeping heuristics as explicit priors/fallbacks
- Changed: moved gap-driven diffusion prompt compilation onto the WM-owned state instead of recomputing it inside orchestration helpers:
  - added `src/world_model/sim_synth_physics/diffusion_contracts.py` as the WM-owned diffusion contract layer
  - `src/orchestrator/diffusion_requests.py` now adapts WM-owned diffusion plans instead of re-ranking coverage gaps locally
  - `src/orchestrator/coverage_loop.py` now compiles one `SimSynthPhysicsWorldState` and derives both the simulation agenda and diffusion prompts from that shared state surface
- Changed: this means the current live control plane no longer has separate agenda-vs-diffusion gap compilers drifting apart inside the coverage loop; diffusion conditioning is now sourced from `DiffusionConditioningState` plus WM-owned branch plans and physics context.
- Changed: updated `docs/economic_world_model/multi_wm_architecture_plan.md` and `docs/economic_world_model/roadmap.md` to make the repo rule explicit across future WMs and enabler phases:
  - no new WM should land as a heuristic-only island
  - bounded learned seams, promotion posture, and receipt traces should exist from the first tranche
- Verification: `python3 -m compileall src/world_model/sim_synth_physics src/orchestrator/semantic_simulation.py src/orchestrator/diffusion_requests.py src/orchestrator/coverage_loop.py tests/test_sim_synth_physics_world_model.py tests/test_gap_agenda_ranking.py -q`, `python3 -m ruff check src/world_model/sim_synth_physics src/orchestrator/semantic_simulation.py src/orchestrator/diffusion_requests.py src/orchestrator/coverage_loop.py tests/test_sim_synth_physics_world_model.py tests/test_gap_agenda_ranking.py`, `python3 -m pytest -q tests/test_sim_synth_physics_world_model.py tests/test_coverage_compilation.py tests/test_gap_agenda_ranking.py tests/test_coverage_loop.py`, and `git diff --check` passed.
- Next recommended task: add package-loading runtime shims plus first trainer/export contracts for the new backend-selector and branch-planner seams, then start routing backend-quality and branch-outcome receipts into replay/training artifacts.

- Changed: added `docs/economic_world_model/advisory_purge_wiring_plan.md` as the advisory counterpart to the earlier heuristic/sidecar sweep. The new document:
  - narrows the repo-wide advisory doctrine
  - separates surfaces that should remain advisory from surfaces that should become canonical metadata, preconditions, work orders, bounded authority, or later benchmark-gated successors
  - ranks the current advisory gaps with epiplexity / inferential signal-yield / inferential work-order promotion as the top remaining tranche
  - updates the architectural posture on frozen Phase B math: keep it as the rollback anchor now, but do not treat it as philosophically immutable forever once successor layers earn replacement through evidence
- Changed: updated `docs/epiplexity.md` so the current posture is explicit: epiplexity remains bounded and non-reward-changing today, but it is now documented as the likely future canonical learnability class rather than a permanently advisory overlay.
- Changed: updated `docs/economic_world_model/multi_wm_architecture_plan.md` and `docs/economic_world_model/roadmap.md` to reflect the same doctrine shift. The multi-WM plan now says internal WM-to-WM state and receipts should not stay culturally advisory once they affect runtime or training, and the roadmap now encodes the same narrower advisory rule plus the benchmark-gated successor stance on frozen Phase B math.
- Verification: `git diff --check` passed. This was a docs-only advisory doctrine pass; no code paths changed.
- Next recommended task: execute the new top tranche in code by promoting epiplexity / inferential signal-yield from overlay-shaped evidence into canonical learnability metadata and executor-facing inferential work-order contracts across replay, training manifests, and benchmark reporting.

- Changed: added `docs/economic_world_model/humanoid_target_readiness.md` as the concrete follow-on artifact for the new G1/R1-facing plan. It turns the hardware-target discussion into:
  - an explicit readiness checklist
  - a benchmark matrix
  - a repo-grounded gap map
  - a model-capacity review target list
  - a Unitree sim-env integration checklist
  - companion-compute / comms / calibration / teleop fallback requirements
- Changed: linked that readiness artifact back into `docs/economic_world_model/multi_wm_architecture_plan.md` so the humanoid-target sections now point to a concrete checklist instead of only future intent.
- Verification: `git diff --check` passed. This was a docs-only planning pass; no code paths changed.
- Next recommended task: when the repo returns from planning to implementation, use `docs/economic_world_model/humanoid_target_readiness.md` as the acceptance checklist for the eventual Unitree sim integration and embodiment-contract refit work.

- Changed: added `docs/economic_world_model/multi_wm_architecture_plan.md` as the new multi-stage architecture plan for the next world-model stack. The document makes the topology explicit:
  - perception / grounding WM
  - embodiment / actuation WM
  - sim / synth / physics WM
  - economic WM over those lower WMs
  - meta-node superposition / control WM above the economic WM
- Changed: made the sequencing rule explicit instead of leaving it implicit in cross-window discussion. The plan argues for:
  - building the sim / synth / physics WM next
  - treating the cross-WM “isomorphic transport” idea as middleware between adjacent canonical WMs rather than as a premature mother-latent
  - delaying deep economic-WM neuralization and the later meta-node WM until lower WMs emit stable canonical state
  - requiring a dedicated local meta-node neuralization / robustness tranche before any overarching meta-node superposition WM
- Changed: documented the concrete Phase 1 module structure for the sim / synth / physics WM, including its proposed package boundary, typed state objects, runtime flow, receipt surfaces, OSS-provider posture, and the current repo files that should be absorbed into that boundary instead of continuing to own agenda compilation independently.
- Changed: extended the plan with explicit G1/R1-class hardware implications instead of treating humanoid readiness as an afterthought. The plan now calls out:
  - a future model-capacity audit for lower-WM and submodule models
  - the fact that current workcell/tabletop envs are useful skill islands but not sufficient humanoid-readiness proxies
  - an explicit later sim-env integration lane for Unitree G1/R1-class simulation
  - the need to refit observation/action contracts around proprioception, IMU, force/torque, whole-body state, latency, and spatial state
  - explicit future phases for companion-compute / communication middleware and operator / teleop / recovery fallback
  - robot asset and calibration management as a first-class future concern
  - a humanoid-specific benchmark taxonomy rather than only stronger workcell benchmarks
  - a dedicated later phase for humanoid target capacity and environment refit before claiming real hardware-readiness
- Changed: added named future phases with explicit preconditions for:
  - perception / grounding WM
  - embodiment / actuation WM
  - humanoid target capacity and environment refit
  - real-time servo vs governance loop separation
  - sensor-fusion shim
  - physical safety layer
  - spatial state / SLAM integration
  - economic-WM consolidation
  - cross-WM typed transport bridges
  - local meta-node neuralization / robustness
  - the later meta-node superposition / control WM
- Verification: `git diff --check` passed. This was a docs-only planning pass; no code paths changed.
- Next recommended task: start Phase 1A / 1B from the new plan by defining the `sim_synth_physics` package boundary and moving simulation/diffusion/branch-agenda ownership out of scattered orchestrator surfaces into that canonical WM layer.

- Changed: completed the `PipelineManager` stage-activation helper pass. `src/orchestrator/pipeline_stage_policy.py` now defines an explicit feature/target contract over pipeline history, execution-precondition truth, progress trends, and stage outcomes; `src/orchestrator/pipeline_stage_policy_training.py` now trains a bounded helper over real `PipelineManager` state receipts; and `scripts/train_pipeline_stage_policy.py` now emits canonical dataset/precondition/package/runtime artifacts under `RegalTrainingRunner`.
- Changed: the pipeline shell now affects the real runtime boundary rather than only a static preview. `src/orchestrator/pipeline_stage_policy_runtime.py` now loads bounded helper packages with `disabled|auto|required` semantics, and `src/orchestrator/pipeline_manager.py` now reorders stage-activation plans by bounded learned priority, lets the helper influence next-iteration config flags, and preserves `policy_source`, `promotion_stage`, and `stage_policy_trace` receipts for future learning.
- Verification: `python3 -m compileall src/orchestrator/pipeline_stage_policy.py src/orchestrator/pipeline_stage_policy_training.py src/orchestrator/pipeline_stage_policy_runtime.py src/orchestrator/pipeline_manager.py scripts/train_pipeline_stage_policy.py tests/test_pipeline_stage_policy.py tests/test_train_pipeline_stage_policy.py -q`, `python3 -m ruff check src/orchestrator/pipeline_stage_policy.py src/orchestrator/pipeline_stage_policy_training.py src/orchestrator/pipeline_stage_policy_runtime.py src/orchestrator/pipeline_manager.py scripts/train_pipeline_stage_policy.py tests/test_pipeline_stage_policy.py tests/test_train_pipeline_stage_policy.py`, `python3 -m pytest -q tests/test_pipeline_stage_policy.py tests/test_train_pipeline_stage_policy.py tests/test_shell_activation.py`, and `python3 scripts/check_training_regality.py --scripts-dir scripts` passed.
- Blocked: the higher-order orchestrator shell gap is no longer in `PipelineManager`; the main remaining live heuristic control-plane seam is now queue/curriculum weighting over replay selection, plus the external grounded-data reality gate on real SAM3D.
- Next recommended task: take `src/orchestrator/queue_selection.py` and `src/rl/episode_sampling.py` next, because that is now the thickest remaining fake boundary inside the production loop itself.

- Changed: completed the `SemanticOrchestratorV2` shell-policy helper pass. `src/orchestrator/orchestrator_shell_policy.py` now defines a shared feature/target contract over semantic snapshots and orchestrator advisories, `src/orchestrator/orchestrator_shell_policy_training.py` now trains a bounded multi-head helper over real snapshot-plus-advisory receipts, and `scripts/train_orchestrator_shell_policy.py` now emits canonical dataset/precondition/package/runtime artifacts under `RegalTrainingRunner` instead of leaving this lane heuristic-only.
- Changed: the shell helper now affects the real runtime boundary instead of only a notebook-like trainer. `src/orchestrator/orchestrator_shell_policy_runtime.py` now loads bounded helper packages with `disabled|auto|required` semantics, and `src/orchestrator/semantic_orchestrator_v2.py` now blends learned preset/strategy/safety/activation preferences against the explicit heuristic prior while recording `policy_source`, `promotion_stage`, and `helper_trace` receipts.
- Verification: `python3 -m compileall src/orchestrator/orchestrator_shell_policy.py src/orchestrator/orchestrator_shell_policy_training.py src/orchestrator/orchestrator_shell_policy_runtime.py src/orchestrator/semantic_orchestrator_v2.py scripts/train_orchestrator_shell_policy.py tests/test_orchestrator_shell_policy.py tests/test_train_orchestrator_shell_policy.py -q`, `python3 -m ruff check src/orchestrator/orchestrator_shell_policy.py src/orchestrator/orchestrator_shell_policy_training.py src/orchestrator/orchestrator_shell_policy_runtime.py src/orchestrator/semantic_orchestrator_v2.py scripts/train_orchestrator_shell_policy.py tests/test_orchestrator_shell_policy.py tests/test_train_orchestrator_shell_policy.py`, `python3 -m pytest -q tests/test_orchestrator_shell_policy.py tests/test_train_orchestrator_shell_policy.py tests/test_shell_activation.py`, and `python3 scripts/check_training_regality.py --scripts-dir scripts` passed.
- Blocked: the remaining orchestration gap is now narrower and higher-order. `SemanticOrchestratorV2` is no longer purely heuristic, but `PipelineManager` still assembles stage activation and pipeline-shell choices mostly deterministically above the newly real shell/selector/meta-transformer/knob lanes.
- Next recommended task: neuralize `PipelineManager` stage activation and then the queue/curriculum weighting core, because those are now the main remaining fake boundaries in the live control loop.

- Changed: completed the meta-transformer planning-helper pass. `src/orchestrator/meta_transformer_planning.py` now defines a shared planning-context contract over semantic-WM, econ, datapack, and selector meta-choice receipts; `src/orchestrator/semantic_runtime_learning.py` now exports those receipts plus explicit objective/backend/energy-mix/data-mix/expected-delta targets into the runtime dataset; and `src/orchestrator/meta_transformer_training.py` now trains those planning heads directly on the real `MetaTransformerNet` substrate instead of leaving them as post-inference heuristics.
- Changed: the runtime helper now affects the actual meta-choice surface instead of only embeddings. `src/orchestrator/meta_transformer_runtime.py` now decodes learned objective/backend/data-mix/energy-profile/expected-delta outputs and records planning traces, while `src/orchestrator/meta_transformer.py` now blends those learned candidates against the explicit heuristic prior with bounded `shadow_candidate` vs `promoted` semantics and records a `planning_application` receipt explaining why the final choice stayed prior-backed or moved toward the helper.
- Changed: synthetic fallback parity is materially better. `scripts/train_meta_transformer_synthetic.py` and the synthetic sample generator now carry the same planning target contract as the heavyweight runtime dataset, so the lightweight path is no longer “authority/tokens only” while the production path carries the real planning surface.
- Verification: `python3 -m compileall src/orchestrator/meta_transformer_planning.py src/orchestrator/meta_transformer_training.py src/orchestrator/meta_transformer_runtime.py src/orchestrator/meta_transformer.py src/orchestrator/semantic_runtime_learning.py scripts/train_meta_transformer_synthetic.py tests/test_train_meta_transformer_synthetic.py tests/test_meta_transformer_runtime.py tests/test_semantic_runtime_learning.py tests/test_semantic_transformer_execution.py -q`, `python3 -m ruff check src/orchestrator/meta_transformer_planning.py src/orchestrator/meta_transformer_training.py src/orchestrator/meta_transformer_runtime.py src/orchestrator/meta_transformer.py src/orchestrator/semantic_runtime_learning.py scripts/train_meta_transformer_synthetic.py tests/test_train_meta_transformer_synthetic.py tests/test_meta_transformer_runtime.py tests/test_semantic_runtime_learning.py tests/test_semantic_transformer_execution.py`, and `python3 -m pytest -q tests/test_train_meta_transformer_synthetic.py tests/test_meta_transformer_runtime.py tests/test_semantic_runtime_learning.py tests/test_semantic_transformer_execution.py` passed.
- Blocked: the remaining gap is no longer the meta-transformer planning seam itself; it is the higher-order orchestrator/control-plane layer above it. `orchestration_plan` is still a deterministic bounded projection from the selected objective/backend/data mixes, and promotion still honestly depends on denser grounded runtime receipts before the helper should move beyond `shadow_candidate`.
- Next recommended task: move up to the higher-order orchestrator/economic-WM control surface that still hand-assembles homeostatic/queue/objective adjustments above the newly real selector + meta-transformer + orchestration sequence lanes, while keeping empirical promotion gates honest.

- Changed: completed the orchestration sequence-supervision pass. `src/orchestrator/orchestration_transformer.py` now emits bounded multi-step tool logits with an explicit PAD/stop label, runtime planning now honors model-predicted tool order before bounded heuristic backfill, and activation metadata records the model sequence trace instead of reconstructing later steps from a hard-coded preferred order alone.
- Changed: upgraded the orchestration trainer/eval contract to `bounded_tool_sequence_v2`. `src/orchestrator/training_dataset.py` now uses an explicit PAD label in target sequences, `scripts/train_orchestration_transformer.py` now trains/evaluates on full bounded tool sequences with full-sequence/active-token/stop-token metrics, and `scripts/eval_orchestration_transformer.py` now reports predicted tool sequences instead of only the first tool.
- Verification: `python3 -m compileall src/orchestrator/orchestration_transformer.py src/orchestrator/training_dataset.py scripts/train_orchestration_transformer.py scripts/eval_orchestration_transformer.py scripts/smoke_test_orchestrator.py scripts/analyze_orchestration_policy.py tests/test_train_orchestration_transformer.py tests/test_semantic_transformer_execution.py tests/test_semantic_runtime_scorers.py tests/test_semantic_runtime_learning.py -q`, `python3 -m ruff check src/orchestrator/orchestration_transformer.py src/orchestrator/training_dataset.py scripts/train_orchestration_transformer.py scripts/eval_orchestration_transformer.py scripts/smoke_test_orchestrator.py scripts/analyze_orchestration_policy.py tests/test_train_orchestration_transformer.py tests/test_semantic_transformer_execution.py tests/test_semantic_runtime_scorers.py tests/test_semantic_runtime_learning.py`, and `python3 -m pytest -q tests/test_train_orchestration_transformer.py tests/test_semantic_transformer_execution.py tests/test_semantic_runtime_scorers.py tests/test_semantic_runtime_learning.py` passed.
- Blocked: the orchestration lane now learns bounded sequences, but objective/backend/data-mix planning above it is still largely heuristic-prior logic in the meta-transformer path; the next gap is higher-order meta-choice learning, not missing sequence supervision.
- Next recommended task: move up to the meta-transformer planning layer and replace the remaining heuristic objective/backend/data-mix chooser with a bounded learned helper/runtime package path that can later condition on economic-WM and meta-node receipts.

- Changed: completed the semantic-selection meta-choice wiring pass. `src/orchestrator/semantic_policy.py` now supports a real bounded neural helper package shape for datapack selection, `src/orchestrator/datapack_selection_training.py` now trains and exports a one-hidden-layer feature MLP plus context-conditioned adjustment caps, and `scripts/train_datapack_selection_scorers.py` now records that contract honestly as `neural_feature_mlp_with_context_conditioned_adjustment_v2`.
- Changed: stopped trapping selector truth in run logs. `src/orchestrator/semantic_simulation.py` now persists per-episode `*_selection_summary_v1.json` sidecars into rollout artifacts, `src/replay/ingest.py` now carries those summaries into replay episodes, and `src/orchestrator/semantic_runtime_learning.py` now preserves selector traces into runtime rows and orchestration samples instead of dropping them before the training bridge.
- Changed: made the downstream conditioning path react to selector meta-choice. `src/orchestrator/semantic_transformer_bridge.py` now encodes selection-feedback features, while `src/orchestrator/orchestration_transformer.py` now records and conditions on selection policy/helper-status/meta-choice summaries rather than treating datapack choice as a past sidecar event with no model-visible effect.
- Verification: `python3 -m compileall src/orchestrator/semantic_policy.py src/orchestrator/datapack_selection_training.py src/orchestrator/semantic_transformer_bridge.py src/orchestrator/orchestration_transformer.py src/orchestrator/semantic_runtime_learning.py src/orchestrator/semantic_simulation.py src/replay/ingest.py scripts/train_datapack_selection_scorers.py tests/test_semantic_policy.py tests/test_datapack_selection_training.py tests/test_train_datapack_selection_scorers.py tests/test_semantic_runtime_learning.py tests/test_replay_dataset.py tests/test_semantic_simulation.py tests/test_semantic_transformer_execution.py -q`, `python3 -m ruff check src/orchestrator/semantic_policy.py src/orchestrator/datapack_selection_training.py src/orchestrator/semantic_transformer_bridge.py src/orchestrator/orchestration_transformer.py src/orchestrator/semantic_runtime_learning.py src/orchestrator/semantic_simulation.py src/replay/ingest.py scripts/train_datapack_selection_scorers.py tests/test_semantic_policy.py tests/test_datapack_selection_training.py tests/test_train_datapack_selection_scorers.py tests/test_semantic_runtime_learning.py tests/test_replay_dataset.py tests/test_semantic_simulation.py tests/test_semantic_transformer_execution.py`, `python3 -m pytest -q tests/test_semantic_policy.py tests/test_datapack_selection_training.py tests/test_train_datapack_selection_scorers.py tests/test_semantic_runtime_learning.py tests/test_replay_dataset.py tests/test_semantic_simulation.py`, and `python3 -m pytest -q tests/test_semantic_transformer_execution.py tests/test_train_orchestration_transformer.py tests/test_semantic_runtime_scorers.py` passed.
- Blocked: semantic selection is now a real neuralized helper lane with replay/runtime receipts, but the orchestration trainer itself is still honestly limited by `first_tool_only_v1` supervision and does not yet learn the full sequence contract.
- Next recommended task: upgrade `train_orchestration_transformer.py`, `eval_orchestration_transformer.py`, and `src/orchestrator/orchestration_transformer.py` from first-tool prediction to bounded multi-step sequence supervision using the now-preserved selector meta-choice traces.

- Changed: completed the gen2sim validity/value admission pass. `scripts/collect_local_synthetic_branches.py` now emits explicit `*_gen2sim_validity.json` branch-sidecar assessments, `src/training/synthetic_branch_corpus.py` now loads those assessments into corpus summaries/training policy, and `scripts/train_offline_with_local_synth.py` now persists that admission truth into the canonical runtime artifacts instead of leaving gen2sim as loose metadata.
- Changed: added the real learned helper substrate for gen2sim admission. `src/evidence/gen2sim_validity.py` now exposes an explicit feature contract plus bounded helper-trace blending, `src/evidence/gen2sim_validity_training.py` / `src/evidence/gen2sim_validity_runtime.py` provide the trained/runtime helper implementation, `scripts/train_gen2sim_validity.py` exports `gen2sim_validity_package.json` under `RegalTrainingRunner`, and `src/regal/data_value.py` now resolves the explicit assessment plus optional learned helper instead of multiplying by a raw scalar.
- Changed: kept the promotion story honest. The learned gen2sim helper is real and runtime-loadable, but its benchmark gate now requires empirical receipt density, so current local-corpus packages remain bounded `shadow_candidate` helpers rather than pretending to be promotion-ready off heuristic distillation alone.
- Verification: `python3 -m compileall src/evidence/gen2sim_validity.py src/evidence/gen2sim_validity_training.py src/evidence/gen2sim_validity_runtime.py src/training/synthetic_branch_corpus.py src/regal/data_value.py scripts/collect_local_synthetic_branches.py scripts/train_offline_with_local_synth.py scripts/train_gen2sim_validity.py tests/test_gen2sim_validity.py tests/test_synthetic_branch_corpus.py tests/test_train_offline_with_local_synth.py tests/test_train_gen2sim_validity.py tests/test_datapack_value_node_integration.py -q`, `python3 -m ruff check src/evidence/gen2sim_validity.py src/evidence/gen2sim_validity_training.py src/evidence/gen2sim_validity_runtime.py src/training/synthetic_branch_corpus.py src/regal/data_value.py scripts/collect_local_synthetic_branches.py scripts/train_offline_with_local_synth.py scripts/train_gen2sim_validity.py tests/test_gen2sim_validity.py tests/test_synthetic_branch_corpus.py tests/test_train_offline_with_local_synth.py tests/test_train_gen2sim_validity.py tests/test_datapack_value_node_integration.py`, and `python3 -m pytest -q tests/test_gen2sim_validity.py tests/test_synthetic_branch_corpus.py tests/test_train_offline_with_local_synth.py tests/test_train_gen2sim_validity.py tests/test_datapack_value_node_integration.py` passed.
- Blocked: promotion of the new gen2sim helper still honestly depends on empirical synth receipts and grounded runtime outcome density; local branch corpora alone are enough for a real helper substrate, not for automatic promotion.
- Next recommended task: move up one layer and deepen semantic/orchestration meta-choice supervision so datapack/scenario routing and multi-step orchestration learn on the newly recorded helper traces instead of leaving that level mostly hand-scored.

- Changed: completed the fill-path routing parity pass inside the coverage loop. `scripts/train_fill_path_policy.py` now emits canonical dataset/precondition/training/runtime-package artifacts under `RegalTrainingRunner`, while `src/world_model/fill_path_runtime.py` and `src/orchestrator/fill_path_routing.py` turn the learned fill-path model into a bounded runtime helper with explicit `disabled|auto|required` semantics.
- Changed: `src/orchestrator/coverage_loop.py` now uses the same honest promotion story for fill decisions that the sim agenda and diffusion gap prompts already use. Fill decisions now record `routing_policy`, helper promotion stage, heuristic vs learned score traces, and helper-conditioned rationale instead of silently switching to a raw `predict_batch()` hook when a checkpoint happens to be present.
- Changed: `CoverageLoopResult.record_outcomes(...)` now preserves fill-routing traces inside append-only fill-outcome records, so later economic-WM/orchestrator trainers can learn not just which fill method won but how the runtime chose it.
- Verification: `python3 -m compileall src/world_model/fill_path_policy.py src/world_model/fill_path_runtime.py src/orchestrator/fill_path_routing.py src/orchestrator/coverage_loop.py scripts/train_fill_path_policy.py scripts/run_coverage_loop.py tests/test_fill_path_policy.py tests/test_train_fill_path_policy.py -q`, `python3 -m ruff check src/world_model/fill_path_policy.py src/world_model/fill_path_runtime.py src/orchestrator/fill_path_routing.py src/orchestrator/coverage_loop.py scripts/train_fill_path_policy.py scripts/run_coverage_loop.py tests/test_fill_path_policy.py tests/test_train_fill_path_policy.py`, and `python3 -m pytest -q tests/test_fill_path_policy.py tests/test_train_fill_path_policy.py tests/test_coverage_loop.py` passed.
- Blocked: the remaining synth-side honesty gap is no longer fill-path routing itself; it is later gen2sim validity/value admission, which still needs the same package/promotion discipline.
- Next recommended task: wire gen2sim validity scoring and synth-value admission onto the same bounded helper contract used by agenda ranking and fill-path routing.

- Changed: completed the sim/gen2sim agenda wiring pass around the learned gap-ranker substrate. `scripts/train_gap_ranker.py` now emits canonical runtime/package artifacts and explicit benchmark/precondition gates, while `src/world_model/gap_ranker_runtime.py` plus `src/orchestrator/gap_agenda_ranking.py` now let the simulation agenda and gap-driven diffusion prompts consume that helper through bounded `disabled|auto|required` semantics.
- Changed: `src/orchestrator/semantic_simulation.py`, `src/orchestrator/diffusion_requests.py`, and `src/orchestrator/coverage_loop.py` now share one ranking contract for agenda selection. Agenda items and governed diffusion prompts now record `ranking_policy`, helper promotion stage, and score traces instead of silently inheriting a pure heuristic order even when a learned gap ranker exists.
- Changed: updated the heuristic inventory and training backlog notes so this lane is no longer described as “learned model exists off to the side.” The real remaining gap is narrower: extend the same helper maturity contract to the remaining coverage-loop decisions like fill-path routing and later gen2sim validity, not basic agenda ranking.
- Verification: `python3 -m compileall scripts/train_gap_ranker.py src/world_model/gap_ranker_runtime.py src/orchestrator/gap_agenda_ranking.py src/orchestrator/semantic_simulation.py src/orchestrator/diffusion_requests.py src/orchestrator/coverage_loop.py tests/test_train_gap_ranker.py tests/test_gap_agenda_ranking.py tests/test_coverage_compilation.py tests/test_coverage_loop.py -q`, `python3 -m ruff check scripts/train_gap_ranker.py src/world_model/gap_ranker_runtime.py src/orchestrator/gap_agenda_ranking.py src/orchestrator/semantic_simulation.py src/orchestrator/diffusion_requests.py src/orchestrator/coverage_loop.py tests/test_train_gap_ranker.py tests/test_gap_agenda_ranking.py tests/test_coverage_compilation.py tests/test_coverage_loop.py`, and `python3 -m pytest -q tests/test_train_gap_ranker.py tests/test_gap_agenda_ranking.py tests/test_coverage_compilation.py tests/test_coverage_loop.py` passed.
- Blocked: fill-path routing still uses the learned gap ranker more directly than the new agenda-helper contract, so the last consistency gap inside the coverage loop is now localized there rather than across the agenda itself.
- Next recommended task: push the same helper-package and shadow/promoted semantics into fill-path routing and then revisit broader economic-WM conditioning over the meta/orchestration layers.

- Changed: hardened the meta-transformer from “real trainer but heuristic runtime” into a bounded runtime helper lane. `scripts/train_meta_transformer_synthetic.py` now emits `meta_transformer_package.json`, applies materially stricter benchmark gating over runtime density/grounding/success counts, and `src/orchestrator/meta_transformer.py` can now load that package in `disabled|auto|required` modes through `src/orchestrator/meta_transformer_runtime.py`.
- Changed: the trained meta-transformer now affects runtime honestly but conservatively. In `auto`, benchmark-unready packages stay `shadow_candidate` and only exert bounded influence on authority, shared policy state, diffusion conditioning, and ontology tokens; in `required`, runtime now fails unless the package is benchmark-gated ready. `src/policies/meta_advisor.py` now accepts the same helper-package path/mode.
- Changed: updated the heuristic inventory and training notes so the remaining meta gap is no longer “does this lane exist?” but “how much of the higher-level meta control surface is still heuristic above the bounded learned helper.”
- Verification: `python3 -m compileall scripts/train_meta_transformer_synthetic.py src/orchestrator/meta_transformer.py src/orchestrator/meta_transformer_runtime.py src/orchestrator/meta_transformer_training.py src/policies/meta_advisor.py tests/test_train_meta_transformer_synthetic.py tests/test_meta_transformer_runtime.py tests/test_semantic_transformer_execution.py -q`, `python3 -m ruff check scripts/train_meta_transformer_synthetic.py src/orchestrator/meta_transformer.py src/orchestrator/meta_transformer_runtime.py src/orchestrator/meta_transformer_training.py src/policies/meta_advisor.py tests/test_train_meta_transformer_synthetic.py tests/test_meta_transformer_runtime.py tests/test_semantic_transformer_execution.py`, and `python3 -m pytest -q tests/test_train_meta_transformer_synthetic.py tests/test_meta_transformer_runtime.py tests/test_semantic_transformer_execution.py` passed.
- Blocked: the meta-transformer now has real training-plus-runtime package plumbing, but objective-preset/data-mix/backend planning still remain heuristic-prior outputs; the higher meta-node/economic-WM conditioning stack above that helper path is still future work.
- Next recommended task: apply the same explicit helper-package and promotion-stage discipline to the sim/gen2sim agenda in `semantic_simulation.py` / `diffusion_requests.py`.

- Changed: completed the orchestration-transformer parity pass. `scripts/train_orchestration_transformer.py` now prefers semantic-runtime exports, emits canonical runtime artifacts/checkpoint-registry outputs under `RegalTrainingRunner`, and benchmark-gates the lane on real runtime-corpus density instead of looking production-ready just because a wrapper exists.
- Changed: removed the last dummy-token drift in the orchestration lane. `src/orchestrator/training_dataset.py` now persists deterministic instruction text/tokens through save-load, `src/orchestrator/semantic_runtime_learning.py` preserves runtime instruction metadata in exported orchestration samples, and `scripts/eval_orchestration_transformer.py` now uses the same token contract instead of random placeholder tokens.
- Changed: updated the heuristic inventory and training backlog notes so the remaining orchestration limitation is explicit: the lane is now runtime-backed and honest, but the current supervision contract is still `first_tool_only_v1` rather than a full sequence learner.
- Verification: `python3 -m compileall scripts/train_orchestration_transformer.py scripts/eval_orchestration_transformer.py src/orchestrator/training_dataset.py src/orchestrator/semantic_runtime_learning.py tests/test_train_orchestration_transformer.py -q`, `python3 -m ruff check scripts/train_orchestration_transformer.py scripts/eval_orchestration_transformer.py src/orchestrator/training_dataset.py src/orchestrator/semantic_runtime_learning.py tests/test_train_orchestration_transformer.py`, and `python3 -m pytest -q tests/test_train_orchestration_transformer.py` passed.
- Blocked: orchestration is no longer blocked on dummy instructions or missing runtime-corpus plumbing; the remaining honest gap is richer sequence supervision and later packet/event-native labels beyond first-tool prediction.
- Next recommended task: harden meta-transformer promotion/readiness semantics so the existing architecture/runtime/training lane is treated as real but only promoted when the runtime corpus is materially dense and grounded.

- Changed: completed the semantic datapack-selection training/export tranche. Added `src/orchestrator/datapack_selection_training.py` plus `scripts/train_datapack_selection_scorers.py`, so `selection_summary` run-log receipts now compile into a real scorer package with dataset summaries, execution preconditions, training summaries, canonical runtime manifests, and checkpoint-registry entries instead of stopping at a runtime-only helper seam.
- Changed: tightened selector neuralization and promotion semantics at the runtime boundary. `src/orchestrator/semantic_policy.py` now exposes `DatapackSelectionContext` and conditions helper strength on candidate-pool/gap/readiness/history context instead of a flat literal cap, while `src/orchestrator/semantic_simulation.py` now distinguishes `shadow_candidate` vs `promoted` helper stages and refuses `selection_scorer_mode='required'` unless the scorer package is benchmark-gated ready.
- Changed: updated `docs/economic_world_model/heuristic_advisory_sidecar_inventory.md` and `scripts/TRAINING_MIGRATION_BACKLOG.json` to mark the datapack-selection helper lane as migrated while keeping the next honest gaps on orchestration-transformer supervision and stricter meta-transformer promotion/readiness.
- Verification: `python3 -m compileall src/orchestrator/semantic_policy.py src/orchestrator/semantic_simulation.py src/orchestrator/datapack_selection_training.py scripts/train_datapack_selection_scorers.py tests/test_datapack_selection_training.py tests/test_train_datapack_selection_scorers.py tests/test_semantic_policy.py tests/test_semantic_simulation.py -q`, `python3 -m ruff check src/orchestrator/semantic_policy.py src/orchestrator/semantic_simulation.py src/orchestrator/datapack_selection_training.py scripts/train_datapack_selection_scorers.py tests/test_datapack_selection_training.py tests/test_train_datapack_selection_scorers.py tests/test_semantic_policy.py tests/test_semantic_simulation.py`, and `python3 -m pytest -q tests/test_datapack_selection_training.py tests/test_train_datapack_selection_scorers.py tests/test_semantic_policy.py tests/test_semantic_simulation.py` passed.
- Blocked: this lands the learned helper lane, but full counterfactual supervision for datapack choice is still sparse; the scorer is honest about that via benchmark gates and shadow-stage clamping rather than pretending the selector is fully learned already.
- Next recommended task: keep moving down the remaining top follow-ons: stricter meta-transformer promotion/readiness, then learned-helper discipline for sim/gen2sim agenda selection.

- Changed: completed the remaining observation-adapter / runtime-backbone vision-truth bridge. `src/semantic/runtime_backbone.py` now synthesizes compact semantic-runtime truth, benchmark signals, and execution-precondition summaries from the semantic world model, while `src/observation/adapter.py` and `src/observation/condition_vector_builder.py` now preserve and react to those signals instead of dropping them as inert metadata.
- Changed: condition vectors now treat benchmark-unready grounding, failed execution preconditions, and blocked semantic fusion as bounded OOD/recovery signals. This means sidecar-carried runtime truth now influences `ood_risk_level` and `recovery_priority` in the actual conditioning path rather than staying JSON-only.
- Changed: updated `docs/economic_world_model/heuristic_advisory_sidecar_inventory.md` and `scripts/RUNTIME_WIRING_BACKLOG.json` so the remaining follow-ons shift from runtime wiring toward training/export work: datapack-selection scorers, orchestration-transformer supervision, and stricter meta-transformer promotion gating.
- Verification: `python3 -m compileall src/observation/adapter.py src/observation/condition_vector_builder.py src/semantic/runtime_backbone.py tests/test_observation_semantic_truth.py tests/test_semantic_world_model_backbone.py -q`, `python3 -m ruff check src/observation/adapter.py src/observation/condition_vector_builder.py src/semantic/runtime_backbone.py tests/test_observation_semantic_truth.py tests/test_semantic_world_model_backbone.py`, and `python3 -m pytest -q tests/test_observation_semantic_truth.py tests/test_semantic_world_model_backbone.py tests/test_semantic_transformer_execution.py` passed.
- Blocked: the next bottleneck is no longer runtime truth propagation in these modules; it is the absence of learned/helper training lanes that consume the now-richer selection and conditioning receipts.
- Next recommended task: implement `train_datapack_selection_scorers.py` so the existing `selection_scorer_mode=disabled|auto|required` runtime promotion path can move beyond heuristic-plus-empty-hook operation.

- Changed: wired the rollout-labeler / semantic-fusion vision sidecars into the actual datapack contract. `src/motor_backend/datapacks.py` and `src/ontology/datapack_registry.py` now preserve datapack metadata plus quality/novelty through YAML and ontology upserts, `src/vla/rollout_labeler.py` now aggregates teacher-runtime / SceneTracks / benchmark / execution-precondition truth into derived VLA-labeled datapacks, and `src/orchestrator/semantic_simulation.py` now enriches those datapacks again with semantic-fusion artifacts and readiness instead of dropping fusion outputs after labeling.
- Changed: this closes the main “thin datapack after rich vision sidecars” gap. Later semantic selection can now see real labeled-datapack readiness/provenance fields rather than only tags/description, while unavailable teacher or non-real SceneTracks remain explicit non-ready states instead of silently disappearing.
- Changed: updated `docs/economic_world_model/heuristic_advisory_sidecar_inventory.md` and `scripts/RUNTIME_WIRING_BACKLOG.json` to mark the rollout-labeler/datapack truth contract as landed and narrow the remaining vision-side runtime backlog to observation-adapter/runtime-backbone bridges.
- Verification: `python3 -m compileall src/motor_backend/datapacks.py src/ontology/datapack_registry.py src/vla/rollout_labeler.py src/orchestrator/semantic_simulation.py tests/test_datapack_loader.py tests/test_datapack_registry.py tests/test_rollout_labeler.py tests/test_semantic_simulation.py tests/test_semantic_simulation_e2e.py -q`, `python3 -m ruff check src/motor_backend/datapacks.py src/ontology/datapack_registry.py src/vla/rollout_labeler.py src/orchestrator/semantic_simulation.py tests/test_datapack_loader.py tests/test_datapack_registry.py tests/test_rollout_labeler.py tests/test_semantic_simulation.py tests/test_semantic_simulation_e2e.py`, and `python3 -m pytest -q tests/test_datapack_loader.py tests/test_datapack_registry.py tests/test_rollout_labeler.py tests/test_semantic_simulation.py tests/test_semantic_simulation_e2e.py` passed.
- Blocked: the remaining vision-side honesty gap is no longer the labeled datapack object itself; it is the smaller set of observation-adapter and runtime-backbone bridges that still need the same sidecar-to-runtime truth promotion.
- Next recommended task: audit `src/observation/adapter.py` and `src/semantic/runtime_backbone.py` for any remaining sidecar-only grounding/quality semantics, then connect that work to the datapack-selection scorer corpus/export lane from `scripts/TRAINING_MIGRATION_BACKLOG.json`.

- Changed: made the shadow-advisory scorer fallback explicit in runtime artifacts. `src/orchestrator/shadow_advisory.py` now emits `semantic_runtime_scorer_preconditions` plus blocking `semantic_runtime_scorer_work_orders` whenever no semantic-runtime scorer package is available, rather than silently falling back to the heuristic branch with no outward trace.
- Changed: the main advisory/training consumers now persist those artifacts into real outputs. `scripts/run_shadow_advisory_pass.py`, `scripts/train_shadow_replay_policy.py`, `scripts/train_shadow_offline_rl.py`, `scripts/train_shadow_pricing_models.py`, and `scripts/train_sac_with_ontology_logging.py` now write/register scorer-precondition and scorer-work-order artifacts so scored vs unscored shadow runs are distinguishable in manifests and backlog scans.
- Changed: updated `scripts/RUNTIME_WIRING_BACKLOG.json` to mark the shadow-advisory precondition tranche complete and open the next honest runtime backlog item around remaining vision-side sidecar semantics.
- Verification: `python3 -m compileall src/orchestrator/shadow_advisory.py scripts/run_shadow_advisory_pass.py scripts/train_shadow_replay_policy.py scripts/train_shadow_offline_rl.py scripts/train_shadow_pricing_models.py scripts/train_sac_with_ontology_logging.py tests/test_receipt_ingest.py tests/test_shadow_advisory_pass.py -q`, `python3 -m ruff check src/orchestrator/shadow_advisory.py scripts/run_shadow_advisory_pass.py scripts/train_shadow_replay_policy.py scripts/train_shadow_offline_rl.py scripts/train_shadow_pricing_models.py scripts/train_sac_with_ontology_logging.py tests/test_receipt_ingest.py tests/test_shadow_advisory_pass.py`, and `python3 -m pytest -q tests/test_receipt_ingest.py tests/test_shadow_advisory_pass.py tests/test_econ_regal_sampling.py tests/test_queue_dispatch_integration.py tests/test_online_queue_dispatch_integration.py` passed.
- Blocked: the fallback is now honest, but the next missing piece is still learned-package density and auto-export policy, not the absence of runtime artifact wiring.
- Next recommended task: audit the remaining vision-side sidecars across observation adapters, rollout labeling, and runtime-backbone bridges so density/quality signals either affect routing/preconditions materially or stay explicitly quarantined.

- Changed: replaced the `scripts/train_meta_transformer_synthetic.py` random-noise placeholder with a real trainer over the existing meta-transformer substrate. The script now consumes `meta_transformer_runtime_dataset.json` exports or saved dataset JSONs, instantiates the real `MetaTransformerNet`, uses the existing batching/loss/eval helpers, and emits canonical runtime manifests/checkpoint registry/training summaries.
- Changed: synthetic generation is still available, but only as an explicit fallback corpus source. The trainer now records dataset provenance, benchmark-gate and execution-precondition artifacts, and `meta_transformer_sample_projection` trajectory audits so a synthetic dev run cannot be mistaken for a serious runtime-corpus promotion lane.
- Changed: updated `docs/economic_world_model/heuristic_advisory_sidecar_inventory.md` and `scripts/TRAINING_MIGRATION_BACKLOG.json` to mark the meta-transformer trainer migrated while leaving meta-transformer promotion/readiness density as an honest remaining follow-on rather than pretending the lane is now benchmark-ready by default.
- Verification: `python3 -m compileall scripts/train_meta_transformer_synthetic.py tests/test_train_meta_transformer_synthetic.py -q`, `python3 -m ruff check scripts/train_meta_transformer_synthetic.py tests/test_train_meta_transformer_synthetic.py`, and `python3 -m pytest -q tests/test_train_meta_transformer_synthetic.py` passed.
- Blocked: the trainer is now real, but serious promotion still depends on a materially non-synthetic runtime corpus; script parity alone is not enough.
- Next recommended task: make the shadow-advisory scorer fallback explicit in runtime artifacts/work orders, then add the actual training/export lane for the semantic datapack-selection helper.

- Changed: tightened the semantic datapack-selection lane into an explicit promotion path instead of a forever-optional helper hook. `src/orchestrator/semantic_policy.py` now exposes first-class `DatapackSelectionFeatures` plus a bounded `DatapackSelectionScorerPackage`, so the hand-set prior terms become an explicit feature contract and trainable reranking seam rather than a hidden permanent policy.
- Changed: `src/orchestrator/semantic_simulation.py` now carries `selection_scorer_mode=disabled|auto|required`, resolves learned helper packages through a small canonical search path when not passed explicitly, and records helper status into `selection_summary` so the runtime tells the truth about whether selection ran with a learned helper, fell back heuristically, or required a missing scorer package.
- Changed: updated `docs/economic_world_model/heuristic_advisory_sidecar_inventory.md` to describe the new `disabled -> auto -> required` promotion model and to make the remaining missing piece explicit: the trainer/export path for the semantic datapack-selection helper itself.
- Verification: `python3 -m compileall src/orchestrator/semantic_policy.py src/orchestrator/semantic_simulation.py tests/test_semantic_policy.py tests/test_semantic_simulation.py tests/test_semantic_simulation_e2e.py -q`, `python3 -m ruff check src/orchestrator/semantic_policy.py src/orchestrator/semantic_simulation.py tests/test_semantic_policy.py tests/test_semantic_simulation.py tests/test_semantic_simulation_e2e.py`, and `python3 -m pytest -q tests/test_semantic_policy.py tests/test_semantic_simulation.py tests/test_semantic_simulation_e2e.py` passed.
- Blocked: the learned helper runtime seam is now real, but the actual training/export job for datapack-selection helpers still needs to be built and benchmarked before `required` should become the default production mode.
- Next recommended task: replace `scripts/train_meta_transformer_synthetic.py` with the actual meta-transformer runtime dataset/training path, then add the dedicated training/export lane for semantic datapack-selection helper packages.

- Changed: migrated `scripts/train_vla_recap_offline.py` into the canonical runtime envelope without breaking the existing direct `train_offline(...)` API used by the smoke and inference scripts. The trainer now emits recap dataset summaries, feature-config artifacts, execution-precondition and benchmark-gate reports, training summaries, training-job receipts, and explicit latest/best checkpoints while preserving the checkpoint fields expected by `src/vla/recap_inference.py`.
- Changed: the CLI RECAP path now runs under `RegalTrainingRunner`, registers recap artifacts and checkpoints in the unified manifest/checkpoint registry, and projects per-episode recap rows into explicit `recap_row_projection` trajectory audits so the lane is no longer a silent lightweight bypass around the runtime/training contract.
- Changed: updated `docs/economic_world_model/heuristic_advisory_sidecar_inventory.md` and `scripts/TRAINING_MIGRATION_BACKLOG.json` to mark RECAP offline training as wired now and benchmark-gated rather than still pending as a lightweight-only lane.
- Verification: `python3 -m compileall scripts/train_vla_recap_offline.py tests/test_train_vla_recap_offline.py -q`, `python3 -m ruff check scripts/train_vla_recap_offline.py tests/test_train_vla_recap_offline.py`, and `python3 -m pytest -q tests/test_train_vla_recap_offline.py` passed.
- Blocked: this makes the RECAP trainer contractually honest, but it does not create a real production recap corpus; the benchmark gate remains conservative and small local recap datasets still stay non-promotion-ready.
- Next recommended task: replace the `train_meta_transformer_synthetic.py` random-data placeholder with the real meta-transformer runtime dataset/training substrate that already exists elsewhere in the repo.

- Changed: completed the semantic datapack/scenario selection wiring tranche. `src/orchestrator/semantic_policy.py` now ranks datapacks with bounded runtime-facing signals instead of only tag overlap: historical scenario outcomes (ARH-adjusted), candidate quality/novelty, benchmark/readiness support, and explicit gap-fill pressure all contribute to one scored selection report.
- Changed: `src/orchestrator/semantic_simulation.py` now consumes that ranking directly, merges ontology and local-YAML fallback candidates instead of abruptly replacing one with the other, and records a `selection_summary` into both the live simulation result and the semantic run log so datapack admission is no longer opaque.
- Changed: updated `docs/economic_world_model/heuristic_advisory_sidecar_inventory.md` and `scripts/RUNTIME_WIRING_BACKLOG.json` to mark semantic datapack/scenario selection as wired now while keeping the remaining shadow-advisory scorer-coverage fallback honest as the active runtime backlog item.
- Verification: `python3 -m compileall src/orchestrator/semantic_policy.py src/orchestrator/semantic_simulation.py tests/test_semantic_policy.py tests/test_semantic_simulation.py tests/test_semantic_simulation_e2e.py -q`, `python3 -m ruff check src/orchestrator/semantic_policy.py src/orchestrator/semantic_simulation.py tests/test_semantic_policy.py tests/test_semantic_simulation.py tests/test_semantic_simulation_e2e.py`, and `python3 -m pytest -q tests/test_semantic_policy.py tests/test_semantic_simulation.py tests/test_semantic_simulation_e2e.py` passed.
- Blocked: the selection lane is still deterministic and bounded rather than learned. That is now an explicit later neuralization task, not a hidden tag-match-only runtime path.
- Next recommended task: migrate `scripts/train_vla_recap_offline.py` into the canonical training runtime so the RECAP lane stops sitting outside manifests/checkpoint registries/receipt-aware training artifacts.

- Changed: completed the broader SceneTracks/SAM3D truth-consumer sweep that remained after the earlier replay/bootstrap fix. `src/evidence/scene_tracks_truth.py` now resolves backend identity from nested runner metadata instead of inferring `real` from any sidecar presence, while `scripts/run_stage1_pipeline.py`, `scripts/collect_local_synthetic_branches.py`, `src/training/synthetic_branch_corpus.py`, `src/orchestrator/semantic_runtime_scorers.py`, and `src/orchestrator/semantic_fusion_runner.py` now all consume that same truth helper so passthrough/auto lanes stop resurfacing as `scene_tracks_non_stub` in Stage 1, local synth metadata, live scorer summaries, or degraded-fusion artifacts.
- Changed: made the GPU-plus-SAM3D requirement executable instead of just documentary. Added `src/evidence/grounded_data_host.py`; `src/vision/scene_ir_tracker/io/scene_tracks_runner.py` now emits `grounded_data_host_capabilities` and `grounded_data_host_preconditions`, and `src/orchestrator/loop_run_backlog.py` now reuses the same host-capability scan so local/runtime assessment and the recurring loop backlog share one real-SAM3D readiness story.
- Changed: updated `docs/economic_world_model/heuristic_advisory_sidecar_inventory.md` and `scripts/RUNTIME_WIRING_BACKLOG.json` to mark the remaining SceneTracks truth-consumer sweep as completed. The main runtime heuristic lane left in this backlog is still semantic datapack/scenario selection.
- Verification: `python3 -m compileall src/evidence/scene_tracks_truth.py src/evidence/grounded_data_host.py src/orchestrator/semantic_runtime_scorers.py src/orchestrator/semantic_fusion_runner.py src/orchestrator/loop_run_backlog.py src/training/synthetic_branch_corpus.py src/vision/scene_ir_tracker/io/scene_tracks_runner.py scripts/run_stage1_pipeline.py scripts/collect_local_synthetic_branches.py tests/test_scene_tracks_truth.py tests/test_grounded_data_host.py tests/test_synthetic_branch_corpus.py tests/test_semantic_runtime_scorers.py -q`, `python3 -m ruff check src/evidence/scene_tracks_truth.py src/evidence/grounded_data_host.py src/orchestrator/semantic_runtime_scorers.py src/orchestrator/semantic_fusion_runner.py src/orchestrator/loop_run_backlog.py src/training/synthetic_branch_corpus.py src/vision/scene_ir_tracker/io/scene_tracks_runner.py scripts/run_stage1_pipeline.py scripts/collect_local_synthetic_branches.py tests/test_scene_tracks_truth.py tests/test_grounded_data_host.py tests/test_synthetic_branch_corpus.py tests/test_semantic_runtime_scorers.py`, `python3 -m pytest -q tests/test_scene_tracks_truth.py tests/test_grounded_data_host.py tests/test_synthetic_branch_corpus.py tests/test_semantic_runtime_scorers.py tests/test_stage1_pipeline_governed.py tests/test_loop_run_backlog.py`, and `git diff --check` passed.
- Blocked: real grounded-data rows are still host-blocked here because this workspace does not have the actual Linux/NVIDIA + SAM3D setup. The new host-precondition artifact makes that limitation explicit in runner metadata instead of letting downstream consumers infer from sidecar existence alone.
- Next recommended task: replace the tag-overlap datapack/scenario policy in `src/orchestrator/semantic_policy.py` with bounded evidence/runtime-aware scoring and thread that selection rationale into `src/orchestrator/semantic_simulation.py`.

- Changed: completed the bootstrap workcell honesty tranche. `scripts/bootstrap_semantic_workcell_loop.py` now emits canonical per-episode `*_runtime_packet_v1.json`, `*_event_spine_v1.json`, and `*_decision_ledger_v1.json` artifacts, records `runtime_packet_id` / `event_refs` / `decision_refs` into `metadata.json`, and distinguishes trace-complete replay rows from `grounded_data_ready` rows that actually require real SAM3D plus GPU-backed execution.
- Changed: tightened rollout replay import to honor those bootstrap artifacts directly. `src/replay/ingest.py` now discovers runtime/event/decision refs from rollout metadata or sidecar filenames, carries `runtime_packet_id` / `event_refs` / `decision_refs` into replay metadata, and lets the bootstrap runtime corpus stop failing readiness solely because the canonical refs were missing.
- Changed: fixed the workcell coverage-graph contract instead of just inflating bootstrap summaries. `src/world_model/coverage_evidence_harvester.py` now canonicalizes env ids like `workcell_env`, emits canonical skill ids aligned with the graph, and maps `peg_in_hole` affordance evidence into a built-in workcell skill chain added in `src/hrl/skill_graph.py`; `src/orchestrator/coverage_loop.py` now enables that chain automatically for workcell envs.
- Changed: updated `docs/economic_world_model/heuristic_advisory_sidecar_inventory.md` and `scripts/RUNTIME_WIRING_BACKLOG.json` to mark bootstrap trace completeness and workcell coverage-graph mapping as wired now, leaving shadow advisory sampling and semantic policy selection as the main remaining runtime heuristic lanes.
- Verification: `python3 -m compileall src/hrl/skill_graph.py src/world_model/coverage_evidence_harvester.py src/orchestrator/coverage_loop.py src/replay/ingest.py scripts/bootstrap_semantic_workcell_loop.py tests/test_skill_graph.py tests/test_coverage_evidence_harvester.py tests/test_replay_dataset.py tests/integration/test_bootstrap_semantic_workcell_loop.py -q`, `python3 -m ruff check src/hrl/skill_graph.py src/world_model/coverage_evidence_harvester.py src/orchestrator/coverage_loop.py src/replay/ingest.py scripts/bootstrap_semantic_workcell_loop.py tests/test_skill_graph.py tests/test_coverage_evidence_harvester.py tests/test_replay_dataset.py tests/integration/test_bootstrap_semantic_workcell_loop.py`, `python3 -m pytest -q tests/test_skill_graph.py tests/test_coverage_evidence_harvester.py tests/test_replay_dataset.py tests/integration/test_bootstrap_semantic_workcell_loop.py`, and `python3 -m pytest -q tests/test_coverage_loop.py tests/test_semantic_coverage_graph.py tests/test_gap_ranker.py` passed.
- Blocked: the current workspace still does not have a real SAM3D host/checkpoint setup, so bootstrap runs can now tell the truth about grounded-data readiness and trace completeness, but they still cannot claim real grounded data locally.
- Next recommended task: wire learned semantic/runtime scorer outputs into `src/orchestrator/shadow_advisory.py`, `src/rl/econ_regal_sampling.py`, and `src/orchestrator/queue_selection.py` so bounded replay selection stops relying mainly on rule-weighted heuristics.

- Changed: completed that shadow-advisory sampling tranche. `src/orchestrator/shadow_advisory.py` now auto-loads semantic runtime scorer packages when present, scores replay-native semantic runtime rows before advisory emission, and passes bounded learned route/regret/counterfactual/authority signals into `src/rl/econ_regal_sampling.py`.
- Changed: `src/orchestrator/queue_selection.py` now preserves `semantic_runtime_score` in live queue metadata, so the bounded replay reweighting lane can carry learned-support evidence all the way through the actual queue-selection shim instead of collapsing back to rule-only metadata.
- Changed: updated `docs/economic_world_model/heuristic_advisory_sidecar_inventory.md` and `scripts/RUNTIME_WIRING_BACKLOG.json` to mark shadow advisory sampling as wired now, leaving semantic policy datapack/scenario selection as the main remaining runtime heuristic lane in this backlog.
- Verification: `python3 -m compileall src/orchestrator/shadow_advisory.py src/rl/econ_regal_sampling.py src/orchestrator/queue_selection.py scripts/run_shadow_advisory_pass.py tests/test_econ_regal_sampling.py tests/test_receipt_ingest.py -q`, `python3 -m ruff check src/orchestrator/shadow_advisory.py src/rl/econ_regal_sampling.py src/orchestrator/queue_selection.py scripts/run_shadow_advisory_pass.py tests/test_econ_regal_sampling.py tests/test_receipt_ingest.py`, `python3 -m pytest -q tests/test_econ_regal_sampling.py tests/test_receipt_ingest.py tests/test_shadow_advisory_pass.py`, and `python3 -m pytest -q tests/test_queue_dispatch_integration.py tests/test_online_queue_dispatch_integration.py` passed.
- Blocked: the fallback path still remains active when no semantic runtime scorer package exists beside the replay dataset, so the remaining limitation here is scorer coverage/package availability rather than missing queue wiring.
- Next recommended task: tackle `src/orchestrator/semantic_policy.py` and adjacent sim/datapack selection code so scenario/datapack routing stops depending mainly on tag overlap and ARH penalties.

- Changed: reclassified the canonical workcell refresh/replay lane as a real-SAM3D run rather than an A100-optional corpus sweep. `docs/economic_world_model/full_stack_training_backlog.md` and `scripts/runpod/FULL_STACK_TRAINING_BUNDLES.json` now treat `workcell_data_refresh` as the first recurring Runpod job specifically because it should run with `--backend-policy real` on a Linux/NVIDIA A100-class host; local passthrough refreshes remain dev-only.
- Changed: tightened Runpod readiness around grounding truth instead of raw corpus counts alone. `scripts/runpod/full_stack_training.py` now discovers `bootstrap_summary.json` files, tracks real-grounded replay episodes/steps separately from generic replay counts, and lets bundle readiness require real-grounded replay density before semantic-runtime training is considered honestly ready.
- Verification: `python3 -m compileall scripts/runpod -q` and `python3 scripts/runpod/assess_full_stack_training.py --bundle auto` passed.
- Blocked: the current workspace still lacks installed SAM3D repos/checkpoints, so the canonical `workcell_data_refresh` bundle is correctly not locally ready even though passthrough bootstrap runs exist.
- Next recommended task: layer the same host/Hugging Face/SAM3D truth into the loop-run backlog so the real-grounding bring-up path and the recurring training path share one consistent readiness story.

- Changed: completed the first serious heuristic/advisory/sidecar purge tranche for the local synthetic training lane. Added `src/training/synthetic_branch_corpus.py` to load synthetic-branch NPZ corpora plus metadata/gap-label sidecars, emit execution-precondition and benchmark-gate artifacts, and compile bounded training policy directly from corpus truth.
- Changed: upgraded `scripts/train_offline_with_local_synth.py` into a canonical runtime-emitting trainer. It now loads explicit branch-corpus sidecars, caps effective synthetic share when metadata/gap labels/non-heuristic grounding are missing, multiplies branch influence by gap/value/readiness signals, and emits `RegalTrainingRunner` runtime artifacts plus canonical checkpoints when not run with `--skip-regal-runner`.
- Changed: upgraded `scripts/collect_local_synthetic_branches.py` to emit explicit seed-runtime provenance (`scene_tracks_backend`, `teacher_runtime_backend_selected`, `vision_backbone_selected`, `semantic_grounding_mode`, `semantic_memory_grounded`) together with `future_training_signals` and `future_training_artifacts`, so local synthetic corpora stop pretending they are self-describing when they are not.
- Changed: added `docs/economic_world_model/heuristic_advisory_sidecar_inventory.md` with a ranked code-grounded gap matrix covering local synth, Stage-1 diffusion, replay ingest truthiness, queue selection, lightweight trainers, semantic policy selection, rollout labeling, and SceneTracks fallback lanes.
- Changed: created `scripts/RUNTIME_WIRING_BACKLOG.json` for remaining non-training runtime gaps and updated `scripts/TRAINING_MIGRATION_BACKLOG.json` to mark `train_offline_with_local_synth.py` migrated while leaving the RECAP/meta-transformer lightweight lanes honestly pending.
- Verification: `python3 -m compileall src/training/synthetic_branch_corpus.py scripts/train_offline_with_local_synth.py scripts/collect_local_synthetic_branches.py tests/test_synthetic_branch_corpus.py tests/test_train_offline_with_local_synth.py -q`, `python3 -m ruff check src/training/synthetic_branch_corpus.py scripts/train_offline_with_local_synth.py scripts/collect_local_synthetic_branches.py tests/test_synthetic_branch_corpus.py tests/test_train_offline_with_local_synth.py`, and `python3 -m pytest -q tests/test_synthetic_branch_corpus.py tests/test_train_offline_with_local_synth.py` passed.
- Blocked: Stage-1 semantic/diffusion routing is still partly keyword/rule-driven, passthrough SceneTracks still overstates truth in a few early metadata paths, and `train_vla_recap_offline.py` / `train_meta_transformer_synthetic.py` still remain outside heavyweight parity.
- Next recommended task: tighten Stage-1 semantic/diffusion routing and replay/bootstrap SceneTracks truth semantics so governed video, import/readiness, and synthetic-branch admission all agree on what counts as real grounding.

- Changed: completed the next runtime wiring tranche for Stage-1 semantic/diffusion routing. `src/orchestrator/diffusion_requests.py` now compiles governed hypotheses and routing context from guidance and coverage-gap prompts, `src/diffusion/real_video_diffusion_stub.py` reranks governed hypotheses before any fallback lane, and `scripts/run_stage1_pipeline.py` now carries routing/benchmark status into proposal admission and datapack creation.
- Changed: Stage-1 benchmark status is no longer implicit. The pipeline now emits per-video benchmark-gate sidecars, records benchmark status in admission rows and datapack metrics, downgrades heuristic/unbenchmarked proposals into `shadow_stage1_datapack` work orders, and caps their datapack tier/trust instead of letting them look benchmark-ready by default.
- Changed: the simulated/default Stage-1 reference lane now declares `scene_tracks_backend=unavailable`, `vision_backbone_selected=unavailable`, and `semantic_grounding_mode=heuristic_fallback` explicitly, while a manifest with real SceneTracks plus a real vision-backbone declaration now passes the benchmark gate and stays on the benchmark-ready admission path.
- Changed: updated `docs/economic_world_model/heuristic_advisory_sidecar_inventory.md` and `scripts/RUNTIME_WIRING_BACKLOG.json` to mark Stage-1 routing as wired now and leave the remaining runtime backlog focused on SceneTracks passthrough truthiness, shadow advisory learning, and semantic policy routing.
- Verification: `python3 -m compileall src/orchestrator/diffusion_requests.py src/diffusion/real_video_diffusion_stub.py scripts/run_stage1_pipeline.py tests/test_stage1_pipeline_governed.py tests/test_diffusion_prompt_includes_constraints.py tests/test_video_diffusion_stub_routing.py -q`, `python3 -m ruff check src/orchestrator/diffusion_requests.py src/diffusion/real_video_diffusion_stub.py scripts/run_stage1_pipeline.py tests/test_stage1_pipeline_governed.py tests/test_diffusion_prompt_includes_constraints.py tests/test_video_diffusion_stub_routing.py`, and `python3 -m pytest -q tests/test_coverage_compilation.py tests/test_stage1_pipeline_governed.py tests/test_diffusion_prompt_includes_constraints.py tests/test_video_diffusion_stub_routing.py` passed.
- Blocked: Stage-1 is now bounded, but seed-tag extraction is still deterministic bootstrap logic; the next live distortion is the permissive passthrough-as-non-stub truth path in replay/bootstrap metadata.
- Next recommended task: tighten `src/replay/ingest.py` and `scripts/bootstrap_semantic_workcell_loop.py` so passthrough SceneTracks never appear equivalent to real grounded SceneTracks anywhere upstream of benchmark gating.

- Changed: completed the replay/bootstrap SceneTracks truthiness tranche. Added `src/evidence/scene_tracks_truth.py` and routed both `src/replay/ingest.py` and `scripts/bootstrap_semantic_workcell_loop.py` through the same normalization logic so `passthrough`, `stub`, and `auto` no longer count as `scene_tracks_non_stub`, `semantic_grounding_ready`, or `semantic_grounding_non_heuristic`.
- Changed: replay ingest now preserves backend identity and density signals without inflating grounding truth. Rollout bundles with passthrough SceneTracks can still surface semantic density and grounded-world-model side information, but upstream replay metadata no longer claims they are non-stub or non-heuristic just because a passthrough backend or old explicit flag was present.
- Changed: bootstrap semantic workcell metadata now writes fallback truth honestly. Only real SceneTracks keep `scene_tracks_non_stub` and `scene_tracks_training_eligible`; passthrough runs stay explicit fallback lanes in `metadata.json` and downstream semantic runtime metadata.
- Changed: updated `docs/economic_world_model/heuristic_advisory_sidecar_inventory.md` and `scripts/RUNTIME_WIRING_BACKLOG.json` to mark SceneTracks passthrough truthiness as wired now, leaving shadow advisory learning and semantic policy routing as the main remaining runtime backlog items.
- Verification: `python3 -m compileall src/evidence/scene_tracks_truth.py src/evidence/__init__.py src/replay/ingest.py scripts/bootstrap_semantic_workcell_loop.py tests/test_scene_tracks_truth.py tests/test_replay_dataset.py tests/integration/test_bootstrap_semantic_workcell_loop.py -q`, `python3 -m ruff check src/evidence/scene_tracks_truth.py src/evidence/__init__.py src/replay/ingest.py scripts/bootstrap_semantic_workcell_loop.py tests/test_scene_tracks_truth.py tests/test_replay_dataset.py tests/integration/test_bootstrap_semantic_workcell_loop.py`, and `python3 -m pytest -q tests/test_scene_tracks_truth.py tests/test_replay_dataset.py tests/integration/test_bootstrap_semantic_workcell_loop.py` passed.
- Blocked: the next highest-impact runtime distortion is the live but rule-based shadow advisory replay/queue scorer; it now stands out more clearly because the Stage-1 and SceneTracks truth layers are bounded.
- Next recommended task: replace bounded heuristic replay/queue scoring in `src/orchestrator/shadow_advisory.py`, `src/rl/econ_regal_sampling.py`, and `src/orchestrator/queue_selection.py` with learned runtime scorer outputs where replay coverage is sufficient.

- Changed: added a checked-in full-stack training backlog at `docs/economic_world_model/full_stack_training_backlog.md` that ranks the real learned lanes by production importance, data dependency, and honest current readiness instead of treating every `train_*.py` surface as equally actionable.
- Changed: added `scripts/runpod/FULL_STACK_TRAINING_BUNDLES.json` plus `scripts/runpod/assess_full_stack_training.py`, `scripts/runpod/execute_training_bundle.py`, and `scripts/runpod/launch_training_bundle.py` so the repo now has a real recurring-training scaffold for Runpod with explicit readiness gates, bundle costs, pod teardown behavior, and a default preference for data-refresh over premature heavyweight model training.
- Changed: the new Runpod backlog intentionally keeps frozen-baseline lanes out of the recurring automation path and marks perception-neuralization as manual-only until real stage outputs and less synthetic training data exist.
- Changed: wired training-run receipt ingest to preserve per-episode backend-truth evidence from observed online receipts before replay precondition scoring. `src/replay/receipt_ingest.py` now merges `scene_tracks_non_stub`, SceneTracks backend identity, teacher backend identity, and grounding flags into enriched episode metadata so future-training predicate checks can reflect real runtime evidence instead of defaulting false.
- Changed: updated `tests/test_training_run_receipt_ingest.py` to include explicit real backend fields in online receipt rows and assert `signal_bool::scene_tracks_non_stub==1` and `signal_bool::teacher_runtime_real==1` in the execution-precondition summary.
- Changed: updated `scripts/economic_world_model/run_receipt_readiness_probe.py` to emit those same backend-truth fields and re-ran the probe. Current report (`artifacts/economic_world_model/readiness_probe/readiness_probe_summary.json`) now shows all targeted predicates satisfied: `budget_settlement_live=1`, `scene_tracks_non_stub=1`, `teacher_runtime_real=1`.
- Changed: fixed `scripts/economic_world_model/nightly_audit.py` progress freshness logic so `_progress_latest_date()` now returns the chronologically newest dated heading from `docs/economic_world_model/progress_log.md` instead of the last heading in file order. This avoids stale drift checks when the log is maintained newest-first.
- Changed: tightened `tests/test_economic_world_model_nightly_audit.py::test_progress_latest_date_uses_most_recent_heading` to cover newest-first ordering (`2026-03-26` then `2026-03-24`) and assert the audit picks the real newest date.
- Changed: added `scripts/economic_world_model/run_receipt_readiness_probe.py` to run a minimal real training-run finalization through `RegalTrainingRunner` and `build_training_run_receipt_label_bundle(...)`, then emit a stable readiness report under `artifacts/economic_world_model/readiness_probe/`.
- Changed: executed the new probe and recorded live predicate results in `artifacts/economic_world_model/readiness_probe/readiness_probe_summary.json` and `artifacts/economic_world_model/readiness_probe/readiness_probe_summary.md`. Current target counts: `signal_bool::budget_settlement_live=1`, `signal_bool::scene_tracks_non_stub=0`, `signal_bool::teacher_runtime_real=0`.
- Changed: ran the nightly audit loop with `python3 scripts/economic_world_model/nightly_audit.py --output-json artifacts/economic_world_model/nightly_audit_summary.json --output-markdown artifacts/economic_world_model/nightly_audit_summary.md` and refreshed both audit artifacts for this run.
- Changed: audit selection remains `next_task.id=audit_only` with `execute_now=false`, so no new safe additive scaffold was selected in this pass.
- Verification: `./scripts/agent/verify.sh`, `PYTHONPYCACHEPREFIX=/tmp/pycache python3 -m compileall src scripts/economic_world_model -q`, `python3 -m pytest -q tests/test_runtime_packets.py tests/embodiment/test_registry.py tests/test_objective_runtime_builder.py tests/test_constraint_set.py tests/test_pricing_sentinel.py tests/test_value_ledger.py tests/test_economic_world_model_nightly_audit.py tests/test_training_run_receipt_ingest.py`, and `python3 scripts/economic_world_model/run_receipt_readiness_probe.py --output-root artifacts/economic_world_model/readiness_probe --seed 17` passed.
- Verification: `python3 -m compileall scripts/runpod -q` and `python3 scripts/runpod/assess_full_stack_training.py --bundle auto` passed.
- Blocked: no additive code-path blocker was detected; this was an intentional audit-only pass because the current selector found no higher-priority missing scaffold.
- Next recommended task: use the new backlog as the source of truth for the cross-window heuristic/advisory/sidecar inventory, then wire the highest-impact non-authoritative surfaces into the live runtime/training/reward loop before choosing the next canonical WM tranche.

## 2026-03-24

- Changed: hardened the OpenVLA and MetaDINO bring-up paths to `real-or-unavailable` by default instead of silently treating stubs as acceptable runtime behavior. `src/vla/openvla_controller.py` now exposes `backend_policy` and `vision_backbone_policy` (`auto|real|disabled|stub`), reports explicit backend status, and only emits zero-action outputs when `stub` is explicitly requested. `src/vla/backbones/meta_dino_backbone.py` now behaves the same way for DINO embeddings: real model, explicit stub, or unavailable with a hard runtime error when a caller tries to use an unavailable backbone as if it were real.
- Changed: wired those backend states through the teacher and replay/evidence layers instead of keeping them local to the controller. `src/vla/teacher_runtime.py`, `src/vla/rollout_labeler.py`, `src/replay/preconditions.py`, `src/replay/importers.py`, and `src/replay/ingest.py` now preserve explicit OpenVLA / vision-backbone / non-heuristic-grounding signals so later benchmark and promotion code can distinguish `real`, `stub`, `disabled`, and `unavailable` instead of inferring “live enough” from generic availability booleans.
- Changed: added typed benchmark gating in `src/evidence/benchmark_gating.py`. Benchmark readiness can now explicitly require real SceneTracks, real teacher runtime, and real vision backbone lanes, and it blocks passthrough/stub/heuristic cases rather than letting them quietly count as benchmark-ready.
- Changed: added a separate loop-run backlog and automatic scanner rather than mixing loop operations into the training backlog. `scripts/LOOP_RUN_BACKLOG.json` now records concrete loop runs to test, their required host/data/model preconditions, their internal/external data dependencies, and whether they are safe for auto-trigger. `src/orchestrator/loop_run_backlog.py` and `scripts/scan_loop_run_backlog.py` evaluate those preconditions and can optionally execute ready auto-trigger runs.
- Changed: updated CLI/smoke surfaces so strict policy is visible at the operational boundary. `scripts/run_vla_on_episode.py`, `scripts/smoke_test_openvla_controller.py`, and `scripts/smoke_test_vision_backbone.py` now expose or reflect explicit backend-policy selection instead of assuming dummy backbones are an acceptable implicit fallback.
- Verification: `python3 -m compileall src/vla src/evidence src/replay src/orchestrator scripts/scan_loop_run_backlog.py scripts/run_vla_on_episode.py scripts/smoke_test_openvla_controller.py scripts/smoke_test_vision_backbone.py tests/test_vla_backend_policy.py tests/test_benchmark_gating.py tests/test_loop_run_backlog.py tests/test_teacher_runtime.py -q`, `python3 -m ruff check src/vla/backbones/meta_dino_backbone.py src/vla/openvla_controller.py src/vla/rollout_labeler.py src/vla/teacher_runtime.py src/evidence/benchmark_gating.py src/evidence/__init__.py src/replay/preconditions.py src/replay/importers.py src/replay/ingest.py src/orchestrator/loop_run_backlog.py scripts/scan_loop_run_backlog.py scripts/run_vla_on_episode.py scripts/smoke_test_openvla_controller.py scripts/smoke_test_vision_backbone.py tests/test_vla_backend_policy.py tests/test_benchmark_gating.py tests/test_loop_run_backlog.py tests/test_teacher_runtime.py`, and `python3 -m pytest -q tests/test_vla_backend_policy.py tests/test_benchmark_gating.py tests/test_loop_run_backlog.py tests/test_teacher_runtime.py tests/test_rollout_labeler.py tests/test_replay_dataset.py` passed.

- Changed: added the scorer tranche on top of the semantic runtime corpus. `src/orchestrator/semantic_runtime_scorers.py` now trains lightweight route-success, authority-calibration, counterfactual-value, and regret models from replay-backed semantic runtime rows, and it can score live semantic-world-model plus transformer packets in shadow mode.
- Changed: added heavyweight scorer-training plumbing rather than leaving the learned path implicit. `src/orchestrator/semantic_runtime_scorer_training.py` now builds explicit scorer-training datasets from the same runtime rows and exposes an optional torch multitask training/checkpoint path for later learned reranking work.
- Changed: `scripts/train_semantic_runtime_scorers.py` now materializes the full scorer-training surface:
  - `semantic_runtime_scorer_training_dataset.json`
  - `semantic_runtime_scorer_package.json`
  - `semantic_runtime_shadow_scores.jsonl`
  - `semantic_runtime_scorer_model.pt` when torch training is enabled and available
  - `semantic_runtime_scorer_summary.json`
- Changed: `run_pipeline_step_with_causal_order(...)` now supports both transformer callouts live at the same boundary. It can emit `orchestration_transformer_execution` and a shared `semantic_runtime_scoring` packet, so the semantic world model now feeds both transformer lanes and gets shadow route/calibration/regret feedback back out immediately.
- Changed: added `train_semantic_runtime_scorers.py` to `scripts/TRAINING_MIGRATION_BACKLOG.json` so the heavyweight learned scorer path is tracked in the repo's explicit training backlog rather than being left as an informal next step.
- Verification: `python3 -m compileall src/orchestrator scripts/train_semantic_runtime_scorers.py tests/test_semantic_runtime_scorers.py -q`, `python3 -m ruff check src/orchestrator/semantic_runtime_scorers.py src/orchestrator/semantic_runtime_scorer_training.py src/orchestrator/pipeline_manager.py scripts/train_semantic_runtime_scorers.py tests/test_semantic_runtime_scorers.py`, and `python3 -m pytest -q tests/test_semantic_runtime_scorers.py tests/test_semantic_runtime_learning.py tests/test_semantic_transformer_execution.py` passed.

- Changed: added the pre-training semantic runtime learning layer instead of waiting for a learned controller run. `src/orchestrator/semantic_runtime_learning.py` now harvests replay-backed semantic world-model rows, teacher/VLA evidence, DINO/SceneTracks proxy evidence, fusion/outcome summaries, transformer targets, and shadow counterfactuals into one canonical corpus.
- Changed: added runtime-dataset export for both transformer lanes. `scripts/export_semantic_runtime_learning_corpus.py` now loads a canonical replay dataset and emits:
  - `semantic_runtime_learning_rows.jsonl`
  - `semantic_runtime_learning_summary.json`
  - `meta_transformer_runtime_dataset.json`
  - `orchestration_runtime_dataset.json`
- Changed: the corpus now closes the broader semantic feedback loop in code rather than docs only:
  - OpenVLA / teacher semantic evidence feeds the semantic world model through teacher traces and VLA sidecars
  - DINO / SceneTracks / Map-First proxy evidence feeds the same world model through grounding summaries
  - semantic-world-model state feeds both transformer shells
  - replay/outcome evidence plus shadow counterfactuals feed back into future training and inferential labels
- Changed: added `docs/economic_world_model/semantic_runtime_learning_loop.md` to spell out the end-to-end production loop and the distinction between the learning pipeline and the inferential pipeline.
- Verification: `python3 -m compileall src/orchestrator scripts/export_semantic_runtime_learning_corpus.py tests/test_semantic_runtime_learning.py -q`, `python3 -m ruff check src/orchestrator/semantic_runtime_learning.py scripts/export_semantic_runtime_learning_corpus.py tests/test_semantic_runtime_learning.py src/orchestrator/meta_transformer.py src/orchestrator/orchestration_transformer.py src/orchestrator/semantic_transformer_bridge.py`, and `python3 -m pytest -q tests/test_semantic_runtime_learning.py tests/test_semantic_transformer_execution.py` passed.

- Changed: promoted the transformer callouts from semantic-adjacent scaffolds into bounded execution packets. Added `src/orchestrator/semantic_transformer_bridge.py` as the shared semantic-world-model featurization layer, and both `src/orchestrator/meta_transformer.py` and `src/orchestrator/orchestration_transformer.py` now consume semantic-world-model state directly instead of only carrying shallow summary fields.
- Changed: `MetaTransformer.propose_plan(...)` now exists as a live pipeline surface. It compiles econ/datapack/semantic inputs into semantic-aware objective/backend/energy/data-mix choices, bounded orchestration steps, execution preconditions, and a work order instead of silently no-oping from `pipeline_manager`.
- Changed: the orchestration transformer is no longer only a generic context encoder over econ fields. `OrchestratorContext` can now carry semantic-world-model context, `_encode_ctx(...)` appends semantic-WM features, and `propose_orchestrated_plan(...)` now emits execution mode, activation plan, execution preconditions, and activation work order in addition to tool steps.
- Changed: `run_pipeline_step_with_causal_order(...)` now threads semantic-world-model inputs into the meta-transformer call and surfaces the resulting execution packet under `meta_transformer_execution`, so the transformer lane is no longer suggestion-only at the pipeline boundary.
- Changed: added `docs/economic_world_model/semantic_authority_promotion.md` to make the intended promotion path explicit: advisory packet -> preconditioned execution -> bounded meta-node authority -> learned control plane.
- Verification: `python3 -m compileall src tests/test_semantic_transformer_execution.py -q`, `python3 -m ruff check src/orchestrator/context.py src/orchestrator/meta_transformer.py src/orchestrator/orchestration_transformer.py src/orchestrator/pipeline_manager.py src/orchestrator/semantic_transformer_bridge.py tests/test_semantic_transformer_execution.py`, and `python3 -m pytest -q tests/test_semantic_transformer_execution.py` passed.

- Changed: made real on-device SAM3D activation automatic at the runner boundary. `run_scene_tracks(...)` now defaults to `backend_policy="auto"`, which tries a real local SAM3D tracker first with `allow_fallbacks=False`, then falls back to the explicit zero-inference passthrough backend when real weights/deps are unavailable but segmentation masks exist. Silent stub selection is no longer the default path.
- Changed: recorded backend resolution explicitly in run metadata. SceneTracks runner metadata now includes `backend_policy`, `backend_selected`, and any `real_backend_failure`, so downstream consumers can distinguish a truly local SAM3D run from a deterministic passthrough run without inferring it from adapter internals.
- Changed: exposed the same policy to callers. `scripts/run_scene_tracks.py` now supports `--backend-policy auto|real|passthrough|stub`, and `XHumanoidIngestConfig` now carries `scene_tracks_backend_policy` so higher ingestion paths can opt into the same precondition-based backend resolution.
- Verification: `python3 -m compileall src/vision/scene_ir_tracker src/ingestion scripts/run_scene_tracks.py tests/integration/test_scene_tracks_from_workcell_datapack.py tests/vision/scene_ir_tracker/test_fallback_behavior.py tests/vision/scene_ir_tracker/test_zero_inference_passthrough.py -q`, `python3 -m ruff check src/vision/scene_ir_tracker/io/scene_tracks_runner.py src/ingestion/x_humanoid_adapter.py scripts/run_scene_tracks.py tests/integration/test_scene_tracks_from_workcell_datapack.py tests/vision/scene_ir_tracker/test_fallback_behavior.py tests/vision/scene_ir_tracker/test_zero_inference_passthrough.py`, and `python3 -m pytest -q tests/integration/test_scene_tracks_from_workcell_datapack.py tests/vision/scene_ir_tracker/test_fallback_behavior.py tests/vision/scene_ir_tracker/test_zero_inference_passthrough.py` passed.
- Changed: added an explicit zero-inference SceneTracks backend so SAM3D plumbing can stay live without requiring local model execution. `SceneIRTrackerConfig` now supports `zero_inference_passthrough`, `run_scene_tracks(...)` and `scripts/run_scene_tracks.py` expose that mode, and `SceneIRTracker` can now synthesize tracked 3D objects/bodies directly from segmentation, depth, and camera geometry while still emitting the normal `SceneTracks_v1` / semantic sidecars.
- Changed: the new passthrough backend is explicit rather than a silent stub. `adapter_status` now reports `overall_mode=passthrough`, `x_humanoid_adapter` can request the same mode, and training-eligibility gating remains off unless the runner is actually using real SAM3D backends.
- Verification: `python3 -m compileall src/vision/scene_ir_tracker src/ingestion scripts/run_scene_tracks.py tests/vision/scene_ir_tracker/test_zero_inference_passthrough.py tests/integration/test_scene_tracks_from_workcell_datapack.py tests/vision/scene_ir_tracker/test_fallback_behavior.py -q`, `python3 -m ruff check src/vision/scene_ir_tracker/config.py src/vision/scene_ir_tracker/tracker.py src/vision/scene_ir_tracker/io/scene_tracks_runner.py src/ingestion/x_humanoid_adapter.py scripts/run_scene_tracks.py tests/vision/scene_ir_tracker/test_zero_inference_passthrough.py tests/integration/test_scene_tracks_from_workcell_datapack.py tests/vision/scene_ir_tracker/test_fallback_behavior.py`, and `python3 -m pytest -q tests/vision/scene_ir_tracker/test_zero_inference_passthrough.py tests/vision/scene_ir_tracker/test_fallback_behavior.py tests/integration/test_scene_tracks_from_workcell_datapack.py` passed.
- Changed: attacked the upstream semantic-quality gap instead of adding another downstream wrapper. `src/vision/scene_ir_tracker/io/datapack_frame_reader.py` now derives per-frame class labels plus scene/object semantic context from datapack metadata and workcell `scene_spec` information, and `src/vision/scene_ir_tracker/io/scene_tracks_runner.py` now feeds those labels into `SceneIRTracker` rather than emitting geometry-only tracks.
- Changed: `run_scene_tracks(...)` now enriches `SceneTracks_v1` artifacts with track-level semantic metadata (`track_label_confidence`, `track_label_source`, `track_category`, `track_motion_score`, semantic-tag JSON, affordance JSON, and merged semantic summaries), and it mirrors semantic density / grounding readiness back into datapack metadata and execution-precondition signals.
- Changed: `src/evidence/teacher_trace.py`, `src/vla/teacher_runtime.py`, and `src/vla/semantic_evidence.py` now infer and persist structured teacher-side semantic hints (`object_refs`, `affordance_hints`, `risk_hints`, richer `semantic_tags`) from instructions plus VLA outputs instead of treating teacher payloads as action-only.
- Changed: `src/world_model/semantic_world_model.py` now consumes those new producer-side semantic fields, so grounded objects absorb label provenance, label confidence, hint object IDs, extra affordances/risk tags, and teacher-object matching instead of ignoring the richer upstream packets.
- Changed: removed a major remaining heuristic by preserving explicit object identity through the tracking stack. `SceneEntity3D`, `SceneIRTracker`, and `KalmanTrackManager` now carry `source_instance_id` / `source_object_id`, `SceneTracks_v1` persists those refs, and the world model now prefers explicit track-source object IDs over class-name guessing.
- Changed: promoted explicit segmentation-label metadata into the sensor-bundle contract. `src/motor_backend/sensor_bundle.py` and `src/motor_backend/workcell_env_backend.py` now emit `segmentation_label_map` plus `scene_object_catalog`, so datapack readers can consume true object/label joins from sensor bundles instead of reconstructing them only from scene-spec ordering.
- Changed: made stub/fallback dependence explicit instead of silent. The SAM3D adapters now expose backend modes, `SceneIRTracker` reports adapter status, and `run_scene_tracks(...)` now blocks training-ready classification unless the tracker backends are real rather than stub/fallback-degraded.
- Changed: `src/vla/rollout_labeler.py` now preserves structured teacher semantics when it writes teacher traces and VLA semantic-evidence sidecars, so object/risk/affordance hints no longer collapse back into a plain tag list on that path.
- Verification: `python3 -m compileall src scripts/run_stage1_pipeline.py tests -q`, `python3 -m ruff check src/vision/scene_ir_tracker/io/datapack_frame_reader.py src/vision/scene_ir_tracker/io/scene_tracks_runner.py src/evidence/teacher_trace.py src/vla/teacher_runtime.py src/vla/semantic_evidence.py src/world_model/semantic_world_model.py tests/test_teacher_runtime.py tests/integration/test_scene_tracks_from_workcell_datapack.py`, and `python3 -m pytest -q tests/test_teacher_runtime.py tests/integration/test_scene_tracks_from_workcell_datapack.py tests/test_semantic_world_model_backbone.py tests/test_semantic_fusion_orchestrator_smoke.py tests/test_stage1_pipeline_governed.py tests/test_semantic_fusion_emit_flag.py` all passed.
- Verification: `python3 -m ruff check src/vision/scene_ir_tracker/types.py src/vision/scene_ir_tracker/tracker.py src/vision/scene_ir_tracker/kalman_track_manager.py src/vision/scene_ir_tracker/sam3d_objects_adapter.py src/vision/scene_ir_tracker/sam3d_body_adapter.py src/vision/scene_ir_tracker/io/datapack_frame_reader.py src/vision/scene_ir_tracker/io/scene_tracks_runner.py src/motor_backend/sensor_bundle.py src/motor_backend/workcell_env_backend.py src/vla/rollout_labeler.py src/world_model/semantic_world_model.py tests/integration/test_scene_tracks_from_workcell_datapack.py tests/integration/test_multicamera_sensor_bundle.py tests/test_rollout_labeler.py`, `python3 -m pytest -q tests/test_teacher_runtime.py tests/test_rollout_labeler.py tests/integration/test_scene_tracks_from_workcell_datapack.py tests/integration/test_multicamera_sensor_bundle.py tests/test_semantic_world_model_backbone.py tests/test_semantic_fusion_orchestrator_smoke.py tests/test_stage1_pipeline_governed.py tests/test_semantic_fusion_emit_flag.py`, and `git diff --check` all passed.
- Blocked: the remaining gap is now mostly model-side rather than metadata-side. Real sensor paths still need non-stub SAM3D / teacher backends and stronger segmentation/identity exports outside workcell-style bundles; when those are absent the repo now fails or degrades explicitly instead of quietly pretending the grounding is real.
- Next recommended task: attack the actual model boundary next by wiring real sensor-bundle label exporters and non-stub SAM3D/OpenVLA execution into the ingestion paths that still arrive without explicit segmentation/object identity metadata.

## 2026-03-07

- Changed: created the architecture gap analysis, staged roadmap, nightly audit runbook, Codex skill docs, automation spec, repo-local skill, audit/update scripts, scheduled workflow, `RuntimePacket` scaffolding, and `EmbodimentRegistry` scaffolding.
- Verification: `./scripts/agent/verify.sh`, `python3 -m compileall src scripts/economic_world_model -q`, `python3 -m pytest -q tests/test_runtime_packets.py tests/embodiment/test_registry.py tests/test_objective_runtime_builder.py tests/test_constraint_set.py tests/test_pricing_sentinel.py tests/test_value_ledger.py`, `bash -n scripts/economic_world_model/run_nightly_codex_task.sh`, and the audit script all passed.
- Blocked: no live robot telemetry, app automations still require manual UI creation even though the app-first prompt/spec is ready, and GitHub/cloud Codex execution still requires a configured `CODEX_API_KEY` secret. The current audit sees no API key in the local environment.
- Next recommended task: wire `RuntimePacket` sidecars into `src/shadow_runtime/control_plane.py` and `src/replay/ingest.py` without changing default runtime behavior.

## 2026-03-08

- Changed: wired additive `RuntimePacket` sidecar emission into `src/shadow_runtime/control_plane.py`, persisted run-level packet sidecars under `runtime_packets.json`, and threaded packet refs/IDs into replay episode/step/window ingest metadata and provenance in `src/replay/ingest.py`.
- Verification: `python3 -m pytest -q tests/test_runtime_packets.py tests/test_shadow_econ_runner.py tests/test_replay_schema.py tests/test_replay_dataset.py`, plus `python3 -m compileall src -q`.
- Blocked: packet schemas are still shadow-workcell-derived and not yet backed by a generalized observation/action adapter layer; older shadow runs without `runtime_packets.json` still ingest in compatibility mode with no packet refs.
- Next recommended task: add an additive `EventSpine` / `DecisionLedger` sidecar for per-window decisions, vetoes, and pricing/adaptation events, then thread its refs into replay metadata beside the new packet refs.

- Changed: added additive `EventSpine` and `DecisionLedger` sidecars under `event_spine.json` and `decision_ledger.json`, emitted stable event/decision IDs tied to runtime packet IDs, contract IDs, artifact refs, and actor/critic/advisor provenance, and threaded those refs into replay episode/step/window `metadata` and `provenance` without changing replay record shapes.
- Verification: `python3 -m pytest -q tests/test_event_spine.py tests/test_shadow_econ_runner.py tests/test_replay_schema.py tests/test_replay_dataset.py tests/test_receipt_ingest.py`, plus `python3 -m compileall src -q`.
- Blocked: receipt label refs are currently empty placeholders because receipt labels are still attached downstream, and current event producers are shadow-only rather than shared with `sim_rollout` or training-run producers.
- Next recommended task: consume `event_spine.json` and `decision_ledger.json` in promotion reporting and multi-run stage movement so promotion holds, vetoes, pricing suppression, and collect-more-data decisions stop being inferred indirectly from summary fields.

## 2026-03-09

- Changed: added `ActionAdapterV2` and `ObservationAdapterV2`, broadened runtime packet builders to accept schema-producing adapters, and reopened `src/world_model/` with `GovernedVideoWorldModel` while keeping the stable checkpoint baseline intact.
- Changed: added `EvidenceBus`, `BeliefState`, and `TeacherTrace` scaffolding; wired teacher traces into `src/vla/rollout_labeler.py`; and wired semantic fusion to emit `*_evidence_bus_v1.json` and `*_belief_state_v1.json` sidecars.
- Changed: upgraded the Stage-1 video path to support manifest-backed video references, deterministic semantic extraction, governed video-state sidecars/hypotheses, and hypothesis-conditioned diffusion rendering; also made SceneTracks stub adapters configurable in the runner API.
- Changed: aligned repo-level docs and planning artifacts around the new Phase B posture: stable baseline frozen, `src/world_model/` reopened additively for governed successor modules, real-video grounding and governed supervision added as next roadmap stages, and learned video-state training moved into the training backlog as a deferred subset of economic-world-model readiness.
- Changed: tightened the roadmap and automation docs so autonomous execution now explicitly prioritizes Week 6.5 reconstruction/teacher-runtime work and Week 6.75 governed supervision before any learned video-state training pass.
- Verification: `python3 -m compileall src scripts/run_stage1_pipeline.py tests -q`, `python3 -m pytest -q tests/test_evidence_bus.py tests/test_runtime_adapters_v2.py tests/test_governed_video_world_model.py tests/test_rollout_labeler.py tests/test_semantic_fusion_orchestrator_smoke.py tests/test_stage1_pipeline_governed.py`, `python3 -m pytest -q tests/test_runtime_packets.py tests/test_vla_semantic_evidence.py tests/test_semantic_fusion_mvp.py tests/test_diffusion_prompt_includes_constraints.py`, and `python3 scripts/run_stage1_pipeline.py --num-videos 1 --proposals-per-video 1 --output-dir /tmp/stage1_governed_smoke` all passed.
- Blocked: the new video-state service is still heuristic/advisory, SceneTracks still defaults to stub adapters unless configured otherwise, and OpenVLA remains soft-fail instead of production-enforced.
- Next recommended task: add a D4RT-style reconstruction sidecar plus real SceneTracks/OpenVLA adapter plumbing so the governed video-state service stops depending on fallback evidence for real footage.

- Changed: wired Week 6.5 and Week 6.75 artifacts into live paths rather than leaving them as standalone helpers. `scripts/run_stage1_pipeline.py` now emits reconstruction sidecars, runtime packets, branch evaluations, event-spine sidecars, decision-ledger sidecars, governance traces, counterfactual evals, value-target packs, and value-ledger receipts for each governed video episode.
- Changed: tightened `src/vla/rollout_labeler.py` plus `src/vla/teacher_runtime.py` so rollout labeling now emits teacher contract and teacher action-envelope sidecars even when OpenVLA is disabled, missing, or failing; fallback state is now explicit and replayable.
- Changed: expanded focused coverage with reconstruction, teacher-runtime, and governed-supervision tests and strengthened Stage-1 / rollout-labeler assertions around live-loop artifact emission.
- Verification: `python3 -m compileall src scripts/run_stage1_pipeline.py tests -q` and `python3 -m pytest -q tests/test_rollout_labeler.py tests/test_stage1_pipeline_governed.py tests/test_four_d_reconstruction.py tests/test_teacher_runtime.py tests/test_governed_video_supervision.py` passed.
- Blocked: the live Stage-1 path still lacks real SceneTracks adapters, richer calibration sources, and non-stub teacher execution from real video frames; current grounding remains truthful-but-advisory rather than production-final.
- Next recommended task: push the same live-loop discipline into real-video ingestion boundaries, especially SceneTracks calibration joins and remaining teacher-runtime consumers, before any learned predictor training.

## 2026-03-19

- Changed: fixed `scripts/economic_world_model/nightly_audit.py` so progress-log freshness uses the most recent dated heading instead of the first heading, removing a stale false-positive drift signal against `scripts/TRAINING_MIGRATION_BACKLOG.json`.
- Changed: replaced the hardcoded EventSpine pending flag with real completion detection via additive code/doc checks (`src/runtime/event_spine.py`, `src/governance/trace.py`, and roadmap/gap-analysis phrase checks), so the nightly next-task selector no longer recommends already-landed work.
- Changed: updated audit compile verification to use `PYTHONPYCACHEPREFIX=/tmp/pycache` so sandboxed/local runs do not fail on unwritable default Python cache paths.
- Changed: added regression tests in `tests/test_economic_world_model_nightly_audit.py` for latest-date parsing, EventSpine pending detection, and audit-only fallback selection.
- Verification: `PYTHONPYCACHEPREFIX=/tmp/pycache python3 -m compileall src scripts/economic_world_model -q`, `python3 -m pytest -q tests/test_economic_world_model_nightly_audit.py tests/test_runtime_packets.py tests/embodiment/test_registry.py tests/test_objective_runtime_builder.py tests/test_constraint_set.py tests/test_pricing_sentinel.py tests/test_value_ledger.py`, and `python3 scripts/economic_world_model/nightly_audit.py --output-json artifacts/economic_world_model/nightly_audit_summary.json --output-markdown artifacts/economic_world_model/nightly_audit_summary.md` (result: `Status: ok`, drift signals: none).
- Blocked: `codex_api_key_present` remains `no`, so GitHub/cloud Codex execution is still credential-gated even though local CLI/app paths are ready.
- Next recommended task: prioritize a Week 6.5 additive grounding pass that wires richer SceneTracks calibration joins and remaining teacher-runtime consumers into real-video ingestion boundaries, then add focused smoke/test coverage before any learned predictor training.

## 2026-03-21

- Changed: added additive dataset-bridge scaffolding at `src/dataset_bridges/rlds_bridge.py` and `src/dataset_bridges/lerobot_bridge.py` (plus package exports) to provide lossy RLDS/LeRobot adapters while preserving references to internal objective/econ/governance/runtime sidecars in metadata.
- Changed: added focused bridge coverage in `tests/test_dataset_bridges.py` to lock down replay-step conversion semantics and sidecar-reference preservation.
- Changed: extended `scripts/economic_world_model/nightly_audit.py` with Week 7+ detection (`_dataset_bridge_scaffold_pending`) and a new `dataset_bridge_scaffold` task candidate so nightly selection no longer reports `audit_only` when dataset bridges are missing.
- Changed: expanded `tests/test_economic_world_model_nightly_audit.py` with explicit coverage for the new dataset-bridge task selection path.
- Verification: `PYTHONPYCACHEPREFIX=/tmp/pycache python3 -m compileall src scripts/economic_world_model -q`, `python3 -m pytest -q tests/test_dataset_bridges.py tests/test_economic_world_model_nightly_audit.py tests/test_runtime_packets.py tests/embodiment/test_registry.py tests/test_objective_runtime_builder.py tests/test_constraint_set.py tests/test_pricing_sentinel.py tests/test_value_ledger.py`, and `python3 scripts/economic_world_model/nightly_audit.py --output-json artifacts/economic_world_model/nightly_audit_summary.json --output-markdown artifacts/economic_world_model/nightly_audit_summary.md`.
- Blocked: `codex_api_key_present` remains `no` in this environment, so GitHub/cloud Codex execution is still credentials-gated.
- Next recommended task: deepen Week 7+ replay export/import glue so Stage-1 governed supervision artifacts can be emitted through dataset-bridge bundles without bespoke joins.

## 2026-03-22

- Changed: added an additive semantic-world-model packet in `src/world_model/semantic_world_model.py` plus runtime bridging in `src/semantic/runtime_backbone.py`, so Stage 1 and semantic-fusion runtime paths now materialize `SemanticWorldModelState`, `SemanticSnapshot`, and meta-node-oriented `OrchestratorAdvisory` sidecars instead of stopping at flat tags or local fusion artifacts.
- Changed: wired the new packet through `scripts/run_stage1_pipeline.py`, `src/orchestrator/semantic_fusion_runner.py`, `src/semantic/models.py`, `src/semantic/aggregator.py`, `src/orchestrator/semantic_orchestrator_v2.py`, `src/observation/adapter.py`, `src/observation/condition_vector_builder.py`, and `src/rl/episode_sampling.py` so capabilities, topology, and meta-node weights reach runtime observation, conditioning, and sampling surfaces.
- Changed: upgraded `SemanticWorldModelBuilder` so it now consumes real `SceneTracks_v1`, teacher traces, and VLA semantic evidence when those artifacts exist, deriving track-scoped objects plus spatial relations from `track_ids`, `poses_t`, visibility, occlusion, convergence, and class labels before falling back to heuristic semantic priors.
- Changed: Stage 1 manifest inputs and the rollout semantic-fusion runner now both pass SceneTracks and teacher/VLA artifacts into the semantic world model, and new tests lock down grounded-track object and relation emission.
- Changed: added `docs/economic_world_model/semantic_gap_matrix.md`, which translates the semantic stack topologically, functionally, and capability-wise and records the remaining non-wired gaps.
- Verification: `python3 -m compileall src scripts/run_stage1_pipeline.py tests -q`, `python3 -m ruff check scripts/run_stage1_pipeline.py src/world_model/semantic_world_model.py src/semantic/models.py src/semantic/aggregator.py src/semantic/runtime_backbone.py src/orchestrator/semantic_orchestrator_v2.py src/orchestrator/semantic_fusion_runner.py src/observation/adapter.py src/observation/condition_vector_builder.py src/rl/episode_sampling.py tests/test_stage1_pipeline_governed.py tests/test_semantic_world_model_backbone.py tests/test_semantic_fusion_orchestrator_smoke.py`, and `python3 -m pytest -q tests/test_governed_video_world_model.py tests/test_semantic_policy.py tests/test_stage1_pipeline_governed.py tests/test_semantic_world_model_backbone.py tests/test_semantic_fusion_orchestrator_smoke.py tests/test_semantic_fusion_emit_flag.py`.
- Blocked: grounded semantics now works when SceneTracks and teacher/VLA artifacts exist, but Stage 1 still depends on manifest-provided grounding and rollout fusion still inherits any upstream stub-adapter limitations from SceneTracks/OpenVLA.
- Next recommended task: push real class labeling and track-quality joins further upstream so more ingestion paths emit fully populated `SceneTracks_v1` and teacher/VLA semantic evidence instead of partial or stub payloads.

- Changed: deepened Week 7+ replay export glue by adding `src/dataset_bridges/sidecar_refs.py` and switching RLDS/LeRobot adapters to generic sidecar extraction across replay record fields plus `metadata`/`provenance` keys ending in `*_ref`, `*_refs`, `*_id`, or `*_ids`.
- Changed: bridge exports now preserve newly added governed-supervision-style references (for example `counterfactual_eval_ref`, `value_target_refs`, and `belief_state_ref`) without requiring per-key adapter rewrites.
- Changed: extended `tests/test_dataset_bridges.py` coverage so both adapters assert preservation of governed-supervision and teacher-trace oriented references from step and episode records.
- Verification: `PYTHONPYCACHEPREFIX=/tmp/pycache python3 -m compileall src scripts/economic_world_model -q`, `python3 -m pytest -q tests/test_dataset_bridges.py tests/test_economic_world_model_nightly_audit.py tests/test_runtime_packets.py tests/embodiment/test_registry.py tests/test_objective_runtime_builder.py tests/test_constraint_set.py tests/test_pricing_sentinel.py tests/test_value_ledger.py`, and `python3 scripts/economic_world_model/nightly_audit.py --output-json artifacts/economic_world_model/nightly_audit_summary.json --output-markdown artifacts/economic_world_model/nightly_audit_summary.md`.
- Blocked: `codex_api_key_present` remains `no` in this environment, so GitHub/cloud Codex execution remains credentials-gated.
- Next recommended task: add replay import glue that rehydrates bridge-exported sidecar refs into canonical replay metadata/provenance so RLDS/LeRobot roundtrips remain loss-bounded but trace-complete.

- Changed: added `scripts/economic_world_model/publish_codex_change.sh` so nightly work publishes to `origin/main` when possible and otherwise falls back to a timestamped `codex/ewm-nightly-*` feature branch instead of leaving commits local-only.
- Changed: updated the nightly runner prompt, automation spec, live automation prompt, and repo-local roadmap skill so publication is now an explicit completion requirement and every run must report the published ref or the exact push blocker.
- Verification: `bash -n scripts/economic_world_model/run_nightly_codex_task.sh`, `bash -n scripts/economic_world_model/publish_codex_change.sh`, and `bash scripts/economic_world_model/publish_codex_change.sh --base-branch main --feature-prefix codex/ewm-nightly`.
- Blocked: none for local publication on this pass; the direct push to `origin/main` succeeded after this automation-substrate update.
- Next recommended task: add replay import glue that rehydrates bridge-exported sidecar refs into canonical replay metadata/provenance so RLDS/LeRobot roundtrips remain loss-bounded but trace-complete.

- Changed: added `src/economics/inferential_reward.py` as an additive successor compiler for inferential reward and signal yield, keeping the stable Phase B reward path untouched while letting budget-gating and replay-selection paths consume frontier gain plus epiplexity evidence.
- Changed: wired the compiled inferential reward into `src/economics/inferential_training_gate.py`, `src/orchestrator/shadow_advisory.py`, `src/rl/econ_regal_sampling.py`, `src/rl/episode_sampling.py`, `src/policies/sampler_weights.py`, and `src/orchestrator/queue_selection.py` so advisory replay and inferential budget decisions can see a canonical signal-yield term instead of only ad hoc scalar heuristics.
- Verification: `python3 -m pytest -q tests/test_inferential_training_gate.py tests/epiplexity/test_epiplexity_sampling_orchestrator.py tests/test_econ_regal_sampling.py tests/test_receipt_ingest.py tests/test_shadow_advisory_pass.py`, plus `python3 -m compileall src -q`.
- Blocked: epiplexity is still sparse in replay/shadow artifacts, so the new inferential reward path falls back to frontier-gain support when no epiplexity fields are present.
- Next recommended task: persist canonical epiplexity / signal-yield sidecars into replay and datapack overlays so the new inferential reward compiler can consume real learnability evidence end to end rather than fallback frontier proxies.

- Changed: fixed the canonical epiplexity contract so cached tracker results are absolute and compute-accounted (`flops_estimate`) while baseline-relative deltas are derived at read time; also promoted the old requential stub into a real online evaluate-then-update estimator.
- Changed: enriched epiplexity metadata with estimator provenance, compute normalization, explicit default-selector helpers, and additive JSONL overlay helpers, then taught `DataPackRepo` to auto-merge `epiplexity_overlays.jsonl` on load.
- Changed: wired `scripts/run_epiplexity_curated_slices.py` to persist canonical overlays in both full and token-only modes, updated `DatapackEngine` and homeostasis consumers to respect the datapack default selector instead of the baseline slot, and fixed `src/evaluation/probe_harness.py` to report true baseline/after aggregates.
- Changed: extended shadow/inferential advisory paths to join epiplexity overlays by datapack id, so replay-side signal-yield and budget decisions consume real epiplexity evidence whenever overlays are available.
- Verification: `python3 -m compileall src scripts -q` and `python3 -m pytest -q tests/epiplexity/test_epiplexity_tracker.py tests/epiplexity/test_epiplexity_metadata.py tests/epiplexity/test_curated_slices_token_only.py tests/epiplexity/test_curated_slices_portable.py tests/epiplexity/test_epiplexity_sampling_orchestrator.py tests/representation/test_homeostasis.py tests/test_probe_harness.py tests/test_receipt_ingest.py tests/test_shadow_advisory_pass.py tests/test_inferential_training_gate.py tests/test_econ_regal_sampling.py tests/test_online_queue_dispatch_integration.py tests/test_queue_dispatch_integration.py`.
- Blocked: no remaining code-path blocker; replay/shadow still falls back only when no matching datapack overlay exists for a given episode.
- Next recommended task: thread real epiplexity overlays into more replay-building entrypoints (not just curated-slice outputs) so synthetic and import-only replay datasets pick up learnability evidence without an extra join step.

- Changed: checked in `docs/economic_world_model/ewm-nightly.automation.toml` as a Git-tracked mirror of the live Codex app automation so the active prompt, RRULE, execution environment, and workspace roots are no longer app-local only.
- Changed: updated `docs/economic_world_model/AUTOMATION_SPEC.md` so the checked-in automation snapshot is documented as the source of truth to keep aligned with the live app automation config.
- Verification: `git diff --check`.
- Blocked: none.
- Next recommended task: add replay import glue that rehydrates bridge-exported sidecar refs into canonical replay metadata/provenance so RLDS/LeRobot roundtrips remain loss-bounded but trace-complete.

- Changed: added `docs/economic_world_model/self_improvement_preconditions_sweep.md`, a repo-wide sweep that separates modules that should remain advisory from modules that should graduate into training-eligibility, work-order, promotion, and replay-roundtrip preconditions.
- Changed: the sweep identifies queue dispatch as the existing bounded-influence template, then pinpoints the next concrete promotion seams in adaptation budgeting, promotion reporting, replay import/rehydration, governed-video admission, semantic-fusion failure handling, and teacher/SceneTracks grounding classification.
- Verification: `git diff --check`.
- Blocked: this pass is analysis/documentation only; no replay import glue, work-order substrate, or new executor wiring landed yet.
- Next recommended task: implement replay import glue plus trace-completeness rehydration in `src/replay/` and `src/dataset_bridges/` so preserved sidecar refs stop being export-only.

- Changed: added a typed execution-precondition/work-order substrate in `src/evidence/preconditions.py` and replay-specific trace-completeness synthesis in `src/replay/preconditions.py`, then wired `src/replay/dataset.py` to persist per-episode readiness plus manifest-level `execution_precondition_summary`.
- Changed: replay roundtrip is now real instead of export-only: `src/dataset_bridges/rlds_bridge.py` and `src/dataset_bridges/lerobot_bridge.py` can rehydrate canonical replay rows, while `ReplayDatasetBuilder` now accepts RLDS and LeRobot imports and emits trace-complete windows/readiness metadata for them.
- Changed: adaptation budgeting and shadow advisory now emit executable work-order artifacts (`adaptation_training`, `data_collection`, `human_review`) when inferential decisions meet explicit replay preconditions, and queue dispatch now preserves that readiness/work-order evidence in bounded live-influence metadata.
- Changed: promotion reporting now incorporates replay trace-readiness/work-order coverage, governed-video now emits proposal-admission work orders plus a stable ledger path, semantic fusion writes degraded-evidence artifacts instead of silently dropping failures, and teacher/SceneTracks paths now expose non-advisory eligibility classes through execution-precondition metadata.
- Changed: the weak execution substrates are now threaded into higher shells without granting them sovereignty: datapack/replay eligibility can be blocked by execution preconditions, and `semantic_orchestrator_v2`, `pipeline_manager`, `phase_h/controller.py`, and `phase_h/economic_learner.py` now surface precondition summaries as advisory routing/repair context.
- Verification: `python3 -m compileall src`, `git diff --check`, `python3 -m pytest -q tests/test_dataset_bridges.py tests/test_replay_dataset.py tests/test_inferential_training_gate.py tests/test_receipt_ingest.py tests/test_promotion_reporting.py tests/test_online_promotion_reporting.py tests/test_stage1_pipeline_governed.py tests/test_semantic_fusion_emit_flag.py tests/test_semantic_fusion_orchestrator_smoke.py tests/test_teacher_runtime.py tests/test_unified_quality_policy_backward_compat.py`, and `python3 -m pytest -q tests/test_queue_dispatch_integration.py tests/test_online_queue_dispatch_integration.py tests/integration/test_scene_tracks_from_workcell_datapack.py tests/smoke_tests/test_phase_h_advisory_integration.py tests/smoke_tests/test_economic_learner_bounds.py`.
- Blocked: none in code; the only observed warning was the existing PyTorch nested-tensor warning during the Stage 1 pipeline tests, and SceneTracks still reports `training_eligible=False` by default when stub adapters are in use, which is the intended new gating behavior.
- Next recommended task: add importer-side replay adapters for governed-video admission logs and semantic degraded-evidence artifacts so those new work orders become first-class training/review inputs in the same canonical replay substrate.

- Changed: added `scripts/SHELL_ACTIVATION_BACKLOG.json` plus `src/orchestrator/shell_activation.py` as a typed higher-shell promotion backlog keyed to execution-precondition summaries, separating auto-activating present-tense shell promotions from future-training-only sovereignty candidates.
- Changed: `semantic_orchestrator_v2`, `pipeline_manager`, `phase_h/advisory_integration.py`, `phase_h/controller.py`, and `phase_h/economic_learner.py` now emit bounded activation plans and typed shell work orders once the new backlog evaluator marks their current-mode promotion ready, while still surfacing future-training backlog items as explicitly pending.
- Changed: documented the new boundary in `docs/economic_world_model/shell_activation_backlog.md`, with the JSON backlog as the machine-readable source of truth for when higher shells may stop being advisory.
- Verification: `python3 -m compileall src tests -q`, `python3 -m pytest -q tests/test_shell_activation.py tests/smoke_tests/test_phase_h_advisory_integration.py tests/smoke_tests/test_economic_learner_bounds.py`, `python3 -m ruff check src/evidence/preconditions.py src/orchestrator/shell_activation.py src/orchestrator/semantic_orchestrator_v2.py src/orchestrator/pipeline_manager.py src/phase_h/advisory_integration.py src/phase_h/controller.py src/phase_h/economic_learner.py tests/test_shell_activation.py tests/smoke_tests/test_phase_h_advisory_integration.py tests/smoke_tests/test_economic_learner_bounds.py`, and `git diff --check`.
- Blocked: future-training-only backlog items remain intentionally pending because current readiness summaries still do not assert real non-stub SceneTracks grounding, live teacher-runtime grounding, replay-roundtrip completion for training manifests, promotion-trace completeness, or budget-settlement evidence.
- Next recommended task: promote governed-video admission logs and degraded semantic-fusion artifacts into importer-side replay adapters, then start satisfying the new future-training backlog checks with explicit `signal_bool::*` and `artifact::*` readiness reports rather than latent assumptions.

## 2026-03-24

- Changed: added importer-side replay adapters in `src/replay/importers.py` plus `ReplayDatasetBuilder` entrypoints in `src/replay/dataset.py` so governed-video admission logs and semantic degraded-evidence artifacts now land in canonical replay instead of remaining sidecar-only outputs.
- Changed: extended `src/evidence/preconditions.py` and `src/replay/preconditions.py` so replay/import paths can carry soft `signal_bool::*` and `artifact::*` future-training checks without turning them into present-tense hard blockers; Stage 1 governed admission logs and semantic degraded artifacts now emit explicit `future_training_signals` and `future_training_artifacts`.
- Changed: tightened importer semantics so replay-owned readiness facts are recomputed on ingress rather than copied blindly from upstream sidecars; in particular, replay roundtrip completion now flips true only after import, while source-computed grounding signals such as `scene_tracks_non_stub` remain preservable.
- Changed: updated `scripts/economic_world_model/nightly_audit.py` and `tests/test_economic_world_model_nightly_audit.py` so the nightly selector now recommends `future_training_evidence_wiring` instead of falling back to `audit_only` when shell-activation backlog items still lack explicit training manifests, promotion ledgers, or budget-settlement evidence.
- Verification: `python3 -m compileall src scripts/economic_world_model tests -q`, `python3 -m ruff check src/evidence/preconditions.py src/replay/preconditions.py src/replay/importers.py src/replay/dataset.py src/replay/__init__.py src/orchestrator/semantic_fusion_runner.py scripts/run_stage1_pipeline.py scripts/economic_world_model/nightly_audit.py tests/test_replay_dataset.py tests/test_stage1_pipeline_governed.py tests/test_semantic_fusion_emit_flag.py tests/test_economic_world_model_nightly_audit.py`, `python3 -m pytest -q tests/test_replay_dataset.py tests/test_stage1_pipeline_governed.py tests/test_semantic_fusion_emit_flag.py tests/test_economic_world_model_nightly_audit.py`, `python3 scripts/economic_world_model/nightly_audit.py --output-json artifacts/economic_world_model/nightly_audit_summary.json --output-markdown artifacts/economic_world_model/nightly_audit_summary.md`, and `git diff --check`.
- Blocked: future-training backlog items are still correctly pending because no live path yet emits `promotion_ledger_ref` or concrete `budget_settlement_live` evidence into replay/training readiness, even though the importer/readiness substrate can now carry them.
- Next recommended task: wire training runtime manifests, promotion ledgers, and settlement evidence through `src/replay/receipt_ingest.py` and `src/training/regal_training_runner.py` so shell-activation future-training gates can flip on explicit artifacts instead of assumptions.
- Changed: `src/training/regal_training_runner.py` now emits normalized `promotion_ledger_v1.json` and `budget_settlement_v1.json` artifacts during canonical training finalization, writes them into the unified training runtime manifest, and records explicit `budget_settlement_live` state rather than leaving promotion/settlement evidence implicit in ad hoc sidecars.
- Changed: `src/training/training_manifest.py` now carries typed `promotion_ledger_*` and `budget_settlement_*` fields, and `src/replay/receipt_ingest.py` now rehydrates those artifacts back into replay-style `future_training_artifacts`, `future_training_signals`, and recomputed execution-precondition summaries when loading a training run.
- Changed: training-run receipt ingestion now upgrades loaded replay bundles in-memory with `training_runtime_manifest`, `promotion_ledger_ref`, and `budget_settlement_live` evidence before building receipt labels, so completed training runs finally surface the exact `artifact::training_runtime_manifest`, `artifact::promotion_ledger_ref`, and `signal_bool::budget_settlement_live` counts the higher-shell future-training backlog expects.
- Verification: `python3 -m compileall src scripts/economic_world_model tests -q`, `python3 -m ruff check src/training/training_manifest.py src/training/regal_training_runner.py src/replay/receipt_ingest.py scripts/economic_world_model/nightly_audit.py tests/test_regal_training_runner.py tests/test_training_run_receipt_ingest.py tests/test_online_promotion_reporting.py tests/test_training_manifest.py tests/test_economic_world_model_nightly_audit.py`, `python3 -m pytest -q tests/test_regal_training_runner.py tests/test_training_run_receipt_ingest.py tests/test_online_promotion_reporting.py tests/test_training_manifest.py tests/test_economic_world_model_nightly_audit.py`, `python3 scripts/economic_world_model/nightly_audit.py --output-json artifacts/economic_world_model/nightly_audit_summary.json --output-markdown artifacts/economic_world_model/nightly_audit_summary.md`, and `git diff --check`.
- Blocked: no additive code-path blocker remains; future-training shell promotions are now gated by whether real runs actually produce positive grounding/promotion/settlement evidence, not by missing wiring.
- Next recommended task: no new safe additive task is currently missing; the nightly audit now returns `audit_only`, so the next meaningful move is to drive real training runs through the new artifacts and observe which future-training backlog predicates still stay false in practice.

- Changed: added a semantic coverage/cybernetics substrate across `src/world_model/semantic_coverage_graph.py`, `src/world_model/coverage_evidence_harvester.py`, `src/orchestrator/coverage_loop.py`, `src/hrl/skill_graph.py`, and `src/envs/primitive_inventory.py`, so the repo now has a typed task × skill × env-primitive coverage graph built from runtime evidence instead of only semantic WM snapshots and advisory tags.
- Changed: added append-only fill-outcome learning data plus learned gap/fill models in `src/world_model/fill_outcome_store.py`, `src/world_model/gap_ranker.py`, `src/world_model/fill_path_policy.py`, and `src/world_model/semantic_state_encoder.py`, together with standalone trainers in `scripts/train_gap_ranker.py` and `scripts/train_fill_path_policy.py`, so the semantic runtime loop can accumulate counterfactual supervision before any full learned controller run exists.
- Changed: wired the new coverage loop into downstream agenda builders: `src/orchestrator/diffusion_requests.py` now compiles gap-driven diffusion prompts, `src/orchestrator/semantic_simulation.py` now compiles ranked simulation agendas from missing edges, `src/orchestrator/pipeline_manager.py` can emit `semantic_coverage` sidecars, `scripts/collect_local_synthetic_branches.py` can bias branch harvesting by coverage deficits, and `scripts/train_latent_diffusion.py` now accepts semantic conditioning sidecars for latent dynamics training.
- Changed: added additive runtime guards and signal adapters around the loop, including `src/process_reward/evidence_adapter.py`, `src/evidence/backend_health.py`, and `src/governance/assessment.py`, so process-reward quality, degraded backend state, and governance coverage can flow into readiness and later into the coverage graph rather than remaining isolated side signals.
- Verification: `python3 -m compileall src scripts/run_coverage_loop.py scripts/train_gap_ranker.py scripts/train_fill_path_policy.py -q`, `python3 -m ruff check src/world_model/semantic_coverage_graph.py src/world_model/coverage_evidence_harvester.py src/world_model/fill_outcome_store.py src/world_model/gap_ranker.py src/world_model/fill_path_policy.py src/world_model/semantic_state_encoder.py src/orchestrator/coverage_loop.py src/orchestrator/diffusion_requests.py src/orchestrator/semantic_simulation.py src/orchestrator/pipeline_manager.py src/hrl/skill_graph.py src/envs/primitive_inventory.py src/process_reward/evidence_adapter.py src/evidence/backend_health.py src/governance/assessment.py scripts/run_coverage_loop.py scripts/train_gap_ranker.py scripts/train_fill_path_policy.py tests/test_semantic_coverage_graph.py tests/test_skill_graph.py tests/test_primitive_inventory.py tests/test_coverage_evidence_harvester.py tests/test_fill_outcome_store.py tests/test_gap_ranker.py tests/test_fill_path_policy.py tests/test_coverage_compilation.py tests/test_coverage_loop.py tests/test_semantic_state_encoder.py tests/test_backend_health.py tests/test_governance_assessment.py`, `python3 -m pytest -q tests/test_semantic_coverage_graph.py tests/test_skill_graph.py tests/test_primitive_inventory.py tests/test_coverage_evidence_harvester.py tests/test_fill_outcome_store.py tests/test_gap_ranker.py tests/test_fill_path_policy.py tests/test_coverage_compilation.py tests/test_coverage_loop.py tests/test_semantic_state_encoder.py tests/test_backend_health.py tests/test_governance_assessment.py`, and `git diff --check`.
- Blocked: the new loop still logs and trains on coverage/fill outcomes rather than fully routing process-reward, trust recalibration, or ontology expansion back into upstream graph topology; the substrate is now present, but not all cybernetic edges are closed yet.
- Next recommended task: route outcome attribution, WM self-correction, trust updates, and ontology-expansion proposals through meta-node-governed transformer packets so the coverage graph becomes a learned control-plane substrate rather than a typed but mostly feed-forward agenda compiler.
- Changed: closed the first runtime return path across `src/world_model/semantic_feedback_packets.py`, `src/orchestrator/coverage_loop.py`, `src/orchestrator/semantic_transformer_bridge.py`, `src/orchestrator/meta_transformer.py`, `src/orchestrator/orchestration_transformer.py`, and `src/orchestrator/pipeline_manager.py`, so coverage outcomes, WM validation packets, governance blocks, graph-mutation proposals, and trust/econ calibration overlays are now compiled once, attached to coverage edges, surfaced in `semantic_coverage`, and injected into both transformer callouts as bounded execution metadata.
- Changed: upgraded synth/world-model training consumers so semantic conditioning is no longer sidecar-only. `scripts/train_latent_diffusion.py` now actually feeds semantic conditioning vectors into both the MLP and transformer latent models, `scripts/train_trust_aware_world_model.py` carries that conditioning through trust-aware reconstruction and rollout loss, `scripts/train_world_model_from_datapacks.py` now appends semantic-gap/process-reward/coverage features plus additive semantic-gap weighting, and checkpoint consumers in `scripts/sample_zv_rollouts.py`, `scripts/eval_world_model_rollouts.py`, and `scripts/train_horizon_agnostic_world_model.py` can reopen semantic-conditioned latent checkpoints without shape mismatch.
- Changed: tightened agenda compilation so governance-blocked edges no longer quietly re-enter sim/diffusion planning; `src/orchestrator/semantic_simulation.py` and `src/orchestrator/diffusion_requests.py` now skip blocked gaps and carry WM-validation pressure through rationale fields instead of flattening it away.
- Verification: `python3 -m compileall src/orchestrator src/world_model scripts/train_latent_diffusion.py scripts/train_trust_aware_world_model.py scripts/train_world_model_from_datapacks.py scripts/sample_zv_rollouts.py scripts/eval_world_model_rollouts.py scripts/train_horizon_agnostic_world_model.py tests/test_semantic_feedback_packets.py tests/test_coverage_loop.py tests/test_semantic_coverage_graph.py tests/test_semantic_transformer_execution.py tests/test_semantic_synth_training.py -q`, `python3 -m ruff check src/world_model/semantic_feedback_packets.py src/world_model/semantic_coverage_graph.py src/orchestrator/coverage_loop.py src/orchestrator/semantic_transformer_bridge.py src/orchestrator/meta_transformer.py src/orchestrator/orchestration_transformer.py src/orchestrator/pipeline_manager.py src/orchestrator/semantic_runtime_learning.py src/orchestrator/semantic_simulation.py src/orchestrator/diffusion_requests.py scripts/train_latent_diffusion.py scripts/train_trust_aware_world_model.py scripts/train_world_model_from_datapacks.py scripts/sample_zv_rollouts.py scripts/eval_world_model_rollouts.py scripts/train_horizon_agnostic_world_model.py tests/test_semantic_feedback_packets.py tests/test_coverage_loop.py tests/test_semantic_coverage_graph.py tests/test_semantic_transformer_execution.py tests/test_semantic_synth_training.py`, `python3 -m pytest -q tests/test_semantic_feedback_packets.py tests/test_coverage_loop.py tests/test_semantic_coverage_graph.py tests/test_semantic_transformer_execution.py tests/test_semantic_synth_training.py`, and `git diff --check`.
- Blocked: the return path is now packetized, but the trust/econ overlays are still summary overlays rather than learned topological adapters over the econ tensor/meta-node lattice, and ontology mutation still stays as bounded proposals rather than fully promoted graph writes.
- Next recommended task: add learned trust/econ overlay trainers over the new feedback packets and a governed graph-mutation executor so runtime packets can promote from bounded review into calibrated online updates without bypassing the frozen Phase B floor.
- Changed: closed that remaining semantic cybernetics gap across `src/world_model/semantic_wm_correction.py`, `src/world_model/graph_mutation_executor.py`, `src/world_model/feedback_topology_adapters.py`, `src/orchestrator/coverage_loop.py`, and `src/orchestrator/pipeline_manager.py`. WM validation packets now compile into an additive correction overlay and corrected semantic WM, graph-mutation proposals now flow through a governed executor before coverage-graph construction, and the coverage loop now optionally shadow-fits a learned trust/econ/readiness/correction adapter over real edge outcomes instead of relying only on scalar heuristics.
- Changed: the runtime outputs now carry those promoted layers end to end. `semantic_coverage` now includes `graph_mutation_execution`, `semantic_wm_correction_overlay`, and `corrected_semantic_world_model`, while `semantic_transformer_bridge`, `meta_transformer`, `orchestration_transformer`, and `semantic_runtime_learning` all see graph-mutation execution counts, WM-correction pressure, and corrected semantic WM state as part of the same bounded control-plane packet.
- Changed: added a heavyweight trainer path for the new overlay layer in `scripts/train_semantic_feedback_adapters.py`, so the learned trust/econ/readiness/correction overlay no longer depends only on in-memory shadow fitting and can be promoted into persisted checkpoints when enough coverage-loop artifacts exist.
- Verification: `python3 -m compileall src/world_model src/orchestrator scripts/train_semantic_feedback_adapters.py tests/test_semantic_gap_closure.py tests/test_coverage_loop.py -q`, `python3 -m ruff check src/world_model/semantic_wm_correction.py src/world_model/graph_mutation_executor.py src/world_model/feedback_topology_adapters.py src/orchestrator/coverage_loop.py src/orchestrator/pipeline_manager.py src/orchestrator/semantic_transformer_bridge.py src/orchestrator/semantic_runtime_learning.py scripts/train_semantic_feedback_adapters.py tests/test_semantic_gap_closure.py tests/test_coverage_loop.py`, `python3 -m pytest -q tests/test_semantic_gap_closure.py tests/test_coverage_loop.py tests/test_semantic_feedback_packets.py tests/test_semantic_transformer_execution.py`, and `git diff --check`.
- Blocked: no additive repo-owned semantic cybernetics blocker remains in this lane; the remaining uncertainty is now data/evidence quality and whether future training artifacts provide enough real coverage-loop examples to move from shadow-fit overlays to persisted learned adapters in practice.
- Next recommended task: drive real coverage-loop artifact accumulation and train the new feedback adapter package on actual runtime histories, then decide which governed graph-mutation actions are ready to promote from provisional overlay into stronger online authority.
- Changed: added `src/world_model/semantic_wm_refiner.py` as the learned successor/refiner over the deterministic semantic WM instead of mutating the frozen builder. It predicts bounded correction overlays and learned graph-mutation proposal scores, then merges those outputs back into the existing governed correction/mutation path rather than bypassing it.
- Changed: `src/orchestrator/coverage_loop.py` and `src/orchestrator/pipeline_manager.py` now support `semantic_wm_refiner_package` / `semantic_wm_refiner_checkpoint` plus shadow-fit fallback, emit `input_semantic_world_model.json` and `semantic_wm_refiner_summary.json`, and expose the learned successor layer through `semantic_coverage` metadata and runtime-learning summaries.
- Changed: added `scripts/train_semantic_wm_refiner.py` as the heavyweight trainer for persisted semantic-WM successor checkpoints, and recorded it as a migrated training entry in `scripts/TRAINING_MIGRATION_BACKLOG.json`.
- Verification: `python3 -m compileall src/world_model src/orchestrator scripts/train_semantic_wm_refiner.py tests/test_semantic_wm_refiner.py tests/test_coverage_loop.py -q`, `python3 -m ruff check src/world_model/semantic_wm_refiner.py src/orchestrator/coverage_loop.py src/orchestrator/pipeline_manager.py src/orchestrator/semantic_runtime_learning.py scripts/train_semantic_wm_refiner.py tests/test_semantic_wm_refiner.py tests/test_coverage_loop.py`, `python3 -m pytest -q tests/test_semantic_wm_refiner.py tests/test_coverage_loop.py`, and `git diff --check`.
- Blocked: the remaining uncertainty is no longer the absence of a learned semantic-WM successor. It is whether accumulated runtime histories are dense and stable enough to promote persisted refiner checkpoints over repeated shadow-fit overfitting on sparse evidence.

## 2026-03-26

- Changed: removed the fake D4 learned knob boundary. `src/regal/knob_model.py` no longer exposes a stub-learned provider; it now resolves a real runtime package/checkpoint through `src/regal/knob_model_runtime.py`, with explicit `required` semantics for benchmark-gated use.
- Changed: added `src/regal/knob_model_training.py`, `src/regal/knob_model_runtime.py`, and `scripts/train_knob_model.py` as the canonical trainer/runtime substrate for bounded knob calibration. The new trainer emits dataset/precondition/training/package artifacts plus unified runtime manifest/checkpoint registry outputs under `RegalTrainingRunner`.
- Changed: wired the homeostatic planner to preserve actual knob-model training context. `src/orchestrator/policy_hooks.py` now builds real exposure/datapack/objective regime features, and `src/orchestrator/homeostatic_plan_writer.py` now records `knob_policy`, `knob_policy_used`, `knob_regime_features`, and `knob_base_config` in `GateStatus`.
- Changed: `scripts/run_closed_loop_smoke.py` now supports explicit knob package loading (`--knob-model-path`, `--require-learned-knobs`) and emits `knob_policy_receipt.json`, so the D4 lane now has a real runtime receipt substrate for future training instead of a fake learned label.
- Changed: updated the heuristic/advisory/sidecar inventory plus runtime/training backlogs so the next remaining mandate-level gaps are now accurately tracked as higher-order orchestrator shell policy, queue/curriculum weighting, real-SAM3D grounded-data refresh, and other data-limited full-loop trainers rather than the knob lane.
- Verification: `python3 -m compileall src/regal/knob_model.py src/regal/knob_model_runtime.py src/regal/knob_model_training.py src/orchestrator/policy_hooks.py src/orchestrator/homeostatic_plan_writer.py scripts/train_knob_model.py scripts/run_closed_loop_smoke.py tests/test_knob_model_clamping.py tests/test_knob_model_runtime.py tests/test_train_knob_model.py -q`, `python3 -m ruff check src/regal/knob_model.py src/regal/knob_model_runtime.py src/regal/knob_model_training.py src/orchestrator/policy_hooks.py src/orchestrator/homeostatic_plan_writer.py scripts/train_knob_model.py scripts/run_closed_loop_smoke.py tests/test_knob_model_clamping.py tests/test_knob_model_runtime.py tests/test_train_knob_model.py`, `python3 -m pytest -q tests/test_knob_model_clamping.py tests/test_knob_model_runtime.py tests/test_train_knob_model.py tests/test_plan_policy.py`, `python3 scripts/check_training_regality.py --scripts-dir scripts`, and `git diff --check`.
- Blocked: the remaining production blocker in this lane is no longer missing runtime/training plumbing. It is receipt density and grounded-data reality: the knob helper is real but should stay benchmark-gated until it accumulates enough runtime receipts, and broader workcell grounding is still blocked on real GPU + SAM3D execution.

- Changed: added `src/orchestrator/queue_dispatch_policy.py`, `src/orchestrator/queue_dispatch_policy_training.py`, `src/orchestrator/queue_dispatch_policy_runtime.py`, and `scripts/train_queue_dispatch_policy.py` as the canonical queue-dispatch trainer/runtime substrate. The new lane learns bounded dispatch desirability over live queue entries, emits canonical dataset/precondition/training/package artifacts, and registers checkpoints/runtime outputs through `RegalTrainingRunner`.
- Changed: wired the learned queue helper into the real replay/training loop. `src/orchestrator/queue_selection.py` now resolves queue-policy helpers with `disabled|auto|required` semantics, blends learned dispatch scores against the explicit heuristic multiplier prior, and preserves queue-policy traces on live decisions; `src/rl/episode_sampling.py` plus the main shadow/online training entrypoints now pass the helper into actual sampling/dispatch runtime.
- Changed: the queue seam is now honest instead of over-claimed. Learned selectors/runtime scorers can materially affect replay weighting through the queue, but the underlying sampler base-weight and curriculum-strategy core in `src/rl/episode_sampling.py` remains explicitly heuristic and stays in the backlog as the next queue/curriculum tranche.
- Verification: `python3 -m compileall src/policies/meta_advisor.py src/orchestrator/queue_dispatch_policy.py src/orchestrator/queue_dispatch_policy_training.py src/orchestrator/queue_dispatch_policy_runtime.py src/orchestrator/queue_selection.py src/rl/episode_sampling.py scripts/train_queue_dispatch_policy.py scripts/train_shadow_replay_policy.py scripts/train_shadow_offline_rl.py scripts/train_shadow_pricing_models.py scripts/train_sac_with_ontology_logging.py tests/test_queue_dispatch_policy.py tests/test_train_queue_dispatch_policy.py -q`, `python3 -m ruff check src/policies/meta_advisor.py src/orchestrator/queue_dispatch_policy.py src/orchestrator/queue_dispatch_policy_training.py src/orchestrator/queue_dispatch_policy_runtime.py src/orchestrator/queue_selection.py src/rl/episode_sampling.py scripts/train_queue_dispatch_policy.py scripts/train_shadow_replay_policy.py scripts/train_shadow_offline_rl.py scripts/train_shadow_pricing_models.py scripts/train_sac_with_ontology_logging.py tests/test_queue_dispatch_policy.py tests/test_train_queue_dispatch_policy.py`, `python3 -m pytest -q tests/test_queue_dispatch_policy.py tests/test_train_queue_dispatch_policy.py tests/test_queue_dispatch_integration.py tests/test_online_queue_dispatch_integration.py`, `python3 scripts/check_training_regality.py --scripts-dir scripts`, and `git diff --check`.
- Blocked: this does not yet replace the sampler’s frontier/econ/curriculum base-weight heuristics. Full queue/curriculum neuralization still needs denser queue-outcome receipts and replay counterfactual labels below the now-real queue-dispatch helper layer.

- Changed: added `src/rl/sampler_policy.py`, `src/rl/sampler_policy_training.py`, `src/rl/sampler_policy_runtime.py`, and `scripts/train_sampler_policy.py` as the canonical sampler-policy trainer/runtime substrate. The new lane learns bounded strategy distributions, frontier/econ plan parameters, and strategy-conditioned per-episode base-weight targets from `sampler_policy_receipt_v1` artifacts under `RegalTrainingRunner`.
- Changed: wired the sampler helper into the actual replay/training loop. `src/rl/episode_sampling.py` now resolves bounded sampler-policy helpers with `disabled|auto|required` semantics, blends learned strategy/weight/plan outputs against explicit heuristic priors, preserves sampler strategy and weight traces in `sampling_metadata`, and emits `sampler_policy_receipt_v1` artifacts; the main shadow/online training entrypoints now persist `sampler_policy_receipt.json` beside queue-dispatch outputs.
- Changed: the remaining queue/curriculum truth boundary is now narrowed to receipt-density/promotion rather than missing runtime wiring. The helper is real and materially affects training distribution, but it should stay benchmark-gated until real queue outcome receipts and replay counterfactual labels are denser than the current heuristic-bootstrap substrate.
- Verification: `python3 -m compileall src/rl/sampler_policy.py src/rl/sampler_policy_training.py src/rl/sampler_policy_runtime.py src/rl/episode_sampling.py scripts/train_sampler_policy.py scripts/train_shadow_replay_policy.py scripts/train_shadow_offline_rl.py scripts/train_shadow_pricing_models.py scripts/train_sac_with_ontology_logging.py tests/test_sampler_policy.py tests/test_train_sampler_policy.py -q`, `python3 -m ruff check src/rl/sampler_policy.py src/rl/sampler_policy_training.py src/rl/sampler_policy_runtime.py src/rl/episode_sampling.py scripts/train_sampler_policy.py scripts/train_shadow_replay_policy.py scripts/train_shadow_offline_rl.py scripts/train_shadow_pricing_models.py scripts/train_sac_with_ontology_logging.py tests/test_sampler_policy.py tests/test_train_sampler_policy.py`, `python3 -m pytest -q tests/test_sampler_policy.py tests/test_train_sampler_policy.py tests/test_sampling_determinism_seeded.py tests/test_queue_dispatch_policy.py tests/test_train_queue_dispatch_policy.py tests/test_queue_dispatch_integration.py tests/test_online_queue_dispatch_integration.py`, `python3 scripts/check_training_regality.py --scripts-dir scripts`, and `git diff --check`.
- Blocked: promotion in this lane is now honestly blocked by data, not missing substrate. The sampler helper still trains mostly on heuristic-bootstrap targets because the repo does not yet have dense enough queue outcome receipts and replay counterfactual labels to promote it beyond `shadow_candidate`.

- Changed: canonicalized the semantic runtime scorer lane. `scripts/train_semantic_runtime_scorers.py` now emits dataset/precondition/model/training/runtime-package artifacts under `RegalTrainingRunner`, preserves the legacy linear scorer package as a stable runtime contract, and `src/orchestrator/shadow_advisory.py` now prefers `semantic_runtime_scorer_runtime_package.json` while recording contract type, promotion stage, and benchmark-gate truth instead of treating any scorer JSON as equally production-ready.
- Changed: canonicalized the learned coverage helpers. `scripts/train_semantic_feedback_adapters.py` and `scripts/train_semantic_wm_refiner.py` now emit canonical dataset/precondition/model/training/runtime-package artifacts under `RegalTrainingRunner`; `src/world_model/feedback_topology_runtime.py` and `src/world_model/semantic_wm_refiner_runtime.py` add bounded `disabled|auto|required` runtime loading; and `src/orchestrator/coverage_loop.py` / `src/orchestrator/pipeline_manager.py` now consume those runtime packages directly instead of raw checkpoint blobs or implicit shadow-fit-only behavior.
- Changed: narrowed the remaining non-GPU mandate blockers to density/promotion rather than missing plumbing. The semantic runtime scorer, semantic feedback adapter, and semantic WM refiner are now real helper lanes in the live loop, but they stay benchmark-gated until replay and coverage-artifact density rises beyond the current shadow/bootstrap substrate.
- Verification: `python3 -m compileall scripts/train_semantic_runtime_scorers.py src/orchestrator/semantic_runtime_scorer_runtime.py src/orchestrator/shadow_advisory.py tests/test_train_semantic_runtime_scorers.py tests/test_receipt_ingest.py scripts/train_semantic_feedback_adapters.py scripts/train_semantic_wm_refiner.py src/world_model/feedback_topology_adapters.py src/world_model/feedback_topology_runtime.py src/world_model/semantic_wm_refiner_runtime.py src/orchestrator/coverage_loop.py src/orchestrator/pipeline_manager.py tests/test_train_semantic_feedback_adapters.py tests/test_train_semantic_wm_refiner.py -q`, `python3 -m ruff check scripts/train_semantic_runtime_scorers.py src/orchestrator/semantic_runtime_scorer_runtime.py src/orchestrator/shadow_advisory.py tests/test_train_semantic_runtime_scorers.py tests/test_receipt_ingest.py scripts/train_semantic_feedback_adapters.py scripts/train_semantic_wm_refiner.py src/world_model/feedback_topology_adapters.py src/world_model/feedback_topology_runtime.py src/world_model/semantic_wm_refiner_runtime.py src/orchestrator/coverage_loop.py src/orchestrator/pipeline_manager.py tests/test_train_semantic_feedback_adapters.py tests/test_train_semantic_wm_refiner.py`, `python3 -m pytest -q tests/test_train_semantic_runtime_scorers.py tests/test_receipt_ingest.py tests/test_train_semantic_feedback_adapters.py tests/test_train_semantic_wm_refiner.py tests/test_coverage_loop.py tests/test_semantic_gap_closure.py tests/test_semantic_wm_refiner.py`, and `git diff --check`.
- Blocked: the honest remainder here is data, not missing runtime wiring. Semantic runtime scorer promotion still needs execution-ready / semantic-grounded replay density; feedback adapter and WM-refiner promotion still need repeated coverage-loop artifacts; real grounded vision promotion still needs GPU + SAM3D.

- Changed: hardened the new sim/synth helper runtime-package seams. `scripts/train_sim_synth_backend_selector.py` and `scripts/train_sim_synth_branch_planner.py` now emit relocatable package refs instead of baking in absolute artifact paths, `src/world_model/sim_synth_physics/backend_selector_runtime.py` / `src/world_model/sim_synth_physics/branch_planner_runtime.py` now resolve relative checkpoints against the package location, and loaded helpers now preserve `package_id`, `package_path`, `promotion_stage`, and subsystem metadata all the way into WM inference.
- Changed: added end-to-end package-loading coverage in `tests/test_sim_synth_physics_world_model.py`, proving that canonical sim/synth runtime packages can be reloaded outside their training directory and still drive backend selection and branch planning through the WM boundary.
- Verification: `python3 -m compileall src/world_model/sim_synth_physics scripts/train_sim_synth_backend_selector.py scripts/train_sim_synth_branch_planner.py tests/test_sim_synth_physics_world_model.py -q`, `python3 -m ruff check src/world_model/sim_synth_physics scripts/train_sim_synth_backend_selector.py scripts/train_sim_synth_branch_planner.py tests/test_sim_synth_physics_world_model.py`, `python3 -m pytest -q tests/test_sim_synth_physics_world_model.py tests/test_train_sim_synth_backend_selector.py tests/test_train_sim_synth_branch_planner.py`, and `git diff --check`.
- Blocked: no code-path blocker remains in this helper seam; the honest remainder is still corpus density and eventual benchmark-gated promotion of the helpers.

- Changed: started the advisory-purge follow-through on the live training-distribution surfaces. `src/orchestrator/queue_selection.py` no longer describes queue entries and dispatch receipts as advisory-only; it now emits explicit `authority_class`, `decision_scope`, `reward_math_mutation`, and `receipt_kind` fields for both queue-selection inputs and queue-dispatch outputs.
- Changed: `src/rl/episode_sampling.py` and `src/rl/sac.py` now preserve that bounded-authority classification through sampler-policy receipts, `dispatch_queue(...)`, and online replay sampling artifacts, so queue/curriculum influence stops degrading back into anonymous metadata once it reaches runtime training and replay.
- Verification: `python3 -m compileall src/orchestrator/queue_selection.py src/rl/episode_sampling.py src/rl/sac.py tests/test_econ_regal_sampling.py tests/test_queue_dispatch_integration.py tests/test_online_queue_dispatch_integration.py tests/test_sampler_policy.py tests/test_queue_dispatch_policy.py tests/test_train_sampler_policy.py -q`, `python3 -m ruff check src/orchestrator/queue_selection.py src/rl/episode_sampling.py src/rl/sac.py tests/test_econ_regal_sampling.py tests/test_queue_dispatch_integration.py tests/test_online_queue_dispatch_integration.py tests/test_sampler_policy.py tests/test_queue_dispatch_policy.py tests/test_train_sampler_policy.py`, `python3 -m pytest -q tests/test_econ_regal_sampling.py tests/test_queue_dispatch_integration.py tests/test_online_queue_dispatch_integration.py tests/test_sampler_policy.py tests/test_queue_dispatch_policy.py tests/test_train_sampler_policy.py`, and `git diff --check`.
- Blocked: this narrows the doctrine gap, but it does not yet fully neuralize orchestration-level bounded authority. The remaining advisory cleanup is still above the queue/sampler layer in orchestration and higher-shell control surfaces.

## 2026-03-27

- Changed: continued Phase 1 sim/synth/physics implementation by turning backend execution from a descriptor-only concept into a WM-owned binding surface:
  - added concrete backend-binding modules in `src/world_model/sim_synth_physics/adapters/backend_pybullet.py`, `backend_holosoma.py`, and `backend_isaac.py`
  - added `src/world_model/sim_synth_physics/backend_bindings.py` plus `BackendExecutionBindingState` / `backend_execution_binding_receipt_v1`, so the WM now emits executor entrypoints, observation-adapter entrypoints, runtime stack, asset profiles, and missing-asset truth for the selected backend
  - the Isaac / Unitree-target path still stays honest: it now carries explicit asset readiness for robot description, joint mapping, sensor extrinsics, and actuator-latency profiles instead of pretending that “isaac” is already executable
- Changed: deepened the NAG/LSD/GGDS provider seam from provider-kind selection into materialization configuration:
  - `src/world_model/sim_synth_physics/render_providers.py` now resolves materialization entrypoints, provider config payloads, and `materialization_status` for NAG counterfactual generation and GGDS scene texturing
  - `BranchRenderProviderState` and `render_provider_receipt_v1` now preserve materialization entrypoints and config instead of only provider-kind/status metadata
  - `src/world_model/sim_synth_physics/training_corpus.py` now harvests backend-binding receipts and the richer render-provider receipts so those contracts survive into downstream trainer datasets
- Verification: `python3 -m compileall src/world_model/sim_synth_physics scripts/run_sim_synth_physics_loop.py tests/test_sim_synth_physics_world_model.py tests/test_sim_synth_physics_scripts.py tests/test_sim_synth_training_corpus.py -q`, `python3 -m ruff check src/world_model/sim_synth_physics scripts/run_sim_synth_physics_loop.py tests/test_sim_synth_physics_world_model.py tests/test_sim_synth_physics_scripts.py tests/test_sim_synth_training_corpus.py`, `python3 -m pytest -q tests/test_sim_synth_physics_world_model.py tests/test_sim_synth_physics_scripts.py tests/test_sim_synth_training_corpus.py tests/test_train_sim_synth_backend_selector.py tests/test_train_sim_synth_branch_planner.py tests/test_nag_lsd_integration.py`, and `git diff --check`.
- Blocked: this keeps shrinking the implementable Phase 1 gap list, but the main honest blockers are still concrete external assets/execution rather than missing WM ownership:
  - real Isaac Sim / Isaac Gym execution through the new binding seam
  - Unitree robot assets and calibration sidecars
  - richer Holosoma runtime asset binding on an actual host
  - concrete GGDS/LDM execution on a host with the required rendering stack

- Changed: continued Phase 1 sim/synth/physics WM implementation with typed backend-adapter, physics-adaptation, and branch/render-provider ownership:
  - added `src/world_model/sim_synth_physics/backend_adapters.py` so backend routing now resolves explicit adapter descriptors for PyBullet, Holosoma, and the still-honest Isaac/Unitree target gap instead of treating backend names as flat strings
  - added `src/world_model/sim_synth_physics/randomization.py` and extended `state.py`, `physics_contracts.py`, `receipts.py`, `compiler.py`, `calibration.py`, and `runtime.py` so the WM now compiles `PhysicsAdaptationPolicyState`, emits `physics_adaptation_receipt_v1`, and threads domain-randomization / system-identification / robot-asset targets into live loop artifacts rather than leaving them implicit
  - added `src/world_model/sim_synth_physics/render_providers.py` and extended branch planning / diffusion / runtime receipts so NAG / LSD / GGDS materialization is now routed through WM-owned `BranchRenderProviderState` contracts and `render_provider_receipt_v1` artifacts instead of sitting only as adjacent provider code
  - `src/world_model/sim_synth_physics/training_corpus.py` now harvests the new adaptation/provider receipts and projects them into backend-selector / branch-planner rows, which closes another Phase 1 gap between WM-owned receipts and downstream training datasets
  - `scripts/run_sim_synth_physics_loop.py` and the sim/synth tests now preserve and verify the new receipt surfaces end-to-end
- Changed: updated `docs/economic_world_model/multi_wm_architecture_plan.md` so Phase 1 explicitly includes backend-adapter ownership, physics adaptation policy/receipt ownership, and WM-owned NAG/LSD/GGDS provider contracts before the phase can be considered exhausted apart from GPU/data/asset limits.
- Verification: `python3 -m compileall src/world_model/sim_synth_physics src/orchestrator/diffusion_requests.py scripts/run_sim_synth_physics_loop.py tests/test_sim_synth_physics_world_model.py tests/test_sim_synth_physics_scripts.py tests/test_sim_synth_training_corpus.py -q`, `python3 -m ruff check src/world_model/sim_synth_physics src/orchestrator/diffusion_requests.py scripts/run_sim_synth_physics_loop.py tests/test_sim_synth_physics_world_model.py tests/test_sim_synth_physics_scripts.py tests/test_sim_synth_training_corpus.py`, `python3 -m pytest -q tests/test_sim_synth_physics_world_model.py tests/test_sim_synth_physics_scripts.py tests/test_sim_synth_training_corpus.py`, `python3 -m pytest -q tests/test_train_sim_synth_backend_selector.py tests/test_train_sim_synth_branch_planner.py tests/test_gap_agenda_ranking.py tests/test_coverage_compilation.py`, and `git diff --check`.
- Blocked: Phase 1 is narrower but still not externally blocked yet. The remaining explicit in-phase work is now:
  - real Isaac Sim / Isaac Gym execution adapters and Unitree robot assets behind the new backend contract
  - richer Holosoma execution/runtime asset binding under the same contract
  - concrete GGDS/LDM execution and richer NAG/LSD materialization under the new WM-owned render-provider seam
  - real GPU-backed grounded video state for perception-conditioned sim

- Changed: finished the external-provider doctrine cleanup for the remaining teacher / SceneTracks seams so provider status is now canonical metadata while provider outputs stay advisory:
  - added `src/evidence/provider_truth.py` as the shared `external_provider_truth_v1` contract for provider availability, fallback, calibration class, and grounding class
  - `src/vla/teacher_runtime.py` and `src/evidence/teacher_trace.py` now emit `provider_truth` on teacher contracts, teacher action envelopes, and teacher traces, including backend, fallback, and vision-backbone metadata
  - `src/vision/scene_ir_tracker/io/scene_tracks_runner.py` and `src/evidence/scene_tracks_truth.py` now emit/preserve `scene_tracks_provider_truth`, so passthrough/stub/real grounding class is carried as canonical metadata instead of being reverse-engineered later
  - `src/vla/rollout_labeler.py` and `src/replay/ingest.py` now preserve teacher and SceneTracks provider truth into datapack/replay metadata, which closes the doctrine gap where downstream consumers could see advisory predictions without reliable provider-status truth
- Verification: `python3 -m compileall src/evidence/provider_truth.py src/evidence/teacher_trace.py src/evidence/scene_tracks_truth.py src/vla/teacher_runtime.py src/vla/rollout_labeler.py src/vision/scene_ir_tracker/io/scene_tracks_runner.py src/replay/ingest.py tests/test_teacher_runtime.py tests/test_rollout_labeler.py tests/test_scene_tracks_truth.py tests/test_semantic_fusion_orchestrator_smoke.py tests/test_replay_dataset.py tests/integration/test_scene_tracks_from_workcell_datapack.py -q`, `python3 -m ruff check src/evidence/provider_truth.py src/evidence/teacher_trace.py src/evidence/scene_tracks_truth.py src/vla/teacher_runtime.py src/vla/rollout_labeler.py src/vision/scene_ir_tracker/io/scene_tracks_runner.py src/replay/ingest.py tests/test_teacher_runtime.py tests/test_rollout_labeler.py tests/test_scene_tracks_truth.py tests/test_semantic_fusion_orchestrator_smoke.py tests/test_replay_dataset.py tests/integration/test_scene_tracks_from_workcell_datapack.py`, `python3 -m pytest -q tests/test_teacher_runtime.py tests/test_rollout_labeler.py tests/test_scene_tracks_truth.py tests/test_semantic_fusion_orchestrator_smoke.py tests/test_replay_dataset.py tests/integration/test_scene_tracks_from_workcell_datapack.py`, and `git diff --check`.
- Blocked: this closes the provider-status truth gap, but it does not create real grounded data where the host cannot. Real non-passthrough SceneTracks promotion is still blocked on GPU + SAM3D and grounded-data receipt density.

- Changed: completed the remaining shell-level advisory cleanup for this mandate without prematurely promoting the shells to sovereign control:
  - `src/phase_h/advisory_integration.py`, `src/phase_h/controller.py`, and `src/phase_h/economic_learner.py` now carry explicit shell receipt fields (`receipt_kind`, `authority_class`, `decision_scope`, `reward_math_mutation`) and preserve `input_receipt_context` so Phase H budget/routing shells consume canonical execution/precondition/work-order context rather than only free-form summaries
  - `src/orchestrator/pipeline_manager.py` now builds and propagates typed input receipt context into iteration activation plans, preview reports, and emitted work-order metadata, so the top shell stops flattening canonical receipts back into anonymous advisory state
  - `src/rl/curriculum.py` is now explicitly treated as bounded training-distribution authority instead of “purely advisory”, and `sample_batch(...)` now emits `curriculum_dispatch_receipt_v1`
  - added `tests/test_curriculum.py` and expanded the Phase H / pipeline-shell tests so the new receipt context and authority typing are covered
- Verification: `python3 -m compileall src/phase_h/advisory_integration.py src/phase_h/controller.py src/phase_h/economic_learner.py src/orchestrator/pipeline_manager.py src/rl/curriculum.py tests/test_curriculum.py tests/smoke_tests/test_phase_h_advisory_integration.py tests/smoke_tests/test_economic_learner_bounds.py tests/test_shell_activation.py tests/test_pipeline_stage_policy.py -q`, `python3 -m ruff check src/phase_h/advisory_integration.py src/phase_h/controller.py src/phase_h/economic_learner.py src/orchestrator/pipeline_manager.py src/rl/curriculum.py tests/test_curriculum.py tests/smoke_tests/test_phase_h_advisory_integration.py tests/smoke_tests/test_economic_learner_bounds.py tests/test_shell_activation.py tests/test_pipeline_stage_policy.py`, `python3 -m pytest -q tests/test_curriculum.py tests/smoke_tests/test_phase_h_advisory_integration.py tests/smoke_tests/test_economic_learner_bounds.py tests/test_shell_activation.py tests/test_pipeline_stage_policy.py`, and `git diff --check`.
- Blocked: the higher shells are now more honest about what they consume and emit, but they still remain intentionally non-sovereign. The honest remainder is not contract ambiguity anymore; it is whether later lower-WM receipts and benchmark evidence ever justify promoting any shell beyond bounded planning/work-order generation.

- Changed: promoted inferential admission out of advisory-only summaries and into a canonical emitted contract. `src/economics/inferential_contract.py` now defines `inferential_admission_contract_v1`, `src/economics/inferential_training_gate.py` now emits typed work-order-class decisions, and `src/orchestrator/adaptation_budgeting.py` / `src/orchestrator/shadow_advisory.py` now carry per-episode admission rows plus an aggregate admission contract instead of only adaptation-budget rollups.
- Changed: threaded that contract through the canonical training runtime. `src/training/regal_training_runner.py` and `src/training/training_manifest.py` now preserve `inferential_admission_summary`, while `scripts/train_shadow_replay_policy.py`, `scripts/train_shadow_offline_rl.py`, `scripts/train_shadow_pricing_models.py`, `scripts/train_sac_with_ontology_logging.py`, and `scripts/run_shadow_advisory_pass.py` now emit/register `inferential_admission_contract.json` beside the existing learnability/work-order artifacts.
- Changed: promoted epiplexity-based learnability into datapack-owned canonical metadata. `src/valuation/datapack_schema.py` now carries `inferential_learnability_contract`, `src/valuation/datapack_repo.py` now attaches or preserves that contract when datapacks are loaded and epiplexity overlays are applied, and `src/rl/episode_sampling.py` now preserves the canonical contract in RL descriptors instead of re-deriving learnability purely from local summary fields.
- Verification: `python3 -m compileall src/economics/inferential_contract.py src/economics/inferential_training_gate.py src/orchestrator/adaptation_budgeting.py src/orchestrator/shadow_advisory.py src/training/training_manifest.py src/training/regal_training_runner.py src/valuation/datapack_schema.py src/valuation/datapack_repo.py src/rl/episode_sampling.py scripts/train_shadow_replay_policy.py scripts/train_shadow_offline_rl.py scripts/train_shadow_pricing_models.py scripts/train_sac_with_ontology_logging.py scripts/run_shadow_advisory_pass.py tests/test_inferential_contract.py tests/test_inferential_training_gate.py tests/test_receipt_ingest.py tests/test_training_manifest.py tests/test_shadow_advisory_pass.py tests/epiplexity/test_curated_slices_portable.py -q`, `python3 -m ruff check src/economics/inferential_contract.py src/economics/inferential_training_gate.py src/orchestrator/adaptation_budgeting.py src/orchestrator/shadow_advisory.py src/training/training_manifest.py src/training/regal_training_runner.py src/valuation/datapack_schema.py src/valuation/datapack_repo.py src/rl/episode_sampling.py scripts/train_shadow_replay_policy.py scripts/train_shadow_offline_rl.py scripts/train_shadow_pricing_models.py scripts/train_sac_with_ontology_logging.py scripts/run_shadow_advisory_pass.py tests/test_inferential_contract.py tests/test_inferential_training_gate.py tests/test_receipt_ingest.py tests/test_training_manifest.py tests/test_shadow_advisory_pass.py tests/epiplexity/test_curated_slices_portable.py`, `python3 -m pytest -q tests/test_inferential_contract.py tests/test_inferential_training_gate.py tests/test_receipt_ingest.py tests/test_training_manifest.py tests/test_shadow_advisory_pass.py tests/epiplexity/test_curated_slices_portable.py tests/test_replay_dataset.py`, and `git diff --check`.
- Blocked: the next advisory purge cut is now clearly the internal orchestrator companion receipts: `semantic_fusion_runner`, `runtime_backbone`, Stage-1 pipeline emissions, and replay ingest still need a canonical control-plane context artifact so bounded internal selectors stop falling back to sidecar semantics.
- Changed: closed that next advisory gap with a companion `orchestrator_control_plane_context_v1` artifact. `src/semantic/runtime_backbone.py` now emits a canonical control-plane context beside `semantic_world_model`, `semantic_snapshot`, and `orchestrator_advisory`, carrying meta-node weights, focus presets, benchmark signals, execution preconditions, semantic-runtime truth, and typed authority metadata.
- Changed: threaded the new control-plane context through the main producers and consumers. `scripts/run_stage1_pipeline.py`, `src/orchestrator/semantic_fusion_runner.py`, and `scripts/bootstrap_semantic_workcell_loop.py` now write/preserve `*_control_plane_context_v1.json`, while `src/replay/ingest.py` now discovers `control_plane_context_path`, preserves the artifact ref in provenance, and hydrates the parsed context into replay episode metadata.
- Verification: `python3 -m compileall src/semantic/runtime_backbone.py src/orchestrator/semantic_fusion_runner.py scripts/run_stage1_pipeline.py scripts/bootstrap_semantic_workcell_loop.py src/replay/ingest.py tests/test_semantic_fusion_emit_flag.py tests/test_stage1_pipeline_governed.py tests/integration/test_bootstrap_semantic_workcell_loop.py tests/test_replay_dataset.py tests/test_semantic_world_model_backbone.py -q`, `python3 -m ruff check src/semantic/runtime_backbone.py src/orchestrator/semantic_fusion_runner.py scripts/run_stage1_pipeline.py scripts/bootstrap_semantic_workcell_loop.py src/replay/ingest.py tests/test_semantic_fusion_emit_flag.py tests/test_stage1_pipeline_governed.py tests/integration/test_bootstrap_semantic_workcell_loop.py tests/test_replay_dataset.py tests/test_semantic_world_model_backbone.py`, `python3 -m pytest -q tests/test_semantic_fusion_emit_flag.py tests/test_stage1_pipeline_governed.py tests/integration/test_bootstrap_semantic_workcell_loop.py tests/test_replay_dataset.py tests/test_semantic_world_model_backbone.py`, and `git diff --check`.
- Blocked: the remaining advisory cleanup is now higher than the semantic-runtime companion artifacts. The next likely buckets are external-provider doctrine cleanup and higher-shell orchestration surfaces that still consume preview/advisory blobs instead of canonical receipts.
- Changed: added an explicit anti-regression guardrail for the advisory/receipt doctrine. `scripts/check_canonical_receipt_contracts.py` now scans the main internal control-plane packages and fails when internal receipt emitters carry `receipt_kind` without the full canonical authority fields (`authority_class`, `decision_scope`, `reward_math_mutation`) or when provider-truth surfaces are present without the canonical provider-truth contract path. `scripts/run_full_repo_verification.py` now runs that checker by default unless `--skip-contract-checks` is passed.
- Changed: finished the default-path sim/synth corpus hardening so the backend-selector and branch-planner trainers can build their datasets from live WM receipt directories instead of only from manually pre-bundled receipt exports:
  - `src/world_model/sim_synth_physics/training_corpus.py` now harvests `sim_synth_physics_world_state_v1`, `physics_calibration_receipt_v1`, and `simulation_outcome_receipt_v1` files directly from receipt directories and assembles canonical bundles with de-duplication by `state_id`
  - `scripts/train_sim_synth_backend_selector.py` and `scripts/train_sim_synth_branch_planner.py` now accept `--receipt-dir`, auto-harvest receipt bundles when no explicit dataset or receipt bundle is passed, and record `receipt_source_kind`, `receipt_dirs`, and `receipt_bundle_count` in their dataset summaries
  - `src/world_model/sim_synth_physics/__init__.py` now exposes the new harvester without forcing eager compiler/runtime imports, which also fixed a real circular-import bug between the sim/synth WM package and orchestrator imports
- Changed: surfaced the newly canonical advisory-replacement classes in promotion/readiness reporting. `src/regality/promotion_reporting.py` now reports:
  - `work_order_ready_count`
  - `control_plane_context_summary`
  - `teacher_provider_truth_summary`
  - `scene_tracks_provider_truth_summary`
  - per-node control-plane / provider-truth episode counts
  This makes the doctrine operational in readiness reports rather than leaving it buried in artifacts.
- Verification: `python3 -m compileall scripts/check_canonical_receipt_contracts.py scripts/run_full_repo_verification.py src/world_model/sim_synth_physics/training_corpus.py src/world_model/sim_synth_physics/__init__.py scripts/train_sim_synth_backend_selector.py scripts/train_sim_synth_branch_planner.py src/regality/promotion_reporting.py tests/test_canonical_receipt_contracts.py tests/test_sim_synth_training_corpus.py tests/test_train_sim_synth_backend_selector.py tests/test_train_sim_synth_branch_planner.py tests/test_promotion_reporting.py tests/test_online_promotion_reporting.py -q`, `python3 -m ruff check scripts/check_canonical_receipt_contracts.py scripts/run_full_repo_verification.py src/world_model/sim_synth_physics/training_corpus.py src/world_model/sim_synth_physics/__init__.py scripts/train_sim_synth_backend_selector.py scripts/train_sim_synth_branch_planner.py src/regality/promotion_reporting.py tests/test_canonical_receipt_contracts.py tests/test_sim_synth_training_corpus.py tests/test_train_sim_synth_backend_selector.py tests/test_train_sim_synth_branch_planner.py tests/test_promotion_reporting.py tests/test_online_promotion_reporting.py`, `python3 -m pytest -q tests/test_canonical_receipt_contracts.py tests/test_sim_synth_training_corpus.py tests/test_train_sim_synth_backend_selector.py tests/test_train_sim_synth_branch_planner.py tests/test_promotion_reporting.py tests/test_online_promotion_reporting.py`, `python3 scripts/check_canonical_receipt_contracts.py`, and `git diff --check`.
- Blocked: this closes the remaining non-GPU contract-hygiene gap inside the already-wired loop, but honest promotion remains constrained by receipt density and grounded-data availability rather than missing contracts. The next additive work should be keeping these checks on every new helper or WM boundary so the repo does not regress into anonymous advisory blobs.
- Changed: started concrete Phase 1 sim/synth/physics WM ownership transfer beyond compile-only state:
  - added typed economic and embodiment input adapters in `src/world_model/sim_synth_physics/adapters/economic_inputs.py` and `src/world_model/sim_synth_physics/adapters/embodiment_inputs.py`, so the WM no longer treats those lanes as raw passthrough mappings
  - added `src/world_model/sim_synth_physics/physics_contracts.py`, `src/world_model/sim_synth_physics/backend_router.py`, and `src/world_model/sim_synth_physics/calibration.py` so backend routing, fallback honesty, and calibration quality are now first-class WM-owned contracts instead of implicit metadata
  - expanded `src/world_model/sim_synth_physics/runtime.py` from a compile-only facade into a WM-owned compile/run boundary that can:
    - compile legacy agenda views for compatibility
    - compile diffusion plans from the same runtime boundary
    - execute a planning window into canonical `physics_execution_contract`, `physics_calibration_receipt`, `simulation_outcome_receipts`, and `sim_synth_training_feedback` artifacts
  - `src/orchestrator/semantic_simulation.py` and `src/orchestrator/diffusion_requests.py` now call through those runtime entrypoints instead of open-coding the same config/compile flow
  - added WM-owned entry scripts:
    - `scripts/compile_sim_synth_physics_plan.py`
    - `scripts/run_sim_synth_physics_loop.py`
- Changed: the Phase 1 posture is now explicit in the architecture docs. `docs/economic_world_model/multi_wm_architecture_plan.md` now states that no later phase should start while the current phase still has implementable ownership/runtime/adapter/package gaps, and that Phase 1 specifically should keep pushing toward real Isaac Sim / Isaac Gym / Unitree-class adapter functionality rather than normalizing PyBullet fallback.
- Verification: `python3 -m compileall src/world_model/sim_synth_physics src/orchestrator/semantic_simulation.py src/orchestrator/diffusion_requests.py scripts/compile_sim_synth_physics_plan.py scripts/run_sim_synth_physics_loop.py tests/test_sim_synth_physics_world_model.py tests/test_sim_synth_physics_scripts.py -q`, `python3 -m ruff check src/world_model/sim_synth_physics src/orchestrator/semantic_simulation.py src/orchestrator/diffusion_requests.py scripts/compile_sim_synth_physics_plan.py scripts/run_sim_synth_physics_loop.py tests/test_sim_synth_physics_world_model.py tests/test_sim_synth_physics_scripts.py`, `python3 -m pytest -q tests/test_sim_synth_physics_world_model.py tests/test_sim_synth_physics_scripts.py tests/test_gap_agenda_ranking.py tests/test_coverage_compilation.py`, and `git diff --check`.
- Blocked: Phase 1 is now further along the honest-complete-subsystem path, but the remaining explicit gaps are still real and still in-phase:
  - real Isaac Sim / Isaac Gym adapter implementation
  - Unitree-class sim-env integration behind typed backend routing
  - richer Holosoma adapter behavior
  - domain-randomization and system-ID policy
  - NAG / LSD / GGDS productionization
  - grounded GPU-backed perception-conditioned sim
- Changed: continued the next in-phase ownership transfer by absorbing synthetic branch generation and gen2sim admission further into the sim/synth/physics WM:
  - added `src/world_model/sim_synth_physics/synthetic_branches.py` as the WM-owned home for:
    - local synthetic branch rollout/gating helpers
    - branch gap labeling
    - branch-plan compilation
    - synthetic branch corpus metadata construction
  - added `src/world_model/sim_synth_physics/gen2sim_admission.py` as the WM-owned home for:
    - compilation of `Gen2SimAdmissionState`
    - local synthetic branch corpus gen2sim assessment rows and summaries
  - `src/world_model/sim_synth_physics/compiler.py` now calls through those modules instead of keeping branch-plan compilation and gen2sim admission logic inline
  - `scripts/collect_local_synthetic_branches.py` is now a thinner WM worker: it still loads the stable world model and trust-net locally, but it no longer owns the branch rollout/gating rules or gen2sim corpus assessment logic
- Verification: `python3 -m compileall src/world_model/sim_synth_physics scripts/collect_local_synthetic_branches.py tests/test_sim_synth_branch_helpers.py tests/test_sim_synth_physics_world_model.py -q`, `python3 -m ruff check src/world_model/sim_synth_physics scripts/collect_local_synthetic_branches.py tests/test_sim_synth_branch_helpers.py tests/test_sim_synth_physics_world_model.py`, `python3 -m pytest -q tests/test_sim_synth_branch_helpers.py tests/test_sim_synth_physics_world_model.py tests/test_sim_synth_physics_scripts.py tests/test_gap_agenda_ranking.py tests/test_coverage_compilation.py tests/test_gen2sim_validity.py`, and `git diff --check`.
- Blocked: this narrows the script-owned branch gap, but the remaining explicit Phase 1 blockers are still the same honest ones:
  - real Isaac Sim / Isaac Gym backend implementation
  - Unitree-class sim-env integration
  - richer Holosoma execution contract
  - domain-randomization and system-identification policy
  - NAG / LSD / GGDS productionization
  - grounded GPU-backed perception-conditioned sim
- Changed: pushed the next concrete Phase 1 backend-runtime tranche so the WM gets closer to a real humanoid execution substrate instead of just request/receipt scaffolding:
  - `src/world_model/sim_synth_physics/backend_runtime_execution.py` now supports a real Holosoma train-or-eval split:
    - evaluate an existing runtime policy when a policy id exists
    - train from motion datapacks / direct motion clips when a policy id does not exist but a motion source bundle does
    - emit honest `runtime_training_completed` status instead of pretending a missing policy blocks the lane when the runtime can actually train
  - `src/world_model/sim_synth_physics/runtime_evidence.py` and `src/world_model/sim_synth_physics/calibration.py` now treat concrete runtime training as real runtime evidence, so adaptation/calibration receipts react to train-path execution instead of only eval-path execution
  - added `src/world_model/sim_synth_physics/asset_manifest.py` and threaded it through `adapters/backend_isaac.py`, `asset_contracts.py`, `backend_runtime_execution.py`, `adapters/backend_holosoma.py`, and `shadow_execution.py`
  - Unitree-target asset manifests are now normalized into canonical hardware contracts (`unitree_robot_description`, `whole_body_joint_map`, camera/IMU/force-torque calibration, actuator latency, joint limits, safety watchdog) instead of being treated as arbitrary manifest keys
  - the resulting robot-asset contract now honestly unions backend-specific requirements with Unitree-class hardware requirements, which makes backend readiness reflect real humanoid control prerequisites rather than a thin manifest
- Verification: `python3 -m compileall src/world_model/sim_synth_physics scripts/run_sim_synth_physics_loop.py tests/test_sim_synth_physics_world_model.py tests/test_sim_synth_training_corpus.py tests/test_sim_synth_physics_scripts.py -q`, `python3 -m ruff check src/world_model/sim_synth_physics scripts/run_sim_synth_physics_loop.py tests/test_sim_synth_physics_world_model.py tests/test_sim_synth_training_corpus.py tests/test_sim_synth_physics_scripts.py`, `python3 -m pytest -q tests/test_sim_synth_physics_world_model.py tests/test_sim_synth_training_corpus.py tests/test_sim_synth_physics_scripts.py`, `python3 -m json.tool scripts/FOUNDATION_MODEL_BRINGUP_BACKLOG.json >/dev/null`, and `git diff --check`.
- Blocked: this removes another structural Phase 1 excuse, but the remaining blockers are still concrete runtime/asset realities:
  - a real Isaac Lab / Isaac Sim / Unitree backend module and assets
  - real Holosoma host/runtime plus motion data and reward/retargeting context
  - real GGDS/LDM and grounded video materialization on GPU
- Changed: pushed the next Phase 1 runtime-materialization tranche so backend bring-up is operationally queueable rather than just documented:
  - added `src/world_model/sim_synth_physics/runtime_targets.py` and threaded it through the Isaac and Holosoma backend bindings plus backend runtime materialization
  - runtime bindings and runtime-request artifacts now carry explicit runtime-target contracts for Isaac/Unitree/Holosoma roots, SDKs, and asset trees, so missing host/repo/runtime pieces are named directly in canonical metadata and sidecars
  - extended `src/orchestrator/loop_run_backlog.py` host capability detection to cover `diffusers`, Isaac runtime modules, Holosoma, and Unitree runtime roots, which makes GPU/runtime bring-up assessment more honest
  - added `scripts/local_holosoma_smoke.py` as a real non-training Holosoma evaluation smoke
  - added `scripts/NON_TRAINING_GPU_RUN_BACKLOG.json`, `src/orchestrator/non_training_gpu_run_backlog.py`, and `scripts/scan_non_training_gpu_run_backlog.py` so non-training GPU runs live in their own explicit queue instead of being mixed into the training backlog
- Verification: `python3 -m compileall src/world_model/sim_synth_physics src/orchestrator/loop_run_backlog.py src/orchestrator/non_training_gpu_run_backlog.py scripts/local_holosoma_smoke.py scripts/scan_non_training_gpu_run_backlog.py tests/test_sim_synth_runtime_targets.py tests/test_non_training_gpu_run_backlog.py tests/test_sim_synth_physics_world_model.py tests/test_sim_synth_physics_scripts.py -q`, `python3 -m ruff check src/world_model/sim_synth_physics src/orchestrator/loop_run_backlog.py src/orchestrator/non_training_gpu_run_backlog.py scripts/local_holosoma_smoke.py scripts/scan_non_training_gpu_run_backlog.py tests/test_sim_synth_runtime_targets.py tests/test_non_training_gpu_run_backlog.py tests/test_sim_synth_physics_world_model.py tests/test_sim_synth_physics_scripts.py`, `python3 -m pytest -q tests/test_sim_synth_runtime_targets.py tests/test_non_training_gpu_run_backlog.py tests/test_sim_synth_physics_world_model.py tests/test_sim_synth_training_corpus.py tests/test_sim_synth_physics_scripts.py tests/test_loop_run_backlog.py`, `python3 -m json.tool scripts/NON_TRAINING_GPU_RUN_BACKLOG.json >/dev/null`, and `git diff --check`.
- Blocked: the remaining Phase 1 backend/runtime blockers are now more clearly “missing external runtime roots, SDKs, assets, GPUs, or checkpoints” rather than missing inventory of those surfaces inside the WM.
- Changed: pushed the next Phase 1 external-runtime tranche so upstream runtime launches now feed canonical WM outcome evidence rather than terminating at `launch_completed` / `launch_failed`:
  - added `src/world_model/sim_synth_physics/runtime_outcomes.py`
  - the WM now compiles `backend_runtime_output_contract_v1` from runtime bundles/launch specs and harvests upstream outputs into `backend_runtime_outcome_receipt_v1`
  - this is currently shaped around the upstream layouts we explicitly want to support:
    - `unitree_sim_isaaclab`
    - `unitree_rl_gym`
    - `HumanoidVerse`
    - `xr_teleoperate`
    - Holosoma repo / motion bank / policy bank / retargeting roots
  - `backend_runtime_execution.py` now threads that receipt into runtime execution metadata and artifact emission
  - `runtime.py`, `runtime_evidence.py`, `runtime_work_orders.py`, `training_corpus.py`, `run_sim_synth_physics_loop.py`, and `run_phase1_runtime_launch.py` now preserve/use that truth end to end
  - practical effect:
    - the WM can distinguish `launch_not_executed`, `runtime_outputs_missing`, and `runtime_outputs_harvested`
    - runtime work orders can now be `satisfied_by_external_runtime_outcomes`
    - calibration/adaptation receipts now react to harvested upstream runtime evidence instead of only in-process runtime completion
    - backend-selector and branch-planner corpora now preserve external-runtime outcome ids/status/counts rather than only launch status
- Changed: deepened the concrete runtime-root / asset / policy posture using upstream repo conventions instead of only root existence:
  - `runtime_layouts.py` now surfaces deploy/policy/data candidates per profile
  - `runtime_targets.py` now includes additional optional Unitree-adjacent roots such as:
    - `unitree_sdk2_python_root`
    - `teleimager_root`
    - `unitree_il_lerobot_root`
  - `runtime_bundles.py` now carries the WM-owned output contract directly so launch artifacts and the full WM runtime speak the same upstream-runtime contract
- Verification: `python3 -m compileall src/world_model/sim_synth_physics scripts/run_phase1_runtime_launch.py scripts/run_sim_synth_physics_loop.py tests/test_sim_synth_runtime_outcomes.py tests/test_sim_synth_runtime_launch.py tests/test_sim_synth_runtime_bundles.py tests/test_sim_synth_runtime_layouts.py tests/test_sim_synth_runtime_work_orders.py tests/test_sim_synth_training_corpus.py tests/test_sim_synth_physics_world_model.py tests/test_sim_synth_physics_scripts.py -q`, `python3 -m ruff check src/world_model/sim_synth_physics scripts/run_phase1_runtime_launch.py scripts/run_sim_synth_physics_loop.py tests/test_sim_synth_runtime_outcomes.py tests/test_sim_synth_runtime_launch.py tests/test_sim_synth_runtime_bundles.py tests/test_sim_synth_runtime_layouts.py tests/test_sim_synth_runtime_work_orders.py tests/test_sim_synth_training_corpus.py tests/test_sim_synth_physics_world_model.py tests/test_sim_synth_physics_scripts.py`, `python3 -m pytest -q tests/test_sim_synth_runtime_outcomes.py tests/test_sim_synth_runtime_launch.py tests/test_sim_synth_runtime_bundles.py tests/test_sim_synth_runtime_layouts.py tests/test_sim_synth_runtime_work_orders.py tests/test_sim_synth_training_corpus.py`, `python3 -m pytest -q tests/test_sim_synth_physics_world_model.py tests/test_sim_synth_physics_scripts.py`, and `git diff --check`.
- Blocked: the honest remainder is now even more clearly external-runtime/GPU reality rather than missing WM plumbing:
  - actual Isaac Lab / Isaac Sim / Unitree execution adapters and assets
  - actual Holosoma host/runtime/policy/motion/retargeting assets
  - actual GGDS / video-diffusion materialization on GPU
## 2026-04-01

- Changed: pushed the next concrete Isaac/Unitree runtime-execution mediation cut so the WM now owns not just the executable-adapter request but the consumer over that request:
  - added `src/world_model/sim_synth_physics/adapters/isaac_unitree_executable_consumer.py`
  - `runtime_bundles.py` now emits `backend_executable_adapter_consumer_v1` beside `backend_executable_adapter_request_v1`
  - the consumer names:
    - consumer mode
    - consumer status
    - local-python-bridge vs external-launch responsibility
    - remaining preconditions
  - `runtime_launch.py` now consumes that consumer surface during launch preparation instead of flattening the adapter mediation into generic launch metadata
  - `backend_runtime_execution.py` now writes and preserves the consumer artifact/metadata inside the live Phase-1 runtime path
  - `scripts/run_isaac_unitree_executable_adapter.py` now exposes both request and consumer
- Verification: `python3 -m compileall src/world_model/sim_synth_physics/adapters/isaac_unitree_executable_consumer.py src/world_model/sim_synth_physics/adapters/__init__.py src/world_model/sim_synth_physics/runtime_bundles.py src/world_model/sim_synth_physics/runtime_launch.py src/world_model/sim_synth_physics/backend_runtime_execution.py scripts/run_isaac_unitree_executable_adapter.py tests/test_isaac_unitree_executable_consumer.py tests/test_sim_synth_runtime_bundles.py tests/test_sim_synth_runtime_launch.py tests/test_sim_synth_physics_world_model.py -q`, `python3 -m ruff check src/world_model/sim_synth_physics/adapters/isaac_unitree_executable_consumer.py src/world_model/sim_synth_physics/adapters/__init__.py src/world_model/sim_synth_physics/runtime_bundles.py src/world_model/sim_synth_physics/runtime_launch.py src/world_model/sim_synth_physics/backend_runtime_execution.py scripts/run_isaac_unitree_executable_adapter.py tests/test_isaac_unitree_executable_consumer.py tests/test_sim_synth_runtime_bundles.py tests/test_sim_synth_runtime_launch.py tests/test_sim_synth_physics_world_model.py`, `python3 -m pytest -q tests/test_isaac_unitree_executable_consumer.py tests/test_sim_synth_runtime_bundles.py tests/test_sim_synth_runtime_launch.py tests/test_sim_synth_physics_world_model.py`, and `git diff --check`.
- Blocked: this is a meaningful Phase-1 closure step, but it is still consumer mediation, not full execution realization:
  - a concrete Isaac Lab / Isaac Sim / Unitree adapter still needs to consume this consumer surface against real upstream runtime/assets
  - Holosoma still needs an equivalent runtime-execution deepening
  - GPU-backed GGDS / video materialization remains outstanding

- Changed: pushed the next concrete Phase 1 Isaac/Unitree executable-adapter cut so the WM does more than emit generic launch commands:
  - added `src/world_model/sim_synth_physics/adapters/isaac_unitree_executable_adapter.py`
  - `runtime_bundles.py` now emits `backend_executable_adapter_request_v1` for Isaac/Unitree bundles and launch specs, carrying:
    - deployment mode
    - adapter entrypoint
    - robot variant / placement class
    - required target ids and required asset ids
    - normalized asset refs
    - calibration / observation / action contract ids
    - output expectations
    - environment overrides for the executable lane
  - `runtime_launch.py` now treats that executable-adapter request as a load-bearing part of launch preparation instead of leaving the Unitree specifics implied only by command strings
  - added `scripts/run_isaac_unitree_executable_adapter.py` as a dedicated WM-facing runner over the existing launch artifacts
  - the result is that the Isaac/Unitree lane now has a concrete executable-adapter surface even when the remaining blocker is still the upstream runtime/assets/GPU rather than local repo logic
- Changed: updated the Phase-1 master docs so the executable-adapter request is now an explicit part of the acceptance posture for the external-runtime lane:
  - `docs/economic_world_model/multi_wm_architecture_plan.md`
  - `docs/economic_world_model/roadmap.md`
- Verification: `python3 -m compileall src/world_model/sim_synth_physics/adapters/isaac_unitree_executable_adapter.py src/world_model/sim_synth_physics/adapters/__init__.py src/world_model/sim_synth_physics/runtime_bundles.py src/world_model/sim_synth_physics/runtime_launch.py src/world_model/sim_synth_physics/backend_runtime_execution.py scripts/run_isaac_unitree_executable_adapter.py tests/test_isaac_unitree_executable_adapter.py tests/test_sim_synth_runtime_bundles.py tests/test_sim_synth_runtime_launch.py -q`, `python3 -m ruff check src/world_model/sim_synth_physics/adapters/isaac_unitree_executable_adapter.py src/world_model/sim_synth_physics/adapters/__init__.py src/world_model/sim_synth_physics/runtime_bundles.py src/world_model/sim_synth_physics/runtime_launch.py src/world_model/sim_synth_physics/backend_runtime_execution.py scripts/run_isaac_unitree_executable_adapter.py tests/test_isaac_unitree_executable_adapter.py tests/test_sim_synth_runtime_bundles.py tests/test_sim_synth_runtime_launch.py`, `python3 -m pytest -q tests/test_isaac_unitree_executable_adapter.py tests/test_sim_synth_runtime_bundles.py tests/test_sim_synth_runtime_launch.py tests/test_sim_synth_physics_world_model.py`, and `git diff --check`.
- Blocked: this is another structural closure step, but Phase 1 still honestly needs:
  - a concrete Isaac Lab / Isaac Sim / Unitree executable adapter that can consume these requests against real upstream runtime/assets
  - deeper Holosoma runtime execution under the same contract quality
  - GPU-backed GGDS / video materialization

- Changed: reconciled branch-truth doctrine for the active multi-WM implementation arc:
  - landed the previously local-only tranche/doctrine/collaboration artifacts:
    - `docs/economic_world_model/neuralization_bridge_doctrine.md`
    - `docs/economic_world_model/codex_tranche_sim_synth_closure.md`
    - `docs/economic_world_model/codex_tranche_perception_wm_schema.md`
    - `.agent/claude_copilot.md`
    - `CLAUDE.md` now includes the Claude copilot doctrine entrypoint
  - this makes the branch’s effective operating posture explicit in git rather than leaving the active/held tranche split and Codex/Claude collaboration doctrine as local-only state
- Changed: completed the next highest-leverage Phase 1 Isaac/Unitree contract cut:
  - added `src/world_model/sim_synth_physics/adapters/isaac_unitree_deployment.py`
  - `backend_isaac.py` now emits a real `deployment_contract` and can distinguish:
    - `runtime_ready`
    - `external_launch_ready`
    - `external_launch_assets_missing`
    - older shadow/assets-missing states
  - `runtime_targets.py`, `runtime_layouts.py`, `runtime_bundles.py`, `runtime_bridge.py`, `runtime_launch.py`, `runtime_outcomes.py`, and `backend_runtime_execution.py` now understand:
    - `unitree_lerobot`
    - XR teleop + `sdk2_python` / `teleimager`
    - deployment-contract-aware preferred-profile selection
    - richer external-launch transport profiles and output harvesting
  - the main practical fix is that the WM now treats Unitree teleop / LeRobot / external launch posture as explicit runtime reality instead of flattening everything into generic Isaac shadow status
- Verification: `python3 -m compileall src/world_model/sim_synth_physics/adapters/isaac_unitree_deployment.py src/world_model/sim_synth_physics/adapters/backend_isaac.py src/world_model/sim_synth_physics/adapters/__init__.py src/world_model/sim_synth_physics/runtime_targets.py src/world_model/sim_synth_physics/runtime_layouts.py src/world_model/sim_synth_physics/runtime_bundles.py src/world_model/sim_synth_physics/runtime_bridge.py src/world_model/sim_synth_physics/runtime_launch.py src/world_model/sim_synth_physics/runtime_outcomes.py src/world_model/sim_synth_physics/backend_runtime_execution.py tests/test_isaac_unitree_deployment.py tests/test_sim_synth_runtime_targets.py tests/test_sim_synth_runtime_layouts.py tests/test_sim_synth_runtime_bundles.py tests/test_sim_synth_physics_world_model.py -q`, `python3 -m ruff check src/world_model/sim_synth_physics/adapters/isaac_unitree_deployment.py src/world_model/sim_synth_physics/adapters/backend_isaac.py src/world_model/sim_synth_physics/adapters/__init__.py src/world_model/sim_synth_physics/runtime_targets.py src/world_model/sim_synth_physics/runtime_layouts.py src/world_model/sim_synth_physics/runtime_bundles.py src/world_model/sim_synth_physics/runtime_bridge.py src/world_model/sim_synth_physics/runtime_launch.py src/world_model/sim_synth_physics/runtime_outcomes.py src/world_model/sim_synth_physics/backend_runtime_execution.py tests/test_isaac_unitree_deployment.py tests/test_sim_synth_runtime_targets.py tests/test_sim_synth_runtime_layouts.py tests/test_sim_synth_runtime_bundles.py tests/test_sim_synth_physics_world_model.py`, `python3 -m pytest -q tests/test_isaac_unitree_deployment.py tests/test_sim_synth_runtime_targets.py tests/test_sim_synth_runtime_layouts.py tests/test_sim_synth_runtime_bundles.py tests/test_sim_synth_physics_world_model.py`, and `git diff --check`.
- Blocked: Phase 1 remains the active implementation center of gravity. The honest remainder is still external runtime, assets, GPU, and provider maturity:
  - real Isaac Lab / Isaac Sim / Unitree executable adapters and assets
  - real Holosoma host/runtime + motion/policy/retargeting assets
  - GGDS / video-diffusion GPU materialization

- Changed: completed the next Phase 1 Isaac/Unitree execution-mediation cut so the runtime lane no longer jumps straight from consumer selection to generic launch status:
  - added `src/world_model/sim_synth_physics/adapters/isaac_unitree_adapter_execution.py`
  - the new layer emits `backend_executable_adapter_execution_v1` over the existing request/consumer pair and distinguishes:
    - `local_bridge_ready`
    - `local_bridge_missing`
    - `local_bridge_handed_off`
    - `external_launch_ready`
    - `external_launch_completed`
    - `external_launch_failed`
  - `src/world_model/sim_synth_physics/backend_runtime_execution.py` now writes and preserves that mediation artifact plus a new `backend_runtime_adapter_receipt_v1` inside the live Phase 1 runtime path
  - `src/world_model/sim_synth_physics/runtime.py` now surfaces the adapter receipt as a first-class loop artifact, carries it into loop summaries and training feedback, and preserves the distinction between executable mediation, launch, and harvested outcomes
  - `src/world_model/sim_synth_physics/training_corpus.py` now harvests the adapter receipt into backend-selector and branch-planner rows, so downstream training surfaces can see adapter readiness/execution-path truth instead of inferring everything from launch status
  - `scripts/run_isaac_unitree_executable_adapter.py` now emits adapter execution plus adapter receipt alongside the existing launch report
- Verification: `python3 -m compileall src/world_model/sim_synth_physics scripts/run_isaac_unitree_executable_adapter.py tests/test_isaac_unitree_adapter_execution.py tests/test_sim_synth_runtime_launch.py tests/test_sim_synth_physics_world_model.py tests/test_sim_synth_training_corpus.py -q`, `python3 -m ruff check src/world_model/sim_synth_physics scripts/run_isaac_unitree_executable_adapter.py tests/test_isaac_unitree_adapter_execution.py tests/test_sim_synth_runtime_launch.py tests/test_sim_synth_physics_world_model.py tests/test_sim_synth_training_corpus.py`, `python3 -m pytest -q tests/test_isaac_unitree_adapter_execution.py tests/test_sim_synth_runtime_launch.py tests/test_sim_synth_runtime_bundles.py tests/test_sim_synth_training_corpus.py tests/test_sim_synth_physics_world_model.py`, and `git diff --check`.
- Blocked: this is another real Phase-1 closure step, but the remaining honest gap is still the final executable realization, not more contract naming:
  - a concrete Isaac Lab / Isaac Sim / Unitree adapter still needs to consume the new request/consumer/adapter-execution chain against real upstream runtime/assets
  - Holosoma still needs equivalent runtime-execution mediation and receipt depth
  - GGDS / video-diffusion still need GPU-backed materialization

- Changed: pushed the next concrete Isaac/Unitree realization cut so the local runtime lane no longer relies on an implicit backend-factory jump:
  - added `src/world_model/sim_synth_physics/adapters/isaac_unitree_adapter_realization.py`
  - the Isaac/Unitree lane now emits `backend_executable_adapter_realization_v1`, which distinguishes:
    - `local_backend_factory`
    - `external_launch_delegate`
    - blocked realization
  - this is intentionally not the final hardware adapter; it is the typed surface that says how the current branch concretely realizes the adapter chain today
  - `src/world_model/sim_synth_physics/backend_runtime_execution.py` now rebuilds that realization after adapter-execution finalization, preserves it in runtime metadata, and writes `backend_runtime_adapter_realization.json`
  - `src/world_model/sim_synth_physics/runtime.py` now promotes that realization to a root-level loop artifact instead of leaving it nested only inside metadata
  - `src/world_model/sim_synth_physics/training_corpus.py` now preserves realization path/status into backend-selector and branch-planner rows, so downstream training surfaces can tell “external delegate” from “local backend factory” rather than inferring from launch state alone
  - `scripts/run_isaac_unitree_executable_adapter.py` now emits the same realization surface alongside request / consumer / execution / launch artifacts
- Verification: `python3 -m compileall src/world_model/sim_synth_physics scripts/run_isaac_unitree_executable_adapter.py tests/test_isaac_unitree_adapter_realization.py tests/test_sim_synth_runtime_launch.py tests/test_sim_synth_physics_world_model.py tests/test_sim_synth_training_corpus.py -q`, `python3 -m ruff check src/world_model/sim_synth_physics scripts/run_isaac_unitree_executable_adapter.py tests/test_isaac_unitree_adapter_realization.py tests/test_sim_synth_runtime_launch.py tests/test_sim_synth_physics_world_model.py tests/test_sim_synth_training_corpus.py`, `python3 -m pytest -q tests/test_isaac_unitree_adapter_execution.py tests/test_isaac_unitree_adapter_realization.py tests/test_sim_synth_runtime_launch.py tests/test_sim_synth_runtime_bundles.py tests/test_sim_synth_training_corpus.py tests/test_sim_synth_physics_world_model.py`, and `git diff --check`.
- Blocked: the honest remainder is narrower again, but still real:
  - a final concrete Isaac Lab / Isaac Sim / Unitree adapter implementation still needs to consume the new request / consumer / execution / realization chain against real upstream runtime/assets
  - Holosoma still needs the same realization depth
  - GGDS / video-diffusion still need GPU-backed materialization

- Changed: pushed the next Phase 1 runtime-materialization tranche across both backend lanes so the local-runtime seam is explicit instead of being partly implicit and partly backend-specific:
  - added `src/world_model/sim_synth_physics/adapters/local_backend_factory_adapter.py`
  - the branch now emits a typed local backend-factory invocation/result surface over executable-adapter realization, so explicit local adapter materialization is no longer hidden inside a direct `make_motor_backend(...)` jump
  - `backend_runtime_execution.py` now uses that explicit invocation/result surface before concrete runtime evaluation/training for both Isaac/Unitree and Holosoma
  - local materialization truth is now preserved into adapter receipt metadata and downstream corpus rows, so replay/training surfaces can tell “local adapter was attempted and materialized” from “local path was only contract-shaped”
- Changed: brought Holosoma up one major structural rung so it no longer lags Isaac/Unitree as a special-case runtime lane:
  - added:
    - `src/world_model/sim_synth_physics/adapters/holosoma_executable_adapter.py`
    - `src/world_model/sim_synth_physics/adapters/holosoma_executable_consumer.py`
    - `src/world_model/sim_synth_physics/adapters/holosoma_adapter_execution.py`
    - `src/world_model/sim_synth_physics/adapters/holosoma_adapter_realization.py`
  - `runtime_bundles.py` now emits executable-adapter request/consumer surfaces for Holosoma too
  - `backend_runtime_execution.py` now emits Holosoma adapter execution / realization / receipt metadata instead of leaving Holosoma as a concrete-runtime special case beside the typed lane
  - the Holosoma motion-train path now explicitly drops `policy_checkpoint` when train-from-motion is the real bounded mode, so the adapter ladder stays honest instead of blocking a valid local training lane
- Verification: `python3 -m compileall src/world_model/sim_synth_physics scripts/run_isaac_unitree_executable_adapter.py tests/test_local_backend_factory_adapter.py tests/test_holosoma_executable_adapter.py tests/test_holosoma_adapter_execution.py tests/test_holosoma_adapter_realization.py tests/test_sim_synth_runtime_bundles.py tests/test_sim_synth_physics_world_model.py tests/test_sim_synth_training_corpus.py -q`, `python3 -m ruff check src/world_model/sim_synth_physics scripts/run_isaac_unitree_executable_adapter.py tests/test_local_backend_factory_adapter.py tests/test_holosoma_executable_adapter.py tests/test_holosoma_adapter_execution.py tests/test_holosoma_adapter_realization.py tests/test_sim_synth_runtime_bundles.py tests/test_sim_synth_physics_world_model.py`, `python3 -m pytest -q tests/test_local_backend_factory_adapter.py tests/test_holosoma_executable_adapter.py tests/test_holosoma_adapter_execution.py tests/test_holosoma_adapter_realization.py tests/test_sim_synth_runtime_bundles.py tests/test_sim_synth_physics_world_model.py tests/test_sim_synth_training_corpus.py`, and `git diff --check`.
- Blocked: Phase 1 is now closer to the right honest remainder:
  - actual Isaac Lab / Isaac Sim / Unitree upstream runtime/assets/policies
  - actual Holosoma host/runtime/motion/policy/retargeting assets
  - GPU-backed GGDS / video materialization

- Changed: pushed the next backend-specific closure tranche so upstream runtime surfaces stop living only as implicit repo roots and start becoming canonical runtime-pack truth inside the Phase 1 WM:
  - added:
    - `src/world_model/sim_synth_physics/adapters/isaac_unitree_runtime_pack.py`
    - `src/world_model/sim_synth_physics/adapters/holosoma_deployment.py`
    - `src/world_model/sim_synth_physics/adapters/holosoma_runtime_pack.py`
  - `backend_isaac.py` now emits an explicit Isaac/Unitree upstream runtime pack over runtime targets, runtime layouts, deployment modes, policy-bank surfaces, telemetry surfaces, and normalized robot-asset refs
  - `backend_holosoma.py` now emits both a Holosoma deployment contract and a Holosoma upstream runtime pack, and it no longer treats retargeting / reward-overlay / policy surfaces as universally required for all Holosoma modes
  - `runtime_bundles.py`, `runtime_bridge.py`, `runtime.py`, `runtime_work_orders.py`, and `training_corpus.py` now preserve upstream runtime-pack truth as load-bearing metadata rather than leaving it stranded beside bindings
  - `scripts/scan_phase1_runtime_layouts.py` now exports deployment contracts plus upstream runtime packs for both backends, so Phase 1 runtime scanning can name pack-ready vs pack-partial vs pack-blocked posture directly
- Changed: this removes another fake-readiness seam:
  - the WM can now say whether the Isaac/Unitree or Holosoma lane has a real upstream runtime/profile/policy/asset pack available
  - Holosoma can now distinguish `sim_eval`, `motion_train`, and `retarget_eval` instead of pretending one universal asset posture
  - downstream corpus/work-order surfaces now preserve pack status and missing components, so later training or GPU bring-up does not need to rediscover them from scattered roots
- Verification:
  - `python3 -m compileall src/world_model/sim_synth_physics scripts/scan_phase1_runtime_layouts.py tests/test_holosoma_deployment.py tests/test_holosoma_runtime_pack.py tests/test_isaac_unitree_runtime_pack.py tests/test_scan_phase1_runtime_layouts.py tests/test_sim_synth_runtime_bundles.py tests/test_sim_synth_physics_world_model.py -q`
  - `python3 -m ruff check src/world_model/sim_synth_physics scripts/scan_phase1_runtime_layouts.py tests/test_holosoma_deployment.py tests/test_holosoma_runtime_pack.py tests/test_isaac_unitree_runtime_pack.py tests/test_scan_phase1_runtime_layouts.py tests/test_sim_synth_runtime_bundles.py tests/test_sim_synth_physics_world_model.py`
  - `python3 -m pytest -q tests/test_holosoma_deployment.py tests/test_holosoma_runtime_pack.py tests/test_isaac_unitree_runtime_pack.py tests/test_scan_phase1_runtime_layouts.py tests/test_sim_synth_runtime_bundles.py tests/test_sim_synth_physics_world_model.py`
  - `python3 -m pytest -q tests/test_sim_synth_training_corpus.py tests/test_sim_synth_runtime_work_orders.py tests/test_sim_synth_runtime_launch.py`
- Blocked: the honest Phase 1 remainder is narrower again and increasingly external:
  - real Isaac Lab / Isaac Sim / Unitree runtime packs still need the actual upstream repos/assets/policies behind them
  - real Holosoma runtime packs still need actual host/runtime/motion/policy/retargeting assets
  - GPU-backed GGDS / video materialization still remains external-runtime / checkpoint / host work

- Changed: pushed the next Phase 1 closure rung so upstream runtime packs now feed a typed runtime-binding layer instead of being consumed as loose pack metadata:
  - added:
    - `src/world_model/sim_synth_physics/adapters/isaac_unitree_runtime_binding.py`
    - `src/world_model/sim_synth_physics/adapters/holosoma_runtime_binding.py`
  - `runtime_bundles.py` now emits `runtime_binding` and writes `backend_runtime_binding.json`
  - `runtime_launch.py`, `runtime_work_orders.py`, `runtime.py`, `training_corpus.py`, `scripts/scan_phase1_runtime_layouts.py`, and `scripts/run_isaac_unitree_executable_adapter.py` now preserve runtime-binding status, selected surfaces, and mode-relevant missing components
  - this removes a real fake-readiness seam: pack-level gaps are no longer blindly inherited as execution blockers when the selected local mode is already satisfied by explicit policy refs or motion datapacks
- Changed: fixed the Holosoma local concrete-runtime path so `motion_train` no longer mutates a stale `sim_eval` request in place:
  - `backend_runtime_execution.py` now rebuilds the Holosoma executable-adapter request from the patched runtime binding when train-from-motion is the honest local mode
  - local Holosoma eval can now run with an explicit policy ref even when external repo/launch roots are absent
  - local Holosoma train-from-motion can now run with datapacks / inline clips without inheriting irrelevant `policy_surface` blockers
- Verification:
  - `python3 -m compileall src/world_model/sim_synth_physics scripts/scan_phase1_runtime_layouts.py scripts/run_isaac_unitree_executable_adapter.py tests/test_isaac_unitree_runtime_binding.py tests/test_holosoma_runtime_binding.py tests/test_scan_phase1_runtime_layouts.py tests/test_sim_synth_runtime_bundles.py tests/test_sim_synth_runtime_launch.py tests/test_sim_synth_physics_world_model.py tests/test_sim_synth_training_corpus.py tests/test_sim_synth_runtime_work_orders.py -q`
  - `python3 -m ruff check src/world_model/sim_synth_physics scripts/scan_phase1_runtime_layouts.py scripts/run_isaac_unitree_executable_adapter.py tests/test_isaac_unitree_runtime_binding.py tests/test_holosoma_runtime_binding.py tests/test_scan_phase1_runtime_layouts.py tests/test_sim_synth_runtime_bundles.py tests/test_sim_synth_runtime_launch.py tests/test_sim_synth_physics_world_model.py tests/test_sim_synth_training_corpus.py tests/test_sim_synth_runtime_work_orders.py`
  - `python3 -m pytest -q tests/test_isaac_unitree_runtime_binding.py tests/test_holosoma_runtime_binding.py tests/test_scan_phase1_runtime_layouts.py tests/test_sim_synth_runtime_bundles.py tests/test_sim_synth_runtime_launch.py tests/test_sim_synth_physics_world_model.py tests/test_sim_synth_training_corpus.py tests/test_sim_synth_runtime_work_orders.py`
  - result: `44 passed`
  - `git diff --check`
- Blocked: the remaining honest Phase 1 gap is now even more external:
  - real Isaac/Unitree upstream runtime, assets, checkpoints, and host setup still need to sit behind the new runtime-pack -> runtime-binding -> adapter ladder
  - real Holosoma host/runtime/motion/retargeting assets still need to sit behind the same ladder
  - GPU-backed GGDS / video materialization is still outside the current host

- Changed: closed the next concrete-runtime evidence gap so local backend execution no longer falls back to launch-shaped truth:
  - `runtime_outcome_parsers.py` now classifies rollout `trajectory` artifacts as dataset capture surfaces before generic motion-dataset classification, so local runtime rollouts count as trainer/replay-ready dataset evidence
  - `runtime_outcomes.py` now supports explicit local runtime artifact harvest and can emit `backend_runtime_outcome_receipt_v1` without a launch receipt, with `harvest_mode=local_runtime_execution`
  - `backend_runtime_execution.py` now harvests policy / metrics / rollout artifacts directly after successful concrete local execution and writes:
    - `backend_runtime_output_contract.json`
    - `backend_runtime_output_summary.json`
    - `backend_runtime_outcome_receipt.json`
  - this means the concrete local Isaac/Unitree and Holosoma paths now preserve policy / dataset / metrics surface readiness as canonical outcome truth instead of leaving that evidence implicit in runtime-execution metadata
- Changed: fixed a real fake-readiness bug on the Isaac/Unitree local bridge lane:
  - `isaac_unitree_runtime_binding.py` now filters out stale upstream-pack/runtime-profile gaps when the selected local `sim_eval` path already has the concrete local requirements it actually needs
  - `isaac_unitree_executable_adapter.py` now takes binding-selected missing components as primary request truth and only supplements them with still-missing required robot assets or policy state
  - consequence: a real local Isaac bridge with policy ref + SDK root + asset root is no longer blocked by irrelevant external-pack placeholders before it reaches the backend factory
- Verification:
  - `python3 -m compileall src/world_model/sim_synth_physics tests/test_sim_synth_runtime_outcomes.py tests/test_sim_synth_physics_world_model.py -q`
  - `python3 -m ruff check src/world_model/sim_synth_physics tests/test_sim_synth_runtime_outcomes.py tests/test_sim_synth_physics_world_model.py`
  - `python3 -m pytest -q tests/test_sim_synth_runtime_outcomes.py tests/test_sim_synth_physics_world_model.py`
  - `python3 -m pytest -q tests/test_sim_synth_training_corpus.py tests/test_sim_synth_physics_scripts.py`
  - `git diff --check`
  - result: `26 passed` and `5 passed`
- Blocked: the remaining honest Phase 1 backend gap is increasingly external rather than receipt-chain-local:
  - real Isaac/Unitree upstream runtime/assets/checkpoints still need to sit behind the now-honest local concrete evidence path
  - real Holosoma host/runtime/motion/policy/retargeting assets still need to sit behind the same path
  - GPU-backed GGDS / video materialization remains external host/model work

- Changed: pushed the next “real upstream evidence” Phase-1 closure tranche across both backend lanes:
  - `runtime_layouts.py` now emits profile-level evidence instead of only root/candidate names:
    - repo git metadata when a runtime root is a real local clone
    - deploy / policy / data candidate counts
    - primary deploy / policy / data refs
  - `describe_isaac_policy_contract(...)` and `describe_holosoma_policy_contract(...)` now emit:
    - primary checkpoint ref
    - primary deploy-config ref
    - primary runtime-report ref
    - candidate-record inventories and counts
  - `isaac_unitree_runtime_pack.py` now carries selected profile evidence plus declared-vs-verified asset truth:
    - `verified_asset_ids`
    - `declared_only_asset_ids`
    - `asset_evidence_summary`
  - `holosoma_runtime_pack.py` now carries selected profile evidence plus motion-source existence truth:
    - `existing_motion_sources`
    - `missing_motion_sources`
  - `runtime_work_orders.py` and `training_corpus.py` now preserve that evidence into work-order metadata and trainer rows, so downstream consumers no longer see only `pack_ready/partial` but also the exact primary refs and evidence density behind that status
- Changed: this removes another Phase-1 pseudo-readiness seam:
  - “runtime pack ready” no longer only means root/candidate presence
  - “asset present” for Isaac no longer means only “manifest key existed”
  - local/runtime consumers now preserve which upstream surfaces are concrete and which are only declared
- Verification:
  - `python3 -m compileall src/world_model/sim_synth_physics tests/test_sim_synth_runtime_layouts.py tests/test_isaac_unitree_runtime_pack.py tests/test_sim_synth_runtime_bundles.py tests/test_sim_synth_runtime_work_orders.py tests/test_scan_phase1_runtime_layouts.py tests/test_sim_synth_training_corpus.py tests/test_sim_synth_physics_world_model.py tests/test_sim_synth_physics_scripts.py -q`
  - `python3 -m ruff check src/world_model/sim_synth_physics tests/test_sim_synth_runtime_layouts.py tests/test_isaac_unitree_runtime_pack.py tests/test_sim_synth_runtime_bundles.py tests/test_sim_synth_runtime_work_orders.py tests/test_scan_phase1_runtime_layouts.py tests/test_sim_synth_training_corpus.py tests/test_sim_synth_physics_world_model.py tests/test_sim_synth_physics_scripts.py`
  - `python3 -m pytest -q tests/test_sim_synth_runtime_layouts.py tests/test_isaac_unitree_runtime_pack.py tests/test_sim_synth_runtime_bundles.py tests/test_sim_synth_runtime_work_orders.py tests/test_scan_phase1_runtime_layouts.py`
  - `python3 -m pytest -q tests/test_sim_synth_training_corpus.py tests/test_sim_synth_physics_world_model.py tests/test_sim_synth_physics_scripts.py`
  - `python3 -m pytest -q tests/test_sim_synth_runtime_launch.py tests/test_isaac_unitree_executable_adapter.py tests/test_holosoma_executable_adapter.py`
  - `git diff --check`
- Blocked: the honest remainder is narrower again:
  - the branch now has richer evidence about upstream runtime roots/checkpoints/assets, but it still needs the actual upstream runtimes/assets/checkpoints on host
  - Holosoma still needs actual host/runtime/motion/retargeting/provider assets behind those evidence surfaces
  - GPU-backed materialization still remains external

- Changed: pushed the next Category-B Phase 1 install-evidence tranche so runtime profiles now carry explicit install/preflight truth rather than only root/candidate truth:
  - `runtime_layouts.py` now emits per-profile install evidence:
    - selected install entrypoint paths
    - matched/missing entrypoints
    - primary entrypoint ref
    - install preflight status
    - install missing/verified components
  - Isaac/Unitree and Holosoma upstream runtime packs now preserve that profile-level install truth, including a `profile_install_by_id` map so downstream consumers can reason about whichever profile is actually selected rather than only the pack’s preferred profile
  - Isaac/Unitree and Holosoma runtime bindings now use the selected profile’s install truth when computing:
    - `runtime_profile_surface`
    - selected-profile missing components
    - host-preflight requirements
  - this removed a real false-blocker seam on the Holosoma motion-train lane: when the branch selects `holosoma_motion_bank`, it no longer inherits `holosoma_repo` install gaps like `profile_entrypoint`
- Changed: trainer/work-order surfaces now preserve the stronger install truth:
  - `training_corpus.py` now exports upstream/runtime-binding profile install status, selected primary entrypoint refs, and selected profile install-missing components
  - `runtime_work_orders.py` now preserves the same fields so runtime bring-up tasks can distinguish:
    - root discovered but install-blocked
    - selected profile install-ready
    - host-preflight blocked for an actually selected surface
- Verification:
  - `python3 -m compileall src/world_model/sim_synth_physics -q`
  - `python3 -m ruff check src/world_model/sim_synth_physics tests/test_sim_synth_runtime_layouts.py tests/test_scan_phase1_runtime_layouts.py tests/test_isaac_unitree_runtime_pack.py tests/test_holosoma_runtime_pack.py tests/test_isaac_unitree_runtime_binding.py tests/test_holosoma_runtime_binding.py tests/test_sim_synth_runtime_bundles.py tests/test_sim_synth_runtime_launch.py tests/test_sim_synth_runtime_work_orders.py tests/test_sim_synth_training_corpus.py`
  - `python3 -m pytest -q tests/test_sim_synth_runtime_layouts.py tests/test_scan_phase1_runtime_layouts.py tests/test_isaac_unitree_runtime_pack.py tests/test_holosoma_runtime_pack.py tests/test_isaac_unitree_runtime_binding.py tests/test_holosoma_runtime_binding.py tests/test_sim_synth_runtime_bundles.py tests/test_sim_synth_runtime_launch.py tests/test_sim_synth_runtime_work_orders.py tests/test_sim_synth_training_corpus.py`
  - result: `34 passed`

- Changed: proved that Phase 1 can still move meaningfully without a GPU once real public upstream roots are present on host:
  - pulled public runtime roots onto `/Users/amarmurray/code` for the Isaac/Unitree lane (`IsaacLab`, `unitree_sim_isaaclab`, `unitree_rl_gym`, `HumanoidVerse`, `xr_teleoperate`, `unitree_IL_lerobot`, `unitree_sdk2`, `unitree_models`) and for the Holosoma lane (`holosoma`)
  - the host scan now sees real local Isaac/Unitree runtime roots plus verified `unitree_sdk2_root` and `unitree_asset_root`
  - the Holosoma lane now consumes repo-derived motion/policy/retargeting subroots directly from the cloned repo and reaches `host_preflight_status=preflight_ready`
  - explicit deployment context now outranks background autodiscovery, so local public clones add evidence without hijacking requested Isaac profile/bridge selection
- Changed: closed two real non-GPU internal incompleteness items:
  - Holosoma policy selection no longer mistakes retargeting demo `.pt` files for runtime policy
  - Holosoma motion/retargeting surfaces no longer require manual restatement when they already exist under a real local repo
- Verification:
  - `python3 -m compileall src/world_model/sim_synth_physics scripts/scan_phase1_runtime_layouts.py tests/test_sim_synth_runtime_targets.py tests/test_sim_synth_runtime_layouts.py tests/test_holosoma_deployment.py tests/test_holosoma_runtime_pack.py tests/test_holosoma_runtime_binding.py tests/test_sim_synth_runtime_bundles.py tests/test_sim_synth_runtime_launch.py tests/test_sim_synth_runtime_work_orders.py tests/test_sim_synth_training_corpus.py tests/test_sim_synth_physics_world_model.py tests/test_scan_phase1_runtime_layouts.py -q`
  - `python3 -m ruff check src/world_model/sim_synth_physics scripts/scan_phase1_runtime_layouts.py tests/test_sim_synth_runtime_targets.py tests/test_sim_synth_runtime_layouts.py tests/test_holosoma_deployment.py tests/test_holosoma_runtime_pack.py tests/test_holosoma_runtime_binding.py tests/test_sim_synth_runtime_bundles.py tests/test_sim_synth_runtime_launch.py tests/test_sim_synth_runtime_work_orders.py tests/test_sim_synth_training_corpus.py tests/test_sim_synth_physics_world_model.py tests/test_scan_phase1_runtime_layouts.py`
  - `python3 -m pytest -q tests/test_sim_synth_runtime_targets.py tests/test_sim_synth_runtime_layouts.py tests/test_holosoma_deployment.py tests/test_holosoma_runtime_pack.py tests/test_holosoma_runtime_binding.py tests/test_sim_synth_runtime_bundles.py tests/test_sim_synth_runtime_launch.py tests/test_sim_synth_runtime_work_orders.py tests/test_sim_synth_training_corpus.py tests/test_sim_synth_physics_world_model.py tests/test_scan_phase1_runtime_layouts.py`
  - `python3 scripts/scan_phase1_runtime_layouts.py --output-path /tmp/phase1_runtime_scan_post_holosoma_fix.json`
  - result: `80 passed`

- Additional non-GPU Unitree asset-normalization pass:
  - `src/world_model/sim_synth_physics/asset_manifest.py` now derives verified local Unitree asset surfaces from already-discovered public roots rather than waiting for a hand-authored manifest.
  - Verified/derived surfaces now include:
    - `unitree_robot_description`
    - `whole_body_joint_map`
    - `joint_limit_profile`
    - recommended:
      - `control_frequency_profile`
      - `teleop_recovery_contract`
  - The derivation is intentionally conservative:
    - it uses real local files from `unitree_models`, `unitree_rl_gym`, `HumanoidVerse`, `unitree_sim_isaaclab`, and `xr_teleoperate`
    - it does not count loose README prose or generic control-loop mentions as `actuator_latency_profile` or `safety_watchdog_profile`
  - The live host scan after this pass now reports:
    - `verified_asset_count = 5` on the Isaac/Unitree runtime pack
    - remaining Isaac host-preflight blockers:
      - `asset::actuator_latency_profile`
      - `asset::safety_watchdog_profile`
  - This narrowed the useful non-GPU Phase-1 remainder again: public repos helped materially, and the remaining Isaac asset blockers are now the two contract surfaces that still lack clean public artifacts.

- Phase 2 Perception / Grounding first functional compiler tranche:
  - `src/world_model/perception_grounding/compiler.py` now compiles `PerceptionGroundingWorldState` from real upstream inputs already present in-repo:
    - scene tracks
    - belief state
    - VLA semantic evidence
    - existing semantic-world-model heuristic grounding
  - the compiled state now owns:
    - canonical scene graph
    - temporal grounding state
    - evidence routing state
    - provider/dataset/task/deployment-resource surfaces
    - semantic bridge registry with first heuristic bridge outputs
  - the semantic bridge family is no longer only declared:
    - Sim / Synth semantic bridge now feeds sim-synth semantic input context and inferential summary fields
    - annotation / evidence semantic bridge now feeds rollout-labeler tags and labeling metadata
  - `VisionBackboneStub` and `HeuristicVisionEncoderPolicy` now expose typed provider/advisory posture instead of remaining ambient placeholder consumers
  - closure effect:
    - Phase 2 is no longer just a schema/receipt shell
    - the Perception / Grounding WM is now starting to behave like a loop-facing subsystem at `shadow_runtime`
    - remaining internal work is now the next honest cluster:
      - live provider/deployment/headroom receipt emission
      - provider/runtime inventory compilation
      - replay/export surfaces
      - more downstream consumers

### 2026-04-03: Perception Seam Training Infrastructure

- Created complete seam training infrastructure for Phase 2 Perception / Grounding WM:

  **Loss Functions Module** (`src/training/perception_seam_losses.py`):
  - `evidence_fusion_loss`: held-out provider reconstruction + task correlation + availability contrastive
  - `sam_calibration_loss`: calibrated confidence vs downstream quality + uncertainty correlation + prompt satisfaction
  - `depth_metric_calibration_loss`: metric depth vs GT + uncertainty calibration + gradient preservation + scale consistency
  - `vjepa_temporal_alignment_loss`: future state prediction + confidence calibration + temporal ordering + smoothness
  - `vision_backbone_projection_loss`: object identity prediction + scene contrastive + cross-provider alignment
  - Each loss returns `SeamLossResult` with total loss, component breakdown, and metrics

  **Data Loaders Module** (`src/training/perception_seam_data.py`):
  - `ProviderAgreementDataset`: base dataset for multi-provider observations
  - `EvidenceFusionDataset` + `EvidenceFusionBatch`: held-out provider training data
  - `SAMCalibrationDataset` + `SAMCalibrationBatch`: mask quality calibration data
  - `DepthCalibrationDataset` + `DepthCalibrationBatch`: metric depth ground truth data
  - `VJEPATemporalDataset` + `VJEPATemporalBatch`: temporal alignment training data
  - Synthetic data generators for testing/bootstrapping each dataset type
  - Factory functions: `create_evidence_fusion_loader`, `create_sam_calibration_loader`, etc.

  **Trainer Module** (`src/training/perception_seam_trainer.py`):
  - `PerceptionSeamTrainer`: full training orchestrator with:
    - Gradient accumulation and mixed precision support
    - Validation loop with early stopping
    - Checkpoint saving via `PerceptionSeamRegistry`
    - Benchmark gate evaluation for promotion decisions
    - Receipt emission: `SeamTrainingStepReceipt`, `SeamValidationReceipt`, `BenchmarkGateReceipt`
  - `SeamTrainingConfig`: LR scheduling, warmup, gradient clipping, promotion thresholds
  - Convenience functions: `train_evidence_fusion_seam`, `train_sam_calibration_seam`, etc.

  **Benchmark Gate Evaluation** (`src/training/perception_seam_benchmarks.py`):
  - `EvidenceFusionBenchmark`: reconstruction accuracy, task correlation, provider dropout robustness
  - `SAMCalibrationBenchmark`: ECE, uncertainty-error correlation, confidence-quality correlation
  - `DepthCalibrationBenchmark`: abs-rel error, delta accuracy, uncertainty calibration
  - `VJEPATemporalBenchmark`: prediction accuracy, confidence calibration, temporal consistency
  - `BenchmarkGateResult`: overall score, per-metric breakdown, promotion decision
  - `BenchmarkGateConfig`: promotion/demotion/shadow thresholds, robustness testing options

  **Tests** (`tests/test_perception_seam_training.py`):
  - 26 tests covering loss functions, data loaders, collation, and benchmark evaluation
  - All tests pass

- This closes the "CRITICAL GAP: Seam Training Infrastructure" identified in Phase 2 assessment
- Seams can now be trained, validated, and promoted via benchmark gates
- Receipt-backed training enables honest promotion decisions without manual intervention

- Verification:
  - `python3 -m compileall src/training/perception_seam_*.py -q`
  - `python3 -m pytest tests/test_perception_seam_training.py tests/test_perception_grounding_neural_seams.py -v`
  - result: `63 passed` (26 new + 37 existing)

### 2026-04-08: Integrated Sim / Synth / Physics doctrinal pass

- Updated the owning docs only:
  - `docs/economic_world_model/multi_wm_architecture_plan.md`
  - `docs/economic_world_model/roadmap.md`
  - `docs/actuation_embodiment_world_model.md`
- Unified Newton, UnrealRoboticsLab, Habitat-style scene/materialization
  learnings, and WinDiNet-style surrogate physics into one Sim / Synth /
  Physics Phase 1.x doctrine pass instead of separate architecture waves.
- Made the Sim↔Embodiment transfer boundary explicit while keeping provider
  doctrine in Sim / Synth and remap/deployment adaptation ownership in
  Embodiment.
- Scope stayed docs-only: no code changes and no new standalone architecture
  docs.

### 2026-04-08: Sim-to-online stabilization doctrine pass

- Added `docs/economic_world_model/doctrine_sim_to_online_stabilization.md` to
  record what Ixion borrows from the sim-to-online RL paper and what it
  explicitly does not borrow.
- Updated the owning docs so future sim-to-online stabilization work is legible
  inside the current topology:
  - `roadmap.md`
  - `multi_wm_architecture_plan.md`
  - `actuation_embodiment_world_model.md`
  - `implementation_notes.md`
- Kept the pass additive and downstream-facing:
  - no topology rewrite
  - no SAC-centered repo pivot
  - no code changes
  - no fake claim that real-robot online RL is the current bottleneck

### 2026-04-08: UE5 / Unreal provider-family doctrine pass

- Added `docs/economic_world_model/doctrine_unreal_ue5_provider_posture.md`
  as the owning doctrine note for UE5 / Unreal posture inside Ixion.
- Updated the existing owning docs so UE5 is legible as a major Sim / Synth /
  Physics provider family without becoming the stack ontology:
  - `multi_wm_architecture_plan.md`
  - `roadmap.md`
  - `actuation_embodiment_world_model.md`
  - `implementation_notes.md`
- Tightened the doctrine around:
  - UE5 capability placement by subsystem
  - hybrid backend posture
  - provider-not-truth-owner boundaries
  - sensor/timing and transfer-boundary implications for Embodiment
  - later headless / cloud / industrial twin execution posture
- Kept the tranche additive and staged:
  - no WM ordering rewrite
  - no Unreal-centered repo pivot
  - no code changes
  - no fake active backlog items where no honest runtime entrypoint exists yet

### 2026-04-09: Nightly verification hardening (CLAUDE shim template)

- Completed the nightly-selected additive task `agent_verify_regression` by
  removing shim drift between `CLAUDE.md` and agent scripts.
- Added a canonical shim template:
  - `scripts/agent/claude_shim_template.md`
  - includes `@.agent/claude_copilot.md` so the current collaboration posture is
    explicitly represented in the enforced shim.
- Updated both agent ergonomics scripts to consume the canonical template
  instead of duplicated inline shim strings:
  - `scripts/agent/verify.sh`
  - `scripts/agent/bootstrap.sh`
- Added regression tests:
  - `tests/test_agent_shim_template.py`
  - asserts `CLAUDE.md` matches the canonical template and the copilot import is
    present.
- Verification run for this tranche:
  - `./scripts/agent/verify.sh` (pass)
  - `python3 -m compileall src/` (pass)
  - `python3 -m pytest -q tests/test_agent_shim_template.py tests/test_economic_world_model_nightly_audit.py` (pass; `9 passed`)
  - `python3 scripts/economic_world_model/nightly_audit.py --output-json artifacts/economic_world_model/nightly_audit_summary.json --output-markdown artifacts/economic_world_model/nightly_audit_summary.md` (pass)
- Refreshed nightly audit status is now `ok`, with all verification gates
  passing.
- Next recommended task from the refreshed audit:
  - `docs_only`: no missing additive scaffold detected; keep nightly verification
    and docs refresh only until a higher-priority gap appears.
