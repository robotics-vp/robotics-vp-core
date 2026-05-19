# Phase 1.x Closure Assessment — 2026-05-19

## Verdict

Phase 1.x is **structurally closure-ready on audited local surfaces**.

- **Category A**: `0`
- **Category B**: remaining blockers are external provider / GPU / asset /
  calibration / benchmark / native-runtime evidence items
- **Category C unresolved**: `0`
- **Maturity floor reached**: `shadow_runtime` with local deploy-smoke evidence
  for the Holosoma ONNX policy path

This does **not** mean Isaac/Unitree provider readiness, full Holosoma runtime
readiness, benchmark promotion readiness, or hardware readiness. It means the
current repo no longer has an identified local structural gap that should keep
the Sim / Synth / Physics Phase 1.x return leg open before Phase 3 spec work can
begin.

## Evidence snapshot

Latest local runtime-layout scan:

```bash
python3 scripts/scan_phase1_runtime_layouts.py \
  --output-path artifacts/sim_synth_runtime_layout_scan.json
```

Key scan results:

| Lane | Current local status | Interpretation |
|------|----------------------|----------------|
| Holosoma | `pack_ready`, `binding_ready`, `preflight_ready`; selected policy is local ONNX `fastsac_g1_29dof.onnx`; ready flags include `policy_ready`, `motion_train_ready`, and `sim_launch_ready` | Local deploy/action smoke is real enough to keep the lane unblocked for cheap checks; full native simulated episodes remain provider/runtime evidence, not promotion evidence |
| Isaac / Unitree | `pack_partial`, `binding_partial`, `preflight_blocked`; selected policy is local `unitree_rl_gym/deploy/pre_train/g1/motion.pt`; missing `actuator_latency_profile` and `safety_watchdog_profile` | The repo has concrete local upstream roots/assets/policy visibility, but true latency/watchdog profiles should not be invented locally; remaining gap is external calibration/runtime truth |

The Holosoma local smoke now has a no-pip shim bootstrap and action-level ONNX
proof. The Isaac / Unitree lane now distinguishes real visible local surfaces
from true deployment blockers instead of collapsing the entire lane into generic
provider absence.

## Closure sheet

| Finding | Category | Rationale | Evidence |
|---------|----------|-----------|----------|
| Canonical Sim / Synth / Physics state compiles with Phase 1.x subsystem identity | resolved A | `phase1x_subsystem_index_v1` maps all ten Phase 1.x subsystems into compiled state metadata and preserves ownership, typed surfaces, receipt families, provider families, promotion gates, and honest blockers | `src/world_model/sim_synth_physics/subsystems.py`, `src/world_model/sim_synth_physics/compiler.py`, `tests/test_sim_synth_phase1x_subsystems.py` |
| Trainer-row projection preserves subsystem identity | resolved A | backend-selector and branch-planner rows carry subsystem index ID, coverage summary, subsystem IDs, ownership rule, structural status, and honest blocker class | `src/world_model/sim_synth_physics/training_corpus.py`, `tests/test_sim_synth_phase1x_subsystems.py` |
| Runtime receipt manifests and validation exist | resolved A | emitted receipt families, artifact refs, optional absences, missing-required checks, and harvested bundle counts are consolidated and validated before training use | `src/world_model/sim_synth_physics/*manifest*`, `tests/test_sim_synth_training_corpus.py`, `tests/test_sim_synth_physics_world_model.py` |
| Trainer-side admissibility is enforced | resolved A | receipt rows are classified as positive training, negative supervision, or diagnostic-only; positive helper losses exclude diagnostic/negative rows until negative losses exist | `src/world_model/sim_synth_physics/training_corpus.py`, `scripts/train_sim_synth_backend_selector.py`, `scripts/train_sim_synth_branch_planner.py` |
| Negative-supervision sidecars and bounded reject heads are live | resolved A | excluded rows persist as JSONL/Regal artifacts and train bounded reject-probability heads without overriding runtime decisions | `scripts/train_sim_synth_backend_selector.py`, `scripts/train_sim_synth_branch_planner.py`, `tests/test_train_sim_synth_backend_selector.py`, `tests/test_train_sim_synth_branch_planner.py` |
| Promotion/precondition gate exists for Phase 1.x helper packages | resolved A | `phase1x_training_gate_v1` requires selected-row consistency, clean manifest validation, diagnostic exclusion, and reject-head coverage where negative sidecars exist | `src/world_model/sim_synth_physics/training_corpus.py`, helper trainer tests |
| Holosoma local proof is reproducible without heavy provider install | resolved A | setup script installs/dry-runs/removes a user-site path shim; ONNX deploy smoke proves actor observation to finite action output without claiming native episode execution | `scripts/setup_holosoma_local_smoke.py`, `scripts/local_holosoma_smoke.py`, `tests/test_setup_holosoma_local_smoke.py`, `tests/test_local_holosoma_smoke.py` |
| Runtime layout scanner captures host truth for Isaac/Unitree and Holosoma | resolved A | scan now reports pack, binding, preflight, selected policy, usable profiles, and missing components instead of forcing manual interpretation | `scripts/scan_phase1_runtime_layouts.py`, `tests/test_scan_phase1_runtime_layouts.py`, `artifacts/sim_synth_runtime_layout_scan.json` |
| Isaac / Unitree missing latency and safety-watchdog profiles | C→B | superficially writable YAML would be worse than absence; truthful values require deployment/runtime calibration and safety-envelope evidence. The typed slots and scan diagnostics already exist. | runtime-layout scan, `tests/test_scan_phase1_runtime_layouts.py` |
| Isaac / Unitree native runtime execution and deployment-mode confirmation | B | requires provider install/runtime execution, likely GPU/runtime host work, and non-fake deployment traces | runtime/provider backlog |
| Full Holosoma native simulated episode evidence | B | the current proof is ONNX deploy/action smoke; native Holosoma eval episode requires the upstream runtime/config/checkpoint shape and later provider evidence | `scripts/local_holosoma_smoke.py`, runtime backlog |
| GGDS / LDM / high-fidelity video materialization | B | requires external model weights, runtime dependencies, and GPU-backed materialization evidence | provider/materialization backlog |
| Benchmark promotion evidence for learned helpers | B | promotion requires real execution outcomes and benchmark density, not more local scaffolding | training gate / benchmark backlog |
| Hardware calibration, drift, and transfer measurements for Unitree G1-class deployment | B | requires real robot/sim provider measurements; Phase 3 should own deployment-side state once this evidence exists | Embodiment / Actuation backlog |

## Remaining Category B items

| Blocker | Why it remains external | Expected owner / timing |
|---------|-------------------------|--------------------------|
| Isaac / Unitree latency and safety-watchdog profiles | Values must come from real runtime/hardware/safety evidence, not placeholder defaults | provider/GPU/hardware season |
| Native Isaac / Unitree provider execution | Requires full runtime stack and validated provider execution | provider/GPU season |
| Full native Holosoma episode execution | Requires native upstream runtime/config/checkpoint execution beyond ONNX action smoke | provider/GPU or local provider season if dependencies are safe |
| GGDS/LDM or equivalent video materialization | Requires model weights and GPU-backed materialization runs | provider/GPU season |
| Benchmark-grade promotion evidence | Requires real run outcomes and held-out evaluation density | after provider execution |
| Real sim-real / embodiment transfer measurements | Requires calibrated runtime or hardware traces | Phase 3+ provider/hardware season |

## Closure recommendation

Treat Phase 1.x as **locally structurally closure-ready**, pending owner/Claude
review. The next repo-local work should move to Phase 3 spec and then Phase 3.1
canonical Embodiment / Actuation state contracts, while preserving the Phase 1.x
external backlog for provider/GPU bring-up.

Do **not** close the external backlog. Do **not** fabricate latency/watchdog
profiles. Do **not** treat Holosoma ONNX action smoke as native episode or
benchmark promotion evidence. Do **not** enable `ROBOTICS_VP_ENABLE_HOLOSOMA_RUNTIME=1`
by default.

## Verification for this assessment

- `git diff --check && python3 -m compileall src/`
- `python3 -m pytest tests/test_scan_phase1_runtime_layouts.py tests/test_setup_holosoma_local_smoke.py tests/test_local_holosoma_smoke.py tests/test_sim_synth_phase1x_subsystems.py -q`
  - `12 passed`
- `python3 scripts/economic_world_model/nightly_audit.py --output-json artifacts/economic_world_model/nightly_audit_summary.json --output-markdown artifacts/economic_world_model/nightly_audit_summary.md`
  - `status: ok`
- `python3 -m pytest tests/ -q`
  - `1647 passed, 2 skipped, 24 warnings`
