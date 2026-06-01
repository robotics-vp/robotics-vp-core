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
| 1 | Broad static hygiene is not clean | structural | Future GPU/provider sessions should not start with noisy lint/type failures unrelated to the run target. | Burn down ruff first by bucket, then mypy by WM. Keep commits small and avoid behavior changes unless a real bug is exposed. | `python3 -m ruff check .`; `python3 -m mypy src/` |
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

1. **RunPod prerequisite closeout**
   - What: install/auth `runpodctl`, set `RUNPOD_API_KEY`, and set
     `RUNPOD_VOLUME_ID`.
   - Why now: without this, provider/loop/train profiles are only manifests.
   - Verify: `./scripts/runpod/ensure_cli.sh`.
   - Do not: launch training before provider/loop proof-of-life manifests are
     reviewed.

2. **Orchestration / semantic type debt**
   - What: fix the largest mypy bucket:
     `src/orchestrator/pipeline_manager.py`,
     `src/orchestrator/orchestration_transformer.py`,
     `src/world_model/semantic_feedback_packets.py`, and
     `src/world_model/semantic_world_model.py`.
   - Why now: these surfaces are shared glue for multiple WMs.
   - Verify: `python3 -m mypy --follow-imports=silent src/orchestrator src/world_model/semantic_feedback_packets.py src/world_model/semantic_world_model.py`.
   - Do not: change routing semantics while doing type cleanup.

3. **Perception seam static hygiene**
   - What: fix ruff/mypy clusters in perception seam trainer/loss/data and
     neural seams.
   - Why now: provider bring-up will be easier if seam code is quiet.
   - Verify: `python3 -m ruff check src/training/perception_seam_trainer.py src/training/perception_seam_losses.py src/world_model/perception_grounding/neural_seams.py`.
   - Do not: turn provisional evidence into promotion evidence.

4. **Sim/Synth type and naming cleanup**
   - What: type `synthetic_branches.py`, `render_materialization.py`, and
     runtime target/layout payloads; add G1/curriculum aliases where naming is
     misleading.
   - Why now: Phase 1.x is the documented current implementation priority once
     local cheap hardening is useful.
   - Verify: `python3 -m pytest -q tests/test_sim_synth_runtime_targets.py tests/test_sim_synth_runtime_bundles.py tests/test_sim_synth_phase1x_subsystems.py`.
   - Do not: claim Isaac/Unitree runtime truth from local type cleanup.

5. **Embodiment/humanoid type cleanup**
   - What: fix optional/annotation mypy debt in `bipedal_chassis.py` and
     `downstream_controller.py`.
   - Why now: these are direct G1 readiness surfaces.
   - Verify: `python3 -m pytest -q tests/test_humanoid_phase35_bipedal_chassis.py tests/test_humanoid_phase4_downstream_controller.py`.
   - Do not: edit physical limits or safety thresholds without measured evidence.
