# Multi-WM Unwired Surface Audit - 2026-06-01

## Scope

This audit answers what remains unwired after the local bio/neuro substrate
pass. It is a repo-grounded claim-vs-code audit, not a GPU/provider/hardware
proof claim.

Baseline before this pass:

- `270ce19 docs: add wm subsystem debt sweep`

Local checks used:

```bash
python3 scripts/economic_world_model/nightly_audit.py \
  --output-json /tmp/nightly_audit_bio_neuro.json \
  --output-markdown /tmp/nightly_audit_bio_neuro.md

python3 scripts/economic_world_model/check_bio_neuro_substrate.py \
  --output-dir /tmp/bio_neuro_substrate_check

python3 -m ruff check \
  src/world_model/embodiment_actuation/bio_neuro_surfaces.py \
  src/world_model/perception_grounding/bio_neuro_receipts.py \
  src/world_model/economic_world_model/regime_broadcast.py \
  src/regal/bio_neuro_anomaly.py \
  scripts/economic_world_model/check_bio_neuro_substrate.py \
  tests/test_bio_neuro_substrate.py

python3 -m pytest -q tests/test_bio_neuro_substrate.py
```

Observed results:

- nightly audit: `status=ok`; no higher-priority missing additive scaffold
  detected by the existing nightly scan
- bio/neuro substrate check: `status=ok_bio_neuro_substrate_passed`;
  `surface_count=14`; `provider_or_hardware_proof=false`;
  `trained_model_proof=false`; `promotion_eligible=false`
- targeted ruff: pass
- targeted pytest: `5 passed`
- training backlog JSON validation: pass

## Bio / Neuro Substrate Status

| Principle | Local substrate now wired | Still not proven / not wired |
| --- | --- | --- |
| Efference copy | `SelfMotionExpectation` in Embodiment and `SelfDisturbanceReceipt` in Perception | no trained predictor, no real observed-motion corpus, no automatic live-loop emission |
| Active sensing | `ActiveSensingProposal` in Embodiment and `ActiveSensingReceipt` in Perception | no executed active-sensing action, no measured information gain, no full Economic WM value-of-information shaper |
| Neuromodulation / allostasis | `RegimeBroadcast` and `RegimeAcknowledgmentReceipt` in Economic WM | no trained switching-SSM regime estimator, no real downstream adaptation, no authority |
| Plasticity gating | Existing Perception promotion gates remain the strongest implementation | no stack-wide consolidation receipt family across all WMs |
| Motor synergies + interoception | `SynergyCodebookEntry` and `InteroceptiveState` in Embodiment | no learned synergy codebook, no real interoceptive telemetry corpus, no hardware-calibrated activation patterns |
| Immune-style anomaly | `AnomalySuspicionReceipt` and `GovernanceEscalationEvent` in Regal governance | no trained domain anomaly critics, no meta-regal immune-style composition |

The future training lanes for these surfaces are tracked in
`scripts/TRAINING_MIGRATION_BACKLOG.json` as:

- `train_self_motion_expectation_v0.py`
- `train_active_sensing_policy_v0.py`
- `train_economic_regime_broadcast_v0.py`
- `train_embodiment_synergy_interoception_v0.py`
- `train_regal_anomaly_governance_v0.py`
- `train_plasticity_consolidation_gates_v0.py`

## Claim vs Code Audit

| Roadmap surface | Code / artifact evidence | Status |
| --- | --- | --- |
| G1/humanoid primary posture | `configs/humanoid/g1_primary_env.yaml`, `src/world_model/humanoid_readiness/g1_primary_environment.py`, G1 hygiene gate | locally wired; no hardware/sim proof |
| Sim/Synth/Physics Phase 1.x subsystem split | `src/world_model/sim_synth_physics/` subsystem modules, runtime targets, backend adapters, branch/admission receipts | locally structural; real Isaac/Unitree/Holosoma execution remains external |
| Perception canonical state and bridge family | `src/world_model/perception_grounding/state.py`, `semantic_bridges.py`, compiler, receipts, benchmark evidence | locally structural; real SAM/DINO/SigLIP/V-JEPA/depth provider execution remains external |
| Embodiment/Actuation G1 scaffolds | G1 morphology, bipedal chassis/readiness, Phase 4 Unitree harnesses, bio/neuro substrate | locally structural; no live Unitree command, DDS echo, hardware calibration, or trained whole-body policy |
| Economic WM local scaffold | `scaffold.py`, `resource_surfaces.py`, `phase5_local_prep.py`, `shadow_execution.py`, `lower_wm_maturity_sweep.py`, `regime_broadcast.py` | locally structural; no trained estimator/dynamics/allocator/governance model |
| Cross-WM transport | `src/world_model/transport/` contracts, rows, topology, uncertainty, neural manifest, advisory runtime | locally structural; no trained bridge/receiver, latency benchmark, or provider/hardware transport evidence |
| Phase 6.5 meta-node | `src/world_model/humanoid_readiness/phase65.py`, `phase65_trainer.py`, scaffold scripts/tests | locally structural; no trained meta-node weights or heldout robustness proof |
| Phase 7 meta-regal control | `phase7.py`, `phase7_runtime.py`, `phase7_eval.py`, `phase7_signal_adapters.py`, `phase7_hypernetwork.py` | shadow/advisory scaffold; no authority, no trained composition, no live lower-WM proof |
| RunPod execution plane | `scripts/runpod/*`, `src/runpod/launch_profiles.py`, `configs/runpod/examples/*` | manifest-ready; local launch prerequisites still external/config-bound |

## Still-Unwired Or Externally Blocked Surfaces

### 1. Provider-backed perception stack

What remains:

- SAM 3 / 3.1 or SAM3D runtime-backed object/video tracking lane
- DINOv2 / SigLIP / MetaDINO replacement for deterministic latent defaults
- V-JEPA 2 upstream runtime/provider lane for Perception temporal grounding
- provider-backed benchmark tokens feeding promotion gates

Why it matters:

- Perception has the typed state and promotion discipline, but provider truth
  is still mostly unavailable or injected for local tests.

Verify when ready:

```bash
python3 scripts/runpod/prepare_launch_manifest.py --profile provider_bringup
python3 -m pytest -q tests/test_perception_grounding_world_model.py tests/test_perception_grounding_neural_seams.py
```

Do not:

- treat stub or injected provider tokens as promotion-grade perception truth.

### 2. Real Sim/Synth/Physics runtime execution

What remains:

- concrete Isaac Lab / Isaac Sim / Unitree asset execution
- concrete Holosoma runtime host/policies/motion data
- real SDS/LDM rendering at scale
- calibrated runtime outcome receipts from actual providers

Why it matters:

- Phase 1.x is structurally rich, but the roadmap's current implementation
  priority still depends on reducing Sim/Synth provider/runtime debt before GPU
  loop work should be treated as proof.

Verify when ready:

```bash
python3 scripts/economic_world_model/probe_phase4_unitree_blockers.py --output-dir /tmp/unitree_probes
python3 -m pytest -q tests/test_isaac_unitree_runtime_binding.py tests/test_holosoma_runtime_binding.py
```

Do not:

- collapse PyBullet/shadow-contract success into Isaac/Unitree/Holosoma truth.

### 3. RunPod launch prerequisites

What remains:

- install/auth `runpodctl`
- set `RUNPOD_API_KEY`
- set `RUNPOD_VOLUME_ID`
- run provider/loop/train profiles and record `.agent/runs/<run_id>/manifest.json`

Why it matters:

- provider bring-up, loop runs, and training are now profile-shaped, but not
  launchable from this machine until local auth/volume prerequisites exist.

Verify when ready:

```bash
./scripts/runpod/ensure_cli.sh
python3 scripts/runpod/prepare_launch_manifest.py --profile provider_bringup --volume-id "$RUNPOD_VOLUME_ID"
```

Do not:

- call manifest preparation a remote run.

### 4. Training and promotion lanes

What remains:

- trained lower-WM models
- trained Economic WM estimator/dynamics/allocator/governance heads
- trained transport bridge/exporter/receiver components
- trained Phase 6.5 meta-node and Phase 7 composition models
- promotion-grade benchmark evidence

Why it matters:

- The repo now has many non-training manifests and CPU smoke forwards. Those
  are useful but not promotion evidence.

Verify when ready:

```bash
python3 scripts/runpod/assess_full_stack_training.py
python3 -m pytest -q tests/test_economic_wm_trainer_scaffold.py tests/test_wm_transport_phase63_neural_scaffold.py tests/test_humanoid_phase65_meta_node_trainer_scaffold.py
```

Do not:

- write weights or mark local scaffolds as promotion eligible.

### 5. Phase 8 production-loop operations

What remains:

- weekly GPU/provider operations loop
- backlog exhaustion governance for uncalled providers/trainers/runs
- automatic run comparison summaries over real artifacts
- routine cost/runtime/quality dashboards

Why it matters:

- Phase 8 is the layer that prevents provider and training backlogs from
  staying theoretical. It is not meaningful before provider/run prerequisites
  exist, but its manifest/run-record contracts should be kept ready.

Verify when ready:

```bash
find .agent/runs -maxdepth 3 -name manifest.json -print
python3 scripts/economic_world_model/nightly_audit.py --output-json /tmp/nightly.json --output-markdown /tmp/nightly.md
```

Do not:

- imply weekly operations have started without real run manifests.

## Ranked Next Local Actions

### 1. Burn down subsystem static debt

- **What**: execute the ranked work in
  `docs/economic_world_model/wm_subsystem_debt_sweep_2026_06_01.md`, starting
  with orchestration/semantic mypy debt.
- **Why now**: it removes noise before GPU/provider sessions.
- **Unblocks**: cleaner RunPod bring-up and loop/debug sessions.
- **Verify**: `python3 -m ruff check .`; `python3 -m mypy src/`
- **Do NOT**: change routing semantics while doing type cleanup.
- **Confidence**: high
- **Blocking**: blocks-downstream

### 2. Add a provider bring-up readiness ledger

- **What**: create a local ledger that maps each provider family
  (SAM/SAM3D, DINO/SigLIP, V-JEPA2, OpenVLA, Isaac/Unitree, Holosoma) to
  command, expected receipts, unavailable mode, RunPod profile, and owner WM.
- **Why now**: the provider backlog is spread across roadmap and JSON backlog
  files.
- **Unblocks**: first GPU/provider day without rediscovering contracts.
- **Verify**: provider ledger lint plus `provider_bringup` manifest generation.
- **Do NOT**: download weights or claim provider execution locally.
- **Confidence**: high
- **Blocking**: blocks-downstream

### 3. Wire bio/neuro substrate into normal receipt joins

- **What**: after the static debt pass, add optional joins from
  `check_bio_neuro_substrate.py` outputs into lower-WM/economic consumption
  rows.
- **Why now**: the typed surfaces exist; they should become queryable evidence
  rather than isolated smoke receipts.
- **Unblocks**: future Economic WM value-of-information and anomaly-critic
  training rows.
- **Verify**: `python3 scripts/economic_world_model/check_bio_neuro_substrate.py --output-dir /tmp/bio_neuro`
- **Do NOT**: make the joins promotion evidence.
- **Confidence**: medium
- **Blocking**: nice-to-have

### 4. Keep Phase 7 bounded

- **What**: consume the new lower-WM receipts through existing Phase 7 signal
  adapters only when receipt joins exist.
- **Why now**: this preserves the rule that Phase 7 consumes better lower-WM
  evidence instead of adding abstract vocabulary.
- **Unblocks**: later meta-regal composition training with real inputs.
- **Verify**: `python3 -m pytest -q tests/test_humanoid_phase7_signal_adapters.py tests/test_humanoid_phase7_shadow_runtime_wiring.py`
- **Do NOT**: grant authority or add new Phase 7 abstractions just because
  typed bio/neuro surfaces now exist.
- **Confidence**: high
- **Blocking**: blocks-downstream
