# GR00T VisualSim2Real Borrowing Doctrine

## Executive Summary

GR00T / VIRAL / DoorMan is a concrete sim-to-real training plant: Hydra
experiment composition, Isaac-backed humanoid training, privileged teacher
policies, vision students, domain randomization, demonstration-seeded resets,
checkpoint/eval/export flows, and run-directory discipline.

Ixion is a typed multi-WM cybernetic/economic architecture: Perception /
Grounding, Embodiment / Actuation, Sim / Synth / Physics, Economic,
Meta-Regal / governance, and WM transport remain separate canonical state
owners connected by typed receipts.

The useful extraction is operational discipline, not ontology. GR00T is a
strong pattern source for training/eval/config/promotion surfaces. It is not
the architecture, not a replacement for the multi-WM topology, and not a
reason to make Isaac Lab, PPO, DAgger, ResNet, or ONNX sovereign across the
stack.

## Phase-Sequencing Posture

This note preserves Phase 2 Perception / Grounding as the active
implementation center.

After Phase 2, the roadmap returns to Sim / Synth / Physics Phase 1.x because
additional Sim/Synth/Physics obligations were added after Phase 1 structural
closure: subsystem decomposition, provider-family placement, run manifests,
runtime/materialization discipline, sim-to-online preparation, and
Sim-to-Embodiment transfer receipts. That later return is not a switch back
to Phase 1 right now.

Near-term Phase 2 can borrow only observation, provider-truth,
randomization-provenance, receipt, and benchmark discipline. The heavier
teacher/student sim-to-real training patterns belong primarily to the later
Phase 1.x Sim/Synth/Physics revisit and Phase 3 Embodiment prep.

## Anti-Overfit Rules

- GR00T is not our ontology.
- Isaac Lab / Isaac Sim are provider/backend lanes, not truth owners.
- PPO is one teacher-training example, not the repo-wide RL mandate.
- DAgger is one student-distillation example, not the only imitation route.
- ResNet is one deployable vision backbone example, not the Perception WM
  backbone decision.
- ONNX is one export artifact family, not the only deployment contract.
- G1 task primitives from GR00T do not become Ixion primitive ontology.
- Experiment directories are useful evidence containers, but typed receipts
  and manifests remain the canonical decision surfaces.
- No learned teacher, simulator, or deployment artifact bypasses promotion
  gates, provider truth, benchmark evidence, or economic/governance receipts.

## Pattern Extraction Table

| GR00T pattern | Portable Ixion extraction | Candidate typed contract | Primary WM home | Current posture |
|---|---|---|---|---|
| Hydra-composed experiment specs | Composable run specs with explicit algo/env/robot/reward/obs/domain-rand/callback inputs | `TrainingRunManifest` | Sim / Synth / Physics, Embodiment | Doctrine now, implementation later |
| Privileged-state PPO teacher | Privileged teacher lane that may use sim-only state but exports bounded traces and receipts | `TeacherStudentTrainingManifest`, teacher checkpoint ref | Sim / Synth / Physics, Embodiment | Future Phase 1.x / Phase 3 |
| Vision DAgger student | Deployable student lane distilled from teacher traces and constrained observations | student checkpoint ref, observation profile ref | Perception, Embodiment | Phase 2 observation discipline now; training later |
| RGB-delay student config | Degraded-observation and latency surfaces as first-class eval axes | observation-delay profile, degraded-observation receipt | Perception / Grounding | Phase 2 borrow now |
| Camera enablement and egocentric RGB | Sensor bundle profiles with camera resolution, placement, timing, and provider truth | camera observation bundle, egocentric sensor profile | Perception / Grounding | Phase 2 borrow now |
| Camera extrinsics randomization | Randomization as typed provenance, not hidden augmentation | `DomainRandomizationProfile`, `PerceptionCalibrationReceipt` | Perception, Sim / Synth / Physics | Phase 2 docs now, Phase 1.x implementation later |
| Visual/depth augmentation blocks | Augmentation provenance attached to training/eval evidence | visual augmentation profile ref | Perception / Grounding | Phase 2 borrow now |
| Reset-from-dataset curricula | Demonstration-seeded resets and staged curricula as typed profiles | `DatasetResetProfile` | Sim / Synth / Physics, Embodiment | Future Phase 1.x / Phase 3 |
| Checkpoint save and autoresume callbacks | Restart-complete run ledger and checkpoint provenance | teacher/student checkpoint refs, checkpoint completeness receipt | Run ledger, Sim, Embodiment | Future manifest fields |
| Eval callback and metrics JSON | Evaluation gates emit machine-readable metrics before promotion | `EvalExportGate`, task measurement receipt | Sim / Synth / Physics, Embodiment | Future gates |
| ONNX export during eval | Deployment artifact gate after eval, not promotion by existence | export artifact ref, deployment candidate status | Embodiment, deployment lane | Future gate |
| W&B and experiment directories | Measurement/logging callbacks become receipt emitters or manifest attachments | `TaskMeasurementSurfaceEmitter` | Run ledger, Economic WM | Future receipt mapping |
| G1 config/action-space discipline | Robot config, joint limits, action dimensions, primitive maps become body state | capability/action-space profile | Embodiment / Actuation | Phase 3 prep |

## Mapping by World Model

### Sim / Synth / Physics WM

GR00T is most useful here during the later Phase 1.x revisit. Borrow:

- composable experiment specs for backend, task, robot, reward, observation,
  domain randomization, callbacks, checkpoints, and export gates
- privileged teacher training lanes as sim-side training plants
- domain randomization as typed provenance, including physics, camera,
  lighting, material, delay, and control perturbation axes
- dataset-reset and demonstration-seeded curricula as explicit profiles
- eval/checkpoint/export gates as promotion preconditions
- callbacks that emit task measurement, sim-real gap, and training-worthiness
  receipts

Do not borrow Isaac sovereignty. Isaac Lab / Isaac Sim may execute the lane,
but the WM owns agenda, provider truth, domain-randomization policy, replay
provenance, transfer-risk summaries, and admission/training-worthiness.

### Perception / Grounding WM

Phase 2 can borrow GR00T-style deployable observation discipline now:

- camera observation bundles with resolution, modality, timing, placement,
  and provider truth
- egocentric sensor profiles as future humanoid-facing inputs
- extrinsics randomization receipts instead of hidden camera perturbations
- RGB delay, dropped frames, degraded observation, and latency profiles as
  explicit evaluation surfaces
- visual/depth augmentation provenance linked to seam training and benchmark
  evidence

This strengthens the current Phase 2 stack without changing priorities:
embodiment-facing shadow consumption remains the highest usefulness lens;
provider truth and receipt emission remain the next discipline layer; cheap
prototype-train proof-of-life stays bounded; promotion claims remain held
until benchmark and GPU evidence exist.

### Embodiment / Actuation WM

Borrow the teacher-to-student training shape and the G1-facing config
discipline, but route them through the six Embodiment subsystems:

- privileged sim teachers produce bounded demonstrations, affordance/contact
  traces, and action priors
- deployable students consume Perception-owned observation bundles and
  Embodiment-owned capability/action state
- dataset-reset curricula seed local contact and retargeting regimes
- export artifacts must pass evaluation/export gates before deployment claims
- G1 action spaces, joint names, limits, primitive maps, and sensor placement
  become typed embodiment state, not global primitive ontology

### Economic WM

The Economic WM should consume GR00T-like run outputs only after lower WMs emit
typed receipts. It may value:

- teacher/student training cost
- domain-randomization contribution
- dataset-reset curriculum yield
- eval/export pass/fail evidence
- sim-real gap reduction
- deployment-artifact cost, storage, runtime latency, and operational value

It does not own teacher training, policy choice, randomization mechanics, or
export mechanics.

### Meta-Regal / Governance WM

The Meta-Regal layer should govern promotion and admissibility, not training
mechanics. It should use GR00T-derived surfaces to ask:

- was the teacher privileged, and are those privileges absent from deployment
  claims?
- is the student evaluated under the observation, delay, and degradation
  profile it will face?
- are domain-randomization and dataset-reset profiles recorded?
- is checkpoint/export evidence complete enough to carry the claimed
  epistemic status?
- are benchmark, safety, and economic gates satisfied before promotion?

### WM Transport / Bridge Layer

GR00T-style transfer discipline belongs in bridge receipts, not in a
mother-latent:

- Sim-to-Embodiment transfer receipts record sim-side randomization,
  dataset-reset, teacher checkpoint, student checkpoint, export, and eval
  gates.
- Perception-to-Embodiment bridge receipts record observation bundle,
  egocentric profile, extrinsics/randomization provenance, and delayed or
  degraded observation posture.
- Economic bridge receipts carry cost/yield summaries derived from lower-WM
  receipts.

The transport layer translates typed evidence between WMs. It does not become
the first owner of sim, perception, embodiment, or economic truth.

## Mapping by Embodiment Subsystem

### 1. Capability / Embodiment State Surface

Borrow G1-facing config discipline: robot family, joint names, DOF count,
joint limits, action-space dimensions, hand/action primitive maps, sensors,
camera placement, and compute/deployment posture. These become typed
capability state and action-space profiles.

Do not import GR00T's task/action primitive names as Ixion ontology. They are
examples that compile into local body/action contracts.

### 2. Contact / Affordance Graph Builder

Borrow sim-trained affordance/contact teacher traces where useful. Teacher
policies with privileged state can label contact preconditions, approach
quality, grasp/hold feasibility, obstruction, slip, and stage transitions for
shadow graph evaluation.

The graph builder remains the owner of local actionable contact truth. Teacher
outputs are evidence, not graph truth.

### 3. Local Contact Dynamics Model

Borrow privileged teacher rollouts, randomized physics profiles, and
degraded-observation tests as training/eval slices for short-horizon contact
dynamics. RGB delay and control delay should be treated as explicit stress
profiles because local dynamics and action feasibility depend on timing.

Do not collapse this subsystem into a full teacher policy. It predicts local
contact evolution and risk; it is not the whole controller.

### 4. Inverse-Dynamics / Retargeting Lane

Borrow reset-from-dataset and demonstration-curriculum discipline:
demonstrations, teleop traces, sim rollouts, and real robot logs should enter
through typed dataset-reset profiles and retargeting receipts.

Teacher/student traces can seed inverse dynamics and retargeting, but source
embodiment, target embodiment, kinematic feasibility, failure points, and
quality scores must be explicit.

### 5. Joint Skill / Action Proposal Head

Borrow the deployable student posture: action proposals may be distilled from
privileged teachers but must be evaluated as students under deployable
observation surfaces. Export artifacts are deployment candidates only after
evaluation gates pass.

PPO/DAgger/ResNet are examples. The Ixion head may use ACT-style chunking,
diffusion proposal heads, inverse-dynamics priors, or other bounded seams.

### 6. Drift / Calibration / Cost Evaluator

Borrow the measurement and eval loop discipline: record sim-real gap,
calibration drift, observation-delay sensitivity, randomization profile,
checkpoint provenance, export artifact, eval metrics, runtime latency, and
cost. These become `EmbodimentDriftReceipt`, `SimRealGapReceipt`, and
economic run receipts downstream.

## Candidate Contract Names

These are documentation targets, not implementation claims:

- `TrainingRunManifest`
- `TeacherStudentTrainingManifest`
- `DomainRandomizationProfile`
- `DatasetResetProfile`
- `EvalExportGate`
- `TaskMeasurementSurfaceEmitter`
- `SimRealGapReceiptEmitter`
- `EmbodimentDriftReceiptEmitter`
- `PerceptionCalibrationReceiptEmitter`
- `EconomicRunReceiptEmitter`

Future run manifests may cross-reference:

- `teacher_checkpoint_ref`
- `student_checkpoint_ref`
- `domain_randomization_profile_ref`
- `dataset_reset_profile_ref`
- `eval_export_gate_ref`

See `docs/agent_ergonomics/run_manifest_schema.md` for the existing remote-run
ledger. That schema remains the ledger anchor; this note only names future
sim-to-real fields that can attach to it.

## Do Not Borrow

- Do not replace Ixion's multi-WM topology with GR00T's training stack.
- Do not make Isaac Lab, Isaac Sim, or Isaac Gym the owner of truth.
- Do not make PPO the required teacher algorithm.
- Do not make DAgger the required student algorithm.
- Do not make ResNet the required deployable vision backbone.
- Do not treat ONNX export as deployment readiness by itself.
- Do not treat W&B logs or experiment directories as canonical receipts.
- Do not collapse privileged teacher observations into deployable student
  claims.
- Do not route economic reward directly into low-level control because GR00T
  uses concrete reward configs.
- Do not import GR00T task primitives as our primitive ontology.
- Do not create a new mother-latent for sim-to-real transfer.

## Adoption Order

1. Documentation and contract vocabulary.
2. Phase 2 observation and receipt discipline where it directly supports
   Perception / Grounding.
3. Phase 1.x Sim / Synth / Physics profile and receipt surfaces after Phase 2.
4. Phase 3 Embodiment teacher/student, reset curriculum, and export-gate
   surfaces.
5. Economic and Meta-Regal consumption after lower-WM receipts are real.

This keeps GR00T useful as a concrete sim-to-real training/eval/config
discipline while preserving Ixion's typed WM topology.
