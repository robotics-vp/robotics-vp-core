# Sim-to-Online Stabilization Doctrine

## Purpose

This document records what Ixion borrows from the paper
_What Matters for Sim-to-Online Reinforcement Learning on Real Robots_
([arXiv:2602.20220](https://arxiv.org/abs/2602.20220)) and, just as
importantly, what it does **not** borrow.

This is a future execution and training doctrine inside the existing Ixion
stack. It is **not** a topology document, not a sequencing override, and not a
license to collapse the stack into "sim pretrain + online SAC finetuning."

## What We Borrow

The useful lessons are practical stabilization doctrines:

- sim-to-online transfer can fail through a real instability mechanism rather
  than only through coarse "transfer worked / failed" narratives
- retained prior data and provenance-aware replay mixtures can stabilize online
  adaptation
- warm-start collection before updates can matter when prior data is not
  retained
- asymmetric actor/critic update schedules can materially improve stability
- incomplete checkpoint restore can silently invalidate transfer conclusions
- massively parallel simulation quality is a real transfer precondition, not
  just a throughput trick
- real-hardware adaptation is often best treated as asynchronous, episodic, and
  receipt-driven rather than per-step synchronous optimization

## What We Explicitly Do Not Borrow

We are **not** borrowing:

- a sovereign architecture for the stack
- a replacement for the multi-WM decomposition
- a claim that robotics should be centered on SAC specifically
- a claim that current repo priorities should pivot away from structural lower-WM
  work toward immediate real-robot finetuning
- an ontology in which the stack is mainly "sim pretraining plus online RL"

Ixion keeps its own topology:

- Sim / Synth / Physics WM owns simulation-side assumptions, calibration state,
  branch/transfer receipts, and sim-real gap evidence
- Embodiment / Actuation WM owns deployment-side drift, remap, capability
  filtering, action-feasibility truth, and local adaptation
- Economic WM later consumes transfer cost, stability, and yield evidence, but
  does not become the owner of transfer mechanics or transfer physics truth

## Why This Is Downstream of Current Work

This doctrine matters because later real-hardware learning can silently unlearn
what simulation produced if the stack treats transfer as a vague success/fail
story. But this is downstream of the current active bottlenecks:

- Sim / Synth / Physics WM structural closure and provider truth
- Perception / Grounding benchmark evidence and provider-backed token paths
- later Embodiment / Actuation buildout

So the right move now is **docs and contract preparation**, not a premature
online-RL implementation campaign.

## Failure Mode

The failure we care about is:

1. a policy is pretrained in simulation
2. it is deployed onto real hardware under distribution shift
3. value/critic error grows on the newly visited state-action distribution
4. biased updates push the actor toward unstable behavior
5. the system enters a downward spiral and silently unlearns the useful prior

In Ixion terms, this is not just "training instability." It is a
cross-boundary transfer failure involving:

- simulation assumptions and domain randomization
- transfer-risk calibration
- deployment-side mismatch and action-feasibility degradation
- replay composition and update policy
- resume / checkpoint integrity

## Mitigation Classes We Care About

### 1. Retained Prior-Data / Replay-Mixture Discipline

Retaining useful prior data can regularize online adaptation, but Ixion should
not flatten this into "keep all old data forever." The relevant doctrine is a
**provenance-aware replay-mixture policy** that distinguishes:

- simulation-pretraining data
- retained prior real-world data
- new online adaptation data
- recent-window versus older-window online data

The important object is not just a replay buffer. It is a typed policy for how
those sources are mixed, annealed, audited, and tied back to transfer outcomes.

### 2. Warm-Start Collection Discipline

When prior data is unavailable, the system should be able to warm-start the
online buffer by collecting some number of deployment episodes before updates.
For Ixion this should later become an explicit `WarmStartPolicy`, not an
implicit experimenter habit.

### 3. Asymmetric Actor/Critic Updates

Less frequent actor updates and lower actor learning rates can stabilize
transfer. In Ixion this should become a future typed training-manifest field,
not a repo-wide SAC dogma. The doctrine generalizes beyond SAC:

- policy/value update cadence should be explicit
- slower policy adaptation under transfer stress should be expressible
- benchmark receipts should record which update schedule was actually used

### 4. Checkpoint-Completeness Discipline

The paper calls out a practical failure mode that matters directly to Ixion:
restoring model weights without restoring the rest of the training state can
invalidate transfer conclusions.

For Ixion, checkpoint completeness should later cover:

- optimizer state
- target-network state where relevant
- entropy/temperature state where relevant
- scheduler / normalization / auxiliary-state continuity
- replay provenance continuity where the training regime depends on it

This should generalize beyond SAC rather than remaining SAC-only doctrine.

### 5. Massive-Parallel Sim Pretraining Quality as a Transfer Precondition

Better transfer is not just about what happens online. Simulation-side
pretraining quality, diversity, calibration, and randomization density shape
whether online adaptation starts from a robust prior or from a brittle one.
This is a Sim / Synth / Physics WM precondition, not an afterthought owned by
later deployment logic.

### 6. Asynchronous Episodic Real-Hardware Update Discipline

Real-hardware training often does not look like neat per-step synchronous
optimization. For Ixion this fits naturally into the repo's existing
receipt/replay discipline:

- collect bounded episodes or windows
- emit receipts
- run updates asynchronously off the control-critical path
- preserve training-window provenance

### 7. Explicit Sim-Real Gap and Transfer-Stability Receipts

We do not want future transfer evaluation to collapse into vague status labels.
The stack should later emit explicit receipts for:

- sim-real gap
- transfer-risk summary
- replay-mixture policy used
- warm-start posture used
- checkpoint completeness
- realized deployment-side drift and degradation
- transfer stability across online windows

## Fit Inside Ixion Topology

### Sim / Synth / Physics WM

Owns:

- simulation-side assumptions
- domain-randomization regime
- calibration proposals
- sim-real gap evidence
- branch receipts and transfer-risk summaries
- training-worthiness under transfer instability

### Embodiment / Actuation WM

Owns:

- realized local drift after transfer
- deployment-side mismatch
- remap / retarget / capability filtering
- action-feasibility degradation
- local recovery and degradation posture

### Economic WM

Later consumes:

- transfer cost
- stability evidence
- replay/training yield
- adaptation spend versus gain

But it does **not** become the owner of transfer mechanics or transfer physics
truth.

## Future Typed Artifacts / Manifest Fields / Receipts

The following should later become typed surfaces. Reserving them now is not a
claim that the full online RL loop should be built immediately.

- `ReplayMixturePolicy`: provenance-aware mixture/annealing policy across
  simulation, retained prior, and online data
- `WarmStartPolicy`: number/type of pre-update collection windows, gating
  conditions, and provenance tags
- `CheckpointCompletenessReceipt`: what exact training state was restored,
  missing fields, and resume risk
- `ActorCriticUpdateSchedule`: update cadence, asymmetry, and any slower-policy
  or lower-LR posture
- `TransferStabilityReceipt`: evidence about whether online adaptation is
  preserving or degrading transfer performance
- `SimOnlineTrainingWindow`: bounded training/update window keyed to the
  receipts and buffers it used
- `OnlineAdaptationEpisodeReceipt`: per-episode adaptation record for
  real-hardware collection windows

Closely related lower-WM surfaces should include existing and adjacent receipts:

- `SimRealGapReceipt`
- `PhysicsAdaptationReceipt`
- `BackendMismatchReceipt`
- `DeploymentTransferDriftReceipt`
- `ActionFeasibilityDegradationReceipt`
- `EmbodimentTransferOutcomeReceipt`
- `ControllerLatencyMismatchReceipt`

## What We Are Explicitly Not Doing Now

- not changing WM order
- not centering the stack on SAC
- not treating online RL on real robots as the current bottleneck
- not rewriting the active roadmap around real-robot finetuning
- not letting this paper redefine the repo's sovereign architecture

The current task is to make the stack **doctrinally and contractually ready**
for later sim-to-online stabilization work while the real near-term focus
remains structural lower-WM readiness.
