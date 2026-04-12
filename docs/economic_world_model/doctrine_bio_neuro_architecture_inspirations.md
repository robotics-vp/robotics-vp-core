# Doctrine: Bio/Neuro Architecture Inspirations for the Multi-WM Stack

## Status

Active doctrine note. No implementation implied. This document defines which
biological and computational-neuroscience organizational principles are
genuinely useful for the multi-WM robostack, where exactly they fit, what
neural architecture families they suggest, what typed surfaces they imply,
and what they must not be allowed to collapse into.

## Anti-Metaphor Framing

This document is not about copying named brain regions as replacement
ontologies. It is not a "brain-inspired architecture" proposal. The useful
imports are **multi-scale control principles** that help solve real
engineering problems in the existing WM topology.

Ground rules:

- Lower WMs remain canonical typed state owners. Nothing here changes
  Perception, Embodiment, Sim/Synth, or Economic WM ownership boundaries.
- Higher layers shape regime, budgets, and governance posture rather than
  collapsing lower realities into one latent representation.
- Every inspiration must land as a typed, receipt-emitting, bounded module
  inside an existing WM boundary — not as a new floating abstraction layer.
- The layer classification taxonomy from `neuralization_bridge_doctrine.md`
  (`WM-native`, `semantic-to-WM bridge`, `WM-to-WM transport-shaped`,
  `provider-backed interpretation`) governs where each module sits.

## Evaluation Rule for Bio Inspiration

A bio/neuro inspiration is useful for this stack **only if** it helps one or
more of:

1. typed cross-scale compression (fast/mid/slow state separation)
2. local subsystem sovereignty under global constraint
3. fast / mid / slow loop separation with multi-rate observation
4. plasticity / promotion gating (what trains, what consolidates, what stays fixed)
5. regime shaping rather than micromanagement
6. honest upward receipts / constrained downward modulation

**Explicitly reject:**

- mother-latent analogies ("this is like the thalamus routing everything
  through one hub")
- vague brain-region mapping ("this module is our hippocampus")
- any move that makes the Economic WM or a meta-regal layer silently
  redefine lower-WM truth
- any framing that implies the stack should converge toward a unified
  neural architecture that erases WM boundaries

## Candidate Inspirations

### A. Efference Copy / Corollary Discharge

**Principle.** When a motor system issues a command, it simultaneously emits
a predictive copy of the expected sensory consequences. Downstream sensory
systems use this copy to distinguish self-caused from externally-caused
state changes.

**Robostack insert point.** Perception ↔ Embodiment WM boundary.

The Embodiment WM (which already owns body state, contact state, action
proposals, and local dynamics prediction — see
`docs/actuation_embodiment_world_model.md`) should later emit a typed
**self-motion / self-disturbance expectation** into the Perception /
Grounding WM. This helps Perception with:

- temporal grounding: expected next-frame body displacement reduces
  false-positive track breaks
- track persistence: predicted self-caused occlusion events are typed, not
  latent confusion
- uncertainty attribution: Perception can distinguish "object moved" from
  "camera moved" via the expectation
- self-caused occlusion handling: robot arm occluding workspace objects is
  predicted rather than surprising

**Boundary discipline.** This must NOT cause Perception to own body truth.
Perception consumes the expectation as a typed input; Embodiment remains the
canonical body-state owner. This must NOT become an excuse to blur
Perception and Embodiment into one WM. The expectation flows through the
existing Perception ↔ Embodiment transport contract, not through a merged
latent space.

**Architecture candidate.**

- Embodiment-native predictive seam: compact recurrent or SSM model
  (Mamba-style gated sequential) over proprio + issued action chunks + body
  pose → predicted self-motion delta, predicted occlusion mask, predicted
  camera-ego-motion. Order of 1–3M params.
- Small bridge module: projects Embodiment-native prediction into
  Perception's temporal grounding token space. Order of 200K–500K params.
  Layer classification: `WM-to-WM transport-shaped`.

**Typed surfaces.**

- `SelfMotionExpectation`: predicted camera/body displacement, predicted
  occlusion regions, predicted force/contact changes, confidence,
  embodiment_state_id, timestamp
- `SelfDisturbanceReceipt`: emitted by Perception after comparing
  expectation to observed reality — mismatch magnitude, attribution
  (self-caused vs external), temporal alignment quality

**Phase timing.** Preserve doctrine now. Implementation belongs in Phase 3
(Embodiment WM) or later, once Embodiment owns real body state and
Perception has wired temporal grounding.

**What not to let it become.** A bidirectional latent fusion layer between
Perception and Embodiment. A reason for Perception to internalize body
modeling. A bypass of the typed transport contract.

---

### B. Active Sensing

**Principle.** Biological sensory systems actively reposition sensors and
make exploratory contact to reduce uncertainty, rather than passively
waiting for better data.

**Robostack insert point.** Two locations:

1. Embodiment WM action proposal surface (near-term)
2. Economic WM value-of-information shaping (later)

"Acquire better evidence" should be a **legitimate typed action family**
alongside task-directed actions. Concrete instances:

- head/camera reposition under ambiguity (gaze direction to resolve
  occlusion)
- cautious exploratory contact to disambiguate object identity, weight, or
  compliance
- body repositioning to reduce workspace occlusion
- sensor-mode switching (depth vs RGB emphasis under lighting conditions)

This is NOT generic exploration. It is **bounded, receipt-emitting, and
economically shaped later**:

- each active-sensing action should emit an `ActiveSensingReceipt` with
  information gain estimate, cost vector, and outcome
- the Economic WM should later shape the value-of-information field that
  determines when active sensing is worth the time/energy/wear cost

**Architecture candidate.**

- Embodiment-local proposal head with info-gain / ambiguity-reduction
  auxiliary scoring: extends the existing Action Proposal Head (subsystem 5
  in `docs/actuation_embodiment_world_model.md`) with a parallel scoring
  branch that estimates expected uncertainty reduction per candidate action.
  Order of 500K–2M params additional. Layer classification: `WM-native` to
  Embodiment.
- Later: Economic WM value-of-information shaping field, consumed by the
  Embodiment active-sensing scorer as a downward-shaped budget/priority
  signal. Layer classification: `WM-to-WM transport-shaped`.

**Typed surfaces.**

- `ActiveSensingProposal`: action type (reposition/contact/mode-switch),
  expected information gain, cost vector, target uncertainty region,
  embodiment_state_id
- `ActiveSensingReceipt`: action taken, actual information gain, cost
  incurred, uncertainty before/after, perception state delta

**Phase timing.** Preserve doctrine now. Near-term implementation:
Embodiment-local scoring branch alongside Phase 3 action proposal work.
Economic value-of-information shaping: Phase 5+ only.

**What not to let it become.** A generic exploration bonus. An RL reward
hack. A reason for Embodiment to own perception uncertainty (Perception owns
uncertainty; Embodiment reads it and may propose actions to reduce it).

---

### C. Neuromodulation / Allostasis

**Principle.** In biology, neuromodulators broadcast slow, wide-area changes
in gain, plasticity, exploration/exploitation balance, vigilance, and
effective learning rate. They do not micromanage individual units. Allostasis
extends this: viability is maintained across a multidimensional
resource/constraint envelope, not a single setpoint. **Robostack import:**
only that *broadcast conditioning* pattern—no neurotransmitter model, no
named-brain-region roles.

**Robostack insert point.** Two locations:

1. Future Economic WM (intra-domain regime broadcast)
2. Later meta-regal superposition layer (inter-domain regime broadcast)

The operational pattern is **regime broadcast, not micromanaging control**.
The Economic WM should be able to alter operating mode across the stack:

- trust posture (conservative vs exploratory)
- compute posture (compute-rich vs compute-scarce behavior)
- exploration posture (information-seeking vs exploitation)
- training-open vs training-closed posture
- energy conservation mode
- degraded-mode / graceful-fallback posture

This should remain **low-bandwidth and typed**. It must not become a
sovereign god-object. The Economic WM doctrine
(`doctrine_economic_wm_future_architecture.md`) already defines regime
switching, `EconomicRegime`, and the multi-timescale fast/meso/slow
separation. The delta here is naming the **design stance** explicitly:

- regime broadcast is a **small typed signal** that conditions downstream
  WM behavior (broadcast conditioning, not per-actuator directives)
- it is NOT a high-bandwidth control channel
- it is NOT a replacement for local WM autonomy
- the meta-regal layer (see `doctrine_meta_regal_node_wm.md`) should later
  compose multiple regime broadcasts (economic, safety, plausibility,
  deployment-truth) without any one becoming dominant

**Architecture candidate.**

- Economic state estimator backbone: regime-conditioned switching
  state-space model (DS3M or RED-SDS, already confirmed in
  `doctrine_economic_wm_future_architecture.md`). That choice matches this
  pattern: regime identity should be a discrete latent that conditions
  continuous dynamics, not a continuous variable lost in the state vector.
  Layer classification: `WM-native` to Economic WM.
- Downward regime broadcast interface: typed `RegimeBroadcast` object
  consumed by lower WMs as parametric context (not as optimizable
  variables), consistent with the adiabatic separation rule already in
  the Economic WM doctrine.
- Later meta-regal composition: hypernetwork-conditioned shaping/allocation
  head that composes multiple domain-regime broadcasts. This is acceptable
  post-Phase 7, not before. Layer classification: `WM-to-WM
  transport-shaped`.

**Why this is different from a monolithic global policy.** A global policy
learns a single mapping from state to action. The neuromodulatory pattern
instead learns a **small set of regime-level signals** that condition many
local policies simultaneously. Each local policy retains its own autonomy
and optimization objective; the regime signal only shifts the operating
envelope. This preserves local WM sovereignty under global constraint —
evaluation rule #2.

**Typed surfaces.**

- `RegimeBroadcast`: regime_id, regime_class (from `EconomicRegime`),
  posture settings (trust, compute, exploration, training, energy,
  degraded-mode), confidence, persistence annotation, provenance
- `RegimeAcknowledgmentReceipt`: emitted by each downstream WM confirming
  receipt and local adaptation (or explicit non-compliance with reason)

**Phase timing.** Later. Full neuromodulatory/allostatic economic shaping
belongs after the Economic WM backbone (switching SSM estimator + dynamics
model) is structurally real. Until then, regime signals should remain typed
non-neural state.

**What not to let it become.** A sovereign god-object. A high-bandwidth
control channel that micromanages lower WMs. A replacement for local WM
optimization objectives. A mechanism for economics to silently redefine
physical, safety, or deployment truth.

---

### D. Plasticity Gating / Consolidation

**Principle.** Biological learning is not continuous or uniform. The nervous
system selectively consolidates experiences based on novelty, reward
prediction error, emotional salience, and sleep/wake state. Not every
experience becomes a long-term memory; not every synapse is plastic at all
times.

**Robostack insert point.** Cross-cutting across:

- replay selection
- training eligibility
- promotion evidence evaluation
- work-order creation
- self-improvement surfaces

The repo already wants artifact completeness, receipt quality, and trace
completeness to become preconditions for training (see the Complete
Subsystem Rule and Mechanics-First WM Readiness Rule in
`multi_wm_architecture_plan.md`). The bio-inspired delta: formalize this as
**selective consolidation / gated plasticity** with a trained eligibility
scorer.

Design principles:

- not every trace becomes training signal
- not every subsystem learns online
- eligibility should be a typed multi-class decision, not a generic scalar
  "quality score"
- the eligibility classes should include: `train-ready`, `consolidate-later`,
  `archive-only`, `discard`, `needs-annotation`, `needs-calibration`,
  `blocked-on-provider`

**Architecture candidate.**

- Receipt-set scorer / trace-completeness scorer: set-attention or compact
  transformer over episode receipts and manifest metadata. Takes as input
  the full receipt family (perception, embodiment, sim, economic) from an
  episode and outputs typed eligibility classes. Order of 2–5M params.
  Layer classification: `WM-native` to whichever WM owns the training
  pipeline (likely Economic WM or a cross-cutting training governance
  surface).
- Training: supervised on realized training improvement conditioned on
  episode characteristics. Later: indirect economic yield shaping.

**Typed surfaces.**

- `TrainingEligibilityAssessment`: episode_id, eligibility_class (enum),
  receipt completeness score, trace quality dimensions (temporal coverage,
  annotation density, calibration freshness, provider truth posture),
  recommended training targets, blocking reasons
- `ConsolidationReceipt`: emitted after training run confirming what was
  trained, what was deferred, and why

**Phase timing.** Preserve doctrine now. Near-term implementation: typed
eligibility schema and heuristic scorer alongside replay/training pipeline
work. Neural scorer: Phase 5+ (Economic WM), once receipt streams from
lower WMs are structurally real.

**What not to let it become.** A generic scalar quality filter. A mechanism
that prevents low-quality data from ever being examined (some low-quality
data reveals important failure modes). A reason to delay training until
"everything is perfect."

---

### E. Motor Synergies + Richer Interoception

**Principle.** Biological motor control operates through structured
synergies — coordinated patterns of multi-joint activation — rather than
independent control of each degree of freedom. Separately, organisms
maintain rich interoceptive state (temperature, fatigue, energy reserves,
pain, proprioceptive confidence) that shapes motor behavior before conscious
planning intervenes.

**Robostack insert point.** Embodiment / Actuation WM.

The existing Embodiment WM doc (`docs/actuation_embodiment_world_model.md`)
already specifies joint state, contact state, tool state, safety envelope,
and action proposal architecture. The delta:

**Motor synergies.** Upper layers (task policy π_H, Economic WM) should not
micromanage every actuator DOF. The Embodiment WM should own structured
posture/contact/skill manifolds:

- skill/synergy codebook: a discrete or continuous codebook of coordinated
  multi-joint patterns (reach, grasp, insert, stabilize, reposition) that
  the action proposal head selects from
- structured chunk proposer: action chunks that encode synergy-level
  commands, not individual joint targets
- graph-conditioned skill selection: kinematic-tree-aware selection of which
  synergy applies given current body topology and contact state

**Richer interoception.** Body schema should include not just geometry and
joint state but also:

- thermal state per subsystem / joint group
- battery reserve and depletion rate
- actuator wear estimation
- communication latency to companion compute
- compute placement and availability
- controllability confidence per joint group
- recent fault / near-miss history

The Embodiment WM's Capability / Embodiment State Surface (subsystem 1) and
Compute/Battery/Thermal Forecaster (already in
`neuralization_bridge_doctrine.md`) partially cover this. The delta is
making interoceptive state **explicit and typed** as a first-class encoder
input, not just a side telemetry stream.

**Architecture candidate.**

- Synergy codebook: VQ-VAE-style discrete codebook over demonstrated
  multi-joint coordination patterns, or continuous manifold learned from
  demonstration retargeting traces. Order of 1–5M params. Layer
  classification: `WM-native` to Embodiment.
- Graph-conditioned skill selector: GNN over kinematic tree + contact
  topology → synergy selection. Can share structure with the existing
  whole-body state encoder (GNN, already specified in the bridge doctrine).
  Minimal additional params.
- Interoceptive encoder: explicit typed encoder over thermal/battery/
  wear/latency/compute/fault state vectors → interoceptive embedding
  consumed by action proposal head and feasibility checker. Order of
  500K–2M params. Layer classification: `WM-native` to Embodiment.

**Typed surfaces.**

- `SynergyCodebookEntry`: synergy_id, joint_group_mask, activation_pattern,
  contact_precondition, energy_cost_estimate, typical_duration
- `InteroceptiveState`: thermal_vector, battery_state, wear_estimates,
  latency_map, compute_availability, controllability_confidence,
  fault_history_summary

**Phase timing.** Later. Full synergy codebook and richer interoceptive
encoder belong after Embodiment WM owns real body state and has wired the
action proposal pipeline (Phase 3+). The typed interoceptive schema can
be preserved now as doctrine.

**What not to let it become.** A replacement for the Embodiment WM ontology.
A reason for the action proposal head to bypass typed body state and read
raw sensor telemetry. A mechanism for upper layers to micromanage joint
torques through synergy-level commands (synergies are Embodiment-local; the
task policy selects goals, not synergies).

---

### F. Immune-Style Anomaly Governance

**Principle.** The immune system uses distributed, local, heterogeneous
detectors with escalation, tolerance, and memory — not one central
classifier. Anomalies are detected locally, escalated through typed
channels, and governed by adaptive tolerance thresholds.

**Robostack insert point.** Two levels:

1. Local governance nodes first (per-domain anomaly detection)
2. Later meta-regal composition (immune-system-like multi-detector
   composition)

The meta-regal doctrine (`doctrine_meta_regal_node_wm.md`) already
specifies domain-governance nodes: economic, anti-reward-hacking,
plausibility, deployment-truth, safety, data-value, coordination. The delta
here is the explicit design stance: **per-domain suspicion receipts are
better than one safety god-model**.

Good fit for:

- anti-reward-hacking: per-domain reward-channel consistency detectors
- deployment truth: per-provider honesty scorers
- data quality: per-source quality / conflict detectors
- safety: per-subsystem safety-margin anomaly detectors
- provider honesty: cross-provider disagreement scorers (partially covered
  by the existing `GroundingCalibrationReceipt` pattern)

**Architecture candidate.**

- Local sequential anomaly models: small sequential models (LSTM, SSM, or
  energy-based scorers) per governance domain, operating over that domain's
  receipt stream. Order of 200K–1M params each. Layer classification:
  `WM-native` to whatever WM owns the domain (e.g., anti-reward-hacking
  anomaly detector is WM-native to the governance surface that owns reward
  integrity).
- Receipt critics: per-domain receipt critics that score whether a receipt
  stream is consistent with expected operating behavior. Lightweight
  attention or energy-scoring architecture.
- Meta-regal composition: later, the meta-regal-node WM composes local
  anomaly signals using the regime-sensitive composition logic already
  specified in `doctrine_meta_regal_node_wm.md`. This is NOT a new
  architecture — it is a concrete instantiation of the governance-node
  neuralization (Stage B) already documented.

**Typed surfaces.**

- `AnomalySuspicionReceipt`: domain, anomaly type, severity, evidence
  summary, suggested escalation, tolerance context, timestamp
- `GovernanceEscalationEvent`: source domain, escalation level,
  triggering anomaly receipts, recommended action class

**Phase timing.** Later. Local governance nodes belong in the governance
node neuralization stage (Stage B in `doctrine_meta_regal_node_wm.md`).
Meta-regal immune-style composition belongs in Stage C. Near-term: preserve
the design stance that per-domain detectors are better than one classifier.

**What not to let it become.** One vague "governance model" that collapses
domain-specific anomaly detection into a single latent. A reason to delay
domain-specific governance nodes in favor of a monolithic safety layer. A
system that generates so many suspicion receipts that the escalation channel
becomes noise.

**Local calibrated-head note.** HALO-like bounded geometric confidence can be
a useful shaping rule for per-domain anomaly heads or receipt critics that
must say "insufficient evidence" rather than force a positive anomaly label.
Use this only as a local head-level abstention discipline; do not let
classifier geometry become the ontology of governance.

## Cross-Cutting Neural Architecture Table

This table uses the layer classification taxonomy from
`neuralization_bridge_doctrine.md`.

| Bio Principle | Likely WM / Subsystem Owner | Layer Classification | Candidate Architecture Family | Likely Timescale | Primary Typed Outputs |
|---|---|---|---|---|---|
| Efference copy | Embodiment WM (predictor); Perception WM (consumer) | Predictor: `WM-native`; Bridge: `WM-to-WM transport-shaped` | Compact recurrent / SSM (Mamba-style) + small projection bridge | Mid-loop (control rate, 10–50 Hz) | `SelfMotionExpectation`, `SelfDisturbanceReceipt` |
| Active sensing | Embodiment WM (proposer); Economic WM (value-of-info shaper) | Proposer: `WM-native`; Value shaping: `WM-to-WM transport-shaped` | Info-gain scoring branch on action proposal head | Mid-loop (skill rate, 1–10 Hz) | `ActiveSensingProposal`, `ActiveSensingReceipt` |
| Neuromodulation / allostasis | Economic WM (intra-domain); Meta-regal WM (inter-domain) | Regime estimator: `WM-native`; Regime broadcast: downward governance transport | Switching SSM (DS3M / RED-SDS); later hypernetwork-conditioned composition | Slow (governance rate, ~0.1–1 Hz) | `RegimeBroadcast`, `RegimeAcknowledgmentReceipt` |
| Plasticity gating | Economic WM or cross-cutting training governance | `WM-native` to training pipeline owner | Set-attention / compact transformer over receipt sets | Episodic / batch | `TrainingEligibilityAssessment`, `ConsolidationReceipt` |
| Motor synergies + interoception | Embodiment WM | `WM-native` | VQ-VAE codebook; GNN skill selector; typed interoceptive encoder | Fast–mid (motor to skill rate) | `SynergyCodebookEntry`, `InteroceptiveState` |
| Immune-style anomaly | Per-domain governance nodes; later meta-regal composition | `WM-native` to each governance domain | Sequential anomaly models / energy scorers / receipt critics | Varies by domain (fast safety → slow deployment-truth) | `AnomalySuspicionReceipt`, `GovernanceEscalationEvent` |

## Sequencing: Now vs Later

### Preserve as doctrine now

These are near-term doctrine worth preserving because they constrain current
design decisions without requiring implementation:

- **Efference copy doctrine**: Ensures the Perception ↔ Embodiment
  transport contract reserves a slot for self-motion expectation. Prevents
  Perception from accidentally internalizing body modeling. Constrains
  Phase 3 Embodiment WM design.
- **Active sensing doctrine**: Ensures the Embodiment WM action proposal
  surface includes uncertainty-reduction as a typed action family. Prevents
  active sensing from being bolted on later as an RL exploration hack.
  Constrains Phase 3 action proposal head design.
- **Plasticity gating doctrine**: Ensures the replay/training pipeline
  designs for typed eligibility classes rather than generic quality scores.
  Constrains replay selection and work-order creation across all phases.

### Defer implementation until lower-WM ownership is structurally real

These require lower-WM maturity before implementation makes sense:

- **Full neuromodulatory / allostatic economic shaping**: Requires the
  Economic WM backbone (switching SSM estimator + dynamics model) to be
  structurally real. Regime broadcast without a real regime estimator is
  just a config flag.
- **Richer synergy codebook + interoceptive stacks**: Requires Embodiment
  WM to own real body state with demonstration traces and action proposal
  pipeline. Synergy learning without real multi-joint data is premature.

### Post-September-2026 training / calibration era

These clearly belong after the structural plumbing is laid and training
has begun:

- **Meta-regal immune-style anomaly composition**: Requires governance
  nodes to be neuralized (Stage B) and emitting typed receipts. Composing
  anomaly signals requires anomaly signals to exist first.
- **Hypernetwork-conditioned regime composition**: Requires meta-regal-node
  WM infrastructure (Phase 7+).
- **Full economic value-of-information shaping for active sensing**:
  Requires Economic WM to be consuming real lower-WM receipts and emitting
  allocation envelopes.

## Related Doctrine

- WM topology, sequencing, phase exit rules: `multi_wm_architecture_plan.md`
- Layer classification, bridge topology, reward placement: `neuralization_bridge_doctrine.md`
- Embodiment WM subsystems and typed interfaces: `docs/actuation_embodiment_world_model.md`
- Economic WM ontology, regime switching, multi-timescale design: `doctrine_economic_wm_future_architecture.md`
- Meta-regal governance pluralism and staged neuralization: `doctrine_meta_regal_node_wm.md`
- PINN posture for Economic WM submodules: `doctrine_economic_wm_future_architecture.md` § Constraint-Informed Submodules / PINN Posture
