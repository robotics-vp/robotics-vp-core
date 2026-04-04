# Doctrine: Future Economic WM Architecture

## Status

Future-architecture doctrine. The Economic WM described here has not been
built. This document defines its target shape so that current lower-WM work
preserves the right upward receipt and downward shaping contracts.

## What the future Economic WM is

The future Economic WM is:

> **the canonical world model of productive flow, constraint, dissipation, and
> allocative opportunity across the robotics stack.**

It models the system as a nonequilibrium constrained productive organism
spanning task throughput, energy, onboard compute, sim budget, replay/data
inventory, embodiment wear, operator scarcity, queue pressure, uncertainty,
and coordination burden.

It is **not**:

- a scalar reward head
- a dashboard
- a PnL tracker
- a thin weight-picker over objectives
- a learned oracle that silently decides value
- a mother-latent that erases typed contracts

Scalar reward should be a downstream compilation artifact, not the native
ontology of the Economic WM.

## Sovereignty clarification

The Economic WM is a **first-class allocative world model**, but it is **not
the sovereign governor of the stack**. The stack's telos is not "optimize
economics." The telos is governed robot control under multiple
non-collapsible realities: physical plausibility, safety, anti-reward-hacking,
deployment truth, embodiment limits, coordination integrity — and only one of
those is economic allocation.

The Economic WM participates as one major contributor within a higher-order
superpositioning meta-regal-node WM (see
`docs/economic_world_model/doctrine_meta_regal_node_wm.md`).

Its role is:

- **first-class contributor**: resource allocation, opportunity cost,
  compute/energy/time/wear tradeoffs, value of information, data valuation,
  task prioritization
- **not sole governor**: the meta-regal-node WM can override or constrain
  economic recommendations when safety, plausibility, reward integrity, or
  deployment truth require it

This means the Economic WM provides a major slice of the meta-governance
surface, but it lives inside a broader superposed governance field. If the
Economic WM becomes too central, the stack risks translating everything into
economic language and treating physical/safety/deployment reality as
constraints subordinate to an economic worldview — which is dangerous in a
control stack.

## Intra-domain vs inter-domain Pareto

Two fundamentally different Pareto problems exist in the stack:

1. **Intra-domain** (inside the Economic WM): throughput vs energy vs wear vs
   compute vs error vs exploration vs data yield. Multi-objective optimization
   within a single evaluative domain.
2. **Inter-domain** (inside the meta-regal-node WM): economics vs
   anti-reward-hacking vs plausibility vs deployment truth vs safety vs
   coordination integrity. More fundamental, because it governs whether the
   intra-domain optimization can even be trusted in a given regime.

The Economic WM should focus on being the strongest possible intra-domain
Pareto allocator. The inter-domain composition is the meta-regal-node WM's
job.

## Core design claim

Before RL abstractions, reward-shaping equations, or Pareto-signal-outputting
meta-nodes, the future Economic WM should be defined as a **state estimator +
dynamics model + allocator + governance layer**.

The sequencing should be:

1. define economic ontology and state
2. define upward typed receipts from lower WMs
3. define regimes / macro order parameters / bottlenecks
4. define downward allocative and governance fields
5. define how local Pareto policy slices are exposed downward
6. only then compile those fields into local RL shaping interfaces

This sequencing matters because otherwise the first reward decomposition or
hyperparameterization becomes the de facto worldview of the entire Economic WM.

## Economic WM ontology

### Resource reservoirs

- battery / energy budget
- onboard inference capacity / availability
- thermal headroom
- actuation wear budget
- human supervision / intervention budget
- training budget
- sim / fidelity budget
- replay / datapack inventory
- wall-clock / latency budget

### Flow fields

- task throughput flow
- useful-data generation flow
- policy-improvement flow
- failure propagation flow
- queue accumulation / dissipation
- energy burn flow
- compute burn flow
- coordination traffic flow

### Dissipation and friction

- latency
- uncertainty
- embodiment mismatch
- sim-to-real mismatch
- semantic disagreement
- coordination overhead
- exploration waste
- thermal stress
- actuator degradation pressure

### Regimes

- energy-scarce
- compute-scarce
- uncertainty-heavy
- coordination-heavy
- deployment-sensitive
- exploration-heavy
- operator-scarce
- queue-congested

Regime-switching and switching state-space models treat the system as a
mixture of latent operating modes with different local dynamics, rather than
as one stationary process. Deep regime-switching models and explicit-duration
switching systems let regime persistence and transitions be learned.

### Reserved type names

- `EconomicState`
- `EconomicReceipt`
- `EconomicRegime`
- `ResourceReservoir`
- `FlowField`
- `DissipationField`
- `BottleneckMap`
- `OpportunityField`
- `AllocationEnvelope`
- `ShapingField`
- `EconomicTransition`
- `EconomicCounterfactual`
- `EconomicInvariant`
- `SlowManifoldProjection` — compiled slow state from the estimator;
  prevents local receipt noise from reparameterizing macro economic state
- `ShadowPriceField` — per-resource marginal prices (Lagrangian
  multipliers) updated on meso timescale
- `ParetoFrontierSlice` — distributional frontier segment with tail-risk
  metadata from the allocator
- `PersistenceAnnotation` — hysteresis / hold-steady metadata on
  governance transport

`EconTensor` can remain as a portable compiled representation but should not
be treated as the full ontology.

## Multi-timescale / near-adiabatic structure

The future Economic WM must be explicitly multi-timescale. This is not a
metaphor — adiabatic elimination (slow/fast variable separation via
quasi-steady-state manifold reduction) provides a rigorous design pattern for
the internal decomposition boundaries.

**Important constraint:** these are **design patterns for decomposition and
control**, not a replacement ontology. Terms like "pressure," "flow," and
"fluid" must not become vague substitutes for typed state, typed receipts,
and typed transport. The adiabatic pattern gives us a multi-timescale
control/estimation structure; it does not redefine the Economic WM as a
fluid system.

### Fast variables

- local routing
- queue dispatch
- sim-branch selection
- local shaping coefficients
- immediate uncertainty penalties
- short-horizon task priority

### Meso variables

- task-family allocation
- episode / shift budget routing
- which datapacks or slices to train on
- embodiment assignment
- exploration quotas
- shadow prices / Lagrangian multipliers for resource constraints

### Slow / near-adiabatic variables

- objective-contract structure
- topology-level allocation priors
- global constraint manifolds
- promotion thresholds for governance nodes
- trusted transport semantics
- deployment-trust / safety invariants
- regime identity and duration

### Adiabatic separation design rules

The following rules are concrete, not metaphorical. They govern the data
flow between the four internal parts of the Economic WM:

1. **No fast→slow feedback without explicit gating.** Fast receipt noise must
   not reparameterize slow manifold state. The `SlowManifoldProjection`
   interface enforces this: fast receipts are aggregated and projected before
   affecting the estimator's slow state.
2. **Slow state conditions fast dynamics.** The dynamics model takes slow
   manifold state (regime, constraint manifold, macro-pressure vector) as
   parametric context, not as optimizable variables.
3. **Meso variables are the update channel for shadow prices.** Lagrangian
   multipliers / shadow prices for resource budgets update on a meso
   timescale — slower than per-step allocation but faster than regime
   transitions.
4. **Persistence / hysteresis prevents thrashing.** Governance transport
   downward carries persistence annotations. Lower WMs should track
   governance fields without overreacting to marginal changes.

**Design rule:** slow variables should not swing violently in response to
local noise. Superstatistics formalizes a similar idea by treating fast local
fluctuations as occurring under slowly varying macro-parameters.

## Bidirectional but asymmetric transport

### Upward transport: lower WMs → Economic WM

Purpose: receipt transport, abstraction, bottleneck aggregation, macro-state
estimation.

Carries:

- capability receipts
- uncertainty receipts
- local feasibility bounds
- local opportunity estimates
- local bottleneck statistics
- temporal summaries
- provenance / confidence

### Downward transport: Economic WM → lower WMs

Purpose: allocative transport, shaping transport, admissible-region transport,
governance transport.

Carries:

- local budget envelopes
- local priority gradients
- local penalties / incentives
- target operating regime
- trust weights
- persistence / hysteresis annotations
- timescale annotations
- local admissible Pareto slices

The downward object is **asymmetric in abstraction and time** relative to the
upward one. It is not the same tensor reversed. Upward carries raw receipts
and local statistics; downward carries compiled allocative fields with regime
context and governance provenance.

## Internal decomposition

### 1. Economic State Estimator

Consumes typed receipts from lower WMs and estimates a legible
`EconomicState`.

Inputs:

- perception uncertainty and scene stability receipts
- embodiment capability / wear / thermal / latency / energy receipts
- sim/synth branch value / realism / failure / coverage receipts
- data valuation receipts
- training loop receipts
- queue/backlog/operator/deployment receipts

Outputs:

- `EconomicState`
- `EconomicRegime`
- `BottleneckMap`
- confidence / uncertainty
- marginal tension / shadow-price-like signals

Architecture family: switching state-space models and regime-aware sequence
models rather than plain transformers first. Confirmed candidates:

- **DS3M** (Deep Switching State Space Model) — RNN + nonlinear SSSM with
  discrete regime latents + continuous stochastic latents. Good for
  long-range dependencies and abrupt regime changes. PyTorch reference:
  `Sherry-Xu/Deep-Switching-State-Space-Model`.
- **RED-SDS** (Recurrent Explicit Duration Switching Dynamical Systems) —
  explicitly models regime duration, not only regime identity. Critical for
  the Economic WM because regime persistence is a first-class concern
  (energy-scarce regimes last shifts, not steps). PyTorch reference:
  `abdulfatir/REDSDS`.

The estimator should emit a `SlowManifoldProjection` that downstream layers
consume, enforcing the adiabatic separation: fast receipts are projected
onto the slow manifold via a typed aggregation layer.

### 2. Economic Dynamics Model

Forecasts how economic state evolves under candidate allocations and policy
choices. Dynamics are **conditioned on slow manifold state** (regime,
constraint manifold, macro-pressure vector) from the estimator.

Questions:

- what happens if sim budget is spent here?
- what happens if onboard inference is reallocated?
- what future fragility is induced by over-exploitation?
- what queue / wear / thermal / latency costs accumulate?

Model families:

- regime-switching state-space models
- sequence models over typed receipts
- counterfactual rollouts under learned regime transitions
- differentiable-physics-coupled forecasting where appropriate

### 3. Economic Allocator / Compiler

Converts estimated state + forecasts into structured allocative fields.

Outputs should not be scalar rewards first. They should be:

- task priority fields
- sim budget recommendations
- training slice priorities
- exploration quotas
- compute allocation envelopes
- energy-aware routing preferences
- data retention / pruning guidance
- later, coordination suggestions

The Pareto allocator should be **distributional, regime-aware, and
execution-aware**, preserving uncertainty over multivariate returns rather
than only their expectations.

Confirmed architecture patterns:

- **DPMORL** (Distributional Pareto Multi-Objective RL, NeurIPS 2023) —
  return-distribution-aware utility functions for Pareto policy training.
  Emits distributional frontier slices, not point estimates.
- **PGMORL** (Prediction-Guided MORL, ICML 2020) — evolutionary, dense
  Pareto front generation for continuous control. Good benchmark pattern
  for frontier generation.
- **Risk budgeting via augmented Lagrangian** — shadow prices (Lagrange
  multipliers) as dynamic resource prices, updated on meso timescale. Each
  resource constraint has an associated `ShadowPriceField` that shapes
  downstream allocation.

The allocator should emit `ParetoFrontierSlice` objects with tail-risk
metadata, not raw scalar weights.

### 4. Economic Governance / Reciprocity Layer

Makes the Economic WM reciprocally coupled to lower WMs.

Bottom-up: lower WMs send receipts, disagreement, feasibility, uncertainty,
counterfactuals.

Top-down: Economic WM sends shaping fields, budget envelopes, admissible
operating regions, persistence instructions, and hold-steady guidance.

Governance stays typed and auditable instead of collapsing into an opaque
latent controller.

## Quant-inspired algorithmic imports

These are **algorithmic pattern imports**, not worldview imports.

### Coherent risk measures

Borrow CVaR-style tail sensitivity, coherent risk measures, and dynamic
time-consistent risk handling. Lower WMs should not just maximize expected
improvement; they should avoid rare but catastrophic deployment, wear, or
safety outcomes.

Roadmap implication: add a future `RiskField` / `TailRiskSlice` concept to
allocator outputs. Let admissible Pareto slices be filtered by coherent-risk
constraints before local compilation.

Repos to inspect: `acoache/RL-DynamicRobustRisk`,
`Silvicek/cvar-algorithms`, `rllab-snu/Trust-Region-CVaR`.

### Distributional Pareto policies

Learn the distribution over multivariate outcomes. Preserve Pareto structure
under uncertainty. A single mean-optimal Pareto point is too brittle for
deployment-sensitive robotics.

Roadmap implication: the allocator should become a distributional,
regime-aware, execution-aware Pareto policy field. Local shaping compilers
should consume frontier slices plus confidence / risk metadata.

Repos to inspect: DPMORL, `mit-gfx/PGMORL`.

### Risk budgeting / shadow prices

Implicit marginal prices for scarce resources. Dynamic budget routing rather
than only hard caps. Energy, onboard inference, operator attention, sim
budget, and actuator wear should all behave like scarce factors with local
marginal costs.

Roadmap implication: emit `AllocationEnvelope`s plus marginal-tension fields
instead of only reward multipliers.

### Regime switching

Latent operating-mode inference, persistent regimes with explicit durations,
state-dependent switching. The stack will not operate in one stationary mode.

Roadmap implication: treat `EconomicRegime` as a first-class typed object.
Train regime detectors before end-to-end neural allocation.

Repos to inspect: DS3M, RED-SDS, regime-switching RNN patterns.

### Stress testing and scenario analysis

Shock testing, scenario-conditioned allocation, robustification against
misspecification. A good policy under nominal conditions may fail under queue
spikes, degraded sensing, reduced oversight, or thermal stress.

Roadmap implication: the allocator should maintain frontier slices under
stress scenarios, not only base-case forecasts.

### Execution-cost awareness

No allocation is good if execution costs dominate theoretical edge. In
robotics the analogs are latency, heat, battery, wear, coordination traffic,
and operator dependence.

Roadmap implication: execution friction should appear natively in
`DissipationField`, not as a later penalty hack.

## Superstatistics: what to borrow

### Keep

- local near-stationarity under slowly varying macro-parameters
- heavy tails / anomalous behavior arising from mixtures of local regimes
- macro intensive variables drifting more slowly than micro state fluctuations

### Do not keep

- vague temperature metaphors with no operational meaning
- ungrounded claims that every heavy tail is thermodynamic

### Robotics translation

Potential slow macro-parameters: uncertainty pressure, coordination burden,
energy stress, queue pressure, oversight scarcity, deployment sensitivity.

Potential fast local variables: action routing, branch simulation choices,
local shaping coefficients, micro queue dispatch.

Roadmap implication: represent macro regime variables separately from fast
local control/state. Allow local policies to condition on slowly varying
`EconomicRegime` or `MacroPressureVector`.

## Nonequilibrium / adiabatic control: what is useful

This area is more useful as a control/abstraction source than as a copy-paste
code source right now. The transfer value is conceptual and architectural.

### Reduced dynamics on structured manifolds

Complex dynamics can be projected to reduced coordinates that preserve
problem geometry.

Roadmap implication: introduce a future reduced `EconomicState` manifold
rather than letting raw receipts explode in dimension.

### Cycle thinking

Expansion, relabeling, synthesis, contraction to stable policy, then renewed
expansion. Represent training / deployment / recovery / resynchronization as
explicit modes rather than one continuous training blur.

### Stochastic hardware

Thermodynamic-computing material is long-range infrastructure inspiration,
not near-term Economic-WM implementation guidance.

## Staged neuralization

### Stage A: typed non-neural scaffolding

- define receipts, regimes, bottleneck maps
- define upward/downward surfaces
- define stress-test scenarios
- define allocation envelope schemas

### Stage B: neural state estimation

Train regime/state estimators over typed receipt sequences.

Candidates: switching state-space models, explicit-duration regime models,
typed sequence encoders over receipt streams.

### Stage C: neural dynamics / counterfactual forecasting

Train counterfactual rollout modules: state transition forecasting,
bottleneck evolution, queue / energy / wear / uncertainty propagation.

### Stage D: neural Pareto allocator

Train an allocator that emits frontier slices, risk-conditioned envelopes,
local shaping fields, confidence / uncertainty metadata. Distributional
Pareto MORL rather than classical single-policy MORL.

### Stage E: local compilers into subsystem shaping

Only after A-D: compile allocator outputs into local reward-shaping /
governance knobs for lower WMs. Preserve explicit provenance of which shaping
came from which macro signal.

**Rule:** neuralization follows typed ontology and transport design, not
precedes it.

**Auxiliary compression note:** Stages B–C may use **bounded**
autoencoder/codebook modules on high-rate receipt streams **before** slow
aggregation (see § Autoencoder / Manifold-Compression Posture). Those modules
are scaffolding for compression and motif extraction; they do **not** replace
switching-SSM / regime-aware sequence estimators or the dynamics/allocator
backbones.

## Scalability requirements

### Sparse / typed factorization

Do not make one giant latent. Factor by resource reservoirs, flow fields,
dissipation terms, regimes, frontier slices, local transport surfaces.

### Modular sequence ingestion

Each subsystem should emit typed receipts that can be ingested independently,
then fused. Regime-aware sequence models are a better near-term fit than
unrestricted end-to-end transformer soup.

### Counterfactual sample efficiency

Use structured models where possible: differentiable physics for embodied
consequences, typed receipt forecasts for economics/governance, stress
testing rather than pure brute-force exploration.

### Distributional outputs

Distributional frontier slices should scale better to heterogeneous
deployments than one brittle expected-value policy.

## Research buckets

### Bucket A — regime-aware neural state estimation

Focus: DS3M / switching SSMs, RED-SDS / explicit duration switching,
regime-switching RNN patterns.

Deliverable: proposed `EconomicStateEstimator` candidates and typed I/O.

### Bucket B — risk-aware / distributional Pareto allocation

Focus: DPMORL, coherent-risk RL, CVaR / dynamic robust risk repos,
PGMORL-style Pareto set generation.

Deliverable: proposed `EconomicAllocator` that emits frontier slices,
tail-risk metadata, and allocation envelopes.

### Bucket C — superstatistical / multiscale abstractions

Focus: foundational superstatistics, dynamic validity conditions, translating
fluctuating intensive variables into robotics macro-pressures.

Deliverable: proposed `EconomicRegime` / `MacroPressureVector` abstraction.

### Bucket D — nonequilibrium / adiabatic control abstractions

Focus: reduced thermodynamic dynamics, control on structured manifolds,
cyclic training/deployment/recovery modes.

Deliverable: proposed slow/meso/fast manifold split and where invariants live.

### Bucket E — differentiable/scalable simulation coupling

Focus: differentiable physics precedents (JaxSim etc.), where structured
simulators can support economic counterfactuals without collapsing the WM
abstraction.

Deliverable: proposal for which consequences should be estimated through
learned models versus simulator-backed modules.

**Auxiliary to A/B (not a substitute):** research into bounded receipt
compressors, VQ/codebook motifs, or denoising summaries on typed streams is
valid **supporting** work for ingestion and visualization of evidence—it does
**not** replace Bucket A (DS3M / RED-SDS / regime-aware estimation) or Bucket B
(distributional Pareto allocation).

## Autoencoder / Manifold-Compression Posture

### Core answer

Autoencoder-family models (VAE, VQ-VAE, contractive/denoising bottlenecks,
codebook learners) **can** sit **inside** the future Economic WM as **bounded
auxiliaries**. The Economic WM must **not** be reframed as an
“autoencoder-native” stack.

The backbone remains:

- typed economic ontology and receipts
- **regime-aware state estimation** and dynamics / forecasting
- allocator / compiler and typed governance transport

**Primary estimator posture (unchanged):** switching state-space and
regime-aware sequence models first—**DS3M**, **RED-SDS**, and related
candidates remain the **primary** families for `EconomicState` /
`EconomicRegime` inference over typed receipt sequences. Autoencoders do not
supplant that choice.

### Good candidate placements

Where bounded AE/codebook modules help **without** becoming the backbone:

- **Receipt-path compression** before `SlowManifoldProjection`: high-rate
  typed receipts aggregated into a **compact** sequence representation that
  still feeds the switching-SSM (the SSM reads compressed tokens, not raw
  firehose dimensionality).
- **Manifold learning around slow state**: auxiliary bottlenecks that
  stabilize recurring macro patterns while slow variables remain **typed** and
  projection-gated (adiabatic rules still apply).
- **Regime-archetype / motif codebooks**: discrete codes for recurring
  economic episode types, queue motifs, or shift patterns—**inputs or side
  channels** to the regime estimator, not a replacement for explicit
  `EconomicRegime` inference.
- **Invariant / residual compression**: bottlenecks paired with PINN-style
  or balance constraints (see § Constraint-Informed Submodules / PINN Posture)
  where the AE learns what is **left over** after known structure.
- **Meso/slow summarization**: compact summaries of heterogeneous WM receipt
  tokens **before** meso shadow-price updates or slow governance fields—always
  subordinate to typed envelopes and provenance.

### What autoencoders must not replace

- **Economic State Estimator backbone** (switching SSM / explicit-duration
  regime models—not “one latent is the economy”).
- **Economic Dynamics Model backbone** (counterfactual rollouts, reservoir
  evolution—AE is not the dynamics).
- **Economic Allocator / Compiler** (distributional Pareto, risk-aware
  compilation—not reconstruction loss).
- **Downward governance shaping** (budget envelopes, persistence,
  admissible slices—not latent decoding).
- **Upward/downward asymmetric transport** (typed asymmetry stays primary).
- **Meta-regal composition** (inter-domain adjudication is not AE training).

### Design stance

**Bounded auxiliary / manifold-compression role: yes.**  
**Economic-WM backbone: no.**

### Relationship to other model families

This section **preserves** the existing doctrine: sparse typed factorization,
staged neuralization A→E, DS3M/RED-SDS-first estimation, DPMORL/PGMORL-style
distributional allocator patterns, and PINN-like **constraint-informed**
submodules. Autoencoder/codebook work is **orthogonal compression and motif
machinery**—useful where dimensionality or recurring structure is expensive,
never as the defining architecture of the Economic WM.

## Near-term roadmap insertions

### Phase 0 — ontology / doc hardening (current)

- define future Economic WM as canonical model of productive flow /
  dissipation / allocation
- add multi-timescale design language
- split upward vs downward transport docs
- reserve type names and interfaces

### Phase 1 — typed receipts and macro-state surfaces

- create receipt schemas from lower WMs
- define `EconomicReceipt`, `EconomicState`, `EconomicRegime`,
  `AllocationEnvelope`, `ShapingField`
- add scenario/stress-test interfaces

### Phase 2 — regime-aware state estimation

- prototype noninvasive `EconomicStateEstimator`
- start with typed sequence models / switching SSMs
- no direct shaping authority yet

### Phase 3 — counterfactual forecasting

- predict queue / energy / wear / uncertainty evolution under candidate choices
- support stress scenarios and confidence estimates

### Phase 4 — allocator / frontier engine

- emit local frontier slices and allocation envelopes
- preserve vector-valued and distributional outputs
- add coherent-risk filters

### Phase 5 — local shaping compilers

- compile frontier slices into subsystem-local shaping knobs
- keep provenance and governance receipts explicit

### Phase 6 — later neural meta-node integration

- only after the above are stable, connect into cross-WM meta-node
  neuralization

## Constraint-Informed Submodules / PINN Posture

Here *PINN-like* means **equation- or identity-constrained residuals**
(conservation, balance, monotonicity) on learned predictors—not branding,
not a requirement to pose economics as a PDE.

### Core answer

A PINN-like (physics-informed / equation-informed / constraint-informed)
component **can** have a place inside the Economic WM. But the Economic WM
should **not** be framed as a PINN-native monolith.

The backbone should remain regime-aware state estimation / dynamics /
allocation — switching-SSM-first (DS3M, RED-SDS) as already specified in
the Internal Decomposition section above. PINNs are useful as **constraint-
informed residual submodules** within that backbone, not as the backbone
itself.

The distinction: a PINN enforces known equations as structural constraints
on a learned model. That is powerful when the Economic WM has conserved or
monotone quantities that should be respected structurally rather than
learned from scratch. But most of the Economic WM's dynamics are hybrid,
regime-switching, and partially observed — not a clean closed PDE system.

### Good candidate placements

Where PINN-like constraint-informed modules fit inside the Economic WM:

- **Slow-manifold consistency / invariant residual modeling**: The
  `SlowManifoldProjection` interface enforces that fast receipt noise does
  not reparameterize slow macro state. A PINN-style residual layer can
  enforce known invariant relationships (e.g., energy conservation across
  subsystems, monotone wear accumulation, budget balance identities) as
  hard or soft constraints on the dynamics model's slow-manifold
  predictions.
- **Meso-timescale reservoir-flow transitions**: Battery depletion, thermal
  accumulation, queue pressure evolution, and compute-budget draw-down
  follow approximately known differential relationships with
  regime-dependent parameters. A PINN-informed dynamics submodule can
  encode these relationships and learn only the residual / regime-dependent
  corrections.
- **Battery / thermal / wear / queue / compute-budget evolution**: These
  `ResourceReservoir` quantities have partially known physics (thermal
  models, battery discharge curves, queue service models). Constraint-
  informed prediction layers can use those known dynamics as structural
  priors.
- **Counterfactual resource-transition rollouts**: When the dynamics model
  rolls out "what happens if sim budget is spent here?", the resource-
  transition component can be PINN-constrained to respect balance and
  conservation laws rather than learning them from data alone.
- **Invariant regularization**: Regularization terms around conserved
  quantities (energy budget balance), monotone quantities (cumulative wear),
  and dissipation-like quantities (entropy production in queue/flow systems)
  can structurally prevent the dynamics model from predicting physically
  impossible economic state transitions.

### Bad candidate placements

Where PINN framing is actively wrong for the Economic WM:

- **Allocator as a PINN**: The allocator's job is multi-objective
  optimization under constraints, not equation-solving. PINN structure
  would inappropriately rigidify the allocation policy.
- **Downward governance shaping as a PINN**: Governance transport carries
  budget envelopes, persistence annotations, and Pareto slices. These are
  policy artifacts, not solutions to differential equations.
- **Meta-regal composition as a PINN**: The inter-domain governance
  composition problem is regime-sensitive Pareto adjudication, not
  equation-constrained dynamics.
- **WM-to-WM transport as a PINN**: Transport bridges preserve
  representational structure through learned affine maps and contrastive
  alignment. No known PDE governs this translation.
- **Anything that would make economics look like a clean closed PDE world**:
  The Economic WM models a nonequilibrium, regime-switching, partially
  observed productive organism. The majority of its dynamics are
  stochastic, multi-agent, and subject to discrete regime transitions that
  are not PDE-governed.

### Design stance

**Constraint-informed Economic submodule: yes.**
**PINN-shaped Economic backbone: no.**

The PINN components should enter as bounded residual layers or
regularization terms inside the dynamics model and state estimator — not as
the architectural identity of the Economic WM. The backbone remains
switching-SSM + distributional Pareto allocator + typed governance
transport, with PINN-style constraint enforcement providing structural
priors where the underlying dynamics are partially known.

**Repo precedent:** the Embodiment WM uses analytic contact models as
heuristic fallbacks inside a promotion-gated learned dynamics seam—the same
division of labor: known structure where it holds, learned residual where it
does not.

## Anti-patterns

The future Economic WM must not collapse into:

- a glorified dashboard
- a reward oracle
- a scalarizer
- a PnL metaphor
- a mother-latent that erases typed contracts

It should remain:

- typed
- receipt-legible
- regime-aware
- multi-timescale
- tail-aware
- execution-aware
- allocative rather than merely evaluative
