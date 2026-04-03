# Doctrine: Meta-Regal-Node Superposition / Control WM

## Status

Future-architecture doctrine. This WM has not been built. This document
defines its structural role, why it is needed, how it relates to the
Economic WM, and what it must compose.

## Why the Economic WM cannot be sovereign

The stack's telos is not "optimize economics." The telos is **governed robot
control under multiple non-collapsible realities**:

- physical plausibility
- safety
- anti-reward-hacking
- deployment truth
- embodiment limits
- coordination integrity
- and only one of those is economic allocation

The Economic WM can be powerful, upstream, and nontrivial without being the
final court. If it becomes too central, the stack risks a subtle overfit:

- everything gets translated into economic language
- local subsystem truths get compressed into allocative abstractions
- physical/safety/deployment reality gets treated as constraints subordinate
  to an economic worldview

That is dangerous in a control stack. For a real robot, economics should shape
action selection, training selection, simulation allocation, compute routing,
and deployment priorities — but it should never be allowed to silently
redefine:

- what is physically real
- what is safe
- what is robust
- what is non-deceptive
- what is epistemically trustworthy

The Economic WM is a **first-class allocative world model**, but it is **not
the sovereign governor of the stack**. It participates as one major
contributor within a higher-order superpositioning meta-regal-node WM.

## Three levels of governance

The stack has at least three governance levels:

### Level 1: Subsystem / local WM level

Perception, embodiment, sim/synth, economic, etc. Each has local truths and
local shaping logic. Each owns canonical state, receipts, and promotion
posture within its boundary.

### Level 2: Domain governance level

Multiple distinct normative and descriptive evaluative-governance surfaces:

- **Economic WM**: resource allocation, opportunity cost, compute/energy/wear
  tradeoffs, value of information, data valuation, task prioritization
- **Anti-reward-hacking governance**: reward integrity, exploit detection,
  deceptive-optimization suspicion, reward-channel consistency
- **Plausibility / geometry governance**: physical plausibility, sim-real
  consistency, grounding quality, geometric truth
- **Deployment truth governance**: deployment posture honesty, provider truth,
  runtime availability, install/preflight reality
- **Safety governance**: safety constraint satisfaction, emergency override,
  degraded-mode handling, human oversight compliance
- **Data value governance**: data quality, annotation integrity, replay
  fidelity, training corpus honesty
- **Coordination governance** (later): multi-agent, multi-site, operator
  handoff, communication integrity

Each of these becomes a neuralized or semi-neuralized evaluative-governance
process. Their outputs must themselves be composed, adjudicated, and
stabilized.

### Level 3: Superposition / meta-governance level

The WM that models and composes the governance nodes themselves. This is the
meta-regal-node WM.

This third layer keeps the system from overfitting to any one domain ontology,
including economics.

## Why "superposition" is the right word

The governance nodes should not necessarily collapse immediately into one
scalar or one strict hierarchy.

Sometimes:

- anti-reward-hacking is dominant
- sometimes plausibility is dominant
- sometimes deployment-truth is dominant
- sometimes economics matters most within a safe feasible region
- sometimes coordination integrity becomes the relevant macro pressure

The nodes may coexist in a partially unresolved relation until regime,
embodiment state, task family, deployment mode, and confidence conditions tell
the meta-layer how to compose them.

This is very different from "economics outputs reward, regal nodes just clip
it." Instead:

- a structured superposition
- regime-sensitive composition
- partial vetoes
- admissible regions
- soft and hard constraint interplay
- typed provenance for why a control decision was shaped the way it was

## Two kinds of Pareto optimization

The stack has two fundamentally different Pareto problems:

### 1. Intra-domain Pareto optimization

Inside the Economic WM: throughput vs energy vs wear vs compute vs error vs
exploration vs data yield, etc.

This is multi-objective optimization within a single evaluative domain.

### 2. Inter-domain Pareto optimization

Inside the meta-regal-node WM: economics vs anti-reward-hacking vs
plausibility vs deployment truth vs safety vs coordination integrity.

This is more fundamental, because it governs whether the intra-domain
optimization can even be trusted in a given regime.

The inter-domain Pareto problem is not simply "add weights across domains."
It is: how do multiple trained evaluative-governance processes interact, when
do they veto, when do they soften, when do they enter Pareto relation versus
hard constraint relation, and how do they remain legible under conflict?

This distinction must be made explicit in the architecture.

## What the meta-regal-node WM must model

### Meta-governance state

- current regime
- conflict structure among node outputs
- node confidence / trust
- active hard constraints
- admissible Pareto region across node surfaces
- persistence / hysteresis in governance mode

### Meta-governance composition

The meta-layer must know when node outputs are:

- in Pareto relation (tradeoffs exist, no node strictly dominates)
- lexicographic (one node takes priority, others shape within feasible region)
- veto-like (one node imposes hard constraint that others must satisfy)
- advisory (one node contributes information but not binding constraint)
- confidence-weighted (composition depends on node epistemic confidence)

### Meta-governance transport

Downward:

- node-composed shaping/constraint fields to lower WMs
- composed budget envelopes filtered through safety/plausibility/integrity
- explicit provenance of which nodes shaped which decisions

Upward:

- conflict receipts (which nodes disagreed, how severely)
- override receipts (which node overrode which, with what justification)
- governance failure receipts (when composition failed or was incoherent)
- reward-hack suspicion receipts
- deployment-truth discrepancy receipts

## The Economic WM's role under meta-governance

The Economic WM is **not demoted into irrelevance**. It still matters a lot
because it contributes things the other nodes do not:

- resource allocation logic
- opportunity cost accounting
- compute/energy/time/wear tradeoffs
- value of information
- value of simulation
- value of exploration
- data valuation
- task/deployment prioritization

Its role becomes: **first-class contributor, not sole governor**.

The Economic WM provides a major slice of the meta-governance surface, but it
lives inside a broader superposed governance field. It is the strongest
allocative voice in the stack, but the meta-layer can override or constrain
its recommendations when safety, plausibility, reward integrity, or
deployment truth require it.

## Staged neuralization

The meta-regal-node WM should be neuralized later, after the lower WMs and
Economic WM are mature. The staging is:

### Stage A: typed non-neural governance scaffolding

- define governance node surfaces and their outputs
- define conflict/override receipts
- define composition modes (Pareto, lexicographic, veto, advisory)
- define admissible-region schemas

### Stage B: governance node neuralization

Each domain-governance node becomes a trained evaluative process:

- economic allocator (see Economic WM doctrine)
- anti-reward-hacking detector / critic
- plausibility evaluator (grounding quality, physical consistency)
- deployment-truth verifier
- safety constraint checker
- data value estimator

### Stage C: meta-composition learning

The meta-layer learns how to compose node outputs:

- regime-conditioned composition weights
- conflict resolution under uncertainty
- when to escalate to hard constraint vs soft Pareto
- persistence / hysteresis in governance mode

Architecture: multi-objective actor-critic with Pareto front tracking,
conditioned on regime state and node confidence. Possibly hypernetwork-
conditioned policy over the composition space.

### Stage D: transport and feedback

The meta-layer learns:

- how to shape lower WMs without thrashing
- how to detect governance failures and self-correct
- how to maintain legibility under node conflict

## Preconditions before building

1. Lower WMs (perception, embodiment, sim/synth) are at `bounded_runtime_authority` or beyond
2. Economic WM is consuming canonical lower-WM receipts and emitting allocation envelopes
3. Transport bridges between adjacent WMs are working
4. Local meta-node objects have passed their own neuralization tranche
5. Governance nodes are emitting typed receipts (not just heuristic gates)
6. Meta-node actions and governance satisfaction are logged as trainable data

## Anti-patterns

Do not:

- build the meta-regal-node WM before the lower WMs are mature
- let the Economic WM silently become the meta-layer by default
- let any single governance node become the total governance surface
- collapse the meta-layer into a scalar "governance score"
- build an opaque meta-controller that cannot explain its composition decisions
- let the meta-layer directly replace lower-WM local truths
- treat governance composition as a solved add-weights problem
- confuse intra-domain Pareto (within Economic WM) with inter-domain Pareto
  (across governance nodes)

## Governance pluralism principle

The architecture preserves **pluralism at the governance layer** while allowing
strong specialization below it.

- each governance node specializes in its domain
- the meta-layer composes without collapsing
- no single domain ontology (including economics) can silently redefine the
  others
- the composition is regime-sensitive, confidence-aware, and typed
- the provenance chain from governance decision back to contributing nodes
  remains legible and auditable

This is not political philosophy. It is an engineering requirement for a
control stack that must remain safe, honest, and non-deceptive under real
deployment conditions.
