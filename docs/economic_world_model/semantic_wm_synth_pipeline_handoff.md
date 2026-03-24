# Semantic WM Synth Loop Handoff

Date: 2026-03-24

Audience: Claude follow-on implementation pass

## What Exists Now

The semantic world model is now real at the runtime packet level:

- Stage 1 and semantic fusion both materialize `SemanticWorldModelState`
- the semantic WM feeds:
  - `SemanticSnapshot`
  - `SemanticOrchestratorV2`
  - `MetaTransformer`
  - `OrchestrationTransformer`
  - observation / condition / sampler surfaces
- replay-backed runtime rows now join:
  - semantic WM
  - OpenVLA / teacher evidence
  - DINO / SceneTracks / Map-First proxy evidence
  - runtime outcomes
  - shadow counterfactuals
  - route / authority / regret labels
- lightweight and heavyweight scorer training paths now exist on top of that row schema

This means the semantic WM is no longer only descriptive. It is now a bounded control-plane substrate.

## Where The Loop Is Still Incomplete

The semantic WM is still under-instantiated relative to diffusion, simulation, env primitives, and the synthetic-data flywheel.

### 1. Diffusion is semantic-tag aware, not semantic-WM aware

Current path:

- `src/orchestrator/diffusion_requests.py`

What it does now:

- builds prompts from datapack guidance plus `ConstraintSet`
- carries `skill_ids`, `semantic_tags`, `objective_vector`, `customer_segment`, and VLA hints

What is missing:

- semantic WM topology
- semantic WM capability scores
- meta-node priorities
- grounded object/relation state
- explicit gap-driven prompt generation

Effect:

Diffusion requests are still generated from datapack-level guidance and flat tag mixtures, not from a compiled semantic deficit over task x env-primitive x skill coverage.

### 2. Semantic simulation is scenario-aware, not semantic-WM compiled

Current path:

- `src/orchestrator/semantic_simulation.py`
- `src/orchestrator/semantic_policy.py`

What it does now:

- selects datapacks/scenarios by tags, robot family, objective hint
- runs backend training/eval
- labels rollouts with VLA
- runs semantic fusion for rollouts

What is missing:

- semantic WM does not compile simulation agendas
- no shared representation of which task/environment primitive combinations are under-covered
- no semantic-WM-driven simulation scheduling across env backend, objective preset, skill family, risk family, and data gaps

Effect:

The sim loop closes after rollout labeling/fusion, but the next sim to run is not yet chosen from the semantic WM.

### 3. There is no global skill graph tied to env primitives

Current local assets:

- `src/hrl/skills.py` defines drawer+vase low-level skills
- `src/sima/co_agent.py` emits skill sequences for that domain
- `src/hrl/skill_termination.py` gives completion/reward logic for those skills
- some envs expose skill-energy accounting, but not a shared primitive graph

What is missing:

- a repo-level skill graph spanning tasks and envs
- an env-primitive inventory per environment/backend
- a task-to-required-primitive map
- a semantic coverage matrix saying:
  - which task primitives exist
  - which skill edges exist
  - which env backends realize them
  - which datapacks / rollouts / synthetic branches cover them

Effect:

The semantic WM cannot yet answer the important production question:

"What task coupled with what env primitive or skill edge do we not have enough grounded evidence for?"

### 4. Synthetic pipeline is trust/econ gated, but semantically blind

Current paths:

- `scripts/train_world_model_from_datapacks.py`
- `scripts/collect_local_synthetic_branches.py`
- `scripts/train_latent_diffusion.py`
- `scripts/train_trust_aware_world_model.py`
- `scripts/train_offline_policy.py`

What they do now:

- generate or consume synthetic branches
- gate with trust and economic weights
- train latent/world-model/offline-policy components

What is missing:

- semantic WM does not decide which local bubbles to branch around
- synthetic branches are not labeled by task primitive coverage gap
- latent diffusion is not conditioned on semantic WM topology/meta-nodes
- there is no semantic target saying which missing primitive transitions or risky affordance cases diffusion should synthesize

Effect:

The synth loop is economically and trust filtered, but not semantically compiled.

### 5. Broader regal nodes are not yet compiled into the semantic WM

Important broader nodes already present in the repo:

- econ signals
- datapack signals
- trust
- shell activation / readiness
- governance trace / decision ledger / event spine
- promotion evidence
- epiplexity / novelty surfaces
- counterfactual / value-target artifacts

What is missing:

- semantic WM does not persist a canonical "coverage + value + readiness" slice that composes these nodes
- the transformers see semantic/econ/data state, but diffusion/sim/synth selection does not yet use the same compiled packet

Effect:

The system still has parallel control surfaces instead of one shared semantic-routing substrate.

## What Claude Should Build Next

The next pass should make the semantic WM the compiler for simulation and synthesis agendas, not just the observer of them.

### A. Add a semantic coverage graph beside the semantic WM

Recommended additive module:

- `src/world_model/semantic_coverage_graph.py`

Core object:

- `SemanticCoverageGraph`

Fields it should contain:

- `task_nodes`
- `skill_nodes`
- `env_primitive_nodes`
- `backend_nodes`
- `object_family_nodes`
- `risk_family_nodes`
- `affordance_family_nodes`
- `coverage_edges`
- `missing_edges`
- `evidence_strength`
- `economic_priority`
- `trust_priority`
- `promotion_readiness`

This should be built from:

- semantic WM
- replay/runtime corpus
- datapack summaries
- scenario metadata
- HRL skill definitions
- SIMA primitive tags
- env/backend inventories
- broader regal/readiness nodes

### B. Introduce environment primitive inventories

Recommended additive modules:

- `src/envs/primitive_inventory.py`
- per-env adapters under `src/envs/...`

Each env/backend should export:

- manipulable object families
- contact primitives
- navigation/motion primitives
- risk primitives
- recovery primitives
- observation limitations
- backend-specific constraints

Examples:

- drawer+vase:
  - locate handle
  - detect fragile obstacle
  - safe approach
  - grasp handle
  - open while maintaining clearance
  - retract
- dishwashing:
  - locate dish
  - grasp utensil/object
  - contact surface
  - scrub
  - rinse
  - place/stow
- workcell:
  - pick
  - place
  - insert
  - align
  - avoid collision
  - recover from occlusion / uncertainty

### C. Promote HRL skills into a real repo-level skill graph

Recommended additive module:

- `src/hrl/skill_graph.py`

This should unify:

- `src/hrl/skills.py`
- SIMA primitive sequences
- VLA semantic action hints
- Stage 2 task-graph proposals

Needed outputs:

- canonical skill graph
- task-to-skill requirements
- skill-to-env-primitive requirements
- transition coverage counts
- missing skill transition edges

This is the substrate for answering:

- which skill edges are missing
- which env backends can realize them
- which objects/risk families they involve
- which data source should fill the gap

### D. Compile diffusion requests from semantic deficits

Update:

- `src/orchestrator/diffusion_requests.py`

It should consume:

- semantic WM
- semantic coverage graph
- meta-node priorities
- capability shortfalls
- economic urgency
- trust / readiness / promotion signals

New prompt fields should include:

- `topology_slice`
- `meta_node_targets`
- `missing_skill_edges`
- `missing_env_primitives`
- `risk_family_targets`
- `affordance_family_targets`
- `coverage_gap_score`
- `economic_priority_score`
- `trust_priority_score`

Diffusion should then be asked for specific missing semantics, not generic tag mixtures.

Example:

"Generate drawer-vase rollouts for `PLAN_SAFE_APPROACH -> GRASP_HANDLE` under partial occlusion with fragile-object proximity, because this edge is economically high-value, trust-safe to synthesize, and under-covered in real data."

### E. Compile simulation agendas from the same graph

Update:

- `src/orchestrator/semantic_simulation.py`

Instead of only selecting by tags/objective hint, it should build ranked simulation jobs such as:

- task family
- env/backend
- skill edge or primitive edge
- risk family
- object family
- objective preset
- energy profile
- data-collection intent

Simulation ranking should consider:

- missing semantic coverage
- expected economic upside
- trust/readiness constraints
- backend realism needed
- whether diffusion or direct sim is the better data source

This should produce an explicit sim agenda artifact, not just one selected run.

Recommended artifact:

- `simulation_agenda_v1.json`

### F. Feed broader regal nodes into semantic WM / coverage graph

The semantic WM and coverage graph should ingest:

- econ urgency
- datapack coverage / diversity / gaps
- trust
- execution readiness
- shell activation backlog state
- governance / decision / event trace presence
- counterfactual value signals
- epiplexity / novelty

These should affect:

- which missing edges matter
- whether to run real sim, passthrough sim, diffusion, or no-op
- whether a route is learnable, promotable, or blocked

### G. Make synthetic branch collection semantic-gap aware

Update:

- `scripts/collect_local_synthetic_branches.py`
- `scripts/train_world_model_from_datapacks.py`
- `scripts/train_trust_aware_world_model.py`

Needed changes:

- branch seeds should be selected from semantically under-covered states
- branches should be labeled with:
  - skill edge
  - env primitive edge
  - risk family
  - affordance family
  - coverage-gap contribution
  - expected economic contribution
- downstream weighting should combine:
  - trust
  - w_econ
  - semantic gap value
  - readiness / promotion safety

### H. Condition latent diffusion on semantic WM state

Update:

- `scripts/train_latent_diffusion.py`

The diffusion stack should learn over conditioning that includes:

- semantic tokens
- object/relation summaries
- skill-edge targets
- env-primitive targets
- risk/affordance families
- economic objective preset
- trust/readiness class

This lets diffusion synthesize targeted missing semantics rather than generic latent variations.

### I. Add a full-loop agenda/evidence cycle

The intended loop should be:

1. ingest real data
2. update semantic WM
3. build semantic coverage graph
4. rank missing task x skill x env-primitive edges
5. choose fill path:
   - real sim
   - diffusion synthesis
   - local latent branch expansion
   - no-op / blocked
6. run selected generation path
7. label/fuse outputs back into semantic WM
8. update replay/runtime corpus
9. retrain scorers and later controllers

## Concrete File-Level Plan

### Priority 1: graph and inventory substrate

Add:

- `src/world_model/semantic_coverage_graph.py`
- `src/hrl/skill_graph.py`
- `src/envs/primitive_inventory.py`

Tests:

- coverage graph construction from semantic WM + skill graph + env inventory
- missing-edge ranking
- skill-edge / primitive-edge aggregation

### Priority 2: simulation agenda compilation

Update:

- `src/orchestrator/semantic_simulation.py`
- `src/orchestrator/semantic_policy.py`

Add:

- ranked `simulation_agenda_v1`
- semantic-gap-to-sim-job compiler
- backend selection using realism / trust / cost / readiness

Tests:

- semantic gap ranking produces stable sim agendas
- economically urgent missing edges outrank low-value edges

### Priority 3: diffusion integration

Update:

- `src/orchestrator/diffusion_requests.py`
- Stage 1 diffusion request path

Add:

- semantic-WM-conditioned diffusion prompt fields
- skill-edge and env-primitive targets
- coverage-gap rationale

Tests:

- diffusion prompt includes missing-edge fields
- objective/risk/capability alignment is preserved

### Priority 4: synthetic branch integration

Update:

- `scripts/collect_local_synthetic_branches.py`
- `scripts/train_world_model_from_datapacks.py`
- `scripts/train_trust_aware_world_model.py`

Add:

- semantic gap labels on synthetic branches
- semantic-gap-aware branch selection
- semantic gap value in weighting/export

Tests:

- branch labels survive export/import
- branch ranking shifts with semantic coverage deficits

### Priority 5: loop closure into runtime learning

Update:

- `src/orchestrator/semantic_runtime_learning.py`
- `src/orchestrator/semantic_runtime_scorers.py`
- `src/orchestrator/semantic_runtime_scorer_training.py`

Add labels for:

- missing-edge coverage improvement
- sim agenda success
- diffusion fill quality
- synthetic branch value by semantic gap family

## Specific Repo Truths Claude Should Respect

- Keep all of this additive.
- Do not modify the frozen Phase B baseline math/checkpoint/trust-net/w_econ lattice/lambda controller core logic.
- Keep external teacher/VLA outputs advisory as truth sources, but do make their availability/confidence/non-stub status hard metadata.
- Prefer compiling shared packets and artifacts over creating more parallel side systems.

## Short Plain-English Translation

Right now the semantic WM can describe grounded scene/task state and guide bounded transformer routing. It still does not decide, in a unified way, what synthetic or simulated data should be generated next.

The next pass should make the semantic WM plus a new coverage graph answer:

- what task-skill-env-primitive combinations are missing
- which of those gaps are economically meaningful
- which are safe and ready enough to pursue
- whether they should be filled by real sim, diffusion, or local synthetic branching

That is the missing semantic control loop for the synth pipeline.
