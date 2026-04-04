# Ixion: Precondition Engine and Robotics Control Stack

> A multi-WM, economics-aware robotics stack built to train, control, and deploy real robots.

This repository contains the software, learning, simulation, control, and coordination substrate being built **before** the acquisition of operating assets. **This is a real robotics control stack.** It is being built to stand up perception, simulation, control, training, and coordination for real robot deployment. 

The explicit progression is:
**software-first standup &rarr; workcell training loop &rarr; first hardware integration &rarr; autonomous micro-workcell operation &rarr; richer loop &rarr; multi-robot coordination &rarr; future deployment across operating assets.**

This stack is the precondition engine for that future humanoid deployment and fleet coordination. It holds an embedded, learned economic logic that directly shapes simulation, training, and actuation.

## Bridging Economics and Deployment

Economics in this stack is a first-class control signal used to decide what to train on, what to simulate, what to prioritize, and how to allocate constrained compute and battery. Those choices are meant to eventually shape real workcell behavior, multi-robot coordination, and fleet management across future operating assets. This is why Ixion is the technical half of a broader organism—providing the actionable intelligence that **Industrial Cybernetica** will physically deploy.

## Why This Exists

Ixion exists to stand up the learning and control loop before any operating business is acquired. The capabilities being accumulated here are the concrete ingredients needed to train, control, and improve robots in real workcell settings before broader deployment:

- Technical ontology and state representation
- Data pipelines and evaluation loops
- Simulation priors and synthetic generation
- Embodiment-aware control logic for hardware
- Transferable task primitives
- Multi-robot coordination capabilities

The point is to de-risk the future industrial system by developing these robotics capabilities in embryo first. Future acquired businesses are intended to become deployment surfaces, learning surfaces, and cash-generating nodes governed by this control stack.

## Robotics Control and Deployment Path

This program is fundamentally a robotics control stack driven by continuous simulation and training. We are explicitly building:

- **Perception + semantic grounding** for stable, robot-relevant state
- **Simulation + synthetic branch generation** for rigorous policy and data improvement
- **Embodiment-aware control logic** for integration with real hardware
- **Replay, evaluation, and training loops** that iteratively improve robot behavior over time
- **Coordination logic** that becomes critical once the system scales to multiple robots and sites

## The Target Loop

`real video -> representation + semantics -> constrained generation -> physics/geometric sim -> policy improvement + datapacks -> economic telemetry + constraint evaluation -> programmable objectives -> better data decisions`

Economics sits inside the loop, acting as a first-class control signal. It exists to inform and prioritize our training runs, simulation decisions, and deployment behavior.

## Two Failure Modes We Must Prevent

1. `Gen->Sim garbage`
- Pretty synthetic outputs that are not trainable, transferable, or physically/plausibly valid for real robot behavior.

2. `Premature scalarization`
- Collapsing multi-objective tradeoffs too early, which destroys Pareto structure and breaks programmable governance contracts.

## Current Subsystem Maturity

There is already real technical substance functioning in the loop today, though the stack is not yet fully unified end-to-end.

**What is already concretely working and scaffolded:**
- Economics-aware training and logging infrastructure (`MPL`, error, energy, wage-parity style signals).
- Datapack valuation scaffolding and advisory orchestrator components.
- SceneIR / map-first / VLA evidence artifact boundaries.
- Seeded, deterministic sampling and curriculum infrastructure for reproducible learning.

**What is additive and available now (Shadow Implementation):**
- A functioning shadow economic control plane (`docs/shadow_economic_control_plane.md`).
- Execution scripts for shadow control and ablations (`scripts/run_shadow_econ_control_plane.py`, `scripts/run_shadow_econ_ablations.py`).

**What is not yet unified end-to-end:**
- `ObjectiveTensor`-first training/inference contracts natively across all layers.
- `EconTensor`-first accounting as the strict system of record.
- Meta-regal governance nodes as the default control plane.
- Real-time deployment-legibility sentinel fully integrated across learning loops.

## Canonical Multi-WM Topology

The stack is a series of multiple adjacent canonical World Models (WMs) communicating via **typed, replayable state surfaces**.

In this stack, **"typed"** is the primary defense against the "mother-latent" trap—the failure mode where all perception, control, and physics are collapsed into one uninterpretable vector embedding. Instead of passing opaque floats, WMs pass explicit, schema-backed contracts (e.g., `BeliefState`, `ObjectiveTensor`, `ConstraintSet`).

**These contracts are the typed surfaces that keep perception, simulation, control, economics, and governance legible to one another as the robot learns and deploys.** This ensures that critical metadata—like geometry, uncertainty, and safety boundaries—survive translation across models, allowing the Economic WM and governance nodes to audit and allocate resources based on legible reality rather than black-box approximations.

Additionally, datapacks are treated as structured composite objects whose source parts, transformation lineage, and functional contributions are tracked across WMs and later allocated under economic objectives. Epiplexity helps govern which compositions preserve the most actionable structure under constrained compute, training, and deployment conditions.

Each WM serves a concrete function in making the robot ready for deployment:

1. **Perception / grounding WM**: turns raw sensor/video streams into stable scene state the robot can actually act on. The stack expects to integrate real open-vocabulary concept segmentation and video object tracking provider lanes (e.g., SAM 3 / 3.1) here to feed canonical perception state, synthetic branch evaluation, and semantic annotation surfaces.
2. **Embodiment / actuation WM**: turns task intent + local world state + embodiment constraints into body-aware, capability-aware, contact-aware control state and action proposals for real robot embodiments. Owns six subsystems: capability/embodiment state surface, contact/affordance graph builder, local contact dynamics model, inverse-dynamics/retargeting lane, joint skill/action proposal head, and drift/calibration/cost evaluator. See `docs/actuation_embodiment_world_model.md`.
3. **Sim / synth / physics WM**: owns the simulated, synthesized, and physics-evaluated branch of the data engine. Internally decomposed into ten subsystems: backend/runtime/provider surface, task/measurement/episode layer, scene/asset/materialization layer, branch planner/evaluator, sim-real gap evaluator, fidelity/randomization/calibration allocator, render/diffusion/materialization lane, differentiable-physics provider lane, drift/calibration/mismatch evaluator, and training-worthiness/synthetic-yield evaluator. See `docs/economic_world_model/multi_wm_architecture_plan.md`.
4. **Economic WM**: decides what tasks, environments, data, and training runs matter most under throughput, error, energy, and labor constraints.
5. **Meta-node superposition / control WM**: the later cross-WM policy and governance layer for multi-objective Pareto optimization.

The multi-WM roadmap increasingly moves toward **canonical WM ownership**, **internal subsystem decomposition**, **typed receipts and interfaces**, and **bounded neural seams** inside each WM. Each WM section is held to a 9-point readiness standard documented in the architecture plan.

Lower WMs own typed canonical state, the economic WM sits above them to allocate resources logically, and cross-WM transport acts as middleware. The next high-leverage priority is the **sim / synth / physics WM**.

## Roadmap

### Architecture Sequence
- Sim / synth / physics WM
- Perception / grounding WM
- Embodiment / actuation WM
- Economic WM consolidation over lower-WM receipts
- Local meta-node neuralization
- Cross-WM transport
- Meta-node superposition / control WM

### Program Timing
The sequence is programmatic and cumulative, building toward live deployment:

- **Mar–Aug 2026**: Software-first loop standup and structural plumbing. Establish the deterministic data engine.
- **Sep 2026 onward**: Rhythmic workcell training and evaluation. Provider bring-up and continuous replay accumulation driven by economic constraints.
- **2027**: First Unitree/G1 hardware integration window. Ground the control loop on real embodiments.
- **By Sep 30, 2027**: Autonomous micro-workcell regime. Demonstrate a closed cybernetic loop running locally and safely.
- **Longer-horizon**: Broader loop maturity, multi-robot coordination, and expansion into acquisition-facing deployment surfaces across an industrial fleet.

### Execution Model

The repo has an intended execution model for the Sep 2026 GPU phase onward:

- **Local**: lightweight edits, fast verification, deterministic checks
- **Codex cloud**: code-only parallel work, scans, reviews, additive scaffolding
- **RunPod**: GPU-backed loop runs, provider bring-up, replay generation, training/eval, heavy refactor validation
- **Roadmap execution companion**: a repo-native agent pattern that reads roadmap docs, artifacts, receipts, and runs to surface bottlenecks and propose next-highest-leverage work

See `AGENTS.md` and `docs/agent_ergonomics/` for details.

## Ixion and Industrial Cybernetica

Ixion is one half of a planned organismal relationship:
- **Industrial Cybernetica**: The acquisition, asset-governance, and industrial-allocation side. This is intended to be tied directly to physical deployment surfaces, fleet deployment, and industrial coordination—not just a financial wrapper around the repo.
- **Ixion**: The learning, simulation, control, coordination, and primitive-transfer side.

Ixion is the precondition engine that lets future acquisitions become more than conventional EBITDA streams. Without Ixion, Industrial Cybernetica collapses toward a conventional holdco. Without Industrial Cybernetica, Ixion risks becoming merely robotics software.

## Intended Architectural Contracts

The stack relies on these internal architecture, control, and governance contracts:
- `ObjectiveTensor`: portable multi-objective representation that remains intact until explicit compile/scalarization.
- `ObjectiveCompiler`: explicit scalarization by contract/profile, never implicit upstream collapse.
- `ConstraintSet`: VLA + geometry manifold constraints used to condition generation/simulation.
- `EconTensor`: accounting representation coupled to objective outcomes and constraints.
- `PricingSentinel`: high-frequency pricing stream with auditable aggregated ledger writes.
- `Regal* nodes`: policy/governance gates for objective integrity, reward safety, plausibility, deployment-truth, and data value.

## Existing Build and Verification Commands

```bash
# Install
python3 -m pip install -r requirements-dev.txt

# Core checks
python3 -m compileall src/
pytest tests/ -v
ruff check .
ruff format .
mypy src/
```

Recommended fast verification loop after edits:
```bash
python3 -m compileall src/ && pytest tests/ -v
```

## Key Smoke Tests

```bash
# Feature extractor sanity
python3 scripts/test_episode_features.py

# Dishwashing smoke + summaries
python3 scripts/smoke_test_dishwashing_sac.py --episodes 5 --econ-preset toy

# Phase C HRL/VLA smoke
python3 scripts/smoke_test_phase_c_hrl_vla.py --episodes 3

# Workcell env suite
python3 scripts/smoke_workcell_env.py
```

## Golden Path Demo (5 Minutes)

Run one deterministic end-to-end proof that emits ObjectiveTensor, compiler scalar reward, Objective->Econ deltas, governance pass/fail reasons, and plots:

```bash
python3 scripts/run_golden_path.py --env workcell --episodes 10 --seed 0 --emit artifacts/golden_path
```

Expected outputs:
- `artifacts/golden_path/objective_tensors.jsonl`
- `artifacts/golden_path/scalar_rewards.json`
- `artifacts/golden_path/econ_deltas.json`
- `artifacts/golden_path/governance_report.json`
- `artifacts/golden_path/artifact_bundle.json`
- `artifacts/golden_path/plots/objective_scalar.png`
- `artifacts/golden_path/plots/econ_governance.png`

Whitepaper and architecture notes:
- `docs/whitepaper/README.md`
- `docs/whitepaper_objective_tensor_stack.md`

## Quality Ratchet (Ruff + Mypy)

The repo carries legacy lint/type debt. CI is configured to block regressions while cleanup proceeds.

Ratchet baselines:
- `config/quality_ratchet.json`

Commands:
```bash
python3 scripts/ci/check_ruff_ratchet.py
python3 scripts/ci/check_mypy_ratchet.py
```

Cleanup sequencing reference:
- `docs/quality_ratcheting.md`

## Frozen and Additive Zones

Phase B now has a split posture:
- the stable baseline remains frozen
- additive successor modules beside that baseline are allowed

Do not modify:
- the stable checkpoint or legacy baseline world-model math
- `checkpoints/stable_world_model.pt`
- Trust net, `w_econ` lattice objective, lambda controller equations
- `src/controllers/synthetic_weight_controller.py` core logic

Allowed additive work:
- governed video-state or other successor modules in `src/world_model/` that preserve the stable baseline as the rollback anchor
- evidence, governance, and runtime sidecars that supervise future world-model work without collapsing external teachers into native truth

Additive-only zones:
- Energy bench extensions
- Orchestrator advisory wiring
- Phase C scaffolding (`HRL/VLA/SIMA`)

## Repository Map (Economics and Loop-Relevant)

- `src/rl/reward_shaping.py`: shared scalar reward shaping contract (current)
- `src/economics/reward_engine.py`: advisory reward decomposition and econ aggregation
- `src/ontology/models.py`: `EconVector` and core ontology dataclasses
- `src/ontology/store.py`: JSONL persistence for episodes/events/econ vectors
- `src/valuation/datapack_schema.py`: datapack metadata and objective profile fields
- `src/policies/unified_quality.py`: training-time quality weighting/eligibility
- `src/rl/episode_sampling.py`: deterministic sampling/curriculum hooks
- `src/vision/scene_ir_tracker/*`: SceneIR artifact serialization and provenance
- `src/vision/map_first_supervision/*`: geometry/map-first pseudo-supervision artifacts
- `src/vla/semantic_evidence.py`: VLA semantic evidence sidecar format
- `src/orchestrator/diffusion_requests.py`: diffusion prompt/request bridge (current conditioning seam)

## Program Teleology in One Line

Build the precondition engine through which simulation, control, transfer, coordination, and economic allocation can be accumulated before live industrial deployment.
