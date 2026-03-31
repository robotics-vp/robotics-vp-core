# Ixion: Precondition Engine and Robotics Control Stack

> A multi-WM, economics-aware robotics stack built to train, control, and deploy real robots.

This repository contains the software, learning, simulation, control, and coordination substrate being built **before** the acquisition of operating assets. It is a concrete **robotics control stack** built to stand up perception, simulation, control, training, and coordination for real robot deployment, with an embedded and learned economic logic shaping simulation, training, and actuation—beginning with workcell environments and later scaling outward.

This stack is the technical foundation of **Ixion**—understood as the intelligence and control side of a broader future organism, where **Industrial Cybernetica** will serve as the acquisition, industrial-allocation, and operating-asset side.

## Why This Exists

Ixion exists to stand up the learning and control loop before any operating business is acquired. The capabilities being accumulated here are not abstract priors—they are the concrete ingredients needed to train, control, and improve robots in real workcell settings before broader deployment:

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

The intended programmatic progression is:
**software-first standup &rarr; workcell training loop &rarr; first hardware integration &rarr; autonomous micro-workcell operation &rarr; richer loop and broader deployment**

## The Target Loop

`real video -> representation + semantics -> constrained generation -> physics/geometric sim -> policy improvement + datapacks -> economic telemetry + constraint evaluation -> programmable objectives -> better data decisions`

Economics sits inside the loop, acting as a first-class control signal. It exists to inform and prioritize our training runs, simulation decisions, and deployment behavior, rather than functioning merely as an abstract thesis.

## Two Failure Modes We Must Prevent

1. `Gen->Sim garbage`
- Pretty synthetic outputs that are not trainable, transferable, or physically/plausibly valid for real robot behavior.

2. `Premature scalarization`
- Collapsing multi-objective tradeoffs too early, which destroys Pareto structure and breaks programmable governance contracts.

## Canonical Multi-WM Topology

The stack is a series of multiple adjacent canonical World Models (WMs) communicating via **typed, replayable state surfaces**.

In this stack, **"typed"** is the primary defense against the "mother-latent" trap—the failure mode where all perception, control, and physics are collapsed into one uninterpretable vector embedding. Instead of passing opaque floats, WMs pass explicit, schema-backed contracts (e.g., `BeliefState`, `ObjectiveTensor`, `ConstraintSet`). This ensures that critical metadata—like geometry, uncertainty, and safety boundaries—survive translation across models, allowing the Economic WM and governance nodes to audit and allocate resources based on legible reality rather than black-box approximations.

Each WM serves a concrete function in making the robot ready for deployment:

1. **Perception / grounding WM**: turns raw sensor/video streams into stable scene state the robot can actually act on.
2. **Embodiment / actuation WM**: turns task intent into body/action/capability-aware control state for real robot embodiments.
3. **Sim / synth / physics WM**: decides what to simulate, what to synthesize, what backend/fidelity to use, and what synthetic branches are worth feeding back into training.
4. **Economic WM**: decides what tasks, environments, data, and training runs matter most under throughput, error, energy, and labor constraints.
5. **Meta-node superposition / control WM**: the later cross-WM policy and governance layer for multi-objective Pareto optimization.

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
- **Mar–Aug 2026**: Software-first loop standup and structural plumbing.
- **Sep 2026 onward**: Workcell training/eval rhythm, provider bring-up, and continuous replay accumulation.
- **2027**: First Unitree/G1 hardware integration window.
- **By Sep 30, 2027**: Target of an autonomous micro-workcell regime.
- **Longer-horizon**: Later broader loop maturity, multi-robot coordination, and expansion into acquisition-facing deployment surfaces.

## Current Subsystem Maturity

There is not yet a complete economic control layer in production.

What exists today:
- Economics-aware training and logging scaffolding (`MPL`, error, energy, wage-parity style signals).
- Datapack valuation scaffolding and advisory orchestrator components.
- SceneIR / map-first / VLA evidence artifact boundaries.
- Seeded, deterministic sampling and curriculum infrastructure.

What does not yet exist as a fully unified runtime layer:
- ObjectiveTensor-first training/inference contracts end-to-end.
- EconTensor-first accounting end-to-end.
- Meta-regal governance nodes as the default control plane.
- Real-time deployment-legibility sentinel integrated across learning loops.

Additive shadow implementation available now:
- [`docs/shadow_economic_control_plane.md`](docs/shadow_economic_control_plane.md)
- [`scripts/run_shadow_econ_control_plane.py`](scripts/run_shadow_econ_control_plane.py)
- [`scripts/run_shadow_econ_ablations.py`](scripts/run_shadow_econ_ablations.py)

## Ixion and Industrial Cybernetica

Ixion is one half of a planned organismal relationship:
- **Industrial Cybernetica** = future acquisition / asset-governance / industrial-allocation side.
- **Ixion** = learning / simulation / control / coordination / primitive-transfer side.

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
