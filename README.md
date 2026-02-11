# robotics-vp-core: Economics-First Robotics Stack

> Robots priced like labor, not software.

This repository is building a closed cybernetic loop where economics is a first-class control signal, not an afterthought.

## Why This Exists
Most robotics programs fail commercially for the same reason:
- Unit economics are weak at deployment time.
- Pricing is static even when performance and risk are dynamic.
- Customers are not economically incentivized to share data.
- Data generation is decoupled from marginal business value.

This project targets those failures directly.

The target loop is:

`real video -> representation + semantics -> constrained generation -> physics/geometric sim -> policy improvement + datapacks -> economic telemetry + pricing -> customer-programmable objectives -> better data decisions`

## Two Failure Modes We Must Prevent
1. `Gen->Sim garbage`
- Pretty synthetic outputs that are not trainable, transferable, or physically/plausibly valid.

2. `Premature scalarization`
- Collapsing multi-objective tradeoffs too early, which destroys Pareto structure and breaks programmable customer contracts.

## Current Reality (Important)
There is not yet a complete roboeconomic layer in production in this repo.

What exists today:
- Economics-aware training and logging scaffolding (`MPL`, error, energy, wage-parity style signals).
- Datapack valuation scaffolding and advisory orchestrator components.
- SceneIR / map-first / VLA evidence artifact boundaries.
- Seeded, deterministic sampling and curriculum infrastructure.

What does not yet exist as a fully unified runtime layer:
- ObjectiveTensor-first training/inference contracts end-to-end.
- EconTensor-first accounting end-to-end.
- Meta-regal governance nodes as the default control plane.
- Real-time pricing sentinel integrated across deployment + learning loops.

This README defines that direction concretely.

## Economic Thesis
Robotics will not scale if pricing assumes static software margins.

For many deployments, hardware economics force a different model:
- A robot at `$20k-$30k` per unit cannot be monetized reliably with flat pricing.
- Price must follow realized task-hour value under constraints (error, safety, energy, uptime).
- Data sharing must have explicit rebates/credits tied to marginal frontier gain.

That implies:
- Real-time economic telemetry must be in the loop.
- Inferential training at deployment must be economically gated.
- Sim generation must be conditioned on expected economic value for the specific `task x env x trajectory`.

## Concrete Economics Layer (Current vs Target)
Current (implemented today, partial):
- Episode-level economics: `MPL`, error, energy, wage-parity style metrics.
- Advisory valuation and sampling infrastructure tied to datapacks.
- Seeded deterministic loops for replay/sampling and orchestration overlays.

Target (explicitly staged in this repo):
- `ObjectiveTensor` stays intact across real/diffusion/sim/training until contract compile.
- `ObjectiveCompiler` performs explicit scalarization at run/contract boundary.
- `ConstraintSet` carries VLA + geometry manifold bounds into generation requests.
- `ObjectiveEconFunctor` maps objective outcomes + violations + uncertainty into econ deltas.
- `PricingSentinel` emits high-frequency task-hour pricing ticks; `ValueLedger` keeps sparse, auditable receipts.

This is the shift from \"economics as logging\" to \"economics as control and accounting\".

## What "Economics-First" Means Here
Economics-first means we optimize and price against deployment outcomes:
- Throughput / MPL
- Error and safety risk
- Energy cost under real grid conditions
- Reliability and uncertainty
- Marginal data value (frontier expansion per unit compute)

Not all customers optimize the same objective.
Objective programmability is required to map customer contracts into policy behavior.

## Concrete Program Roadmap
The roadmap below is ordered as execution layers and capital-efficiency milestones.

| Stage | Goal | Concrete Output | Economic Impact |
|---|---|---|---|
| 1 | Synthetic data engine loop | Real video -> diffusion -> sim -> policy updates/datapacks | Faster capability iteration |
| 2 | Economic telemetry in-loop | Per-episode/per-task telemetry streams (`MPL`, error, safety, energy, uncertainty) | Sim/training focus shifts to value-relevant regimes |
| 3 | Real-time pricing + data sharing credits | Task-hour dynamic pricing + rebate/credit primitives for shared data | Makes deployment and contribution economically rational |
| 4 | Fleet coordination by economics | Assignment/routing/scheduling uses marginal value and risk | Improves blended fleet ROI |
| 5 | Objective programmability | Customer objective contracts compiled into scalarization profiles | Honest, programmable contracts across verticals |
| 6 | Inferential training economics | Deployment-time adaptation spend admitted only when expected gain > cost/risk | Prevents compute burn and pricing distortion |
| 7 | Securitization + insurance pricing | Risk-adjusted productivity/reliability curves | Enables financing and risk transfer products |
| 8 | Automated GTM + leasing + cross-customer coordination | Transfer playbooks + lease-aware shared-capacity optimization | Faster rollout, higher utilization |
| 9 | Lifecycle fleet management | Productivity decay as repair/replacement trigger | Better uptime and lifecycle economics |

## Deployment-Time Inferential Training (Planned)
As inference compute share increases, deployment-time adaptation must be economically bounded.

Target behavior:
- Inferential updates are admitted only when expected value exceeds cost/risk threshold.
- Objective profile and contract constraints define acceptable adaptation directions.
- Economic telemetry decides whether to spend compute on adaptation, data collection, or no-op.

## Pricing and Data-Sharing Contract (Concrete)
Planned deployment contract primitives:
- `task_hour_price_tick`: real-time estimate of value-backed task-hour price.
- `constraint_adjustment`: discount when constraints or uncertainty degrade trust.
- `data_share_credit`: explicit credit tied to measured marginal frontier gain.
- `net_customer_rate`: tick minus credits plus any risk/insurance adjustment.

This is designed to avoid dishonest pricing claims from metric gaming or constraint violations.

## Intended Architectural Contracts
The stack is moving toward these first-class artifacts:
- `ObjectiveTensor`: portable multi-objective representation that remains intact until explicit compile/scalarization.
- `ObjectiveCompiler`: explicit scalarization by contract/profile, never implicit upstream collapse.
- `ConstraintSet`: VLA + geometry manifold constraints used to condition generation/simulation.
- `EconTensor`: accounting representation coupled to objective outcomes and constraints.
- `PricingSentinel`: high-frequency pricing stream with auditable aggregated ledger writes.
- `Regal* nodes`: policy/governance gates for objective integrity, reward safety, plausibility, pricing truth, and data value.

## Existing Build and Verification Commands
```bash
# Install
pip install -e .

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

## Frozen and Additive Zones
Phase B is frozen. Do not modify:
- `src/world_model/`
- `checkpoints/stable_world_model.pt`
- Trust net, `w_econ` lattice objective, lambda controller equations
- `src/controllers/synthetic_weight_controller.py` core logic

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
Build a robotics platform where policy improvement, simulation generation, customer objective programming, and pricing are all governed by the same economic truth at deployment time.
