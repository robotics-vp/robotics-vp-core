============================================
PHASE_G_SPEC.md
============================================
Phase G: Universal Policy Abstraction & Neural-Ready Surfaces
1. Teleology

Phase G completes a critical architectural transformation:

Every heuristic becomes a policy module with a stable interface, unified features, optional heuristic backend, and NN-ready scaffolding.

This turns your entire stack into a learnable economic engine, not a collection of hand-coded logic.

The architecture after Phase G:

      ┌─────────────────────────────────────┐
      │     Sovereign Layers (NEVER ML)     │
      │   • Economic Controller             │
      │   • Datapack Engine                 │
      └─────────────────────────────────────┘
                      ↓
      ┌─────────────────────────────────────┐
      │  Meta-Semantic Layer (Orchestrator) │
      └─────────────────────────────────────┘
                      ↓
      ┌─────────────────────────────────────┐
      │   Phase G Policy Abstraction Layer  │
      │    (Learnable Decision Surfaces)    │
      └─────────────────────────────────────┘
                      ↓
      ┌─────────────────────────────────────┐
      │ Sampler/Curriculum/Training/Backends│
      └─────────────────────────────────────┘
                      ↓
      ┌──────────────────────────────────────┐
      │  RL Policy + Physics Environment     │
      └──────────────────────────────────────┘


Phase G defines the contract that each of your future ML modules must satisfy.

2. Scope

Phase G covers nine policy classes:

Data Valuation Policy

Pricing Policy

Safety Risk Policy

Energy Cost Policy

Episode Quality / Anomaly Policy

Orchestrator Policy

Sampler Weight Policy

Meta-Advisor Policy (Meta-Transformer abstraction)

Vision Encoder Policy (VLA/DINO backbone abstraction)

These become the switching points where ML replaces heuristics later.

3. Core Requirements
3.1 Deterministic + JSON-safe

All policies must:

accept dict-like inputs

produce dict-like outputs

be JSON-safe

be deterministic under fixed seed

log features in the same shape they'll later be trained on

3.2 Config-driven policy selection

via config/policies.yaml

data_valuation: heuristic

pricing: heuristic

safety_risk: heuristic

energy_cost: heuristic

episode_quality: heuristic

sampler_weights: heuristic

orchestrator: heuristic

meta_advisor: heuristic

vision_encoder: stub


Later you can swap any to:

data_valuation: neural
...

3.3 Unified Feature Builders

Each policy has a centralized build_features() function.

3.4 Advisory-only

Policies output recommendations, not enforced commands.

3.5 No breaking changes

Phase G MUST NOT modify:

Reward math

SAC/PPO logic

Phase B economics

Datapack/Episode schemas

Semantic Orchestrator core logic (besides advisory hooks)

4. Policy Specs

For each policy:

4.1 Data Valuation Policy

Input:

datapack metadata

econ slice

semantic tags

recap scores

Output:

scalar valuation_score

metadata dict

4.2 Pricing Policy

Input:

task econ distribution

datapack value

semantic context

Output:

datapack price

robot-hour price

reasoning metadata

4.3 Safety Risk Policy

Input:

primitives

ontology updates

task graph risk nodes

Output:

categorical risk level

numeric damage estimate

4.4 Energy Cost Policy

Input: episode events
Output: estimated energy (Wh)

4.5 Episode Quality Policy

Input: rewards, variance, collisions
Output: quality_score, anomaly_score

4.6 Orchestrator Policy

Input: SemanticSnapshot
Output: OrchestratorAdvisory

4.7 Sampler Weight Policy

Input: descriptor batch
Output: per-descriptor weights

4.8 Meta-Advisor Policy

Input: MetaTransformerSlice
Output: MetaTransformerOutputs

4.9 Vision Encoder Policy

Input: VisionFrame
Output: VisionLatent

5. Dataset Specification

Every policy logs:

{
  "policy": "<name>",
  "features": {...},
  "target": {...},
  "timestamp": ...,
  "task_id": ...,
  "episode_id": ...
}

6. Integration Contracts

Each module (sampler, curriculum, orchestrator, analytics, VLA, etc.) must use policy instances created via:

from policies.registry import build_all_policies
policies = build_all_policies()


No direct heuristic calls allowed after Phase G.
