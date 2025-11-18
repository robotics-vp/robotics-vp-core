============================================
PHASE_G_PIPELINE_DIAGRAMS.md
============================================
1. Global Pipeline Diagram
Stage 1 → Stage 2 → Stage 3 → Stage 4 → Stage 5
   ↓          ↓         ↓         ↓         ↓
Datapacks → Semantics → Episodes → Sims → Real Logs
        ↓        ↓         ↓         ↓         ↓
        └──────────── Ontology Spine ─────────┘
                       ↓
                Phase G Policies
                       ↓
          Sampler / Curriculum / Orchestrator
                       ↓
                  RL + Backends

2. Policy Invocation Diagram

Example: DataValuationPolicy in sampling loop:

Descriptor
   ↓
Build valuation features
   ↓
DataValuationPolicy.score()
   ↓
Weight adjustment
   ↓
Sampler strategy

3. Orchestrator Advisory Flow
SemanticSnapshot ← aggregator
   ↓
OrchestratorPolicy
   ↓
OrchestratorAdvisory (JSONL)
   ↓
Sampler / Curriculum (flag-gated)

4. Meta-Advisor Flow
MetaTransformerSlice
   ↓
MetaAdvisorPolicy.propose()
   ↓
hint vector for sampler/curriculum (advisory)

5. Vision Policy Flow
Physics Backend → VisionFrame → VisionEncoderPolicy → VisionLatent → PolicyObservation

6. Dataset Logging Flow
Episode/Event → FeatureBuilders → Policy Evaluators → JSONL Logs
