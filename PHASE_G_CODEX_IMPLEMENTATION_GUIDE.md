============================================
PHASE_G_CODEX_IMPLEMENTATION_GUIDE.md
============================================

This is the exact step-by-step coding order.

Step 1 — policies.yaml

Create baseline with all entries = "heuristic".

Step 2 — registry.py

Loads yaml, returns instantiated policy classes.

Step 3 — Create src/policies/interfaces.py

Define Protocol for all 9 policies.

Step 4 — Create heuristic implementations

One file per domain:

src/policies/data_valuation.py
src/policies/pricing.py
src/policies/safety_risk.py
src/policies/energy_cost.py
src/policies/episode_quality.py
src/policies/orchestrator_policy.py
src/policies/sampler_weights.py
src/policies/meta_advisor.py
src/policies/vision_encoder.py

Step 5 — Feature Builders

Create:

src/policies/features_data_valuation.py
src/policies/features_pricing.py
...

Step 6 — Integration

Replace inline logic in:

sampler

curriculum

orchestrator v2

semantic aggregator

analytics reports

recap dataset

train_sac_with_ontology_logging

Everything must call a policy instance.

Step 7 — Dataset Builders

Create:

scripts/build_policy_datasets.py


Outputs every policy’s dataset.

Step 8 — Smoke Tests

Add to run_all_smokes:

scripts/smoke_test_policy_registry.py
scripts/smoke_test_<policy>.py
scripts/smoke_test_policy_datasets.py

Step 9 — Integration Smokes

sampler + policy

curriculum + policy

orchestrator + policy

pricing report + policy

vision frame → latent
