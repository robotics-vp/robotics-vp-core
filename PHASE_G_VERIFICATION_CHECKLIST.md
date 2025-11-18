============================================
PHASE_G_VERIFICATION_CHECKLIST.md
============================================
✔ Pre-Implementation

 policies.yaml created

 registry.py loads all policies

 all interfaces declared using Protocol

 every heuristic isolated into its own policy class

 feature builders exist

✔ Integration

 sampler uses SamplingWeightPolicy

 curriculum uses SamplingWeightPolicy

 orchestrator uses OrchestratorPolicy

 pricing report uses PricingPolicy

 reward engine uses Safety + Energy policies

 recap integrates through EpisodeQualityPolicy

✔ Determinism

 all policies deterministic given seed

 policies’ outputs JSON-safe

 dataset builder writes & reads back identically

✔ Dataset completeness

For each policy:

 logs features

 logs target

 logs metadata

 logs timestamp

✔ Smoke Tests

 per-policy smokes

 per-dataset smokes

 registry smokes

 integration smokes

✔ Completion

Phase G is complete when:

RL training unchanged

reward logic unchanged

sampler behavior unchanged unless flag-gated

ALL decisions go through policies
