# Fast/Slow Economic Sync Bridge

This document describes a hybrid strategy for millisecond control loops that must
remain economically auditable.

## Why

The runtime has a classic split:

- **Fast loop** (≈1ms): policy/action inference
- **Slow loop** (100-500ms+): ontology + economic valuation updates

Direct per-step reads from ontology/ledger can create avoidable latency spikes.

## Hybrid design

Implemented in `src/orchestrator/fast_slow_econ_bridge.py`.

1. **Geometric Shadow**
   - Slow loop publishes a `ConstraintShadow` (hard bounds per action dimension).
   - Fast loop projects candidate actions locally via `ConstraintShadow.project`.

2. **Hierarchical Ledgering (L1/L2)**
   - Fast loop appends `EconTensorSample` to `TransientLedger` (L1 circular buffer).
   - Orchestrator periodically calls `settle()` to produce batched `SettlementRecord`s
     for async L2/global persistence.
   - `deploy_gate()` can block execution when L1/L2 drift exceeds a threshold.

3. **Predictive Ontology Masking**
   - `OntologyMask` stores zone-scoped shadows (e.g., workcell zones).
   - Controller resolves by zone first, then falls back to global shadow.

## Entry point

`FastSlowEconBridge` combines all three mechanisms and exposes:

- `update_shadow(...)`
- `project_action(...)`
- `ingest_tick(...)`
- `settle_to_l2(...)`
- `deploy_gate(...)`

This gives a single interface for zero-network hot-path control plus explicit
safety/audit gating.
