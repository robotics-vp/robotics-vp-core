# Phase H System Integration Report

**Date:** 2025-01-23
**Status:** Integration Complete
**Version:** 1.0

---

## Executive Summary

This document describes the complete Phase H system integration for the video-to-policy economics model. Phase H introduces **economic learner-driven skill portfolio management** with **advisory-only signals** that guide sampling, orchestration, and vision conditioning without mutating reward or economic controllers.

All components are:
- **Deterministic**: Same inputs → same outputs
- **Bounded**: Advisory changes ≤ ±20%
- **Flag-gated**: `enable_phase_h_advisories=True` to activate
- **JSON-safe**: All artifacts serializable
- **Advisory-only**: No reward/econ mutations

---

## 1. System Architecture

### Component Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     Phase H Integration Layer                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐  │
│  │ TFDSession   │      │ SIMA-2       │      │ Economic     │  │
│  │ (streaming)  │      │ Tags         │      │ Learner      │  │
│  └──────┬───────┘      └──────┬───────┘      └──────┬───────┘  │
│         │                     │                      │           │
│         ▼                     ▼                      ▼           │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │          ConditionVector (Fusion Point)                   │  │
│  │  - TFD fields (skill_mode, target_mpl_uplift, etc.)      │  │
│  │  - SIMA-2 fields (ood_risk_level, recovery_priority)     │  │
│  │  - Phase H fields (exploration_uplift, skill_roi_est.)   │  │
│  └──────┬───────────────────────────────────────────────────┘  │
│         │                                                        │
│         ├──────────────┬─────────────────┬──────────────────┐  │
│         ▼              ▼                 ▼                  ▼  │
│  ┌──────────┐   ┌──────────┐    ┌───────────┐    ┌──────────┐│
│  │Conditioned│   │Sampler   │    │Orchestrator│   │Policy    ││
│  │Vision     │   │Weights   │    │Routing    │    │Head      ││
│  │Adapter    │   │(±20%)    │    │(±20%)     │    │          ││
│  └──────┬────┘   └──────────┘    └───────────┘    └────┬─────┘│
│         │                                                │       │
│         ▼                                                ▼       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │             PolicyObservation → TrunkNet                  │  │
│  │             (FiLM fusion + conditioned features)          │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
                               │
                               ▼
                     ┌──────────────────┐
                     │ EpisodeLogger     │
                     │ (TFD + Phase H    │
                     │  metadata)        │
                     └──────────────────┘
                               │
                               ▼
                     ┌──────────────────┐
                     │ Economic Learner  │
                     │ (measure returns, │
                     │  compute ROI)     │
                     └──────────────────┘
                               │
                               ▼
                     ┌──────────────────┐
                     │ Phase H Advisory  │
                     │ (exploration      │
                     │  priorities,      │
                     │  routing)         │
                     └──────────────────┘
                               │
                  ┌────────────┼────────────┐
                  ▼            ▼            ▼
            Sampler    Orchestrator  ConditionVector
         (Advisory ±20%) (Advisory ±20%) (Phase H fields)
```

---

## 2. Dataflow Mapping

### End-to-End Flow

```
[1] TFD Text Input
      ↓
    TFDSession.process_streaming_text()
      ↓
    TFDInstruction (canonical, JSON-safe)
      ↓
    ConditionVectorBuilder.build()
      ├─ TFD fields: skill_mode, target_mpl_uplift, energy_budget_wh
      ├─ SIMA-2 fields: ood_risk_level, recovery_priority
      └─ Phase H fields: exploration_uplift, skill_roi_estimate
      ↓
    ConditionVector (frozen dataclass)
      ↓
[2] ConditionedVisionAdapter.forward()
      ├─ FiLM modulation (gamma, beta from ConditionVector)
      ├─ BiFPN fusion (condition-weighted)
      └─ Risk/Affordance maps (condition-scaled)
      ↓
    Conditioned Features (z_v, modulated, fused, risk_map, affordance_map)
      ↓
[3] PolicyObservationBuilder.build_policy_features()
      ├─ Merge conditioned features
      └─ Build PolicyObservation
      ↓
[4] TrunkNet.forward()
      ├─ Extract vision (conditioned or base)
      ├─ Extract state
      ├─ Extract condition vector
      ├─ Apply FiLM modulation
      └─ Fuse features
      ↓
    Trunk Features (ready for policy head)
      ↓
[5] PolicyHead (PPO/SAC/etc.)
      ↓
    Action Distribution
      ↓
[6] Environment Step
      ↓
    Reward + Next Observation
      ↓
[7] EpisodeLogger.log_phase_h_cycle()
      ├─ TFD metadata
      ├─ Phase H cycle summary
      └─ Skill ROI estimates
      ↓
[8] EconomicLearner.run_cycle()
      ├─ Measure returns (ΔMP, ΔSuccess, ΔEnergy)
      ├─ Compute ROI per skill
      ├─ Reallocate budgets
      └─ Update skill statuses
      ↓
    Save artifacts (SkillMarketState, ExplorationBudget, SkillReturns)
      ↓
[9] PhaseHCycleOrchestrator.run_cycle_once()
      ├─ Load Phase H artifacts
      ├─ Create PhaseHAdvisory
      ├─ Push to Sampler (apply_sampler_advisory, ±20%)
      ├─ Push to Orchestrator (apply_orchestrator_advisory, ±20%)
      └─ Push to ConditionVector (build_phase_h_condition_fields)
      ↓
    [Loop back to [1]]
```

---

## 3. Advisory Boundaries

All Phase H advisory changes are **bounded** to prevent instability:

| Component | Boundary | Mechanism |
|-----------|----------|-----------|
| **Skill Multipliers** | [0.8, 1.2] | `MIN_MULTIPLIER`, `MAX_MULTIPLIER` in `advisory_integration.py` |
| **Sampler Weights** | ±20% | `MAX_ROUTING_DELTA = 0.20` |
| **Orchestrator Routing** | ±20% | Safety emphasis, strategy overrides bounded |
| **FiLM Gamma** | [0.1, 10.0] | `min_scale`, `max_scale` in `ConditionedVisionAdapter` |

**Enforcement:**
- All multipliers clamped: `np.clip(value, MIN, MAX)`
- Delta checks: `abs(new - old) / old <= MAX_ROUTING_DELTA`
- Deterministic: Same advisory → same outputs

---

## 4. Invariants

### Vision Invariant
- **z_v (base representation) unchanged by conditioning**
  - `ConditionedVisionAdapter.forward()` always returns `z_v` from base encoder
  - Conditioned features stored separately (`features_modulated`, `fused_features`)
  - TrunkNet can use either conditioned or base features

### Reward/Econ Invariant
- **No mutations to reward signals or economic controllers**
  - Phase H operates purely advisory
  - Sampler weights adjusted, but reward math untouched
  - EconomicLearner reads ontology, never writes to reward

### Determinism Invariant
- **Same inputs → same outputs**
  - All hashing deterministic (SHA256 for strings)
  - Sorted dict keys for consistent ordering
  - Fixed random seeds where needed

### JSON Safety Invariant
- **All artifacts serializable to JSON**
  - `to_json_safe()` utility for recursive conversion
  - No non-serializable types (functions, objects with state)
  - All timestamps ISO 8601 format

---

## 5. Integration Points

### A. ConditionVectorBuilder

**Location:** `src/observation/condition_vector_builder.py`

**New Parameters:**
- `phase_h_advisory: Optional[PhaseHAdvisory]`
- `enable_phase_h_advisories: bool = False`

**Behavior:**
```python
if enable_phase_h_advisories and phase_h_advisory is not None:
    from src.phase_h.advisory_integration import build_phase_h_condition_fields
    phase_h_fields = build_phase_h_condition_fields(
        phase_h_advisory,
        skill_id=episode_metadata.get("skill_id"),
        enable_phase_h_advisories=True,
    )
    exploration_uplift = phase_h_fields.get("exploration_uplift")
    skill_roi_estimate = phase_h_fields.get("skill_roi_estimate")
```

**Fields Added to ConditionVector:**
- `exploration_uplift: Optional[float]`
- `skill_roi_estimate: Optional[float]`

---

### B. Sampler Integration

**Location:** `src/policies/sampler_weights.py` (indirect via `advisory_integration.py`)

**Function:** `apply_sampler_advisory()`

**Behavior:**
```python
def apply_sampler_advisory(
    base_weights: Dict[str, float],
    advisory: PhaseHAdvisory,
    enable_phase_h_advisories: bool = False,
) -> Dict[str, float]:
    if not enable_phase_h_advisories:
        return base_weights

    # Apply global uplift based on exploration priorities
    avg_priority = sum(advisory.exploration_priorities.values()) / len(...)
    global_mult = 0.9 + (avg_priority * 0.2)  # [0.9, 1.1]

    adjusted_weights = {}
    for ep_key, base_weight in base_weights.items():
        adjusted = base_weight * global_mult

        # Enforce ±20% boundary
        max_allowed = base_weight * 1.2
        min_allowed = base_weight * 0.8
        adjusted = max(min_allowed, min(max_allowed, adjusted))

        adjusted_weights[ep_key] = adjusted

    return adjusted_weights
```

---

### C. Orchestrator Integration

**Location:** `src/policies/orchestrator_policy.py` (indirect via `advisory_integration.py`)

**Function:** `apply_orchestrator_advisory()`

**Behavior:**
```python
def apply_orchestrator_advisory(
    base_advisory: OrchestratorAdvisory,
    phase_h_advisory: PhaseHAdvisory,
    enable_phase_h_advisories: bool = False,
) -> OrchestratorAdvisory:
    if not enable_phase_h_advisories:
        return base_advisory

    routing = phase_h_advisory.routing_advisories

    # Blend safety_emphasis (bounded ±20%)
    base_safety = base_advisory.safety_emphasis
    suggested_safety = routing["safety_emphasis"]

    delta = suggested_safety - base_safety
    clamped_delta = max(-0.20, min(0.20, delta))  # ±20%
    adjusted_safety = base_safety + clamped_delta

    base_advisory.safety_emphasis = adjusted_safety
    base_advisory.metadata["phase_h_routing"] = routing

    return base_advisory
```

---

### D. ConditionedVisionAdapter Integration

**Location:** `src/vision/conditioned_adapter.py`

**Integration Point:** PolicyObservationBuilder

**Usage:**
```python
from src.vision.conditioned_adapter import ConditionedVisionAdapter

# In PolicyObservationBuilder:
if enable_conditioned_vision and condition_vector:
    adapter = ConditionedVisionAdapter(config=vision_config)
    conditioned = adapter.forward(frame, condition_vector)

    features["conditioned_vision_z_v"] = conditioned["z_v"]
    features["conditioned_vision_modulated"] = conditioned["features_modulated"]
    features["conditioned_vision_fused"] = conditioned["fused_features"]
    features["conditioned_vision_risk_map"] = conditioned["risk_map"]
    features["conditioned_vision_affordance_map"] = conditioned["affordance_map"]
```

**TrunkNet Handling:**
```python
# In TrunkNet._extract_vision():
if "conditioned_vision_fused" in obs:
    conditioned_fused = obs["conditioned_vision_fused"]
    return _tensor_from_iterable(flatten_pyramid(conditioned_fused), device, self.vision_dim)
```

---

### E. Phase H Cycle Controller

**Location:** `src/phase_h/controller.py`

**Class:** `PhaseHCycleOrchestrator`

**Methods:**
- `run_cycle_once(episode_count)`: Run Phase H cycle every N episodes
- `get_current_advisory()`: Get current PhaseHAdvisory
- `should_run_cycle(episode_count)`: Check if it's time for cycle

**Cycle Flow:**
```python
advisory = load_phase_h_advisory(ontology_root)

summary = {
    "cycle_count": cycle_count,
    "episode_count": episode_count,
    "skill_multipliers": advisory.skill_multipliers,
    "routing_advisories": advisory.routing_advisories,
    "exploration_priorities": advisory.exploration_priorities,
}

save_cycle_summary(summary)
```

---

## 6. Smoke Tests

### Phase H Advisory Integration Tests

**File:** `tests/smoke_tests/test_phase_h_advisory_integration.py`

**Tests Passed (9/9):**
1. ✅ `test_advisory_multipliers_bounded()` - Multipliers ∈ [0.8, 1.2]
2. ✅ `test_sampler_advisory_bounded()` - Sampler changes ≤ ±20%
3. ✅ `test_sampler_advisory_disabled_by_default()` - Flag=False → no change
4. ✅ `test_orchestrator_advisory_bounded()` - Orchestrator changes ≤ ±20%
5. ✅ `test_orchestrator_advisory_disabled_by_default()` - Flag=False → no change
6. ✅ `test_condition_fields_only_when_enabled()` - Fields only when flag=True
7. ✅ `test_determinism()` - Identical inputs → identical outputs
8. ✅ `test_json_safe_export()` - All artifacts JSON-serializable
9. ✅ `test_load_phase_h_advisory()` - Load from ontology artifacts

**Run Command:**
```bash
cd "/Users/amarmurray/robotics v-p economics model"
PYTHONPATH=. python3 tests/smoke_tests/test_phase_h_advisory_integration.py
```

**Output:**
```
=== Phase H Advisory Integration Smoke Tests ===

✓ Advisory multipliers bounded to [0.8, 1.2]
✓ Sampler advisory changes bounded to ±20%
✓ Sampler advisory disabled by default (flag=False)
✓ Orchestrator advisory changes bounded to ±20%
✓ Orchestrator advisory disabled by default (flag=False)
✓ ConditionVector fields only present when flag enabled
✓ Advisory computation is deterministic
✓ Advisory is JSON-safe
✓ Phase H advisory loads from ontology

=== All Tests Passed ===
```

---

## 7. CLI Usage

### Run Phase H Controller

**Command:**
```bash
python3 run_phase_h_controller.py \
  --ontology-root data/ontology \
  --episodes-per-cycle 1000 \
  --enable-phase-h \
  --total-episodes 5000 \
  --total-budget 10000.0
```

**Output:**
- Phase H artifacts saved to: `data/ontology/phase_h/`
  - `skill_market_state.json`
  - `exploration_budget.json`
  - `skill_returns.json`
- Cycle logs saved to: `logs/phase_h/`
  - `cycle_summaries.jsonl`

**Example Output:**
```
=== Phase H Cycle Controller ===

Ontology Root: data/ontology
Episodes Per Cycle: 1000
Enable Phase H: True
Total Episodes: 5000
Log Directory: logs/phase_h

=== Starting Phase H Cycle Loop ===

[Episode 1000] Economic Learner Cycle:
  - Total Budget: $10000.00
  - Skills: 2
  - ROI by Skill: {'dishwashing_precision': 15.3, 'drawer_open_v2': 42.7}
  - Artifacts saved to data/ontology/phase_h

[Episode 1000] Phase H Cycle 0:
  - Timestamp: 2025-01-23T10:30:00Z
  - Skill Multipliers: {'dishwashing_precision': 1.1, 'drawer_open_v2': 1.2}
  - Routing Advisories:
    - frontier_emphasis: 0.6
    - safety_emphasis: 0.3
    - efficiency_emphasis: 0.4
    - skill_mode_suggestion: frontier_exploration

[Episode 2000] Economic Learner Cycle:
  ...

=== Phase H Controller Summary ===
{
  "cycle_count": 5,
  "last_cycle_episode": 5000,
  "cycle_period_episodes": 1000,
  "enable_phase_h": true
}

Phase H artifacts saved to: data/ontology/phase_h/
Cycle logs saved to: logs/phase_h/
```

---

## 8. Files Modified/Created

### Created Files

1. **`src/phase_h/advisory_integration.py`** (347 lines)
   - `PhaseHAdvisory` class
   - `load_skill_market_state()`, `load_exploration_budget()`, `load_skill_returns()`
   - `apply_sampler_advisory()`, `apply_orchestrator_advisory()`
   - `build_phase_h_condition_fields()`, `load_phase_h_advisory()`

2. **`src/phase_h/controller.py`** (95 lines)
   - `PhaseHCycleOrchestrator` class
   - `run_cycle_once()`, `get_current_advisory()`, `should_run_cycle()`

3. **`run_phase_h_controller.py`** (165 lines)
   - CLI for running Phase H controller
   - Simulates training loop with economic learner integration

4. **`tests/smoke_tests/test_phase_h_advisory_integration.py`** (291 lines)
   - 9 smoke tests for Phase H advisory integration
   - All tests passing

5. **`PHASE_H_INTEGRATION_COMPLETE.md`** (summary document)

6. **`reports/PHASE_H_SYSTEM_INTEGRATION.md`** (this document)

### Modified Files

1. **`src/observation/condition_vector.py`**
   - Added `exploration_uplift: Optional[float]` field
   - Added `skill_roi_estimate: Optional[float]` field
   - Updated `to_dict()` to export Phase H fields
   - Updated `from_dict()` to import Phase H fields

2. **`src/observation/condition_vector_builder.py`**
   - Added `phase_h_advisory: Optional[Any]` parameter to `build()`
   - Added `enable_phase_h_advisories: bool` flag
   - Integrated `build_phase_h_condition_fields()` call
   - Pass Phase H fields to ConditionVector constructor

---

## 9. Configuration

### Enable Phase H Integration

**Config File:** `configs/phase_h.yaml`

```yaml
phase_h:
  enabled: true
  cycle_period_episodes: 1000
  ontology_root: data/ontology
  log_dir: logs/phase_h

  # Advisory boundaries
  min_skill_multiplier: 0.8
  max_skill_multiplier: 1.2
  max_routing_delta: 0.20

  # Economic learner
  total_exploration_budget: 10000.0
  price_per_unit: 0.30
  hours_deployed: 1000
  energy_price_per_kwh: 0.12

  # Data valuation
  tier0_threshold: 0.5  # Redundant
  tier1_threshold: 3.0  # Context-novel
  # Above tier1: Frontier
```

**Usage in Code:**
```python
from src.utils.config import load_config

config = load_config("configs/phase_h.yaml")

orchestrator = PhaseHCycleOrchestrator(config["phase_h"])
```

---

## 10. Deployment Checklist

### Pre-Deployment

- [x] All smoke tests passing
- [x] Advisory boundaries verified
- [x] Determinism verified
- [x] JSON safety verified
- [x] Flag-gating verified (disabled by default)

### Deployment Steps

1. **Enable Phase H in Config:**
   ```yaml
   phase_h:
     enabled: true
   ```

2. **Initialize Ontology Artifacts:**
   ```bash
   python3 run_phase_h_controller.py \
     --ontology-root data/ontology \
     --episodes-per-cycle 1000 \
     --enable-phase-h \
     --total-episodes 1000
   ```

3. **Verify Artifacts Created:**
   ```bash
   ls data/ontology/phase_h/
   # Expected:
   #   skill_market_state.json
   #   exploration_budget.json
   #   skill_returns.json
   ```

4. **Integrate into Training Loop:**
   ```python
   from src.phase_h.controller import PhaseHCycleOrchestrator
   from src.phase_h.advisory_integration import apply_sampler_advisory

   orchestrator = PhaseHCycleOrchestrator(config)

   for episode in range(total_episodes):
       # ... training code ...

       # Phase H cycle
       if orchestrator.should_run_cycle(episode):
           advisory = orchestrator.get_current_advisory()
           if advisory:
               # Apply to sampler
               weights = apply_sampler_advisory(base_weights, advisory, enable_phase_h_advisories=True)

               # Apply to orchestrator
               orch_advisory = apply_orchestrator_advisory(base_orch, advisory, enable_phase_h_advisories=True)

               # Pass to condition vector builder
               condition_vector = builder.build(
                   ...,
                   phase_h_advisory=advisory,
                   enable_phase_h_advisories=True,
               )
   ```

5. **Monitor Logs:**
   ```bash
   tail -f logs/phase_h/cycle_summaries.jsonl
   ```

### Post-Deployment Monitoring

- Monitor skill multipliers: Should stay in [0.8, 1.2]
- Monitor routing deltas: Should stay ≤ ±20%
- Monitor ROI trends: Should increase over time
- Monitor exploration budgets: Should reallocate based on ROI

---

## 11. Troubleshooting

### Issue: Phase H artifacts not found

**Symptom:** `load_phase_h_advisory()` returns `None`

**Fix:**
```bash
# Initialize artifacts
python3 run_phase_h_controller.py \
  --ontology-root data/ontology \
  --episodes-per-cycle 100 \
  --enable-phase-h \
  --total-episodes 100
```

### Issue: Advisory changes exceed boundaries

**Symptom:** Sampler weights change by >20%

**Debug:**
```python
from src.phase_h.advisory_integration import apply_sampler_advisory

weights_before = {...}
weights_after = apply_sampler_advisory(weights_before, advisory, enable_phase_h_advisories=True)

for key in weights_before:
    delta_pct = abs(weights_after[key] - weights_before[key]) / weights_before[key]
    assert delta_pct <= 0.20, f"Delta {delta_pct:.2%} exceeds 20% for {key}"
```

### Issue: ConditionVector fields missing

**Symptom:** `exploration_uplift` is `None`

**Fix:**
```python
# Ensure flag enabled
condition_vector = builder.build(
    ...,
    phase_h_advisory=advisory,
    enable_phase_h_advisories=True,  # Must be True
)

# Verify advisory loaded
assert advisory is not None, "Phase H advisory not loaded"
```

---

## 12. Future Work

### Phase H+: Extended Features

1. **Multi-Agent Skill Sharing:**
   - Share skill ROI across robot fleet
   - Dynamic budget reallocation across agents

2. **Causal Skill Dependencies:**
   - Model dependencies between skills (e.g., "grasp" → "place")
   - Prioritize prerequisite skills in exploration budget

3. **Adaptive Cycle Periods:**
   - Dynamic cycle period based on learning velocity
   - Early phases: short cycles (100 eps), mature phases: long cycles (5000 eps)

4. **Skill Deprecation Automation:**
   - Auto-deprecate skills with ROI < -50% for >3 cycles
   - Archive deprecated skills for analysis

5. **Tier-Aware Data Pricing:**
   - Integrate tier 0/1/2 classification from data valuation
   - Adjust budgets based on tier distribution

---

## 13. Conclusion

Phase H system integration is **complete** and **production-ready**. All components are:

- ✅ **Tested** (9/9 smoke tests passing)
- ✅ **Bounded** (±20% advisory changes, [0.8, 1.2] multipliers)
- ✅ **Deterministic** (same inputs → same outputs)
- ✅ **JSON-safe** (all artifacts serializable)
- ✅ **Flag-gated** (disabled by default, explicit opt-in)
- ✅ **Advisory-only** (no reward/econ mutations)
- ✅ **Documented** (this report + inline comments)

**Key Integration Points:**
1. ConditionVector (Phase H fields added)
2. Sampler (advisory weight modulation)
3. Orchestrator (advisory routing)
4. ConditionedVision (FiLM modulation from ConditionVector)
5. Phase H Controller (cycle orchestration)

**Deployment:** Ready for staging/production with `enable_phase_h=true` flag.

**Contact:** See IMPLEMENTATION_HANDOFF.md for team contacts and escalation paths.

---

**End of Report**
