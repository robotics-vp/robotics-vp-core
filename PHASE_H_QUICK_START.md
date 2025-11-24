# Phase H Quick Start Guide

**TL;DR:** Phase H adds economic learner-driven skill management with advisory signals (bounded ±20%, flag-gated, deterministic).

---

## 1. Enable Phase H (5 minutes)

### Step 1: Run Initial Setup

```bash
cd "/Users/amarmurray/robotics v-p economics model"

# Create ontology directory
mkdir -p data/ontology/phase_h

# Run controller to initialize artifacts
PYTHONPATH=. python3 run_phase_h_controller.py \
  --ontology-root data/ontology \
  --episodes-per-cycle 1000 \
  --enable-phase-h \
  --total-episodes 1000 \
  --total-budget 10000.0
```

### Step 2: Verify Artifacts Created

```bash
ls data/ontology/phase_h/
# Expected:
#   exploration_budget.json
#   skill_market_state.json
#   skill_returns.json

ls logs/phase_h/
# Expected:
#   cycle_summaries.jsonl
```

### Step 3: Integrate into Training Loop

```python
from pathlib import Path
from src.phase_h.controller import PhaseHCycleOrchestrator
from src.phase_h.advisory_integration import (
    load_phase_h_advisory,
    apply_sampler_advisory,
    apply_orchestrator_advisory,
)
from src.observation.condition_vector_builder import ConditionVectorBuilder

# Initialize Phase H controller
config = {
    "ontology_root": "data/ontology",
    "cycle_period_episodes": 1000,
    "enable_phase_h": True,
    "log_dir": "logs/phase_h",
}
orchestrator = PhaseHCycleOrchestrator(config)

# In training loop:
for episode in range(total_episodes):
    # ... existing training code ...

    # Check if it's time for Phase H cycle
    if orchestrator.should_run_cycle(episode):
        # Run cycle
        summary = orchestrator.run_cycle_once(episode)

        # Get current advisory
        advisory = orchestrator.get_current_advisory()

        if advisory:
            # Apply to sampler
            base_weights = {...}  # Your existing sampler weights
            adjusted_weights = apply_sampler_advisory(
                base_weights,
                advisory,
                enable_phase_h_advisories=True,
            )

            # Apply to orchestrator
            base_orch_advisory = {...}  # Your existing orchestrator advisory
            adjusted_orch = apply_orchestrator_advisory(
                base_orch_advisory,
                advisory,
                enable_phase_h_advisories=True,
            )

            # Pass to condition vector builder
            builder = ConditionVectorBuilder()
            condition_vector = builder.build(
                episode_config=...,
                econ_state=...,
                curriculum_phase=...,
                sima2_trust=...,
                phase_h_advisory=advisory,  # NEW
                enable_phase_h_advisories=True,  # NEW
            )

    # ... rest of training code ...
```

---

## 2. Verify Integration (2 minutes)

### Run Smoke Tests

```bash
PYTHONPATH=. python3 tests/smoke_tests/test_phase_h_advisory_integration.py
```

**Expected Output:**
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

## 3. Monitor Phase H (Ongoing)

### Check Cycle Logs

```bash
tail -f logs/phase_h/cycle_summaries.jsonl
```

### Inspect Skill Market State

```bash
cat data/ontology/phase_h/skill_market_state.json | python3 -m json.tool
```

**Example Output:**
```json
{
  "timestamp": "2025-11-23T23:53:54Z",
  "total_budget_usd": 5000.0,
  "allocated_usd": 4312.5,
  "skills": {
    "dishwashing_precision": {
      "skill_id": "dishwashing_precision",
      "mpl_current": 55.0,
      "mpl_target": 60.0,
      "success_rate": 0.75,
      "status": "training"
    },
    ...
  }
}
```

### Monitor Skill ROI

```bash
cat data/ontology/phase_h/skill_returns.json | python3 -c "
import json, sys
data = json.load(sys.stdin)
for r in data['returns']:
    print(f\"{r['skill_id']}: ROI {r['roi_pct']:.1f}%\")
"
```

**Example Output:**
```
dishwashing_precision: ROI 200.0%
drawer_open_v2: ROI 900.0%
```

---

## 4. Configuration

### Create Config File

**File:** `configs/phase_h.yaml`

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
```

### Load in Code

```python
import yaml

with open("configs/phase_h.yaml") as f:
    config = yaml.safe_load(f)

orchestrator = PhaseHCycleOrchestrator(config["phase_h"])
```

---

## 5. Troubleshooting

### Issue: Phase H artifacts not found

**Symptom:** `load_phase_h_advisory()` returns `None`

**Fix:**
```bash
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
weights_before = {...}
weights_after = apply_sampler_advisory(weights_before, advisory, enable_phase_h_advisories=True)

for key in weights_before:
    delta_pct = abs(weights_after[key] - weights_before[key]) / weights_before[key]
    if delta_pct > 0.20:
        print(f"WARNING: {key} delta {delta_pct:.2%} exceeds 20%")
```

### Issue: ConditionVector fields missing

**Symptom:** `exploration_uplift` is `None`

**Fix:**
```python
# Ensure flag enabled
condition_vector = builder.build(
    ...,
    phase_h_advisory=advisory,
    enable_phase_h_advisories=True,  # MUST be True
)
```

---

## 6. Common Use Cases

### Use Case 1: Adjust Sampling Based on ROI

```python
advisory = load_phase_h_advisory(Path("data/ontology"))

# Get skill multipliers
multipliers = advisory.skill_multipliers
# {'skill_a': 1.2, 'skill_b': 0.8}

# Apply to sampler
weights = apply_sampler_advisory(base_weights, advisory, enable_phase_h_advisories=True)
```

### Use Case 2: Route to Safety vs Frontier

```python
routing = advisory.routing_advisories

if routing["safety_emphasis"] > 0.7:
    # Emphasize safety mode
    skill_mode = "safety_critical"
elif routing["frontier_emphasis"] > 0.6:
    # Emphasize frontier exploration
    skill_mode = "frontier_exploration"
else:
    # Balanced
    skill_mode = "efficiency_throughput"
```

### Use Case 3: Log Phase H Metadata

```python
logger.log_phase_h_cycle(
    learner_suggested_skill_mode=routing["skill_mode_suggestion"],
    exploration_budget_at_episode=budget_state,
    learner_roi_estimate_per_skill={s.skill_id: s.roi for s in returns},
    tfd_session_state=tfd_summary,
)
```

---

## 7. Key Files Reference

| File | Purpose |
|------|---------|
| `src/phase_h/advisory_integration.py` | Advisory layer, sampler/orchestrator integration |
| `src/phase_h/controller.py` | Phase H cycle orchestrator |
| `src/phase_h/economic_learner.py` | Economic learner (ROI, budget allocation) |
| `src/phase_h/models.py` | Data models (Skill, ExplorationBudget, SkillReturns) |
| `run_phase_h_controller.py` | CLI for running Phase H controller |
| `tests/smoke_tests/test_phase_h_advisory_integration.py` | Smoke tests |
| `reports/PHASE_H_SYSTEM_INTEGRATION.md` | Comprehensive integration report |

---

## 8. Decision Tree: When to Use Phase H?

```
Do you want to:
  ├─ Dynamically adjust sampling based on skill ROI?
  │   └─ YES → Enable Phase H, use apply_sampler_advisory()
  ├─ Route tasks to safety vs frontier based on skill quality?
  │   └─ YES → Enable Phase H, use apply_orchestrator_advisory()
  ├─ Track exploration budgets per skill?
  │   └─ YES → Enable Phase H, monitor exploration_budget.json
  ├─ Condition vision features based on economic signals?
  │   └─ YES → Enable Phase H, pass phase_h_advisory to ConditionVectorBuilder
  └─ None of the above
      └─ NO → Phase H not needed, keep disabled
```

---

## 9. Minimal Example (Copy-Paste Ready)

```python
from pathlib import Path
from src.phase_h.controller import PhaseHCycleOrchestrator
from src.phase_h.advisory_integration import load_phase_h_advisory, apply_sampler_advisory

# Setup
config = {"ontology_root": "data/ontology", "cycle_period_episodes": 1000, "enable_phase_h": True}
orchestrator = PhaseHCycleOrchestrator(config)

# Training loop
for episode in range(10000):
    # Your training code here
    pass

    # Phase H cycle (every 1000 episodes)
    if orchestrator.should_run_cycle(episode):
        summary = orchestrator.run_cycle_once(episode)
        advisory = orchestrator.get_current_advisory()

        if advisory:
            # Apply advisory to sampler
            base_weights = {"ep1": 1.0, "ep2": 0.5}
            adjusted_weights = apply_sampler_advisory(base_weights, advisory, enable_phase_h_advisories=True)
            print(f"[Episode {episode}] Adjusted weights: {adjusted_weights}")
```

---

## 10. Next Steps

### After Enabling Phase H

1. **Monitor logs:** `tail -f logs/phase_h/cycle_summaries.jsonl`
2. **Track ROI trends:** Plot skill ROI over time
3. **Validate boundaries:** Ensure multipliers ∈ [0.8, 1.2], routing ≤ ±20%
4. **Tune cycle period:** Adjust `cycle_period_episodes` based on learning velocity

### Optional Enhancements

- Integrate ConditionedVision for FiLM modulation
- Add skill dependency tracking
- Implement adaptive cycle periods
- Enable multi-agent skill sharing

---

**Questions?** See [PHASE_H_SYSTEM_INTEGRATION.md](reports/PHASE_H_SYSTEM_INTEGRATION.md) for full documentation.

**End of Quick Start**
