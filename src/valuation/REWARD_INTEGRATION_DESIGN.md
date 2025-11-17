# RewardBuilder Integration Design

This document describes how `RewardBuilder` + objective presets integrate with the existing economics system, and maps out the path to objective-conditioned rewards.

**Status**: Scaffolding only. Current training behavior unchanged.

---

## 1. Compatibility Check: RewardBuilder vs Current Economics

### Current Economics (from CLAUDE.md)

The existing reward system computes:

```python
# Layer 1: Training Economics
r(t) = α · ΔMPᵣ(t) - β · e(t) - γ · |ŵᵣ(t)/wₕ - 1|
```

Where:
- `α` = productivity improvement weight
- `β` = error penalty
- `γ` = wage convergence weight
- `MPᵣ(t)` = robot marginal product (units/hour)
- `e(t)` = error rate
- `ŵᵣ(t)` = robot implied wage

### RewardBuilder Terms

`src/valuation/reward_builder.py` computes:

```python
reward_terms = {
    "r_mpl": mpl,                    # ✓ Matches MPᵣ (throughput)
    "r_error": -error_rate,          # ✓ Matches -e(t)
    "r_energy": -energy_cost,        # NEW: Energy efficiency term
    "r_safety": -1.0 if vase_broken, # NEW: Safety/fragility penalty
    "r_novelty": 0.0,                # PLACEHOLDER: Future exploration bonus
}

combined = Σ w[i] * r[i]  # Weighted combination
```

### Compatibility Analysis

**Compatible**:
- MPL term directly maps to throughput/productivity improvement
- Error rate term directly maps to precision/damage costs
- Safety term captures catastrophic failures (vase_broken)

**Extended**:
- Energy term is NEW but aligns with EP (energy productivity) goal
- Safety term is MORE EXPLICIT than current cₐ · e(t) (separates "routine errors" from "catastrophic failures")

**Missing from RewardBuilder**:
- Wage parity convergence: `|ŵᵣ(t)/wₕ - 1|`
- This is intentional: wage parity is a META constraint, not a per-step reward

**Verdict**: RewardBuilder is compatible but operates at a different level:
- **Current**: Step-level reward with wage convergence penalty
- **RewardBuilder**: Episode-level objective scoring for meta-optimization

---

## 2. default_objective_vector() Analysis

```python
def default_objective_vector():
    return [1.0, 0.2, 0.1, 0.05, 0.0]
    #      [MPL, error, energy, safety, novelty]
```

**Why this approximates current behavior**:
- `w_mpl = 1.0`: Primary focus on throughput (matches α ≫ β in early training)
- `w_error = 0.2`: Mild error penalty (matches moderate β)
- `w_energy = 0.1`: Low energy weight (not explicit in current reward)
- `w_safety = 0.05`: Small safety penalty (vase_break_cost indirectly penalizes)
- `w_novelty = 0.0`: No exploration bonus (matches current deterministic training)

**Backwards Compatibility**:
If we plug RewardBuilder into training loop with default_objective_vector(), the policy should:
1. Prioritize MPL improvement (same as current)
2. Penalize errors proportionally (same as current)
3. Add mild energy cost awareness (NEW but low weight)
4. Add explicit safety penalty for catastrophic failures (NEW but low weight)

This is **backwards-compatible** because the dominant term (MPL) matches, and new terms have low weights. The policy won't suddenly change behavior.

---

## 3. Integration Architecture: Objective-Conditioned Reward

### Where RewardBuilder Will Be Called

**Current Training Loop** (simplified):

```python
# train_sac.py or train_ppo.py
for episode in range(num_episodes):
    obs = env.reset()
    while not done:
        action = policy(obs)
        obs, raw_reward, done, info = env.step(action)

        # CURRENT: Fixed reward from env
        reward = raw_reward

        buffer.add(obs, action, reward, ...)

    # Log episode metrics
    log_episode_metrics(info)
```

**Future Objective-Conditioned Loop**:

```python
# train_sac_objective.py  <-- NEW FILE
from src.valuation.reward_builder import build_reward_terms, combine_reward
from src.config.objective_profile import ObjectiveVector

# Load objective preset
objective_vector = ObjectiveVector.from_preset(args.objective_preset).to_list()
econ_params = load_econ_params(args.config)

for episode in range(num_episodes):
    obs = env.reset()
    backend = make_backend(args.engine_type, env, env_name=args.env)

    while not done:
        action = policy(obs, objective_vector)  # Conditioned policy
        obs, raw_reward, done, info = backend.step(action)

        # TODO: Plug RewardBuilder here
        # Build reward terms from step info
        step_summary = extract_step_summary(info)  # Convert step info to summary
        reward_terms = build_reward_terms(step_summary, econ_params)
        shaped_reward = combine_reward(objective_vector, reward_terms)

        buffer.add(obs, action, shaped_reward, ...)

    # Episode-level summary for datapack
    episode_summary = backend.get_episode_info()
    episode_reward_terms = build_reward_terms(episode_summary, econ_params)
    episode_J = combine_reward(objective_vector, episode_reward_terms)

    # Log with objective profile
    log_objective_metrics(episode_summary, objective_vector, episode_J)
```

### Where to Find the Change Point

**File**: `src/rl/train_sac_objective.py` (to be created)

**Line to Change** (marked with TODO):

```python
# In train_sac_objective.py
def train_step(...):
    # ... policy forward, env step ...

    # === CURRENT BEHAVIOR (for testing) ===
    # reward = raw_reward  # Use environment's raw reward

    # === FUTURE BEHAVIOR (objective-conditioned) ===
    # TODO: Enable when ready to flip to objective-conditioned reward
    reward_terms = build_reward_terms(step_summary, econ_params)
    reward = combine_reward(objective_vector, reward_terms)
```

---

## 4. Data Flow: Objective + Econ + Energy → Reward Weights

### Current Flow (Fixed)

```
EconParams (static)
    ↓
compute_econ_reward(mpl, ep, error, alpha, beta, gamma)
    ↓
reward (scalar)
```

### Future Flow (Objective-Conditioned)

```
ObjectiveVector (from preset)     EconContext (market)     EnergyResponseNet (energy sensitivity)
         ↓                              ↓                              ↓
    [w_mpl, w_error, ...]        [wage_human, ...]           [energy_delta, ...]
                   \                    |                         /
                    \                   |                        /
                     \                  |                       /
                      ↓                 ↓                      ↓
                    EconObjectiveNet (learned mapping)
                              ↓
                    RewardWeights (α_mpl, α_error, ...)
                              ↓
                    combine_reward(weights, reward_terms)
                              ↓
                        reward (scalar)
```

### Where EconObjectiveNet Will Live

**Module**: `src/networks/econ_objective_net.py`

```python
class EconObjectiveNet(nn.Module):
    """
    Maps (ObjectiveVector, EconContext, EnergyResponse) → RewardWeights.

    Learned mapping that adapts reward shaping to economic context.
    """
    def forward(self, objective_vec, econ_context, energy_response):
        # Encode objective priorities
        obj_enc = self.obj_encoder(objective_vec)

        # Encode economic context
        econ_enc = self.econ_encoder(econ_context)

        # Encode energy sensitivity
        energy_enc = self.energy_encoder(energy_response)

        # Combine and output reward weights
        combined = torch.cat([obj_enc, econ_enc, energy_enc], dim=-1)
        reward_weights = self.mlp(combined)

        return reward_weights  # [α_mpl, α_ep, α_error, α_energy, α_safety]
```

---

## 5. Constraints Integration

### Offline Solver: solve_objective_surface_from_datapacks.py

This script evaluates objective presets under constraint bundles:

```python
# Constraint set example
constraint_set = {
    "min_wage_parity": 0.9,       # Must maintain 90% wage parity
    "max_error_rate": 0.05,       # Error rate must stay below 5%
    "max_energy_Wh": 20.0,        # Energy budget per task
}

# Evaluate each preset
for preset in ["throughput", "safety", "energy_saver"]:
    objective_vector = ObjectiveVector.from_preset(preset)

    # Score datapacks under this objective
    valid, utility = evaluate_datapacks_under_constraints(
        datapacks, objective_vector, constraint_set
    )

    # Log results
    results[preset] = {"valid": valid, "utility": utility}
```

### How Constraints Feed Into Training

**Current**: Constraints are implicit in reward weights (e.g., high β = low error)

**Future**: Constraints are explicit and enforceable:

```python
# In train_sac_objective.py
def apply_constraint_penalties(reward, summary, constraints):
    """Apply soft penalties for constraint violations."""
    penalty = 0.0

    if constraints.get("min_wage_parity"):
        if summary.wage_parity < constraints["min_wage_parity"]:
            penalty += (constraints["min_wage_parity"] - summary.wage_parity) * 10.0

    if constraints.get("max_error_rate"):
        if summary.error_rate_episode > constraints["max_error_rate"]:
            penalty += (summary.error_rate_episode - constraints["max_error_rate"]) * 10.0

    return reward - penalty
```

---

## 6. Migration Plan

### Phase 1: Schema and Logging (CURRENT - DONE)

- ✅ ObjectiveProfile in DataPackMeta
- ✅ RewardBuilder scaffolding
- ✅ Objective presets defined
- ✅ No training behavior changed

### Phase 2: Offline Analysis (IN PROGRESS)

- ✅ solve_objective_surface_from_datapacks.py
- ⏳ Evaluate which objective presets work best for which scenarios
- ⏳ Identify constraint violations in current policy

### Phase 3: Shadow Mode (NEXT)

- [ ] Create `train_sac_objective.py` with dual reward computation
- [ ] Log both old_reward and new_reward (RewardBuilder) side-by-side
- [ ] Verify they correlate as expected
- [ ] Run for 1000+ episodes to gather statistics

### Phase 4: Flip Switch (FUTURE)

- [ ] Replace old_reward with new_reward in training loop
- [ ] Monitor wage_parity, MPL, error_rate convergence
- [ ] Adjust objective_vector weights as needed
- [ ] Validate against constraint sets

---

## 7. Key Sanity Checks

### Check 1: RewardBuilder returns sensible magnitudes

```python
# Expected ranges:
# r_mpl: 0 - 100 (units/hour, matches human ~60)
# r_error: -1 - 0 (error rate 0-100%)
# r_energy: -50 - 0 (Wh consumed, expect 10-50 Wh/task)
# r_safety: -1 - 0 (binary: catastrophic or not)

# With default_objective_vector() [1.0, 0.2, 0.1, 0.05, 0.0]:
# Good episode: 60 * 1.0 + (-0.02) * 0.2 + (-15) * 0.1 + 0 * 0.05 = 58.5
# Bad episode:  20 * 1.0 + (-0.3) * 0.2 + (-40) * 0.1 + (-1) * 0.05 = 15.9

# Range: 15-60 (sensible for RL)
```

### Check 2: Objective presets differentiate behavior

```python
# "throughput" preset: [2.0, 1.0, 0.5, 1.0, 0.0]
# Should prioritize MPL over energy efficiency

# "energy_saver" preset: [1.0, 1.0, 3.0, 1.0, 0.0]
# Should penalize high energy consumption heavily

# "safety" preset: [0.5, 2.0, 1.0, 3.0, 0.0]
# Should avoid catastrophic failures strongly
```

### Check 3: Wage parity is tracked, not reward-shaped

RewardBuilder intentionally omits wage_parity convergence term because:
- Wage parity is a META objective (we want to converge to it)
- It's tracked at episode level, not step level
- Constraints can enforce it: `min_wage_parity: 0.9`

This is correct. RewardBuilder shapes behavior; wage parity is a constraint.

---

## 8. TODOs

1. **Create train_sac_objective.py stub** with shadow mode (log both rewards)
2. **Add EconObjectiveNet placeholder** that outputs identity mapping
3. **Validate reward magnitudes** on real episodes
4. **Run offline solver** on existing datapacks
5. **Document expected behavior** for each objective preset

---

## Summary

RewardBuilder + objective presets are **compatible** with current economics and **ready for integration**. The key insight is:

- **Current reward**: Step-level, fixed weights, includes wage parity
- **RewardBuilder**: Episode-level, programmable weights, excludes wage parity
- **Integration**: Shadow mode first, then flip when ready

The path is well-defined. When you implement, just fill in the TODOs in `train_sac_objective.py` and enable the `combine_reward()` call.
