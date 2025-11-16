# SyntheticWeightController: Conceptual Design

**Purpose:** Unified mother-module for computing final per-sample synthetic weights, coordinating all gating components with safety rails.

---

## 1. High-Level Architecture

```
                    ┌─────────────────────────────────────┐
                    │       SyntheticWeightController     │
                    └─────────────────────────────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    ▼                                   ▼
           ┌─────────────┐                    ┌─────────────┐
           │  trust_net  │                    │  λ_controller│
           │  (per-sample)│                    │  (global)    │
           └──────┬──────┘                    └──────┬──────┘
                  │                                  │
                  ▼                                  ▼
           ┌─────────────┐                    ┌─────────────┐
           │ w_econ_lattice │                 │ objective    │
           │ (per-sample) │                   │ vector       │
           └──────┬──────┘                    └──────┬──────┘
                  │                                  │
                  └─────────────┬───────────────────┘
                                ▼
                    ┌─────────────────────────────────────┐
                    │   w_final = trust × econ × λ_budget │
                    │   (with safety caps)                │
                    └─────────────────────────────────────┘
```

---

## 2. Module Inputs

### Per-Sample Inputs (for each synthetic transition):
- `branch_states`: (H+1, latent_dim) - rollout trajectory
- `branch_actions`: (H, action_dim) - actions taken
- `source_episode`: int - which real episode this branched from
- `source_timestep`: int - when the branch started
- `trust_score`: float - from trust_net (already computed, or recompute)
- `delta_mpl`: float - ΔMPL for the synthetic branch
- `delta_error`: float - Δerror rate
- `delta_ep`: float - Δenergy productivity
- `novelty_score`: float - state-action space novelty
- `brick_id`: int - categorical brick embedding

### Global Inputs (shared across batch):
- `objective_vector`: (4,) - [productivity, precision, energy, novelty]
- `current_mpl`: float - current policy's MPL
- `current_error`: float - current error rate
- `current_ep`: float - current energy productivity
- `baseline_mpl`: float - baseline from profile
- `baseline_error`: float - baseline error rate
- `baseline_ep`: float - baseline EP
- `training_progress`: float - epoch / total_epochs (for curriculum)
- `max_synth_share`: float - hard cap from profile (e.g., 0.4)
- `target_synth_share`: float - desired synthetic contribution (e.g., 0.2)

---

## 3. Module Outputs

### Per-Sample Outputs:
- `w_final`: float ∈ [0, 1] - final weight for each synthetic sample
- `trust_component`: float - contribution from trust_net
- `econ_component`: float - contribution from w_econ_lattice
- `lambda_component`: float - global λ_synth scaling factor

### Global Outputs:
- `effective_synth_share`: float - actual contribution ratio
- `lambda_prediction`: float - λ_controller's predicted optimal share
- `safety_caps_applied`: dict - which safety rails were triggered
- `statistics`: dict - mean/std of weights, trust, econ components

---

## 4. Internal Components

### 4.1 Trust Gate (trust_net)
```python
def compute_trust(self, branch):
    """
    Computes per-sample trust ∈ [0, 1].
    High trust = trajectory is physically plausible.
    """
    return self.trust_net(branch_states)
```

### 4.2 Economic Weighting (w_econ_lattice)
```python
def compute_econ_weight(self, branch):
    """
    Computes per-sample economic value ∈ [0, 1].
    High weight = branch improves MPL, reduces error, good energy use.
    """
    return self.w_econ_model(
        delta_mpl, delta_error, delta_ep, novelty, brick_id, objective_vector
    )
```

### 4.3 Lambda Controller (λ_controller)
```python
def compute_lambda(self, global_context):
    """
    Computes global λ_synth ∈ [0, max_synth_share].
    Controls overall synthetic budget based on meta-objective.
    """
    features = build_feature_vector(
        objective_vector, current_mpl, baseline_mpl, ...
    )
    return self.lambda_controller(features, max_synth_share)
```

### 4.4 Safety Rails
```python
def apply_safety_rails(self, weights):
    """
    Enforce hard constraints:
    1. Trust floor: w = 0 if trust < min_trust_threshold
    2. Econ cap: w_econ ≤ econ_weight_cap (default 1.0)
    3. Lambda cap: λ ≤ max_synth_share (hard cap)
    4. Final cap: effective_synth_share ≤ max_synth_share
    """
    # Never bypass trust_net
    trust_mask = trust_scores >= self.min_trust_threshold
    weights = weights * trust_mask

    # Apply caps
    weights = torch.clamp(weights, 0, self.econ_weight_cap)

    # Scale to achieve target share (respecting max cap)
    return self.balance_source_contributions(weights)
```

---

## 5. Weight Computation Strategy

### Current (Multiplicative):
```python
w_final = trust × w_econ × λ
```
**Problem:** Over-penalization when all components are conservative.

### Proposed (Separated Concerns):
```python
# Per-sample quality gate (multiplicative)
w_quality = trust × w_econ

# Global budget controller (additive scaling)
λ_budget = λ_controller.predict(global_context)

# Balance source contributions
w_final = rescale_to_achieve_budget(w_quality, λ_budget)
```

This separates:
1. **Quality gating** (trust × econ): "Is this sample trustworthy and valuable?"
2. **Budget control** (λ): "How much synthetic data do we want overall?"

---

## 6. Interaction Diagram

```
Input: Real data buffer R, Synthetic branch buffer S

Step 1: Compute per-sample trust scores
    for each branch in S:
        trust[i] = trust_net(branch)

Step 2: Compute per-sample econ weights
    for each branch in S:
        econ[i] = w_econ_lattice(ΔMPL, Δerror, ΔEP, novelty, brick_id)

Step 3: Quality gate
    quality[i] = trust[i] × econ[i]

Step 4: Lambda budget prediction
    λ = λ_controller(global_context)

Step 5: Balance contributions
    # Total real weight = |R| × 1.0
    # Desired synth weight = |R| × λ / (1 - λ)
    # Scale quality weights to achieve this
    w_final[i] = quality[i] × scale_factor

Step 6: Safety rails
    - Enforce trust floor
    - Cap effective share at max_synth_share
    - Log any rail triggers

Step 7: Return weights
    return w_final, metadata
```

---

## 7. Integration with Training Pipeline

### In train_offline_with_local_synth.py:
```python
# Initialize controller
synth_weight_ctrl = SyntheticWeightController(
    trust_net=load_trust_net(profile['trust_net_path']),
    w_econ_model=load_w_econ_lattice(profile['w_econ_lattice_path']),
    lambda_controller=load_lambda_controller(profile['synth_lambda_controller_path']),
    profile=profile
)

# During training
for epoch in range(n_epochs):
    # Compute weights for current epoch
    global_context = {
        'current_mpl': compute_current_mpl(),
        'current_error': compute_current_error(),
        'training_progress': epoch / n_epochs,
        ...
    }

    w_final, metadata = synth_weight_ctrl.compute_weights(
        synthetic_branches, global_context
    )

    # Sample batch with weights
    batch = sample_weighted_batch(real_buffer, synth_buffer, w_final)

    # Train policy
    loss = update_policy(batch)

    # Log metadata
    log_synthetic_weighting(metadata)
```

---

## 8. Future Extensions

### 8.1 Horizon Selection Module
```python
def select_branch_horizon(self, world_model, state):
    """
    Dynamically choose rollout horizon based on:
    - Trust decay rate at different horizons
    - Computational budget
    - Variance growth rate
    """
    for H in [10, 20, 40, 60]:
        trust_H = world_model.rollout_trust(state, H)
        if trust_H >= min_trust_threshold:
            best_H = H
    return best_H
```

### 8.2 Objective-Conditioned Weighting
```python
def compute_objective_conditioned_weight(self, branch, objective_vector):
    """
    Weight branches based on how well they align with current task objectives.
    High productivity objective → favor high ΔMPL branches
    High precision objective → favor low Δerror branches
    """
    # Already in w_econ_lattice via objective_vector input
    pass
```

### 8.3 Distributional λ Prediction
```python
def predict_lambda_distribution(self, global_context):
    """
    Instead of point estimate, predict distribution over λ.
    Use mean for budget, uncertainty for exploration.
    """
    mean_lambda, std_lambda = self.lambda_ctrl(global_context)
    # Sample from distribution for curriculum learning
    sampled_lambda = mean_lambda + std_lambda * torch.randn(1)
    return torch.clamp(sampled_lambda, 0, max_synth_share)
```

---

## 9. Safety Invariants

**CRITICAL: These must NEVER be violated:**

1. **Trust is ALWAYS gated first:**
   ```python
   w_final = trust × ...  # Trust is always multiplicative, never bypassed
   ```

2. **Max synth share is a HARD CAP:**
   ```python
   assert effective_synth_share <= max_synth_share
   ```

3. **No synthetic data if trust < threshold:**
   ```python
   if trust < min_trust_threshold:
       w_final = 0  # Reject branch entirely
   ```

4. **All weights are logged:**
   ```python
   log_to_trajectory(trust, econ, lambda, w_final, metadata)
   ```

5. **RL weighting is 100% DL-driven:**
   ```python
   # No heuristics in final weighting
   # trust_net = learned from data
   # w_econ_lattice = learned from outcomes
   # λ_controller = learned from meta-objectives
   ```

---

## 10. Implementation Priority

1. **Phase 1 (Current):** Use trust-only with proven -0.13% improvement
2. **Phase 2:** Retrain w_econ_lattice on actual J-based targets
3. **Phase 3:** Integrate λ_controller as budget controller (not additional gate)
4. **Phase 4:** Build full SyntheticWeightController module
5. **Phase 5:** Add distributional λ and horizon selection
6. **Phase 6:** Connect to objective-conditioning stack

---

## Summary

The SyntheticWeightController is the "brain" of synthetic data integration:
- **Inputs:** Per-sample branch stats + global training context
- **Outputs:** Final per-sample weights with full traceability
- **Components:** trust_net (quality gate) + w_econ_lattice (value gate) + λ_controller (budget)
- **Safety:** Trust is NEVER bypassed, max_synth_share is hard cap
- **Modularity:** Each component can be updated independently
- **Logging:** Full trajectory of all weight components for analysis

This design allows the flywheel to:
1. Trust the world model's branches (trust_net)
2. Value economically sound samples (w_econ_lattice)
3. Control synthetic budget based on meta-objectives (λ_controller)
4. Maintain safety bounds at all times (safety rails)
5. Learn from actual outcomes, not heuristics (100% DL-driven)
