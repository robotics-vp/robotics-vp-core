# Hydra Policy Neuralization Plan

## Architecture: Shared Trunk + Multi-Head

### 1. The Trunk
The `TrunkNet` (already defined in `src/rl/trunk_net.py`) fuses:
*   **Vision:** $z_t$ (from RegNet/BiFPN) or $h_t$ (from Spatial RNN).
*   **State:** Proprioception + Task State.
*   **Condition:** `ConditionVector` embedding (via FiLM or Concatenation).

### 2. The Heads
We replace the current stub/registry approach with true neural heads.
*   **Actor Heads:** One per `skill_mode`. Outputs $\mu, \sigma$ for the action distribution.
    *   `Head_Grasp`: Specialized for grasping logic.
    *   `Head_Move`: Specialized for navigation/approach.
    *   `Head_Insert`: Specialized for precision assembly.
*   **Critic Heads:** One per `skill_mode`. Outputs $Q(s, a)$.

### 3. Head Selection Logic
Selection remains **deterministic** based on `ConditionVector.skill_mode`.
$$ \pi(a|s, c) = \text{Head}_{c.skill\_mode}(\text{Trunk}(s, c)) $$
*   **Invariant:** The policy *never* chooses its own skill mode. The Orchestrator controls the mode via the ConditionVector.

## Multi-Objective RL Loss

We employ **SAC (Soft Actor-Critic)** as the base algorithm, augmented with economic auxiliary objectives.

### Primary Loss (Task Reward)
Standard SAC objectives:
1.  **Critic Loss:** MSE between predicted Q and target Q (Bellman update).
2.  **Actor Loss:** Maximize $Q(s, \pi(s)) - \alpha \log \pi(a|s)$.

### Auxiliary Loss 1: Economic Value (EV) Prediction
The Critic must also predict the *economic value* of the state, not just the task reward.
$$ L_{EV} = || Q_{econ}(s, a) - V_{econ}^{target} ||^2 $$
*   $V_{econ}^{target}$ comes from the `EconomicLearner` (Phase H).
*   This ensures the representations in the trunk are "economy-aware".

### Auxiliary Loss 2: Energy Cost Regularization
To encourage efficiency:
$$ L_{energy} = \beta \cdot \text{EnergyModel}(s, a) $$
*   Penalizes high-torque/high-velocity actions unless necessary for high reward.

## Training Strategy: "Hydra-Dropout"
To ensure the trunk learns robust features for *all* skills, we randomly sample skill modes during training rollouts (if the task allows) or ensure a balanced mix of tasks in the replay buffer.

*   **Gradient Blocking:** Gradients from Head A do not affect Head B directly, but they both shape the Trunk.
*   **Trunk Stability:** If one head becomes unstable, we can freeze the trunk and only train that head.

## Implementation Steps
1.  **Head Definitions:** Implement `GaussianActorHead` and `DoubleQCriticHead` classes in `hydra_heads.py`.
2.  **Loss Integration:** Update `hydra_losses.py` to include the EV and Energy terms.
3.  **Training Loop:** Ensure `train_sac.py` correctly routes batches to the active head based on the recorded `skill_mode`.

## Stage 5 Dependencies & Constraints
*   **Pre-requisites:**
    *   **Vision:** Backbone must be **frozen** (Stage 1).
    *   **Memory:** Spatial RNN must be **frozen** (Stage 2).
    *   **Segmentation:** Neural Segmenter must be at acceptable accuracy (Stage 3).
*   **Economic Sovereignty:**
    *   The policy optimizes the **existing** multi-objective reward defined in `EconomicLearner`.
    *   It **must not** introduce new reward terms that bypass the `EconDomainAdapter` or Phase H bounds.
*   **Conditioning:**
    *   `ConditionVector` is the **only** mechanism for skill selection. The policy cannot "decide" to switch skills internally.
