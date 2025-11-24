# SIMA-2 Neural Segmenter Design

## Model Specification

### Objective
Replace the rule-based `HeuristicSegmenter` with a neural network that predicts semantic primitive boundaries and labels from robot experience.

### Inputs
1.  **Proprioception:** Joint positions, velocities, gripper width, force/torque (if available).
2.  **Vision/Memory:** Spatial RNN hidden state $h_t$ (captures context like "approaching object").
3.  **Action:** Previous action $a_{t-1}$.

### Architecture: Temporal 1D CNN (TCN)
A lightweight 1D Convolutional network operating on a sliding window of history (e.g., 32 timesteps).

*   **Backbone:** 3-layer 1D CNN with dilated convolutions to increase receptive field.
*   **Heads:**
    1.  **Boundary Head:** Sigmoid output $P(B_t | \text{context})$. Predicts if time $t$ is a segment start/end.
    2.  **Classification Head:** Softmax output $P(C_t | \text{context})$. Predicts primitive class (Approach, Grasp, Move, Release, etc.).
    3.  **Failure Head:** Sigmoid output $P(F_t | \text{context})$. Predicts if the current segment is failing.

## Loss Functions

### 1. Boundary Loss (Focal Loss)
Since boundaries are rare (sparse), we use Focal Loss to handle class imbalance.
$$ L_{bound} = -\alpha (1-p_t)^\gamma \log(p_t) $$
where $p_t$ is the probability of the true class (boundary vs. no-boundary).

### 2. Classification Loss (Cross Entropy)
Standard CE loss for the primitive label, masked to only apply during active segments (ignoring transitions if ambiguous).

### 3. Smoothness Regularization
To prevent flickering predictions:
$$ L_{smooth} = || P(C_t) - P(C_{t-1}) ||^2 $$
(Applied only when $P(B_t)$ is low).

## Bootstrapping Strategy

### Stage 1: Heuristic Distillation (Weak Supervision)
1.  **Generate Data:** Run the existing `HeuristicSegmenter` on 50k offline trajectories.
2.  **Label:** Create "silver" labels for boundaries and classes.
3.  **Weighting:** Use `TrustMatrix` scores to down-weight low-confidence heuristic outputs.
4.  **Train:** Train the Neural Segmenter to mimic the Heuristic Segmenter.
    *   *Benefit:* The neural net learns to smooth out noise and can run faster/more consistently.
    *   *Risk:* It learns the heuristics' bugs.

### Stage 2: Human-in-the-Loop Refinement
1.  **Active Learning:** Identify low-confidence segments or segments where Neural and Heuristic disagree.
2.  **Annotation:** Human annotators (or GPT-4V) correct the labels for these hard cases.
3.  **Finetune:** Update the model on the high-quality "gold" dataset.

### Stage 3: Self-Supervised Consistency
Enforce that the predicted segments align with keyframes in the visual latent space (e.g., significant changes in $z_t$ should correlate with boundaries).

## Stage 5 Dependencies
*   **Labels:** `HeuristicSegmenter` outputs + `OODTag`/`RecoveryTag` from Stage 5 stress runs provide the initial labels.
*   **Trust:** `TrustMatrix` is used to weight the contribution of each sample (trust high-confidence heuristic labels more).
*   **Input:** `ConditionVector` provides the semantic context for the segmenter.

## Integration
*   **Interface:** The `NeuralSegmenter` must implement the same interface as `HeuristicSegmenter` (taking a rollout/window and returning `DetectedPrimitive` objects).
*   **Fallback:** Keep `HeuristicSegmenter` as a fallback if Neural confidence is low ($< 0.7$).
