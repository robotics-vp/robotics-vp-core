# Spatial RNN Design

## Architecture Comparison

### 1. S4D (Structured State Space with Diagonalization)
*   **Pros:** Extremely efficient for long sequences ($O(L \log L)$), handles long-range dependencies well, parallelizable training.
*   **Cons:** Newer, potentially less stable in mixed-modality RL settings than gated RNNs.
*   **Verdict:** **Primary Candidate for Long-Horizon Tasks.**

### 2. ConvGRU (Convolutional Gated Recurrent Unit)
*   **Pros:** Preserves spatial structure (unlike standard GRU), computationally cheaper than LSTM, stable training.
*   **Cons:** Struggles with very long sequences (vanishing gradients), sequential computation limits training speed.
*   **Verdict:** **Robust Baseline / Fallback.**

### 3. ConvLSTM (Convolutional Long Short-Term Memory)
*   **Pros:** Powerful gating mechanism, proven track record in video prediction.
*   **Cons:** Heavy compute/memory footprint, often overkill compared to GRU.
*   **Verdict:** **Discard** (too heavy for real-time inference constraints).

## Selected Architecture: Hybrid S4D-Conv Block
We propose a hybrid approach where S4D handles the temporal dynamics of flattened spatial features, or a lightweight ConvGRU processes the spatial map directly. Given the need for spatial awareness in manipulation:

**Decision:** **ConvGRU** (1-2 layers) for the initial Phase I to ensure spatial feature stability, transitioning to S4D in Phase II if horizon length becomes a bottleneck.

### ConvGRU Specification
*   **Input:** BiFPN Features $P_t$ (e.g., $128 \times 14 \times 14$).
*   **State:** Hidden state $h_t$ (same dim).
*   **Update:**
    $$ z_t = \sigma(W_z * [h_{t-1}, x_t]) $$
    $$ r_t = \sigma(W_r * [h_{t-1}, x_t]) $$
    $$ \tilde{h}_t = \tanh(W_h * [r_t \odot h_{t-1}, x_t]) $$
    $$ h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t $$

## Temporal Latent Prediction ($z_{t+1} | z_t, a_t$)
To force the RNN to capture dynamics (and not just memorize), we add an auxiliary **Forward Dynamics Loss**.

### Architecture
*   **Predictor:** MLP or shallow ConvNet taking $(h_t, a_t)$.
*   **Target:** Next frame vision features $z_{t+1}$ (from the frozen Vision encoder).
*   **Loss:** Cosine Similarity or MSE.
    $$ L_{dyn} = || \text{Predict}(h_t, a_t) - \text{StopGrad}(z_{t+1}) ||^2 $$

### Integration with Downstream
*   **Policy:** Receives $h_t$ (spatial memory) + $z_t$ (current vision).
*   **Segmenter:** Receives $h_t$ to detect temporal boundaries (e.g., "velocity dropped *after* contact").

## Stage 5 Dependencies
*   **Inputs:**
    *   **Vision:** Frozen features from `regnet_backbone.py` (Stage 1).
    *   **Conditioning:** `ConditionVector` (from `isaac_adapter` or `ros_bridge`) provides the context for memory gating.
    *   **Tags:** `OODTag` and `RecoveryTag` serve as auxiliary supervision signals (e.g., "reset memory on recovery").
*   **Supervision:**
    *   **TrustMatrix:** Used to weight the loss during training (trust high-confidence segments more).

## Implementation Plan
1.  **Stub Replacement:** Replace `spatial_rnn_adapter.py` with a PyTorch `ConvGRUCell`.
2.  **Training:**
    *   **Data:** Segmented trajectories + Econ slices + ConditionVectors.
    *   **Phase A:** Train purely as a dynamics model (predict $z_{t+1}$) on offline rollouts.
    *   **Phase B:** Freeze and use as a feature extractor for Policy/Segmenter.
    *   **Phase C:** End-to-end finetuning (optional, risky).
