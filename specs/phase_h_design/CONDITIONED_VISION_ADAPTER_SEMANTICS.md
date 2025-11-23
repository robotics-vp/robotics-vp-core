# Conditioned Vision Adapter: Complete Semantics

**Status**: Canonical Specification
**Owner**: Claude (Semantic Architect)
**Context**: How ConditionVector modulates visual perception

---

## 1. Vision Modulation Principles

### 1.1. Core Idea

**The robot should "see" what it "needs" to see based on economic/semantic context.**

- **Safety-first mode**: Vision emphasizes fragile objects, collision risks, high-cost regions
- **Frontier-explore mode**: Vision amplifies novel objects, under-explored regions, anomalies
- **Energy-efficient mode**: Vision suppresses low-value clutter, focuses on minimal-motion paths

---

## 2. ConditionVector → Vision Mapping

### 2.1. Which Fields Modulate Vision?

```python
VISION_RELEVANT_FIELDS = {
    "risk_tolerance": "Controls risk saliency amplification",
    "safety_language_modulation": "Boosts fragile object detection",
    "skill_mode": "Changes attention routing (precision vs speed vs exploration)",
    "novelty_tier": "Amplifies OOD/novel feature importance",
    "objective_vector": "Selectively highlights/suppresses object classes",
    "energy_budget_wh": "Influences spatial downsampling / feature sparsity"
}
```

---

### 2.2. Modulation Mechanisms

#### A. FiLM (Feature-wise Linear Modulation)

**Apply to RegNet features**:
```python
def apply_film_modulation(
    features: Tensor,  # [B, C, H, W]
    condition_vector: ConditionVector
) -> Tensor:
    """
    Apply FiLM: features_out = γ * features + β

    γ (scale), β (shift) derived from condition_vector
    """
    # Compute scale/shift from condition latent
    z_c = condition_vector.to_vector()  # [D] dimensional latent
    gamma = learned_fc_gamma(z_c)  # [C] channel-wise scale
    beta = learned_fc_beta(z_c)    # [C] channel-wise shift

    # Apply modulation
    features_modulated = gamma.view(1, -1, 1, 1) * features + beta.view(1, -1, 1, 1)

    return features_modulated
```

#### B. Attention Reweighting

**Modify BiFPN fusion weights**:
```python
def compute_fusion_weights(
    feature_pyramid: List[Tensor],
    condition_vector: ConditionVector
) -> List[float]:
    """
    Compute fusion weights for BiFPN based on condition.

    Example:
    - If skill_mode = "precision": Boost high-resolution features
    - If skill_mode = "speed": Boost low-resolution (coarse) features
    """
    base_weights = [1.0] * len(feature_pyramid)

    if condition_vector.skill_mode == "precision":
        # Amplify fine-grained features (higher resolutions)
        base_weights[-1] *= 2.0  # Boost finest level
    elif condition_vector.skill_mode == "speed":
        # Amplify coarse features (lower resolutions)
        base_weights[0] *= 2.0  # Boost coarsest level

    # Normalize
    total = sum(base_weights)
    return [w / total for w in base_weights]
```

#### C. Spatial RNN Gating

**Modulate temporal smoothing**:
```python
def compute_rnn_gate(
    prev_hidden: Tensor,
    curr_features: Tensor,
    condition_vector: ConditionVector
) -> Tensor:
    """
    Compute gating coefficient for spatial RNN.

    Logic:
    - If skill_mode = "exploration": Low gate (favor new observations)
    - If skill_mode = "precision": High gate (smooth over time)
    """
    base_gate = sigmoid(learned_gate_fn(curr_features))

    if condition_vector.skill_mode == "exploration":
        gate = base_gate * 0.5  # Reduce temporal smoothing
    elif condition_vector.skill_mode == "precision":
        gate = base_gate * 1.5  # Increase temporal smoothing

    return gate.clamp(0, 1)
```

---

## 3. Three Canonical Regimes

### 3.1. Safety-First Regime

**Condition Vector**:
```json
{
  "risk_tolerance": 0.2,
  "safety_language_modulation": 0.9,
  "skill_mode": "safety_critical"
}
```

**Vision Modulation Effects**:

1. **Risk Saliency Amplification**:
   - RiskMap values multiplied by `(1.0 - risk_tolerance + 0.5)` = 1.3
   - Fragile objects highlighted with 2x intensity
   - Collision zones expanded by 20%

2. **Feature Weighting**:
   - High-resolution features boosted (fine-grained obstacle detection)
   - Edge detection filters amplified
   - Temporal smoothing increased (reduce false positives)

3. **BiFPN Fusion**:
   - Fine-scale features get 2x weight
   - Coarse features downweighted to 0.5x

4. **Object Saliency**:
   ```python
   if object.fragility > 0.7:
       saliency *= 3.0  # Highlight fragile objects aggressively
   ```

**Result**: Robot "sees" the environment as full of hazards; moves cautiously, wide berth around obstacles.

---

### 3.2. Frontier-Explore Regime

**Condition Vector**:
```json
{
  "curriculum_phase": "frontier",
  "novelty_tier": 2,
  "skill_mode": "exploration"
}
```

**Vision Modulation Effects**:

1. **Novelty Amplification**:
   - OOD features boosted by `(novelty_tier + 1.0)` = 3x
   - Novel object embedding distance → higher saliency
   - Familiar objects suppressed to 0.3x

2. **Feature Weighting**:
   - Mid-resolution features emphasized (balance detail vs context)
   - Texture/color variance features amplified
   - Temporal smoothing reduced (detect transient novelties)

3. **BiFPN Fusion**:
   - Balanced weights across all scales
   - Slight boost to mid-scale features (optimal for object discovery)

4. **Attention Routing**:
   ```python
   if object.embedding_distance > ood_threshold:
       attention_weight *= 5.0  # Focus on novel objects
   ```

**Result**: Robot "sees" the environment as a treasure map of novel features; attention drawn to unfamiliar objects/regions.

---

### 3.3. Energy-Efficient Regime

**Condition Vector**:
```json
{
  "energy_budget_wh": 50.0,
  "skill_mode": "energy_efficient",
  "risk_tolerance": 0.5
}
```

**Vision Modulation Effects**:

1. **Spatial Downsampling**:
   - Process every other frame (50% compute reduction)
   - Reduce BiFPN iterations from 3 to 2
   - Use lower-resolution feature pyramid

2. **Feature Sparsity**:
   - Apply top-k sparsification: Keep only top 30% most salient features
   - Low-saliency regions processed at 1/4 resolution

3. **Temporal Smoothing**:
   - Increase RNN gate to 0.9 (reuse prev features more)
   - Reduce redundant re-computation

4. **Object Filtering**:
   ```python
   if object.econ_value < min_value_threshold:
       saliency = 0.0  # Suppress low-value clutter
   ```

**Result**: Robot "sees" a simplified environment; ignores clutter, focuses on high-value objects, reuses temporal context.

---

## 4. Field-Specific Modulation Rules

### 4.1. risk_tolerance → Risk Saliency

```python
def modulate_risk_map(
    risk_map: Tensor,
    risk_tolerance: float
) -> Tensor:
    """
    Modulate risk map based on tolerance.

    Low tolerance (0.2): Amplify risks
    High tolerance (0.8): Dampen risks
    """
    scale = 1.0 - risk_tolerance + 0.5
    return risk_map * scale
```

**Examples**:
- `risk_tolerance = 0.2`: scale = 1.3 (risks amplified)
- `risk_tolerance = 0.8`: scale = 0.7 (risks dampened)

---

### 4.2. novelty_tier → OOD Focus

```python
def modulate_ood_saliency(
    ood_score: Tensor,
    novelty_tier: int
) -> Tensor:
    """
    Amplify OOD features based on novelty tier.

    Tier 0 (redundant): 0.5x (suppress novelty)
    Tier 2 (frontier): 3.0x (amplify novelty)
    """
    multipliers = {0: 0.5, 1: 1.0, 2: 3.0}
    mult = multipliers.get(novelty_tier, 1.0)
    return ood_score * mult
```

---

### 4.3. skill_mode → Attention Routing

```python
SKILL_MODE_CONFIGS = {
    "precision": {
        "resolution_boost": "high",  # Fine-grained features
        "temporal_smoothing": "high",  # Stable tracking
        "saliency_threshold": "low"  # Notice small details
    },
    "speed": {
        "resolution_boost": "low",  # Coarse features
        "temporal_smoothing": "low",  # React to changes quickly
        "saliency_threshold": "high"  # Ignore distractions
    },
    "exploration": {
        "resolution_boost": "medium",
        "temporal_smoothing": "low",  # Detect novelty
        "saliency_threshold": "medium"
    },
    "safety_critical": {
        "resolution_boost": "high",
        "temporal_smoothing": "high",
        "saliency_threshold": "very_low"  # Notice everything
    }
}
```

---

### 4.4. objective_vector → Object Highlighting

```python
def apply_objective_vector(
    object_features: Tensor,
    object_labels: List[str],
    objective_vector: Dict[str, float]
) -> Tensor:
    """
    Selectively boost/suppress object classes.

    Example: {"blue": 1.0, "red": 0.0} → Boost blue, suppress red
    """
    for i, label in enumerate(object_labels):
        weight = objective_vector.get(label, 1.0)
        object_features[i] *= weight

    return object_features
```

---

### 4.5. energy_budget_wh → Compute Sparsity

```python
def compute_feature_sparsity(
    energy_budget_wh: Optional[float],
    base_compute_wh: float = 10.0
) -> float:
    """
    Compute feature sparsity ratio based on energy budget.

    Returns: Fraction of features to keep (0.3-1.0)
    """
    if energy_budget_wh is None:
        return 1.0  # No sparsity

    # If budget is tight, increase sparsity
    ratio = energy_budget_wh / base_compute_wh
    sparsity = max(0.3, min(1.0, ratio))

    return sparsity
```

---

## 5. Implementation Architecture

### 5.1. ConditionedVisionAdapter Structure

```python
class ConditionedVisionAdapter(nn.Module):
    """
    Vision encoder modulated by ConditionVector.

    Architecture:
    1. RegNet backbone → multi-scale features
    2. FiLM conditioning → feature modulation
    3. BiFPN → multi-scale fusion (condition-weighted)
    4. Spatial RNN → temporal integration (condition-gated)
    5. Risk/Affordance heads → semantic maps (condition-scaled)
    """

    def __init__(self, config):
        self.regnet = RegNet(...)
        self.film_layers = nn.ModuleList([FiLMLayer(...) for _ in range(4)])
        self.bifpn = BiFPN(...)
        self.spatial_rnn = SpatialRNN(...)
        self.risk_head = RiskHead(...)
        self.affordance_head = AffordanceHead(...)

    def forward(self, image: Tensor, condition: ConditionVector) -> Dict[str, Tensor]:
        # 1. Base encode (unchanged by condition)
        features_pyramid = self.regnet(image)

        # 2. Apply FiLM modulation
        z_c = condition.to_vector()
        features_modulated = [
            self.film_layers[i](f, z_c) for i, f in enumerate(features_pyramid)
        ]

        # 3. Compute condition-dependent fusion weights
        fusion_weights = self._compute_fusion_weights(condition)

        # 4. BiFPN fusion
        fused_features = self.bifpn(features_modulated, weights=fusion_weights)

        # 5. Temporal integration with condition-gated RNN
        gate = self._compute_rnn_gate(condition)
        h_t = self.spatial_rnn(fused_features, gate=gate)

        # 6. Semantic heads with condition-scaled outputs
        risk_map = self.risk_head(h_t)
        risk_map = self._modulate_risk_map(risk_map, condition.risk_tolerance)

        affordance_map = self.affordance_head(h_t)

        return {
            "z_v": h_t,  # Vision latent (base representation)
            "risk_map": risk_map,
            "affordance_map": affordance_map,
            "fused_features": fused_features
        }
```

---

### 5.2. Invariants

**Invariant 1: Base Representation Stability**
- `z_v` (vision latent) MUST be identical for same image, regardless of ConditionVector
- Only downstream heads and fusion are modulated

**Invariant 2: Determinism**
- Same (image, ConditionVector) → same output
- No randomness in modulation

**Invariant 3: Boundedness**
- All modulation scales clamped to [0.1, 10.0] (prevent explosion/collapse)

---

## 6. Integration Points

### 6.1. ObservationAdapter Wiring

```python
class ObservationAdapter:
    def __init__(self, config):
        if config.get("use_conditioned_vision", False):
            self.vision_adapter = ConditionedVisionAdapter(config)
        else:
            self.vision_adapter = VisionEncoderWithHeads(config)  # Legacy

    def build_observation(
        self,
        frame: VisionFrame,
        proprioception: Dict,
        condition_vector: Optional[ConditionVector] = None
    ) -> Observation:
        if isinstance(self.vision_adapter, ConditionedVisionAdapter):
            vision_output = self.vision_adapter(frame.image, condition_vector)
        else:
            vision_output = self.vision_adapter(frame.image)

        return Observation(
            vision=vision_output,
            proprioception=proprioception,
            condition=condition_vector
        )
```

---

## 7. Smoke Tests

```python
def test_regime_differences():
    """Test that three regimes produce different outputs."""
    image = load_test_image()

    cv_safety = ConditionVector(risk_tolerance=0.2, skill_mode="safety_critical")
    cv_explore = ConditionVector(novelty_tier=2, skill_mode="exploration")
    cv_efficient = ConditionVector(energy_budget_wh=50.0, skill_mode="energy_efficient")

    adapter = ConditionedVisionAdapter(config)

    out_safety = adapter(image, cv_safety)
    out_explore = adapter(image, cv_explore)
    out_efficient = adapter(image, cv_efficient)

    # Risk maps should differ
    assert not torch.allclose(out_safety["risk_map"], out_explore["risk_map"])

    # z_v should be identical (base representation)
    assert torch.allclose(out_safety["z_v"], out_explore["z_v"], atol=1e-5)
```

---

**End of Conditioned Vision Adapter Semantics**
