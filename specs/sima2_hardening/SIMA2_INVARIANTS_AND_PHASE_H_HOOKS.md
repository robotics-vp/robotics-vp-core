# SIMA-2 Invariants & Phase H Hooks

**Status**: Canonical Specification
**Owner**: Claude (Semantic Architect)
**Context**: Defines exact firing conditions, trust semantics, and Phase H integration

---

## 1. Tag Firing Conditions (Deterministic Semantics)

### 1.1. OODTag Firing Rules

**OODTag fires when ANY of the following conditions are met:**

#### Visual OOD
```python
def should_fire_visual_ood(embedding_distance: float, threshold: float = 0.7) -> bool:
    """
    Fire if CLIP embedding distance from training centroid exceeds threshold.

    Args:
        embedding_distance: Cosine distance in [0, 1]
        threshold: Default 0.7 (70th percentile of training distribution)

    Returns:
        True if embedding_distance > threshold
    """
    return embedding_distance > threshold
```

**Severity Mapping**:
- `0.7 < distance < 0.85`: severity = 0.5 (medium OOD)
- `0.85 < distance < 0.95`: severity = 0.8 (high OOD)
- `distance >= 0.95`: severity = 1.0 (critical OOD)

#### Kinematic OOD
```python
def should_fire_kinematic_ood(
    joint_velocity: np.ndarray,
    force: float,
    training_stats: Dict[str, float]
) -> bool:
    """
    Fire if joint velocities or contact forces exceed 99th percentile of training data.

    Args:
        joint_velocity: Current joint velocities (rad/s)
        force: Contact force magnitude (N)
        training_stats: {"vel_p99": float, "force_p99": float}

    Returns:
        True if max(joint_velocity) > vel_p99 OR force > force_p99
    """
    vel_ood = np.max(np.abs(joint_velocity)) > training_stats["vel_p99"]
    force_ood = force > training_stats["force_p99"]
    return vel_ood or force_ood
```

**Severity Mapping**:
- 99th percentile exceeded by 10-25%: severity = 0.6
- 99th percentile exceeded by 25-50%: severity = 0.8
- 99th percentile exceeded by >50%: severity = 1.0

#### Temporal OOD
```python
def should_fire_temporal_ood(
    segment_duration: int,
    primitive_type: str,
    mean_durations: Dict[str, float]
) -> bool:
    """
    Fire if segment duration exceeds 3x mean duration for that primitive type.

    Args:
        segment_duration: Timesteps in current segment
        primitive_type: "grasp", "pull", "transport", etc.
        mean_durations: {"grasp": 4.2, "pull": 6.5, ...}

    Returns:
        True if duration > 3 * mean_durations[primitive_type]
    """
    expected_mean = mean_durations.get(primitive_type, 10.0)  # Default 10
    return segment_duration > (3.0 * expected_mean)
```

**Severity Mapping**:
- 3-4x mean: severity = 0.5
- 4-6x mean: severity = 0.7
- >6x mean: severity = 1.0

#### OOD Suppression Rules

**DO NOT fire OODTag if:**
1. Segment is labeled as `recovery` (recovery primitives are expected to be atypical)
2. `TrustMatrix["OODTag"].trust_score < 0.5` (untrusted OOD detector)
3. Episode metadata contains `{"debug_mode": true}` (test/calibration rollouts)

---

### 1.2. RecoveryTag Firing Rules

**RecoveryTag fires when ALL of the following conditions are met:**

#### Pattern Match
```python
def should_fire_recovery_tag(
    segments: List[Segment],
    i: int  # Current segment index
) -> bool:
    """
    Fire if segments form: failure → correction → (success OR recovered).

    Pattern:
        segments[i-1].outcome in ["failure"]
        segments[i].metadata["recovery_observed"] == True
        segments[i+1].outcome in ["success", "recovered"] (if exists)

    Args:
        segments: Ordered list of segments
        i: Index of potential recovery segment

    Returns:
        True if recovery pattern detected
    """
    if i == 0 or i >= len(segments):
        return False

    prev_failed = segments[i-1].outcome == "failure"
    curr_is_recovery = segments[i].metadata.get("recovery_observed", False)

    # Check if next segment succeeds (if it exists)
    if i + 1 < len(segments):
        next_succeeds = segments[i+1].outcome in ["success", "recovered"]
    else:
        next_succeeds = True  # Terminal recovery counts

    return prev_failed and curr_is_recovery and next_succeeds
```

#### Value-Add Classification
```python
def classify_recovery_value(
    failure_severity: str,
    recovery_duration: int,
    final_success: bool
) -> str:
    """
    Classify recovery value-add.

    Returns: "low", "medium", "high"
    """
    if not final_success:
        return "low"

    if failure_severity == "critical" and recovery_duration < 10:
        return "high"  # Fast recovery from critical failure
    elif failure_severity in ["high", "critical"]:
        return "medium"
    else:
        return "low"  # Recovered from minor failure
```

#### Energy Cost Estimation
```python
def estimate_recovery_cost_wh(
    recovery_duration: int,
    primitive_type: str,
    base_power_per_timestep: float = 0.5
) -> float:
    """
    Estimate Watt-hours consumed during recovery.

    Heuristic: 0.5 Wh/timestep baseline, with multipliers for complex primitives.

    Args:
        recovery_duration: Timesteps spent in recovery
        primitive_type: "regrasp", "pick_from_drop", etc.
        base_power_per_timestep: Base power draw (Wh/timestep)

    Returns:
        Estimated energy cost in Wh
    """
    multipliers = {
        "regrasp": 1.0,
        "pick_from_drop": 1.5,  # Requires repositioning
        "wiggle_pull": 0.8,  # Low-force manipulation
    }
    mult = multipliers.get(primitive_type, 1.0)
    return recovery_duration * base_power_per_timestep * mult
```

#### RecoveryTag Suppression Rules

**DO NOT fire RecoveryTag if:**
1. No prior failure segment exists
2. Recovery primitive fails (no subsequent success)
3. `TrustMatrix["RecoveryTag"].trust_score < 0.5` (untrusted recovery detector)

---

## 2. TrustMatrix Semantics

### 2.1. Trust Score Interpretation

**Trust Score**: Scalar in [0.0, 1.0] representing predictive power of a tag.

```python
@dataclass
class TrustTier:
    min_score: float
    max_score: float
    tier_name: str
    policy_action: str

TRUST_TIERS = [
    TrustTier(0.8, 1.0, "Tier 1: Trusted", "Heavy sampling bias + auto-stop authority"),
    TrustTier(0.5, 0.8, "Tier 2: Provisional", "Moderate sampling bias + advisory warnings"),
    TrustTier(0.0, 0.5, "Tier 3: Untrusted", "Ignored by policies; debug logging only"),
]
```

### 2.2. Tier 1: Trusted Tags (Score > 0.8)

**Definition**: Tag is a proven predictor of economic reality.

**Policy Actions**:
1. **DataPackRLSampler**: Oversample episodes with this tag by 5x
   ```python
   if tag in trusted_tags:
       sampling_weight *= 5.0
   ```

2. **SemanticOrchestrator**: Tag can trigger `SafetyStop` proposals
   ```python
   if tag == "OODTag" and severity > 0.9 and trust_score > 0.8:
       propose_safety_stop()
   ```

3. **Curriculum Phasing**: Tag influences phase transitions
   ```python
   if "RiskTag" in trusted_tags and risk_exposure > threshold:
       curriculum_phase = "advanced"
   ```

4. **Reporting**: Tag appears in executive summaries

**Example**: `RiskTag` with trust_score = 0.92 → High-risk episodes are aggressively sampled; high-risk proposals trigger safety reviews.

---

### 2.3. Tier 2: Provisional Tags (0.5 < Score ≤ 0.8)

**Definition**: Tag shows promise but has noise or weak correlation.

**Policy Actions**:
1. **DataPackRLSampler**: Moderate sampling bias (1.5x weight)
   ```python
   if tag in provisional_tags:
       sampling_weight *= 1.5
   ```

2. **SemanticOrchestrator**: Advisory warnings only (no auto-stop)
   ```python
   if tag == "OODTag" and severity > 0.9 and trust_score in (0.5, 0.8):
       log_warning("High OOD detected (provisional trust)")
   ```

3. **DatapackAuditor**: Flag for human review
   ```python
   if datapack_score relies_on provisional_tags:
       audit_status = "requires_human_review"
   ```

4. **Reporting**: Tag appears in detailed diagnostics, not summaries

**Example**: `OODTag` with trust_score = 0.65 → OOD episodes get moderate sampling boost; operator sees warnings but system doesn't auto-stop.

---

### 2.4. Tier 3: Untrusted Tags (Score ≤ 0.5)

**Definition**: Tag is uncorrelated with economic outcomes; likely noisy or miscalibrated.

**Policy Actions**:
1. **All Policies**: Tag is **completely ignored**
   ```python
   if tag in untrusted_tags:
       continue  # Skip tag processing
   ```

2. **EconCorrelator**: Emit `CalibrationAlert` to engineering team
   ```json
   {
     "alert_type": "UntrustedTagDetected",
     "tag": "OODTag",
     "trust_score": 0.42,
     "recommendation": "Recalibrate OOD thresholds or disable detector"
   }
   ```

3. **Reporting**: Tag does not appear in any reports (only debug logs)

**Example**: `NoveltyTag` with trust_score = 0.38 → System behaves as if NoveltyTag doesn't exist; engineers investigate detector.

---

## 3. How Tags Influence Downstream Modules

### 3.1. Stage 2.4 (SemanticTagPropagator)

**Input**: Segments with OODTag / RecoveryTag
**Output**: Enhanced `SemanticEnrichmentProposal` with trust-weighted tags

```python
def propagate_with_trust(segment: Segment, trust_matrix: Dict) -> SemanticEnrichmentProposal:
    """
    Propagate tags, weighted by trust.

    Logic:
    - Trusted tags: Included with high confidence
    - Provisional tags: Included with medium confidence + warning flag
    - Untrusted tags: Excluded entirely
    """
    tags = []

    for tag_type, tag_data in segment.tags.items():
        trust_entry = trust_matrix.get(tag_type, {"trust_score": 0.0})
        trust_score = trust_entry["trust_score"]

        if trust_score > 0.8:
            tags.append({
                "type": tag_type,
                "data": tag_data,
                "confidence": 0.9,
                "trust_tier": "Tier1_Trusted"
            })
        elif trust_score > 0.5:
            tags.append({
                "type": tag_type,
                "data": tag_data,
                "confidence": 0.6,
                "trust_tier": "Tier2_Provisional",
                "warning": "Tag has provisional trust; verify manually"
            })
        # Untrusted tags (≤0.5) are dropped

    return SemanticEnrichmentProposal(tags=tags, ...)
```

---

### 3.2. DataPackRLSampler

**Input**: Datapacks with trust-weighted tags
**Output**: Sampling probabilities for RL training

```python
def compute_sampling_weight(datapack: Datapack, trust_matrix: Dict) -> float:
    """
    Compute sampling weight based on trusted tags.

    Base weight: 1.0
    Multipliers:
    - Trusted RiskTag: 5x
    - Trusted RecoveryTag: 3x
    - Provisional tags: 1.5x
    - Untrusted tags: 1x (ignored)
    """
    weight = 1.0

    for tag in datapack.tags:
        trust_score = trust_matrix.get(tag.type, {}).get("trust_score", 0.0)

        if trust_score > 0.8:
            if tag.type == "RiskTag":
                weight *= 5.0
            elif tag.type == "RecoveryTag":
                weight *= 3.0
        elif trust_score > 0.5:
            weight *= 1.5
        # Untrusted (≤0.5): no multiplier

    return weight
```

**Curriculum Interaction**:
```python
def select_curriculum_batch(
    datapacks: List[Datapack],
    curriculum_phase: str,
    trust_matrix: Dict
) -> List[Datapack]:
    """
    Sample datapacks for curriculum training.

    Phases:
    - "early": Sample low-risk, high-success episodes
    - "mid": Sample mixed risk, including recoveries
    - "advanced": Oversample high-risk, high-recovery episodes
    """
    if curriculum_phase == "early":
        # Filter to trusted low-risk episodes
        return [dp for dp in datapacks
                if "RiskTag" in dp.tags
                and trust_matrix["RiskTag"]["trust_score"] > 0.8
                and dp.tags["RiskTag"].severity < 0.5]

    elif curriculum_phase == "mid":
        # Balanced sampling with recovery boost
        weights = [compute_sampling_weight(dp, trust_matrix) for dp in datapacks]
        return weighted_sample(datapacks, weights)

    elif curriculum_phase == "advanced":
        # Heavily oversample risk + recovery
        return [dp for dp in datapacks
                if ("RiskTag" in dp.tags and trust_matrix["RiskTag"]["trust_score"] > 0.8)
                or ("RecoveryTag" in dp.tags and trust_matrix["RecoveryTag"]["trust_score"] > 0.8)]
```

---

### 3.3. DatapackAuditor

**Input**: Datapack with tags and quality score
**Output**: Audit status (passed / requires_review / rejected)

```python
def audit_datapack(datapack: Datapack, trust_matrix: Dict) -> AuditResult:
    """
    Audit datapack quality using trust-weighted tags.

    Rules:
    1. If quality_score relies on untrusted tags → downgrade score
    2. If OODTag (trusted) + severity > 0.9 → flag for review
    3. If RecoveryTag (trusted) → boost score
    """
    audit_status = "passed"
    score_adjustment = 0.0
    notes = []

    # Check if datapack relies on untrusted tags
    untrusted_tags = [t for t in datapack.tags
                      if trust_matrix.get(t.type, {}).get("trust_score", 0) <= 0.5]
    if untrusted_tags:
        score_adjustment -= 0.2
        notes.append(f"Datapack relies on untrusted tags: {untrusted_tags}")
        audit_status = "requires_review"

    # Check for trusted high-severity OOD
    for tag in datapack.tags:
        if tag.type == "OODTag" and trust_matrix.get("OODTag", {}).get("trust_score", 0) > 0.8:
            if tag.severity > 0.9:
                audit_status = "requires_review"
                notes.append("High-severity OOD detected (trusted detector)")

        # Boost for trusted recovery
        if tag.type == "RecoveryTag" and trust_matrix.get("RecoveryTag", {}).get("trust_score", 0) > 0.8:
            score_adjustment += 0.15
            notes.append("Trusted recovery pattern detected")

    adjusted_score = datapack.quality_score + score_adjustment

    return AuditResult(
        status=audit_status,
        adjusted_score=adjusted_score,
        notes=notes
    )
```

---

### 3.4. Reporting & Dashboards

**Executive Report** (Tier 1 tags only):
```json
{
  "sima2_summary": {
    "total_episodes": 10000,
    "high_risk_episodes": 1200,
    "recovery_successes": 450,
    "trusted_tags_used": ["RiskTag", "RecoveryTag"],
    "risk_premium": 2.3,
    "recovery_value_avg": 12.5
  }
}
```

**Diagnostic Report** (Tier 1 + Tier 2 tags):
```json
{
  "sima2_diagnostics": {
    "ood_detections": 230,
    "ood_trust_score": 0.67,
    "ood_status": "provisional",
    "warnings": ["OOD detector shows weak correlation; consider recalibration"]
  }
}
```

**Debug Logs** (All tags including Tier 3):
```json
{
  "debug_sima2": {
    "untrusted_tags": ["NoveltyTag"],
    "novelty_trust_score": 0.42,
    "calibration_alert": "NoveltyTag detector uncorrelated with MPL; recommend disable"
  }
}
```

---

## 4. Phase H Integration: SIMA-2 → Economic Learner

### 4.1. Skill Quality Signals

**Definition**: Metrics from SIMA-2 that indicate skill proficiency.

```python
@dataclass
class SkillQualitySignal:
    skill_id: str
    quality_metrics: Dict[str, float]

def extract_skill_quality(segments: List[Segment], skill_id: str) -> SkillQualitySignal:
    """
    Extract skill quality from SIMA-2 segments.

    Quality Metrics:
    - success_rate: Fraction of segments with outcome="success"
    - recovery_rate: Fraction of failures that led to recovery
    - avg_duration: Mean segment duration (shorter = more efficient)
    - fragility_score: Inverse of failure rate (higher = more robust)
    """
    skill_segments = [s for s in segments if s.metadata.get("skill_id") == skill_id]

    if not skill_segments:
        return SkillQualitySignal(skill_id, {})

    success_count = sum(1 for s in skill_segments if s.outcome == "success")
    recovery_count = sum(1 for s in skill_segments if s.metadata.get("recovery_observed"))
    failure_count = sum(1 for s in skill_segments if s.outcome == "failure")
    total = len(skill_segments)

    success_rate = success_count / total
    recovery_rate = recovery_count / failure_count if failure_count > 0 else 0.0
    avg_duration = np.mean([s.end_t - s.start_t for s in skill_segments])
    fragility_score = 1.0 - (failure_count / total)

    return SkillQualitySignal(
        skill_id=skill_id,
        quality_metrics={
            "success_rate": success_rate,
            "recovery_rate": recovery_rate,
            "avg_duration": avg_duration,
            "fragility_score": fragility_score
        }
    )
```

**Usage in Phase H**:
- High `success_rate` + high `fragility_score` → Skill is mature; harvest returns
- Low `recovery_rate` → Skill needs more failure-case training
- High `avg_duration` → Skill is inefficient; explore faster variants

---

### 4.2. Exploration Signals

**Definition**: Metrics from SIMA-2 that indicate exploration value.

```python
def extract_exploration_value(segments: List[Segment]) -> Dict[str, float]:
    """
    Extract exploration value from SIMA-2 segments.

    Exploration Metrics:
    - ood_rate: Fraction of segments with OODTag (novelty)
    - recovery_density: Recoveries per episode (learning opportunities)
    - failure_diversity: Number of unique failure modes
    """
    ood_count = sum(1 for s in segments if any(t.type == "OODTag" for t in s.tags))
    recovery_count = sum(1 for s in segments if s.metadata.get("recovery_observed"))

    failure_modes = set()
    for s in segments:
        if s.outcome == "failure":
            failure_modes.add(s.metadata.get("failure_mode", "unknown"))

    total = len(segments)

    return {
        "ood_rate": ood_count / total,
        "recovery_density": recovery_count / total,
        "failure_diversity": len(failure_modes)
    }
```

**Usage in Phase H**:
- High `ood_rate` → High exploration value; allocate more budget
- High `recovery_density` → Rich learning signal; prioritize for training
- High `failure_diversity` → Good coverage of edge cases

---

### 4.3. Phase H Skill Market Hooks

**Skill Pricing**:
```python
def price_skill_training_budget(
    skill_id: str,
    quality_signal: SkillQualitySignal,
    exploration_signal: Dict[str, float],
    current_mpl: float,
    target_mpl: float
) -> float:
    """
    Determine training budget allocation for a skill.

    Logic:
    - If skill quality is low but exploration value is high → Invest more
    - If skill quality is high → Reduce investment, harvest returns
    - If MPL gap is large → Prioritize skills with highest expected ΔMPL
    """
    mpl_gap = target_mpl - current_mpl

    # Quality penalty: low quality → need more training
    quality_penalty = 1.0 - quality_signal.quality_metrics.get("success_rate", 0.5)

    # Exploration bonus: high novelty → more valuable
    exploration_bonus = exploration_signal.get("ood_rate", 0.0) + exploration_signal.get("recovery_density", 0.0)

    # Base budget proportional to MPL gap
    base_budget = mpl_gap * 100.0  # $100 per MPL point

    adjusted_budget = base_budget * (1 + quality_penalty + exploration_bonus)

    return adjusted_budget
```

**Skill Retirement**:
```python
def should_retire_skill(quality_signal: SkillQualitySignal, min_success_rate: float = 0.95) -> bool:
    """
    Retire skill from active training if quality is sufficient.

    Criteria: success_rate > 0.95 AND fragility_score > 0.9
    """
    return (quality_signal.quality_metrics.get("success_rate", 0) > min_success_rate and
            quality_signal.quality_metrics.get("fragility_score", 0) > 0.9)
```

---

## 5. Contracts & Invariants Summary

### 5.1. MUST-ENFORCE Invariants

1. **OOD Containment**: Any OODTag with `severity > 0.9` AND `trust_score > 0.8` MUST trigger SafetyStop unless RecoveryTag immediately follows.

2. **Recovery Imperative**: If `RiskTag.severity > 0.8` AND `outcome == "success"` AND NO `RecoveryTag` exists, MUST flag as "Lucky" (false-positive safety).

3. **Trust Monotonicity**: Trust scores MUST NOT decrease without new calibration data (prevents oscillation).

4. **Sampling Conservation**: Total sampling weight across all datapacks MUST normalize to 1.0 (prevents drift).

5. **Advisory Boundary**: Tags MUST NOT directly modify:
   - Reward functions
   - EconController pricing
   - Task ordering
   These are strictly advisory inputs.

---

### 5.2. JSON Artifact Contracts

**TrustMatrix** (`results/sima2/trust_matrix.json`):
```json
{
  "RiskTag": {
    "trust_score": 0.92,
    "correlation_strength": "strong",
    "mean_damage": 7.3,
    "mean_energy": 14.2,
    "count": 1200,
    "economic_impact": "high_damage_predictor"
  },
  "RecoveryTag": {
    "trust_score": 0.87,
    "correlation_strength": "strong",
    "mean_damage": 2.1,
    "mean_energy": 18.5,
    "count": 450,
    "economic_impact": "resilience_marker"
  }
}
```

**Phase H Skill Signals** (`results/phase_h/skill_quality_signals.json`):
```json
{
  "skill_drawer_open": {
    "success_rate": 0.92,
    "recovery_rate": 0.78,
    "avg_duration": 12.3,
    "fragility_score": 0.88,
    "exploration_value": {
      "ood_rate": 0.05,
      "recovery_density": 0.12,
      "failure_diversity": 3
    },
    "recommended_budget": 450.0,
    "status": "active_training"
  }
}
```

---

## 6. Implementation Checklist for Codex

- [ ] Implement `OODTag` firing logic with visual/kinematic/temporal branches
- [ ] Implement `RecoveryTag` pattern matching and value classification
- [ ] Wire `TrustMatrix` into `DataPackRLSampler` with 5x/1.5x/1x multipliers
- [ ] Wire `TrustMatrix` into `SemanticOrchestrator` for SafetyStop authority
- [ ] Implement `DatapackAuditor` trust-weighted scoring
- [ ] Add `extract_skill_quality` and `extract_exploration_value` to SIMA-2 pipeline
- [ ] Create smoke tests for all invariants (OOD containment, recovery imperative, etc.)
- [ ] Generate `trust_matrix.json` and `skill_quality_signals.json` artifacts deterministically

---

**End of SIMA-2 Invariants & Phase H Hooks**
