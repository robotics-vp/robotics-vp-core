# Stage 3: DataPack RL Sampler & Curriculum Specification

**Version**: 1.0
**Status**: Design Complete
**Dependencies**: Stage 1 (datapacks), Stage 2.4 (semantic tags), Economics Module, RewardBuilder

---

## 1. Overview

### 1.1 Purpose

Stage 3 provides **advisory-only orchestration** for RL training via two core components:

1. **DataPackRLSampler**: Samples RL episodes from datapacks using semantic tags, economics urgency, and episode descriptors
2. **DataPackCurriculum**: Formal curriculum logic (warmup → skill-building → frontier → fine-tuning) that produces sampling schedules

**Critical Constraint**: Stage 3 is **purely advisory**. It chooses **which episodes to present** but does NOT modify:
- RewardBuilder logic
- Objective vectors
- Economic parameters (MPL, wage, damage costs)
- RL update rules (SAC/PPO)

### 1.2 Design Principles

**Advisory-Only**:
- Sampler presents episodes; RL loop sees "just another replay sampler"
- Curriculum provides weight schedules; reward math remains unchanged
- No mutations to datapacks, tags, or econ outputs

**Deterministic**:
- Given identical inputs (datapacks, tags, seed), produces identical sampling sequence
- Stable sorting, reproducible RNG
- Determinism smoke tests required

**JSON-Safe**:
- All configs, schedules, and sampling logs serializable
- Schema-driven validation
- Backward-compatible with Stage 1/2 outputs

**Contract-Compliant**:
- Respects forbidden fields from Stage 2.4
- Honors economic constraints (read-only tier, novelty, MPL)
- Uses episode descriptors as advisory metadata

---

## 2. DataPackRLSampler

### 2.1 Core Responsibilities

**Inputs**:
- Stage 1 datapacks (enriched with Stage 2.4 semantic tags)
- Episode descriptors (tier, trust_score, sampling_weight, objective_vector)
- Curriculum schedule (from DataPackCurriculum)
- Sampling strategy config (balanced, frontier-prioritized, tag-aware)

**Outputs**:
- Sampled RL episodes (batches for training)
- Sampling logs (which episodes chosen, why, with what frequency)
- Diagnostics (coverage metrics, tier distribution, tag diversity)

**Process**:
1. Load enriched datapacks from Stage 1/2 merge
2. Parse semantic tags and supervision hints
3. Apply curriculum weights (from DataPackCurriculum)
4. Filter episodes based on strategy config
5. Sample with prioritization (tier, novelty, safety, econ-urgency)
6. Log sampling decisions for reproducibility

### 2.2 Sampling Strategies

#### 2.2.1 Balanced Sampling

**Goal**: Uniform coverage across tiers, tasks, and semantic profiles

**Algorithm**:
```python
def balanced_sample(episodes, batch_size, seed):
    # 1. Stratify by tier (0/1/2)
    tier_groups = group_by(episodes, key="tier")

    # 2. Sample proportionally from each tier
    samples_per_tier = {
        0: batch_size * 0.2,  # 20% Tier 0 (redundant)
        1: batch_size * 0.5,  # 50% Tier 1 (context-novel)
        2: batch_size * 0.3,  # 30% Tier 2 (frontier)
    }

    # 3. Within each tier, sample uniformly (or by trust_score)
    batch = []
    for tier, count in samples_per_tier.items():
        tier_episodes = tier_groups[tier]
        weights = [ep["trust_score"] for ep in tier_episodes]
        sampled = weighted_sample(tier_episodes, count, weights, seed)
        batch.extend(sampled)

    # 4. Shuffle to avoid tier-correlated batches
    shuffle(batch, seed=seed)
    return batch
```

**Tunable Parameters**:
- `tier_ratios`: `{0: 0.2, 1: 0.5, 2: 0.3}` (default)
- `use_trust_weighting`: `True` (weight by trust_score within tier)
- `stratify_by`: `["tier", "task_type"]` (additional stratification dims)

#### 2.2.2 Frontier-Prioritized Sampling

**Goal**: Focus on high-novelty, high-urgency, high-MPL-delta episodes

**Algorithm**:
```python
def frontier_prioritized_sample(episodes, batch_size, seed, urgency_threshold=0.7):
    # 1. Compute urgency score per episode
    for ep in episodes:
        urgency = compute_urgency(ep)  # combines tier, novelty, expected_mpl_gain
        ep["urgency_score"] = urgency

    # 2. Split into urgent vs. non-urgent
    urgent = [ep for ep in episodes if ep["urgency_score"] >= urgency_threshold]
    non_urgent = [ep for ep in episodes if ep["urgency_score"] < urgency_threshold]

    # 3. Sample 80% from urgent, 20% from non-urgent (for diversity)
    urgent_count = int(batch_size * 0.8)
    non_urgent_count = batch_size - urgent_count

    urgent_weights = [ep["urgency_score"] for ep in urgent]
    non_urgent_weights = [ep["sampling_weight"] for ep in non_urgent]

    batch = (
        weighted_sample(urgent, urgent_count, urgent_weights, seed) +
        weighted_sample(non_urgent, non_urgent_count, non_urgent_weights, seed)
    )

    shuffle(batch, seed=seed)
    return batch
```

**Urgency Computation**:
```python
def compute_urgency(ep):
    # Combines:
    # - tier (higher tier → higher urgency)
    # - novelty_score (from NoveltyTag)
    # - expected_mpl_gain (from economics)
    # - safety_critical flag (from SupervisionHints)

    tier_weight = {0: 0.2, 1: 0.5, 2: 1.0}[ep["tier"]]

    novelty = get_max_novelty_score(ep["enrichment"]["novelty_tags"])
    expected_mpl = sum(tag["expected_mpl_gain"] for tag in ep["enrichment"]["novelty_tags"])

    safety_boost = 1.5 if ep["enrichment"]["supervision_hints"]["safety_critical"] else 1.0

    urgency = tier_weight * (0.5 + 0.3 * novelty + 0.2 * min(expected_mpl / 10.0, 1.0)) * safety_boost

    return min(urgency, 1.0)  # Clip to [0, 1]
```

**Tunable Parameters**:
- `urgency_threshold`: `0.7` (default)
- `urgent_ratio`: `0.8` (default 80% urgent episodes)
- `tier_weights`: `{0: 0.2, 1: 0.5, 2: 1.0}`
- `safety_boost_factor`: `1.5`

#### 2.2.3 Tag-Aware Sampling

**Goal**: Ensure coverage of specific semantic profiles (safety, fragility, energy, novelty, affordances)

**Algorithm**:
```python
def tag_aware_sample(episodes, batch_size, seed, tag_quotas):
    # tag_quotas = {
    #     "safety_critical": 0.2,      # 20% safety-critical
    #     "fragile_objects": 0.15,     # 15% fragile objects
    #     "high_energy_cost": 0.1,     # 10% high-energy
    #     "novel_affordance": 0.15,    # 15% novel affordances
    #     "intervention": 0.1,         # 10% human interventions
    #     "baseline": 0.3,             # 30% unconstrained
    # }

    batch = []
    remaining = batch_size

    for tag_key, quota in tag_quotas.items():
        if tag_key == "baseline":
            continue

        count = int(batch_size * quota)
        matched = filter_by_tag(episodes, tag_key)

        if len(matched) > 0:
            weights = [ep["sampling_weight"] for ep in matched]
            sampled = weighted_sample(matched, min(count, len(matched)), weights, seed)
            batch.extend(sampled)
            remaining -= len(sampled)

    # Fill remaining with baseline (no tag constraints)
    if remaining > 0:
        baseline_weights = [ep["sampling_weight"] for ep in episodes]
        baseline_sampled = weighted_sample(episodes, remaining, baseline_weights, seed)
        batch.extend(baseline_sampled)

    shuffle(batch, seed=seed)
    return batch
```

**Tag Matching Logic**:
```python
def filter_by_tag(episodes, tag_key):
    if tag_key == "safety_critical":
        return [ep for ep in episodes if ep["enrichment"]["supervision_hints"]["safety_critical"]]

    elif tag_key == "fragile_objects":
        return [ep for ep in episodes
                if len(ep["enrichment"]["fragility_tags"]) > 0
                and any(tag["fragility_level"] in {"high", "critical"}
                        for tag in ep["enrichment"]["fragility_tags"])]

    elif tag_key == "high_energy_cost":
        return [ep for ep in episodes
                if any(tag["metric"] == "energy" and tag["score"] < 0.5
                       for tag in ep["enrichment"]["efficiency_tags"])]

    elif tag_key == "novel_affordance":
        return [ep for ep in episodes
                if any(not tag["demonstrated"]
                       for tag in ep["enrichment"]["affordance_tags"])]

    elif tag_key == "intervention":
        return [ep for ep in episodes
                if len(ep["enrichment"]["intervention_tags"]) > 0]

    else:
        return episodes  # Unknown tag → no filter
```

**Tunable Parameters**:
- `tag_quotas`: Dict mapping tag keys to quota fractions
- `allow_overlap`: `True` (episodes can satisfy multiple quotas)
- `fallback_to_baseline`: `True` (fill unfilled quotas with baseline)

### 2.3 Curriculum Integration

The sampler consumes a **curriculum schedule** from DataPackCurriculum:

```python
class CurriculumSchedule:
    stage: str  # "warmup" | "skill_building" | "frontier" | "fine_tuning"
    episode: int
    strategy: str  # "balanced" | "frontier_prioritized" | "tag_aware"
    strategy_params: Dict[str, Any]  # tier_ratios, urgency_threshold, tag_quotas, etc.
    weight_multipliers: Dict[str, float]  # {"safety_critical": 2.0, "fragile": 1.5}
    filter_constraints: Dict[str, Any]  # {"min_tier": 1, "require_tags": ["validated"]}
```

**Sampler Usage**:
```python
def sample_with_curriculum(episodes, schedule, batch_size, seed):
    # 1. Apply filter constraints
    filtered = apply_filters(episodes, schedule.filter_constraints)

    # 2. Apply weight multipliers (from supervision_hints)
    for ep in filtered:
        base_weight = ep["sampling_weight"]
        multiplier = get_multiplier(ep, schedule.weight_multipliers)
        ep["adjusted_weight"] = base_weight * multiplier

    # 3. Sample using strategy from schedule
    if schedule.strategy == "balanced":
        batch = balanced_sample(filtered, batch_size, seed, **schedule.strategy_params)
    elif schedule.strategy == "frontier_prioritized":
        batch = frontier_prioritized_sample(filtered, batch_size, seed, **schedule.strategy_params)
    elif schedule.strategy == "tag_aware":
        batch = tag_aware_sample(filtered, batch_size, seed, **schedule.strategy_params)
    else:
        raise ValueError(f"Unknown strategy: {schedule.strategy}")

    return batch
```

### 2.4 Sampling Logs

Every sampling call produces a log entry for reproducibility:

```json
{
  "sample_id": "sample_12345",
  "timestamp": "2025-11-17T12:34:56Z",
  "episode_count": 1234,
  "curriculum_stage": "frontier",
  "strategy": "frontier_prioritized",
  "strategy_params": {"urgency_threshold": 0.7, "urgent_ratio": 0.8},
  "batch_size": 64,
  "seed": 42,

  "sampled_episodes": [
    {"pack_id": "pack_001", "tier": 2, "urgency_score": 0.85, "weight": 1.5},
    {"pack_id": "pack_042", "tier": 1, "urgency_score": 0.72, "weight": 1.2},
    ...
  ],

  "diagnostics": {
    "tier_distribution": {"0": 12, "1": 32, "2": 20},
    "avg_urgency": 0.68,
    "avg_novelty": 0.54,
    "safety_critical_count": 8,
    "fragile_object_count": 15,
    "tag_coverage": {
      "fragility": 0.23,
      "risk": 0.31,
      "affordance": 0.67,
      "efficiency": 0.54,
      "novelty": 1.0,
      "intervention": 0.12
    }
  }
}
```

### 2.5 Interface Specification

```python
class DataPackRLSampler:
    """
    Samples RL training episodes from datapacks using curriculum + strategy.

    Advisory-only: does not modify datapacks, tags, or econ outputs.
    Deterministic: given same inputs + seed, produces same samples.
    """

    def __init__(
        self,
        datapacks: List[DataPackMeta],
        curriculum: DataPackCurriculum,
        strategy: str = "balanced",  # "balanced" | "frontier_prioritized" | "tag_aware"
        strategy_params: Optional[Dict[str, Any]] = None,
        seed: int = 42,
    ):
        """
        Args:
            datapacks: Enriched datapacks from Stage 1 + Stage 2.4 merge
            curriculum: Curriculum instance (provides schedule)
            strategy: Default sampling strategy (can be overridden by curriculum)
            strategy_params: Strategy-specific parameters
            seed: RNG seed for reproducibility
        """
        self.datapacks = datapacks
        self.curriculum = curriculum
        self.strategy = strategy
        self.strategy_params = strategy_params or {}
        self.rng = np.random.RandomState(seed)

        self.current_episode = 0
        self.sampling_logs = []

    def sample_batch(self, batch_size: int) -> List[Dict[str, Any]]:
        """
        Sample a batch of episodes for RL training.

        Args:
            batch_size: Number of episodes to sample

        Returns:
            List of episode dictionaries (with obs, actions, rewards, etc.)
        """
        # 1. Get curriculum schedule for current episode
        schedule = self.curriculum.get_schedule(self.current_episode)

        # 2. Sample using schedule (or fallback to default strategy)
        batch = sample_with_curriculum(
            self.datapacks,
            schedule,
            batch_size,
            self.rng.randint(0, 2**31)
        )

        # 3. Log sampling decision
        self._log_sample(batch, schedule)

        # 4. Increment episode counter
        self.current_episode += 1

        return batch

    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Returns:
            Sampling diagnostics (tier distribution, tag coverage, urgency stats)
        """
        return compute_sampling_diagnostics(self.sampling_logs)

    def reset(self, seed: Optional[int] = None):
        """Reset sampler state (for reproducibility testing)."""
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        self.current_episode = 0
        self.sampling_logs = []
```

---

## 3. DataPackCurriculum

### 3.1 Core Responsibilities

**Inputs**:
- Semantic tags from Stage 2.4 (SupervisionHints, prerequisite_tags, curriculum_stage)
- Econ profiles / objective presets (throughput, safety, energy_saver, balanced)
- Episode descriptors (tier, trust_score, novelty, MPL deltas)
- Curriculum config (stage definitions, transition rules, weight schedules)

**Outputs**:
- CurriculumSchedule per episode (strategy, params, weight multipliers, filters)
- Curriculum diagnostics (progress metrics, stage transitions, prerequisite satisfaction)

**Process**:
1. Define curriculum stages (warmup → skill-building → frontier → fine-tuning)
2. Define transition rules (when to advance from one stage to next)
3. Generate weight schedules (how to prioritize episodes within each stage)
4. Enforce prerequisite dependencies (don't sample advanced episodes too early)

### 3.2 Curriculum Stages

#### Stage 1: Warmup (Episodes 0–1000)

**Goal**: Build foundational motor skills, avoid damage

**Strategy**: Balanced sampling with safety bias

**Characteristics**:
- Focus on Tier 0 and Tier 1 (redundant + context-novel)
- Avoid Tier 2 (frontier) until basics mastered
- Prioritize safety_critical episodes
- Filter out fragility_level="critical"
- Use objective preset: "safety" `[1.0, 1.0, 0.5, 3.0, 0.0]`

**Schedule**:
```python
CurriculumSchedule(
    stage="warmup",
    episode=500,
    strategy="balanced",
    strategy_params={
        "tier_ratios": {0: 0.5, 1: 0.5, 2: 0.0},  # No Tier 2 yet
        "use_trust_weighting": True,
    },
    weight_multipliers={
        "safety_critical": 2.0,
        "fragile_high": 0.5,  # De-prioritize high-fragility
        "fragile_critical": 0.0,  # Skip critical fragility
    },
    filter_constraints={
        "max_tier": 1,
        "exclude_tags": ["fragile_critical", "novel_affordance"],
        "curriculum_stage": "early",  # From SupervisionHints
    }
)
```

#### Stage 2: Skill-Building (Episodes 1001–5000)

**Goal**: Master task-specific affordances, improve efficiency

**Strategy**: Tag-aware sampling with affordance coverage

**Characteristics**:
- Introduce Tier 2 at low rate (10%)
- Ensure coverage of affordances (demonstrated + novel)
- Balance efficiency metrics (time, energy, precision)
- Use objective preset: "balanced" `[1.0, 1.0, 1.0, 1.0, 0.0]`

**Schedule**:
```python
CurriculumSchedule(
    stage="skill_building",
    episode=3000,
    strategy="tag_aware",
    strategy_params={
        "tag_quotas": {
            "demonstrated_affordance": 0.3,
            "novel_affordance": 0.2,
            "efficiency_time": 0.2,
            "efficiency_energy": 0.1,
            "baseline": 0.2,
        }
    },
    weight_multipliers={
        "curriculum_stage_mid": 1.5,  # Boost episodes tagged "mid"
    },
    filter_constraints={
        "max_tier": 2,  # Allow all tiers
        "curriculum_stage": ["early", "mid"],
    }
)
```

#### Stage 3: Frontier (Episodes 5001–15000)

**Goal**: Maximize MPL, explore edge cases, pursue novel data

**Strategy**: Frontier-prioritized sampling with urgency

**Characteristics**:
- Heavily prioritize Tier 2 (frontier)
- Focus on high expected_mpl_gain episodes
- Explore intervention_tags (failure recovery, human corrections)
- Use objective preset: "throughput" `[2.0, 1.0, 0.5, 1.0, 0.0]`

**Schedule**:
```python
CurriculumSchedule(
    stage="frontier",
    episode=10000,
    strategy="frontier_prioritized",
    strategy_params={
        "urgency_threshold": 0.7,
        "urgent_ratio": 0.8,
        "tier_weights": {0: 0.1, 1: 0.3, 2: 1.0},
    },
    weight_multipliers={
        "tier_2": 2.0,
        "high_novelty": 1.5,
        "intervention": 1.8,
    },
    filter_constraints={
        "min_tier": 1,  # Skip Tier 0 (redundant)
        "curriculum_stage": ["mid", "late", "advanced"],
    }
)
```

#### Stage 4: Fine-Tuning (Episodes 15001+)

**Goal**: Polish policy, eliminate remaining errors, converge to wage parity

**Strategy**: Balanced sampling with error-focused weighting

**Characteristics**:
- Return to balanced sampling (avoid overfitting to frontier)
- Prioritize episodes where policy underperforms
- Focus on efficiency_tags with low scores
- Use objective preset: "balanced" or custom

**Schedule**:
```python
CurriculumSchedule(
    stage="fine_tuning",
    episode=20000,
    strategy="balanced",
    strategy_params={
        "tier_ratios": {0: 0.2, 1: 0.5, 2: 0.3},
        "use_trust_weighting": True,
    },
    weight_multipliers={
        "low_efficiency_score": 2.0,  # Prioritize weak areas
        "intervention": 1.5,
    },
    filter_constraints={
        "curriculum_stage": ["late", "advanced"],
    }
)
```

### 3.3 Transition Rules

**Automatic Stage Advancement**:
```python
def should_advance_stage(current_stage, episode, diagnostics):
    if current_stage == "warmup":
        # Advance if safety converged (low error rate for 100 episodes)
        if diagnostics["rolling_avg_error_rate_100ep"] < 0.05:
            return True

    elif current_stage == "skill_building":
        # Advance if affordances mastered (high success rate on demonstrated affordances)
        if diagnostics["affordance_success_rate"] > 0.8:
            return True

    elif current_stage == "frontier":
        # Advance if MPL growth plateaus
        if diagnostics["mpl_improvement_rate_1000ep"] < 0.05:
            return True

    # Fallback: advance by episode count
    thresholds = {"warmup": 1000, "skill_building": 5000, "frontier": 15000}
    return episode >= thresholds.get(current_stage, float('inf'))
```

**Manual Stage Override** (for debugging):
```python
curriculum.set_stage("frontier", episode=5001)
```

### 3.4 Prerequisite Enforcement

**From SupervisionHints**:
```python
supervision_hints = {
    "curriculum_stage": "late",
    "prerequisite_tags": ["basic_drawer_open", "fragile_object_awareness"],
    ...
}
```

**Curriculum Logic**:
```python
def filter_by_prerequisites(episodes, satisfied_prerequisites):
    """
    Only include episodes whose prerequisites are satisfied.

    Args:
        episodes: List of episode descriptors
        satisfied_prerequisites: Set of prerequisite tags learned so far

    Returns:
        Filtered list of episodes
    """
    filtered = []
    for ep in episodes:
        prereqs = ep["enrichment"]["supervision_hints"]["prerequisite_tags"]
        if all(prereq in satisfied_prerequisites for prereq in prereqs):
            filtered.append(ep)

    return filtered
```

**Updating Satisfied Prerequisites**:
```python
def update_prerequisites(diagnostics):
    """
    Mark prerequisites as satisfied based on policy performance.

    Example:
        If policy achieves >90% success on "basic_drawer_open" episodes,
        mark "basic_drawer_open" as satisfied.
    """
    satisfied = set()

    for tag, success_rate in diagnostics["tag_success_rates"].items():
        if success_rate > 0.9:
            satisfied.add(tag)

    return satisfied
```

### 3.5 Interface Specification

```python
class DataPackCurriculum:
    """
    Generates curriculum schedules for DataPackRLSampler.

    Advisory-only: provides weight schedules, doesn't modify rewards/objectives.
    Deterministic: same config + episode → same schedule.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        datapacks: List[DataPackMeta],
    ):
        """
        Args:
            config: Curriculum configuration (stages, transition rules, presets)
            datapacks: Enriched datapacks (for prerequisite analysis)
        """
        self.config = config
        self.datapacks = datapacks

        self.current_stage = "warmup"
        self.satisfied_prerequisites = set()
        self.diagnostics_history = []

    def get_schedule(self, episode: int) -> CurriculumSchedule:
        """
        Get curriculum schedule for a given episode.

        Args:
            episode: Current RL training episode

        Returns:
            CurriculumSchedule with strategy, params, multipliers, filters
        """
        # 1. Check if should advance stage
        if should_advance_stage(self.current_stage, episode, self.get_diagnostics()):
            self.current_stage = self._next_stage()

        # 2. Load stage config
        stage_config = self.config["stages"][self.current_stage]

        # 3. Build schedule
        schedule = CurriculumSchedule(
            stage=self.current_stage,
            episode=episode,
            strategy=stage_config["strategy"],
            strategy_params=stage_config["strategy_params"],
            weight_multipliers=stage_config["weight_multipliers"],
            filter_constraints=self._build_filters(stage_config),
        )

        return schedule

    def update_diagnostics(self, diagnostics: Dict[str, Any]):
        """
        Update curriculum state with latest training diagnostics.

        Args:
            diagnostics: RL training metrics (error rate, MPL, success rates)
        """
        self.diagnostics_history.append(diagnostics)
        self.satisfied_prerequisites = update_prerequisites(diagnostics)

    def get_diagnostics(self) -> Dict[str, Any]:
        """Returns curriculum diagnostics (stage, progress, prerequisites)."""
        return {
            "current_stage": self.current_stage,
            "satisfied_prerequisites": list(self.satisfied_prerequisites),
            "stage_progress": self._compute_stage_progress(),
            "next_stage_eta": self._estimate_next_stage_episode(),
        }

    def _next_stage(self) -> str:
        """Returns next curriculum stage."""
        stage_order = ["warmup", "skill_building", "frontier", "fine_tuning"]
        idx = stage_order.index(self.current_stage)
        if idx + 1 < len(stage_order):
            return stage_order[idx + 1]
        return self.current_stage  # Stay in fine_tuning
```

---

## 4. Integration with RL Training Scripts

### 4.1 Training Loop Modifications

**Before Stage 3** (simplified):
```python
# Old training loop
for episode in range(num_episodes):
    # Sample from replay buffer (no curriculum, no datapack-awareness)
    batch = replay_buffer.sample(batch_size)

    # Train policy
    for obs, action, reward, next_obs, done in batch:
        agent.store_transition(obs, action, reward, next_obs, done)
        agent.update()
```

**After Stage 3** (with sampler + curriculum):
```python
# New training loop with DataPackRLSampler
sampler = DataPackRLSampler(
    datapacks=enriched_datapacks,
    curriculum=curriculum,
    strategy="frontier_prioritized",
    seed=42
)

for episode in range(num_episodes):
    # Sample from datapacks using curriculum
    batch = sampler.sample_batch(batch_size)

    # Train policy (UNCHANGED - still uses RewardBuilder, objective_vector)
    for ep_data in batch:
        obs, action, reward, next_obs, done = ep_data
        agent.store_transition(obs, action, reward, next_obs, done)
        agent.update()

    # Update curriculum diagnostics
    diagnostics = compute_training_diagnostics(agent, env)
    curriculum.update_diagnostics(diagnostics)
```

**Key Points**:
- Sampler replaces old replay buffer sampling
- RewardBuilder, objective_vector, RL update rules UNCHANGED
- Curriculum diagnostics updated from training metrics

### 4.2 No Changes to RewardBuilder

```python
# RewardBuilder usage IDENTICAL before and after Stage 3

from valuation.reward_builder import build_reward_terms, combine_reward

# Compute reward terms (UNCHANGED)
reward_terms = build_reward_terms(episode_summary, econ_params)

# Combine with objective vector (UNCHANGED)
objective_vector = episode_descriptor["objective_vector"]  # From datapack
total_reward = combine_reward(objective_vector, reward_terms)

# Store transition (UNCHANGED)
agent.store_transition(obs, action, total_reward, next_obs, done)
```

### 4.3 No Changes to Objective Vectors

Objective vectors remain **read-only metadata** from datapacks:

```python
# Loaded from enriched datapack
episode_descriptor = datapack_to_rl_episode_descriptor(datapack)
objective_vector = episode_descriptor["objective_vector"]  # [1.0, 1.0, 1.0, 1.0, 0.0]

# Used for reward combination (UNCHANGED)
total_reward = combine_reward(objective_vector, reward_terms)
```

**Stage 3 does NOT modify objective_vector**. It only chooses which datapacks (with which objective vectors) to present.

### 4.4 Train Script CLI Integration

**New CLI Args**:
```bash
python -m train \
  --env dishwashing \
  --algorithm sac \
  --episodes 20000 \
  --datapack-dir results/stage1/datapacks \
  --enrichment-dir results/stage2/enrichments \
  --curriculum-config configs/curriculum_dishwashing.yaml \
  --sampling-strategy frontier_prioritized \
  --curriculum-auto-advance \
  --seed 42
```

**Config File** (`configs/curriculum_dishwashing.yaml`):
```yaml
stages:
  warmup:
    strategy: balanced
    strategy_params:
      tier_ratios: {0: 0.5, 1: 0.5, 2: 0.0}
      use_trust_weighting: true
    weight_multipliers:
      safety_critical: 2.0
      fragile_high: 0.5
      fragile_critical: 0.0
    filter_constraints:
      max_tier: 1
      exclude_tags: ["fragile_critical", "novel_affordance"]
      curriculum_stage: "early"
    transition_rule:
      type: rolling_error_rate
      threshold: 0.05
      window: 100
    fallback_episode: 1000

  skill_building:
    strategy: tag_aware
    strategy_params:
      tag_quotas:
        demonstrated_affordance: 0.3
        novel_affordance: 0.2
        efficiency_time: 0.2
        efficiency_energy: 0.1
        baseline: 0.2
    # ... (similar structure for other stages)
```

---

## 5. Forbidden Mutations (Contract Compliance)

### 5.1 What Stage 3 MUST NOT Change

**Economics Outputs** (read-only):
- `mpl_value`, `wage_parity`, `damage_cost`, `price_per_unit`
- Tier classifications (`tier: 0/1/2`)
- `novelty_score`, `expected_mpl_gain`
- `data_premium`, `trust_score`

**Semantic Tags** (read-only):
- All tag dataclasses (FragilityTag, RiskTag, AffordanceTag, etc.)
- `coherence_score`, `validation_status`
- `supervision_hints` (can consume, but not mutate)

**Datapacks** (read-only):
- Episode observations, actions, rewards
- Metadata (pack_id, task_type, engine_type)

**RewardBuilder Logic** (no override):
- `build_reward_terms()` function
- `combine_reward()` function
- Economic parameters (alpha, beta, gamma from econ config)

**Objective Vectors** (no override):
- Read from datapacks, used as-is
- No dynamic adjustment during training

### 5.2 What Stage 3 MAY Do

**Sampling Control** (advisory):
- Choose which episodes to present
- Apply weight multipliers (from supervision_hints)
- Filter by tier, tags, curriculum_stage
- Prioritize by urgency, novelty, safety

**Curriculum Orchestration** (advisory):
- Define stage transitions
- Enforce prerequisite dependencies
- Adjust sampling strategy per stage

**Logging & Diagnostics** (informational):
- Track sampling decisions
- Compute tag coverage metrics
- Monitor curriculum progress

---

## 6. Determinism Requirements

### 6.1 Reproducibility Contract

**Given identical inputs**:
- Same datapacks (with same enrichments)
- Same curriculum config
- Same seed

**Sampler MUST produce**:
- Identical sampling sequence
- Identical batch contents (same episodes in same order)
- Identical diagnostics

### 6.2 Determinism Strategy

**RNG Management**:
```python
class DataPackRLSampler:
    def __init__(self, ..., seed: int = 42):
        self.rng = np.random.RandomState(seed)  # Dedicated RNG instance

    def sample_batch(self, batch_size):
        # Use self.rng for all random operations
        batch_seed = self.rng.randint(0, 2**31)
        batch = sample_with_curriculum(..., seed=batch_seed)
        return batch
```

**Stable Sorting**:
```python
# Always sort by deterministic keys before sampling
episodes = sorted(episodes, key=lambda ep: (ep["pack_id"], ep["tier"], ep["timestamp"]))
```

**No External Randomness**:
- Don't use `random.random()` or `np.random.rand()` (global RNG)
- Don't depend on system time for sampling decisions
- Don't use multithreading/multiprocessing without deterministic seeding

### 6.3 Determinism Smoke Tests

**Test 1: Identical Inputs → Identical Outputs**:
```python
def test_determinism():
    sampler1 = DataPackRLSampler(datapacks, curriculum, seed=42)
    batch1 = sampler1.sample_batch(64)

    sampler2 = DataPackRLSampler(datapacks, curriculum, seed=42)
    batch2 = sampler2.sample_batch(64)

    assert batch1 == batch2  # Exact match
```

**Test 2: Curriculum State Reproducibility**:
```python
def test_curriculum_determinism():
    curriculum1 = DataPackCurriculum(config, datapacks)
    schedule1 = curriculum1.get_schedule(5000)

    curriculum2 = DataPackCurriculum(config, datapacks)
    schedule2 = curriculum2.get_schedule(5000)

    assert schedule1 == schedule2
```

---

## 7. JSON-Safe Schema

### 7.1 CurriculumSchedule Schema

```json
{
  "stage": "frontier",
  "episode": 10000,
  "strategy": "frontier_prioritized",
  "strategy_params": {
    "urgency_threshold": 0.7,
    "urgent_ratio": 0.8,
    "tier_weights": {"0": 0.1, "1": 0.3, "2": 1.0}
  },
  "weight_multipliers": {
    "tier_2": 2.0,
    "high_novelty": 1.5,
    "intervention": 1.8
  },
  "filter_constraints": {
    "min_tier": 1,
    "curriculum_stage": ["mid", "late", "advanced"]
  }
}
```

### 7.2 Sampling Log Schema

```json
{
  "sample_id": "sample_12345",
  "timestamp": "2025-11-17T12:34:56Z",
  "episode_count": 1234,
  "curriculum_stage": "frontier",
  "strategy": "frontier_prioritized",
  "batch_size": 64,
  "seed": 42,

  "sampled_episodes": [
    {"pack_id": "pack_001", "tier": 2, "urgency_score": 0.85}
  ],

  "diagnostics": {
    "tier_distribution": {"0": 12, "1": 32, "2": 20},
    "tag_coverage": {"fragility": 0.23, "risk": 0.31}
  }
}
```

---

## 8. Summary

**Stage 3 provides**:
1. **DataPackRLSampler**: Advisory episode selection with multiple strategies
2. **DataPackCurriculum**: Formal curriculum stages with automatic progression
3. **RL Integration**: Plug-and-play replacement for replay buffer sampling

**Stage 3 respects**:
- RewardBuilder logic (no override)
- Objective vectors (read-only from datapacks)
- Economic constraints (tier, novelty, MPL are read-only)
- Semantic tags (consume supervision_hints, don't mutate)

**Stage 3 guarantees**:
- Deterministic sampling (given same inputs + seed)
- JSON-safe serialization (all configs and logs)
- Contract compliance (no forbidden mutations)
- Backward compatibility (existing RL code works unchanged)

**Next**: See [STAGE_3_PIPELINE_DIAGRAMS.md](STAGE_3_PIPELINE_DIAGRAMS.md) for visual architecture.
