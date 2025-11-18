# Stage 3: Pipeline Diagrams

**Version**: 1.0
**Status**: Design Complete

---

## 1. Overall Stage 3 Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          STAGE 3: SAMPLER & CURRICULUM                       │
│                            (Advisory Orchestration)                          │
└─────────────────────────────────────────────────────────────────────────────┘

INPUT LAYER (Read-Only)
┌────────────────────┬──────────────────────┬──────────────────────────────────┐
│  Stage 1 Datapacks │ Stage 2.4 Enrichments│   Economics Outputs              │
├────────────────────┼──────────────────────┼──────────────────────────────────┤
│ • Episodes (obs,   │ • Semantic Tags:     │ • tier (0/1/2)                   │
│   actions, states) │   - FragilityTag     │ • novelty_score                  │
│ • Metadata         │   - RiskTag          │ • expected_mpl_gain              │
│ • Task types       │   - AffordanceTag    │ • trust_score                    │
│ • Engine types     │   - EfficiencyTag    │ • sampling_weight                │
│                    │   - NoveltyTag       │ • damage_cost, price_per_unit    │
│                    │   - InterventionTag  │ • mpl_value, wage_parity         │
│                    │ • SupervisionHints:  │                                  │
│                    │   - priority_level   │                                  │
│                    │   - curriculum_stage │                                  │
│                    │   - safety_critical  │                                  │
│                    │   - prerequisite_tags│                                  │
│                    │ • coherence_score    │                                  │
└────────────────────┴──────────────────────┴──────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      EPISODE DESCRIPTOR NORMALIZATION                        │
│                                                                              │
│  Input: Datapack JSONL (Stage 1) + Enrichment (Stage 2.4)                   │
│  Output: Normalized Episode Descriptor                                      │
│                                                                              │
│  {                                                                           │
│    "pack_id": "pack_001",                                                    │
│    "objective_vector": [1.0, 1.0, 1.0, 1.0, 0.0],                           │
│    "tier": 2,                                                                │
│    "trust_score": 0.87,                                                      │
│    "sampling_weight": 1.305,  # trust * (1.0 + 0.5 * tier)                  │
│    "semantic_tags": ["fragile", "novel_affordance"],                        │
│    "enrichment": {                                                           │
│      "supervision_hints": {                                                  │
│        "priority_level": "high",                                             │
│        "curriculum_stage": "late",                                           │
│        "safety_critical": false,                                             │
│        "prerequisite_tags": ["basic_drawer_open"]                           │
│      },                                                                      │
│      "fragility_tags": [...],                                                │
│      "novelty_tags": [{"expected_mpl_gain": 6.5}],                          │
│      ...                                                                     │
│    }                                                                         │
│  }                                                                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DATA PACK CURRICULUM                                 │
│                  (Generates Curriculum Schedules)                            │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ Curriculum Config (YAML)                                             │   │
│  │ ─────────────────────────                                            │   │
│  │ stages:                                                              │   │
│  │   warmup:                                                            │   │
│  │     strategy: balanced                                               │   │
│  │     tier_ratios: {0: 0.5, 1: 0.5, 2: 0.0}                            │   │
│  │     weight_multipliers: {safety_critical: 2.0}                       │   │
│  │     filter_constraints: {max_tier: 1}                                │   │
│  │     transition_rule: {type: rolling_error_rate, threshold: 0.05}     │   │
│  │   skill_building:                                                    │   │
│  │     strategy: tag_aware                                              │   │
│  │     tag_quotas: {affordance: 0.3, efficiency: 0.2, ...}              │   │
│  │   frontier:                                                          │   │
│  │     strategy: frontier_prioritized                                   │   │
│  │     urgency_threshold: 0.7                                           │   │
│  │   fine_tuning:                                                       │   │
│  │     strategy: balanced                                               │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ Curriculum Logic                                                     │   │
│  │ ─────────────────                                                    │   │
│  │ 1. Check episode number → determine current stage                    │   │
│  │ 2. Evaluate transition rules → advance if conditions met             │   │
│  │ 3. Load stage config → generate CurriculumSchedule                   │   │
│  │ 4. Enforce prerequisites → filter episodes by satisfied_prereqs      │   │
│  │ 5. Return schedule to sampler                                        │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Output: CurriculumSchedule                                                  │
│  {                                                                           │
│    "stage": "frontier",                                                      │
│    "episode": 10000,                                                         │
│    "strategy": "frontier_prioritized",                                       │
│    "strategy_params": {"urgency_threshold": 0.7},                            │
│    "weight_multipliers": {"tier_2": 2.0},                                    │
│    "filter_constraints": {"min_tier": 1}                                     │
│  }                                                                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       DATA PACK RL SAMPLER                                   │
│                  (Samples Episodes for Training)                             │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ Sampling Strategy Selection                                          │   │
│  │ ───────────────────────────────                                      │   │
│  │                                                                       │   │
│  │  ┌────────────────┐  ┌─────────────────────┐  ┌─────────────────┐   │   │
│  │  │   BALANCED     │  │ FRONTIER-PRIORITIZED│  │   TAG-AWARE     │   │   │
│  │  ├────────────────┤  ├─────────────────────┤  ├─────────────────┤   │   │
│  │  │ • Stratify by  │  │ • Compute urgency   │  │ • Tag quotas    │   │   │
│  │  │   tier (0/1/2) │  │   score (tier +     │  │   (safety,      │   │   │
│  │  │ • Sample       │  │   novelty + MPL)    │  │   fragility,    │   │   │
│  │  │   proportional │  │ • Split urgent vs.  │  │   affordance)   │   │   │
│  │  │   (20/50/30%)  │  │   non-urgent        │  │ • Sample to     │   │   │
│  │  │ • Weight by    │  │ • Sample 80% urgent │  │   meet quotas   │   │   │
│  │  │   trust_score  │  │ • Weight by urgency │  │ • Fill baseline │   │   │
│  │  └────────────────┘  └─────────────────────┘  └─────────────────┘   │   │
│  │                                                                       │   │
│  │  Strategy chosen by CurriculumSchedule.strategy                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ Filtering & Weighting Pipeline                                       │   │
│  │ ───────────────────────────────                                      │   │
│  │ 1. Apply filter_constraints from schedule:                           │   │
│  │    - min_tier, max_tier                                              │   │
│  │    - curriculum_stage match                                          │   │
│  │    - exclude_tags, require_tags                                      │   │
│  │    - prerequisite satisfaction                                       │   │
│  │                                                                       │   │
│  │ 2. Apply weight_multipliers from schedule:                           │   │
│  │    - Base weight = sampling_weight (from descriptor)                 │   │
│  │    - Adjusted weight = base * multiplier (e.g., 2.0 for tier_2)      │   │
│  │                                                                       │   │
│  │ 3. Sample using strategy algorithm:                                  │   │
│  │    - balanced: weighted_sample(tier_groups, tier_ratios)             │   │
│  │    - frontier: weighted_sample(urgent, urgency_scores)               │   │
│  │    - tag_aware: weighted_sample(tag_groups, tag_quotas)              │   │
│  │                                                                       │   │
│  │ 4. Shuffle batch (with seeded RNG for determinism)                   │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Output: Sampled Batch (list of episode descriptors)                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SAMPLING DIAGNOSTICS                                 │
│                                                                              │
│  • Tier distribution (count per tier)                                       │
│  • Tag coverage (% episodes with each tag type)                             │
│  • Avg urgency score                                                         │
│  • Avg novelty score                                                         │
│  • Safety-critical count                                                    │
│  • Curriculum stage distribution                                            │
│                                                                              │
│  Logged to JSONL for reproducibility & analysis                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
OUTPUT LAYER
┌─────────────────────────────────────────────────────────────────────────────┐
│                            RL TRAINING LOOP                                  │
│                        (UNCHANGED from pre-Stage 3)                          │
│                                                                              │
│  for episode in range(num_episodes):                                         │
│      # Sample batch from DataPackRLSampler (NEW)                             │
│      batch = sampler.sample_batch(batch_size)                                │
│                                                                              │
│      for ep_data in batch:                                                   │
│          obs, action, next_obs, done = ep_data                               │
│                                                                              │
│          # Compute reward using RewardBuilder (UNCHANGED)                    │
│          reward_terms = build_reward_terms(summary, econ_params)             │
│          reward = combine_reward(objective_vector, reward_terms)             │
│                                                                              │
│          # Store transition (UNCHANGED)                                      │
│          agent.store_transition(obs, action, reward, next_obs, done)         │
│                                                                              │
│          # Update policy (UNCHANGED - SAC/PPO logic)                         │
│          agent.update()                                                      │
│                                                                              │
│      # Update curriculum diagnostics (NEW)                                   │
│      diagnostics = compute_training_diagnostics(agent, env)                  │
│      curriculum.update_diagnostics(diagnostics)                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Curriculum Stage Progression Flow

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                      CURRICULUM STAGE LIFECYCLE                               │
└──────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 1: WARMUP (Episodes 0–1000)                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│  Goal: Build foundational skills, avoid damage                               │
│                                                                              │
│  Strategy: Balanced sampling (safety bias)                                   │
│  Tier Filter: max_tier=1 (exclude Tier 2 frontier episodes)                 │
│  Weight Multipliers: safety_critical=2.0, fragile_critical=0.0              │
│  Objective Preset: "safety" [1.0, 1.0, 0.5, 3.0, 0.0]                       │
│                                                                              │
│  Transition Rule: rolling_error_rate < 0.05 (over 100 episodes)             │
│  Fallback: episode >= 1000                                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ ✓ Error rate converged OR episode >= 1000
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 2: SKILL-BUILDING (Episodes 1001–5000)                                │
├─────────────────────────────────────────────────────────────────────────────┤
│  Goal: Master affordances, improve efficiency                                │
│                                                                              │
│  Strategy: Tag-aware sampling (affordance coverage)                          │
│  Tier Filter: max_tier=2 (allow Tier 2 at ~10% rate)                        │
│  Tag Quotas:                                                                 │
│    - demonstrated_affordance: 30%                                            │
│    - novel_affordance: 20%                                                   │
│    - efficiency_time: 20%                                                    │
│    - efficiency_energy: 10%                                                  │
│    - baseline: 20%                                                           │
│  Objective Preset: "balanced" [1.0, 1.0, 1.0, 1.0, 0.0]                     │
│                                                                              │
│  Transition Rule: affordance_success_rate > 0.8                              │
│  Fallback: episode >= 5000                                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ ✓ Affordances mastered OR episode >= 5000
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 3: FRONTIER (Episodes 5001–15000)                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│  Goal: Maximize MPL, explore edge cases                                      │
│                                                                              │
│  Strategy: Frontier-prioritized sampling (urgency-driven)                    │
│  Tier Filter: min_tier=1 (skip Tier 0 redundant episodes)                   │
│  Urgency Threshold: 0.7 (80% urgent, 20% non-urgent)                        │
│  Weight Multipliers: tier_2=2.0, high_novelty=1.5, intervention=1.8         │
│  Objective Preset: "throughput" [2.0, 1.0, 0.5, 1.0, 0.0]                   │
│                                                                              │
│  Urgency Score = tier_weight * (0.5 + 0.3*novelty + 0.2*mpl_gain)           │
│                                                                              │
│  Transition Rule: mpl_improvement_rate < 0.05 (over 1000 episodes)          │
│  Fallback: episode >= 15000                                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ ✓ MPL plateaued OR episode >= 15000
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 4: FINE-TUNING (Episodes 15001+)                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│  Goal: Eliminate errors, converge to wage parity                             │
│                                                                              │
│  Strategy: Balanced sampling (error-focused weighting)                       │
│  Tier Filter: none (all tiers allowed)                                      │
│  Weight Multipliers: low_efficiency_score=2.0, intervention=1.5             │
│  Objective Preset: "balanced" or custom                                      │
│                                                                              │
│  Focus: Areas where policy underperforms (low efficiency scores)             │
│                                                                              │
│  Transition: None (terminal stage)                                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Sampling Strategy Comparison

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                     SAMPLING STRATEGY DECISION TREE                           │
└──────────────────────────────────────────────────────────────────────────────┘

                          CurriculumSchedule.strategy
                                      │
        ┌─────────────────────────────┼─────────────────────────────┐
        │                             │                             │
        ▼                             ▼                             ▼
┌──────────────────┐      ┌──────────────────────┐      ┌──────────────────┐
│    BALANCED      │      │ FRONTIER-PRIORITIZED │      │    TAG-AWARE     │
└──────────────────┘      └──────────────────────┘      └──────────────────┘

┌────────────────────────────────────────────────────────────────────────────┐
│ BALANCED SAMPLING                                                          │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  Episodes → Group by Tier                                                  │
│              │                                                             │
│      ┌───────┼───────┬────────┐                                           │
│      ▼       ▼       ▼        ▼                                           │
│   Tier 0  Tier 1  Tier 2   (Unknown)                                      │
│   (20%)   (50%)   (30%)                                                    │
│                                                                            │
│  Within each tier: weighted_sample(episodes, weights=trust_score)          │
│                                                                            │
│  Output: Batch with proportional tier distribution                         │
│                                                                            │
│  Use Cases:                                                                │
│  • Warmup stage (broad coverage, safety focus)                             │
│  • Fine-tuning stage (avoid overfitting to frontier)                       │
│                                                                            │
│  Pros: Stable, predictable, good coverage                                  │
│  Cons: May miss urgent/novel episodes                                      │
└────────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────────┐
│ FRONTIER-PRIORITIZED SAMPLING                                              │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  Episodes → Compute Urgency Score                                          │
│              │                                                             │
│      urgency = tier_weight * (0.5 + 0.3*novelty + 0.2*mpl_gain) * safety  │
│              │                                                             │
│      ┌───────┴────────┐                                                    │
│      ▼                ▼                                                    │
│   Urgent          Non-Urgent                                               │
│   (≥0.7)           (<0.7)                                                  │
│   [80% of batch]  [20% of batch]                                           │
│                                                                            │
│  Urgent: weighted_sample(urgent_episodes, weights=urgency_score)           │
│  Non-Urgent: weighted_sample(non_urgent, weights=sampling_weight)          │
│                                                                            │
│  Output: Batch skewed toward high-urgency episodes                         │
│                                                                            │
│  Use Cases:                                                                │
│  • Frontier stage (maximize MPL growth)                                    │
│  • Data-hungry training (pursue novel/high-value data)                     │
│                                                                            │
│  Pros: Fast learning, targets high-value data                              │
│  Cons: May neglect baseline skills                                         │
└────────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────────┐
│ TAG-AWARE SAMPLING                                                         │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  Tag Quotas: {                                                             │
│    "safety_critical": 20%,                                                 │
│    "fragile_objects": 15%,                                                 │
│    "novel_affordance": 15%,                                                │
│    "efficiency_time": 20%,                                                 │
│    "baseline": 30%                                                         │
│  }                                                                         │
│                                                                            │
│  Episodes → Filter by Tag                                                  │
│              │                                                             │
│      ┌───────┼───────┬────────┬────────┬────────┐                         │
│      ▼       ▼       ▼        ▼        ▼        ▼                         │
│   Safety  Fragile  Novel   Efficiency Baseline                             │
│   (20%)   (15%)    (15%)    (20%)      (30%)                               │
│                                                                            │
│  For each quota: weighted_sample(matched_episodes, sampling_weight)        │
│                                                                            │
│  Output: Batch with enforced tag coverage                                  │
│                                                                            │
│  Use Cases:                                                                │
│  • Skill-building stage (ensure affordance coverage)                       │
│  • Safety-critical training (guarantee safety episode inclusion)           │
│                                                                            │
│  Pros: Guarantees semantic diversity                                       │
│  Cons: May over-sample rare tags                                           │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Prerequisite Dependency Graph

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                 PREREQUISITE ENFORCEMENT (from SupervisionHints)              │
└──────────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────────┐
│  Example Task: Multi-Object Dishwashing                                    │
└────────────────────────────────────────────────────────────────────────────┘

                     ┌─────────────────────────┐
                     │  basic_motor_control    │
                     │  (curriculum_stage:     │
                     │   "early")              │
                     └───────────┬─────────────┘
                                 │
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
      ┌──────────────────────┐       ┌──────────────────────┐
      │  basic_drawer_open   │       │  basic_object_grasp  │
      │  (curriculum_stage:  │       │  (curriculum_stage:  │
      │   "early")           │       │   "early")           │
      └──────────┬───────────┘       └──────────┬───────────┘
                 │                               │
                 └───────────────┬───────────────┘
                                 ▼
                  ┌──────────────────────────────┐
                  │  fragile_object_awareness    │
                  │  (curriculum_stage: "mid")   │
                  │  prerequisite_tags:          │
                  │    ["basic_object_grasp"]    │
                  └──────────┬───────────────────┘
                             │
                 ┌───────────┴────────────┐
                 ▼                        ▼
      ┌─────────────────────┐   ┌────────────────────────┐
      │  multi_object_sort  │   │  delicate_placement    │
      │  (stage: "mid")     │   │  (stage: "mid")        │
      │  prereqs:           │   │  prereqs:              │
      │   ["fragile_aware"] │   │   ["fragile_aware"]    │
      └──────────┬──────────┘   └──────────┬─────────────┘
                 │                         │
                 └──────────┬──────────────┘
                            ▼
              ┌──────────────────────────────────┐
              │  complex_multi_object_dishwash   │
              │  (curriculum_stage: "late")      │
              │  prerequisite_tags:              │
              │    ["multi_object_sort",         │
              │     "delicate_placement"]        │
              └──────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────────┐
│  Filtering Logic                                                           │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  satisfied_prerequisites = {                                               │
│    "basic_motor_control",                                                  │
│    "basic_drawer_open",                                                    │
│    "basic_object_grasp",                                                   │
│    "fragile_object_awareness"                                              │
│  }                                                                         │
│                                                                            │
│  Episode: "multi_object_sort"                                              │
│    prerequisite_tags = ["fragile_aware"]                                   │
│    → All prerequisites satisfied → INCLUDE                                 │
│                                                                            │
│  Episode: "complex_multi_object_dishwash"                                  │
│    prerequisite_tags = ["multi_object_sort", "delicate_placement"]        │
│    → "multi_object_sort" NOT satisfied → EXCLUDE                           │
│                                                                            │
│  As training progresses:                                                   │
│    - Curriculum monitors success rates per tag                             │
│    - When tag success > 90%, mark prerequisite as satisfied                │
│    - Unlock dependent episodes in next sampling cycle                      │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. DataPackRLSampler Internal Flow

```
┌──────────────────────────────────────────────────────────────────────────────┐
│              DATAPACK RL SAMPLER: sample_batch() CALL FLOW                    │
└──────────────────────────────────────────────────────────────────────────────┘

Input: batch_size=64
       current_episode=10000

       ┌──────────────────────────────────────────┐
       │  1. Get Curriculum Schedule              │
       │     schedule = curriculum.get_schedule(  │
       │         episode=10000                    │
       │     )                                    │
       └──────────────┬───────────────────────────┘
                      │
                      ▼
       ┌──────────────────────────────────────────┐
       │  schedule = {                            │
       │    stage: "frontier",                    │
       │    strategy: "frontier_prioritized",     │
       │    strategy_params: {                    │
       │      urgency_threshold: 0.7              │
       │    },                                    │
       │    weight_multipliers: {                 │
       │      tier_2: 2.0,                        │
       │      high_novelty: 1.5                   │
       │    },                                    │
       │    filter_constraints: {                 │
       │      min_tier: 1,                        │
       │      curriculum_stage: ["mid", "late"]   │
       │    }                                     │
       │  }                                       │
       └──────────────┬───────────────────────────┘
                      │
                      ▼
       ┌──────────────────────────────────────────┐
       │  2. Apply Filters                        │
       │     filtered_episodes = []               │
       │     for ep in datapacks:                 │
       │       if ep["tier"] >= 1:  # min_tier    │
       │         if ep["enrichment"]["supervision │
       │            _hints"]["curriculum_stage"]  │
       │            in ["mid", "late"]:           │
       │           filtered_episodes.append(ep)   │
       └──────────────┬───────────────────────────┘
                      │
                      ▼
       ┌──────────────────────────────────────────┐
       │  3. Apply Weight Multipliers             │
       │     for ep in filtered_episodes:         │
       │       base = ep["sampling_weight"]       │
       │       multiplier = 1.0                   │
       │       if ep["tier"] == 2:                │
       │         multiplier *= 2.0                │
       │       if max_novelty(ep) > 0.7:          │
       │         multiplier *= 1.5                │
       │       ep["adjusted_weight"] = base *     │
       │                              multiplier  │
       └──────────────┬───────────────────────────┘
                      │
                      ▼
       ┌──────────────────────────────────────────┐
       │  4. Run Sampling Strategy                │
       │     if strategy == "frontier_prioritized"│
       │       urgent = [ep for ep if             │
       │                 urgency(ep) >= 0.7]      │
       │       non_urgent = [ep for ep if         │
       │                     urgency(ep) < 0.7]   │
       │       batch = (                          │
       │         weighted_sample(urgent, 51,      │
       │           weights=urgency_scores) +      │
       │         weighted_sample(non_urgent, 13,  │
       │           weights=adjusted_weights)      │
       │       )                                  │
       └──────────────┬───────────────────────────┘
                      │
                      ▼
       ┌──────────────────────────────────────────┐
       │  5. Shuffle (Deterministic)              │
       │     batch_seed = self.rng.randint(2**31) │
       │     shuffle(batch, seed=batch_seed)      │
       └──────────────┬───────────────────────────┘
                      │
                      ▼
       ┌──────────────────────────────────────────┐
       │  6. Log Sampling Decision                │
       │     log_entry = {                        │
       │       "sample_id": "sample_10000",       │
       │       "curriculum_stage": "frontier",    │
       │       "strategy": "frontier_prioritized",│
       │       "batch_size": 64,                  │
       │       "sampled_episodes": [...],         │
       │       "diagnostics": {                   │
       │         "tier_distribution": {...},      │
       │         "tag_coverage": {...}            │
       │       }                                  │
       │     }                                    │
       │     self.sampling_logs.append(log_entry) │
       └──────────────┬───────────────────────────┘
                      │
                      ▼
       ┌──────────────────────────────────────────┐
       │  7. Increment Episode Counter            │
       │     self.current_episode += 1            │
       └──────────────┬───────────────────────────┘
                      │
                      ▼
Output: batch (list of 64 episode descriptors)
```

---

## 6. RL Training Integration (Before vs. After)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                      BEFORE STAGE 3 (Baseline RL)                             │
└──────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────┐
│  Replay Buffer      │  ← Random sampling (or PER)
│  (no curriculum,    │  ← No semantic awareness
│   no datapack-aware)│  ← No econ-urgency prioritization
└──────────┬──────────┘
           │ sample(batch_size)
           ▼
┌─────────────────────────────────────────────────────────┐
│  RL Training Loop                                       │
│  ─────────────────                                      │
│  for ep in range(num_episodes):                         │
│    batch = replay_buffer.sample(64)                     │
│    for obs, action, reward, next_obs, done in batch:    │
│      agent.store_transition(...)                        │
│      agent.update()                                     │
└─────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│                      AFTER STAGE 3 (Curriculum-Driven RL)                     │
└──────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────┐     ┌──────────────────────┐     ┌─────────────────────┐
│  Stage 1 Datapacks  │────▶│ Stage 2.4 Enrichments│────▶│ Enriched Datapacks  │
└─────────────────────┘     └──────────────────────┘     └──────────┬──────────┘
                                                                     │
                                                                     ▼
                                                          ┌──────────────────────┐
                                                          │ DataPackCurriculum   │
                                                          │ (generates schedules)│
                                                          └──────────┬───────────┘
                                                                     │
                                                                     ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│  DataPackRLSampler                                                             │
│  ──────────────────                                                            │
│  • Consumes curriculum schedule                                               │
│  • Applies tier/tag/urgency filtering                                         │
│  • Samples with strategy (balanced, frontier, tag-aware)                      │
│  • Logs sampling decisions                                                    │
└────────────────────┬───────────────────────────────────────────────────────────┘
                     │ sample_batch(64)
                     ▼
┌─────────────────────────────────────────────────────────┐
│  RL Training Loop (UNCHANGED reward/objective logic)    │
│  ────────────────────────────────────────────────────   │
│  sampler = DataPackRLSampler(datapacks, curriculum)     │
│                                                         │
│  for ep in range(num_episodes):                         │
│    # NEW: curriculum-aware sampling                     │
│    batch = sampler.sample_batch(64)                     │
│                                                         │
│    # UNCHANGED: reward/training logic                   │
│    for obs, action, reward, next_obs, done in batch:    │
│      reward = combine_reward(                           │
│        objective_vector,  # from datapack (read-only)   │
│        build_reward_terms(summary, econ_params)         │
│      )                                                  │
│      agent.store_transition(obs, action, reward, ...)   │
│      agent.update()  # SAC/PPO update (unchanged)       │
│                                                         │
│    # NEW: update curriculum diagnostics                 │
│    diagnostics = compute_diagnostics(agent, env)        │
│    curriculum.update_diagnostics(diagnostics)           │
└─────────────────────────────────────────────────────────┘

KEY DIFFERENCES:
  • Sampling: Random/PER → Curriculum-driven with semantic awareness
  • Rewards: Unchanged (still uses RewardBuilder + objective_vector)
  • Policy Updates: Unchanged (still SAC/PPO)
  • New: Curriculum diagnostics feedback loop
```

---

## 7. Data Flow Summary

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        STAGE 3 DATA FLOW (End-to-End)                         │
└──────────────────────────────────────────────────────────────────────────────┘

Stage 1                Stage 2.4              Economics              Stage 3
──────────            ───────────            ──────────            ──────────

Datapacks     ───┐
                 ├──▶ Semantic Tags    ───┐
Video Demos   ───┤    (fragility,         │
                 │     risk,               │
Task Runs     ───┘     affordance,         │
                       efficiency,         │
                       novelty,            ├──▶ Enriched      ───┐
                       intervention)       │    Datapacks        │
                                           │                     │
                  Supervision Hints   ───┘                      ├──▶ Episode
                  (priority,                                    │    Descriptors
                   curriculum_stage,                            │    (normalized)
                   prerequisites)                               │
                                                                │
Economics     ────────────────────────────────────────┐         │
Module                                                │         │
(tier,                                                ├─────────┘
 novelty_score,                                       │
 expected_mpl_gain,                                   │
 trust_score,                                         │
 sampling_weight)                                     │
                                                      │
                                                      ▼
                                         ┌──────────────────────┐
                                         │  DataPackCurriculum  │
                                         │  (schedule generator)│
                                         └──────────┬───────────┘
                                                    │
                                         CurriculumSchedule
                                         (stage, strategy,
                                          params, multipliers,
                                          filters)
                                                    │
                                                    ▼
                                         ┌──────────────────────┐
                                         │ DataPackRLSampler    │
                                         │ (episode selector)   │
                                         └──────────┬───────────┘
                                                    │
                                         Sampled Batch (64 episodes)
                                                    │
                                                    ▼
                                         ┌──────────────────────┐
                                         │   RL Training Loop   │
                                         │   (SAC/PPO)          │
                                         └──────────┬───────────┘
                                                    │
                                         Training Diagnostics
                                         (error_rate, MPL,
                                          success_rates)
                                                    │
                                                    │ (feedback)
                                                    ▼
                                         ┌──────────────────────┐
                                         │  DataPackCurriculum  │
                                         │  (update state)      │
                                         └──────────────────────┘
```

---

## 8. Key Invariants

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         STAGE 3 CONTRACT INVARIANTS                           │
└──────────────────────────────────────────────────────────────────────────────┘

✓ ADVISORY-ONLY
  ├─ Sampler chooses WHICH episodes to present
  ├─ Curriculum chooses WHEN to advance stages
  ├─ NO modification to reward math, objectives, or econ params
  └─ RL loop sees "just another sampler"

✓ DETERMINISTIC
  ├─ Same (datapacks, config, seed) → same sampling sequence
  ├─ Stable sorting (by pack_id, tier, timestamp)
  ├─ Seeded RNG (no global randomness)
  └─ Reproducible curriculum transitions

✓ JSON-SAFE
  ├─ All configs serializable (YAML → JSON roundtrip)
  ├─ All logs serializable (sampling_logs, diagnostics)
  └─ Schema-driven validation

✓ CONTRACT-COMPLIANT
  ├─ Respects forbidden fields (tier, novelty_score, trust_score read-only)
  ├─ Honors supervision_hints (advisory, not mandatory)
  ├─ Uses RewardBuilder without modification
  └─ Objective vectors from datapacks (no override)

✓ BACKWARD-COMPATIBLE
  ├─ Enriched datapacks work with pre-Stage 3 code (enrichment is optional)
  ├─ Sampler can be disabled (fallback to random sampling)
  └─ No breaking changes to existing RL scripts
```

---

**Next**: See [STAGE_3_SMOKE_TESTS.md](STAGE_3_SMOKE_TESTS.md) for validation strategy.
