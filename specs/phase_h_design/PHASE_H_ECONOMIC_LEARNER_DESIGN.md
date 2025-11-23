# Phase H: Economic Learner & Skill Market

**Status**: Canonical Specification
**Owner**: Claude (Semantic Architect)
**Context**: Dynamic skill acquisition via economic returns

---

## 1. Core Concept

**Skills are assets with measurable returns.**

The Economic Learner treats each skill as an investment:
- **Cost to train**: Data collection, compute, human supervision
- **Returns**: ΔMPL (productivity), ΔWh (efficiency), Δdamage (quality)
- **Decision**: Buy more training data vs harvest current returns

---

## 2. Skill Representation

### 2.1. Skill-as-Asset Model

```python
@dataclass
class Skill:
    """
    A skill is a trainable capability with economic attributes.
    """
    skill_id: str  # "drawer_open_v2", "dish_place_precision", etc.
    display_name: str
    description: str

    # Economic Stats
    mpl_baseline: float  # Initial MPL (units/hr)
    mpl_current: float   # Current MPL after training
    mpl_target: float    # Target MPL (economic goal)

    # Cost Stats
    training_cost_usd: float  # Cumulative training cost
    data_cost_per_episode: float  # Cost per additional training episode

    # Risk/Quality Stats
    success_rate: float  # Fraction of successful executions
    failure_rate: float  # Fraction of failures
    recovery_rate: float  # Fraction of failures recovered
    fragility_score: float  # 1.0 - failure_rate

    # Exploration Stats
    ood_exposure: float  # Fraction of training data that was OOD
    novelty_tier_avg: float  # Average novelty tier of training data

    # Metadata
    training_episodes: int
    last_updated: str  # ISO timestamp
    status: str  # "training", "mature", "deprecated"
```

---

### 2.2. Skill Lifecycle States

```python
class SkillStatus(Enum):
    EXPLORATION = "exploration"  # Early phase: high exploration budget
    TRAINING = "training"        # Active training: balanced exploration/exploitation
    MATURE = "mature"            # High performance: harvest returns
    DEPRECATED = "deprecated"    # Replaced by better skill variant
```

**Transition Rules**:
```python
def update_skill_status(skill: Skill) -> str:
    """
    Update skill status based on performance.

    Rules:
    - If success_rate > 0.95 AND mpl_current >= mpl_target → MATURE
    - If success_rate < 0.6 → EXPLORATION
    - Else → TRAINING
    """
    if skill.success_rate > 0.95 and skill.mpl_current >= skill.mpl_target:
        return SkillStatus.MATURE.value

    if skill.success_rate < 0.6:
        return SkillStatus.EXPLORATION.value

    return SkillStatus.TRAINING.value
```

---

## 3. Exploration Budget Allocation

### 3.1. Budget Definition

```python
@dataclass
class ExplorationBudget:
    """
    Exploration budget for a skill.
    """
    skill_id: str
    budget_usd: float  # Total budget for this skill
    spent_usd: float   # Cumulative spend
    remaining_usd: float  # budget_usd - spent_usd

    # Allocation breakdown
    data_collection_pct: float  # % for new data collection
    compute_training_pct: float  # % for model training
    human_supervision_pct: float  # % for expert demonstrations

    # Derived allocations
    max_episodes: int  # remaining_usd / data_cost_per_episode
```

---

### 3.2. Budget Allocation Logic

```python
def allocate_exploration_budget(
    skill: Skill,
    total_budget_usd: float,
    mpl_gap: float
) -> ExplorationBudget:
    """
    Allocate exploration budget for a skill.

    Logic:
    - Larger MPL gap → larger budget
    - Lower success rate → more budget for data
    - Higher novelty exposure → more budget for diverse data
    """
    # Base allocation proportional to MPL gap
    base_budget = mpl_gap * 200.0  # $200 per unit MPL gap

    # Quality penalty: low success → need more training
    quality_penalty = 1.0 - skill.success_rate
    adjusted_budget = base_budget * (1.0 + quality_penalty)

    # Novelty bonus: high OOD exposure → more valuable
    novelty_bonus = skill.ood_exposure * 0.5
    final_budget = adjusted_budget * (1.0 + novelty_bonus)

    # Cap at total available budget
    final_budget = min(final_budget, total_budget_usd)

    # Allocate breakdown based on skill status
    if skill.status == SkillStatus.EXPLORATION.value:
        # Heavy data collection
        data_pct = 0.6
        compute_pct = 0.3
        human_pct = 0.1
    elif skill.status == SkillStatus.TRAINING.value:
        # Balanced
        data_pct = 0.4
        compute_pct = 0.5
        human_pct = 0.1
    else:  # MATURE
        # Minimal budget (maintenance only)
        data_pct = 0.2
        compute_pct = 0.7
        human_pct = 0.1

    max_episodes = int(final_budget * data_pct / skill.data_cost_per_episode)

    return ExplorationBudget(
        skill_id=skill.skill_id,
        budget_usd=final_budget,
        spent_usd=0.0,
        remaining_usd=final_budget,
        data_collection_pct=data_pct,
        compute_training_pct=compute_pct,
        human_supervision_pct=human_pct,
        max_episodes=max_episodes
    )
```

---

## 4. Returns Measurement

### 4.1. Economic Returns

```python
@dataclass
class SkillReturns:
    """
    Measured returns from a skill investment.
    """
    skill_id: str

    # Productivity Returns
    delta_mpl: float  # MPL improvement (units/hr)
    delta_mpl_pct: float  # Percentage improvement

    # Efficiency Returns
    delta_energy_wh: float  # Energy savings (Wh)
    delta_time_sec: float  # Time savings (sec/task)

    # Quality Returns
    delta_damage: float  # Damage cost reduction ($)
    delta_success_rate: float  # Success rate improvement

    # Exploration Returns
    delta_novelty_coverage: float  # New state-action coverage
    unique_failure_modes_discovered: int

    # Financial Returns
    roi_pct: float  # (returns - costs) / costs * 100
```

---

### 4.2. ROI Calculation

```python
def compute_skill_roi(
    skill: Skill,
    returns: SkillReturns,
    wage_target_usd_per_hr: float = 18.0
) -> float:
    """
    Compute return on investment for a skill.

    ROI = (Value Created - Training Cost) / Training Cost * 100

    Value Created = ΔMP product value + energy savings + damage savings
    """
    # Productivity value: ΔMP * price_per_unit * hours_deployed
    price_per_unit = 0.30  # Assume $0.30/unit (dishwashing example)
    hours_deployed = 1000  # Assume 1000 hours deployment horizon

    productivity_value = returns.delta_mpl * price_per_unit * hours_deployed

    # Efficiency savings
    energy_price_per_kwh = 0.12  # $0.12/kWh
    energy_savings = (returns.delta_energy_wh / 1000.0) * energy_price_per_kwh * hours_deployed

    # Quality savings
    damage_savings = abs(returns.delta_damage) * hours_deployed

    # Total value created
    total_value = productivity_value + energy_savings + damage_savings

    # ROI
    if skill.training_cost_usd > 0:
        roi_pct = (total_value - skill.training_cost_usd) / skill.training_cost_usd * 100.0
    else:
        roi_pct = 0.0

    return roi_pct
```

---

## 5. Economic Learner Loop

### 5.1. Learner Cycle

**The Economic Learner runs every N episodes (e.g., every 1000 episodes) to:**

1. **Measure Returns**: Compute SkillReturns for each active skill
2. **Compute ROI**: Determine which skills are profitable
3. **Reallocate Budget**: Shift budget from low-ROI to high-ROI skills
4. **Update Status**: Transition skills between exploration/training/mature
5. **Retire/Deprecate**: Mark low-performing skills as deprecated

---

### 5.2. Learner Implementation

```python
class EconomicLearner:
    """
    Manages skill portfolio and budget allocation.
    """

    def __init__(self, config):
        self.total_budget_usd = config["total_exploration_budget"]
        self.reallocation_period = config["reallocation_period_episodes"]  # e.g., 1000
        self.skills: Dict[str, Skill] = {}
        self.budgets: Dict[str, ExplorationBudget] = {}

    def run_cycle(self, episode_count: int):
        """
        Run one learner cycle.

        Steps:
        1. Measure returns for all skills
        2. Compute ROI
        3. Reallocate budgets
        4. Update skill statuses
        5. Generate market state report
        """
        if episode_count % self.reallocation_period != 0:
            return  # Not time for cycle yet

        # 1. Measure returns
        returns_by_skill = self._measure_all_returns()

        # 2. Compute ROI
        roi_by_skill = {
            skill_id: compute_skill_roi(skill, returns_by_skill[skill_id])
            for skill_id, skill in self.skills.items()
        }

        # 3. Reallocate budgets
        self._reallocate_budgets(roi_by_skill)

        # 4. Update statuses
        for skill_id, skill in self.skills.items():
            skill.status = update_skill_status(skill)

        # 5. Generate report
        self._generate_market_state_report(roi_by_skill)

    def _reallocate_budgets(self, roi_by_skill: Dict[str, float]):
        """
        Reallocate exploration budget based on ROI.

        Strategy:
        - High ROI skills: Increase budget by 20%
        - Low ROI skills: Decrease budget by 30%
        - Negative ROI skills: Consider deprecation
        """
        for skill_id, roi in roi_by_skill.items():
            skill = self.skills[skill_id]
            current_budget = self.budgets[skill_id].budget_usd

            if roi > 50.0:
                # High ROI: invest more
                new_budget = current_budget * 1.2
            elif roi > 0.0:
                # Positive ROI: maintain
                new_budget = current_budget
            else:
                # Negative ROI: reduce or deprecate
                new_budget = current_budget * 0.7

                if roi < -50.0:
                    skill.status = SkillStatus.DEPRECATED.value

            # Update budget
            mpl_gap = skill.mpl_target - skill.mpl_current
            self.budgets[skill_id] = allocate_exploration_budget(skill, new_budget, mpl_gap)
```

---

## 6. SIMA-2 + TFD + RECAP Integration

### 6.1. SIMA-2 Skill Quality Signals

**Inputs from SIMA-2**:
- `success_rate`, `recovery_rate`, `fragility_score` → Skill quality
- `ood_rate`, `failure_diversity` → Exploration value

```python
def update_skill_from_sima2(skill: Skill, sima2_signals: Dict):
    """
    Update skill stats from SIMA-2 quality signals.
    """
    skill.success_rate = sima2_signals["success_rate"]
    skill.recovery_rate = sima2_signals["recovery_rate"]
    skill.fragility_score = sima2_signals["fragility_score"]
    skill.ood_exposure = sima2_signals["ood_rate"]

    # Derived stats
    skill.failure_rate = 1.0 - skill.success_rate
```

---

### 6.2. TFD Skill Deployment

**TFD instruction → Skill activation**:

```python
def deploy_skill_from_tfd(instruction: str, skill_market: Dict[str, Skill]) -> Optional[str]:
    """
    Deploy a skill based on TFD instruction.

    Example: "Use the dishwashing skill" → skill_id = "dishwashing"
    """
    parsed = parse_instruction(instruction)

    if parsed.intent_type == InstructionType.DEPLOY_SKILL:
        skill_id = parsed.parameters.get("skill_id")

        if skill_id in skill_market:
            skill = skill_market[skill_id]

            if skill.status == SkillStatus.MATURE.value:
                return skill_id  # Activate mature skill
            else:
                log_warning(f"Skill {skill_id} is in {skill.status} status; may be unstable")
                return skill_id

    return None
```

---

### 6.3. RECAP Score Integration

**RECAP provides skill-level performance scores**:

```python
def update_skill_from_recap(skill: Skill, recap_score: float):
    """
    Update skill MPL based on RECAP evaluation.

    RECAP score ∈ [0, 1] → maps to MPL improvement estimate
    """
    # Assume RECAP score correlates with MPL
    # High RECAP score → skill is performing well → high MPL
    mpl_improvement = recap_score * 10.0  # Heuristic: score 0.9 → +9 units/hr

    skill.mpl_current = skill.mpl_baseline + mpl_improvement
```

---

## 7. Phase H Artifacts

### 7.1. Skill Market State (`results/phase_h/skill_market_state.json`)

```json
{
  "timestamp": "2025-11-23T14:30:00Z",
  "total_budget_usd": 10000.0,
  "allocated_usd": 7500.0,
  "skills": {
    "drawer_open_v2": {
      "skill_id": "drawer_open_v2",
      "status": "training",
      "mpl_current": 58.5,
      "mpl_target": 65.0,
      "mpl_gap": 6.5,
      "success_rate": 0.87,
      "roi_pct": 42.3,
      "budget_allocated_usd": 1300.0,
      "budget_spent_usd": 450.0,
      "budget_remaining_usd": 850.0
    },
    "dish_place_precision": {
      "skill_id": "dish_place_precision",
      "status": "mature",
      "mpl_current": 72.0,
      "mpl_target": 70.0,
      "mpl_gap": -2.0,
      "success_rate": 0.96,
      "roi_pct": 125.7,
      "budget_allocated_usd": 200.0,
      "budget_spent_usd": 1500.0,
      "budget_remaining_usd": -1300.0
    }
  }
}
```

---

### 7.2. Exploration Budget (`results/phase_h/exploration_budget.json`)

```json
{
  "total_budget_usd": 10000.0,
  "reallocation_period_episodes": 1000,
  "last_reallocation_episode": 5000,
  "budgets_by_skill": {
    "drawer_open_v2": {
      "budget_usd": 1300.0,
      "spent_usd": 450.0,
      "remaining_usd": 850.0,
      "data_collection_pct": 0.4,
      "compute_training_pct": 0.5,
      "human_supervision_pct": 0.1,
      "max_episodes": 2125
    }
  }
}
```

---

### 7.3. Skill Returns Report (`results/phase_h/skill_returns.json`)

```json
{
  "skill_id": "drawer_open_v2",
  "measurement_period": "2025-11-16 to 2025-11-23",
  "returns": {
    "delta_mpl": 5.2,
    "delta_mpl_pct": 9.7,
    "delta_energy_wh": -3.5,
    "delta_damage": -1.2,
    "delta_success_rate": 0.12,
    "unique_failure_modes_discovered": 4,
    "roi_pct": 42.3
  },
  "costs": {
    "training_cost_usd": 450.0,
    "data_cost_usd": 180.0,
    "compute_cost_usd": 225.0,
    "supervision_cost_usd": 45.0
  }
}
```

---

## 8. Module Consumption Rules (Advisory Boundaries)

### 8.1. Who Can Read Phase H Artifacts?

```python
ARTIFACT_CONSUMERS = {
    "skill_market_state.json": [
        "DataPackRLSampler",  # Sample based on skill training needs
        "TFD",  # Deploy mature skills
        "Dashboard/Reports"  # Display to operators
    ],
    "exploration_budget.json": [
        "DataCollectionOrchestrator",  # Decide which skills need more data
        "CurriculumManager",  # Prioritize skill training
        "EconomicLearner"  # Update allocations
    ],
    "skill_returns.json": [
        "EconomicLearner",  # Compute ROI
        "Dashboard/Reports"  # Show performance metrics
    ]
}
```

**Prohibited Consumers**:
- **EconController**: MUST NOT read skill_market_state (pricing is independent of skill training)
- **Reward Functions**: MUST NOT use skill ROI (rewards are task-based, not skill-based)
- **Ontology**: MUST NOT depend on budget allocations (ontology is descriptive, not economic)

---

### 8.2. Advisory-Only Policy

**Phase H artifacts are strictly advisory**:

```python
def use_skill_market_state(artifact_path: str, consumer: str):
    """
    Validate that consumer is allowed to use Phase H artifacts.

    Enforcement:
    - Artifacts influence sampling, curriculum, deployment decisions
    - Artifacts DO NOT modify rewards, pricing, task ordering
    """
    if consumer in ["EconController", "RewardFunction", "OntologyUpdateEngine"]:
        raise ValueError(f"{consumer} is prohibited from consuming Phase H artifacts")

    # Advisory use allowed
    return load_artifact(artifact_path)
```

---

## 9. Implementation Checklist

- [ ] Define `Skill` dataclass with economic attributes
- [ ] Implement `allocate_exploration_budget()` logic
- [ ] Implement `compute_skill_roi()` with productivity/efficiency/quality returns
- [ ] Implement `EconomicLearner.run_cycle()` with budget reallocation
- [ ] Wire `update_skill_from_sima2()` to consume SIMA-2 quality signals
- [ ] Wire `deploy_skill_from_tfd()` to handle TFD skill requests
- [ ] Generate `skill_market_state.json`, `exploration_budget.json`, `skill_returns.json` artifacts
- [ ] Add smoke tests for skill lifecycle transitions
- [ ] Enforce advisory boundaries (no EconController/Reward consumption)

---

**End of Phase H: Economic Learner & Skill Market**
