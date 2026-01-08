# DataPack Repository and Skill Tree Integration

## Overview

The DataPack system provides a **unified 2.0-energy schema** for robotics data valuation across Phase B and Phase C. It sits on top of:
- Episode summaries (MPL, error rate, EP, wage parity)
- Economic parameters (EconParams)
- Trust network (trust_net) - **FROZEN, read-only**
- Economic weight lattice (w_econ_lattice) - **FROZEN, read-only**
- λ-controller (synthetic budget) - **FROZEN, read-only**
- World model (horizon planning)
- HRL skill graph (6-skill drawer_vase)
- VLA transformer (language → skill plans)
- SIMA-2 co-agent (narrations)
- Energy breakdown (per-limb, per-skill, per-joint, per-effector)

**Schema Version**: `2.0-energy` (legacy) and `2.1-portable` (self-contained artifacts)

## Two-Bucket Taxonomy

### Positive Bucket
- **Definition**: Data that improved meta-objective J
- **ΔJ ≥ 0**: Episode performed better than baseline
- **Contains**: Successful skill executions, high-trust data, good economic weight
- **Use case**: Imitation learning, policy improvement, skill refinement

### Negative Bucket
- **Definition**: Data that worsened meta-objective J
- **ΔJ < 0**: Episode performed worse than baseline
- **Contains**: Failed executions with counterfactual plans
- **Use case**: Failure mode analysis, contrastive learning, safety training

## Core Components

### DataPackMeta Schema (Unified 2.0-Energy)
```python
@dataclass
class DataPackMeta:
    # Schema version (always "2.0-energy")
    schema_version: str = "2.0-energy"  # use "2.1-portable" when embedding portable artifacts

    # Identification
    pack_id: str                    # Unique identifier
    task_name: str                  # "drawer_vase", "dishwashing"
    env_type: str                   # Environment type
    brick_id: Optional[str]         # Data unit tracing ID
    bucket: Literal["positive", "negative"]

    # Tags
    semantic_tags: List[str]        # ["fragile", "offset_vase", etc.]
    energy_driver_tags: List[str]   # ["energy_driver:long_reach", ...]

    # Core profiles
    condition: ConditionProfile     # Environment conditions
    attribution: AttributionProfile # Impact metrics + gating signals
    energy: EnergyProfile           # Energy breakdown (2.0-energy fields)
    agent_profile: Dict[str, Any]   # Policy info

    # Skill trace
    skill_trace: List[Dict]         # Time-ordered skill usage
    episode_metrics: Dict[str, Any] # Raw EpisodeInfoSummary fields

    # Language annotations
    sima_annotation: Optional[SimaAnnotation]  # SIMA-2 narrations
    vla_plan: Optional[Dict[str, Any]]         # VLA skill plan

    # Counterfactual (for negative bucket)
    counterfactual_plan: Optional[Dict]
    counterfactual_source: Optional[str]
```

### EnergyProfile (2.0-Energy Fields)
```python
@dataclass
class EnergyProfile:
    total_Wh: float = 0.0
    Wh_per_unit: float = 0.0
    Wh_per_hour: float = 0.0

    # Hierarchical breakdown
    energy_per_limb: Dict[str, Dict[str, float]]      # {"shoulder": {"Wh": 0.5, ...}}
    energy_per_skill: Dict[str, Dict[str, float]]     # {"grasp": {"Wh": 0.2, ...}}
    energy_per_joint: Dict[str, Dict[str, float]]     # {"joint_0": {"Wh": 0.1, ...}}
    energy_per_effector: Dict[str, Dict[str, float]]  # {"gripper": {"Wh": 0.05, ...}}
    coordination_metrics: Dict[str, float]             # {"mean_active_joints": 2.5, ...}

    # Legacy compatibility
    limb_energy_Wh: Dict[str, float]
    skill_energy_Wh: Dict[str, float]
```

### ConditionProfile (Engine-Agnostic)
```python
@dataclass
class ConditionProfile:
    task_name: str = "drawer_vase"
    engine_type: Literal["pybullet", "isaac", "ue5"] = "pybullet"
    world_id: str = "pyb_drawer_v1"

    # Environment configuration
    vase_offset: tuple = (0.0, 0.0, 0.0)
    drawer_friction: float = 0.3
    lighting_profile: str = "normal"
    occlusion_level: float = 0.0

    # Economic regime
    econ_preset: str = "drawer_vase"
    price_per_unit: float = 5.0
    vase_break_cost: float = 50.0
    energy_price_kWh: float = 0.12

    # Objective vector
    objective_vector: List[float] = [1.0, 1.0, 1.0, 0.5]  # α_mpl, α_error, α_ep, α_safety

    tags: Dict[str, Any] = {}
```

### AttributionProfile (World Model Hooks)
```python
@dataclass
class AttributionProfile:
    # Core impact deltas
    delta_mpl: float = 0.0
    delta_error: float = 0.0
    delta_ep: float = 0.0
    delta_J: float = 0.0

    # Phase B gating signals
    trust_score: float = 0.0         # From trust_net
    w_econ: float = 0.0              # From J-trained lattice
    lambda_budget: float = 0.0       # Synthetic budget used

    # World model integration
    source_type: Literal["real", "synthetic", "hybrid"] = "real"
    wm_model_id: Optional[str] = None
    wm_horizon_used: Optional[int] = None
    wm_branch_depth: Optional[int] = None
    wm_trust_over_horizon: Optional[List[float]] = None

    # Marginal value-of-data
    mvd_score: Optional[float] = None
    econ_weight_final: Optional[float] = None  # trust * w_econ

    # Training tracking
    used_in_training_runs: List[str] = []
    wm_role: Optional[Literal["wm_train", "wm_eval", "wm_synth_source", "wm_synth_target"]] = None

    # Per-skill contributions
    skill_contribs: Dict[int, Dict[str, float]] = {}
```

## DataPackRepo API

### Storage
- JSONL backing store (one file per task)
- Append-only writes for data integrity
- In-memory caching for fast queries

### Query Methods
```python
repo = DataPackRepo(base_dir="data/datapacks")

# Append single datapack
repo.append(datapack)

# Append batch
repo.append_batch(datapacks)

# Portable Datapacks (2.1-portable)
Portable datapacks embed the minimal artifacts needed to run curated slices without raw rehydration.
They include `scene_tracks_v1`, `rgb_features_v1`, and `slice_labels_v1` alongside standard metadata.

Example JSONL entry (fields truncated):

```json
{
  "schema_version": "2.1-portable",
  "pack_id": "drawer_vase_scripted_0001",
  "task_name": "drawer_vase",
  "raw_data_path": null,
  "scene_tracks_v1": {
    "scene_tracks_v1/version": ["v1"],
    "scene_tracks_v1/poses_t": "...",
    "scene_tracks_v1/occlusion": "..."
  },
  "rgb_features_v1": {
    "encoder": "vision_rgb::deterministic_pool_v1",
    "dim": 64,
    "pooling": "mean",
    "stride_seconds": 1.0,
    "features": "..."
  },
  "slice_labels_v1": {
    "occlusion_level": 0.6,
    "is_occluded": true,
    "motion_score": 0.2,
    "is_dynamic": true,
    "is_static": false
  }
}
```

# Query with filters
results = repo.query(
    task_name="drawer_vase",
    bucket="positive",           # Filter by bucket
    skill_id=3,                  # Filter by skill usage
    engine_type="pybullet",      # Filter by engine
    min_trust=0.9,               # Minimum trust score
    min_delta_j=0.0,             # Minimum ΔJ
    min_mvd_score=0.5,           # Minimum MVD score
    source_type="real",          # Data source type
    limit=100,                   # Max results
    include_counterfactual=True, # Require counterfactual (for negative)
    include_sima=True,           # Require SIMA annotation
    sort_by="delta_j",           # Sort field
    sort_descending=True         # Sort order
)

# Convenience methods
positive = repo.get_positive_for_skill("drawer_vase", skill_id=4, top_k=10)
negative = repo.get_negative_for_skill("drawer_vase", skill_id=4, top_k=10)
stats = repo.get_statistics("drawer_vase")
```

### Statistics
```python
stats = {
    'total': 1000,
    'positive': 650,
    'negative': 350,
    'positive_ratio': 0.65,
    'delta_j_mean': 0.125,
    'delta_j_std': 0.35,
    'trust_mean': 0.92,
    'unique_skills': [0, 1, 2, 3, 4, 5],
    'engine_types': {'pybullet': 1000},
    'source_types': {'real': 800, 'synthetic': 200},
    'with_sima': 500,
    'with_counterfactual': 320
}
```

## Skill Tree Integration

### SkillDataPackAdapter
Bridges HRL skill graph with datapack repository:

```python
from src.hrl.skill_datapack_adapter import SkillDataPackAdapter

adapter = SkillDataPackAdapter(repo)

# Per-skill statistics
stats = adapter.get_skill_statistics("drawer_vase", skill_id=4)
# Returns: usage count, success rate, avg ΔJ, trust, w_econ, etc.

# All skills performance
all_stats = adapter.get_all_skills_performance("drawer_vase")

# Skill-specific training data
training_data = adapter.get_skill_training_data(
    "drawer_vase",
    skill_id=3,  # GRASP_HANDLE
    bucket="positive",
    min_trust=0.9,
    top_k=100
)

# Failure mode analysis
failures = adapter.get_skill_failure_modes("drawer_vase", skill_id=4, top_k=20)

# Sequence pattern analysis
patterns = adapter.get_skill_sequence_patterns("drawer_vase", bucket="positive")

# Contrastive pairs for training
pairs = adapter.extract_contrastive_pairs("drawer_vase", skill_id=4, n_pairs=50)

# Curriculum generation
curriculum = adapter.generate_skill_curriculum("drawer_vase", skill_id=4, n_levels=3)

# Value matrix: skill × condition
value_matrix = adapter.compute_skill_value_matrix("drawer_vase")

# Skill recommendations
recommendations = adapter.recommend_skill_for_condition(
    "drawer_vase",
    condition_tags=["offset_vase", "high_friction"]
)
```

### Skill IDs
```python
SkillID.LOCATE_DRAWER = 0
SkillID.LOCATE_VASE = 1
SkillID.PLAN_SAFE_APPROACH = 2
SkillID.GRASP_HANDLE = 3
SkillID.OPEN_WITH_CLEARANCE = 4
SkillID.RETRACT_SAFE = 5
```

## Episode-to-DataPack Pipeline

### Builder Script
```bash
# Generate synthetic episodes for testing
python scripts/build_datapacks_from_episodes.py --generate-synthetic 100

# Build from real episode data
python scripts/build_datapacks_from_episodes.py \
    --data-dir data/episodes \
    --output-dir data/datapacks \
    --task drawer_vase \
    --engine pybullet \
    --verbose

# Export portable datapacks (embed tracks/features/labels)
python -m scripts.export_portable_datapacks \
    --datapack-dir data/datapacks/phase_c \
    --task drawer_vase
```

### ΔJ Computation
```python
J = α_mpl * ΔMPL - α_error * Δerror + α_ep * ΔEP + α_safety * safety_bonus
```

Where:
- ΔMPL = (episode_mpl - baseline_mpl) / baseline_mpl
- Δerror = episode_error - baseline_error
- ΔEP = (episode_ep - baseline_ep) / baseline_ep
- safety_bonus = -10.0 if vase_broken else 0.0

## World Model Integration

### Source Types
- **real**: From actual environment execution
- **synthetic**: Generated by world model rollouts
- **hybrid**: Mix of real start + synthetic continuation

### WM Roles
- **wm_train**: Used to train world model
- **wm_eval**: Used to evaluate world model
- **wm_synth_source**: Source for synthetic data generation
- **wm_synth_target**: Target of synthetic data (generated)

### MVD Score (Marginal Value-of-Data)
Estimated economic value of the datapoint:
```python
mvd_score = trust_score * w_econ * expected_delta_j
```

Higher MVD → more valuable for training.

## SIMA-2 Integration

### SimaAnnotation Schema
```python
@dataclass
class SimaAnnotation:
    instruction: str = "open the drawer without hitting the vase"
    step_narrations: List[str] = []
    sima_agent_id: str = "sima_v1"
    source_world: str = "pyb_drawer_v1"
    derived_skill_plan: List[int] = [0, 1, 2, 3, 4, 5]
```

### Querying SIMA Data
```python
# Get datapacks with SIMA annotations
sima_packs = repo.query(
    task_name="drawer_vase",
    include_sima=True,
    limit=500
)

# Extract VLA training data
vla_samples = []
for dp in sima_packs:
    if dp.sima_annotation:
        vla_samples.append({
            'instruction': dp.sima_annotation.instruction,
            'skill_sequence': dp.get_skill_ids(),
            'narrations': dp.sima_annotation.step_narrations,
        })
```

## Multi-Engine Design

The system is engine-agnostic by design:

```python
# PyBullet (current)
condition = ConditionProfile(engine_type="pybullet", world_id="pyb_drawer_v1")

# Isaac Gym (future)
condition = ConditionProfile(engine_type="isaac", world_id="isaac_drawer_v1")

# Unreal Engine 5 (future)
condition = ConditionProfile(engine_type="ue5", world_id="ue5_drawer_v1")
```

Query by engine:
```python
isaac_packs = repo.query(task_name="drawer_vase", engine_type="isaac")
```

## Typical Workflows

### 1. Collect and Classify Data
```python
# Run episodes
episodes = run_training_episodes(env, policy, n=1000)

# Build datapacks with automatic bucketing
datapacks = build_datapacks_from_episodes(episodes, econ, baseline)

# Store
repo.append_batch(datapacks)
```

### 2. Train Skill Policy
```python
# Get positive training data for GRASP_HANDLE
adapter = SkillDataPackAdapter(repo)
training_data = adapter.get_skill_training_data(
    "drawer_vase",
    skill_id=SkillID.GRASP_HANDLE,
    bucket="positive",
    min_trust=0.9
)

# Train with high-trust positive examples
train_skill_policy(skill_id=3, data=training_data)
```

### 3. Analyze Failures
```python
# Find worst failures for OPEN_WITH_CLEARANCE
failures = adapter.get_skill_failure_modes("drawer_vase", skill_id=4)

# Extract counterfactual plans
for failure in failures:
    print(f"Failed: {failure['semantic_tags']}")
    print(f"Counterfactual: {failure['counterfactual']}")
```

### 4. Curriculum Learning
```python
curriculum = adapter.generate_skill_curriculum("drawer_vase", skill_id=4, n_levels=3)

for level in curriculum:
    print(f"Level {level['level']}: {level['n_samples']} samples")
    print(f"  Difficulty: {level['difficulty']:.2f}")
    print(f"  Avg offset: {level['conditions_summary']['avg_offset']:.3f}")
    train_on_level(level['datapacks'])
```

### 5. World Model Training
```python
# Get datapacks for world model training
wm_train_packs = repo.query(
    task_name="drawer_vase",
    source_type="real",
    min_trust=0.95,
    limit=1000
)

# Mark as used
for dp in wm_train_packs:
    dp.attribution.wm_role = "wm_train"
    dp.attribution.used_in_training_runs.append("wm_run_001")

# Generate synthetic data
synthetic_packs = world_model.generate_rollouts(wm_train_packs)
for dp in synthetic_packs:
    dp.attribution.source_type = "synthetic"
    dp.attribution.wm_model_id = "wm_v1"
    dp.attribution.wm_role = "wm_synth_target"

repo.append_batch(synthetic_packs)
```

## Phase B/C Datapack Flow

### Phase B (Dishwashing) → DataPackRepo
```python
# In Phase B evaluation scripts
from src.valuation.datapacks import build_datapack_meta_from_episode
from src.valuation.datapack_repo import DataPackRepo

repo = DataPackRepo(base_dir="data/datapacks/phase_b")

# After each episode
summary = env.get_episode_summary()
dp = build_datapack_meta_from_episode(
    summary, econ_params,
    condition_profile={"task": "dishwashing", "tags": []},
    agent_profile={"policy": "ppo_baseline"},
    env_type="dishwashing",
    baseline_mpl=baseline_mpl,
    baseline_error=baseline_error,
    baseline_ep=baseline_ep
)
repo.append(dp)
```

### Phase C (Drawer+Vase HRL/VLA/SIMA) → DataPackRepo
```python
# In Phase C evaluation scripts (e.g., eval_drawer_vase_scripted.py)
from src.valuation.datapacks import build_datapack_meta_from_episode
from src.valuation.datapack_repo import DataPackRepo

repo = DataPackRepo(base_dir="data/datapacks/phase_c")

# After each episode
summary = summarize_drawer_vase_episode(info_history)
dp = build_datapack_meta_from_episode(
    summary, econ_params,
    condition_profile={"task": "drawer_vase", "engine_type": "pybullet"},
    agent_profile={"policy": "scripted"},
    env_type="drawer_vase",
    skill_trace=[...],                    # HRL skill trace
    sima_annotation=sima_annotation,      # SIMA language
    vla_plan=vla_plan,                    # VLA skill plan
    baseline_mpl=baseline_mpl,
    baseline_error=baseline_error,
    baseline_ep=baseline_ep
)
repo.append(dp)
```

### Scripts Writing to DataPackRepo
- `scripts/eval_drawer_vase_scripted.py` → `data/datapacks/phase_c/`
- `scripts/eval_feifei_benchmark.py` → `data/datapacks/phase_c/`
- `scripts/smoke_test_phase_c_hrl_vla.py` → `data/datapacks/phase_c/`
- Phase B training scripts → `data/datapacks/phase_b/`

## Directory Structure
```
data/datapacks/
├── phase_b/
│   ├── dishwashing_datapacks.jsonl
│   └── bricklaying_datapacks.jsonl
├── phase_c/
│   ├── drawer_vase_datapacks.jsonl
│   └── isaac_drawer_datapacks.jsonl   # Future
└── test/
    └── test_datapacks.jsonl

src/valuation/
├── datapack_schema.py             # Core schemas (unified 2.0-energy)
├── datapack_repo.py               # Repository with query API
├── datapacks.py                   # Build functions (legacy + unified)
├── energy_tags.py                 # Energy driver tag inference
└── __init__.py                    # Exports

src/hrl/
├── skill_datapack_adapter.py      # Skill ↔ DataPack bridge
└── skills.py                      # Skill definitions

scripts/
├── test_datapack_repo_and_adapter.py  # Comprehensive test suite
├── eval_drawer_vase_scripted.py       # Now writes to DataPackRepo
└── build_datapacks_from_episodes.py   # Episode converter
```

## Key Invariants

1. **Positive ΔJ → Positive Bucket**: Automatic classification based on metric improvement
2. **Negative Bucket ⇒ Counterfactual**: All negative datapacks must have counterfactual plans
3. **Trust Score Gating**: High-trust data (≥0.9) prioritized for training
4. **Engine Agnostic**: Same schema works across PyBullet, Isaac Gym, UE5
5. **Append-Only**: JSONL storage is append-only for data integrity
6. **Skill Trace Required**: Every datapack has time-ordered skill trace

## Future Extensions

1. **Isaac Gym Integration**: Higher throughput simulation
2. **UE5 Photo-Realistic**: Visual domain transfer
3. **Active Learning**: Query datapacks by uncertainty
4. **Federated DataPacks**: Cross-deployment aggregation
5. **Differential Privacy**: Privacy-preserving data sharing
6. **Causal Discovery**: Learn skill → outcome causality from datapacks
