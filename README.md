# robotics-v-p-core — Economics-First Robotics Stack

> Robots priced like labor, not software. Full-stack econ + data valuation for video-to-policy, HRL, VLA, and synthetic flywheels.

## 1) Project Overview
This repo implements a robotics economics and learning engine that:
- Learns manipulation from video → latent → policy (SAC / HRL / VLA)
- Models marginal productivity of labor (MPL) and economics (wage parity, energy efficiency)
- Generates and prices synthetic data with a closed-loop world model + trust/econ/λ gating
- Anchors data valuation (ΔMPL/Δerror/ΔEP) into RL, offline RL, and datapacks
- Provides scaffolding for HRL skill graphs, transformer planners, and SIMA-style co-agents

Target flywheel: better data → better robots → better economics → better data.

## 2) Motivation & Teleology
**Why economics:** Deployment is constrained by MPL, wage parity, energy efficiency, error SLAs, and data value—not just task success.  
**Why multi-layered reward:** Real factories need MPL, EP, error, wage parity, safety, and meta-value (ΔMPL, novelty) simultaneously.  
**Why data valuation:** Data is capital. We compute how each episode shifts MPL/error/EP and gate synthetic via trust × w_econ × λ_budget to price data, build datapacks, and control synthetic budgets.

## 3) Layered Architecture (Phase A → B → C)
- **Phase A (Calibration):** Dishwashing physics + aligned encoder z_V, novelty/ΔMPL estimators, stable dynamics, wage/MPL reward shaping.
- **Phase B (Frozen Synthetic Flywheel):** Contractive world model, trust_net, J-trained w_econ lattice, λ budget controller, SyntheticWeightController (trust × econ quality; λ as budget), local synthetic branches, data bricks. Frozen and validated.
- **Phase C (Generalization Stack):** Drawer+Vase PyBullet env, HRL skill graph (π_H/π_L), vision affordance heads (risk/no-go/fragility), VLA transformer planner, SIMA co-agent/narrator, Fei-Fei benchmark scaffolding.

## Current Status (Short)
- Phase B frozen: world model, trust_net, w_econ_lattice, λ controller, SyntheticWeightController — do not change math.
- Phase C scaffolding: HRL/VLA/SIMA for drawer+vase with datapack hooks.
- Energy bench: experimental articulated-arm envs (`dishwashing_arm_env`, `drawer_vase_arm_env`) plus energy interventions/analysis; additive only, no changes to Phase B weighting or rewards.
- Orchestrator: advisory meta-planner that proposes tool sequences/run specs (energy profiles, data mix, backend), annotates datapacks via GuidanceProfile, and emits structured diffusion requests; analysis-only, no reward/weight changes.
- Datapacks: two-bucket taxonomy with ObjectiveProfile and optional GuidanceProfile; diffusion request/ingest scaffold connects synthetic video generation back into datapacks/valuation.

## 4) Data Valuation Loop (Heart of the System)
```
Episode → EpisodeInfoSummary
        → Episode features (ΔMPL, Δerror, ΔEP, novelty, trust, brick_id…)
        → w_econ (lattice, J-trained)
        → λ_budget (global synthetic share)
        → w_quality = trust × w_econ
        → w_final = scale_to_budget(w_quality, λ, max_synth_share)
```
Rules: trust gate for plausibility, econ weighting for impact, λ for global budget. Synthetic is admitted only if safe, valuable, and within budget.

## 5) Core Modules & Files
- **Econ config:** `src/config/econ_params.py`, `src/config/internal_profile.py`
- **Env + summaries:** `src/envs/dishwashing_env.py`, `EpisodeInfoSummary`, `summarize_episode_info`
- **Reward shaping:** `src/rl/reward_shaping.py` (MPL/EP/error + wage penalty hook)
- **Synthetic weighting:** `src/controllers/synthetic_weight_controller.py` (trust/econ quality + λ budget), `src/controllers/synth_lambda_controller.py`, `src/valuation/w_econ_lattice.py`
- **World model trust:** `src/world_model/...`, `checkpoints/stable_world_model.pt` (frozen, do not alter math)
- **Phase C env/HRL/VLA:** `src/envs/drawer_vase_physics_env.py`, `src/hrl/*`, `src/vision/*`, `src/vla/*`, `src/sima/*`
- **Embodiment:** `src/embodiment/*` (contacts/affordances + econ attribution), `docs/embodiment_module.md`
- **Datapacks:** `src/valuation/datapacks.py` → build datapacks from `EpisodeInfoSummary`
- **Utilities/Smoke tests:** `scripts/smoke_test_dishwashing_sac.py`, `scripts/test_episode_features.py`, `scripts/eval_drawer_vase_scripted.py`, `scripts/smoke_test_phase_c_hrl_vla.py`, `docs/synthetic_weight_controller_design.md`

## 6) Training & Evaluation (Phase B smoke path)
```bash
# Feature extractor sanity
python3 scripts/test_episode_features.py

# Dishwashing smoke + summaries (JSON/CSV)
python3 scripts/smoke_test_dishwashing_sac.py --episodes 5 --econ-preset toy --out-json tmp_smoke.json

# SAC (aligned physics config)
python3 train_sac_v2.py configs/dishwashing_physics_aligned.yaml --episodes 20 --econ-preset toy

# Novelty/ΔMPL validation + zV dataset + WM training/eval
python3 scripts/validate_dmpl_novelty.py --config configs/dishwashing_physics_aligned.yaml --episodes 30
python3 scripts/collect_zv_dataset.py --config configs/dishwashing_physics_aligned.yaml --episodes 30
python3 scripts/train_stable_world_model.py --epochs 20
python3 scripts/eval_world_model_rollouts.py --world-model checkpoints/stable_world_model.pt

# Synthetic branches + offline A/B
python3 scripts/collect_local_synthetic_branches.py --horizon 10
python3 scripts/train_offline_with_local_synth.py --synth-weight 0.1
# 4-mode A/B: baseline, trust, trust+econ, trust+econ+λ
python3 scripts/run_4mode_synth_ab_test.py
```

## 7) Phase C Plumbing
```bash
# Drawer+Vase scripted eval + datapacks
python3 scripts/eval_drawer_vase_scripted.py --episodes 5 --emit-datapacks data/phase_c_datapacks/scripted.json

# HRL/VLA smoke (imports + episodes + datapacks)
python3 scripts/smoke_test_phase_c_hrl_vla.py --episodes 3 --out-datapacks data/phase_c_datapacks/smoke.json
```

## Dev Workflow
```bash
# Create venv
python3 -m venv .venv && .venv/bin/python -m pip install -U pip && .venv/bin/python -m pip install -e .

# Sanity check
python -m scripts.doctor

# Fast tests
.venv/bin/python -m pytest -q -m "not slow"

# Slow tests
.venv/bin/python -m pytest -m slow -v
```

## 7.1) Workcell Manufacturing Suite (Blue-Collar Environments)

The workcell environment suite provides modular manufacturing task environments for kitting, assembly, inspection, and conveyor operations.

### Quick Start

```bash
# Run smoke tests
python3 scripts/smoke_workcell_env.py

# Run via motor backend factory
python3 -c "
from src.motor_backend import make_motor_backend
backend = make_motor_backend('workcell_env')
result = backend.train_policy(episodes=5, config={'task_type': 'kitting'})
print(f'Training result: {result.metrics}')
"

# Promptable task generation (orchestrator path)
python3 -c "
from src.orchestrator.workcell_adapter import WorkcellOrchestrationAdapter
adapter = WorkcellOrchestrationAdapter()
result = adapter.request_task('Pack 6 bolts into a tray with 2mm tolerance')
print(f'Generated task: {result.inferred_task_type}, nodes: {len(result.task_graph.nodes)}')
"

# Export datapack from workcell episodes
python3 -c "
from src.motor_backend.workcell_env_backend import WorkcellEnvBackend
backend = WorkcellEnvBackend()
backend.train_policy(episodes=10)
datapack = backend.export_datapack()
print(f'Datapack episodes: {len(datapack[\"episodes\"])}')
"

# Reconstruct workcell from SceneTracks (video-to-env replay)
python3 -c "
from src.envs.workcell_env.reconstruction.scene_tracks_adapter import SceneTracksAdapter
import numpy as np

# Load your SceneTracks_v1 npz
# tracks = np.load('path/to/scene_tracks.npz')
# adapter = SceneTracksAdapter()
# result = adapter.reconstruct_from_tracks(dict(tracks))
# print(f'Reconstructed: {len(result.scene_spec.parts)} parts')
"
```

### Available Tasks

| Task Type | Description | Config Key |
|-----------|-------------|------------|
| `kitting` | Pack items into trays/boxes | `PRESETS["assembly_bench_simple"]` |
| `peg_in_hole` | Precision insertion | tolerance_mm parameter |
| `bin_picking` | Pick from cluttered bins | occlusion_level parameter |
| `conveyor_sorting` | Sort items on moving belt | `PRESETS["conveyor_sorting"]` |
| `assembly` | Multi-step assembly with fasteners | tool_changes_required |
| `inspection` | Visual defect detection | `PRESETS["inspection_simple"]` |

### Key Files

- `src/envs/workcell_env/` — Core environment module
- `src/motor_backend/workcell_env_backend.py` — Motor backend integration
- `src/orchestrator/workcell_adapter.py` — Promptable task compiler
- `src/analytics/workcell_analytics.py` — Episode metrics and reports
- `docs/workcell_ontology.md` — Entity schema (Station, Fixture, Part, Tool, Container)
- `THIRD_PARTY_NOTICES.md` — License attribution for referenced patterns

## 8) Design Principles
- **Economics-first:** MPL, wage parity, EP, error, safety, and data value drive rewards and gating.
- **Multi-gating:** trust × w_econ for quality; λ for budget. Never stack λ as another gate.
- **Stability:** contractive WM; trust_net gate; J-trained w_econ; EpisodeInfoSummary as single truth.
- **Portability:** PyBullet today for speed; interfaces kept simple for future Isaac Gym + real robot ports.
- **Data as capital:** datapacks carry attribution (ΔMPL/Δerror/ΔEP, trust, econ weight, novelty) for pricing and budgeting.

## 9) Roadmap (what’s next for Codex/Claude)
- **Energy economics everywhere:** Wh/unit and Wh/hr into reward shaping, datapacks, WM eval, econ weighting, λ budgeting, VLA conditioning.
- **Distributional metrics in datapacks:** realism/variance gaps (real vs synthetic), fragility/safety footprint, HRL skill impact.
- **Phase C → B feedback:** funnel Drawer+Vase HRL/VLA/SIMA datapacks into Phase B valuation and synthetic weighting.
- **Env interface unification:** BaseEnv stub to align PyBullet and future Isaac Gym wrappers.
- **Video→latent validation:** scripts for encoding external video frames and checking latent distribution match.

## 10) Warnings & Frozen Zones
- Phase B core is frozen: do **not** change WM architecture/training, trust_net, w_econ lattice objective, λ controller objective, or synthetic A/B math. Only wire/consume them.
- Checkpoints and artifacts are large; avoid committing binaries.

## 11) Pointers to Docs
- `docs/synthetic_weight_controller_design.md` — mother-module spec (trust × econ + λ budget)
- `docs/drawer_vase_env.md`, `docs/phase_c_hrl_vla_architecture.md` — Phase C env/HRL/VLA
- `docs/ECON_ARCHITECTURE.md`, `docs/V2P_TECHNICAL_OVERVIEW.md` — economics rationale and V2P pipeline

## 12) Flywheel Summary
1) Collect real/video/physics data → encode to latent  
2) World model + trust gate → synthetic branches (local, stable)  
3) Econ valuation: w_econ lattice (J-trained) + trust → w_quality  
4) λ controller sets global synthetic budget → SyntheticWeightController scales w_quality  
5) RL/offline RL trains with economically gated data → higher MPL, lower error/Wh  
6) EpisodeInfoSummary + datapacks capture impact → price data, update λ/w_econ, refresh world model  
7) Repeat; port to Isaac Gym + real robots when ready.

# Update with actual ΔMPL (incremental learning)
estimator.update(novelty, actual_delta_mpl)

# Use for spread allocation
s_cust = delta_mpl_pred / delta_mpl_total
```

Replaces heuristics with learned data value model.

## File Structure

```
robotics v-p economics model/
├── README.md                          # This file
├── ECON_ARCHITECTURE.md               # Economic design
├── V2P_TECHNICAL_OVERVIEW.md          # Technical implementation
├── INVESTOR_STORY.md                  # Investor pitch
├── DEEP_LEARNING_ARCHITECTURE.md      # SAC + encoder details
├── FEASIBILITY_FIX_SUMMARY.md         # V1→V2 transition
│
├── train_sac.py                       # Main training script
├── configs/
│   └── dishwashing_feasible.yaml      # Task configuration
│
├── src/
│   ├── envs/
│   │   └── dishwashing_env.py         # 2D action environment
│   ├── encoders/
│   │   └── mlp_encoder.py             # MLP encoder + aux heads
│   ├── rl/
│   │   └── sac.py                     # Complete SAC implementation
│   ├── economics/
│   │   ├── mpl.py                     # Marginal product calculation
│   │   ├── wage.py                    # Implied wage calculation
│   │   ├── reward.py                  # Lagrangian reward
│   │   ├── spread_allocation.py       # Mechanistic split
│   │   └── data_value.py              # Online ΔMPL estimator
│   └── utils/
│       └── logger.py                  # CSV logging
│
├── experiments/
│   ├── sweep_frontier.py              # Feasibility validation
│   ├── elasticity_curve.py            # Economic generalization
│   ├── plot_spread_allocation.py      # Spread visualization
│   └── summary_snapshots.py           # Quantitative validation
│
├── logs/
│   └── sac_train.csv                  # Training metrics (38 columns)
├── checkpoints/
│   └── sac_final.pt                   # Trained model
└── plots/
    └── spread_allocation.png          # Spread analysis
```

## Economic Metrics (Logged Per Episode)

**Performance:**
- `mp_r` - Marginal product (units/hr)
- `err_rate` - Error rate (fraction)
- `err_target` - SLA target (curriculum)

**Economics:**
- `w_hat_r` - Robot implied wage ($/hr)
- `w_h` - Human wage benchmark ($/hr)
- `wage_parity` - ŵᵣ/wₕ (viability metric)
- `profit` - Revenue - costs ($/hr)
- `lambda_dual` - Shadow price of quality

**Data Value:**
- `novelty_raw` - Raw novelty score (from SAC)
- `delta_mpl_cust_pred` - Predicted ΔMPL from estimator
- `delta_mpl_total` - Actual ΔMPL (units/hr)

**Spread Allocation:**
- `spread` - ŵᵣ - wₕ ($/hr)
- `spread_value` - spread × hours ($)
- `s_cust` - Customer contribution share [0,1]
- `s_plat` - Platform contribution share [0,1]
- `rebate` - Customer rebate ($)
- `captured_spread` - Platform captured ($)

## Development Workflow

### 1. Train Agent
```bash
python3 train_sac.py 250  # 250 episodes (~5 min)
```

### 2. Validate Economics
```bash
python3 experiments/summary_snapshots.py
# Outputs: Platform revenue, customer savings, unit economics
```

### 3. Visualize Spread
```bash
python3 experiments/plot_spread_allocation.py
# Outputs: plots/spread_allocation.png
```

### 4. Check Feasibility
```bash
python3 experiments/sweep_frontier.py
# Validates SLA is achievable
```

## Unit Economics (1000 Robots)

**Platform Revenue:**
- Spread: $9.11/hr (robot vs human wage)
- Platform share: 65.1%
- Revenue per robot-year: $5,926/year
- **Total: $5.93M/year**

**Platform Costs:**
- Compute + hardware + support: $1,700/robot/year
- **Total: $1.70M/year**

**Gross Profit: $4.23M/year (71% margin)**

**Customer Economics:**
- Human wage: $18/hr
- Rebate: $3.71/hr (from data contribution)
- Net cost: $14.29/hr
- **Savings: 20.6% vs human**

*From `experiments/summary_snapshot.txt` (250-episode run)*

## Why This Enables Video-to-Policy

### Modular Encoder Interface

**Current (Sim):**
```python
state (4D) → MLP encoder → latent (128D) → SAC
```

**Future (Video):**
```python
video (T×H×W×3) → Video encoder → latent (128D) → SAC
                                      ↑
                               (same interface!)
```

**No changes needed to:**
- SAC agent (actor, critics, replay)
- Economic rewards (profit, wage parity)
- Lagrangian constraint (λ dual ascent)
- Spread allocation (ΔMPL-based split)

**Only swap:** `MLPEncoder` → `VideoEncoder` (R3D-18 or TimeSformer-B)

## Why Investors Care

### Predictable Cash Flows
- **Not** licensing deals (unpredictable)
- **Not** per-unit sales (one-time)
- **Is** recurring $/hr revenue (labor pricing)

### Securitizable
- Spread mechanistically tied to labor market
- Churn measurable (robot uptime vs labor availability)
- s_plat tracked per episode (attribution clear)

### Defensible
- Data flywheel: share → rebates → more sharing
- Economic lock-in: customers profit from contributing
- First-mover advantage in data collection

**See [INVESTOR_STORY.md](INVESTOR_STORY.md) for full pitch.**

## Key References

**Labor Economics:**
- Human wage benchmarks from BLS (Bureau of Labor Statistics)
- Marginal product theory (MPL = units/hour)
- Implied wage: ŵ = p·MP - c·error·MP

**Deep RL:**
- SAC (Haarnoja et al., 2018): Off-policy, entropy-regularized
- Twin critics (Fujimoto et al., 2018): Reduce Q-overestimation
- Prioritized replay (Schaul et al., 2015): Novelty-weighted sampling

**Video Encoders:**
- R3D-18 (Tran et al., 2018): 3D ResNet for video
- TimeSformer (Bertasius et al., 2021): Temporal transformer

## License

Proprietary - All rights reserved.

## Contact

For investment inquiries or technical questions, see documentation or create an issue.

---

**Status:** Simulation validated, ready for video integration
**Performance:** 1.65× wage parity, 2.7% error, $5.93/hr revenue
**Next:** Deploy 10 pilot robots, integrate video encoder
