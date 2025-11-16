# Video-to-Policy Economics Model

**Robots priced like labor, not software.**

This project grounds robot learning in labor economics, enabling predictable cash flows, mechanistic data pricing, and securitizable revenue streams.

## Core Innovation

Traditional robotics optimizes for task completion. This project optimizes for **economic viability**:

1. **Wage Parity:** Robot policies converge to human wage benchmarks (ŵᵣ/wₕ)
2. **Data Pricing:** Training data valued by expected MPL improvement (E[ΔMPL])
3. **Spread Allocation:** Value split mechanistically based on causal contribution

**Result:** $5.93/hr platform revenue per robot, 71% gross margins, securitizable cash flows.

## Quick Start

```bash
# Train SAC agent with economic objectives (250 episodes)
python3 train_sac.py 250

# Generate quantitative summary
python3 experiments/summary_snapshots.py

# Plot spread allocation
python3 experiments/plot_spread_allocation.py

# Run feasibility sweep
python3 experiments/sweep_frontier.py
```

## Documentation

### Core Concepts
- **[ECON_ARCHITECTURE.md](ECON_ARCHITECTURE.md)** - Why economics first, Lagrangian constraints, spread allocation
- **[V2P_TECHNICAL_OVERVIEW.md](V2P_TECHNICAL_OVERVIEW.md)** - SAC implementation, encoder design, video pathway
- **[INVESTOR_STORY.md](INVESTOR_STORY.md)** - Unit economics, cash flow model, securitization

### Implementation Guides
- **[DEEP_LEARNING_ARCHITECTURE.md](DEEP_LEARNING_ARCHITECTURE.md)** - SAC + encoder details (V3 architecture)
- **[FEASIBILITY_FIX_SUMMARY.md](FEASIBILITY_FIX_SUMMARY.md)** - How 2D actions made SLA feasible (V1→V2)
- **[CLAUDE.md](CLAUDE.md)** - Development guide (for Claude Code)

## Architecture Overview

```
Video Frames → Encoder (R3D-18) → Latent (128D) → SAC Agent
                                                      ↓
                                                   Actions (speed, care)
                                                      ↓
                                                   Environment
                                                      ↓
                        Economics (MP, wage, profit, spread)
                                                      ↓
                        Spread Allocation (mechanistic split)
                                                      ↓
                        Cash Flows (rebate, captured)
```

### Current State (V3)
- ✅ Simulation environment with 2D action space (speed, care)
- ✅ MLP encoder (128D latent)
- ✅ SAC agent with novelty-weighted replay
- ✅ Lagrangian constraint for quality SLA
- ✅ Mechanistic spread allocation (ΔMPL-based)
- ✅ Online data value estimator (novelty → ΔMPL)
- ✅ Economic logging (profit, wage parity, spread)

### Next Steps
- [ ] Video encoder integration (R3D-18 / TimeSformer-B)
- [ ] Real diffusion novelty (Stable-Video-Diffusion)
- [ ] Pilot deployments (10 robots)
- [ ] Expand to 2nd task (bricklaying)

## Results (100-Episode Test)

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **MP** | 109/h | 80/h | ✅ +36% |
| **Error Rate** | 2.7% | 6.0% | ✅ (2.2× margin) |
| **Wage Parity** | 1.65 | 1.0 | ✅ +65% |
| **Profit** | $29.68/hr | $18/hr | ✅ +65% |
| **Platform Capture** | $5.93/hr | - | ✅ 65% of spread |
| **Customer Savings** | 20.6% | - | ✅ vs human wage |
| **Gross Margin** | 71.3% | - | ✅ Software-first |

*From `experiments/summary_snapshot.txt`*

## Key Technical Decisions

### 1. Why 2D Action Space?
**1D (speed only)** → Error rate 17%, infeasible SLA

**2D (speed, care)** → Error rate 2.7%, 1,615 viable operating points

**Reason:** Task feasibility requires controllable error-throughput tradeoff.

### 2. Why Lagrangian Constraints?
Simple reward `r = α·MP - β·error` can be gamed.

Lagrangian `r = profit - λ·max(0, err - e*)` enforces hard SLA:
- λ = 0 when SLA met
- λ increases when SLA violated
- Automatic quality guarantee

### 3. Why Mechanistic Spread Allocation?
Arbitrary splits ("60% customer, 40% platform") lack justification.

Causal split `s_cust = ΔMPL_cust / ΔMPL_total`:
- Customer gets share proportional to their data's contribution
- Platform gets share proportional to foundation models
- Transparent, auditable, incentive-aligned

### 4. Why Online ΔMPL Estimator?
Direct path: **Novelty → E[ΔMPL] → Price**

```python
# Predict ΔMPL from novelty
delta_mpl_pred = estimator.predict(novelty)

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
