# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Video-to-Policy (v-p) Economics Model** for humanoid robotics that operates on two economic layers:

1. **Training Economics**: Robot policies converge to human wage parity via MPL-grounded rewards
2. **Data Economics**: Each datapoint has quantifiable value based on expected MPL inflection, with tiered pricing based on novelty

**Core innovation**: A feedback-priced learning ecosystem where task data has explicit market value, MPL growth is a measurable financial asset, and pricing incentives align data sharing with productivity gains.

## Economic Model

### Layer 1: Training Economics (Robot Wage Convergence)

#### Definitions

- **p**: Price per unit output ($/dish, $/brick, etc.)
- **wₕ**: National average wage for human role ($/hour)
- **MPₕ**: Human marginal product (units/hour)
- **MPᵣ(t)**: Robot marginal product at episode t (units/hour)
- **e(t)**: Robot error rate at episode t (fraction of units with errors)
- **cₐ**: Damage cost per error ($)

#### Core Relationships

**Human wage-productivity equilibrium** (competitive benchmark):
```
wₕ ≈ p · MPₕ
```

**Robot implied wage** at episode t:
```
ŵᵣ(t) = p · MPᵣ(t) - cₐ · e(t)
```

**Parity metrics**:
```
Productivity Parity = MPᵣ(t) / MPₕ
Wage Parity = ŵᵣ(t) / wₕ
```

#### Training Objective (Reward Shaping)

```
r(t) = α · ΔMPᵣ(t) - β · e(t) - γ · |ŵᵣ(t)/wₕ - 1|
       └─────────┘   └──────┘   └──────────────────┘
       learn faster  be precise  converge to human wage
```

Where `ΔMPᵣ(t) = MPᵣ(t) - MPᵣ(t-1)`

**Normalized version** (for stability):
```
ΔMPᵣ(t) ← (MPᵣ(t) - MPᵣ(t-1)) / MPₕ
```

**Alternative: Smooth squared penalty**:
```
WageLoss(t) = (ŵᵣ(t)/wₕ - 1)²
r(t) = α · ΔMPᵣ(t) - β · e(t) - γ · WageLoss(t)
```

#### Numeric Example: Dishwashing

```
Task: Dishwashing
p = $0.30/dish
MPₕ = 60 dishes/hr
wₕ ≈ $18/hr

Robot at episode t:
MPᵣ(t) = 54 dishes/hr
e(t) = 0.05 (5% broken)
cₐ = $1/broken dish

ŵᵣ(t) = 0.30 · 54 - 1 · (0.05 · 54)
       = 16.2 - 2.7
       = $13.5/hr

Wage Parity = 13.5/18 = 0.75
→ Economic penalty triggers; policy should improve speed/precision
```

### Layer 2: Data Economics (Datapoint Valuation via MPL Inflection)

#### Core Idea

Every datapoint (demonstration, correction, runtime feedback) has **quantifiable economic value** based on how much it reduces policy uncertainty and increases expected MPL.

**Marginal value of a datapoint**:
```
Vₐₐₜₐ,ᵢ = ΔE[MPLᵣ]ᵢ · p
```

where `ΔE[MPLᵣ]ᵢ` is the **expected improvement in robot marginal product** attributable to datapoint i.

#### Pricing Formula for Non-Sharing Customers

If a customer opts out of data sharing, they pay an **opportunity cost premium**:

```
Pₙₒₛₕₐᵣₑ = p · κ · E[ΔMPLᵢ] · H · S
```

Where:
- **p**: Price per unit output
- **κ**: Confidence discount (lower_confidence_bound / mean ratio)
- **E[ΔMPLᵢ]**: Expected MPL inflection from withheld datapoint (units/hour)
- **H**: Horizon (robot-hours over which MPL gain applies)
- **S**: Sharing coefficient (0 = full share, 1 = no share)

#### Estimating E[ΔMPLᵢ] via Novelty Metrics

Use **novelty_score** and **causal_gain** to estimate expected MPL improvement:

**novelty_score**: Measures how "different" this datapoint is from existing training data
- Embedding distance, coverage in state-action space, etc.

**causal_gain**: Measures how much this datapoint would improve policy performance
- Expected information gain, prediction error reduction, etc.

**Implementation**:
- Use **rolling regression** or **Bayesian updater** to map (novelty, causal_gain) → realized ΔMPL
- Track historical relationship over training
- Predict E[ΔMPLᵢ] for new datapoints

#### Data Tiering System

Classify datapoints into tiers based on novelty and causal potential:

**Tier 0 (Redundant)**:
- E[ΔMPL] ≈ 0 (no new information)
- Premium: near-zero
- Example: Duplicate demonstrations, states already well-covered

**Tier 1 (Context-Novel)**:
- E[ΔMPL] = moderate (fills gaps in context/coverage)
- Premium: moderate
- Example: New environment conditions, minor task variations

**Tier 2 (Causal-Novel / Frontier)**:
- E[ΔMPL] = high (causally critical for performance)
- Premium: high
- Example: Edge cases, failure modes, novel sub-tasks

#### Ex-Ante Quote + Ex-Post True-Up

**Ex-ante** (upfront pricing):
- Predict E[ΔMPLᵢ] using novelty/causal metrics
- Quote Pₙₒₛₕₐᵣₑ based on prediction
- Customer decides: share data (discount) or pay premium

**Ex-post** (quarterly reconciliation):
- Measure realized ΔMPL using:
  - Held-out training (with vs without datapoint)
  - Off-policy evaluation
  - Counterfactual analysis
- True-up pricing: refund if over-charged, bill if under-charged

#### Numeric Example: Frontier Data Premium

```
Task: Dishwashing
p = $0.30/dish
E[ΔMPLᵢ] = 6 dishes/hr (from novelty/causal model)
H = 1,000 robot-hours (deployment horizon)
S = 1 (no sharing)
κ = 0.8 (80% confidence in prediction)

Pₙₒₛₕₐᵣₑ = 0.30 × 0.8 × 6 × 1,000
          = $1,440

Customer pays $1,440 premium for withholding this frontier datapoint.
If they share, they receive discount/credit instead.
```

#### Why This Matters

This creates a **meta-economic feedback system**:
- Task data has explicit market value tied to learning potential
- Causally novel data commands higher premiums (withholds outsized learning)
- MPL growth is a measurable financial asset
- Pricing incentives align: **share data → cheaper pricing; don't share → pay for foregone productivity**

## Technical Stack

- **Language**: Python
- **Deep Learning**: PyTorch (RL, imitation learning, diffusion models)
- **Simulation**: Gymnasium, Mujoco, Isaac Gym
- **Data/Analysis**: Numpy, Pandas, Matplotlib, Scikit-learn
- **Compute**: RunPod (cloud GPU provider with PyTorch templates)

## Core Architecture

### 1. Video-to-Policy Module
- Input: Video data of human task performance
- Output: Trained robot control policies
- Methods: Imitation learning, diffusion policies

### 2. Simulation Environment
- Primary tasks: Dishwashing (initial), bricklaying (next)
- Environments: Gymnasium / Mujoco / Isaac Gym
- Agents: Humanoid robots performing real-world labor tasks

### 3. Economic Valuation Module
- **Training metrics**: MPᵣ(t), e(t), ŵᵣ(t), parity tracking
- **Data valuation**: E[ΔMPLᵢ] computation via novelty/causal metrics
- **Tiering system**: Classify datapoints into Tier 0/1/2
- Economic reward signals for policy training
- Sources wₕ from config (BLS data for specific roles)

### 4. Data Value Estimator
- Computes novelty_score and causal_gain for each datapoint
- Maintains rolling regression: (novelty, causal_gain) → realized ΔMPL
- Predicts E[ΔMPLᵢ] with confidence intervals
- Computes κ (confidence discount) from prediction uncertainty
- Outputs value_per_datapoint, tier assignment, premium quote

### 5. Pricing Engine
- Ex-ante quotes for data sharing vs non-sharing
- Tier-based pricing (0 = redundant, 1 = context, 2 = frontier)
- Ex-post true-up mechanism using realized ΔMPL
- Simulates equilibrium under different sharing incentives

## Implementation Guide

### Computing Training Economics (Every Episode)

```python
# Marginal Product (units per hour)
MPᵣ_t = completed_units / time_hours

# Error rate (fraction of attempts with errors)
e_t = errors / attempts

# Robot implied wage
ŵᵣ_t = price_per_unit * MPᵣ_t - damage_cost * (e_t * attempts)

# Parity metrics
productivity_parity = MPᵣ_t / MPₕ
wage_parity = ŵᵣ_t / wₕ
```

### Reward Function Implementation

```python
# Delta MP (normalized by human productivity)
delta_mp_norm = (MPᵣ_t - MPᵣ_prev) / MPₕ

# Wage convergence loss
wage_loss = abs(ŵᵣ_t / wₕ - 1)  # or squared: (ŵᵣ_t / wₕ - 1)**2

# Combined reward
reward = alpha * delta_mp_norm - beta * e_t - gamma * wage_loss
```

### Computing Data Value via Novelty/Causal Metrics

**Module**: `economics/data_value_estimator.py`

```python
# 1. Compute novelty score (embedding distance, coverage, etc.)
novelty_score = compute_novelty(datapoint, existing_data)

# 2. Compute causal gain (information gain, prediction error reduction)
causal_gain = compute_causal_gain(datapoint, policy)

# 3. Predict E[ΔMPL] using rolling regression
# Fit: (novelty, causal_gain) → realized_mpl_improvement
# Over past N datapoints
X = [(novelty_i, causal_i) for i in history]
y = [realized_mpl_i for i in history]
model.fit(X, y)

# Predict for new datapoint
expected_mpl_gain, lower_ci, upper_ci = model.predict_with_uncertainty(
    (novelty_score, causal_gain)
)

# 4. Compute confidence discount κ
kappa = lower_ci / expected_mpl_gain  # More uncertainty → smaller κ

# 5. Compute premium for non-sharing
P_noshare = price_per_unit * kappa * expected_mpl_gain * horizon * S

# 6. Assign tier based on expected_mpl_gain thresholds
if expected_mpl_gain < tier0_threshold:
    tier = 0  # Redundant
elif expected_mpl_gain < tier1_threshold:
    tier = 1  # Context-novel
else:
    tier = 2  # Causal-novel/frontier
```

### Novelty Score Implementation

**Embedding-based novelty**:
```python
def compute_novelty(datapoint, existing_data, embedder):
    # Embed datapoint
    embedding = embedder(datapoint)

    # Compute distance to nearest neighbors in existing data
    distances = [euclidean_distance(embedding, emb)
                 for emb in existing_data_embeddings]

    # Novelty = minimum distance (or percentile)
    novelty_score = min(distances)  # or np.percentile(distances, 10)

    return novelty_score
```

**State-action coverage novelty**:
```python
def compute_coverage_novelty(datapoint, state_action_histogram):
    # Discretize state-action space
    bin_idx = discretize(datapoint.state, datapoint.action)

    # Novelty = inverse of bin count
    count = state_action_histogram[bin_idx]
    novelty_score = 1.0 / (count + 1)  # +1 to avoid div by zero

    return novelty_score
```

### Causal Gain Implementation

**Information gain (Bayesian)**:
```python
def compute_information_gain(datapoint, policy_posterior):
    # Entropy before observing datapoint
    H_before = compute_entropy(policy_posterior)

    # Expected entropy after observing datapoint
    H_after_expected = compute_expected_entropy(
        policy_posterior, datapoint
    )

    # Information gain
    causal_gain = H_before - H_after_expected

    return causal_gain
```

**Prediction error reduction**:
```python
def compute_prediction_error_gain(datapoint, policy):
    # Current prediction error on validation set
    error_before = compute_validation_error(policy)

    # Predicted error after including this datapoint
    # (use holdout estimation or cross-validation)
    error_after_predicted = estimate_error_after_update(
        policy, datapoint
    )

    # Causal gain
    causal_gain = error_before - error_after_predicted

    return causal_gain
```

### Rolling Regression for E[ΔMPL] Prediction

```python
class DataValueEstimator:
    def __init__(self, lookback_window=100):
        self.lookback_window = lookback_window
        self.history = []  # [(novelty, causal_gain, realized_mpl), ...]
        self.model = RidgeRegression()  # or BayesianRegression

    def update(self, novelty, causal_gain, realized_mpl):
        """Add new observation to history"""
        self.history.append((novelty, causal_gain, realized_mpl))

        # Keep only recent history
        if len(self.history) > self.lookback_window:
            self.history = self.history[-self.lookback_window:]

        # Refit model
        if len(self.history) >= 10:  # Minimum samples
            X = [(n, c) for n, c, _ in self.history]
            y = [mpl for _, _, mpl in self.history]
            self.model.fit(X, y)

    def predict(self, novelty, causal_gain):
        """Predict E[ΔMPL] with confidence interval"""
        if len(self.history) < 10:
            # Not enough data; use conservative estimate
            return 0.0, 0.0, 0.0  # mean, lower_ci, upper_ci

        mean = self.model.predict([(novelty, causal_gain)])[0]

        # Compute prediction interval (e.g., via bootstrap)
        lower_ci, upper_ci = self.model.predict_interval(
            [(novelty, causal_gain)], confidence=0.80
        )

        return mean, lower_ci, upper_ci
```

### Ex-Post True-Up Implementation

```python
class PricingEngine:
    def quote_premium(self, datapoint, horizon, sharing_coef):
        """Ex-ante pricing quote"""
        novelty = compute_novelty(datapoint)
        causal_gain = compute_causal_gain(datapoint)

        expected_mpl, lower_ci, upper_ci = self.estimator.predict(
            novelty, causal_gain
        )

        kappa = lower_ci / expected_mpl if expected_mpl > 0 else 0.5

        P_noshare = (self.price_per_unit * kappa *
                     expected_mpl * horizon * sharing_coef)

        return {
            'premium': P_noshare,
            'expected_mpl': expected_mpl,
            'kappa': kappa,
            'tier': self.assign_tier(expected_mpl),
            'ex_ante_timestamp': time.now()
        }

    def reconcile_premium(self, quote, realized_mpl):
        """Ex-post true-up using realized MPL"""
        # Compute what premium should have been
        P_realized = (self.price_per_unit * quote['kappa'] *
                      realized_mpl * quote['horizon'] * quote['sharing_coef'])

        # True-up amount (positive = refund, negative = additional charge)
        trueup = quote['premium'] - P_realized

        return {
            'original_premium': quote['premium'],
            'realized_premium': P_realized,
            'trueup_amount': trueup,
            'ex_post_timestamp': time.now()
        }
```

### Curriculum Learning Strategy

**Early training** (focus on task competence):
```
α ≫ γ  # High weight on productivity improvement
```

**As performance stabilizes** (tighten wage convergence):
```
α ↓    # Reduce productivity improvement weight
γ ↑    # Increase wage convergence weight
```

### Logging Requirements

**Training economics** (every episode):
- `episode`
- `MPᵣ_t` (robot marginal product)
- `e_t` (error rate)
- `ŵᵣ_t` (robot implied wage)
- `wage_parity` (ŵᵣ/wₕ)
- `productivity_parity` (MPᵣ/MPₕ)
- `reward_total`
- `reward_mp` (α · ΔMPᵣ component)
- `reward_error` (β · e component)
- `reward_wage` (γ · wage_loss component)
- `alpha`, `beta`, `gamma` (hyperparameters)

**Data economics** (per datapoint):
- `datapoint_id`
- `novelty_score`
- `causal_gain`
- `expected_mpl_gain` (E[ΔMPLᵢ])
- `confidence_lower`, `confidence_upper` (prediction interval)
- `kappa` (confidence discount)
- `tier` (0/1/2)
- `premium_quote` (Pₙₒₛₕₐᵣₑ)
- `sharing_status` (shared/withheld)
- `realized_mpl_gain` (filled in later for true-up)
- `trueup_amount` (ex-post reconciliation)

### Safety & Realism Constraints

1. **Include damage costs**: cₐ prevents "speed-running" with sloppy errors
2. **Clip MP improvements**: Cap ΔMPᵣ(t) per step to avoid reward explosion
3. **Episode timeouts**: Prevent infinite episodes
4. **Error penalties**: Ensure β is large enough to discourage high error rates
5. **Minimum sample size**: Don't fit E[ΔMPL] model until sufficient history (e.g., 10+ datapoints)
6. **Confidence floor**: Set minimum κ (e.g., 0.3) to avoid zero premiums on high-uncertainty predictions
7. **Tier threshold validation**: Calibrate tier boundaries using held-out data

### Convergence Criteria

Stop/flag success when **both** conditions hold for N consecutive evaluations:

1. **Productivity parity**: `MPᵣ(t) / MPₕ ≥ 1 - ε` (e.g., ε = 0.05)
2. **Wage parity**: `|ŵᵣ(t) / wₕ - 1| ≤ δ` (e.g., δ = 0.10)

## Development Workflow

### Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install core dependencies
pip install torch torchvision gymnasium mujoco numpy pandas matplotlib scikit-learn scipy
```

### Running Simulations

```bash
# Run task environment
python -m envs.dishwashing_env

# Run policy training with economic reward + data valuation
python -m train \
  --env dishwashing \
  --algorithm ppo \
  --episodes 10000 \
  --alpha 1.0 \
  --beta 0.5 \
  --gamma 0.1 \
  --track-data-value \
  --enable-tiering

# Evaluate trained policy
python -m evaluate --policy checkpoints/model_latest.pt --env dishwashing
```

### Economic Analysis

```bash
# Compute MPL metrics and wage parity
python -m economics.compute_metrics --policy checkpoints/model_latest.pt

# Generate productivity and wage convergence reports
python -m economics.generate_report \
  --policy checkpoints/model_latest.pt \
  --output reports/

# Plot wage convergence over training
python -m economics.plot_convergence --log logs/training_metrics.csv

# Analyze data valuation, tiering, and premiums
python -m economics.analyze_data_value \
  --log logs/data_value.csv \
  --output reports/data_valuation.pdf

# Run ex-post true-up reconciliation
python -m economics.reconcile_premiums \
  --quotes logs/premium_quotes.csv \
  --realized logs/realized_mpl.csv \
  --output reports/trueup_summary.csv
```

### Pricing Simulation

```bash
# Simulate pricing with vs without data sharing
python -m economics.pricing_simulator \
  --task dishwashing \
  --data-sharing-rate 0.8 \
  --episodes 10000 \
  --output reports/pricing_simulation.csv

# Compare tiering strategies
python -m economics.compare_tiers \
  --config configs/dishwashing.yaml \
  --output reports/tier_comparison.pdf
```

### Training on RunPod

```bash
# Copy training scripts to RunPod container
# Use RunPod's PyTorch template with CUDA support

# SSH into RunPod pod
# Run training with GPU
python -m train \
  --env dishwashing \
  --algorithm ppo \
  --episodes 50000 \
  --gpu \
  --alpha-anneal \
  --gamma-schedule linear \
  --track-data-value \
  --enable-tiering
```

## Project Structure (Planned)

```
robotics v-p economics model/
├── envs/                           # Simulation environments
│   ├── dishwashing_env.py
│   ├── bricklaying_env.py
│   └── base_env.py
├── policies/                       # Policy networks and training code
│   ├── ppo.py
│   ├── sac.py
│   ├── diffusion_policy.py         # For v-p pipeline
│   └── network.py
├── economics/                      # Economic valuation module
│   ├── metrics.py                  # MPᵣ, ŵᵣ, parity computation
│   ├── reward.py                   # Economic reward function
│   ├── data_value_estimator.py     # Novelty/causal → E[ΔMPL] prediction
│   ├── novelty.py                  # Novelty score implementations
│   ├── causal_gain.py              # Causal gain implementations
│   ├── tiering.py                  # Tier 0/1/2 classification
│   ├── pricing_engine.py           # Ex-ante quotes + ex-post true-up
│   ├── wage_data.py                # wₕ data (BLS integration)
│   ├── pricing_simulator.py        # "With sharing" vs "without sharing"
│   └── analysis.py                 # Reports and visualization
├── video_to_policy/                # Video demonstration processing
│   ├── extract.py                  # Feature extraction from video
│   ├── imitation.py                # Behavioral cloning
│   ├── diffusion.py                # Diffusion policy training
│   └── demo_loader.py              # Video dataset loader
├── configs/                        # Configuration files
│   ├── dishwashing.yaml            # Task params: p, MPₕ, wₕ, cₐ, tiers
│   ├── bricklaying.yaml
│   └── training.yaml               # α, β, γ schedules
├── train.py                        # Main training script
├── evaluate.py                     # Policy evaluation script
├── checkpoints/                    # Saved model checkpoints
├── data/                           # Video demonstrations and datasets
├── logs/                           # Training logs and metrics CSVs
└── reports/                        # Economic analysis and plots
```

## Key Performance Metrics

### Task Performance
- Task success rate (% of episodes completed)
- Task completion time (average time per task)
- Error rate e(t) (fraction with errors/damage)

### Training Economics
- MPᵣ(t) (units/hour)
- ŵᵣ(t) ($/hour) - robot implied wage
- Productivity Parity (MPᵣ/MPₕ)
- Wage Parity (ŵᵣ/wₕ)

### Data Economics
- Novelty score distribution (per datapoint)
- Causal gain distribution (per datapoint)
- E[ΔMPLᵢ] prediction accuracy (predicted vs realized)
- Tier distribution (% Tier 0/1/2)
- Premium quotes vs realized true-up
- Cumulative data value (shared vs withheld)

### Training Efficiency
- Episodes to reach wage parity threshold
- Sample efficiency (data units needed)
- Reward component breakdown (MP vs error vs wage)
- Data value convergence rate
- Tier calibration accuracy

## Development Priorities

1. Build dishwashing simulation environment (Gymnasium/Mujoco)
2. Implement economic metrics module (MPᵣ, ŵᵣ, parity calculations)
3. Implement economic reward function with curriculum learning
4. **Implement novelty score computation (embedding-based + coverage-based)**
5. **Implement causal gain computation (information gain + prediction error)**
6. **Implement data value estimator with rolling regression**
7. **Implement tiering system (Tier 0/1/2 classification)**
8. Set up comprehensive logging (training + data economics)
9. Train baseline humanoid policy (PPO/SAC) with economic rewards
10. **Validate E[ΔMPL] prediction accuracy on held-out data**
11. **Implement pricing engine with ex-ante quotes and ex-post true-up**
12. Collect/generate video demonstrations for dishwashing
13. Implement video-to-policy imitation learning / diffusion policy
14. Validate wage convergence AND data valuation model together
15. **Run pricing simulations: sharing vs non-sharing scenarios**
16. Scale to bricklaying task
17. Integrate BLS wage data for multiple roles
18. **Deploy feedback-priced learning ecosystem (tiered pricing + true-up)**
19. Generalize pipeline for arbitrary labor tasks

## Economic Module Config Structure

Each task should have a YAML config with economic parameters:

```yaml
task: dishwashing
price_per_unit: 0.30  # $/dish
human_mp: 60          # dishes/hour
human_wage: 18.0      # $/hour (national avg)
damage_cost: 1.0      # $/broken dish
bls_occupation_code: "35-2021"  # Dishwashers (for wage updates)

reward:
  alpha_initial: 1.0   # Productivity improvement weight
  alpha_final: 0.3
  beta: 0.5            # Error penalty
  gamma_initial: 0.1   # Wage convergence weight
  gamma_final: 1.0
  anneal_episodes: 5000

convergence:
  productivity_threshold: 0.95  # 95% of human MP
  wage_parity_tolerance: 0.10   # Within 10% of wₕ
  eval_window: 50               # Consecutive evals to confirm

data_valuation:
  novelty_metric: "embedding_distance"  # or "state_coverage"
  causal_metric: "information_gain"     # or "prediction_error_reduction"
  lookback_window: 100                  # Episodes to fit E[ΔMPL] regression
  min_samples_for_fit: 10               # Minimum datapoints before fitting
  confidence_level: 0.80                # For prediction intervals
  kappa_floor: 0.3                      # Minimum confidence discount

  # Tier boundaries (expected MPL gain thresholds)
  tier0_threshold: 0.5   # Below this: Tier 0 (redundant)
  tier1_threshold: 3.0   # Below this: Tier 1 (context-novel)
  # Above tier1_threshold: Tier 2 (causal-novel/frontier)

pricing:
  base_price_per_task: 0.30           # Base price (p)
  horizon_hours: 1000                 # Robot-hours for premium calculation
  data_sharing_discount: 0.15         # 15% discount if customer shares
  trueup_period_days: 90              # Quarterly reconciliation
  trueup_payment_terms: "net30"       # Payment terms for true-up
```

## Implementation Details

### Novelty Score Module

**Module**: `economics/novelty.py`

```python
class NoveltyEstimator:
    """Estimates how novel/different a datapoint is from existing data"""

    def __init__(self, method='embedding', embedder=None):
        self.method = method
        self.embedder = embedder
        self.existing_embeddings = []
        self.state_action_histogram = {}

    def compute_novelty(self, datapoint):
        if self.method == 'embedding':
            return self._embedding_novelty(datapoint)
        elif self.method == 'coverage':
            return self._coverage_novelty(datapoint)
        else:
            raise ValueError(f"Unknown novelty method: {self.method}")

    def _embedding_novelty(self, datapoint):
        embedding = self.embedder(datapoint)

        if len(self.existing_embeddings) == 0:
            return 1.0  # First datapoint is maximally novel

        # Distance to k nearest neighbors
        distances = [np.linalg.norm(embedding - emb)
                     for emb in self.existing_embeddings]
        k = min(5, len(distances))
        k_nearest = sorted(distances)[:k]

        # Novelty = average distance to k nearest
        novelty = np.mean(k_nearest)
        return novelty

    def _coverage_novelty(self, datapoint):
        bin_idx = self._discretize(datapoint)
        count = self.state_action_histogram.get(bin_idx, 0)
        novelty = 1.0 / (count + 1)
        return novelty

    def update(self, datapoint):
        """Add datapoint to existing data"""
        if self.method == 'embedding':
            embedding = self.embedder(datapoint)
            self.existing_embeddings.append(embedding)
        elif self.method == 'coverage':
            bin_idx = self._discretize(datapoint)
            self.state_action_histogram[bin_idx] = \
                self.state_action_histogram.get(bin_idx, 0) + 1
```

### Causal Gain Module

**Module**: `economics/causal_gain.py`

```python
class CausalGainEstimator:
    """Estimates expected performance improvement from a datapoint"""

    def __init__(self, method='information_gain', policy=None):
        self.method = method
        self.policy = policy
        self.validation_set = None

    def compute_causal_gain(self, datapoint):
        if self.method == 'information_gain':
            return self._information_gain(datapoint)
        elif self.method == 'prediction_error':
            return self._prediction_error_reduction(datapoint)
        else:
            raise ValueError(f"Unknown causal method: {self.method}")

    def _information_gain(self, datapoint):
        # Compute entropy of policy predictions before/after datapoint
        # (Simplified: use variance of policy outputs as proxy)

        # Variance before
        outputs_before = [self.policy(s) for s in self.validation_set]
        variance_before = np.var(outputs_before)

        # Expected variance after (approximate via importance sampling)
        # ... (implementation details)

        # Information gain = variance reduction
        gain = variance_before - variance_after_expected
        return gain

    def _prediction_error_reduction(self, datapoint):
        # Current prediction error on validation set
        error_before = self._compute_validation_error()

        # Estimate error after including datapoint
        # (Use online learning / incremental update estimate)
        error_after_est = self._estimate_error_after_update(datapoint)

        # Causal gain
        gain = error_before - error_after_est
        return max(0, gain)  # Clip to non-negative
```

### Data Value Estimator Module

**Module**: `economics/data_value_estimator.py`

```python
from sklearn.linear_model import Ridge, BayesianRidge
from sklearn.ensemble import GradientBoostingRegressor

class DataValueEstimator:
    """Maps (novelty, causal_gain) → E[ΔMPL] using historical data"""

    def __init__(self, lookback_window=100, min_samples=10, kappa_floor=0.3):
        self.lookback_window = lookback_window
        self.min_samples = min_samples
        self.kappa_floor = kappa_floor

        self.history = []  # [(novelty, causal_gain, realized_mpl), ...]
        self.model = BayesianRidge()  # Provides prediction intervals
        self.novelty_estimator = None
        self.causal_estimator = None

    def add_observation(self, novelty, causal_gain, realized_mpl):
        """Add new (X, y) pair to history and refit"""
        self.history.append((novelty, causal_gain, realized_mpl))

        # Keep only recent history
        if len(self.history) > self.lookback_window:
            self.history = self.history[-self.lookback_window:]

        # Refit model if enough samples
        if len(self.history) >= self.min_samples:
            X = np.array([[n, c] for n, c, _ in self.history])
            y = np.array([mpl for _, _, mpl in self.history])
            self.model.fit(X, y)

    def predict(self, novelty, causal_gain, confidence=0.80):
        """Predict E[ΔMPL] with confidence interval"""
        if len(self.history) < self.min_samples:
            # Not enough data; return conservative estimate
            return 0.0, 0.0, 0.0  # mean, lower_ci, upper_ci

        X_new = np.array([[novelty, causal_gain]])

        # Mean prediction
        mean = self.model.predict(X_new)[0]

        # Prediction interval (Bayesian Ridge provides std)
        std = np.sqrt(self.model.predict(X_new, return_std=True)[1][0])
        z = 1.28 if confidence == 0.80 else 1.96  # 80% or 95% CI
        lower_ci = mean - z * std
        upper_ci = mean + z * std

        # Clip to non-negative
        mean = max(0, mean)
        lower_ci = max(0, lower_ci)
        upper_ci = max(0, upper_ci)

        return mean, lower_ci, upper_ci

    def compute_kappa(self, mean, lower_ci):
        """Compute confidence discount κ = lower_ci / mean"""
        if mean == 0:
            return self.kappa_floor

        kappa = lower_ci / mean
        kappa = max(self.kappa_floor, min(1.0, kappa))  # Clip to [floor, 1]
        return kappa

    def compute_value(self, datapoint, price_per_unit):
        """Compute full data value for a datapoint"""
        # 1. Compute novelty and causal gain
        novelty = self.novelty_estimator.compute_novelty(datapoint)
        causal_gain = self.causal_estimator.compute_causal_gain(datapoint)

        # 2. Predict E[ΔMPL] with confidence interval
        mean, lower_ci, upper_ci = self.predict(novelty, causal_gain)

        # 3. Compute κ
        kappa = self.compute_kappa(mean, lower_ci)

        # 4. Compute value
        value = price_per_unit * kappa * mean

        return {
            'novelty': novelty,
            'causal_gain': causal_gain,
            'expected_mpl': mean,
            'confidence_lower': lower_ci,
            'confidence_upper': upper_ci,
            'kappa': kappa,
            'value': value
        }
```

### Tiering Module

**Module**: `economics/tiering.py`

```python
class DataTieringSystem:
    """Classifies datapoints into Tier 0/1/2 based on expected MPL"""

    def __init__(self, tier0_threshold=0.5, tier1_threshold=3.0):
        self.tier0_threshold = tier0_threshold
        self.tier1_threshold = tier1_threshold

    def assign_tier(self, expected_mpl):
        """Assign tier based on expected MPL gain"""
        if expected_mpl < self.tier0_threshold:
            return 0  # Redundant
        elif expected_mpl < self.tier1_threshold:
            return 1  # Context-novel
        else:
            return 2  # Causal-novel / Frontier

    def get_tier_description(self, tier):
        descriptions = {
            0: "Redundant (no new information)",
            1: "Context-novel (fills gaps in coverage)",
            2: "Causal-novel / Frontier (critical for performance)"
        }
        return descriptions.get(tier, "Unknown")

    def compute_tier_statistics(self, datapoints_with_tiers):
        """Compute tier distribution statistics"""
        tier_counts = {0: 0, 1: 0, 2: 0}
        total_value_by_tier = {0: 0.0, 1: 0.0, 2: 0.0}

        for dp in datapoints_with_tiers:
            tier = dp['tier']
            value = dp['value']
            tier_counts[tier] += 1
            total_value_by_tier[tier] += value

        total = sum(tier_counts.values())
        tier_percentages = {t: 100 * c / total for t, c in tier_counts.items()}

        return {
            'counts': tier_counts,
            'percentages': tier_percentages,
            'total_value_by_tier': total_value_by_tier,
            'avg_value_by_tier': {
                t: total_value_by_tier[t] / tier_counts[t]
                if tier_counts[t] > 0 else 0
                for t in [0, 1, 2]
            }
        }
```

### Pricing Engine Module

**Module**: `economics/pricing_engine.py`

```python
class PricingEngine:
    """Handles ex-ante quotes and ex-post true-up for data pricing"""

    def __init__(self, config):
        self.price_per_unit = config['price_per_unit']
        self.horizon_hours = config['horizon_hours']
        self.data_sharing_discount = config['data_sharing_discount']
        self.trueup_period_days = config['trueup_period_days']

        self.value_estimator = None  # DataValueEstimator
        self.tiering_system = None   # DataTieringSystem

        self.quotes = []  # Ex-ante quotes issued

    def quote_premium(self, datapoint, sharing_status='no_share'):
        """Generate ex-ante pricing quote"""
        # Compute data value
        value_info = self.value_estimator.compute_value(
            datapoint, self.price_per_unit
        )

        # Assign tier
        tier = self.tiering_system.assign_tier(value_info['expected_mpl'])

        # Compute premium for non-sharing
        S = 0 if sharing_status == 'share' else 1
        P_noshare = (self.price_per_unit *
                     value_info['kappa'] *
                     value_info['expected_mpl'] *
                     self.horizon_hours *
                     S)

        # Apply discount if sharing
        if sharing_status == 'share':
            discount = self.price_per_unit * self.data_sharing_discount
            final_price = -discount  # Negative = credit/discount
        else:
            final_price = P_noshare

        quote = {
            'datapoint_id': id(datapoint),
            'timestamp': time.time(),
            'novelty': value_info['novelty'],
            'causal_gain': value_info['causal_gain'],
            'expected_mpl': value_info['expected_mpl'],
            'confidence_lower': value_info['confidence_lower'],
            'confidence_upper': value_info['confidence_upper'],
            'kappa': value_info['kappa'],
            'tier': tier,
            'sharing_status': sharing_status,
            'premium': final_price,
            'horizon_hours': self.horizon_hours,
            'trueup_due_date': time.time() + (self.trueup_period_days * 86400)
        }

        self.quotes.append(quote)
        return quote

    def reconcile_premium(self, quote_id, realized_mpl):
        """Ex-post true-up using realized MPL"""
        # Find quote
        quote = next((q for q in self.quotes if q['datapoint_id'] == quote_id), None)
        if quote is None:
            raise ValueError(f"Quote {quote_id} not found")

        # Compute what premium should have been
        S = 0 if quote['sharing_status'] == 'share' else 1
        P_realized = (self.price_per_unit *
                      quote['kappa'] *
                      realized_mpl *
                      quote['horizon_hours'] *
                      S)

        # True-up amount (positive = refund to customer, negative = bill customer)
        trueup = quote['premium'] - P_realized

        return {
            'datapoint_id': quote_id,
            'timestamp': time.time(),
            'ex_ante_premium': quote['premium'],
            'ex_post_premium': P_realized,
            'expected_mpl': quote['expected_mpl'],
            'realized_mpl': realized_mpl,
            'prediction_error': abs(realized_mpl - quote['expected_mpl']),
            'trueup_amount': trueup,
            'trueup_pct': 100 * trueup / quote['premium'] if quote['premium'] != 0 else 0
        }

    def generate_reconciliation_report(self, period_start, period_end):
        """Generate quarterly true-up report"""
        # Get all quotes in period
        period_quotes = [
            q for q in self.quotes
            if period_start <= q['timestamp'] <= period_end
        ]

        # ... (aggregate true-up amounts, prediction errors, etc.)

        return report
```

## Notes

- This is a research prototype designed to evolve into a generalizable framework
- **Meta-economic principle**: Task data has explicit market value; causally novel data commands higher premiums
- MPL growth is a measurable financial asset that grounds both training rewards and data pricing
- Pricing and policy incentives align: **share data → cheaper pricing; don't share → pay for foregone productivity**
- Focus on measurable economic value ($/hour), not just abstract task completion
- RunPod provides cost-effective GPU training; use PyTorch-ready templates
- Log all metrics (RL rewards, task metrics, training economics, data economics) for analysis
- Economic module should be modular to swap different valuation functions
- Wage data should be sourced from BLS (Bureau of Labor Statistics) for accuracy
- Clip/normalize rewards to prevent instability during early training
- Use curriculum learning: first teach task, then tighten economic constraints
- **Data valuation must be tracked from day 1** - it defines the feedback-priced learning ecosystem
- **Novelty and causal gain metrics are implementation-flexible**: use embeddings, coverage, information gain, or prediction error as appropriate
- **Ex-post true-up is critical for trust**: customers need confidence that quotes are accurate and reconciled fairly
- **Tier calibration should be validated on held-out data** to ensure boundaries reflect real MPL gains
- **κ (confidence discount) prevents over-charging on uncertain predictions**, aligning pricing with prediction quality
