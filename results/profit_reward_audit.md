# Profit Reward Audit

- train_ppo.py: now uses `compute_econ_reward` (mpl/ep/error) for RL reward; profit remains logging-only.
- train_ppo_ablate.py: now uses `compute_econ_reward` (mpl/ep/error) for RL reward; profit remains logging-only.
- train_ppo_ablate_v2.py: now uses `compute_econ_reward` (mpl/ep/error) for RL reward; profit remains logging-only.
- src/economics/reward.py:12-40 — `econ_lagrangian_reward` definition encodes profit/hr with SLA penalty; currently consumed by PPO scripts. Proposal: redesign helper to compute weighted MPL/EP/error reward and expose optional SLA penalty, then update callers.
- train_sac.py:189 — Computes profit for logging only (post-episode), not used as RL reward. Note for consistency; no reward change needed.
- train_sac_v2.py:399 — Computes profit for logging only (post-episode), not used as RL reward. Note for consistency; no reward change needed.
- experiments/summary_snapshots.py:165-187 — Profit per robot/year metrics for reporting, not tied to RL rewards. Logging-only; no change recommended.
