## Econ & Reward Schema

### EconParams
- `price_per_unit`: revenue per completed unit ($/dish)
- `damage_cost`: cost per broken unit
- `energy_Wh_per_attempt`: energy consumption per attempt (Wh)
- `time_step_s`: seconds per environment step
- `base_rate`: baseline attempts per minute at speed=1
- `p_min`, `k_err`, `q_speed`, `q_care`: error model coefficients
- `care_cost`: throughput penalty multiplier for care
- `max_steps`: episode horizon
- `max_catastrophic_errors`: termination threshold for catastrophic errors
- `max_error_rate_sla`: SLA error-rate cutoff
- `min_steps_for_sla`: minimum steps before SLA enforcement
- `zero_throughput_patience`: patience before terminating on zero throughput
- `preset`: econ preset name (toy | realistic)

### EpisodeInfoSummary (from envs.dishwashing_env)
- `termination_reason`: enum-like string (`max_steps`, `sla_violation`, `catastrophic_error`, `zero_throughput`, `unknown`)
- `mpl_episode`: units per hour over the episode
- `ep_episode`: units per Wh over the episode
- `error_rate_episode`: errors per unit over the episode
- `throughput_units_per_hour`: alias of `mpl_episode`
- `energy_Wh`: total energy consumed
- `profit`: total profit (revenue - error_cost) aggregated per-step
- `wage_parity`: optional robot/human wage ratio if available

Use `summarize_episode_info` to aggregate per-step info into this summary and log it as the canonical episode record.

### Reward Shaping (src/rl/reward_shaping.py)
- `compute_econ_reward(mpl, ep, error_rate, wage_parity=None, mode="mpl_ep_error", alpha_mpl, alpha_error, alpha_ep, alpha_wage=0)` returns:
  - scalar reward: `alpha_mpl*mpl + alpha_ep*ep - alpha_error*error_rate - alpha_wage*max(0, wage_parity-1)`
  - component dict (`mpl_component`, `ep_component`, `error_penalty`, `wage_penalty`)
- Profit is logging-only; RL rewards are built from MPL/EP/error (plus optional wage parity penalty).
