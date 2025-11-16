# Energy Economics Bench

The arm-based “energy bench” environments provide a safe place to exercise τ·ω energy attribution without touching the frozen Phase B stack (trust_net, w_econ_lattice, λ controller, SyntheticWeightController, rewards).

## What it is
- `src/envs/dishwashing_arm_env.py`: articulated KUKA iiwa arm in a dishwashing proxy task; joint-velocity control; emits per-joint/limb/effector energy and coordination metrics.
- `src/envs/drawer_vase_arm_env.py`: articulated arm for the drawer+vase task; same energy attribution plumbing; success if end-effector reaches drawer target, error if near vase at speed.
- Configs: `configs/dishwashing_arm_state.yaml`, `configs/drawer_vase_arm_state.yaml`.
- Smoke tests: `scripts/smoke_test_dishwashing_arm.py`, `scripts/smoke_test_drawer_vase_arm.py`.

## How it plugs into the energy tooling
- **Energy profiles**: `src/controllers/energy_profile.py` (struct) and `src/controllers/energy_profile_policy.py` (small NN) can be used to generate preset or learned energy control knobs; not wired into RL or weighting.
- **Interventions**: `scripts/run_energy_interventions.py` runs BASE/BOOST/SAVER/SAFE profiles on the arm envs and writes `data/energy_interventions.jsonl`; `scripts/analyze_energy_interventions.py` summarizes MPL/error/energy deltas and limb/joint fractions.
- **Datapacks**: `src/valuation/datapacks.py` (schema_version `2.0-energy`) includes `energy_per_limb`, `energy_per_joint`, `energy_per_skill`, `energy_per_effector`, `coordination_metrics`, and `energy_driver_tags` via `src/valuation/energy_tags.py`.
- **Inspection**: `scripts/inspect_phase_c_datapacks.py` can be pointed at arm datapacks to group by env type and energy_driver_tags.

## What it is not
- It does **not** modify Phase B economics, weighting, or rewards.
- It is an additive bench to validate energy attribution and control; integration with RL/econ weighting is future work.
