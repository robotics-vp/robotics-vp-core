#!/usr/bin/env python3
"""
Internal Experiment Profile Configuration

IMPORTANT: These are experimental knobs for testing scripts ONLY.
The actual synthetic weighting in RL is 100% DL-driven (trust_net × w_econ_lattice).
These knobs control A/B test parameters, data collection settings, and evaluation configs.

TODO: Migrate to full PolicyProfile after demo phase.
"""


def get_internal_experiment_profile(env_name: str = "default"):
    """
    Centralized knobs for testing scripts ONLY.
    Not part of the RL or economic weighting logic.

    Args:
        env_name: Environment/task name for task-specific configs

    Returns:
        dict of experimental configuration parameters
    """
    # Base profile (common to all envs)
    base_profile = {
        # A/B test knobs (not used in actual model weighting)
        "target_synth_share": 0.20,  # Target synthetic contribution for A/B tests
        "max_synth_share": 0.40,     # Hard cap on synthetic share (controller cannot exceed)
        "econ_weight_scale": 1.0,    # Scale factor for economic weights in tests

        # Branch collection parameters
        "max_branch_horizon": 20,     # Maximum rollout horizon for branch collection
        "branches_per_episode": 5,    # Branches to sample per episode
        "min_trust_threshold": 0.9,   # Trust gate threshold for branch acceptance
        "min_std_ratio": 0.8,         # Minimum std ratio for variance gate
        "max_std_ratio": 1.2,         # Maximum std ratio for variance gate

        # Objective conditioning (for future task-specific weighting)
        "default_objective_vector": [1.0, 0.7, 0.5, 0.8],  # [productivity, precision, energy, novelty]
        "objective_dim": 4,
        # Dishwashing environment thresholds / costs
        "price_per_unit": 0.30,
        "damage_cost": 1.0,
        "time_step_s": 60.0,
        "energy_Wh_per_attempt": 0.05,
        "base_rate": 2.0,
        "p_min": 0.02,
        "k_err": 0.12,
        "q_speed": 1.2,
        "q_care": 1.5,
        "care_cost": 0.25,
        "max_steps": 240,
        "max_catastrophic_errors": 3,
        "max_error_rate_sla": 0.12,
        "min_steps_for_sla": 5,
        "zero_throughput_patience": 10,
        "econ_preset": "toy",
        # Realistic preset overrides (optional, used by load_econ_params)
        "price_per_unit_realistic": 0.25,
        "damage_cost_realistic": 1.5,
        "base_rate_realistic": 1.6,
        "max_error_rate_sla_realistic": 0.08,
        "energy_Wh_per_attempt_realistic": 0.08,

        # Training defaults
        "lattice_n_keypoints": 16,    # Keypoints per calibrator
        "lattice_hidden_dim": 32,     # MLP hidden dimension
        "lattice_n_bricks": 5,        # Number of brick embeddings
        "lattice_epochs": 100,        # Training epochs for lattice

        # A/B test evaluation
        "ab_test_epochs": 100,        # Epochs per A/B test condition
        "ab_test_batch_size": 256,    # Batch size for A/B tests
        "ab_test_lr": 1e-3,           # Learning rate for A/B tests

        # Gating caps (safety bounds, not model parameters)
        "econ_weight_cap": 1.0,       # Maximum allowed econ weight
        "trust_weight_floor": 0.0,    # Minimum trust weight

        # Data paths (defaults, can be overridden)
        "real_data_path": "data/physics_zv_rollouts.npz",
        "synthetic_branches_path": "data/local_synth_branches.npz",
        "brick_manifest_path": "data/bricks/data_bricks_manifest.json",
        "w_econ_lattice_path": "checkpoints/w_econ_lattice.pt",
        "world_model_path": "checkpoints/world_model_stable_canonical.pt",
        "trust_net_path": "checkpoints/trust_net.pt",
        "synth_lambda_controller_path": "checkpoints/synth_lambda_controller.pt",
    }

    # Task-specific overrides
    if env_name == "dishwashing":
        base_profile.update({
            "default_objective_vector": [1.0, 0.8, 0.3, 0.6],  # High productivity, high precision
            "target_synth_share": 0.15,  # More conservative for safety
        })
    elif env_name == "bricklaying":
        base_profile.update({
            "default_objective_vector": [0.9, 0.9, 0.7, 0.5],  # Balanced productivity and precision
            "max_branch_horizon": 30,  # Longer horizon for brick tasks
        })
    # Add more task-specific profiles as needed

    return base_profile


def get_experiment_knob(env_name: str, knob_name: str):
    """
    Get a single experimental knob value.

    Args:
        env_name: Environment name
        knob_name: Name of the knob to retrieve

    Returns:
        Value of the specified knob
    """
    profile = get_internal_experiment_profile(env_name)
    if knob_name not in profile:
        raise ValueError(f"Unknown experiment knob: {knob_name}")
    return profile[knob_name]


def list_experiment_knobs():
    """List all available experiment knobs with descriptions."""
    knob_descriptions = {
        "target_synth_share": "Target synthetic data contribution ratio for A/B tests",
        "max_synth_share": "Hard cap on synthetic share (controller cannot exceed)",
        "econ_weight_scale": "Scale factor for economic weights in experiments",
        "max_branch_horizon": "Maximum rollout horizon for synthetic branch collection",
        "branches_per_episode": "Number of branches to sample per episode",
        "min_trust_threshold": "Minimum trust score for branch acceptance",
        "min_std_ratio": "Minimum std ratio for variance gate",
        "max_std_ratio": "Maximum std ratio for variance gate",
        "default_objective_vector": "Default task objective conditioning vector",
        "objective_dim": "Dimension of objective conditioning space",
        "lattice_n_keypoints": "Number of keypoints per calibrator in lattice",
        "lattice_hidden_dim": "Hidden dimension of lattice MLP",
        "lattice_n_bricks": "Number of brick embeddings",
        "lattice_epochs": "Training epochs for lattice model",
        "ab_test_epochs": "Epochs per A/B test condition",
        "ab_test_batch_size": "Batch size for A/B tests",
        "ab_test_lr": "Learning rate for A/B tests",
        "econ_weight_cap": "Maximum allowed economic weight (safety cap)",
        "trust_weight_floor": "Minimum trust weight (safety floor)",
        "price_per_unit": "Dishwashing: revenue per completed unit ($/dish)",
        "damage_cost": "Dishwashing: cost for broken dish",
        "time_step_s": "Dishwashing: seconds per env step",
        "energy_Wh_per_attempt": "Dishwashing: energy consumption per attempt (Wh)",
        "base_rate": "Dishwashing: baseline attempts per minute at speed=1",
        "p_min": "Dishwashing: minimum error probability",
        "k_err": "Dishwashing: maximum additional error scale",
        "q_speed": "Dishwashing: speed curvature in error model",
        "q_care": "Dishwashing: care curvature in error model",
        "care_cost": "Dishwashing: throughput penalty multiplier for care",
        "max_steps": "Dishwashing: max horizon (steps)",
        "max_catastrophic_errors": "Dishwashing: termination threshold for catastrophic errors",
        "max_error_rate_sla": "Dishwashing: SLA error-rate cutoff",
        "min_steps_for_sla": "Dishwashing: minimum steps before SLA enforcement",
        "zero_throughput_patience": "Dishwashing: patience before terminating on zero throughput",
        "econ_preset": "Economic parameter preset to use (toy|realistic)",
        "price_per_unit_realistic": "Realistic preset: revenue per completed unit",
        "damage_cost_realistic": "Realistic preset: cost per broken unit",
        "base_rate_realistic": "Realistic preset: baseline attempts/min",
        "max_error_rate_sla_realistic": "Realistic preset: SLA error-rate cutoff",
        "energy_Wh_per_attempt_realistic": "Realistic preset: energy per attempt",
        "real_data_path": "Path to real physics rollouts data",
        "synthetic_branches_path": "Path to synthetic branches file",
        "brick_manifest_path": "Path to brick manifest JSON",
        "w_econ_lattice_path": "Path to trained w_econ_lattice checkpoint",
        "world_model_path": "Path to stable world model checkpoint",
        "trust_net_path": "Path to trust network checkpoint",
    }
    return knob_descriptions
