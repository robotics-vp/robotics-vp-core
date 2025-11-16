"""
Episode Impact Features (Phase B.4.5)

Extracts performance-impact metrics from episodes for causal attribution.
These metrics answer: "What dimension of performance did this episode improve?"

Dimensions tracked:
- MPL (throughput)
- Error types (slip, breakage, misalignment, gripper failure)
- Energy efficiency (Wh per unit)
- Per-limb energy/torque (shoulder, elbow, wrist) if available
"""

import numpy as np
from typing import Dict, Optional, List, Any


def episode_impact_features(ep_row: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract impact-relevant performance metrics from a single episode.

    Args:
        ep_row: Episode data dict (from CSV row or rollout info)
            Expected keys: mpl, error_rate, errors, attempts, completed
            Optional keys: slip_errors, breakage_errors, misalignment_errors,
                          gripper_failures, energy_wh, time_hours,
                          shoulder_torque, elbow_torque, wrist_torque

    Returns:
        Dict of impact features (all numeric, ready for aggregation)
    """
    features = {}

    # --- Core Performance Metrics ---

    # MPL (Marginal Product of Labor)
    features['mpl_units_per_hr'] = float(ep_row.get('mpl', 0.0))

    # Error rate (overall)
    features['error_rate'] = float(ep_row.get('error_rate', 0.0))

    # Raw counts
    completed = float(ep_row.get('completed', 0))
    attempts = float(ep_row.get('attempts', completed))
    errors = float(ep_row.get('errors', 0))

    # Success rate (1 - error_rate)
    features['success_rate'] = 1.0 - features['error_rate'] if attempts > 0 else 0.0

    # --- Error Type Breakdown ---
    # Normalize by attempts to get error rates by type

    # Slip errors (wet conditions, friction issues)
    slip_errors = float(ep_row.get('slip_errors', 0))
    features['slip_error_rate'] = slip_errors / attempts if attempts > 0 else 0.0

    # Breakage errors (fragile handling)
    breakage_errors = float(ep_row.get('breakage_errors', 0))
    features['breakage_error_rate'] = breakage_errors / attempts if attempts > 0 else 0.0

    # Misalignment errors (precision issues)
    misalignment_errors = float(ep_row.get('misalignment_errors', 0))
    features['misalignment_error_rate'] = misalignment_errors / attempts if attempts > 0 else 0.0

    # Gripper failures (mechanical issues)
    gripper_failures = float(ep_row.get('gripper_failures', 0))
    features['gripper_failure_rate'] = gripper_failures / attempts if attempts > 0 else 0.0

    # If breakdown not available, estimate from total errors
    if slip_errors == 0 and breakage_errors == 0 and errors > 0:
        # Default assumption: 50% slip, 30% breakage, 20% misalignment
        features['slip_error_rate'] = features['error_rate'] * 0.5
        features['breakage_error_rate'] = features['error_rate'] * 0.3
        features['misalignment_error_rate'] = features['error_rate'] * 0.2

    # --- Energy Efficiency ---

    # Total energy (Wh)
    energy_wh = float(ep_row.get('energy_wh', 0.0))
    features['energy_wh_total'] = energy_wh

    # Energy per unit (efficiency metric)
    if completed > 0 and energy_wh > 0:
        features['energy_wh_per_unit'] = energy_wh / completed
    else:
        features['energy_wh_per_unit'] = 0.0

    # --- Per-Limb Energy (if available) ---
    # These track which joints are being optimized

    # Shoulder energy/torque
    shoulder_energy = float(ep_row.get('shoulder_energy', 0.0))
    features['shoulder_energy_per_unit'] = shoulder_energy / completed if completed > 0 else 0.0

    # Elbow energy/torque
    elbow_energy = float(ep_row.get('elbow_energy', 0.0))
    features['elbow_energy_per_unit'] = elbow_energy / completed if completed > 0 else 0.0

    # Wrist energy/torque
    wrist_energy = float(ep_row.get('wrist_energy', 0.0))
    features['wrist_energy_per_unit'] = wrist_energy / completed if completed > 0 else 0.0

    # --- Task-Specific Conditions ---

    # Wet/dry conditions (affects slip)
    features['wet_condition'] = float(ep_row.get('wet_condition', 0.5))

    # Object fragility (affects breakage)
    features['fragility_score'] = float(ep_row.get('fragility_score', 0.5))

    # Multi-object complexity
    features['num_objects'] = float(ep_row.get('num_objects', 1))

    # Occlusion level (visibility challenges)
    features['occlusion_level'] = float(ep_row.get('occlusion_level', 0.0))

    # Lighting conditions
    features['lighting_quality'] = float(ep_row.get('lighting_quality', 1.0))

    # --- Economic Metrics ---

    # Robot implied wage
    features['robot_wage'] = float(ep_row.get('robot_wage', 0.0))

    # Wage parity
    features['wage_parity'] = float(ep_row.get('wage_parity', 0.0))

    # Delta MPL (improvement over previous)
    features['delta_mpl'] = float(ep_row.get('delta_mpl', 0.0))

    return features


def batch_impact_features(episodes: List[Dict[str, Any]]) -> np.ndarray:
    """
    Extract impact features for multiple episodes.

    Args:
        episodes: List of episode data dicts

    Returns:
        (N, D) array where N=episodes, D=feature dimensions
    """
    if not episodes:
        return np.array([])

    # Extract features for each episode
    feature_dicts = [episode_impact_features(ep) for ep in episodes]

    # Get consistent feature names
    feature_names = list(feature_dicts[0].keys())

    # Convert to numpy array
    features = np.zeros((len(episodes), len(feature_names)))

    for i, fd in enumerate(feature_dicts):
        for j, name in enumerate(feature_names):
            features[i, j] = fd.get(name, 0.0)

    return features, feature_names


def compute_impact_deltas(
    cluster_episodes: List[Dict[str, Any]],
    all_episodes: List[Dict[str, Any]]
) -> Dict[str, float]:
    """
    Compute how a cluster differs from global population.

    Args:
        cluster_episodes: Episodes in this cluster
        all_episodes: All episodes (global population)

    Returns:
        Dict of delta values (cluster_mean - global_mean) for each feature
    """
    if not cluster_episodes or not all_episodes:
        return {}

    # Extract features
    cluster_features_list = [episode_impact_features(ep) for ep in cluster_episodes]
    global_features_list = [episode_impact_features(ep) for ep in all_episodes]

    # Compute means
    feature_names = list(cluster_features_list[0].keys())
    deltas = {}

    for name in feature_names:
        cluster_values = [f[name] for f in cluster_features_list]
        global_values = [f[name] for f in global_features_list]

        cluster_mean = np.mean(cluster_values)
        global_mean = np.mean(global_values)

        deltas[f'delta_{name}'] = cluster_mean - global_mean

    return deltas


def create_impact_profile(
    cluster_episodes: List[Dict[str, Any]],
    all_episodes: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Create comprehensive impact profile for a cluster.

    This profile answers: "How does this cluster differ from the global
    population along each performance dimension?"

    Args:
        cluster_episodes: Episodes in this cluster
        all_episodes: All episodes

    Returns:
        Impact profile dict with:
        - Raw deltas for each metric
        - Limb-specific deltas
        - Error-type specific deltas
        - Condition-specific patterns
    """
    deltas = compute_impact_deltas(cluster_episodes, all_episodes)

    # Organize into structured profile
    profile = {
        # Core performance deltas
        'delta_mpl_units_per_hr': deltas.get('delta_mpl_units_per_hr', 0.0),
        'delta_error_rate': deltas.get('delta_error_rate', 0.0),
        'delta_success_rate': deltas.get('delta_success_rate', 0.0),

        # Energy efficiency deltas
        'delta_energy_wh_per_unit': deltas.get('delta_energy_wh_per_unit', 0.0),

        # Per-limb energy deltas
        'limb_energy_deltas': {
            'shoulder': deltas.get('delta_shoulder_energy_per_unit', 0.0),
            'elbow': deltas.get('delta_elbow_energy_per_unit', 0.0),
            'wrist': deltas.get('delta_wrist_energy_per_unit', 0.0),
        },

        # Error-type specific deltas
        'error_type_deltas': {
            'slip': deltas.get('delta_slip_error_rate', 0.0),
            'breakage': deltas.get('delta_breakage_error_rate', 0.0),
            'misalignment': deltas.get('delta_misalignment_error_rate', 0.0),
            'gripper_failure': deltas.get('delta_gripper_failure_rate', 0.0),
        },

        # Condition associations
        'condition_profile': {
            'wet_tendency': deltas.get('delta_wet_condition', 0.0),
            'fragility_tendency': deltas.get('delta_fragility_score', 0.0),
            'multi_object_tendency': deltas.get('delta_num_objects', 0.0),
            'occlusion_tendency': deltas.get('delta_occlusion_level', 0.0),
        },

        # Economic deltas
        'economic_deltas': {
            'wage_parity': deltas.get('delta_wage_parity', 0.0),
            'delta_mpl': deltas.get('delta_delta_mpl', 0.0),  # Double delta!
        },
    }

    # Add cluster statistics
    cluster_features = [episode_impact_features(ep) for ep in cluster_episodes]

    profile['cluster_stats'] = {
        'num_episodes': len(cluster_episodes),
        'mean_mpl': np.mean([f['mpl_units_per_hr'] for f in cluster_features]),
        'std_mpl': np.std([f['mpl_units_per_hr'] for f in cluster_features]),
        'mean_error_rate': np.mean([f['error_rate'] for f in cluster_features]),
        'std_error_rate': np.std([f['error_rate'] for f in cluster_features]),
    }

    return profile


def infer_episode_conditions(ep_row: Dict[str, Any]) -> Dict[str, bool]:
    """
    Infer task conditions from episode metrics.

    These are boolean flags indicating what conditions were present.

    Args:
        ep_row: Episode data dict

    Returns:
        Dict of condition flags
    """
    features = episode_impact_features(ep_row)

    conditions = {
        # Wet conditions: high slip error rate
        'wet_conditions': features['slip_error_rate'] > 0.1,

        # Fragile objects: high breakage rate
        'fragile_objects': features['breakage_error_rate'] > 0.05,

        # Multi-object: more than 1 object
        'multi_object': features['num_objects'] > 1,

        # High occlusion: visibility challenges
        'high_occlusion': features['occlusion_level'] > 0.3,

        # Low lighting: poor visibility
        'low_lighting': features['lighting_quality'] < 0.7,

        # High throughput episode
        'high_throughput': features['mpl_units_per_hr'] > 100,

        # Perfect episode (no errors)
        'perfect_execution': features['error_rate'] == 0.0,

        # Energy efficient episode
        'energy_efficient': features['energy_wh_per_unit'] < 0.5,
    }

    return conditions


if __name__ == '__main__':
    """Test impact feature extraction."""
    print("Testing Impact Feature Extraction")
    print("="*60)

    # Create sample episode data
    sample_episode = {
        'mpl': 120.0,
        'error_rate': 0.15,
        'errors': 3,
        'attempts': 20,
        'completed': 17,
        'slip_errors': 2,
        'breakage_errors': 1,
        'energy_wh': 5.0,
        'shoulder_energy': 2.0,
        'elbow_energy': 2.0,
        'wrist_energy': 1.0,
        'robot_wage': 12.0,
        'wage_parity': 0.67,
        'delta_mpl': 10.0,
    }

    features = episode_impact_features(sample_episode)
    print("Episode Impact Features:")
    for k, v in features.items():
        print(f"  {k}: {v:.4f}")

    print()

    # Test condition inference
    conditions = infer_episode_conditions(sample_episode)
    print("Inferred Conditions:")
    for k, v in conditions.items():
        print(f"  {k}: {v}")

    print()

    # Test impact profile (with dummy global data)
    global_episodes = [sample_episode.copy() for _ in range(10)]
    cluster_episodes = [sample_episode.copy() for _ in range(3)]

    # Make cluster slightly better
    for ep in cluster_episodes:
        ep['mpl'] = 130.0
        ep['error_rate'] = 0.10

    profile = create_impact_profile(cluster_episodes, global_episodes)
    print("Impact Profile:")
    print(f"  Delta MPL: {profile['delta_mpl_units_per_hr']:+.2f} units/hr")
    print(f"  Delta Error Rate: {profile['delta_error_rate']:+.4f}")
    print(f"  Delta Energy/Unit: {profile['delta_energy_wh_per_unit']:+.4f} Wh")
    print()
    print("  Limb Energy Deltas:")
    for limb, delta in profile['limb_energy_deltas'].items():
        print(f"    {limb}: {delta:+.4f}")

    print("\n✅ Impact feature extraction complete!")
