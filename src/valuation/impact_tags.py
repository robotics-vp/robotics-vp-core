"""
Impact Tags (Phase B.4.6)

Converts raw impact deltas into human-readable performance improvement tags.
These tags answer: "What does this cluster help IMPROVE?"

Examples:
- "reduces slip errors in wet conditions"
- "improves multi-object throughput"
- "improves shoulder joint efficiency"
"""

from typing import Dict, List, Any


def derive_impact_tags(impact_profile: Dict[str, Any]) -> List[str]:
    """
    Derive human-readable impact tags from raw impact profile.

    Args:
        impact_profile: Dict with delta values for each performance dimension
            (output from create_impact_profile)

    Returns:
        List of human-readable tags describing what this cluster improves
    """
    tags = []

    # --- Core Performance Tags ---

    delta_mpl = impact_profile.get('delta_mpl_units_per_hr', 0.0)
    delta_error = impact_profile.get('delta_error_rate', 0.0)
    delta_energy = impact_profile.get('delta_energy_wh_per_unit', 0.0)

    # Throughput improvement
    if delta_mpl > 10.0:
        tags.append('significantly improves throughput')
    elif delta_mpl > 5.0:
        tags.append('moderately improves throughput')
    elif delta_mpl > 0.0:
        tags.append('slightly improves throughput')

    # Reliability improvement (reduced errors)
    if delta_error < -0.05:
        tags.append('significantly reduces error rate')
    elif delta_error < -0.02:
        tags.append('moderately reduces error rate')
    elif delta_error < 0.0:
        tags.append('slightly improves reliability')

    # Combined: high throughput + low errors
    if delta_mpl > 5.0 and delta_error < -0.02:
        tags.append('improves overall productivity')

    # Energy efficiency
    if delta_energy < -0.1:
        tags.append('significantly improves energy efficiency')
    elif delta_energy < -0.05:
        tags.append('moderately improves energy efficiency')
    elif delta_energy < 0.0:
        tags.append('slightly reduces energy consumption')

    # --- Error Type Specific Tags ---

    error_deltas = impact_profile.get('error_type_deltas', {})

    # Slip error reduction (wet/friction handling)
    slip_delta = error_deltas.get('slip', 0.0)
    if slip_delta < -0.03:
        tags.append('significantly reduces slip errors')
        tags.append('improves wet condition handling')
    elif slip_delta < -0.01:
        tags.append('reduces slip errors')

    # Breakage reduction (fragile handling)
    breakage_delta = error_deltas.get('breakage', 0.0)
    if breakage_delta < -0.02:
        tags.append('significantly reduces breakage')
        tags.append('improves fragile object handling')
    elif breakage_delta < -0.01:
        tags.append('reduces breakage risk')

    # Misalignment reduction (precision)
    misalign_delta = error_deltas.get('misalignment', 0.0)
    if misalign_delta < -0.02:
        tags.append('significantly improves precision')
    elif misalign_delta < -0.01:
        tags.append('improves alignment accuracy')

    # Gripper failure reduction
    gripper_delta = error_deltas.get('gripper_failure', 0.0)
    if gripper_delta < -0.01:
        tags.append('improves gripper reliability')

    # --- Per-Limb Efficiency Tags ---

    limb_deltas = impact_profile.get('limb_energy_deltas', {})

    # Shoulder efficiency
    shoulder_delta = limb_deltas.get('shoulder', 0.0)
    if shoulder_delta < -0.05:
        tags.append('significantly improves shoulder joint efficiency')
    elif shoulder_delta < -0.02:
        tags.append('improves shoulder energy usage')

    # Elbow efficiency
    elbow_delta = limb_deltas.get('elbow', 0.0)
    if elbow_delta < -0.05:
        tags.append('significantly improves elbow joint efficiency')
    elif elbow_delta < -0.02:
        tags.append('improves elbow energy usage')

    # Wrist efficiency
    wrist_delta = limb_deltas.get('wrist', 0.0)
    if wrist_delta < -0.05:
        tags.append('significantly improves wrist joint efficiency')
    elif wrist_delta < -0.02:
        tags.append('improves wrist energy usage')

    # Overall limb efficiency
    total_limb_delta = shoulder_delta + elbow_delta + wrist_delta
    if total_limb_delta < -0.1:
        tags.append('improves overall actuator efficiency')

    # --- Condition-Specific Tags ---

    condition_profile = impact_profile.get('condition_profile', {})

    # Wet conditions
    wet_tendency = condition_profile.get('wet_tendency', 0.0)
    if wet_tendency > 0.1 and slip_delta < 0:
        tags.append('handles wet conditions better than average')

    # Fragility
    fragility_tendency = condition_profile.get('fragility_tendency', 0.0)
    if fragility_tendency > 0.1 and breakage_delta < 0:
        tags.append('handles fragile objects better than average')

    # Multi-object
    multi_object_tendency = condition_profile.get('multi_object_tendency', 0.0)
    if multi_object_tendency > 0.5:
        if delta_mpl > 0:
            tags.append('improves multi-object throughput')
        if delta_error < 0:
            tags.append('improves multi-object reliability')

    # Occlusion
    occlusion_tendency = condition_profile.get('occlusion_tendency', 0.0)
    if occlusion_tendency > 0.1 and delta_error < 0:
        tags.append('handles occlusion challenges effectively')

    # --- Economic Tags ---

    economic_deltas = impact_profile.get('economic_deltas', {})

    wage_parity_delta = economic_deltas.get('wage_parity', 0.0)
    if wage_parity_delta > 0.1:
        tags.append('improves wage parity alignment')

    # If no tags, add neutral
    if not tags:
        tags.append('no significant impact detected')

    return tags


def generate_brick_description(
    semantic_tags: List[str],
    impact_tags: List[str],
    impact_profile: Dict[str, Any]
) -> str:
    """
    Generate human-readable description for a Data Brick.

    Combines semantic labels (what) with impact tags (what it improves).

    Args:
        semantic_tags: Visual/semantic labels (e.g., "fragile", "multi-object")
        impact_tags: Impact labels (e.g., "reduces slip errors")
        impact_profile: Raw impact profile dict

    Returns:
        Human-readable description string
    """
    # Format semantic tags
    semantic_str = ", ".join(semantic_tags) if semantic_tags else "general"

    # Format impact metrics
    delta_mpl = impact_profile.get('delta_mpl_units_per_hr', 0.0)
    delta_error = impact_profile.get('delta_error_rate', 0.0)
    delta_energy = impact_profile.get('delta_energy_wh_per_unit', 0.0)

    # Build description
    description = f"""This data brick contains episodes with {semantic_str} characteristics.

Performance Impact:
- MPL change: {delta_mpl:+.2f} units/hr vs global average
- Error rate change: {delta_error:+.4f} ({delta_error*100:+.2f}%)
- Energy per unit change: {delta_energy:+.4f} Wh

Key improvements:"""

    for tag in impact_tags[:5]:  # Top 5 impact tags
        description += f"\n  • {tag}"

    # Add limb-specific if significant
    limb_deltas = impact_profile.get('limb_energy_deltas', {})
    significant_limbs = []
    for limb, delta in limb_deltas.items():
        if abs(delta) > 0.02:
            change = "decreased" if delta < 0 else "increased"
            significant_limbs.append(f"{limb} ({change} by {abs(delta):.2f})")

    if significant_limbs:
        description += f"\n\nPer-limb energy impact: {', '.join(significant_limbs)}"

    return description


def rank_clusters_by_impact(
    cluster_profiles: List[Dict[str, Any]],
    dimension: str = 'mpl'
) -> List[int]:
    """
    Rank clusters by their impact on a specific performance dimension.

    Args:
        cluster_profiles: List of impact profiles for each cluster
        dimension: Which dimension to rank by:
            - 'mpl': throughput improvement
            - 'error': error reduction
            - 'energy': energy efficiency
            - 'slip': slip error reduction
            - 'breakage': breakage reduction
            - 'shoulder': shoulder efficiency
            - 'elbow': elbow efficiency
            - 'wrist': wrist efficiency

    Returns:
        List of cluster indices, sorted by impact (best first)
    """
    # Extract the relevant delta for ranking
    scores = []

    for i, profile in enumerate(cluster_profiles):
        if dimension == 'mpl':
            score = profile.get('delta_mpl_units_per_hr', 0.0)
        elif dimension == 'error':
            score = -profile.get('delta_error_rate', 0.0)  # Negative is better
        elif dimension == 'energy':
            score = -profile.get('delta_energy_wh_per_unit', 0.0)  # Negative is better
        elif dimension == 'slip':
            score = -profile.get('error_type_deltas', {}).get('slip', 0.0)
        elif dimension == 'breakage':
            score = -profile.get('error_type_deltas', {}).get('breakage', 0.0)
        elif dimension == 'shoulder':
            score = -profile.get('limb_energy_deltas', {}).get('shoulder', 0.0)
        elif dimension == 'elbow':
            score = -profile.get('limb_energy_deltas', {}).get('elbow', 0.0)
        elif dimension == 'wrist':
            score = -profile.get('limb_energy_deltas', {}).get('wrist', 0.0)
        else:
            score = 0.0

        scores.append((i, score))

    # Sort by score (descending)
    scores.sort(key=lambda x: x[1], reverse=True)

    return [idx for idx, _ in scores]


def filter_clusters_by_condition(
    cluster_profiles: List[Dict[str, Any]],
    condition: str
) -> List[int]:
    """
    Filter clusters that excel under specific conditions.

    Args:
        cluster_profiles: List of impact profiles
        condition: Condition to filter by:
            - 'wet': good at wet conditions
            - 'fragile': good at fragile handling
            - 'multi_object': good at multi-object scenarios
            - 'occluded': good at handling occlusion

    Returns:
        List of cluster indices that excel at this condition
    """
    matching = []

    for i, profile in enumerate(cluster_profiles):
        # Get condition tendency
        cond_profile = profile.get('condition_profile', {})

        # Check if cluster has high tendency for this condition AND good performance
        if condition == 'wet':
            if cond_profile.get('wet_tendency', 0.0) > 0.1:
                # And reduces slip errors
                if profile.get('error_type_deltas', {}).get('slip', 0.0) < 0:
                    matching.append(i)

        elif condition == 'fragile':
            if cond_profile.get('fragility_tendency', 0.0) > 0.1:
                # And reduces breakage
                if profile.get('error_type_deltas', {}).get('breakage', 0.0) < 0:
                    matching.append(i)

        elif condition == 'multi_object':
            if cond_profile.get('multi_object_tendency', 0.0) > 0.5:
                # And has good MPL or error rate
                if profile.get('delta_mpl_units_per_hr', 0.0) > 0 or \
                   profile.get('delta_error_rate', 0.0) < 0:
                    matching.append(i)

        elif condition == 'occluded':
            if cond_profile.get('occlusion_tendency', 0.0) > 0.1:
                # And has good performance
                if profile.get('delta_error_rate', 0.0) < 0:
                    matching.append(i)

    return matching


if __name__ == '__main__':
    """Test impact tag generation."""
    print("Testing Impact Tag Generation")
    print("="*60)

    # Create sample impact profile
    sample_profile = {
        'delta_mpl_units_per_hr': 18.5,
        'delta_error_rate': -0.07,
        'delta_energy_wh_per_unit': -0.12,

        'limb_energy_deltas': {
            'shoulder': -0.05,
            'elbow': -0.02,
            'wrist': +0.01,
        },

        'error_type_deltas': {
            'slip': -0.04,
            'breakage': -0.02,
            'misalignment': -0.01,
            'gripper_failure': 0.0,
        },

        'condition_profile': {
            'wet_tendency': 0.3,
            'fragility_tendency': 0.4,
            'multi_object_tendency': 1.2,
            'occlusion_tendency': 0.2,
        },

        'economic_deltas': {
            'wage_parity': 0.15,
            'delta_mpl': 0.0,
        },
    }

    # Generate impact tags
    tags = derive_impact_tags(sample_profile)
    print("Impact Tags:")
    for tag in tags:
        print(f"  • {tag}")

    print()

    # Generate brick description
    semantic_tags = ['fragile glassware', 'multi-object', 'occluded', 'low-light']
    description = generate_brick_description(semantic_tags, tags, sample_profile)
    print("Data Brick Description:")
    print("-"*60)
    print(description)
    print("-"*60)

    print("\n✅ Impact tag generation complete!")
