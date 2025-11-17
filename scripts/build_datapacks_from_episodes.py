#!/usr/bin/env python3
"""
Build DataPacks from Drawer+Vase Episodes.

Converts episode data into two-bucket taxonomy (positive/negative).
"""

import os
import sys
import json
import argparse
import numpy as np
from typing import List, Dict, Any, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.valuation.datapack_schema import (
    ConditionProfile,
    AttributionProfile,
    SimaAnnotation,
    DataPackMeta,
    create_positive_datapack,
    create_negative_datapack,
)
from src.valuation.datapack_repo import DataPackRepo
from src.config.econ_params import EconParams


def load_episode_data(data_dir: str) -> List[Dict[str, Any]]:
    """
    Load episode data from directory.

    Expected format: JSON files with episode summaries.

    Args:
        data_dir: Directory containing episode JSON files

    Returns:
        List of episode dictionaries
    """
    episodes = []

    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} does not exist")
        return episodes

    # Look for JSON files
    for filename in sorted(os.listdir(data_dir)):
        if filename.endswith('.json') and 'episode' in filename.lower():
            filepath = os.path.join(data_dir, filename)
            with open(filepath, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    episodes.extend(data)
                else:
                    episodes.append(data)

    return episodes


def compute_delta_j(
    episode: Dict[str, Any],
    baseline: Dict[str, Any],
    econ: EconParams,
    objective_weights: Optional[List[float]] = None
) -> float:
    """
    Compute ΔJ (change in meta-objective) for an episode.

    J = α_mpl * ΔMPL - α_error * Δerror + α_ep * ΔEP + α_safety * safety_bonus

    Args:
        episode: Episode data dict
        baseline: Baseline episode data
        econ: Economic parameters
        objective_weights: [α_mpl, α_error, α_ep, α_safety] weights

    Returns:
        ΔJ value (positive = improvement, negative = regression)
    """
    if objective_weights is None:
        objective_weights = [1.0, 1.0, 1.0, 0.5]  # Default weights

    alpha_mpl, alpha_error, alpha_ep, alpha_safety = objective_weights

    # Extract metrics (with fallbacks)
    ep_mpl = episode.get('mpl_episode', episode.get('mpl', 0))
    ep_error = episode.get('error_rate_episode', episode.get('error_rate', 0))
    ep_ep = episode.get('ep_episode', episode.get('ep', 1.0))
    ep_vase_broken = episode.get('vase_broken', False)

    base_mpl = baseline.get('mpl_episode', baseline.get('mpl', ep_mpl))
    base_error = baseline.get('error_rate_episode', baseline.get('error_rate', ep_error))
    base_ep = baseline.get('ep_episode', baseline.get('ep', ep_ep))

    # Compute deltas (normalized)
    delta_mpl = (ep_mpl - base_mpl) / max(base_mpl, 1e-6)
    delta_error = (ep_error - base_error)  # Lower is better
    delta_ep = (ep_ep - base_ep) / max(base_ep, 1e-6)

    # Safety bonus (negative if vase broken)
    safety_bonus = -10.0 if ep_vase_broken else 0.0

    # Compute J
    delta_j = (
        alpha_mpl * delta_mpl
        - alpha_error * delta_error
        + alpha_ep * delta_ep
        + alpha_safety * safety_bonus
    )

    return delta_j


def create_condition_profile(
    episode: Dict[str, Any],
    econ: EconParams,
    engine_type: str = "pybullet"
) -> ConditionProfile:
    """
    Create ConditionProfile from episode data.

    Args:
        episode: Episode data dict
        econ: Economic parameters
        engine_type: Simulation engine type

    Returns:
        ConditionProfile instance
    """
    # Extract environment conditions
    vase_offset = tuple(episode.get('vase_offset', [0.0, 0.0, 0.0]))
    drawer_friction = episode.get('drawer_friction', 0.3)
    lighting = episode.get('lighting_profile', 'normal')
    occlusion = episode.get('occlusion_level', 0.0)

    # Economic regime
    objective_vector = episode.get('objective_vector', [1.0, 1.0, 1.0, 0.5])

    # Tags
    tags = {}
    if 'fragile' in episode:
        tags['fragile'] = episode['fragile']
    if 'multi_object' in episode:
        tags['multi_object'] = episode['multi_object']
    if vase_offset[0] != 0 or vase_offset[1] != 0 or vase_offset[2] != 0:
        tags['has_offset'] = True

    return ConditionProfile(
        task_name="drawer_vase",
        engine_type=engine_type,
        world_id=f"{engine_type}_drawer_v1",
        vase_offset=vase_offset,
        drawer_friction=drawer_friction,
        lighting_profile=lighting,
        occlusion_level=occlusion,
        econ_preset="drawer_vase",
        price_per_unit=econ.price_per_unit,
        vase_break_cost=50.0,  # High cost for fragile objects
        energy_price_kWh=econ.energy_price_kWh,
        objective_vector=objective_vector,
        tags=tags
    )


def create_attribution_profile(
    episode: Dict[str, Any],
    baseline: Dict[str, Any],
    delta_j: float,
    trust_score: float = 0.0,
    w_econ: float = 0.0,
    source_type: str = "real"
) -> AttributionProfile:
    """
    Create AttributionProfile from episode data.

    Args:
        episode: Episode data dict
        baseline: Baseline episode data
        delta_j: Computed ΔJ value
        trust_score: Trust network score (0-1)
        w_econ: Economic weight from lattice
        source_type: "real", "synthetic", "hybrid"

    Returns:
        AttributionProfile instance
    """
    # Compute deltas
    ep_mpl = episode.get('mpl_episode', episode.get('mpl', 0))
    ep_error = episode.get('error_rate_episode', episode.get('error_rate', 0))
    ep_ep = episode.get('ep_episode', episode.get('ep', 1.0))

    base_mpl = baseline.get('mpl_episode', baseline.get('mpl', ep_mpl))
    base_error = baseline.get('error_rate_episode', baseline.get('error_rate', ep_error))
    base_ep = baseline.get('ep_episode', baseline.get('ep', ep_ep))

    delta_mpl = ep_mpl - base_mpl
    delta_error = ep_error - base_error
    delta_ep = ep_ep - base_ep

    # Wage parity
    wage_parity = episode.get('wage_parity', 0.0)
    base_wage_parity = baseline.get('wage_parity', wage_parity)
    delta_wage_parity = wage_parity - base_wage_parity

    # World model metadata (if available)
    wm_horizon = episode.get('wm_horizon', 0)
    wm_trust = episode.get('wm_trust_over_horizon', [])

    # Lambda budget
    lambda_budget = episode.get('lambda_budget', 0.0)

    return AttributionProfile(
        delta_mpl=delta_mpl,
        delta_error=delta_error,
        delta_ep=delta_ep,
        delta_wage_parity=delta_wage_parity,
        delta_J=delta_j,
        trust_score=trust_score,
        w_econ=w_econ,
        lambda_budget=lambda_budget,
        world_model_horizon=wm_horizon,
        world_model_trust_over_horizon=wm_trust,
        source_type=source_type,
        wm_model_id=episode.get('wm_model_id'),
        wm_horizon_used=episode.get('wm_horizon_used'),
        wm_branch_depth=episode.get('wm_branch_depth'),
        wm_trust_over_horizon=episode.get('wm_trust_over_horizon'),
        mvd_score=episode.get('mvd_score'),
        econ_weight_final=trust_score * w_econ if trust_score > 0 and w_econ > 0 else None
    )


def extract_skill_trace(episode: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract skill trace from episode data.

    Args:
        episode: Episode data dict

    Returns:
        List of skill trace entries
    """
    # Check for explicit skill trace
    if 'skill_trace' in episode:
        return episode['skill_trace']

    # Try to infer from actions
    if 'actions' not in episode:
        # Generate default skill sequence for drawer_vase
        return [
            {'t': 0, 'skill_id': 0, 'params': {}, 'duration': 10},  # LOCATE_DRAWER
            {'t': 10, 'skill_id': 1, 'params': {}, 'duration': 10},  # LOCATE_VASE
            {'t': 20, 'skill_id': 2, 'params': {}, 'duration': 20},  # PLAN_SAFE_APPROACH
            {'t': 40, 'skill_id': 3, 'params': {}, 'duration': 30},  # GRASP_HANDLE
            {'t': 70, 'skill_id': 4, 'params': {}, 'duration': 40},  # OPEN_WITH_CLEARANCE
            {'t': 110, 'skill_id': 5, 'params': {}, 'duration': 20},  # RETRACT_SAFE
        ]

    # Try to segment actions into skills (simplified)
    actions = episode['actions']
    n_actions = len(actions)

    # Simple segmentation: 6 skills, roughly equal
    skills_per_episode = 6
    steps_per_skill = max(1, n_actions // skills_per_episode)

    skill_trace = []
    for i in range(skills_per_episode):
        start_t = i * steps_per_skill
        skill_trace.append({
            't': start_t,
            'skill_id': i,
            'params': {},
            'duration': steps_per_skill
        })

    return skill_trace


def extract_semantic_tags(episode: Dict[str, Any]) -> List[str]:
    """
    Extract semantic tags from episode.

    Args:
        episode: Episode data dict

    Returns:
        List of semantic tag strings
    """
    tags = []

    # Extract from episode data
    if episode.get('vase_broken'):
        tags.append('vase_collision')

    if episode.get('drawer_stuck'):
        tags.append('drawer_jam')

    if episode.get('success', False):
        tags.append('task_complete')

    # Environment conditions
    vase_offset = episode.get('vase_offset', [0, 0, 0])
    if abs(vase_offset[0]) > 0.1 or abs(vase_offset[1]) > 0.1:
        tags.append('offset_vase')

    if episode.get('drawer_friction', 0.3) > 0.5:
        tags.append('high_friction')

    if episode.get('occlusion_level', 0) > 0.3:
        tags.append('partial_occlusion')

    if episode.get('lighting_profile') == 'low_light':
        tags.append('low_light')

    # Default task-specific tags
    tags.append('fragile_glassware')
    tags.append('top_drawer')

    return tags


def generate_counterfactual_plan(
    episode: Dict[str, Any],
    failure_reason: str = "unknown"
) -> Dict[str, Any]:
    """
    Generate counterfactual plan for negative datapacks.

    Args:
        episode: Failed episode data
        failure_reason: Why the episode failed

    Returns:
        Counterfactual plan dict
    """
    # Standard successful plan for drawer_vase
    standard_plan = {
        'skills': [0, 1, 2, 3, 4, 5],
        'waypoints': [
            [0.5, 0.0, 0.3],  # Approach position
            [0.5, 0.0, 0.35],  # Clearance position
            [0.6, 0.0, 0.35],  # Pull back position
        ],
        'source': 'scripted_teacher',
        'params': {
            'target_clearance': 0.15,
            'approach_speed': 0.8,
            'grasp_force': 0.5,
            'retract_speed': 0.5
        }
    }

    # Adjust based on failure
    if 'vase' in failure_reason.lower() or episode.get('vase_broken'):
        # Increase clearance for vase collision
        standard_plan['params']['target_clearance'] = 0.25
        standard_plan['source'] = 'hrl_teacher'

    if 'friction' in failure_reason.lower():
        # Adjust grasp and pull parameters
        standard_plan['params']['grasp_force'] = 0.7
        standard_plan['params']['approach_speed'] = 0.6

    return standard_plan


def build_datapacks_from_episodes(
    episodes: List[Dict[str, Any]],
    econ: EconParams,
    baseline: Optional[Dict[str, Any]] = None,
    trust_scores: Optional[List[float]] = None,
    w_econ_scores: Optional[List[float]] = None,
    objective_weights: Optional[List[float]] = None,
    engine_type: str = "pybullet"
) -> List[DataPackMeta]:
    """
    Convert episodes to DataPackMeta objects.

    Args:
        episodes: List of episode data dicts
        econ: Economic parameters
        baseline: Baseline episode (for comparison)
        trust_scores: Optional trust scores per episode
        w_econ_scores: Optional economic weights per episode
        objective_weights: Objective function weights
        engine_type: Simulation engine type

    Returns:
        List of DataPackMeta objects
    """
    if not episodes:
        return []

    # Use first episode as baseline if not provided
    if baseline is None:
        baseline = episodes[0]

    # Default scores
    if trust_scores is None:
        trust_scores = [0.9] * len(episodes)
    if w_econ_scores is None:
        w_econ_scores = [1.0] * len(episodes)

    datapacks = []

    for i, episode in enumerate(episodes):
        # Compute ΔJ
        delta_j = compute_delta_j(episode, baseline, econ, objective_weights)

        # Create profiles
        condition = create_condition_profile(episode, econ, engine_type)
        attribution = create_attribution_profile(
            episode, baseline, delta_j,
            trust_score=trust_scores[i % len(trust_scores)],
            w_econ=w_econ_scores[i % len(w_econ_scores)],
            source_type=episode.get('source_type', 'real')
        )

        # Extract skill trace
        skill_trace = extract_skill_trace(episode)

        # Extract semantic tags
        semantic_tags = extract_semantic_tags(episode)

        # SIMA annotation (if available)
        sima_annotation = None
        if 'instruction' in episode or 'narrations' in episode:
            sima_annotation = SimaAnnotation(
                instruction=episode.get('instruction', 'open the drawer without hitting the vase'),
                step_narrations=episode.get('narrations', []),
                sima_agent_id=episode.get('sima_agent_id', 'sima_v1'),
                source_world=f"{engine_type}_drawer_v1",
                derived_skill_plan=[s['skill_id'] for s in skill_trace]
            )
            sima_annotation.compute_stats()

        # Classify into bucket
        if delta_j >= 0:
            # Positive bucket (improved J)
            datapack = create_positive_datapack(
                task_name="drawer_vase",
                condition=condition,
                attribution=attribution,
                skill_trace=skill_trace,
                semantic_tags=semantic_tags,
                sima_annotation=sima_annotation,
                episode_id=episode.get('episode_id', str(i))
            )
        else:
            # Negative bucket (worsened J)
            # Generate counterfactual plan
            failure_reason = episode.get('failure_reason', 'unknown')
            counterfactual = generate_counterfactual_plan(episode, failure_reason)

            datapack = create_negative_datapack(
                task_name="drawer_vase",
                condition=condition,
                attribution=attribution,
                skill_trace=skill_trace,
                counterfactual_plan=counterfactual,
                counterfactual_source=counterfactual['source'],
                semantic_tags=semantic_tags,
                sima_annotation=sima_annotation,
                episode_id=episode.get('episode_id', str(i))
            )

        datapacks.append(datapack)

    return datapacks


def generate_synthetic_episodes(n: int = 100) -> List[Dict[str, Any]]:
    """
    Generate synthetic episode data for testing.

    Args:
        n: Number of episodes to generate

    Returns:
        List of episode dicts
    """
    episodes = []

    for i in range(n):
        # Random performance variations
        base_mpl = 10.0  # units/hour
        base_error = 0.05
        base_ep = 2.0  # units/Wh

        # Add variation
        mpl = base_mpl * (1.0 + np.random.randn() * 0.3)
        error = max(0, base_error + np.random.randn() * 0.02)
        ep = base_ep * (1.0 + np.random.randn() * 0.2)

        # Occasionally break vase
        vase_broken = np.random.random() < 0.1

        # Environment conditions
        vase_offset = (
            np.random.uniform(-0.1, 0.1),
            np.random.uniform(-0.1, 0.1),
            0.0
        )
        drawer_friction = np.random.uniform(0.2, 0.5)

        episode = {
            'episode_id': f'ep_{i:04d}',
            'mpl_episode': mpl,
            'error_rate_episode': error,
            'ep_episode': ep,
            'vase_broken': vase_broken,
            'success': not vase_broken and error < 0.1,
            'vase_offset': list(vase_offset),
            'drawer_friction': drawer_friction,
            'lighting_profile': 'normal',
            'occlusion_level': 0.0,
            'wage_parity': mpl / 20.0,  # Relative to human
            'objective_vector': [1.0, 1.0, 1.0, 0.5],
        }

        if vase_broken:
            episode['failure_reason'] = 'vase_collision'

        episodes.append(episode)

    return episodes


def main():
    parser = argparse.ArgumentParser(description='Build DataPacks from episodes')
    parser.add_argument('--data-dir', type=str, default='data/episodes',
                        help='Directory containing episode data')
    parser.add_argument('--output-dir', type=str, default='data/datapacks',
                        help='Directory for datapack repository')
    parser.add_argument('--task', type=str, default='drawer_vase',
                        help='Task name')
    parser.add_argument('--engine', type=str, default='pybullet',
                        choices=['pybullet', 'isaac', 'ue5'],
                        help='Simulation engine type')
    parser.add_argument('--generate-synthetic', type=int, default=0,
                        help='Generate N synthetic episodes for testing')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')

    args = parser.parse_args()

    print("=" * 70)
    print("BUILDING DATAPACKS FROM EPISODES")
    print("=" * 70)

    # Load or generate episodes
    if args.generate_synthetic > 0:
        print(f"Generating {args.generate_synthetic} synthetic episodes...")
        episodes = generate_synthetic_episodes(args.generate_synthetic)
    else:
        print(f"Loading episodes from {args.data_dir}...")
        episodes = load_episode_data(args.data_dir)

    if not episodes:
        print("No episodes found. Use --generate-synthetic to create test data.")
        return

    print(f"Found {len(episodes)} episodes")

    # Create economic parameters
    econ = EconParams(
        price_per_unit=5.0,  # $/successful drawer open
        mpl_human=20.0,  # Units/hour (human benchmark)
        wage_human=18.0,  # $/hour
        energy_price_kWh=0.12,  # $/kWh
        energy_Wh_per_attempt=5.0,  # Wh per attempt
        max_error_rate_sla=0.10,  # 10% max error rate
        damage_cost_per_error=50.0  # Cost if vase breaks
    )

    # Build datapacks
    print("Converting episodes to datapacks...")
    datapacks = build_datapacks_from_episodes(
        episodes=episodes,
        econ=econ,
        engine_type=args.engine
    )

    # Analyze buckets
    positive_count = sum(1 for dp in datapacks if dp.bucket == 'positive')
    negative_count = sum(1 for dp in datapacks if dp.bucket == 'negative')

    print(f"\nDatapack Statistics:")
    print(f"  Total: {len(datapacks)}")
    print(f"  Positive (ΔJ ≥ 0): {positive_count} ({100*positive_count/len(datapacks):.1f}%)")
    print(f"  Negative (ΔJ < 0): {negative_count} ({100*negative_count/len(datapacks):.1f}%)")

    # Save to repository
    print(f"\nSaving to repository: {args.output_dir}")
    repo = DataPackRepo(base_dir=args.output_dir)

    repo.append_batch(datapacks)

    # Show statistics
    stats = repo.get_statistics(args.task)
    print(f"\nRepository Statistics for '{args.task}':")
    print(f"  Total datapacks: {stats['total']}")
    print(f"  Positive: {stats['positive']}")
    print(f"  Negative: {stats['negative']}")
    print(f"  Positive ratio: {stats['positive_ratio']:.2%}")
    print(f"  ΔJ mean: {stats['delta_j_mean']:.4f}")
    print(f"  ΔJ std: {stats['delta_j_std']:.4f}")
    print(f"  ΔJ range: [{stats['delta_j_min']:.4f}, {stats['delta_j_max']:.4f}]")
    print(f"  Trust mean: {stats['trust_mean']:.4f}")
    print(f"  Unique skills: {stats['unique_skills']}")
    print(f"  With SIMA: {stats['with_sima']}")
    print(f"  With counterfactual: {stats['with_counterfactual']}")

    if args.verbose:
        print("\nSample DataPacks:")
        for i, dp in enumerate(datapacks[:5]):
            print(f"  {dp.summary()}")

    print("\nDatapack building complete!")


if __name__ == "__main__":
    main()
