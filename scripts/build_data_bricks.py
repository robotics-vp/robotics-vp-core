#!/usr/bin/env python3
"""
Build Data Bricks with Semantic + Impact Labeling (Phase B.4.7)

Creates comprehensive Data Brick manifests that combine:
- SEMANTIC labels: what the data IS (fragile, multi-object, occluded, etc.)
- IMPACT labels: what the data IMPROVES (slip errors, MPL, energy efficiency, etc.)

Each Data Brick answers:
"This brick contains episodes with [semantic description] that tend to
[improve/reduce] [performance dimension] under [conditions]."

Usage:
    python scripts/build_data_bricks.py --rollouts data/physics_zv_rollouts.npz --clusters 5
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from collections import defaultdict


def load_rollout_data(rollouts_path):
    """Load z_V rollouts with episode metrics."""
    data = np.load(rollouts_path, allow_pickle=True)

    n_episodes = int(data['n_episodes'])
    latent_dim = int(data['latent_dim'])

    episodes = []
    for ep in range(n_episodes):
        ep_data = {
            'episode': ep,
            'z_sequence': data[f'ep_{ep}_z_sequence'],
            'actions': data[f'ep_{ep}_actions'],
            'length': int(data.get(f'ep_{ep}_length', len(data[f'ep_{ep}_actions']))),
        }

        # Load episode metrics if available
        for key in ['mpl', 'error_rate', 'robot_wage', 'wage_parity',
                    'completed', 'attempts', 'errors', 'time_hours']:
            metric_key = f'ep_{ep}_metric_{key}'
            if metric_key in data:
                ep_data[key] = float(data[metric_key])

        # Load novelty/dmpl if available
        if f'ep_{ep}_novelty' in data:
            ep_data['novelty'] = float(data[f'ep_{ep}_novelty'])
        if f'ep_{ep}_dmpl_estimate' in data:
            ep_data['dmpl_estimate'] = float(data[f'ep_{ep}_dmpl_estimate'])

        episodes.append(ep_data)

    return episodes, latent_dim


def cluster_episodes_by_zv(episodes, n_clusters=5):
    """
    Cluster episodes based on z_V representations.

    Uses mean z_V as episode-level feature.
    """
    # Compute mean z_V for each episode
    z_means = []
    for ep in episodes:
        z_seq = ep['z_sequence']
        z_mean = z_seq.mean(axis=0)
        z_means.append(z_mean)

    z_means = np.array(z_means)

    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(z_means)

    # Assign cluster labels to episodes
    for i, ep in enumerate(episodes):
        ep['cluster_id'] = int(cluster_labels[i])

    # Group episodes by cluster
    clusters = defaultdict(list)
    for ep in episodes:
        clusters[ep['cluster_id']].append(ep)

    return dict(clusters), kmeans


def infer_semantic_tags(cluster_episodes, all_episodes):
    """
    Infer semantic tags for a cluster based on episode characteristics.

    This is a placeholder - in production, would use CLIP/DINOv2 features.
    """
    if not cluster_episodes:
        return ['unknown']

    tags = []

    # Compute cluster statistics
    cluster_mpls = [ep.get('mpl', 0) for ep in cluster_episodes]
    cluster_errors = [ep.get('error_rate', 0) for ep in cluster_episodes]
    cluster_attempts = [ep.get('attempts', 0) for ep in cluster_episodes]

    global_mpl = np.mean([ep.get('mpl', 0) for ep in all_episodes])
    global_error = np.mean([ep.get('error_rate', 0) for ep in all_episodes])

    # Infer object type based on error patterns
    high_breakage = np.mean(cluster_errors) > 0.3
    if high_breakage:
        tags.append('fragile objects')

    # Infer complexity based on attempts
    avg_attempts = np.mean(cluster_attempts)
    if avg_attempts > 5:
        tags.append('multi-object scenarios')
    elif avg_attempts > 2:
        tags.append('moderate complexity')
    else:
        tags.append('single-object scenarios')

    # Infer conditions based on performance patterns
    if np.mean(cluster_errors) > global_error * 1.2:
        tags.append('challenging conditions')

    if np.mean(cluster_mpls) > global_mpl * 1.1:
        tags.append('high-throughput episodes')
    elif np.mean(cluster_mpls) < global_mpl * 0.9:
        tags.append('low-throughput episodes')

    # Add z_V space characteristics
    cluster_z_means = [ep['z_sequence'].mean(axis=0) for ep in cluster_episodes]
    z_variance = np.var(cluster_z_means, axis=0).mean()

    if z_variance < 0.01:
        tags.append('consistent visual features')
    elif z_variance > 0.05:
        tags.append('diverse visual features')

    return tags if tags else ['general']


def build_data_bricks(
    rollouts_path='data/physics_zv_rollouts.npz',
    n_clusters=5,
    output_dir='data/bricks',
):
    """
    Build comprehensive Data Brick manifests.

    Args:
        rollouts_path: Path to z_V rollouts
        n_clusters: Number of clusters (bricks)
        output_dir: Directory to save brick manifests

    Returns:
        List of brick manifests
    """
    # Import impact modules
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.valuation.impact_features import create_impact_profile
    from src.valuation.impact_tags import derive_impact_tags, generate_brick_description

    print(f"Building Data Bricks from {rollouts_path}")
    print(f"Number of clusters: {n_clusters}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    episodes, latent_dim = load_rollout_data(rollouts_path)
    print(f"Loaded {len(episodes)} episodes (latent_dim={latent_dim})")

    # Cluster episodes
    clusters, kmeans = cluster_episodes_by_zv(episodes, n_clusters)
    print(f"Clustered into {len(clusters)} groups")

    # Build brick manifests
    bricks = []

    for cluster_id in sorted(clusters.keys()):
        cluster_eps = clusters[cluster_id]

        print(f"\nProcessing Cluster {cluster_id} ({len(cluster_eps)} episodes)")

        # 1. Semantic tags (what the data IS)
        semantic_tags = infer_semantic_tags(cluster_eps, episodes)
        print(f"  Semantic: {semantic_tags}")

        # 2. Impact profile (what the data IMPROVES)
        impact_profile = create_impact_profile(cluster_eps, episodes)

        # 3. Impact tags (human-readable improvements)
        impact_tags = derive_impact_tags(impact_profile)
        print(f"  Impact: {impact_tags[:3]}...")

        # 4. Economic value
        if 'novelty' in cluster_eps[0]:
            mean_novelty = np.mean([ep.get('novelty', 0) for ep in cluster_eps])
            mean_dmpl = np.mean([ep.get('dmpl_estimate', 0) for ep in cluster_eps])
        else:
            mean_novelty = 0.0
            mean_dmpl = impact_profile['delta_mpl_units_per_hr']

        # 5. Generate comprehensive description
        description = generate_brick_description(semantic_tags, impact_tags, impact_profile)

        # 6. Build manifest
        brick = {
            'brick_id': f'brick_{cluster_id}',
            'cluster_id': cluster_id,
            'num_episodes': len(cluster_eps),
            'episode_ids': [ep['episode'] for ep in cluster_eps],

            # Semantic taxonomy
            'semantic_tags': semantic_tags,

            # Impact attribution
            'impact_tags': impact_tags,
            'impact_profile': impact_profile,

            # Economic value
            'economic_value': {
                'mean_novelty': mean_novelty,
                'mean_dmpl': mean_dmpl,
                'mean_mpl': impact_profile['cluster_stats']['mean_mpl'],
                'mean_error_rate': impact_profile['cluster_stats']['mean_error_rate'],
            },

            # Human-readable description
            'description': description,

            # Cluster centroid (for nearest-neighbor queries)
            'centroid': kmeans.cluster_centers_[cluster_id].tolist(),
        }

        bricks.append(brick)

    # Save manifests
    manifest_path = os.path.join(output_dir, 'data_bricks_manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(bricks, f, indent=2)
    print(f"\nSaved brick manifests to {manifest_path}")

    # Generate summary report
    generate_impact_report(bricks, output_dir)

    return bricks


def generate_impact_report(bricks, output_dir):
    """
    Generate comprehensive impact report answering key questions.

    Questions answered:
    - Which clusters most improve MPL under specific conditions?
    - Which clusters reduce specific error types?
    - Which clusters improve energy efficiency for specific actuators?
    """
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.valuation.impact_tags import rank_clusters_by_impact, filter_clusters_by_condition

    report = []
    report.append("="*70)
    report.append("DATA BRICK IMPACT REPORT")
    report.append("="*70)
    report.append("")

    # Extract profiles for ranking
    profiles = [b['impact_profile'] for b in bricks]

    # 1. Ranking by MPL improvement
    report.append("1. CLUSTERS RANKED BY MPL IMPROVEMENT")
    report.append("-"*70)
    mpl_ranking = rank_clusters_by_impact(profiles, 'mpl')
    for rank, idx in enumerate(mpl_ranking[:5], 1):
        brick = bricks[idx]
        delta_mpl = brick['impact_profile']['delta_mpl_units_per_hr']
        report.append(f"  #{rank}: {brick['brick_id']} (+{delta_mpl:.2f} units/hr)")
        report.append(f"       Semantic: {', '.join(brick['semantic_tags'])}")
        report.append(f"       Key impact: {brick['impact_tags'][0]}")
        report.append("")

    # 2. Ranking by error reduction
    report.append("2. CLUSTERS RANKED BY ERROR REDUCTION")
    report.append("-"*70)
    error_ranking = rank_clusters_by_impact(profiles, 'error')
    for rank, idx in enumerate(error_ranking[:5], 1):
        brick = bricks[idx]
        delta_err = brick['impact_profile']['delta_error_rate']
        report.append(f"  #{rank}: {brick['brick_id']} ({delta_err:+.4f} error rate)")
        report.append(f"       Semantic: {', '.join(brick['semantic_tags'])}")
        error_deltas = brick['impact_profile']['error_type_deltas']
        report.append(f"       Error breakdown: slip={error_deltas['slip']:+.4f}, "
                     f"breakage={error_deltas['breakage']:+.4f}")
        report.append("")

    # 3. Energy efficiency ranking
    report.append("3. CLUSTERS RANKED BY ENERGY EFFICIENCY")
    report.append("-"*70)
    energy_ranking = rank_clusters_by_impact(profiles, 'energy')
    for rank, idx in enumerate(energy_ranking[:5], 1):
        brick = bricks[idx]
        delta_energy = brick['impact_profile']['delta_energy_wh_per_unit']
        report.append(f"  #{rank}: {brick['brick_id']} ({delta_energy:+.4f} Wh/unit)")
        limb_deltas = brick['impact_profile']['limb_energy_deltas']
        report.append(f"       Limbs: shoulder={limb_deltas['shoulder']:+.4f}, "
                     f"elbow={limb_deltas['elbow']:+.4f}, "
                     f"wrist={limb_deltas['wrist']:+.4f}")
        report.append("")

    # 4. Condition-specific recommendations
    report.append("4. CONDITION-SPECIFIC RECOMMENDATIONS")
    report.append("-"*70)

    for condition in ['wet', 'fragile', 'multi_object', 'occluded']:
        matching = filter_clusters_by_condition(profiles, condition)
        report.append(f"  {condition.upper()} CONDITIONS:")
        if matching:
            for idx in matching[:3]:
                brick = bricks[idx]
                report.append(f"    - {brick['brick_id']}: {', '.join(brick['semantic_tags'][:2])}")
        else:
            report.append(f"    - No clusters optimized for this condition")
        report.append("")

    # 5. Per-limb efficiency leaders
    report.append("5. PER-LIMB EFFICIENCY LEADERS")
    report.append("-"*70)

    for limb in ['shoulder', 'elbow', 'wrist']:
        ranking = rank_clusters_by_impact(profiles, limb)
        best_idx = ranking[0]
        brick = bricks[best_idx]
        delta = brick['impact_profile']['limb_energy_deltas'][limb]
        report.append(f"  {limb.upper()}: {brick['brick_id']} ({delta:+.4f} Wh)")

    report.append("")
    report.append("="*70)
    report.append("END OF REPORT")
    report.append("="*70)

    # Save report
    report_text = "\n".join(report)
    report_path = os.path.join(output_dir, 'data_brick_impact_report.txt')
    with open(report_path, 'w') as f:
        f.write(report_text)

    print(f"\nSaved impact report to {report_path}")
    print("\nKey findings:")
    print(report_text[:2000])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build Data Bricks with semantic + impact labels')
    parser.add_argument(
        '--rollouts',
        type=str,
        default='data/physics_zv_rollouts.npz',
        help='Path to z_V rollouts'
    )
    parser.add_argument(
        '--clusters',
        type=int,
        default=5,
        help='Number of clusters (bricks)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/bricks',
        help='Directory to save brick manifests'
    )

    args = parser.parse_args()

    print("="*60)
    print("Phase B.4: Building Data Bricks")
    print("="*60)

    bricks = build_data_bricks(
        rollouts_path=args.rollouts,
        n_clusters=args.clusters,
        output_dir=args.output_dir,
    )

    print("\n" + "="*60)
    print("Data Brick Summary")
    print("="*60)

    for brick in bricks:
        print(f"\n{brick['brick_id']}:")
        print(f"  Episodes: {brick['num_episodes']}")
        print(f"  Semantic: {', '.join(brick['semantic_tags'][:3])}")
        print(f"  Top impacts: {', '.join(brick['impact_tags'][:3])}")
        print(f"  Economic value: MPL={brick['economic_value']['mean_mpl']:.2f}, "
              f"Novelty={brick['economic_value']['mean_novelty']:.4f}")

    print("\n" + "="*60)
    print("Next steps:")
    print("="*60)
    print("1. Use bricks for targeted training:")
    print("   - Upsample bricks with high ΔMPL for specific conditions")
    print("   - Price bricks based on semantic + impact value")
    print()
    print("2. Query system:")
    print("   - 'Which bricks improve slip errors in wet conditions?'")
    print("   - 'Which bricks have best shoulder joint efficiency?'")
    print("="*60)
