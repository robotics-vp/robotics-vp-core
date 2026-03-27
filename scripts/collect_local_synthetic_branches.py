#!/usr/bin/env python3
"""
Collect Local Synthetic Branches from Stable World Model

Conservative augmentation strategy:
- Start from REAL z_t (known-good states)
- Roll forward only H_short steps (5-10)
- Gate hard: trust > 0.9, std ratio in [0.8, 1.2]
- Tag with brick_id and economic metrics

This is the safest way to use the world model:
"Fill in local bubbles around real states" instead of hallucinating entire episodes.

Usage:
    python scripts/collect_local_synthetic_branches.py
    python scripts/collect_local_synthetic_branches.py --horizon 10 --min-trust 0.95
"""
# NOTE: Experimental configuration;
# actual synthetic weighting is DL-driven (trust × w_econ).
# TODO: migrate to full PolicyProfile after demo.

import os
import sys
import argparse
import json
import numpy as np
import torch

sys.path.insert(0, str(os.path.dirname(os.path.dirname(__file__))))
from src.world_model.contractive_dynamics import StableWorldModel
from src.valuation.trust_net import TrustNet
from src.config.internal_profile import get_internal_experiment_profile
from src.world_model.sim_synth_physics import (
    assess_local_branch_corpus_gen2sim,
    build_synthetic_branch_corpus_metadata,
    collect_local_synthetic_branch_records,
)


def _load_coverage_graph(path):
    """Load a coverage graph from JSON for gap-aware branch selection."""
    try:
        from src.world_model.semantic_coverage_graph import SemanticCoverageGraph
        with open(path) as f:
            data = json.load(f)
        return SemanticCoverageGraph.from_dict(data)
    except Exception as e:
        print(f"WARNING: Could not load coverage graph from {path}: {e}")
        return None


def _load_source_runtime_metadata(path):
    """Load optional seed/runtime metadata describing real grounding provenance."""
    if not path:
        return {}
    if not os.path.exists(path):
        print(f"WARNING: Source runtime metadata not found at {path}")
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else {}
    except Exception as exc:
        print(f"WARNING: Could not parse source runtime metadata from {path}: {exc}")
        return {}


def _resolve_runtime_field(source_metadata, explicit_value, *keys, default=""):
    if explicit_value:
        return explicit_value
    for key in keys:
        value = source_metadata.get(key)
        if value not in (None, ""):
            return value
    return default


def load_brick_manifest(manifest_path):
    """Load brick manifest if available."""
    if os.path.exists(manifest_path):
        with open(manifest_path, 'r') as f:
            return json.load(f)
    return None


def get_episode_brick_id(ep_idx, brick_manifest):
    """Get brick ID for an episode."""
    if brick_manifest is None:
        return -1  # Unknown brick

    # brick_manifest is a list of brick objects
    for brick in brick_manifest:
        if ep_idx in brick.get('episode_ids', []):
            # Extract numeric brick ID from string like "brick_0"
            brick_id_str = brick.get('brick_id', 'brick_-1')
            try:
                return int(brick_id_str.split('_')[1])
            except Exception:
                return -1
    return -1


def main():
    # Load experiment profile for defaults
    profile = get_internal_experiment_profile("default")

    parser = argparse.ArgumentParser(description='Collect local synthetic branches')
    parser.add_argument('--world-model', type=str, default=profile['world_model_path'])
    parser.add_argument('--dataset', type=str, default=profile['real_data_path'])
    parser.add_argument('--trust-net', type=str, default=profile['trust_net_path'])
    parser.add_argument('--brick-manifest', type=str, default=profile['brick_manifest_path'])
    parser.add_argument('--output', type=str, default=profile['synthetic_branches_path'])

    # Branch parameters
    parser.add_argument('--horizon', type=int, default=profile['max_branch_horizon'],
                        help='Branch length (steps)')
    parser.add_argument('--branches-per-episode', type=int, default=profile['branches_per_episode'],
                        help='Branches to sample per episode')

    # Gating thresholds
    parser.add_argument('--min-trust', type=float, default=profile['min_trust_threshold'],
                        help='Minimum trust score')
    parser.add_argument('--min-std-ratio', type=float, default=profile['min_std_ratio'],
                        help='Minimum std ratio')
    parser.add_argument('--max-std-ratio', type=float, default=profile['max_std_ratio'],
                        help='Maximum std ratio')

    # Objective conditioning (for future use)
    parser.add_argument('--objective-dim', type=int, default=profile['objective_dim'],
                        help='Dimension of objective vector')

    # Semantic coverage graph integration (Section G)
    parser.add_argument('--use-coverage-graph', action='store_true',
                        help='Use coverage graph for gap-aware seed selection and labeling')
    parser.add_argument('--coverage-graph-path', type=str,
                        default='data/coverage/coverage_graph.json',
                        help='Path to coverage_graph.json from coverage loop')
    parser.add_argument('--source-runtime-metadata', type=str, default='',
                        help='Optional JSON metadata describing the real seed/runtime grounding status')
    parser.add_argument('--scene-tracks-backend', type=str, default='',
                        help='Optional explicit SceneTracks backend for the seed corpus (real/passthrough/stub/unavailable)')
    parser.add_argument('--teacher-runtime-backend', type=str, default='',
                        help='Optional explicit teacher-runtime backend for the seed corpus (real/stub/unavailable)')
    parser.add_argument('--vision-backbone-selected', type=str, default='',
                        help='Optional explicit vision-backbone status for the seed corpus (real/stub/unavailable)')
    parser.add_argument('--semantic-grounding-mode', type=str, default='',
                        help='Optional semantic grounding mode for the seed corpus (non_heuristic/heuristic_fallback/keyword_tags)')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("="*70)
    print("COLLECTING LOCAL SYNTHETIC BRANCHES")
    print("="*70)
    print(f"Branch horizon: {args.horizon} steps")
    print(f"Branches per episode: {args.branches_per_episode}")
    print(f"Trust threshold: >= {args.min_trust}")
    print(f"Std ratio range: [{args.min_std_ratio}, {args.max_std_ratio}]")
    print()

    # Load real data
    print(f"Loading real data from {args.dataset}...")
    data = np.load(args.dataset, allow_pickle=True)
    n_episodes = int(data['n_episodes'])
    latent_dim = int(data['latent_dim'])

    episodes = []
    all_z = []
    for ep in range(n_episodes):
        z_seq = data[f'ep_{ep}_z_sequence']
        actions = data[f'ep_{ep}_actions']
        episodes.append({
            'z_sequence': torch.FloatTensor(z_seq).to(device),
            'actions': torch.FloatTensor(actions).to(device),
            'length': len(actions),
        })
        all_z.extend(z_seq)
    all_z = np.array(all_z)
    action_dim = episodes[0]['actions'].shape[1]

    real_z_mean = all_z.mean()
    real_z_std = all_z.std()
    print(f"Real z_V: mean={real_z_mean:.6f}, std={real_z_std:.6f}")
    print(f"Loaded {n_episodes} episodes")

    # Load brick manifest
    print(f"\nLoading brick manifest from {args.brick_manifest}...")
    brick_manifest = load_brick_manifest(args.brick_manifest)
    if brick_manifest:
        print(f"Loaded {len(brick_manifest)} bricks")
    else:
        print("No brick manifest found, brick_id will be -1")

    # Load world model
    print(f"\nLoading stable world model from {args.world_model}...")
    wm_ckpt = torch.load(args.world_model, map_location=device, weights_only=False)
    world_model = StableWorldModel(
        latent_dim=latent_dim,
        action_dim=action_dim,
        hidden_dim=wm_ckpt.get('hidden_dim', 256),
        n_layers=wm_ckpt.get('n_layers', 3),
        alpha_init=wm_ckpt.get('alpha_init', 0.3),
        max_delta=wm_ckpt.get('max_delta', 0.15),
    )
    world_model.load_state_dict(wm_ckpt['model_state_dict'])
    world_model = world_model.to(device)
    world_model.eval()
    print(f"Model alpha: {world_model.dynamics.alpha.item():.4f}")

    # Load trust_net
    print(f"Loading trust_net from {args.trust_net}...")
    trust_ckpt = torch.load(args.trust_net, map_location=device, weights_only=False)
    trust_net = TrustNet(input_dim=6, hidden_dim=64)
    trust_net.load_state_dict(trust_ckpt['model_state_dict'])
    trust_net = trust_net.to(device)
    trust_net.eval()
    trust_mean = torch.FloatTensor(trust_ckpt['X_mean']).to(device)
    trust_std_norm = torch.FloatTensor(trust_ckpt['X_std']).to(device)

    # Load coverage graph if requested
    coverage_graph = None
    if args.use_coverage_graph:
        print(f"\nLoading coverage graph from {args.coverage_graph_path}...")
        coverage_graph = _load_coverage_graph(args.coverage_graph_path)
        if coverage_graph:
            summary = coverage_graph.coverage_summary()
            print(f"  Total edges: {summary.get('total_edges', 0)}")
            print(f"  Missing edges: {summary.get('missing_edges', 0)}")
            print(f"  Coverage: {summary.get('coverage_ratio', 0):.2%}")
        else:
            print("  Coverage graph not available, proceeding without gap awareness")

    source_runtime_metadata = _load_source_runtime_metadata(args.source_runtime_metadata)
    scene_tracks_backend = str(
        _resolve_runtime_field(
            source_runtime_metadata,
            args.scene_tracks_backend,
            "scene_tracks_backend",
        )
        or "unavailable"
    )
    teacher_runtime_backend = str(
        _resolve_runtime_field(
            source_runtime_metadata,
            args.teacher_runtime_backend,
            "teacher_runtime_backend_selected",
            "openvla_backend_selected",
        )
        or "unavailable"
    )
    vision_backbone_selected = str(
        _resolve_runtime_field(
            source_runtime_metadata,
            args.vision_backbone_selected,
            "vision_backbone_selected",
            "openvla_vision_backbone_selected",
        )
        or "unavailable"
    )
    semantic_grounding_mode = str(
        _resolve_runtime_field(
            source_runtime_metadata,
            args.semantic_grounding_mode,
            "semantic_grounding_mode",
            "grounding_mode",
        )
        or ("non_heuristic" if scene_tracks_backend == "real" else "heuristic_fallback")
    )
    semantic_memory_grounded = bool(
        source_runtime_metadata.get("semantic_memory_grounded", False)
        or source_runtime_metadata.get("grounded_track_object_count", 0)
        or scene_tracks_backend == "real"
    )

    # Collect branches
    print("\n" + "="*70)
    print("GENERATING BRANCHES")
    if coverage_graph:
        print("  (gap-aware mode: seeds from under-covered states)")
    print("="*70)

    branches, stats = collect_local_synthetic_branch_records(
        episodes=episodes,
        world_model=world_model,
        trust_net=trust_net,
        trust_mean=trust_mean,
        trust_std_norm=trust_std_norm,
        real_z_std=real_z_std,
        horizon=args.horizon,
        branches_per_episode=args.branches_per_episode,
        min_trust=args.min_trust,
        min_std_ratio=args.min_std_ratio,
        max_std_ratio=args.max_std_ratio,
        objective_vector=profile['default_objective_vector'],
        coverage_graph=coverage_graph,
        brick_manifest=brick_manifest,
        brick_id_fn=get_episode_brick_id,
    )

    # Summary
    print("\n" + "="*70)
    print("COLLECTION SUMMARY")
    print("="*70)
    print(f"Total branches attempted: {stats['total_attempted']}")
    print(f"Passed trust gate (>= {args.min_trust}): {stats['passed_trust']} "
          f"({100*stats['passed_trust']/max(1,stats['total_attempted']):.1f}%)")
    print(f"Passed std ratio gate [{args.min_std_ratio}, {args.max_std_ratio}]: {stats['passed_std']} "
          f"({100*stats['passed_std']/max(1,stats['total_attempted']):.1f}%)")
    print(f"Final branches collected: {stats['passed_all']}")

    if branches:
        trust_scores = np.array([b['trust_score'] for b in branches])
        std_ratios = np.array([b['std_ratio'] for b in branches])
        print("\nCollected branch statistics:")
        print(f"  Trust: {trust_scores.mean():.6f} +/- {trust_scores.std():.6f}")
        print(f"  Std ratio: {std_ratios.mean():.4f} +/- {std_ratios.std():.4f}")

        print("\nBy brick:")
        for brick_id in sorted(stats['by_brick'].keys()):
            brick_stats = stats['by_brick'][brick_id]
            pct = 100 * brick_stats['passed'] / max(1, brick_stats['attempted'])
            print(f"  Brick {brick_id}: {brick_stats['passed']}/{brick_stats['attempted']} ({pct:.1f}%)")

    # Save
    if len(branches) == 0:
        print("\nWARNING: No branches passed gating! Check thresholds.")
        return

    print(f"\nSaving {len(branches)} branches to {args.output}...")

    # Prepare data for npz
    save_data = {
        'n_branches': len(branches),
        'horizon': args.horizon,
        'latent_dim': latent_dim,
        'action_dim': action_dim,
        'objective_dim': args.objective_dim,
        'real_z_mean': real_z_mean,
        'real_z_std': real_z_std,
        'min_trust_threshold': args.min_trust,
        'min_std_ratio': args.min_std_ratio,
        'max_std_ratio': args.max_std_ratio,
    }

    # Save each branch
    for i, branch in enumerate(branches):
        save_data[f'branch_{i}_z_sequence'] = branch['z_sequence']
        save_data[f'branch_{i}_actions'] = branch['actions']
        save_data[f'branch_{i}_source_episode'] = branch['source_episode']
        save_data[f'branch_{i}_source_timestep'] = branch['source_timestep']
        save_data[f'branch_{i}_trust_score'] = branch['trust_score']
        save_data[f'branch_{i}_std_ratio'] = branch['std_ratio']
        save_data[f'branch_{i}_brick_id'] = branch['brick_id']
        save_data[f'branch_{i}_objective_vector'] = branch['objective_vector']
        save_data[f'branch_{i}_branch_value'] = branch.get('branch_value', branch['trust_score'])

    np.savez(args.output, **save_data)
    print(f"Saved to {args.output}")

    # Save metadata
    gap_labels_path = None
    if any(b.get('gap_labels') for b in branches):
        gap_labels_path = args.output.replace('.npz', '_gap_labels.json')
        gap_data = [
            {'branch_idx': i, **b.get('gap_labels', {}), 'branch_value': b.get('branch_value', 0.0)}
            for i, b in enumerate(branches)
        ]
        with open(gap_labels_path, 'w') as f:
            json.dump(gap_data, f, indent=2)
        print(f"Saved gap labels to {gap_labels_path}")

    gen2sim_validity_rows, gen2sim_summary = assess_local_branch_corpus_gen2sim(
        branches,
        corpus_name=os.path.splitext(os.path.basename(args.output))[0],
        source_runtime_metadata=source_runtime_metadata,
        scene_tracks_backend=scene_tracks_backend,
        teacher_runtime_backend_selected=teacher_runtime_backend,
        vision_backbone_selected=vision_backbone_selected,
        semantic_grounding_mode=semantic_grounding_mode,
        semantic_memory_grounded=semantic_memory_grounded,
        gap_labels_path=gap_labels_path,
    )
    for row in gen2sim_validity_rows:
        branch_idx = int(row.get('branch_idx', -1))
        if 0 <= branch_idx < len(branches):
            branches[branch_idx]['gen2sim_validity'] = {
                key: value for key, value in row.items() if key != 'branch_idx'
            }

    gen2sim_validity_path = args.output.replace('.npz', '_gen2sim_validity.json')
    with open(gen2sim_validity_path, 'w', encoding='utf-8') as f:
        json.dump(gen2sim_validity_rows, f, indent=2)
    print(f"Saved gen2sim validity assessments to {gen2sim_validity_path}")

    metadata = build_synthetic_branch_corpus_metadata(
        output_path=args.output,
        world_model_path=args.world_model,
        dataset_path=args.dataset,
        horizon=args.horizon,
        branches_per_episode=args.branches_per_episode,
        objective_dim=args.objective_dim,
        min_trust=args.min_trust,
        min_std_ratio=args.min_std_ratio,
        max_std_ratio=args.max_std_ratio,
        stats=stats,
        branches=branches,
        coverage_graph_used=args.use_coverage_graph,
        coverage_graph_path=args.coverage_graph_path,
        source_runtime_metadata=source_runtime_metadata,
        source_runtime_metadata_artifact=args.source_runtime_metadata or None,
        scene_tracks_backend=scene_tracks_backend,
        teacher_runtime_backend_selected=teacher_runtime_backend,
        vision_backbone_selected=vision_backbone_selected,
        semantic_grounding_mode=semantic_grounding_mode,
        semantic_memory_grounded=semantic_memory_grounded,
        gap_labels_path=gap_labels_path,
        gen2sim_validity_path=gen2sim_validity_path,
        gen2sim_summary=gen2sim_summary,
    )

    metadata_path = args.output.replace('.npz', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_path}")

    print("\n" + "="*70)
    print("DONE")
    print("="*70)
    print(f"Collected {len(branches)} trusted local synthetic branches")
    print("Ready for trust + econ-weighted offline RL A/B test")


if __name__ == '__main__':
    main()
