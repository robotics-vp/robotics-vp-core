#!/usr/bin/env python3
"""
NAG edit surface analysis CLI.

Analyzes MPL / success / cost as a function of NAG edit vectors.
Makes NAG a first-class axis in the econ analytics stack.

Usage:
    python scripts/analyze_nag_mpl_surface.py --datapacks-dir <dir> [--output <json>]
    python scripts/analyze_nag_mpl_surface.py --task-id <task_id> --store <path>
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_datapacks_from_dir(datapacks_dir: Path) -> List[Dict[str, Any]]:
    """Load NAG datapacks from a directory of JSON files."""
    datapacks = []

    for json_file in datapacks_dir.glob("*.json"):
        try:
            with open(json_file) as f:
                data = json.load(f)
                if isinstance(data, list):
                    datapacks.extend(data)
                elif isinstance(data, dict):
                    datapacks.append(data)
        except Exception as e:
            logger.warning(f"Failed to load {json_file}: {e}")

    return datapacks


def load_datapacks_from_store(store_path: str, task_id: str) -> List[Dict[str, Any]]:
    """Load NAG datapacks from ontology store."""
    try:
        from src.ontology.store import OntologyStore

        store = OntologyStore(store_path)
        datapacks = store.list_datapacks(task_id=task_id)

        # Filter to NAG datapacks and convert to dicts
        nag_datapacks = []
        for dp in datapacks:
            if hasattr(dp, 'metadata') and isinstance(dp.metadata, dict):
                if 'nag_edit_vector' in dp.metadata or dp.source_type == 'nag_counterfactual':
                    nag_datapacks.append({
                        'datapack_id': dp.datapack_id,
                        'nag_edit_vector': dp.metadata.get('nag_edit_vector', []),
                        'difficulty_features': dp.metadata.get('difficulty_features', {}),
                        'lsd_metadata': dp.metadata.get('lsd_metadata', {}),
                    })

        return nag_datapacks
    except Exception as e:
        logger.error(f"Failed to load from store: {e}")
        return []


def analyze_edit_types(datapacks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze edit type distribution and statistics."""
    edit_type_counts: Dict[str, int] = {}
    edit_magnitudes: Dict[str, List[float]] = {}

    for dp in datapacks:
        edits = dp.get('nag_edit_vector', [])
        for edit in edits:
            edit_type = edit.get('edit_type', 'unknown')
            edit_type_counts[edit_type] = edit_type_counts.get(edit_type, 0) + 1

            # Track magnitude for pose edits
            params = edit.get('parameters', {})
            if 'delta_translation' in params:
                trans = params['delta_translation']
                if isinstance(trans, list):
                    mag = sum(x**2 for x in trans) ** 0.5
                    if edit_type not in edit_magnitudes:
                        edit_magnitudes[edit_type] = []
                    edit_magnitudes[edit_type].append(mag)

    result: Dict[str, Any] = {
        'counts': edit_type_counts,
        'total_edits': sum(edit_type_counts.values()),
        'unique_types': len(edit_type_counts),
    }

    # Add magnitude stats
    for edit_type, mags in edit_magnitudes.items():
        if mags:
            result[f'{edit_type}_magnitude'] = {
                'mean': np.mean(mags),
                'std': np.std(mags),
                'min': min(mags),
                'max': max(mags),
            }

    return result


def analyze_difficulty_impact(datapacks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze how NAG edits affect difficulty features."""
    difficulty_by_edit_count: Dict[int, List[Dict[str, float]]] = {}

    for dp in datapacks:
        edits = dp.get('nag_edit_vector', [])
        difficulty = dp.get('difficulty_features', {})

        num_edits = len(edits)
        if num_edits not in difficulty_by_edit_count:
            difficulty_by_edit_count[num_edits] = []

        # Extract NAG-specific features
        nag_features = {
            k: float(v) for k, v in difficulty.items()
            if k.startswith('nag_') and isinstance(v, (int, float))
        }
        if nag_features:
            difficulty_by_edit_count[num_edits].append(nag_features)

    # Aggregate by edit count
    result: Dict[str, Any] = {}
    for num_edits, features_list in sorted(difficulty_by_edit_count.items()):
        if not features_list:
            continue

        # Average each feature
        all_keys = set()
        for f in features_list:
            all_keys.update(f.keys())

        avg_features = {}
        for key in all_keys:
            vals = [f.get(key, 0) for f in features_list if key in f]
            if vals:
                avg_features[key] = np.mean(vals)

        result[f'{num_edits}_edits'] = {
            'count': len(features_list),
            'avg_features': avg_features,
        }

    return result


def compute_counterfactual_coverage(datapacks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute coverage of counterfactual generation."""
    base_episodes = set()
    counterfactuals_per_base: Dict[str, int] = {}

    for dp in datapacks:
        base_id = dp.get('lsd_metadata', {}).get('scene_id') or dp.get('base_episode_id', '')
        if base_id:
            base_episodes.add(base_id)
            counterfactuals_per_base[base_id] = counterfactuals_per_base.get(base_id, 0) + 1

    counts = list(counterfactuals_per_base.values())

    return {
        'num_base_episodes': len(base_episodes),
        'total_counterfactuals': len(datapacks),
        'avg_counterfactuals_per_base': np.mean(counts) if counts else 0,
        'min_counterfactuals': min(counts) if counts else 0,
        'max_counterfactuals': max(counts) if counts else 0,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Analyze NAG edit surface for MPL impact',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '--datapacks-dir',
        type=Path,
        help='Directory containing NAG datapack JSON files',
    )
    parser.add_argument(
        '--store',
        type=str,
        help='Path to ontology store',
    )
    parser.add_argument(
        '--task-id',
        type=str,
        help='Task ID to analyze (required with --store)',
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='Output JSON file for results',
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output',
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load datapacks
    if args.datapacks_dir:
        if not args.datapacks_dir.exists():
            logger.error(f"Directory not found: {args.datapacks_dir}")
            sys.exit(1)
        datapacks = load_datapacks_from_dir(args.datapacks_dir)
        logger.info(f"Loaded {len(datapacks)} datapacks from {args.datapacks_dir}")
    elif args.store and args.task_id:
        datapacks = load_datapacks_from_store(args.store, args.task_id)
        logger.info(f"Loaded {len(datapacks)} NAG datapacks for task {args.task_id}")
    else:
        logger.error("Must specify --datapacks-dir or (--store and --task-id)")
        sys.exit(1)

    if not datapacks:
        logger.warning("No datapacks found")
        results = {'error': 'no_datapacks'}
    else:
        # Run analyses
        from src.analytics.econ_reports import (
            compute_nag_edit_surface_summary,
            compute_nag_counterfactual_mpl_analysis,
        )

        # Core NAG surface analysis
        surface_summary = compute_nag_edit_surface_summary(datapacks)

        # Additional analyses
        edit_type_analysis = analyze_edit_types(datapacks)
        difficulty_analysis = analyze_difficulty_impact(datapacks)
        coverage_analysis = compute_counterfactual_coverage(datapacks)

        results = {
            'num_datapacks': len(datapacks),
            'surface_summary': surface_summary,
            'edit_type_analysis': edit_type_analysis,
            'difficulty_impact': difficulty_analysis,
            'counterfactual_coverage': coverage_analysis,
        }

    # Output results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results written to {args.output}")
    else:
        print(json.dumps(results, indent=2, default=str))

    # Print summary
    if 'error' not in results:
        logger.info("\n=== NAG Edit Surface Summary ===")
        logger.info(f"Total datapacks: {results['num_datapacks']}")

        coverage = results['counterfactual_coverage']
        logger.info(f"Base episodes: {coverage['num_base_episodes']}")
        logger.info(f"Avg counterfactuals/base: {coverage['avg_counterfactuals_per_base']:.2f}")

        edit_analysis = results['edit_type_analysis']
        logger.info(f"\nEdit types: {edit_analysis['counts']}")
        logger.info(f"Total edits: {edit_analysis['total_edits']}")

        if 'surface_summary' in results:
            impact = results['surface_summary'].get('counterfactual_impact', {})
            logger.info(f"\nAvg edits/counterfactual: {impact.get('avg_edits_per_counterfactual', 0):.2f}")
            logger.info(f"Removal rate: {impact.get('removal_rate', 0):.1%}")
            logger.info(f"Duplication rate: {impact.get('duplication_rate', 0):.1%}")


if __name__ == '__main__':
    main()
