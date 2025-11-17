"""
Skill Tree ↔ DataPack Adapter.

Bridges HRL skill graph with datapack taxonomy for:
- Skill-specific data queries
- Skill performance analysis
- Training data curation per skill

Works with unified 2.0-energy schema across:
- Phase B dishwashing datapacks
- Phase C drawer_vase datapacks (including HRL skills)
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple

from .skills import SkillID, SkillParams
from src.valuation.datapack_repo import DataPackRepo
from src.valuation.datapack_schema import DataPackMeta, AttributionProfile


class SkillDataPackAdapter:
    """
    Adapter between HRL skill graph and DataPack repository.

    Provides skill-centric views of datapack repository:
    - Query datapacks by skill usage
    - Analyze skill-specific performance metrics
    - Extract training data per skill
    - Compute skill value contributions
    """

    def __init__(self, repo: DataPackRepo):
        """
        Args:
            repo: DataPackRepo instance
        """
        self.repo = repo

    def get_skill_statistics(
        self,
        task_name: str,
        skill_id: int
    ) -> Dict[str, Any]:
        """
        Get performance statistics for a specific skill.

        Args:
            task_name: Task identifier
            skill_id: Skill ID to analyze

        Returns:
            Dict with skill statistics
        """
        # Query all datapacks with this skill
        positive = self.repo.query(
            task_name=task_name,
            bucket="positive",
            skill_id=skill_id,
            limit=10000
        )

        negative = self.repo.query(
            task_name=task_name,
            bucket="negative",
            skill_id=skill_id,
            limit=10000
        )

        all_packs = positive + negative

        if not all_packs:
            return {
                'skill_id': skill_id,
                'skill_name': SkillID.name(skill_id),
                'total_usage': 0,
                'positive_count': 0,
                'negative_count': 0,
            }

        # Compute statistics
        delta_js = [dp.attribution.delta_J for dp in all_packs]
        trust_scores = [dp.attribution.trust_score for dp in all_packs]
        w_econs = [dp.attribution.w_econ for dp in all_packs]

        # Per-skill contribution analysis
        skill_delta_mpls = []
        skill_delta_errors = []
        for dp in all_packs:
            if skill_id in dp.attribution.skill_contribs:
                contrib = dp.attribution.skill_contribs[skill_id]
                if 'delta_mpl' in contrib:
                    skill_delta_mpls.append(contrib['delta_mpl'])
                if 'delta_error' in contrib:
                    skill_delta_errors.append(contrib['delta_error'])

        return {
            'skill_id': skill_id,
            'skill_name': SkillID.name(skill_id),
            'total_usage': len(all_packs),
            'positive_count': len(positive),
            'negative_count': len(negative),
            'success_rate': len(positive) / len(all_packs),
            'delta_j_mean': np.mean(delta_js),
            'delta_j_std': np.std(delta_js),
            'delta_j_min': np.min(delta_js),
            'delta_j_max': np.max(delta_js),
            'trust_mean': np.mean(trust_scores),
            'w_econ_mean': np.mean(w_econs),
            'skill_delta_mpl_mean': np.mean(skill_delta_mpls) if skill_delta_mpls else None,
            'skill_delta_error_mean': np.mean(skill_delta_errors) if skill_delta_errors else None,
        }

    def get_all_skills_performance(self, task_name: str) -> Dict[int, Dict[str, Any]]:
        """
        Get performance statistics for all skills.

        Args:
            task_name: Task identifier

        Returns:
            Dict mapping skill_id → statistics
        """
        results = {}
        for skill_id in range(SkillID.NUM_SKILLS):
            results[skill_id] = self.get_skill_statistics(task_name, skill_id)
        return results

    def get_skill_training_data(
        self,
        task_name: str,
        skill_id: int,
        bucket: str = "positive",
        min_trust: float = 0.9,
        top_k: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Extract training data for a specific skill.

        Args:
            task_name: Task identifier
            skill_id: Skill ID to extract data for
            bucket: "positive" or "negative"
            min_trust: Minimum trust score
            top_k: Maximum number of samples

        Returns:
            List of training samples with (state, skill_params, metrics)
        """
        datapacks = self.repo.query(
            task_name=task_name,
            bucket=bucket,
            skill_id=skill_id,
            min_trust=min_trust,
            limit=top_k,
            sort_by="delta_j",
            sort_descending=(bucket == "positive")
        )

        training_data = []
        for dp in datapacks:
            # Extract skill-specific data from trace
            for entry in dp.skill_trace:
                if entry['skill_id'] == skill_id:
                    sample = {
                        'pack_id': dp.pack_id,
                        'skill_id': skill_id,
                        'params': entry.get('params', {}),
                        'duration': entry.get('duration', 0),
                        'local_metrics': entry.get('local_metrics', {}),
                        'delta_j': dp.attribution.delta_J,
                        'trust_score': dp.attribution.trust_score,
                        'w_econ': dp.attribution.w_econ,
                        'condition': dp.condition.to_dict(),
                        'semantic_tags': dp.semantic_tags,
                    }
                    training_data.append(sample)

        return training_data

    def get_skill_failure_modes(
        self,
        task_name: str,
        skill_id: int,
        top_k: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Analyze failure modes for a specific skill.

        Args:
            task_name: Task identifier
            skill_id: Skill ID to analyze
            top_k: Number of worst failures to examine

        Returns:
            List of failure analysis dicts
        """
        # Get worst negative datapacks for this skill
        negative_packs = self.repo.get_negative_for_skill(
            task_name=task_name,
            skill_id=skill_id,
            top_k=top_k,
            require_counterfactual=True
        )

        failure_modes = []
        for dp in negative_packs:
            failure = {
                'pack_id': dp.pack_id,
                'delta_j': dp.attribution.delta_J,
                'semantic_tags': dp.semantic_tags,
                'condition': {
                    'vase_offset': dp.condition.vase_offset,
                    'drawer_friction': dp.condition.drawer_friction,
                    'lighting_profile': dp.condition.lighting_profile,
                    'occlusion_level': dp.condition.occlusion_level,
                },
                'counterfactual': dp.counterfactual_plan,
                'counterfactual_source': dp.counterfactual_source,
            }

            # Extract skill-specific failure info
            for entry in dp.skill_trace:
                if entry['skill_id'] == skill_id:
                    failure['skill_entry'] = entry

            failure_modes.append(failure)

        return failure_modes

    def get_skill_sequence_patterns(
        self,
        task_name: str,
        bucket: str = "positive",
        top_k: int = 100
    ) -> Dict[Tuple[int, ...], int]:
        """
        Analyze common skill sequence patterns.

        Args:
            task_name: Task identifier
            bucket: "positive" or "negative"
            top_k: Number of datapacks to analyze

        Returns:
            Dict mapping skill sequence → frequency count
        """
        datapacks = self.repo.query(
            task_name=task_name,
            bucket=bucket,
            limit=top_k,
            sort_by="delta_j",
            sort_descending=True
        )

        sequence_counts = {}
        for dp in datapacks:
            skill_seq = tuple(dp.get_skill_ids())
            sequence_counts[skill_seq] = sequence_counts.get(skill_seq, 0) + 1

        return sequence_counts

    def compute_skill_value_matrix(
        self,
        task_name: str
    ) -> np.ndarray:
        """
        Compute value matrix: skill_id × condition → average ΔJ.

        Args:
            task_name: Task identifier

        Returns:
            Value matrix (num_skills × num_conditions)
        """
        # Define condition dimensions
        conditions = [
            'normal',
            'offset_vase',
            'high_friction',
            'low_light',
            'partial_occlusion'
        ]

        value_matrix = np.zeros((SkillID.NUM_SKILLS, len(conditions)))
        count_matrix = np.zeros((SkillID.NUM_SKILLS, len(conditions)))

        # Load all datapacks
        all_packs = self.repo.load_all(task_name)

        for dp in all_packs:
            # Determine condition category
            cond_idx = 0  # Default: normal
            if 'offset_vase' in dp.semantic_tags:
                cond_idx = 1
            elif 'high_friction' in dp.semantic_tags:
                cond_idx = 2
            elif 'low_light' in dp.semantic_tags:
                cond_idx = 3
            elif 'partial_occlusion' in dp.semantic_tags:
                cond_idx = 4

            # Accumulate per skill
            for skill_id in dp.get_skill_ids():
                value_matrix[skill_id, cond_idx] += dp.attribution.delta_J
                count_matrix[skill_id, cond_idx] += 1

        # Average
        mask = count_matrix > 0
        value_matrix[mask] = value_matrix[mask] / count_matrix[mask]

        return value_matrix

    def recommend_skill_for_condition(
        self,
        task_name: str,
        condition_tags: List[str]
    ) -> List[Tuple[int, float]]:
        """
        Recommend skills ranked by historical performance for given conditions.

        Args:
            task_name: Task identifier
            condition_tags: List of condition tags

        Returns:
            List of (skill_id, expected_delta_j) sorted by performance
        """
        # Query datapacks matching conditions
        condition_filters = {}
        for tag in condition_tags:
            condition_filters[tag] = True

        positive_packs = self.repo.query(
            task_name=task_name,
            bucket="positive",
            condition_filters=condition_filters,
            min_trust=0.8,
            limit=1000
        )

        # Compute average ΔJ per skill
        skill_performance = {}
        for dp in positive_packs:
            for skill_id in dp.get_skill_ids():
                if skill_id not in skill_performance:
                    skill_performance[skill_id] = []
                skill_performance[skill_id].append(dp.attribution.delta_J)

        # Average and rank
        recommendations = []
        for skill_id, delta_js in skill_performance.items():
            avg_delta_j = np.mean(delta_js)
            recommendations.append((skill_id, avg_delta_j))

        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations

    def extract_contrastive_pairs(
        self,
        task_name: str,
        skill_id: int,
        n_pairs: int = 50
    ) -> List[Tuple[DataPackMeta, DataPackMeta]]:
        """
        Extract contrastive pairs (positive, negative) for skill training.

        Args:
            task_name: Task identifier
            skill_id: Skill ID
            n_pairs: Number of pairs to extract

        Returns:
            List of (positive_pack, negative_pack) tuples
        """
        positive = self.repo.get_positive_for_skill(
            task_name=task_name,
            skill_id=skill_id,
            top_k=n_pairs
        )

        negative = self.repo.get_negative_for_skill(
            task_name=task_name,
            skill_id=skill_id,
            top_k=n_pairs,
            require_counterfactual=True
        )

        # Pair by similar conditions
        pairs = []
        for pos in positive:
            # Find most similar negative
            best_match = None
            best_sim = -1

            for neg in negative:
                # Compute condition similarity
                sim = self._compute_condition_similarity(pos.condition, neg.condition)
                if sim > best_sim:
                    best_sim = sim
                    best_match = neg

            if best_match is not None:
                pairs.append((pos, best_match))
                negative.remove(best_match)  # Don't reuse

            if len(pairs) >= n_pairs:
                break

        return pairs

    def _compute_condition_similarity(self, cond1, cond2) -> float:
        """Compute similarity between two condition profiles."""
        # Compare key attributes
        score = 0.0
        total = 0.0

        # Vase offset similarity (inverse distance)
        offset1 = np.array(cond1.vase_offset)
        offset2 = np.array(cond2.vase_offset)
        offset_dist = np.linalg.norm(offset1 - offset2)
        score += 1.0 / (1.0 + offset_dist)
        total += 1.0

        # Friction similarity
        friction_diff = abs(cond1.drawer_friction - cond2.drawer_friction)
        score += 1.0 / (1.0 + friction_diff * 10)
        total += 1.0

        # Lighting match
        if cond1.lighting_profile == cond2.lighting_profile:
            score += 1.0
        total += 1.0

        # Occlusion similarity
        occ_diff = abs(cond1.occlusion_level - cond2.occlusion_level)
        score += 1.0 / (1.0 + occ_diff * 5)
        total += 1.0

        return score / total

    def generate_skill_curriculum(
        self,
        task_name: str,
        skill_id: int,
        n_levels: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Generate curriculum for skill training (easy → hard conditions).

        Args:
            task_name: Task identifier
            skill_id: Skill ID
            n_levels: Number of difficulty levels

        Returns:
            List of curriculum levels with datapacks
        """
        # Get all positive datapacks for skill
        positive = self.repo.get_positive_for_skill(
            task_name=task_name,
            skill_id=skill_id,
            top_k=1000
        )

        if not positive:
            return []

        # Score difficulty based on conditions
        scored_packs = []
        for dp in positive:
            # Difficulty factors
            offset_dist = np.linalg.norm(np.array(dp.condition.vase_offset))
            friction = dp.condition.drawer_friction
            occlusion = dp.condition.occlusion_level

            # Simple difficulty score (higher = harder)
            difficulty = offset_dist * 10 + friction * 5 + occlusion * 10
            scored_packs.append((difficulty, dp))

        scored_packs.sort(key=lambda x: x[0])

        # Split into levels
        curriculum = []
        packs_per_level = len(scored_packs) // n_levels

        for level in range(n_levels):
            start_idx = level * packs_per_level
            end_idx = start_idx + packs_per_level if level < n_levels - 1 else len(scored_packs)

            level_packs = [dp for _, dp in scored_packs[start_idx:end_idx]]

            level_difficulty = np.mean([d for d, _ in scored_packs[start_idx:end_idx]])

            curriculum.append({
                'level': level,
                'difficulty': level_difficulty,
                'n_samples': len(level_packs),
                'datapacks': level_packs,
                'conditions_summary': {
                    'avg_offset': np.mean([np.linalg.norm(np.array(dp.condition.vase_offset))
                                           for dp in level_packs]),
                    'avg_friction': np.mean([dp.condition.drawer_friction for dp in level_packs]),
                    'avg_occlusion': np.mean([dp.condition.occlusion_level for dp in level_packs]),
                }
            })

        return curriculum

    def query_by_energy_driver_tags(
        self,
        task_name: str,
        energy_tags: List[str],
        bucket: Optional[str] = None,
        match_all: bool = False,
        limit: int = 100
    ) -> List[DataPackMeta]:
        """
        Query datapacks by energy driver tags.

        Args:
            task_name: Task identifier
            energy_tags: List of energy driver tag strings
            bucket: Optional bucket filter ("positive" or "negative")
            match_all: If True, datapack must have all tags; otherwise any
            limit: Maximum results

        Returns:
            List of matching DataPackMeta objects
        """
        all_packs = self.repo.load_all(task_name)

        results = []
        for dp in all_packs:
            # Bucket filter
            if bucket is not None and dp.bucket != bucket:
                continue

            # Energy tag filter
            if match_all:
                # Must have all specified tags
                if all(tag in dp.energy_driver_tags for tag in energy_tags):
                    results.append(dp)
            else:
                # Must have at least one tag
                if any(tag in dp.energy_driver_tags for tag in energy_tags):
                    results.append(dp)

            if len(results) >= limit:
                break

        return results

    def query_by_source_type(
        self,
        task_name: str,
        source_type: str,
        bucket: Optional[str] = None,
        min_trust: float = 0.0,
        limit: int = 100
    ) -> List[DataPackMeta]:
        """
        Query datapacks by source type (real, synthetic, hybrid).

        Args:
            task_name: Task identifier
            source_type: "real", "synthetic", or "hybrid"
            bucket: Optional bucket filter
            min_trust: Minimum trust score
            limit: Maximum results

        Returns:
            List of matching DataPackMeta objects
        """
        return self.repo.query(
            task_name=task_name,
            bucket=bucket,
            source_type=source_type,
            min_trust=min_trust,
            limit=limit
        )

    def query_by_attribution_ranges(
        self,
        task_name: str,
        min_delta_mpl: Optional[float] = None,
        max_delta_mpl: Optional[float] = None,
        min_delta_error: Optional[float] = None,
        max_delta_error: Optional[float] = None,
        min_delta_ep: Optional[float] = None,
        max_delta_ep: Optional[float] = None,
        min_trust: float = 0.0,
        max_trust: float = 1.0,
        bucket: Optional[str] = None,
        limit: int = 100
    ) -> List[DataPackMeta]:
        """
        Query datapacks by attribution metric ranges.

        Args:
            task_name: Task identifier
            min_delta_mpl: Minimum ΔMPL
            max_delta_mpl: Maximum ΔMPL
            min_delta_error: Minimum Δerror
            max_delta_error: Maximum Δerror
            min_delta_ep: Minimum ΔEP
            max_delta_ep: Maximum ΔEP
            min_trust: Minimum trust score
            max_trust: Maximum trust score
            bucket: Optional bucket filter
            limit: Maximum results

        Returns:
            List of matching DataPackMeta objects
        """
        all_packs = self.repo.load_all(task_name)

        results = []
        for dp in all_packs:
            # Bucket filter
            if bucket is not None and dp.bucket != bucket:
                continue

            # Attribution filters
            if min_delta_mpl is not None and dp.attribution.delta_mpl < min_delta_mpl:
                continue
            if max_delta_mpl is not None and dp.attribution.delta_mpl > max_delta_mpl:
                continue
            if min_delta_error is not None and dp.attribution.delta_error < min_delta_error:
                continue
            if max_delta_error is not None and dp.attribution.delta_error > max_delta_error:
                continue
            if min_delta_ep is not None and dp.attribution.delta_ep < min_delta_ep:
                continue
            if max_delta_ep is not None and dp.attribution.delta_ep > max_delta_ep:
                continue
            if dp.attribution.trust_score < min_trust or dp.attribution.trust_score > max_trust:
                continue

            results.append(dp)
            if len(results) >= limit:
                break

        return results

    def get_env_type_statistics(self, task_name: str) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics broken down by environment type.

        Works across Phase B (dishwashing) and Phase C (drawer_vase).

        Args:
            task_name: Task identifier

        Returns:
            Dict mapping env_type → statistics
        """
        all_packs = self.repo.load_all(task_name)

        env_stats = {}
        for dp in all_packs:
            env_type = dp.env_type
            if env_type not in env_stats:
                env_stats[env_type] = {
                    'count': 0,
                    'positive': 0,
                    'negative': 0,
                    'delta_js': [],
                    'delta_mpls': [],
                    'delta_errors': [],
                    'delta_eps': [],
                    'energy_whs': [],
                    'trust_scores': [],
                }

            stats = env_stats[env_type]
            stats['count'] += 1
            if dp.bucket == "positive":
                stats['positive'] += 1
            else:
                stats['negative'] += 1

            stats['delta_js'].append(dp.attribution.delta_J)
            stats['delta_mpls'].append(dp.attribution.delta_mpl)
            stats['delta_errors'].append(dp.attribution.delta_error)
            stats['delta_eps'].append(dp.attribution.delta_ep)
            stats['energy_whs'].append(dp.energy.total_Wh)
            stats['trust_scores'].append(dp.attribution.trust_score)

        # Compute summaries
        for env_type, stats in env_stats.items():
            if stats['count'] > 0:
                stats['mean_delta_j'] = float(np.mean(stats['delta_js']))
                stats['mean_delta_mpl'] = float(np.mean(stats['delta_mpls']))
                stats['mean_delta_error'] = float(np.mean(stats['delta_errors']))
                stats['mean_delta_ep'] = float(np.mean(stats['delta_eps']))
                stats['mean_energy_wh'] = float(np.mean(stats['energy_whs']))
                stats['mean_trust'] = float(np.mean(stats['trust_scores']))
                stats['positive_ratio'] = stats['positive'] / stats['count']

            # Clean up raw lists
            del stats['delta_js']
            del stats['delta_mpls']
            del stats['delta_errors']
            del stats['delta_eps']
            del stats['energy_whs']
            del stats['trust_scores']

        return env_stats

    def get_datapack_ids_for_skill(
        self,
        task_name: str,
        skill_id: int,
        bucket: Optional[str] = None
    ) -> List[str]:
        """
        Get pack_ids for datapacks containing a specific skill.

        Args:
            task_name: Task identifier
            skill_id: Skill ID to filter
            bucket: Optional bucket filter

        Returns:
            List of pack_id strings
        """
        datapacks = self.repo.query(
            task_name=task_name,
            bucket=bucket,
            skill_id=skill_id,
            limit=10000
        )

        return [dp.pack_id for dp in datapacks]

    def summarize_energy_usage(self, task_name: str) -> Dict[str, Any]:
        """
        Summarize energy usage across all datapacks.

        Args:
            task_name: Task identifier

        Returns:
            Energy usage summary dict
        """
        all_packs = self.repo.load_all(task_name)

        if not all_packs:
            return {'n_packs': 0}

        total_whs = [dp.energy.total_Wh for dp in all_packs]
        wh_per_units = [dp.energy.Wh_per_unit for dp in all_packs]

        # Aggregate limb energy
        limb_totals = {}
        for dp in all_packs:
            for limb, data in dp.energy.energy_per_limb.items():
                if limb not in limb_totals:
                    limb_totals[limb] = 0.0
                limb_totals[limb] += data.get('Wh', 0.0)

        # Aggregate skill energy
        skill_totals = {}
        for dp in all_packs:
            for skill, data in dp.energy.energy_per_skill.items():
                if skill not in skill_totals:
                    skill_totals[skill] = 0.0
                skill_totals[skill] += data.get('Wh', 0.0)

        return {
            'n_packs': len(all_packs),
            'total_wh_mean': float(np.mean(total_whs)),
            'total_wh_std': float(np.std(total_whs)),
            'wh_per_unit_mean': float(np.mean(wh_per_units)),
            'wh_per_unit_std': float(np.std(wh_per_units)),
            'limb_totals': limb_totals,
            'skill_totals': skill_totals,
        }

    def export_skill_report(self, task_name: str, output_path: str):
        """
        Export comprehensive skill performance report.

        Args:
            task_name: Task identifier
            output_path: Output file path (JSON)
        """
        import json

        report = {
            'task_name': task_name,
            'skills': {},
            'sequence_patterns': {},
            'env_types': {},
            'energy_summary': {},
        }

        # Per-skill statistics
        for skill_id in range(SkillID.NUM_SKILLS):
            stats = self.get_skill_statistics(task_name, skill_id)
            report['skills'][SkillID.name(skill_id)] = stats

        # Sequence patterns
        pos_patterns = self.get_skill_sequence_patterns(task_name, "positive", top_k=100)
        neg_patterns = self.get_skill_sequence_patterns(task_name, "negative", top_k=100)

        # Convert tuples to strings for JSON
        report['sequence_patterns']['positive'] = {
            str([SkillID.name(s) for s in seq]): count
            for seq, count in pos_patterns.items()
        }
        report['sequence_patterns']['negative'] = {
            str([SkillID.name(s) for s in seq]): count
            for seq, count in neg_patterns.items()
        }

        # Environment type statistics
        report['env_types'] = self.get_env_type_statistics(task_name)

        # Energy summary
        report['energy_summary'] = self.summarize_energy_usage(task_name)

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
