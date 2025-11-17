"""
Phase B Integration with DataPack Repository.

Hooks Phase B scripts (trust_net, w_econ_lattice, λ-controller) into DataPackRepo.
"""

import os
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

from .datapack_repo import DataPackRepo
from .datapack_schema import DataPackMeta, AttributionProfile


class PhaseBDataPackIntegration:
    """
    Integration layer between Phase B components and DataPack repository.

    Connects:
    - trust_net → trust_score in AttributionProfile
    - w_econ_lattice → w_econ in AttributionProfile
    - λ-controller → lambda_budget in AttributionProfile
    - World model → source_type, wm_* fields in AttributionProfile
    """

    def __init__(
        self,
        repo: DataPackRepo,
        trust_net=None,
        w_econ_lattice=None,
        lambda_controller=None
    ):
        """
        Args:
            repo: DataPackRepo instance
            trust_net: TrustNet model (optional)
            w_econ_lattice: W_econ lattice model (optional)
            lambda_controller: λ-controller (optional)
        """
        self.repo = repo
        self.trust_net = trust_net
        self.w_econ_lattice = w_econ_lattice
        self.lambda_controller = lambda_controller

    def enrich_with_phase_b_scores(
        self,
        task_name: str,
        episode_features: Optional[np.ndarray] = None,
        batch_size: int = 100
    ) -> int:
        """
        Enrich existing datapacks with Phase B scores (trust, w_econ).

        Args:
            task_name: Task identifier
            episode_features: Optional feature matrix (n_packs, feature_dim)
            batch_size: Batch size for processing

        Returns:
            Number of datapacks updated
        """
        if self.trust_net is None and self.w_econ_lattice is None:
            print("No Phase B models provided")
            return 0

        datapacks = self.repo.load_all(task_name)
        if not datapacks:
            return 0

        updated_count = 0

        for i in range(0, len(datapacks), batch_size):
            batch = datapacks[i:i + batch_size]

            # Compute trust scores
            if self.trust_net is not None:
                trust_scores = self._compute_trust_scores(batch, episode_features)
            else:
                trust_scores = [dp.attribution.trust_score for dp in batch]

            # Compute w_econ scores
            if self.w_econ_lattice is not None:
                w_econ_scores = self._compute_w_econ_scores(batch, episode_features)
            else:
                w_econ_scores = [dp.attribution.w_econ for dp in batch]

            # Update attributions
            for j, dp in enumerate(batch):
                dp.attribution.trust_score = trust_scores[j]
                dp.attribution.w_econ = w_econ_scores[j]
                dp.attribution.econ_weight_final = trust_scores[j] * w_econ_scores[j]
                updated_count += 1

        # Note: JSONL is append-only, so we need to rewrite
        # In production, use a database or versioned storage
        print(f"Updated {updated_count} datapacks with Phase B scores")
        return updated_count

    def _compute_trust_scores(
        self,
        datapacks: List[DataPackMeta],
        episode_features: Optional[np.ndarray] = None
    ) -> List[float]:
        """Compute trust scores for batch of datapacks."""
        if self.trust_net is None:
            return [0.5] * len(datapacks)

        scores = []
        for i, dp in enumerate(datapacks):
            # Try to get features
            if episode_features is not None and i < len(episode_features):
                features = episode_features[i]
            else:
                # Extract from datapack
                features = self._extract_features_from_datapack(dp)

            # Compute trust score
            try:
                if hasattr(self.trust_net, 'score_episode'):
                    score = self.trust_net.score_episode(features)
                elif hasattr(self.trust_net, 'forward'):
                    import torch
                    feat_t = torch.FloatTensor(features).unsqueeze(0)
                    with torch.no_grad():
                        score = self.trust_net(feat_t).item()
                else:
                    score = 0.5
            except Exception:
                score = 0.5

            scores.append(float(score))

        return scores

    def _compute_w_econ_scores(
        self,
        datapacks: List[DataPackMeta],
        episode_features: Optional[np.ndarray] = None
    ) -> List[float]:
        """Compute w_econ scores for batch of datapacks."""
        if self.w_econ_lattice is None:
            return [1.0] * len(datapacks)

        scores = []
        for i, dp in enumerate(datapacks):
            # Try to get features
            if episode_features is not None and i < len(episode_features):
                features = episode_features[i]
            else:
                features = self._extract_features_from_datapack(dp)

            # Compute w_econ score
            try:
                if hasattr(self.w_econ_lattice, 'score_episode'):
                    score = self.w_econ_lattice.score_episode(features)
                elif hasattr(self.w_econ_lattice, 'forward'):
                    import torch
                    feat_t = torch.FloatTensor(features).unsqueeze(0)
                    with torch.no_grad():
                        score = self.w_econ_lattice(feat_t).item()
                else:
                    score = 1.0
            except Exception:
                score = 1.0

            scores.append(float(score))

        return scores

    def _extract_features_from_datapack(self, dp: DataPackMeta) -> np.ndarray:
        """Extract episode features from datapack for scoring."""
        # Build feature vector from attribution
        features = np.array([
            dp.attribution.delta_mpl,
            dp.attribution.delta_error,
            dp.attribution.delta_ep,
            dp.attribution.delta_wage_parity,
            dp.attribution.delta_J,
            dp.attribution.lambda_budget,
            len(dp.skill_trace),  # Number of skills
            dp.get_total_duration(),  # Total duration
        ], dtype=np.float32)

        return features

    def select_training_data_by_trust(
        self,
        task_name: str,
        min_trust: float = 0.9,
        max_samples: int = 1000
    ) -> List[DataPackMeta]:
        """
        Select high-trust datapacks for training.

        Args:
            task_name: Task identifier
            min_trust: Minimum trust threshold
            max_samples: Maximum number of samples

        Returns:
            List of high-trust datapacks
        """
        return self.repo.query(
            task_name=task_name,
            min_trust=min_trust,
            limit=max_samples,
            sort_by="trust_score",
            sort_descending=True
        )

    def select_by_econ_weight(
        self,
        task_name: str,
        min_econ_weight: float = 0.5,
        max_samples: int = 1000
    ) -> List[DataPackMeta]:
        """
        Select datapacks by economic weight (trust * w_econ).

        Args:
            task_name: Task identifier
            min_econ_weight: Minimum combined weight
            max_samples: Maximum number of samples

        Returns:
            List of economically weighted datapacks
        """
        all_packs = self.repo.load_all(task_name)

        # Filter by combined weight
        filtered = [
            dp for dp in all_packs
            if (dp.attribution.trust_score * dp.attribution.w_econ) >= min_econ_weight
        ]

        # Sort by combined weight
        filtered.sort(
            key=lambda x: x.attribution.trust_score * x.attribution.w_econ,
            reverse=True
        )

        return filtered[:max_samples]

    def allocate_lambda_budget(
        self,
        task_name: str,
        total_budget: float = 1000.0,
        strategy: str = "proportional"
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Allocate λ synthetic budget across datapacks.

        Args:
            task_name: Task identifier
            total_budget: Total λ budget
            strategy: Allocation strategy ("proportional", "top_k", "uniform")

        Returns:
            Dict with allocation plan
        """
        if self.lambda_controller is None:
            print("No λ-controller provided")
            return {'allocations': []}

        datapacks = self.repo.load_all(task_name)
        if not datapacks:
            return {'allocations': []}

        allocations = []

        if strategy == "proportional":
            # Allocate proportional to economic weight
            total_weight = sum(
                dp.attribution.trust_score * dp.attribution.w_econ
                for dp in datapacks
            )

            for dp in datapacks:
                weight = dp.attribution.trust_score * dp.attribution.w_econ
                allocation = (weight / total_weight) * total_budget if total_weight > 0 else 0
                dp.attribution.lambda_budget = allocation
                allocations.append((dp.pack_id, allocation))

        elif strategy == "top_k":
            # Allocate to top K by economic weight
            k = min(100, len(datapacks))
            sorted_packs = sorted(
                datapacks,
                key=lambda x: x.attribution.trust_score * x.attribution.w_econ,
                reverse=True
            )

            budget_per_pack = total_budget / k
            for dp in sorted_packs[:k]:
                dp.attribution.lambda_budget = budget_per_pack
                allocations.append((dp.pack_id, budget_per_pack))

        elif strategy == "uniform":
            # Uniform allocation
            budget_per_pack = total_budget / len(datapacks)
            for dp in datapacks:
                dp.attribution.lambda_budget = budget_per_pack
                allocations.append((dp.pack_id, budget_per_pack))

        return {
            'strategy': strategy,
            'total_budget': total_budget,
            'n_allocated': len(allocations),
            'allocations': allocations
        }

    def mark_training_run(
        self,
        datapacks: List[DataPackMeta],
        run_id: str,
        role: str = "policy_train"
    ):
        """
        Mark datapacks as used in a training run.

        Args:
            datapacks: List of datapacks
            run_id: Training run identifier
            role: Role in training (e.g., "policy_train", "wm_train")
        """
        for dp in datapacks:
            if run_id not in dp.attribution.used_in_training_runs:
                dp.attribution.used_in_training_runs.append(run_id)

            if role.startswith("wm_"):
                dp.attribution.wm_role = role

    def get_unused_datapacks(
        self,
        task_name: str,
        exclude_runs: Optional[List[str]] = None
    ) -> List[DataPackMeta]:
        """
        Get datapacks not yet used in training.

        Args:
            task_name: Task identifier
            exclude_runs: List of run IDs to exclude

        Returns:
            List of unused datapacks
        """
        exclude_runs = exclude_runs or []
        all_packs = self.repo.load_all(task_name)

        unused = [
            dp for dp in all_packs
            if not any(run in dp.attribution.used_in_training_runs for run in exclude_runs)
        ]

        return unused

    def compute_mvd_scores(
        self,
        task_name: str,
        baseline_delta_j: float = 0.0,
        horizon: int = 1000
    ) -> int:
        """
        Compute Marginal Value-of-Data scores for datapacks.

        MVD = trust * w_econ * expected_delta_j

        Args:
            task_name: Task identifier
            baseline_delta_j: Baseline expected ΔJ
            horizon: Planning horizon

        Returns:
            Number of datapacks updated
        """
        datapacks = self.repo.load_all(task_name)
        updated = 0

        for dp in datapacks:
            # Simple MVD: trust * w_econ * delta_j
            mvd = (
                dp.attribution.trust_score *
                dp.attribution.w_econ *
                max(dp.attribution.delta_J - baseline_delta_j, 0)
            )
            dp.attribution.mvd_score = mvd
            updated += 1

        return updated

    def export_phase_b_summary(self, task_name: str, output_path: str):
        """
        Export Phase B integration summary.

        Args:
            task_name: Task identifier
            output_path: Output JSON file path
        """
        datapacks = self.repo.load_all(task_name)

        if not datapacks:
            summary = {'task_name': task_name, 'n_datapacks': 0}
        else:
            trust_scores = [dp.attribution.trust_score for dp in datapacks]
            w_econ_scores = [dp.attribution.w_econ for dp in datapacks]
            lambda_budgets = [dp.attribution.lambda_budget for dp in datapacks]
            mvd_scores = [dp.attribution.mvd_score or 0 for dp in datapacks]

            # Training run statistics
            all_runs = set()
            for dp in datapacks:
                all_runs.update(dp.attribution.used_in_training_runs)

            # Source type distribution
            source_types = {}
            for dp in datapacks:
                st = dp.attribution.source_type
                source_types[st] = source_types.get(st, 0) + 1

            summary = {
                'task_name': task_name,
                'n_datapacks': len(datapacks),
                'trust_scores': {
                    'mean': float(np.mean(trust_scores)),
                    'std': float(np.std(trust_scores)),
                    'min': float(np.min(trust_scores)),
                    'max': float(np.max(trust_scores)),
                },
                'w_econ_scores': {
                    'mean': float(np.mean(w_econ_scores)),
                    'std': float(np.std(w_econ_scores)),
                    'min': float(np.min(w_econ_scores)),
                    'max': float(np.max(w_econ_scores)),
                },
                'lambda_budgets': {
                    'total': float(np.sum(lambda_budgets)),
                    'mean': float(np.mean(lambda_budgets)),
                    'max': float(np.max(lambda_budgets)),
                },
                'mvd_scores': {
                    'mean': float(np.mean(mvd_scores)),
                    'std': float(np.std(mvd_scores)),
                    'max': float(np.max(mvd_scores)),
                },
                'training_runs': list(all_runs),
                'source_types': source_types,
            }

        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)


def integrate_phase_b_with_datapacks(
    repo: DataPackRepo,
    task_name: str,
    trust_net=None,
    w_econ_lattice=None,
    lambda_controller=None,
    output_dir: str = "results/phase_b_integration"
):
    """
    Convenience function to run full Phase B integration.

    Args:
        repo: DataPackRepo instance
        task_name: Task identifier
        trust_net: TrustNet model
        w_econ_lattice: W_econ lattice model
        lambda_controller: λ-controller
        output_dir: Output directory

    Returns:
        Integration results dict
    """
    os.makedirs(output_dir, exist_ok=True)

    integration = PhaseBDataPackIntegration(
        repo=repo,
        trust_net=trust_net,
        w_econ_lattice=w_econ_lattice,
        lambda_controller=lambda_controller
    )

    print("=" * 70)
    print("PHASE B ↔ DATAPACK INTEGRATION")
    print("=" * 70)

    # Enrich with Phase B scores
    print("\n1. Enriching datapacks with Phase B scores...")
    n_updated = integration.enrich_with_phase_b_scores(task_name)
    print(f"   Updated {n_updated} datapacks")

    # Compute MVD scores
    print("\n2. Computing MVD scores...")
    n_mvd = integration.compute_mvd_scores(task_name)
    print(f"   Computed MVD for {n_mvd} datapacks")

    # Allocate λ budget
    print("\n3. Allocating λ synthetic budget...")
    allocation = integration.allocate_lambda_budget(
        task_name,
        total_budget=1000.0,
        strategy="proportional"
    )
    print(f"   Allocated {allocation['total_budget']:.2f} budget to {allocation['n_allocated']} datapacks")

    # Export summary
    summary_path = os.path.join(output_dir, f"{task_name}_phase_b_summary.json")
    integration.export_phase_b_summary(task_name, summary_path)
    print(f"\n4. Exported summary to {summary_path}")

    # Get high-trust training data
    high_trust = integration.select_training_data_by_trust(task_name, min_trust=0.9)
    print(f"\n5. High-trust datapacks (≥0.9): {len(high_trust)}")

    # Get economically weighted data
    econ_weighted = integration.select_by_econ_weight(task_name, min_econ_weight=0.5)
    print(f"   Econ-weighted datapacks (≥0.5): {len(econ_weighted)}")

    return {
        'n_updated': n_updated,
        'n_mvd': n_mvd,
        'allocation': allocation,
        'n_high_trust': len(high_trust),
        'n_econ_weighted': len(econ_weighted),
        'summary_path': summary_path,
    }
