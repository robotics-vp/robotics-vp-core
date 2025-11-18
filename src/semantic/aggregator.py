"""
SemanticAggregator builds a unified SemanticSnapshot from Stage 2/3/meta outputs.
"""
from typing import Any, Dict, Sequence, Optional, List
from datetime import datetime

from src.semantic.models import SemanticSnapshot, EconSlice, MetaTransformerSlice
from src.analytics.econ_reports import compute_task_econ_summary
from src.ontology.store import OntologyStore
from src.orchestrator.meta_transformer import MetaTransformerOutputs
from src.sima2.ontology_proposals import OntologyUpdateProposal
from src.sima2.task_graph_proposals import TaskGraphRefinementProposal
from src.sima2.tags.semantic_tags import SemanticEnrichmentProposal as SemanticTag
from src.policies.registry import build_all_policies


class SemanticAggregator:
    def __init__(self, ontology_store: OntologyStore, policies=None):
        self.store = ontology_store
        self.policies = policies or build_all_policies()

    def build_snapshot(
        self,
        task_id: str,
        stage2_ontology_proposals: Sequence[OntologyUpdateProposal],
        stage2_task_refinements: Sequence[TaskGraphRefinementProposal],
        stage2_tags: Sequence[SemanticTag],
        meta_outputs: Optional[MetaTransformerOutputs] = None,
        recap_scores: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> SemanticSnapshot:
        econ_slice = self._build_econ_slice(task_id)
        meta_slice = self._build_meta_slice(task_id, meta_outputs)
        recap_summary = self._build_recap_summary(task_id, recap_scores or {})

        proposals = sorted(list(stage2_ontology_proposals), key=lambda p: p.proposal_id)
        refinements = sorted(list(stage2_task_refinements), key=lambda r: r.proposal_id)
        tags = sorted(list(stage2_tags), key=lambda t: getattr(t, "proposal_id", repr(t)))

        snapshot = SemanticSnapshot(
            task_id=task_id,
            ontology_proposals=proposals,
            task_refinements=refinements,
            semantic_tags=tags,
            econ_slice=econ_slice,
            meta_slice=meta_slice,
            timestamp=datetime.utcnow().timestamp(),
            metadata={"recap": recap_summary} if recap_summary else {},
        )
        return snapshot.sorted_copy()

    def _build_econ_slice(self, task_id: str) -> EconSlice:
        summary = compute_task_econ_summary(self.store, task_id)
        episodes = self.store.list_episodes(task_id=task_id)
        frontier_episodes = [e.episode_id for e in episodes if getattr(e, "status", "") == "success"][:10]

        return EconSlice(
            task_id=task_id,
            avg_mpl_units_per_hour=summary.get("mpl", {}).get("mean", 0.0),
            avg_wage_parity=summary.get("wage_parity", {}).get("mean", 0.0),
            avg_energy_cost=summary.get("energy_cost", {}).get("mean", 0.0),
            avg_error_rate=summary.get("damage_cost", {}).get("mean", 0.0),
            frontier_episodes=frontier_episodes,
            metadata={"counts": summary.get("counts", {})},
        )

    def _build_meta_slice(self, task_id: str, meta_outputs: Optional[MetaTransformerOutputs]) -> MetaTransformerSlice:
        if not meta_outputs:
            return MetaTransformerSlice(task_id=task_id)
        expected_deltas = {
            "expected_delta_mpl": meta_outputs.expected_delta_mpl,
            "expected_delta_error": meta_outputs.expected_delta_error,
            "expected_delta_energy_Wh": meta_outputs.expected_delta_energy_Wh,
        }
        return MetaTransformerSlice(
            task_id=task_id,
            objective_vectors=[meta_outputs.energy_profile_weights],
            presets=[meta_outputs.objective_preset],
            expected_deltas=expected_deltas,
            backends=[meta_outputs.chosen_backend],
            metadata={
                "data_mix_weights": meta_outputs.data_mix_weights,
                "authority": getattr(meta_outputs, "authority", ""),
            },
        )

    def _build_recap_summary(self, task_id: str, recap_scores: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        if not recap_scores:
            return {}
        policy = getattr(self.policies, "episode_quality", None)
        task_eps = [ep for ep in recap_scores.values() if ep.get("episode_id", "").startswith("")]  # no-op filter for determinism
        goodness_vals = []
        for ep in task_eps:
            score = float(ep.get("recap_goodness_score", 0.0))
            if policy:
                feats = policy.build_features([], [], [], {"recap_goodness_score": score})
                evaluated = policy.evaluate(feats)
                score = float(evaluated.get("quality_score", score))
            goodness_vals.append(score)
            ep["recap_goodness_score"] = score
        mean_goodness = float(sum(goodness_vals) / len(goodness_vals)) if goodness_vals else 0.0
        top = sorted(task_eps, key=lambda e: e.get("recap_goodness_score", 0.0), reverse=True)[:5]
        return {
            "task_id": task_id,
            "count": len(task_eps),
            "mean_goodness": mean_goodness,
            "top_episodes": [t.get("episode_id") for t in top],
        }
