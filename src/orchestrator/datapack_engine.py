from typing import Any, Dict, Optional, List
from dataclasses import dataclass, field
import numpy as np

from src.valuation.datapack_repo import DataPackRepo
from src.valuation.datapack_schema import DataPackMeta, ObjectiveProfile
from src.orchestrator.semantic_metrics import SemanticMetrics
from src.orchestrator.economic_controller import EconSignals


@dataclass
class DatapackSignals:
    """
    Data valuation and novelty signals from datapacks.

    These signals feed into the SemanticOrchestrator alongside EconSignals.

    IMPORTANT: DatapackEngine is UPSTREAM of SemanticOrchestrator.
    SemanticOrchestrator consumes these signals - it does not define them.
    """
    # Coverage metrics
    total_datapacks: int = 0
    positive_fraction: float = 0.0
    negative_fraction: float = 0.0

    # Novelty signals
    mean_novelty: float = 0.0
    max_novelty: float = 0.0
    novelty_variance: float = 0.0

    # Tier distribution
    tier0_fraction: float = 0.0  # Redundant
    tier1_fraction: float = 0.0  # Context-novel
    tier2_fraction: float = 0.0  # Causal-novel / frontier

    # Data quality
    data_coverage_score: float = 0.0  # How well data covers state space
    embedding_diversity: float = 0.0  # Diversity of episode embeddings

    # VLA/semantic richness
    vla_annotation_fraction: float = 0.0
    guidance_annotation_fraction: float = 0.0
    semantic_tag_diversity: int = 0

    # Data economics (aggregated)
    mean_rebate_pct: float = 0.0
    mean_spread_capture: float = 0.0
    mean_data_premium: float = 0.0

    # Recommendations
    data_gaps: List[str] = field(default_factory=list)
    # e.g., ["edge_cases", "occlusion", "low_light"]
    recommended_collection_focus: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_datapacks": self.total_datapacks,
            "positive_fraction": self.positive_fraction,
            "negative_fraction": self.negative_fraction,
            "mean_novelty": self.mean_novelty,
            "max_novelty": self.max_novelty,
            "novelty_variance": self.novelty_variance,
            "tier0_fraction": self.tier0_fraction,
            "tier1_fraction": self.tier1_fraction,
            "tier2_fraction": self.tier2_fraction,
            "data_coverage_score": self.data_coverage_score,
            "embedding_diversity": self.embedding_diversity,
            "vla_annotation_fraction": self.vla_annotation_fraction,
            "guidance_annotation_fraction": self.guidance_annotation_fraction,
            "semantic_tag_diversity": self.semantic_tag_diversity,
            "mean_rebate_pct": self.mean_rebate_pct,
            "mean_spread_capture": self.mean_spread_capture,
            "mean_data_premium": self.mean_data_premium,
            "data_gaps": self.data_gaps,
            "recommended_collection_focus": self.recommended_collection_focus,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DatapackSignals":
        """Create from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class DatapackEngine:
    """
    Engine for computing data valuation and novelty signals.

    IMPORTANT HIERARCHY:
    - DatapackEngine is UPSTREAM (defines data value physics)
    - SemanticOrchestrator is DOWNSTREAM (applies data value to meaning)

    This module DOES NOT import SemanticOrchestrator or MetaTransformer.
    """
    def __init__(self, repo: DataPackRepo):
        self.repo = repo

    def compute_datapack_stats(self) -> Dict[str, Any]:
        stats = {}
        for task_file in self.repo._cache.keys():
            task_name = task_file
            dps = self.repo.load_all(task_name)
            if not dps:
                continue
            stats[task_name] = {
                "mpl_mean": float(np.mean([dp.attribution.delta_mpl for dp in dps])),
                "error_mean": float(np.mean([dp.attribution.delta_error for dp in dps])),
                "energy_Wh_mean": float(np.mean([dp.energy.total_Wh for dp in dps])),
                "rebate_pct_mean": float(np.mean([getattr(dp.attribution, "rebate_pct", 0.0) for dp in dps])),
                "spread_mean": float(np.mean([getattr(dp.attribution, "attributable_spread_capture", 0.0) for dp in dps])),
                "data_premium_mean": float(np.mean([getattr(dp.attribution, "data_premium", 0.0) for dp in dps])),
                "tags_energy": list(self._aggregate_tags(dps, field="energy_driver_tags")),
                "tags_guidance": list(self._aggregate_guidance_tags(dps)),
            }
        return stats

    def _aggregate_tags(self, dps: List[DataPackMeta], field: str):
        tags = {}
        for dp in dps:
            for t in getattr(dp, field, []) or []:
                tags[t] = tags.get(t, 0) + 1
        return tags.items()

    def _aggregate_guidance_tags(self, dps: List[DataPackMeta]):
        tags = {}
        for dp in dps:
            gp = dp.guidance_profile
            if gp:
                for t in gp.semantic_tags:
                    tags[t] = tags.get(t, 0) + 1
        return tags.items()

    def compute_novelty_scores(self) -> Dict[str, float]:
        # Placeholder: novelty based on inverse density of energy_Wh
        scores = {}
        for task_file in self.repo._cache.keys():
            dps = self.repo.load_all(task_file)
            energies = np.array([dp.energy.total_Wh for dp in dps]) if dps else np.array([])
            if len(energies) == 0:
                continue
            mean_energy = np.mean(energies)
            for dp in dps:
                scores[dp.pack_id] = float(abs(dp.energy.total_Wh - mean_energy))
        return scores

    def suggest_sampling_weights(
        self,
        target_skill_id: Optional[int] = None,
        objective_profile: Optional[ObjectiveProfile] = None,
    ) -> Dict[str, float]:
        weights = {}
        novelty = self.compute_novelty_scores()
        for task_file in self.repo._cache.keys():
            dps = self.repo.load_all(task_file)
            for dp in dps:
                w = 1.0
                if dp.pack_id in novelty:
                    w += novelty[dp.pack_id]
                if dp.bucket == "positive":
                    w *= 1.1
                w += float(dp.attribution.delta_mpl)
                w -= float(dp.attribution.delta_error)
                w -= float(dp.energy.total_Wh)
                w += float(getattr(dp.attribution, "rebate_pct", 0.0))
                weights[dp.pack_id] = max(0.0, w)
        return weights

    def update_sampling_from_semantics(
        self,
        metrics: SemanticMetrics,
    ) -> Dict[str, float]:
        """
        Advisory sampling multipliers based on semantic coverage/drift.
        """
        overrides: Dict[str, float] = {}
        if metrics.econ_ignored_task_fraction > 0.4:
            overrides["tag:econ_irrelevant"] = 0.7
        for t in metrics.underrepresented_tasks:
            overrides[f"task:{t}"] = 1.2
        return overrides

    def export_sampling_overrides(
        self,
        metrics: SemanticMetrics,
        out_path: str,
    ) -> None:
        overrides = self.update_sampling_from_semantics(metrics)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(overrides, f, indent=2)

    def compute_signals(
        self,
        datapacks: List[DataPackMeta],
        econ_signals: Optional[EconSignals] = None,
    ) -> DatapackSignals:
        """
        Compute data valuation signals from datapacks.

        PRIMARY entry point for downstream modules (SemanticOrchestrator)
        to get current data quality and novelty state.

        Args:
            datapacks: List of DataPackMeta objects
            econ_signals: Optional EconSignals for context-aware analysis

        Returns:
            DatapackSignals with computed metrics
        """
        signals = DatapackSignals()

        if not datapacks:
            return signals

        signals.total_datapacks = len(datapacks)

        # Bucket distribution
        positive_count = sum(1 for dp in datapacks if dp.bucket == "positive")
        negative_count = len(datapacks) - positive_count
        signals.positive_fraction = positive_count / len(datapacks)
        signals.negative_fraction = negative_count / len(datapacks)

        # Novelty scores
        novelty_scores = []
        for dp in datapacks:
            # Use delta_mpl variance as proxy for novelty (placeholder)
            # In production, use actual novelty computation from embedding_utils
            novelty_scores.append(abs(dp.attribution.delta_mpl))

        if novelty_scores:
            signals.mean_novelty = float(np.mean(novelty_scores))
            signals.max_novelty = float(np.max(novelty_scores))
            signals.novelty_variance = float(np.var(novelty_scores))

        # Tier distribution (heuristic based on novelty)
        tier0_count = 0
        tier1_count = 0
        tier2_count = 0
        tier1_threshold = signals.mean_novelty * 0.5
        tier2_threshold = signals.mean_novelty * 1.5

        for score in novelty_scores:
            if score < tier1_threshold:
                tier0_count += 1
            elif score < tier2_threshold:
                tier1_count += 1
            else:
                tier2_count += 1

        total = len(novelty_scores)
        signals.tier0_fraction = tier0_count / total if total > 0 else 0.0
        signals.tier1_fraction = tier1_count / total if total > 0 else 0.0
        signals.tier2_fraction = tier2_count / total if total > 0 else 0.0

        # VLA/guidance annotation coverage
        vla_count = sum(1 for dp in datapacks if dp.vla_action_summary is not None)
        guidance_count = sum(1 for dp in datapacks if dp.guidance_profile is not None)
        signals.vla_annotation_fraction = vla_count / len(datapacks)
        signals.guidance_annotation_fraction = guidance_count / len(datapacks)

        # Embedding diversity
        embeddings = [dp.episode_embedding for dp in datapacks if dp.episode_embedding is not None]
        if len(embeddings) >= 2:
            # Compute mean pairwise distance as diversity metric
            emb_array = np.array(embeddings)
            pairwise_dists = []
            for i in range(min(50, len(embeddings))):  # Sample pairs
                for j in range(i + 1, min(50, len(embeddings))):
                    pairwise_dists.append(np.linalg.norm(emb_array[i] - emb_array[j]))
            if pairwise_dists:
                signals.embedding_diversity = float(np.mean(pairwise_dists))

        # Semantic tag diversity
        all_tags = set()
        for dp in datapacks:
            all_tags.update(dp.semantic_tags or [])
            all_tags.update(dp.energy_driver_tags or [])
            if dp.guidance_profile:
                all_tags.update(dp.guidance_profile.semantic_tags or [])
        signals.semantic_tag_diversity = len(all_tags)

        # Data economics
        rebates = [getattr(dp.attribution, "rebate_pct", 0.0) for dp in datapacks]
        spreads = [getattr(dp.attribution, "attributable_spread_capture", 0.0) for dp in datapacks]
        premiums = [getattr(dp.attribution, "data_premium", 0.0) for dp in datapacks]

        signals.mean_rebate_pct = float(np.mean(rebates)) if rebates else 0.0
        signals.mean_spread_capture = float(np.mean(spreads)) if spreads else 0.0
        signals.mean_data_premium = float(np.mean(premiums)) if premiums else 0.0

        # Data coverage score (heuristic)
        signals.data_coverage_score = min(1.0, signals.tier2_fraction * 2.0 + signals.tier1_fraction)

        # Identify data gaps
        gaps = []
        if signals.tier2_fraction < 0.1:
            gaps.append("frontier_cases")
        if signals.vla_annotation_fraction < 0.5:
            gaps.append("vla_annotations")
        if signals.embedding_diversity < 0.5 and len(embeddings) > 5:
            gaps.append("embedding_diversity")
        if signals.negative_fraction < 0.2:
            gaps.append("negative_examples")

        # Check for specific condition gaps
        conditions = {}
        for dp in datapacks:
            lighting = dp.condition.lighting_profile
            conditions[lighting] = conditions.get(lighting, 0) + 1
        if "low_light" not in conditions:
            gaps.append("low_light_conditions")
        if "high_contrast" not in conditions:
            gaps.append("high_contrast_conditions")

        signals.data_gaps = gaps

        # Recommend focus area based on urgencies
        if econ_signals:
            if econ_signals.error_urgency > 0.5:
                signals.recommended_collection_focus = "safety_edge_cases"
            elif econ_signals.mpl_urgency > 0.5:
                signals.recommended_collection_focus = "throughput_demonstrations"
            elif econ_signals.energy_urgency > 0.5:
                signals.recommended_collection_focus = "energy_efficient_trajectories"
            elif len(gaps) > 0:
                signals.recommended_collection_focus = gaps[0]
            else:
                signals.recommended_collection_focus = "balanced"
        else:
            signals.recommended_collection_focus = gaps[0] if gaps else "balanced"

        return signals
