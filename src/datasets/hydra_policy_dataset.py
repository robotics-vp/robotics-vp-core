"""
Phase I Hydra policy dataset loader.

Builds ConditionVectors from Stage 5-ready signals and exports a compact set of
training features (econ, OOD, recovery, novelty, TFD, Phase H).
"""
import hashlib
from typing import Any, Dict

from src.datasets.base import Phase1DatasetBase, set_deterministic_seeds
from src.observation.condition_vector_builder import ConditionVectorBuilder


class HydraPolicyDataset(Phase1DatasetBase):
    name = "hydra_policy_phase1"

    def __init__(self, *args, seed: int = 0, curriculum_phase: str = "stage5", **kwargs) -> None:
        set_deterministic_seeds(seed)
        self.curriculum_phase = curriculum_phase
        self.condition_builder = ConditionVectorBuilder()
        super().__init__(*args, seed=seed, **kwargs)

    def _augment_sample(self, sample: Dict[str, Any], idx: int) -> Dict[str, Any]:
        frame = sample.get("stage1_frame", {})
        seg = sample.get("stage2_segments", {})
        stress = sample.get("sima2_stress", {})
        trust_tag = sample.get("trust_tag") or self._resolve_trust_tag(sample)
        trust_score = self._resolve_trust_weight(trust_tag)

        econ_state = self._build_econ_state(frame, seg, idx)
        episode_config = {
            "task_id": frame.get("task", "phase1_task"),
            "env_id": seg.get("source", "stage5_env"),
            "backend_id": "isaac" if sample.get("isaac_rollout", {}).get("source") == "isaac_adapter" else "pybullet",
            "objective_vector": [1.0, 1.0, 1.0, 1.0, 0.0],
        }
        datapack_metadata = {"tags": frame.get("tags", []), "pack_id": frame.get("pack_id")}
        episode_metadata = {"episode_id": frame.get("pack_id"), "tfd_instruction": {"status": "synthetic", "condition_vector": {"novelty_tier": idx % 3}}}
        advisory_context = {"skill_mode": seg.get("source", "default"), "frontier_score": stress.get("severity", 0.0)}
        econ_slice = {"mpl": econ_state["target_mpl"], "energy_wh": econ_state["energy_budget_wh"], "wage_parity": econ_state["current_wage_parity"]}
        trust_summary = {trust_tag: {"trust_score": trust_score}}

        condition = self.condition_builder.build(
            episode_config=episode_config,
            econ_state=econ_state,
            curriculum_phase=self.curriculum_phase,
            sima2_trust=trust_score,
            datapack_metadata=datapack_metadata,
            econ_slice=econ_slice,
            semantic_tags={t: 1.0 for t in frame.get("tags", [])},
            episode_metadata=episode_metadata,
            advisory_context=advisory_context,
            trust_summary=trust_summary,
            enable_tfd_integration=True,
            enable_phase_h_advisories=True,
            phase_h_advisory=None,
        )

        sample["condition_vector"] = condition.to_dict()
        sample["condition_features"] = self.condition_builder.to_training_features(condition)
        sample["trust_tag"] = trust_tag
        return sample

    def _build_econ_state(self, frame: Dict[str, Any], seg: Dict[str, Any], idx: int) -> Dict[str, float]:
        digest_val = int(hashlib.sha256(f"{frame.get('pack_id')}:{idx}:{self.seed}".encode("utf-8")).hexdigest()[:8], 16)
        parity = 0.8 + (digest_val % 20) / 100.0
        target_mpl = 1.0 + (digest_val % 10) * 0.05
        energy_budget = 50.0 + (seg.get("energy_intensity", 0.0) * 10.0)
        return {
            "current_wage_parity": float(parity),
            "target_mpl": float(target_mpl),
            "energy_budget_wh": float(energy_budget),
        }

