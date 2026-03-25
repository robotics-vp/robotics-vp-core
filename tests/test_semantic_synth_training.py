import json

import numpy as np
import torch

from scripts.train_latent_diffusion import ZVTransitionDataset, train_latent_dynamics
from scripts.train_world_model_from_datapacks import DataPackWorldModelDataset
from src.valuation.datapack_schema import (
    AttributionProfile,
    ConditionProfile,
    ProcessRewardProfile,
    create_positive_datapack,
)
from src.valuation.episode_features import make_full_datapack_features


def _write_transition_dataset(tmp_path):
    dataset_path = tmp_path / "zv_rollouts.npz"
    np.savez(
        dataset_path,
        n_episodes=np.array(2),
        latent_dim=np.array(4),
        ep_0_z_sequence=np.array([[0.0, 0.1, 0.2, 0.3], [0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.5]], dtype=np.float32),
        ep_0_actions=np.array([[0.1, 0.2], [0.2, 0.1]], dtype=np.float32),
        ep_1_z_sequence=np.array([[0.2, 0.0, 0.1, 0.3], [0.3, 0.1, 0.2, 0.4], [0.4, 0.2, 0.3, 0.5]], dtype=np.float32),
        ep_1_actions=np.array([[0.3, 0.2], [0.4, 0.1]], dtype=np.float32),
    )
    sidecar_path = tmp_path / "zv_rollouts_semantic_cond.json"
    sidecar_path.write_text(
        json.dumps(
            [
                {"episode_id": 0, "cond_vector": [1.0, 0.0, 0.5]},
                {"episode_id": 1, "cond_vector": [0.2, 1.0, 0.1]},
            ]
        ),
        encoding="utf-8",
    )
    return dataset_path, sidecar_path


def test_zv_transition_dataset_loads_semantic_conditioning(tmp_path) -> None:
    dataset_path, sidecar_path = _write_transition_dataset(tmp_path)
    dataset = ZVTransitionDataset(
        dataset_path,
        use_semantic_conditioning=True,
        semantic_sidecar_path=str(sidecar_path),
    )
    assert dataset.semantic_cond_dim == 3
    item = dataset[0]
    assert len(item) == 4
    assert item[-1].shape[0] == 3


def test_train_latent_dynamics_persists_semantic_conditioning(tmp_path) -> None:
    dataset_path, sidecar_path = _write_transition_dataset(tmp_path)
    _, save_path = train_latent_dynamics(
        dataset_path=str(dataset_path),
        n_epochs=1,
        batch_size=2,
        lr=1e-3,
        hidden_dim=16,
        save_dir=str(tmp_path),
        use_semantic_conditioning=True,
        semantic_sidecar_path=str(sidecar_path),
        device=torch.device("cpu"),
    )
    checkpoint = torch.load(save_path, map_location="cpu", weights_only=False)
    assert checkpoint["use_semantic_conditioning"] is True
    assert checkpoint["semantic_cond_dim"] == 3


def test_datapack_world_model_dataset_appends_semantic_gap_features() -> None:
    datapack = create_positive_datapack(
        task_name="drawer_vase",
        condition=ConditionProfile(task_name="drawer_vase"),
        attribution=AttributionProfile(delta_J=0.2, delta_mpl=0.1, trust_score=0.8, w_econ=0.7),
        skill_trace=[{"skill_id": 1, "duration": 4}],
        semantic_tags=["fragile", "risk:collision", "workcell"],
        episode_id="ep_1",
    )
    datapack.semantic_quality = 0.65
    datapack.process_reward_profile = ProcessRewardProfile(phi_star_mean=0.8, conf_mean=0.9)
    datapack.episode_metrics = {"coverage_gap_contribution": 0.4}
    datapack.regal_annotations = {
        "semantic_coverage": {
            "coverage_summary": {"total_edges": 10, "missing_edges": 3},
            "feedback_summary": {"graph_mutation_pressure": 1.5},
        }
    }

    base_dim = make_full_datapack_features(datapack).shape[0]
    dataset = DataPackWorldModelDataset([datapack])
    assert dataset.features.shape[1] > base_dim
    assert dataset.weights[0] > datapack.attribution.trust_score * datapack.attribution.w_econ
