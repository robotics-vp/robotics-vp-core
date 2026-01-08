import numpy as np
import pytest
import torch

from src.envs.lsd_vector_scene_env import (
    LSDVectorSceneEnv,
    LSDVectorSceneEnvConfig,
    SceneGraphConfig,
    BehaviourConfig,
)
from src.envs.lsd3d_env.motion_hierarchy_integration import build_motion_hierarchy_input_from_lsd_episode
from src.motor_backend.lsd_vector_scene_backend import _serialize_agent_trajectories
from src.vision.motion_hierarchy.config import MotionHierarchyConfig
from src.vision.motion_hierarchy.motion_hierarchy_node import MotionHierarchyNode


@pytest.mark.slow
def test_lsd_motion_hierarchy_integration():
    config = LSDVectorSceneEnvConfig(
        scene_graph_config=SceneGraphConfig(topology_type="WAREHOUSE_AISLES", num_nodes=5, seed=123),
        behaviour_config=BehaviourConfig(num_humans=1, num_robots=0, num_forklifts=0, use_simple_policy=True),
        max_steps=8,
    )
    env = LSDVectorSceneEnv(config)
    env.reset()
    for _ in range(5):
        env.step(np.array([0.5, 0.5]))

    traj_payload = _serialize_agent_trajectories(env.get_agent_trajectories(), env.graph)
    episode = {"episode_id": "test_episode", "trajectory_data": traj_payload}

    mh_input = build_motion_hierarchy_input_from_lsd_episode(episode)

    assert mh_input.positions.ndim == 3
    assert len(mh_input.node_labels) == mh_input.positions.shape[1]
    assert mh_input.mask is None or mh_input.mask.shape[0] == mh_input.positions.shape[1]

    positions = mh_input.positions.unsqueeze(0)
    config = MotionHierarchyConfig(d_model=32, num_gnn_layers=2, k_neighbors=3, l_max=2, device="cpu")
    model = MotionHierarchyNode(config)
    out = model(positions, mask=mh_input.mask.unsqueeze(0) if mh_input.mask is not None else None)
    assert out["hierarchy"].shape[-1] == out["hierarchy"].shape[-2]
    assert not torch.isnan(out["hierarchy"]).any()
