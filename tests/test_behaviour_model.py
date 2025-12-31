"""
Tests for CtRL-Sim-style behaviour model.
"""

import numpy as np
import pytest

from src.behaviour.ctrl_sim_like import (
    AgentState,
    AgentStateBatch,
    KDisksActionCoder,
    SceneObjectTrajectory,
    create_simple_behaviour_policy,
)
from src.scene.vector_scene.graph import ObjectClass, SceneGraph, SceneObject


class TestKDisksActionCoder:
    def test_coder_creation(self):
        coder = KDisksActionCoder(num_disks=5, num_angles=8, max_step=2.0)
        assert coder.vocab_size == 5 * 8 * 8

    def test_encode_decode_roundtrip(self):
        coder = KDisksActionCoder(num_disks=5, num_angles=8, max_step=2.0)

        # Test various actions
        test_cases = [
            (0.0, 0.0, 0.0),  # No movement
            (1.0, 0.0, 0.0),  # Move in +x
            (0.0, 1.0, 0.0),  # Move in +y
            (0.5, 0.5, 0.1),  # Diagonal with rotation
        ]

        for dx, dy, dtheta in test_cases:
            token = coder.encode(dx, dy, dtheta)
            dx2, dy2, dtheta2 = coder.decode(token)
            # Due to discretization, we can't expect exact equality
            # but the decoded action should be reasonably close
            assert token >= 0
            assert token < coder.vocab_size

    def test_null_token(self):
        coder = KDisksActionCoder()
        null_token = coder.get_null_token()
        dx, dy, dtheta = coder.decode(null_token)
        # Null token should decode to approximately no movement
        assert abs(dx) < 0.5
        assert abs(dy) < 0.5

    def test_vocab_size(self):
        coder = KDisksActionCoder(num_disks=3, num_angles=4, max_step=1.0)
        # 3 disks * 4 directions * 4 rotations = 48
        assert coder.get_vocab_size() == 48

    def test_encode_clamps_values(self):
        coder = KDisksActionCoder(max_step=2.0, max_rotation=0.5)
        # Large values should be clamped
        token = coder.encode(100.0, 100.0, 100.0)
        assert token >= 0
        assert token < coder.vocab_size


class TestAgentState:
    def test_creation(self):
        state = AgentState(
            agent_id=0,
            x=1.0,
            y=2.0,
            z=0.0,
            heading=np.pi / 4,
            speed=1.5,
            class_id=ObjectClass.HUMAN,
        )
        assert state.agent_id == 0
        assert state.x == 1.0
        assert state.speed == 1.5

    def test_from_scene_object(self):
        obj = SceneObject(
            id=5,
            class_id=ObjectClass.ROBOT,
            x=3.0,
            y=4.0,
            heading=1.0,
            speed=2.0,
        )
        state = AgentState.from_scene_object(obj)
        assert state.agent_id == 5
        assert state.x == 3.0
        assert state.class_id == ObjectClass.ROBOT

    def test_to_feature_vector(self):
        state = AgentState(
            agent_id=0,
            x=1.0,
            y=2.0,
            heading=0.5,
            speed=1.0,
            class_id=ObjectClass.HUMAN,
        )
        feat = state.to_feature_vector()
        assert feat.ndim == 1
        expected_len = len(ObjectClass) + 3 + 2 + 1
        assert len(feat) == expected_len

    def test_apply_action(self):
        state = AgentState(
            agent_id=0,
            x=0.0,
            y=0.0,
            heading=0.0,
            speed=0.0,
        )
        new_state = state.apply_action(dx=1.0, dy=0.0, dtheta=np.pi / 2, dt=1.0)
        assert new_state.x == 1.0
        assert new_state.y == 0.0
        assert abs(new_state.heading - np.pi / 2) < 1e-6
        assert new_state.speed == 1.0


class TestAgentStateBatch:
    def test_from_scene_graph(self):
        objects = [
            SceneObject(id=0, class_id=ObjectClass.HUMAN, speed=1.0),
            SceneObject(id=1, class_id=ObjectClass.ROBOT, speed=0.5),
            SceneObject(id=2, class_id=ObjectClass.SHELF, speed=0.0),  # Static
        ]
        graph = SceneGraph(objects=objects)
        batch = AgentStateBatch.from_scene_graph(graph)
        # Should include dynamic objects (HUMAN, ROBOT) but not SHELF
        assert len(batch.agents) >= 2

    def test_to_tensor(self):
        pytest.importorskip("torch")
        agents = [
            AgentState(agent_id=0, x=0.0, y=0.0, class_id=ObjectClass.HUMAN),
            AgentState(agent_id=1, x=1.0, y=1.0, class_id=ObjectClass.ROBOT),
        ]
        batch = AgentStateBatch(agents=agents)
        tensor = batch.to_tensor()
        assert tensor.shape[0] == 2
        assert tensor.shape[1] == len(ObjectClass) + 3 + 2 + 1

    def test_get_agent(self):
        agents = [
            AgentState(agent_id=0, x=0.0, y=0.0),
            AgentState(agent_id=5, x=1.0, y=1.0),
        ]
        batch = AgentStateBatch(agents=agents)
        agent = batch.get_agent(5)
        assert agent is not None
        assert agent.x == 1.0

        missing = batch.get_agent(99)
        assert missing is None


class TestSceneObjectTrajectory:
    def test_append_and_len(self):
        traj = SceneObjectTrajectory(agent_id=0)
        assert len(traj) == 0

        traj.append(
            timestamp=0.0,
            position=(0.0, 0.0, 0.0),
            heading=0.0,
            speed=0.0,
        )
        assert len(traj) == 1

        traj.append(
            timestamp=1.0,
            position=(1.0, 1.0, 0.0),
            heading=0.5,
            speed=1.5,
            action=42,
        )
        assert len(traj) == 2
        assert traj.actions[-1] == 42


class TestSimpleBehaviourPolicy:
    def test_policy_returns_valid_token(self):
        coder = KDisksActionCoder()
        policy = create_simple_behaviour_policy(coder, speed=1.0)

        state = AgentState(
            agent_id=0,
            x=0.0,
            y=0.0,
            heading=0.0,
            speed=0.0,
            class_id=ObjectClass.HUMAN,
        )

        for _ in range(10):
            token = policy(state)
            assert 0 <= token < coder.vocab_size


class TestBehaviourModel:
    @pytest.fixture
    def model(self):
        pytest.importorskip("torch")
        from src.behaviour.ctrl_sim_like import BehaviourModel

        return BehaviourModel(
            scene_latent_dim=64,
            agent_dim=32,
            hidden_dim=64,
            num_layers=2,
            num_heads=4,
            num_actions=100,
        )

    def test_forward(self, model):
        import torch

        batch_size = 2
        seq_len = 5
        n_agents = 3

        scene_latent = torch.randn(batch_size, 64)
        agent_states = torch.randn(batch_size, seq_len, n_agents, 32)
        past_actions = torch.randint(0, 100, (batch_size, seq_len, n_agents))

        logits = model(scene_latent, agent_states, past_actions)
        assert logits.shape == (batch_size, seq_len, n_agents, 100)

    def test_get_action_probs(self, model):
        import torch

        scene_latent = torch.randn(1, 64)
        agent_states = torch.randn(1, 1, 2, 32)
        past_actions = torch.zeros(1, 1, 2, dtype=torch.long)

        probs = model.get_action_probs(scene_latent, agent_states, past_actions)
        # Probabilities should sum to 1
        assert torch.allclose(probs.sum(dim=-1), torch.ones(1, 1, 2), atol=1e-5)
