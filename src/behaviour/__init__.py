"""
Behaviour modelling for dynamic agents in scenes.

Provides CtRL-Sim-style autoregressive behaviour models for:
- Action tokenization (K-disks discretization)
- Behaviour prediction conditioned on scene context
- Return-conditional sampling for adversarial/cooperative agents
"""

from src.behaviour.ctrl_sim_like import (
    KDisksActionCoder,
    AgentState,
    AgentStateBatch,
    SceneObjectTrajectory,
    BehaviourModel,
    rollout_behaviour,
)

__all__ = [
    "KDisksActionCoder",
    "AgentState",
    "AgentStateBatch",
    "SceneObjectTrajectory",
    "BehaviourModel",
    "rollout_behaviour",
]
