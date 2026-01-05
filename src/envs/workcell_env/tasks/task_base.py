"""
Task specifications for workcell environments.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal

TaskType = Literal["ASSEMBLY", "SORTING", "INSPECTION", "PACKAGING", "CUSTOM"]
RewardType = Literal["SPARSE", "DENSE", "SHAPED"]
ActionType = Literal["PICK", "PLACE", "INSERT", "FASTEN", "ROUTE", "INSPECT", "PACKAGE"]


class _JsonMixin:
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str):
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))


@dataclass(frozen=True)
class TaskSpec(_JsonMixin):
    """
    Specification for a single workcell task.
    """
    task_id: str
    task_type: TaskType
    success_metric: str
    reward_type: RewardType
    time_limit: int
    parameters: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "success_metric": self.success_metric,
            "reward_type": self.reward_type,
            "time_limit": self.time_limit,
            "parameters": dict(self.parameters),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskSpec":
        """Deserialize from dictionary."""
        return cls(
            task_id=data["task_id"],
            task_type=data.get("task_type", "CUSTOM"),
            success_metric=data.get("success_metric", "success"),
            reward_type=data.get("reward_type", "SPARSE"),
            time_limit=data.get("time_limit", 0),
            parameters=data.get("parameters", {}),
        )


@dataclass(frozen=True)
class ActionStepSpec(_JsonMixin):
    """
    Specification for an atomic action step in a task graph.
    """
    step_id: str
    action_type: ActionType
    target_object_id: str
    constraints: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "step_id": self.step_id,
            "action_type": self.action_type,
            "target_object_id": self.target_object_id,
            "constraints": dict(self.constraints),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ActionStepSpec":
        """Deserialize from dictionary."""
        return cls(
            step_id=data["step_id"],
            action_type=data.get("action_type", "PICK"),
            target_object_id=data.get("target_object_id", ""),
            constraints=data.get("constraints", {}),
        )


@dataclass(frozen=True)
class TaskGraphSpec(_JsonMixin):
    """
    Task graph specification with nodes and dependency edges.
    """
    nodes: List[ActionStepSpec]
    edges: List[tuple[str, str]]
    entry_node: str
    exit_nodes: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [list(edge) for edge in self.edges],
            "entry_node": self.entry_node,
            "exit_nodes": list(self.exit_nodes),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskGraphSpec":
        """Deserialize from dictionary."""
        nodes = [ActionStepSpec.from_dict(n) for n in data.get("nodes", [])]
        edges = []
        for edge in data.get("edges", []):
            if isinstance(edge, tuple):
                edges.append(edge)
            elif isinstance(edge, list) and len(edge) == 2:
                edges.append((edge[0], edge[1]))
        return cls(
            nodes=nodes,
            edges=edges,
            entry_node=data.get("entry_node", ""),
            exit_nodes=data.get("exit_nodes", []),
        )
