"""
Task Graph: Semantic structure for task decomposition.

Provides a hierarchical representation of tasks, sub-tasks, and their dependencies.
Used by the orchestrator to understand task structure and plan execution.

This is additive infrastructure - no changes to Phase B math or RL training loops.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
from enum import Enum
import json


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


class TaskType(Enum):
    """Type of task in the graph."""
    ROOT = "root"  # Top-level task
    COMPOSITE = "composite"  # Task with sub-tasks
    ATOMIC = "atomic"  # Leaf task (no sub-tasks)
    SKILL = "skill"  # Maps to a specific skill_id
    CHECKPOINT = "checkpoint"  # Milestone/verification point


@dataclass
class TaskNode:
    """
    A node in the task graph representing a single task or sub-task.

    Attributes:
        task_id: Unique identifier for the task
        name: Human-readable task name
        description: Detailed description of what the task accomplishes
        task_type: Type of task (ROOT, COMPOSITE, ATOMIC, SKILL, CHECKPOINT)
        status: Current execution status
        skill_id: If SKILL type, the skill_id to execute
        preconditions: List of task_ids that must complete before this task
        postconditions: List of state predicates that should be true after completion
        children: List of sub-task nodes (for COMPOSITE tasks)
        parent_id: ID of parent task (None for ROOT)
        metadata: Additional task-specific metadata
        affordances: List of affordances required for this task
        objects_involved: List of objects involved in this task
    """
    task_id: str
    name: str
    description: str = ""
    task_type: TaskType = TaskType.ATOMIC
    status: TaskStatus = TaskStatus.PENDING
    skill_id: Optional[int] = None
    preconditions: List[str] = field(default_factory=list)
    postconditions: List[str] = field(default_factory=list)
    children: List["TaskNode"] = field(default_factory=list)
    parent_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    affordances: List[str] = field(default_factory=list)
    objects_involved: List[str] = field(default_factory=list)

    def is_leaf(self) -> bool:
        """Check if this is a leaf node (no children)."""
        return len(self.children) == 0

    def is_ready(self) -> bool:
        """Check if task is ready to execute (all preconditions met)."""
        return self.status == TaskStatus.PENDING and len(self.preconditions) == 0

    def add_child(self, child: "TaskNode"):
        """Add a child task."""
        child.parent_id = self.task_id
        self.children.append(child)
        if self.task_type == TaskType.ATOMIC:
            self.task_type = TaskType.COMPOSITE

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "task_id": self.task_id,
            "name": self.name,
            "description": self.description,
            "task_type": self.task_type.value,
            "status": self.status.value,
            "skill_id": self.skill_id,
            "preconditions": self.preconditions,
            "postconditions": self.postconditions,
            "children": [c.to_dict() for c in self.children],
            "parent_id": self.parent_id,
            "metadata": self.metadata,
            "affordances": self.affordances,
            "objects_involved": self.objects_involved,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TaskNode":
        """Create from dictionary."""
        children = [cls.from_dict(c) for c in d.get("children", [])]
        return cls(
            task_id=d["task_id"],
            name=d["name"],
            description=d.get("description", ""),
            task_type=TaskType(d.get("task_type", "atomic")),
            status=TaskStatus(d.get("status", "pending")),
            skill_id=d.get("skill_id"),
            preconditions=d.get("preconditions", []),
            postconditions=d.get("postconditions", []),
            children=children,
            parent_id=d.get("parent_id"),
            metadata=d.get("metadata", {}),
            affordances=d.get("affordances", []),
            objects_involved=d.get("objects_involved", []),
        )


@dataclass
class TaskGraph:
    """
    A directed acyclic graph (DAG) of tasks.

    Represents the hierarchical decomposition of a complex task into
    simpler sub-tasks with dependencies.
    """
    root: TaskNode
    graph_id: str = ""
    name: str = ""
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Ensure root has correct type."""
        if self.root.task_type != TaskType.ROOT:
            self.root.task_type = TaskType.ROOT

    def get_all_nodes(self) -> List[TaskNode]:
        """Get all nodes in the graph (breadth-first)."""
        nodes = []
        queue = [self.root]
        while queue:
            node = queue.pop(0)
            nodes.append(node)
            queue.extend(node.children)
        return nodes

    def get_leaf_nodes(self) -> List[TaskNode]:
        """Get all leaf nodes (atomic tasks)."""
        return [n for n in self.get_all_nodes() if n.is_leaf()]

    def get_skill_nodes(self) -> List[TaskNode]:
        """Get all nodes that map to skills."""
        return [n for n in self.get_all_nodes() if n.task_type == TaskType.SKILL]

    def get_node_by_id(self, task_id: str) -> Optional[TaskNode]:
        """Find a node by its task_id."""
        for node in self.get_all_nodes():
            if node.task_id == task_id:
                return node
        return None

    def get_ready_tasks(self) -> List[TaskNode]:
        """Get all tasks that are ready to execute."""
        ready = []
        for node in self.get_all_nodes():
            if node.status == TaskStatus.PENDING:
                # Check if all preconditions are completed
                all_met = True
                for pre_id in node.preconditions:
                    pre_node = self.get_node_by_id(pre_id)
                    if pre_node and pre_node.status != TaskStatus.COMPLETED:
                        all_met = False
                        break
                if all_met:
                    ready.append(node)
        return ready

    def mark_completed(self, task_id: str):
        """Mark a task as completed."""
        node = self.get_node_by_id(task_id)
        if node:
            node.status = TaskStatus.COMPLETED

    def mark_failed(self, task_id: str):
        """Mark a task as failed."""
        node = self.get_node_by_id(task_id)
        if node:
            node.status = TaskStatus.FAILED

    def get_execution_order(self) -> List[str]:
        """
        Get topological order of tasks respecting dependencies.

        Returns list of task_ids in execution order.
        """
        # Build dependency graph
        in_degree = {}
        graph = {}
        all_nodes = self.get_all_nodes()

        for node in all_nodes:
            in_degree[node.task_id] = len(node.preconditions)
            graph[node.task_id] = []

        # Build reverse edges (task -> dependent tasks)
        for node in all_nodes:
            for pre_id in node.preconditions:
                if pre_id in graph:
                    graph[pre_id].append(node.task_id)

        # Kahn's algorithm for topological sort
        queue = [n.task_id for n in all_nodes if in_degree[n.task_id] == 0]
        order = []

        while queue:
            task_id = queue.pop(0)
            order.append(task_id)

            for dependent_id in graph.get(task_id, []):
                in_degree[dependent_id] -= 1
                if in_degree[dependent_id] == 0:
                    queue.append(dependent_id)

        return order

    def get_critical_path(self) -> List[str]:
        """
        Get the critical path (longest path) through the task graph.

        Useful for estimating minimum completion time.
        """
        # Simple BFS to find longest path from root to any leaf
        all_nodes = self.get_all_nodes()
        if not all_nodes:
            return []

        # Build path lengths
        path_length = {self.root.task_id: 1}
        predecessor = {self.root.task_id: None}

        for node in all_nodes:
            if node.task_id not in path_length:
                path_length[node.task_id] = 1
            for child in node.children:
                new_length = path_length[node.task_id] + 1
                if child.task_id not in path_length or new_length > path_length[child.task_id]:
                    path_length[child.task_id] = new_length
                    predecessor[child.task_id] = node.task_id

        # Find the leaf with longest path
        leaves = self.get_leaf_nodes()
        if not leaves:
            return [self.root.task_id]

        max_leaf = max(leaves, key=lambda n: path_length.get(n.task_id, 0))

        # Reconstruct path
        path = []
        current = max_leaf.task_id
        while current is not None:
            path.append(current)
            current = predecessor.get(current)
        path.reverse()

        return path

    def summary(self) -> Dict[str, Any]:
        """Get summary statistics of the task graph."""
        all_nodes = self.get_all_nodes()
        return {
            "total_nodes": len(all_nodes),
            "leaf_nodes": len(self.get_leaf_nodes()),
            "skill_nodes": len(self.get_skill_nodes()),
            "depth": max(len(self.get_critical_path()), 1),
            "status_counts": {
                status.value: sum(1 for n in all_nodes if n.status == status)
                for status in TaskStatus
            },
            "ready_tasks": len(self.get_ready_tasks()),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "graph_id": self.graph_id,
            "name": self.name,
            "description": self.description,
            "root": self.root.to_dict(),
            "metadata": self.metadata,
            "summary": self.summary(),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TaskGraph":
        """Create from dictionary."""
        root = TaskNode.from_dict(d["root"])
        return cls(
            root=root,
            graph_id=d.get("graph_id", ""),
            name=d.get("name", ""),
            description=d.get("description", ""),
            metadata=d.get("metadata", {}),
        )

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "TaskGraph":
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))


def build_drawer_vase_task_graph() -> TaskGraph:
    """
    Build example task graph for drawer_vase environment.

    This is a canonical example showing task decomposition for the
    "open drawer without breaking vase" task.
    """
    # Root task
    root = TaskNode(
        task_id="open_drawer_safely",
        name="Open Drawer Safely",
        description="Open the drawer without knocking over or breaking the vase",
        task_type=TaskType.ROOT,
        postconditions=["drawer_open", "vase_intact", "vase_upright"],
        objects_involved=["drawer", "vase"],
    )

    # Sub-tasks
    approach = TaskNode(
        task_id="approach_drawer",
        name="Approach Drawer",
        description="Position gripper near drawer handle",
        task_type=TaskType.SKILL,
        skill_id=0,  # APPROACH skill
        postconditions=["gripper_near_handle"],
        affordances=["graspable"],
        objects_involved=["drawer"],
    )

    check_vase = TaskNode(
        task_id="check_vase_position",
        name="Check Vase Position",
        description="Verify vase location to avoid collision",
        task_type=TaskType.CHECKPOINT,
        preconditions=["approach_drawer"],
        postconditions=["vase_position_known"],
        objects_involved=["vase"],
    )

    grasp_handle = TaskNode(
        task_id="grasp_handle",
        name="Grasp Handle",
        description="Close gripper on drawer handle",
        task_type=TaskType.SKILL,
        skill_id=1,  # GRASP skill
        preconditions=["approach_drawer"],
        postconditions=["handle_grasped"],
        affordances=["graspable"],
        objects_involved=["drawer"],
    )

    pull_carefully = TaskNode(
        task_id="pull_drawer",
        name="Pull Drawer",
        description="Pull drawer open while avoiding vase",
        task_type=TaskType.SKILL,
        skill_id=2,  # PULL skill
        preconditions=["grasp_handle", "check_vase_position"],
        postconditions=["drawer_open"],
        affordances=["pullable"],
        objects_involved=["drawer"],
        metadata={"caution_level": "high", "fragile_nearby": True},
    )

    release_handle = TaskNode(
        task_id="release_handle",
        name="Release Handle",
        description="Open gripper to release handle",
        task_type=TaskType.SKILL,
        skill_id=3,  # RELEASE skill
        preconditions=["pull_drawer"],
        postconditions=["handle_released"],
        affordances=["releasable"],
        objects_involved=["drawer"],
    )

    verify_success = TaskNode(
        task_id="verify_success",
        name="Verify Success",
        description="Check that drawer is open and vase is intact",
        task_type=TaskType.CHECKPOINT,
        preconditions=["release_handle"],
        postconditions=["task_verified"],
        objects_involved=["drawer", "vase"],
    )

    # Build tree
    root.add_child(approach)
    root.add_child(check_vase)
    root.add_child(grasp_handle)
    root.add_child(pull_carefully)
    root.add_child(release_handle)
    root.add_child(verify_success)

    return TaskGraph(
        root=root,
        graph_id="drawer_vase_open_v1",
        name="Open Drawer Without Breaking Vase",
        description="Task graph for safely opening a drawer near a fragile vase",
        metadata={"env": "drawer_vase", "difficulty": "medium", "fragility": True},
    )


def build_grasp_place_task_graph() -> TaskGraph:
    """Build example task graph for grasp-and-place task."""
    root = TaskNode(
        task_id="grasp_place_object",
        name="Grasp and Place Object",
        description="Pick up object and place it at target location",
        task_type=TaskType.ROOT,
        postconditions=["object_at_target", "gripper_clear"],
        objects_involved=["object", "target_location"],
    )

    # Grasp phase
    approach_object = TaskNode(
        task_id="approach_object",
        name="Approach Object",
        task_type=TaskType.SKILL,
        skill_id=0,
        postconditions=["gripper_above_object"],
        objects_involved=["object"],
    )

    grasp_object = TaskNode(
        task_id="grasp_object",
        name="Grasp Object",
        task_type=TaskType.SKILL,
        skill_id=1,
        preconditions=["approach_object"],
        postconditions=["object_grasped"],
        objects_involved=["object"],
    )

    lift_object = TaskNode(
        task_id="lift_object",
        name="Lift Object",
        task_type=TaskType.SKILL,
        skill_id=4,  # LIFT skill
        preconditions=["grasp_object"],
        postconditions=["object_lifted"],
        objects_involved=["object"],
    )

    # Place phase
    move_to_target = TaskNode(
        task_id="move_to_target",
        name="Move to Target",
        task_type=TaskType.SKILL,
        skill_id=5,  # MOVE skill
        preconditions=["lift_object"],
        postconditions=["gripper_at_target"],
        objects_involved=["object", "target_location"],
    )

    place_object = TaskNode(
        task_id="place_object",
        name="Place Object",
        task_type=TaskType.SKILL,
        skill_id=6,  # PLACE skill
        preconditions=["move_to_target"],
        postconditions=["object_at_target"],
        objects_involved=["object", "target_location"],
    )

    release_object = TaskNode(
        task_id="release_object",
        name="Release Object",
        task_type=TaskType.SKILL,
        skill_id=3,
        preconditions=["place_object"],
        postconditions=["gripper_clear", "object_released"],
        objects_involved=["object"],
    )

    # Build tree
    root.add_child(approach_object)
    root.add_child(grasp_object)
    root.add_child(lift_object)
    root.add_child(move_to_target)
    root.add_child(place_object)
    root.add_child(release_object)

    return TaskGraph(
        root=root,
        graph_id="grasp_place_v1",
        name="Grasp and Place",
        description="Pick and place task graph",
        metadata={"env": "tabletop", "difficulty": "easy"},
    )
