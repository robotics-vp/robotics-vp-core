#!/usr/bin/env python3
"""
Print a snapshot of the current semantic orchestrator state.

Read-only: no mutations to Phase B, rewards, or sampling.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.orchestrator.semantic_orchestrator import SemanticOrchestrator
from src.orchestrator.task_graph import TaskGraph, TaskNode
from src.orchestrator.ontology import EnvironmentOntology


def main():
    # Minimal dummy graph/ontology for snapshotting
    task_graph = TaskGraph(root=TaskNode("root", "root", "root task"))
    ontology = EnvironmentOntology(ontology_id="ont1", name="default")
    sem = SemanticOrchestrator(
        econ_controller=None,
        datapack_engine=None,
        task_graph=task_graph,
        ontology=ontology,
    )

    snapshot = sem.snapshot()
    consistency = sem.semantic_consistency_checks()
    print("Semantic snapshot:", snapshot)
    print("Consistency:", consistency)
    if hasattr(sem, "_current_constraints"):
        print("Current constraints:", getattr(sem, "_current_constraints"))


if __name__ == "__main__":
    main()
