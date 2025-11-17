#!/usr/bin/env python3
"""
Tiny end-to-end meta-layer pass.

Loads a few datapacks (if available), builds econ/datapack signals,
runs SemanticOrchestrator.build_update_plan, and feeds a MetaTransformer once.

Read-only; no training or reward changes.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.config.econ_params import EconParams
from src.orchestrator.datapack_engine import DatapackEngine
from src.orchestrator.economic_controller import EconomicController
from src.orchestrator.meta_transformer import MetaTransformer
from src.orchestrator.semantic_orchestrator import SemanticOrchestrator
from src.orchestrator.task_graph import TaskGraph, TaskNode
from src.orchestrator.ontology import EnvironmentOntology
from src.valuation.datapack_repo import DataPackRepo


def main():
    repo = DataPackRepo(base_dir="data/datapacks")
    datapacks = [dp for dp in repo.iter_all("drawer_vase") or []][:5]

    econ_params = EconParams(
        price_per_unit=0.3,
        damage_cost=1.0,
        energy_Wh_per_attempt=0.05,
        time_step_s=60.0,
        base_rate=2.0,
        p_min=0.02,
        k_err=0.12,
        q_speed=1.2,
        q_care=1.5,
        care_cost=0.25,
        max_steps=240,
        max_catastrophic_errors=3,
        max_error_rate_sla=0.12,
        min_steps_for_sla=5,
        zero_throughput_patience=10,
        preset="toy",
    )
    econ = EconomicController.from_econ_params(econ_params)
    datapack_engine = DatapackEngine(repo)

    econ_signals = econ.compute_signals(datapacks)
    datapack_signals = datapack_engine.compute_signals(datapacks, econ_signals)

    sem = SemanticOrchestrator(
        econ_controller=econ,
        datapack_engine=datapack_engine,
        task_graph=TaskGraph(root=TaskNode("root", "root", "root task")),
        ontology=EnvironmentOntology(ontology_id="ont1", name="default"),
    )

    meta = MetaTransformer()
    meta_out = meta.forward(
        dino_features=np.zeros(8),
        vla_features=np.zeros(8),
        dino_conf=0.5,
        vla_conf=0.5,
    )

    plan = sem.build_update_plan(
        econ_signals=econ_signals,
        datapack_signals=datapack_signals,
        meta_out=meta_out,
    )

    print("Econ signals:", econ_signals.to_dict())
    print("Datapack signals:", datapack_signals.to_dict())
    print("Meta output authority:", meta_out.authority)
    print("Semantic plan rationale:", plan.rationale)


if __name__ == "__main__":
    main()
