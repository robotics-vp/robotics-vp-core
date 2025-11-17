#!/usr/bin/env python3
"""
Handshakes between meta-transformer outputs and downstream facades.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.orchestrator.meta_transformer import MetaTransformer, MetaTransformerOutputs
from src.orchestrator.economic_controller import EconomicController
from src.orchestrator.datapack_engine import DatapackEngine
from src.orchestrator.context import OrchestratorContext
from src.valuation.datapack_repo import DataPackRepo


def build_ctx():
    return OrchestratorContext(
        env_name="drawer_vase",
        engine_type="pybullet",
        task_type="fragility",
        customer_segment="balanced",
        market_region="US",
        objective_vector=[1.0, 1.0, 1.0, 1.0, 0.0],
        wage_human=18.0,
        energy_price_kWh=0.12,
        mean_delta_mpl=0.0,
        mean_delta_error=0.0,
        mean_delta_j=0.0,
        mean_trust=0.0,
        mean_w_econ=0.0,
        profile_summaries={},
    )


def meta_to_econ_handshake():
    econ = EconomicController.from_econ_params(econ_params=None)
    meta = MetaTransformer()
    ctx = build_ctx()
    meta_out = meta.forward(np.zeros(2), np.zeros(2))
    meta_out.objective_preset = "balanced"
    meta_out.chosen_backend = "pybullet"
    econ.apply_meta_update(meta_out)
    derived = meta.derive_expected_objectives(meta_out, ctx)
    delta = meta.predict_expected_delta(meta_out)
    print("Econ handshake derived objective:", derived)
    print("Econ handshake deltas:", delta)


def meta_to_orchestrator_handshake():
    meta = MetaTransformer()
    ctx = build_ctx()
    meta_out = meta.forward(np.zeros(2), np.zeros(2))
    obj = meta.derive_expected_objectives(meta_out, ctx)
    assert len(obj) >= 4
    print("Orchestrator handshake objective vector length:", len(obj))


def meta_to_datapack_handshake():
    # Load minimal datapack repo if exists
    repo = DataPackRepo(base_dir="data/datapacks")
    first_dp = None
    for env in ["drawer_vase", "dishwashing"]:
        it = repo.iter_all(env)
        if it:
            dps = list(it)
            if dps:
                first_dp = dps[0]
                break
    if first_dp:
        # Attach meta deltas in a metadata-friendly field
        first_dp.agent_profile["meta_expected_delta"] = {
            "mpl": 0.0,
            "error": 0.0,
            "energy_Wh": 0.0,
        }
        print("Datapack handshake: attached meta deltas to agent_profile.")
    else:
        print("Datapack handshake: no datapacks found (ok for smoke).")


def main():
    meta_to_econ_handshake()
    meta_to_orchestrator_handshake()
    meta_to_datapack_handshake()


if __name__ == "__main__":
    main()
