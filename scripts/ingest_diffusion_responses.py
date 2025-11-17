#!/usr/bin/env python3
"""
Stub ingestion of diffusion responses into datapacks.
"""
import argparse
import json
import os

from src.orchestrator.diffusion_requests import DiffusionPromptSpec
from src.valuation.diffusion_ingest import build_episode_stub_from_diffusion_request, attach_rollout_metrics_to_diffusion_stub
from src.valuation.datapacks import build_datapack_from_episode
from src.config.econ_params import EconParams
from src.valuation.datapack_repo import DataPackRepo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=str, default="data/diffusion_requests.jsonl")
    parser.add_argument("--responses", type=str, default="data/diffusion_responses.jsonl")
    parser.add_argument("--out-base", type=str, default="data/datapacks")
    args = parser.parse_args()

    if not os.path.exists(args.responses):
        print("No responses file; nothing to ingest.")
        return

    repo = DataPackRepo(base_dir=args.out_base)
    datapacks = []
    with open(args.responses, "r") as f:
        for line in f:
            if not line.strip():
                continue
            resp = json.loads(line)
            req = DiffusionPromptSpec(**resp.get("request", {}))
            # Build stub then fill metrics from response if present
            stub = build_episode_stub_from_diffusion_request(req)
            metrics = resp.get("metrics", {})
            stub = attach_rollout_metrics_to_diffusion_stub(
                stub,
                mpl=metrics.get("mpl", 0.0),
                error_rate=metrics.get("error_rate", 0.0),
                energy_wh=metrics.get("energy_Wh", 0.0),
            )
            econ = EconParams(
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
                preset="diffusion_stub",
            )
            dp = build_datapack_from_episode(
                episode_info=stub,
                econ_params=econ,
                condition_profile={"env": req.env_name, "engine_type": req.engine_type, "task_type": req.task_type},
                agent_profile={"source": "diffusion_stub"},
                brick_id=None,
                env_type=req.env_name,
                extra_tags=req.semantic_tags,
                semantic_energy_drivers=req.semantic_tags,
            )
            datapacks.append(dp)

    # Write to repo
    repo.append_batch(datapacks)
    print(f"Ingested {len(datapacks)} diffusion responses into {args.out_base}")


if __name__ == "__main__":
    main()
