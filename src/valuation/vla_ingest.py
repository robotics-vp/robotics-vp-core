import json
import os
from typing import Dict

from src.valuation.datapack_schema import DataPackMeta
from src.valuation.datapack_repo import DataPackRepo


def attach_vla_action_to_datapack(datapack: DataPackMeta, vla_json: Dict[str, any]) -> DataPackMeta:
    datapack.vla_action_summary = {
        "has_vla": bool(vla_json.get("action", {}).get("vla_available", False)),
        "instruction": vla_json.get("instruction"),
        "action_7dof": vla_json.get("action", {}).get("raw_action", []),
        "source": "openvla-7b",
        "semantic_tags": vla_json.get("semantic_tags", []),
        "vla_hint_text": vla_json.get("vla_hint_text", ""),
    }

    # Also merge VLA semantic tags into datapack's main semantic_tags
    vla_tags = vla_json.get("semantic_tags", [])
    if vla_tags:
        existing_tags = datapack.semantic_tags or []
        merged_tags = list(set(existing_tags + vla_tags))
        datapack.semantic_tags = merged_tags

    return datapack


def ingest_vla_actions_for_repo(repo: DataPackRepo, vla_results_path: str, output_overlay: str):
    if not os.path.exists(vla_results_path):
        return
    overlays = []
    with open(vla_results_path, "r") as f:
        data = [json.loads(line) for line in f if line.strip()]
    # Build map for quick lookup
    id_to_dp = {}
    for task in os.listdir(repo.base_dir):
        if not task.endswith(".jsonl"):
            continue
        task_name = task.replace("_datapacks.jsonl", "")
        for dp in repo.iter_all(task_name) or []:
            id_to_dp[dp.episode_id] = dp
    for rec in data:
        epi = rec.get("episode_id")
        if epi in id_to_dp:
            dp = attach_vla_action_to_datapack(id_to_dp[epi], rec)
            overlays.append({"pack_id": dp.pack_id, "vla_action_summary": dp.vla_action_summary})
    if overlays:
        os.makedirs(os.path.dirname(output_overlay), exist_ok=True)
        with open(output_overlay, "w") as f:
            for o in overlays:
                f.write(json.dumps(o) + "\n")
