#!/usr/bin/env python3
"""
Import Stage 1/2 datapacks and enrichments into the ontology store.
"""
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

from src.ontology.datapack_adapters import datapack_from_stage1, datapack_from_stage2_enrichment
from src.ontology.store import OntologyStore
from src.ontology.models import Datapack


def _load_stage1(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    if path.suffix == ".jsonl":
        records = []
        with path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records
    with path.open("r") as f:
        data = json.load(f)
        return data if isinstance(data, list) else []


def _load_enrichments(path: Path) -> List[Dict[str, Any]]:
    if not path or not path.exists():
        return []
    records = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def main():
    parser = argparse.ArgumentParser(description="Import Stage1/2 datapacks into ontology store")
    parser.add_argument("--stage1-datapacks-path", type=str, required=True)
    parser.add_argument("--stage2-enrichments-path", type=str, default="")
    parser.add_argument("--ontology-root", type=str, default="data/ontology")
    parser.add_argument("--task-id", type=str, required=True)
    args = parser.parse_args()

    stage1_path = Path(args.stage1_datapacks_path)
    stage2_path = Path(args.stage2_enrichments_path) if args.stage2_enrichments_path else None

    stage1_records = _load_stage1(stage1_path)
    stage2_records = _load_enrichments(stage2_path) if stage2_path else []

    datapacks: List[Datapack] = []
    for rec in stage1_records:
        datapacks.append(datapack_from_stage1(rec, task_id=args.task_id))
    for rec in stage2_records:
        datapacks.append(datapack_from_stage2_enrichment(rec, task_id=args.task_id))

    store = OntologyStore(root_dir=args.ontology_root)
    if datapacks:
        store.append_datapacks(datapacks)

    source_types = {d.source_type for d in datapacks}
    modalities = {d.modality for d in datapacks}
    print(
        f"[import_stage2_datapacks_to_ontology] Imported {len(datapacks)} datapacks "
        f"for task {args.task_id} | sources={sorted(source_types)} | modalities={sorted(modalities)}"
    )


if __name__ == "__main__":
    main()
