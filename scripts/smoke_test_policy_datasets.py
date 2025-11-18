#!/usr/bin/env python3
"""
Smoke test for policy dataset builder.
"""
import json
import shutil
import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from scripts.build_policy_datasets import build_datasets  # noqa: E402
from src.ontology.store import OntologyStore  # noqa: E402


def _load_jsonl(path: Path):
    records = []
    with path.open("r") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def main():
    out_dir = Path("results/policy_datasets_smoke")
    if out_dir.exists():
        shutil.rmtree(out_dir)

    store = OntologyStore(root_dir="data/ontology")
    build_datasets(store, out_dir)

    expected_files = [
        "data_valuation.jsonl",
        "pricing.jsonl",
        "safety_risk.jsonl",
        "energy_cost.jsonl",
        "episode_quality.jsonl",
        "sampler_weights.jsonl",
        "orchestrator.jsonl",
        "meta_advisor.jsonl",
        "vision_encoder.jsonl",
    ]
    contents_first = {}
    for fname in expected_files:
        path = out_dir / fname
        assert path.exists(), f"Missing dataset file {fname}"
        records = _load_jsonl(path)
        contents_first[fname] = records
        for rec in records:
            assert "policy" in rec and "features" in rec and "target" in rec, f"Schema mismatch in {fname}"
    # Determinism: rebuild and compare
    build_datasets(store, out_dir)
    for fname in expected_files:
        path = out_dir / fname
        records = _load_jsonl(path)
        assert records == contents_first.get(fname, []), f"Nondeterministic output detected in {fname}"

    print("[smoke_test_policy_datasets] All tests passed.")


if __name__ == "__main__":
    main()
