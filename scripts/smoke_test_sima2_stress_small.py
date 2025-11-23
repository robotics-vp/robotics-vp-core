#!/usr/bin/env python3
"""
Small-scale smoke for SIMA-2 stress test (100 rollouts, batched).
Ensures artifacts/metrics are produced and guardrails hold.
"""
import json
import subprocess
from pathlib import Path
import sys

repo_root = Path(__file__).parent.parent


def main():
    out_dir = repo_root / "results" / "sima2_stress_smoke"
    if out_dir.exists():
        import shutil

        shutil.rmtree(out_dir)
    cmd = [
        "python3",
        str(repo_root / "scripts" / "stress_test_sima2_pipeline.py"),
        "--num-rollouts",
        "100",
        "--batch-size",
        "20",
        "--output-dir",
        str(out_dir),
        "--ontology-root",
        str(out_dir / "ontology_store"),
    ]
    subprocess.run(cmd, check=True)

    metrics_path = out_dir / "metrics.json"
    assert metrics_path.exists(), "metrics.json missing"
    with metrics_path.open() as f:
        metrics = json.load(f)

    assert metrics.get("rollouts") == 100, "Unexpected rollout count"
    assert metrics.get("primitives", 0) > 0, "Expected primitives"
    assert (out_dir / "primitives.jsonl").exists(), "primitives.jsonl missing"
    assert (out_dir / "ontology_proposals.jsonl").exists(), "ontology_proposals.jsonl missing"
    assert (out_dir / "semantic_tags.jsonl").exists(), "semantic_tags.jsonl missing"

    print("[smoke_test_sima2_stress_small] PASS")


if __name__ == "__main__":
    sys.exit(main())
