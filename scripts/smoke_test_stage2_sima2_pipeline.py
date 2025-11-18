#!/usr/bin/env python3
"""
Smoke test for Stage2 SIMA-2 pipeline.
"""
import json
import tempfile
from pathlib import Path
import sys

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from src.sima2.client import Sima2Client
from scripts.run_stage2_sima2_pipeline import main as pipeline_main


def _write_rollouts(tmpdir: Path) -> Path:
    client = Sima2Client(task_id="drawer_vase")
    path = tmpdir / "rollouts.jsonl"
    with path.open("w") as f:
        for idx in range(2):
            rollout = client.run_episode({"task_id": "drawer_vase", "episode_index": idx})
            f.write(json.dumps(rollout, sort_keys=True))
            f.write("\n")
    return path


def main():
    with tempfile.TemporaryDirectory() as d:
        tmpdir = Path(d)
        rollouts_path = _write_rollouts(tmpdir)
        out_dir = tmpdir / "out"
        args = [
            "--rollouts-path",
            str(rollouts_path),
            "--ontology-root",
            str(tmpdir / "ontology"),
            "--output-dir",
            str(out_dir),
        ]
        sys.argv = ["run_stage2_sima2_pipeline.py"] + args
        pipeline_main()

        primitives = json.loads((out_dir / "sima2_primitives.jsonl").read_text().splitlines()[0])
        proposals = json.loads((out_dir / "sima2_ontology_proposals.jsonl").read_text().splitlines()[0])
        refinements = json.loads((out_dir / "sima2_task_refinements.jsonl").read_text().splitlines()[0])
        tags = json.loads((out_dir / "sima2_semantic_tags.jsonl").read_text().splitlines()[0])
        assert primitives
        assert proposals
        assert refinements
        assert tags
        print("[smoke_test_stage2_sima2_pipeline] PASS")


if __name__ == "__main__":
    main()
