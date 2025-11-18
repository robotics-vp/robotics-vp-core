#!/usr/bin/env python3
"""
Smoke test for importing SIMA-2 semantics into ontology.
"""
import json
import tempfile
from pathlib import Path
import sys

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from scripts.run_sima2_rollouts import main as rollouts_main
from scripts.run_stage2_sima2_pipeline import main as pipeline_main
from scripts.import_sima2_semantics_to_ontology import main as import_main
from src.ontology.store import OntologyStore


def main():
    with tempfile.TemporaryDirectory() as d:
        tmpdir = Path(d)
        rollouts_path = tmpdir / "rollouts.jsonl"
        sys.argv = ["run_sima2_rollouts.py", "--task-id", "drawer_vase", "--num-episodes", "1", "--output-jsonl", str(rollouts_path)]
        rollouts_main()

        out_dir = tmpdir / "stage2"
        sys.argv = [
            "run_stage2_sima2_pipeline.py",
            "--rollouts-path",
            str(rollouts_path),
            "--ontology-root",
            str(tmpdir / "ontology"),
            "--output-dir",
            str(out_dir),
        ]
        pipeline_main()

        tags_path = out_dir / "sima2_semantic_tags.jsonl"
        sys.argv = [
            "import_sima2_semantics_to_ontology.py",
            "--ontology-root",
            str(tmpdir / "ontology"),
            "--task-id",
            "drawer_vase",
            "--rollouts-path",
            str(rollouts_path),
            "--tags-path",
            str(tags_path),
        ]
        import_main()

        store = OntologyStore(root_dir=str(tmpdir / "ontology"))
        datapacks = store.list_datapacks(task_id="drawer_vase")
        assert datapacks, "Expected datapacks imported"
        print("[smoke_test_sima2_ontology_integration] PASS")


if __name__ == "__main__":
    main()
