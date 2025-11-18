#!/usr/bin/env python3
"""
Smoke test for semantic ↔ econ correlation reporting.
"""
import json
import shutil
import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from scripts.report_semantic_econ_correlations import run_report
from src.ontology.store import OntologyStore
from src.ontology.models import Task, Episode, EconVector


def _setup_ontology(root: Path):
    if root.exists():
        shutil.rmtree(root)
    store = OntologyStore(root_dir=str(root))
    task_id = "semantic_econ_smoke"
    store.upsert_task(
        Task(
            task_id=task_id,
            name="SemanticEconSmoke",
            environment_id="env",
            human_mpl_units_per_hour=40.0,
            human_wage_per_hour=18.0,
            default_energy_cost_per_wh=0.1,
        )
    )
    episodes = [
        Episode(episode_id="ep_a", task_id=task_id, robot_id="r1", status="success"),
        Episode(episode_id="ep_b", task_id=task_id, robot_id="r1", status="success"),
        Episode(episode_id="ep_c", task_id=task_id, robot_id="r1", status="failure"),
    ]
    for ep in episodes:
        store.upsert_episode(ep)
    econ_vectors = [
        EconVector(episode_id="ep_a", mpl_units_per_hour=80.0, wage_parity=0.9, energy_cost=0.4, damage_cost=0.05, novelty_delta=0.1, reward_scalar_sum=5.0, components={"error_rate": 0.01}),
        EconVector(episode_id="ep_b", mpl_units_per_hour=60.0, wage_parity=1.1, energy_cost=0.5, damage_cost=0.02, novelty_delta=0.2, reward_scalar_sum=4.0, components={"error_rate": 0.02}),
        EconVector(episode_id="ep_c", mpl_units_per_hour=55.0, wage_parity=1.0, energy_cost=0.6, damage_cost=0.1, novelty_delta=0.3, reward_scalar_sum=3.0, components={"error_rate": 0.08}),
    ]
    for ev in econ_vectors:
        store.upsert_econ_vector(ev)
    return task_id


def _write_semantic_tags(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    enrichments = [
        {
            "episode_id": "ep_a",
            "enrichment": {
                "fragility_tags": [{"object_name": "glass", "fragility_level": "high"}],
                "risk_tags": [{"risk_type": "collision", "severity": "medium"}],
                "novelty_tags": [{"novelty_score": 0.8}],
                "supervision_hints": {"priority_level": "high"},
            },
        },
        {
            "episode_id": "ep_b",
            "enrichment": {
                "fragility_tags": [],
                "risk_tags": [{"risk_type": "tip_over", "severity": "high"}],
                "novelty_tags": [{"novelty_score": 0.4}],
                "supervision_hints": {"priority_level": "medium"},
            },
        },
    ]
    with path.open("w") as f:
        for rec in enrichments:
            f.write(json.dumps(rec, sort_keys=True))
            f.write("\n")


def _write_advisories(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    advisories = [
        {"task_id": "semantic_econ_smoke", "safety_emphasis": 0.8, "metadata": {"frontier_eps": ["ep_a", "ep_b"]}},
        {"task_id": "semantic_econ_smoke", "safety_emphasis": 0.4, "metadata": {"frontier_eps": ["ep_c"]}},
    ]
    with path.open("w") as f:
        for adv in advisories:
            f.write(json.dumps(adv, sort_keys=True))
            f.write("\n")


def main():
    ontology_root = Path("data/ontology_semantic_econ_smoke")
    tags_path = Path("results/semantic_econ/smoke_tags.jsonl")
    advisories_path = Path("results/semantic_econ/smoke_advisories.jsonl")
    output_dir = Path("results/semantic_econ/smoke_report")

    task_id = _setup_ontology(ontology_root)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    _write_semantic_tags(tags_path)
    _write_advisories(advisories_path)

    summary = run_report(
        ontology_root=str(ontology_root),
        task_id=task_id,
        semantic_tags_path=str(tags_path),
        advisories_path=str(advisories_path),
        output_dir=str(output_dir),
    )
    json_path = output_dir / "semantic_econ_correlations.json"
    csv_path = output_dir / "semantic_econ_correlations.csv"
    assert json_path.exists() and csv_path.exists(), "Summary files missing."
    loaded = json.loads(json_path.read_text())
    assert loaded.get("tag_summaries", {}).get("high_risk", {}).get("count", 0) > 0
    # Determinism: rerun and compare outputs
    summary_2 = run_report(
        ontology_root=str(ontology_root),
        task_id=task_id,
        semantic_tags_path=str(tags_path),
        advisories_path=str(advisories_path),
        output_dir=str(output_dir),
    )
    assert summary == summary_2
    print("[smoke_test_semantic_econ_correlations] All tests passed.")


if __name__ == "__main__":
    main()
