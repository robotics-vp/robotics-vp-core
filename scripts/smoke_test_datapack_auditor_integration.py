#!/usr/bin/env python3
"""
Smoke test for DatapackAuditor integration with ontology + reports.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from src.ontology.models import Task, Robot, Datapack, Episode, EconVector  # noqa: E402
from src.ontology.store import OntologyStore  # noqa: E402
from src.policies.datapack_auditor import HeuristicDatapackAuditor  # noqa: E402


def _make_datapack(auditor: HeuristicDatapackAuditor, datapack_id: str, task_id: str, novelty: float, risk_tag: str):
    semantic_tags = [{"novelty_type": "edge_case", "novelty_score": novelty}]
    if risk_tag:
        semantic_tags.append({"risk_type": risk_tag, "severity": "critical" if risk_tag == "collision" else "medium"})
    features = auditor.build_features(
        datapack={"datapack_id": datapack_id, "task_id": task_id, "novelty_score": novelty},
        semantic_tags=semantic_tags,
        econ_slice={"expected_mpl_gain": 5.0 * novelty, "novelty_score": novelty},
    )
    audit = auditor.evaluate(features)
    return Datapack(
        datapack_id=datapack_id,
        source_type="synthetic_video",
        task_id=task_id,
        modality="video",
        storage_uri=f"/tmp/{datapack_id}",
        novelty_score=novelty,
        quality_score=0.5 + 0.1 * novelty,
        tags={"semantic_tags": semantic_tags},
        auditor_rating=audit.get("rating"),
        auditor_score=audit.get("score"),
        auditor_predicted_econ=audit.get("predicted_econ"),
    ), audit


def main():
    root = Path("tmp/datapack_auditor_smoke")
    if root.exists():
        shutil.rmtree(root)
    store = OntologyStore(root_dir=str(root))
    task = Task(task_id="task_auditor_smoke", name="Auditor Smoke Task")
    robot = Robot(robot_id="robot_auditor_smoke", name="AuditBot")
    store.upsert_task(task)
    store.upsert_robot(robot)

    auditor = HeuristicDatapackAuditor()
    datapacks = []
    audits = {}
    for idx, (novelty, risk_tag) in enumerate([(0.9, ""), (0.1, "collision")]):
        dp, audit = _make_datapack(auditor, f"dp_audit_{idx}", task.task_id, novelty, risk_tag)
        datapacks.append(dp)
        audits[dp.datapack_id] = audit.get("predicted_econ")
    store.append_datapacks(datapacks)

    for dp in datapacks:
        ep = Episode(episode_id=f"ep_{dp.datapack_id}", task_id=task.task_id, robot_id=robot.robot_id, datapack_id=dp.datapack_id, status="success")
        store.upsert_episode(ep)
        econ = EconVector(
            episode_id=ep.episode_id,
            mpl_units_per_hour=10.0 * (1.0 + dp.novelty_score),
            wage_parity=1.0,
            energy_cost=2.0 * (1.0 + dp.novelty_score),
            damage_cost=5.0 * (1.0 + (1.0 if dp.auditor_rating == "JUNK" else 0.0)),
            novelty_delta=dp.novelty_score,
            reward_scalar_sum=100.0,
            components={"energy_penalty": 1.0},
            source_domain="pybullet",
        )
        store.upsert_econ_vector(econ)

    # Round-trip check
    reloaded = {dp.datapack_id: dp for dp in store.list_datapacks(task_id=task.task_id)}
    assert reloaded, "Datapacks failed to round-trip through store"
    for dp_id, dp in reloaded.items():
        assert dp.auditor_rating, f"Missing auditor rating for {dp_id}"
        assert dp.auditor_predicted_econ == audits.get(dp_id), "Predicted econ did not survive serialization"

    cmd = [
        sys.executable,
        "scripts/report_task_pricing_and_performance.py",
        "--ontology-root",
        str(root),
        "--task-id",
        task.task_id,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=str(repo_root))
    assert proc.stdout, "Report did not produce output"
    print(proc.stdout.strip().splitlines()[:3])
    print("[smoke_test_datapack_auditor_integration] Passed.")


if __name__ == "__main__":
    main()
