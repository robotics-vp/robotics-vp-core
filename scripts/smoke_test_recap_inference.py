#!/usr/bin/env python3
"""
Smoke test for RECAP inference and scoring script.
"""
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from src.ontology.store import OntologyStore
from src.ontology.models import Task, Robot, Episode, EconVector, EpisodeEvent
from scripts.train_vla_recap_offline import train_offline
from scripts.score_episodes_with_recap import main as score_main


def _build_ontology(root: Path) -> None:
    store = OntologyStore(root_dir=str(root))
    task_id = "recap_infer_task"
    store.upsert_task(Task(task_id=task_id, name="RecapInferTask", environment_id="env", human_mpl_units_per_hour=60.0, human_wage_per_hour=18.0, default_energy_cost_per_wh=0.1))
    store.upsert_robot(Robot(robot_id="recap_robot", name="RecapBot"))
    now = datetime.utcnow()
    eps = []
    for idx in range(2):
        ep_id = f"ep_inf_{idx}"
        ep = Episode(episode_id=ep_id, task_id=task_id, robot_id="recap_robot", started_at=now, status="success", metadata={"objective_preset": "balanced"})
        store.upsert_episode(ep)
        store.upsert_econ_vector(EconVector(episode_id=ep_id, mpl_units_per_hour=70 + 5 * idx, wage_parity=1.0, energy_cost=0.4 + 0.05 * idx, damage_cost=0.02 * idx, novelty_delta=0.1, reward_scalar_sum=5.0 + idx))
        events = [
            EpisodeEvent(episode_id=ep_id, timestep=0, event_type="step", timestamp=now, reward_scalar=1.0 + idx, reward_components={"mpl": 1.0 + idx, "energy_cost": 0.5, "error_rate": 0.01 * idx}),
            EpisodeEvent(episode_id=ep_id, timestep=1, event_type="step", timestamp=now, reward_scalar=2.0 + idx, reward_components={"mpl": 2.0 + idx, "energy_cost": 0.6, "error_rate": 0.02 * idx}),
        ]
        store.append_events(events)


def main():
    ontology_root = Path("data/ontology/recap_inference_smoke")
    if ontology_root.exists():
        shutil.rmtree(ontology_root)
    _build_ontology(ontology_root)

    dataset_path = Path("results/recap/recap_inference_smoke_dataset.jsonl")
    ckpt_dir = Path("checkpoints/vla_recap")
    for p in [dataset_path, ckpt_dir / "recap_infer_smoke_seed0.pt"]:
        if p.exists():
            p.unlink()

    # Build tiny dataset directly from ontology events
    store = OntologyStore(root_dir=str(ontology_root))
    entries = []
    econ_map = {e.episode_id: e for e in store.list_econ_vectors()}
    for ep in store.list_episodes():
        events = store.get_events(ep.episode_id)
        reward_mean = sum(e.reward_scalar for e in events) / len(events)
        for ev in events:
            entries.append(
                {
                    "task_id": ep.task_id,
                    "episode_id": ep.episode_id,
                    "timestep": ev.timestep,
                    "advantage": float(ev.reward_scalar - reward_mean),
                    "metrics": ev.reward_components,
                    "sampler_strategy": "balanced",
                    "curriculum_phase": "early",
                    "objective_preset": ep.metadata.get("objective_preset"),
                }
            )
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    with dataset_path.open("w") as f:
        for e in entries:
            f.write(json.dumps(e, sort_keys=True))
            f.write("\n")

    run_out = train_offline(
        dataset_paths=[str(dataset_path)],
        output_dir="results/vla_recap/infer_smoke",
        checkpoint_dir=str(ckpt_dir),
        advantage_bins=[-1.0, 0.0, 1.0],
        metrics=["mpl", "energy_cost", "error_rate"],
        num_atoms=5,
        hidden_dim=8,
        batch_size=2,
        epochs=1,
        lr=1e-3,
        seed=0,
        log_csv=False,
        run_name="recap_infer_smoke",
    )
    ckpt_path = run_out["checkpoint"]
    assert Path(ckpt_path).exists()

    sys.argv = ["score_episodes_with_recap.py", "--ontology-root", str(ontology_root), "--checkpoint", ckpt_path, "--output-jsonl", "results/recap/episode_scores_smoke.jsonl"]
    score_main()
    out_path = Path("results/recap/episode_scores_smoke.jsonl")
    assert out_path.exists()
    data = out_path.read_text().strip().splitlines()
    assert data
    parsed = [json.loads(l) for l in data]
    assert len(parsed) == len(store.list_episodes())
    # Determinism: re-run and compare output
    score_main()
    assert out_path.read_text() == "\n".join(data) + ("\n" if data else "")
    print("[smoke_test_recap_inference] All tests passed.")


if __name__ == "__main__":
    main()
