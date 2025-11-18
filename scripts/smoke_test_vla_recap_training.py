#!/usr/bin/env python3
"""
Smoke test for RECAP VLA offline training loop.
"""
import json
import math
import shutil
import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from scripts.train_vla_recap_offline import train_offline


def _build_synthetic_dataset(path: Path) -> None:
    """Build a tiny deterministic RECAP JSONL dataset."""
    path.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    for i in range(12):
        entry = {
            "task_id": "recap_smoke_task",
            "episode_id": f"ep_{i//4}",
            "timestep": i,
            "advantage": float((i % 3) - 1),
            "metrics": {
                "mpl": 50.0 + i,
                "energy_cost": 0.5 + 0.01 * i,
                "error_rate": 0.05 * ((i % 4)),
            },
            "sampler_strategy": "balanced" if i % 2 == 0 else "econ_urgency",
            "curriculum_phase": "early" if i < 6 else "mid",
            "objective_preset": "balanced",
        }
        entries.append(entry)
    with path.open("w") as f:
        for e in entries:
            f.write(json.dumps(e, sort_keys=True))
            f.write("\n")


def main():
    results_dir = Path("results/vla_recap/smoke")
    ckpt_dir = Path("checkpoints/vla_recap")
    dataset_path = results_dir / "synthetic_recap.jsonl"
    if results_dir.exists():
        shutil.rmtree(results_dir)
    _build_synthetic_dataset(dataset_path)
    if ckpt_dir.exists():
        for p in ckpt_dir.glob("recap_smoke*.pt"):
            p.unlink()

    run_out = train_offline(
        dataset_paths=[str(dataset_path)],
        output_dir=str(results_dir),
        checkpoint_dir=str(ckpt_dir),
        advantage_bins=[-1.5, 0.0, 1.5],
        metrics=["mpl", "energy_cost", "error_rate"],
        num_atoms=5,
        hidden_dim=16,
        batch_size=4,
        epochs=2,
        lr=1e-3,
        seed=123,
        log_csv=True,
        run_name="recap_smoke",
    )
    history = run_out["history"]
    assert len(history) == 2
    losses = [h["total_loss"] for h in history]
    assert all(math.isfinite(l) for l in losses)
    assert losses[-1] <= losses[0] * 1.1, "Loss did not improve sufficiently."
    ckpt_path = Path(run_out["checkpoint"])
    assert ckpt_path.exists(), "Checkpoint missing."
    csv_path = Path(run_out["csv"])
    assert csv_path.exists(), "CSV log missing."
    print("[smoke_test_vla_recap_training] All tests passed.")


if __name__ == "__main__":
    main()
