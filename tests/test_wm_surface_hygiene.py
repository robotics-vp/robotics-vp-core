import json
from pathlib import Path

from src.world_model.economic_world_model import run_wm_surface_hygiene


def _write_manifest(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "run_id": "runpod-20260901-120000-abc123",
                "mode": "runpod",
                "pod_class": "provider",
                "run_class": "provider",
                "epistemic_status": "proof_of_life",
                "commit_sha": "abcdef1",
                "branch": "main",
                "task": "Provider smoke",
                "config_paths": ["docs/agent_ergonomics/run_manifest_schema.md"],
                "seeds": [0],
                "image": "nvidia/cuda:12.1.0-runtime-ubuntu22.04",
                "template": "gpu",
                "pod_id": None,
                "commands": ["python3 smoke.py"],
                "artifact_paths": ["artifacts/economic_world_model/provider_runs/x/y.json"],
                "status": "pending",
                "started_at": None,
                "finished_at": None,
                "cost_snapshot": None,
                "dependency_chain": ["provider package"],
                "rollback_notes": "discard smoke receipts",
                "replay_notes": "rerun same command",
            }
        ),
        encoding="utf-8",
    )


def _write_required_paths(root: Path) -> None:
    for path in [
        "docs/agent_ergonomics/run_manifest_schema.md",
        "src/world_model/economic_world_model/evidence_hygiene.py",
        "src/world_model/economic_world_model/gpu_run_hygiene.py",
        "scripts/economic_world_model/check_evidence_hygiene.py",
        "scripts/economic_world_model/check_gpu_run_hygiene.py",
        ".github/workflows/economic-world-model-focused.yml",
    ]:
        full = root / path
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text("ok\n", encoding="utf-8")


def _write_target_roots(root: Path) -> None:
    for path in [
        "src/world_model/example.py",
        "src/runtime/example.py",
        "src/evidence/example.py",
        "src/embodiment/example.py",
        "scripts/economic_world_model/example.py",
        "docs/economic_world_model/example.md",
        ".github/workflows/example.yml",
    ]:
        full = root / path
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text("ok\n", encoding="utf-8")
    _write_manifest(root / "configs/runpod/examples/provider.json")


def test_wm_surface_hygiene_passes_clean_surface(tmp_path: Path) -> None:
    _write_target_roots(tmp_path)
    _write_required_paths(tmp_path)

    report = run_wm_surface_hygiene(
        repo_root=tmp_path,
        output_dir=tmp_path / "out",
        changed_paths=[],
    )

    assert report["status"] == "ok_wm_surface_hygiene_passed"
    assert report["blocking_issue_count"] == 0
    assert Path(report["output_paths"]["report_path"]).exists()


def test_wm_surface_hygiene_blocks_risky_true_claim(tmp_path: Path) -> None:
    _write_target_roots(tmp_path)
    _write_required_paths(tmp_path)
    (tmp_path / "docs/economic_world_model/example.md").write_text(
        "gpu_training_executed: true\n",
        encoding="utf-8",
    )

    report = run_wm_surface_hygiene(
        repo_root=tmp_path,
        output_dir=tmp_path / "out",
        changed_paths=[],
    )

    assert report["status"] == "blocked_wm_surface_hygiene_failed"
    assert report["risky_true_claim_count"] == 1


def test_wm_surface_hygiene_blocks_protected_changes(tmp_path: Path) -> None:
    _write_target_roots(tmp_path)
    _write_required_paths(tmp_path)

    report = run_wm_surface_hygiene(
        repo_root=tmp_path,
        output_dir=tmp_path / "out",
        changed_paths=["checkpoints/stable_world_model.pt"],
    )

    assert report["status"] == "blocked_wm_surface_hygiene_failed"
    assert report["protected_change_count"] == 1
