import json
from pathlib import Path

from src.world_model.economic_world_model import (
    run_gpu_run_hygiene,
    validate_gpu_run_manifest_payload,
)


def _valid_manifest() -> dict[str, object]:
    return {
        "run_id": "runpod-20260901-120000-abc123",
        "mode": "runpod",
        "pod_class": "provider",
        "run_class": "provider",
        "epistemic_status": "proof_of_life",
        "commit_sha": "abcdef1",
        "branch": "main",
        "task": "Provider proof-of-life smoke",
        "wm": "economic_world_model",
        "subsystem": "provider_runtime",
        "blocker": "provider_bringup_not_run",
        "config_paths": ["docs/agent_ergonomics/run_manifest_schema.md"],
        "seeds": [0],
        "image": "nvidia/cuda:12.1.0-runtime-ubuntu22.04",
        "template": "robotics-vp-core-gpu",
        "pod_id": None,
        "volume_id": None,
        "commands": [
            "python3 scripts/economic_world_model/check_evidence_hygiene.py "
            "--artifact-root artifacts/economic_world_model"
        ],
        "artifact_paths": [
            "artifacts/economic_world_model/provider_runs/"
            "runpod-20260901-120000-abc123/provider_truth.json"
        ],
        "status": "pending",
        "started_at": None,
        "finished_at": None,
        "cost_snapshot": {"estimated_cost_usd": 1.0},
        "gpu_class": "A10G",
        "wall_clock_seconds": None,
        "artifact_size_bytes": None,
        "storage_or_checkpoint_size_bytes": None,
        "expected_value": "Burns down provider runtime truth.",
        "estimated_cost_usd": 1.0,
        "dependency_chain": ["provider runbook template"],
        "urgency": "high",
        "justified_itself": None,
        "rollback_notes": "Discard provider receipts if smoke fails.",
        "replay_notes": "Run the same command on the recorded commit and image.",
    }


def _blocking_keys(payload: dict[str, object]) -> set[str]:
    return {
        receipt.check_key
        for receipt in validate_gpu_run_manifest_payload(payload)
        if not receipt.passed and receipt.severity == "blocking"
    }


def test_gpu_run_hygiene_accepts_guarded_provider_manifest() -> None:
    assert _blocking_keys(_valid_manifest()) == set()


def test_gpu_run_hygiene_blocks_missing_run_class() -> None:
    payload = _valid_manifest()
    payload.pop("run_class")

    blocked = _blocking_keys(payload)

    assert "required_fields_present" in blocked
    assert "run_class_valid" in blocked


def test_gpu_run_hygiene_blocks_generic_checkpoint_sink() -> None:
    payload = _valid_manifest()
    payload["artifact_paths"] = ["checkpoints/"]

    blocked = _blocking_keys(payload)

    assert "no_generic_checkpoint_sink" in blocked


def test_gpu_run_hygiene_blocks_inline_secret_command() -> None:
    payload = _valid_manifest()
    payload["commands"] = ["PROVIDER_API_KEY=abc python3 scripts/provider.py"]

    blocked = _blocking_keys(payload)

    assert "commands_do_not_inline_secrets" in blocked


def test_gpu_run_hygiene_writes_report(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(_valid_manifest()), encoding="utf-8")

    report = run_gpu_run_hygiene(
        manifest_paths=[manifest],
        output_dir=tmp_path / "out",
    )

    assert report["status"] == "ok_gpu_run_hygiene_passed"
    assert report["manifest_count"] == 1
    assert Path(report["output_paths"]["report_path"]).exists()
    assert Path(report["output_paths"]["receipts_path"]).exists()
