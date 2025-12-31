from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class RolloutCaptureConfig:
    output_dir: str | Path
    max_artifacts: int | None = None
    include_patterns: Sequence[str] = ("*.mp4", "*.npz", "*.json", "*.jsonl")


@dataclass(frozen=True)
class RolloutCaptureResult:
    output_dir: str
    manifest_path: str
    artifacts: Sequence[str]
    status: str


def capture_rollouts(
    *,
    policy_id: str,
    scenario_id: str,
    task_id: str,
    datapack_ids: Sequence[str],
    config: RolloutCaptureConfig,
) -> RolloutCaptureResult:
    output_root = Path(config.output_dir)
    output_dir = output_root / scenario_id
    output_dir.mkdir(parents=True, exist_ok=True)

    run_dir = _resolve_run_dir(policy_id)
    artifacts: list[str] = []
    status = "pending_capture"
    if run_dir and run_dir.exists():
        artifacts = _collect_artifacts(run_dir, config)
        status = "captured" if artifacts else "no_artifacts_found"

    manifest = {
        "scenario_id": scenario_id,
        "task_id": task_id,
        "policy_id": policy_id,
        "datapack_ids": list(datapack_ids),
        "run_dir": str(run_dir) if run_dir else None,
        "status": status,
        "artifacts": artifacts,
    }
    manifest_path = output_dir / "rollout_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    return RolloutCaptureResult(
        output_dir=str(output_dir),
        manifest_path=str(manifest_path),
        artifacts=artifacts,
        status=status,
    )


def _resolve_run_dir(policy_id: str) -> Path | None:
    if not policy_id:
        return None
    path = Path(policy_id)
    if path.exists():
        return path.parent if path.is_file() else path
    return None


def _collect_artifacts(run_dir: Path, config: RolloutCaptureConfig) -> list[str]:
    artifacts: list[str] = []
    patterns = config.include_patterns or ()
    for pattern in patterns:
        for path in run_dir.rglob(pattern):
            if path.is_file():
                artifacts.append(str(path))
                if config.max_artifacts and len(artifacts) >= config.max_artifacts:
                    return artifacts
    return artifacts
