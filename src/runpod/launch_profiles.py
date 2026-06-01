"""Typed RunPod launch profiles for provider, loop, and training runs.

The profiles prepare manifest-shaped launch intent. They do not create pods,
execute provider code, train weights, or claim runtime proof.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

from src.utils.json_safe import to_json_safe
from src.world_model.humanoid_readiness.g1_primary_environment import (
    PRIMARY_EMBODIMENT_ID,
    PRIMARY_ENV_ID,
    PRIMARY_HARDWARE_CLASS,
    PRIMARY_POSTURE_TAG,
    PRIMARY_ROBOT_FAMILY,
    PRIMARY_TASK_ID,
    primary_env_metadata,
)


RUNPOD_LAUNCH_PROFILE_IDS = (
    "provider_bringup",
    "g1_loop_run",
    "g1_sac_training",
)


def _mapping(payload: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    return dict(to_json_safe(dict(payload or {})))


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _git_value(args: list[str], default: str) -> str:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=_repo_root(),
            check=True,
            capture_output=True,
            text=True,
        )
        value = proc.stdout.strip()
        return value or default
    except Exception:
        return default


def _utc_run_id(profile_id: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"runpod-{stamp}-{profile_id.replace('_', '-')}"


@dataclass(frozen=True)
class RunPodLaunchProfile:
    """Reusable RunPod launch intent for a run family."""

    profile_id: str
    pod_class: str
    run_class: str
    epistemic_status: str
    task: str
    wm: str
    subsystem: str
    blocker: str
    gpu_class: str
    estimated_cost_usd: float
    config_paths: list[str] = field(default_factory=list)
    seeds: list[int] = field(default_factory=list)
    commands: list[str] = field(default_factory=list)
    artifact_paths: list[str] = field(default_factory=list)
    dependency_chain: list[str] = field(default_factory=list)
    timeout_seconds: int = 3600
    requires_volume: bool = False
    image: str = "nvidia/cuda:12.1.0-runtime-ubuntu22.04"
    template: str = ""
    expected_value: str = ""
    urgency: str = "medium"
    rollback_notes: str = ""
    replay_notes: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def render(self, *, run_id: str) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "pod_class": self.pod_class,
            "run_class": self.run_class,
            "epistemic_status": self.epistemic_status,
            "task": self.task,
            "wm": self.wm,
            "subsystem": self.subsystem,
            "blocker": self.blocker,
            "gpu_class": self.gpu_class,
            "estimated_cost_usd": float(self.estimated_cost_usd),
            "config_paths": list(self.config_paths),
            "seeds": list(self.seeds),
            "commands": [
                command.format(run_id=run_id, primary_env_id=PRIMARY_ENV_ID)
                for command in self.commands
            ],
            "artifact_paths": [
                path.format(run_id=run_id, primary_env_id=PRIMARY_ENV_ID)
                for path in self.artifact_paths
            ],
            "dependency_chain": list(self.dependency_chain),
            "timeout_seconds": int(self.timeout_seconds),
            "requires_volume": bool(self.requires_volume),
            "image": self.image,
            "template": self.template,
            "expected_value": self.expected_value,
            "urgency": self.urgency,
            "rollback_notes": self.rollback_notes,
            "replay_notes": self.replay_notes,
            "metadata": _mapping(self.metadata),
        }


def _profiles() -> dict[str, RunPodLaunchProfile]:
    g1_metadata = primary_env_metadata(source_curriculum_env="workcell")
    return {
        "provider_bringup": RunPodLaunchProfile(
            profile_id="provider_bringup",
            pod_class="provider",
            run_class="provider",
            epistemic_status="proof_of_life",
            task="G1 perception/provider bring-up smoke for egocentric humanoid grounding",
            wm="perception_grounding",
            subsystem="g1_provider_bringup",
            blocker="provider_gpu_runtime_truth_missing",
            gpu_class="A100-40GB",
            estimated_cost_usd=3.0,
            config_paths=[
                "configs/humanoid/g1_primary_env.yaml",
                "docs/economic_world_model/g1_primary_environment.md",
            ],
            seeds=[0],
            commands=[
                "pip install -r requirements-gpu.txt",
                "python3 scripts/economic_world_model/compile_economic_wm_provider_runbook.py --output-dir artifacts/economic_world_model/provider_runs/{run_id}/runbook",
                "python3 scripts/economic_world_model/validate_economic_wm_provider_runbook.py --output-dir artifacts/economic_world_model/provider_runs/{run_id}/validation --runbook artifacts/economic_world_model/provider_runs/{run_id}/runbook/economic_wm_provider_runbook_v1.json",
            ],
            artifact_paths=[
                "artifacts/economic_world_model/provider_runs/{run_id}/runbook/",
                "artifacts/economic_world_model/provider_runs/{run_id}/validation/",
            ],
            dependency_chain=[
                "runpodctl_ready",
                "provider weights access confirmed on pod",
                "g1 primary environment profile reviewed",
            ],
            timeout_seconds=3600,
            requires_volume=False,
            expected_value="proof-of-life only: separate provider availability from G1 deployment truth",
            rollback_notes="Do not promote provider outputs; discard receipts if provider truth is unavailable.",
            replay_notes="Re-run the same commit, profile, provider weights, and seeds.",
            metadata=g1_metadata,
        ),
        "g1_loop_run": RunPodLaunchProfile(
            profile_id="g1_loop_run",
            pod_class="loop",
            run_class="loop",
            epistemic_status="proof_of_life",
            task="G1 humanoid loop/replay collection with fixed-base curriculum sources explicitly bounded",
            wm="sim_synth_physics",
            subsystem="g1_loop_replay",
            blocker="gpu_loop_replay_runtime_truth_missing",
            gpu_class="A40",
            estimated_cost_usd=8.0,
            config_paths=["configs/humanoid/g1_primary_env.yaml"],
            seeds=[0, 1, 2],
            commands=[
                "pip install -r requirements-gpu.txt",
                "python3 scripts/economic_world_model/run_cpu_august_gap_tranche.py --output-dir artifacts/economic_world_model/provider_runs/{run_id}/cpu_gap_join --no-build-attempt",
                "python3 scripts/bootstrap_semantic_workcell_loop.py --output-root artifacts/economic_world_model/provider_runs/{run_id}/curriculum_loop --episodes 24 --steps 8 --max-frames 8 --seed 24 --backend-policy real",
            ],
            artifact_paths=[
                "artifacts/economic_world_model/provider_runs/{run_id}/cpu_gap_join/",
                "artifacts/economic_world_model/provider_runs/{run_id}/curriculum_loop/",
                "results/run_registry/{run_id}/",
            ],
            dependency_chain=[
                "RUNPOD_VOLUME_ID configured",
                "G1 profile is primary target",
                "fixed-base curriculum sources remain bounded",
            ],
            timeout_seconds=14400,
            requires_volume=True,
            expected_value="proof-of-life loop receipts before interpreting replay as training data",
            rollback_notes="Do not treat workcell curriculum replay as G1 hardware proof.",
            replay_notes="Re-run with the same commit, seeds, G1 profile, and curriculum source settings.",
            metadata=g1_metadata,
        ),
        "g1_sac_training": RunPodLaunchProfile(
            profile_id="g1_sac_training",
            pod_class="train",
            run_class="train",
            epistemic_status="proof_of_life",
            task="G1 primary SAC plumbing proof-of-life with dishwashing fixed-base curriculum source",
            wm="embodiment_actuation",
            subsystem="g1_sac_plumbing",
            blocker="gpu_training_runtime_receipts_missing",
            gpu_class="A100-80GB",
            estimated_cost_usd=12.0,
            config_paths=[
                "configs/humanoid/g1_primary_env.yaml",
                "configs/sac/contract_aware_smoke.yaml",
            ],
            seeds=[0, 42, 137],
            commands=[
                "pip install -r requirements-gpu.txt",
                "python train_sac.py --episodes 1000 --seed 0 --primary-env-id {primary_env_id} --target-task-id "
                + PRIMARY_TASK_ID
                + " --target-robot-family "
                + PRIMARY_ROBOT_FAMILY
                + " --target-posture-tag "
                + PRIMARY_POSTURE_TAG
                + " --target-embodiment-id "
                + PRIMARY_EMBODIMENT_ID
                + " --source-curriculum-env dishwashing --use-condition-vector --log-path results/run_registry/{run_id}/metrics/sac_train.csv --checkpoint-path results/run_registry/{run_id}/checkpoints/sac_final.pt",
            ],
            artifact_paths=[
                "results/run_registry/{run_id}/metrics/",
                "results/run_registry/{run_id}/checkpoints/",
                "artifacts/economic_world_model/provider_runs/{run_id}/training_runtime_manifest_v1.json",
            ],
            dependency_chain=[
                "RUNPOD_VOLUME_ID configured",
                "G1 primary environment profile reviewed",
                "curriculum source boundary remains fixed_base_tabletop",
                "promotion gate remains false",
            ],
            timeout_seconds=28800,
            requires_volume=True,
            expected_value="proof-of-life only: verify training-loop receipts and G1 metadata plumbing",
            rollback_notes="Do not replace stable checkpoints from this smoke; keep outputs run-scoped.",
            replay_notes="Re-run with the same G1 profile, source curriculum env, configs, image, commit, and seeds.",
            metadata=primary_env_metadata(source_curriculum_env="dishwashing"),
        ),
    }


def get_runpod_launch_profile(profile_id: str) -> RunPodLaunchProfile:
    profiles = _profiles()
    if profile_id not in profiles:
        raise KeyError(
            f"Unknown RunPod launch profile {profile_id!r}; expected one of {sorted(profiles)}"
        )
    return profiles[profile_id]


def build_runpod_launch_manifest(
    *,
    profile_id: str,
    run_id: str = "",
    branch: str = "",
    commit_sha: str = "",
    volume_id: str | None = None,
    template: str = "",
    image: str = "",
) -> dict[str, Any]:
    """Build a pending manifest for a RunPod launch profile."""

    profile = get_runpod_launch_profile(profile_id)
    resolved_run_id = run_id or _utc_run_id(profile_id)
    rendered = profile.render(run_id=resolved_run_id)
    manifest = {
        "run_id": resolved_run_id,
        "mode": "runpod",
        "pod_class": rendered["pod_class"],
        "run_class": rendered["run_class"],
        "epistemic_status": rendered["epistemic_status"],
        "commit_sha": commit_sha or _git_value(["rev-parse", "--short", "HEAD"], "unknown"),
        "branch": branch or _git_value(["rev-parse", "--abbrev-ref", "HEAD"], "unknown"),
        "task": rendered["task"],
        "wm": rendered["wm"],
        "subsystem": rendered["subsystem"],
        "blocker": rendered["blocker"],
        "config_paths": rendered["config_paths"],
        "seeds": rendered["seeds"],
        "image": image or rendered["image"],
        "template": template or rendered["template"],
        "pod_id": None,
        "volume_id": volume_id,
        "commands": rendered["commands"],
        "artifact_paths": rendered["artifact_paths"],
        "status": "pending",
        "started_at": None,
        "finished_at": None,
        "cost_snapshot": None,
        "gpu_class": rendered["gpu_class"],
        "wall_clock_seconds": None,
        "artifact_size_bytes": None,
        "storage_or_checkpoint_size_bytes": None,
        "expected_value": rendered["expected_value"],
        "estimated_cost_usd": rendered["estimated_cost_usd"],
        "dependency_chain": rendered["dependency_chain"],
        "urgency": rendered["urgency"],
        "justified_itself": None,
        "rollback_notes": rendered["rollback_notes"],
        "replay_notes": rendered["replay_notes"],
        "metadata": {
            **_mapping(rendered["metadata"]),
            "launch_profile_id": profile.profile_id,
            "requires_volume": bool(rendered["requires_volume"]),
            "timeout_seconds": int(rendered["timeout_seconds"]),
            "primary_env_id": PRIMARY_ENV_ID,
            "primary_task_id": PRIMARY_TASK_ID,
            "primary_robot_family": PRIMARY_ROBOT_FAMILY,
            "primary_hardware_class": PRIMARY_HARDWARE_CLASS,
            "primary_posture_tag": PRIMARY_POSTURE_TAG,
            "launch_sequence": [
                "scripts/runpod/ensure_cli.sh",
                "scripts/runpod/launch_pod.sh",
                "scripts/runpod/sync_up.sh",
                "scripts/runpod/exec_remote.sh",
                "scripts/runpod/sync_down.sh",
                "scripts/runpod/collect_billing.sh",
                "scripts/runpod/cleanup_idle.sh",
            ],
        },
    }
    return dict(to_json_safe(manifest))


def _launch_command(manifest: Mapping[str, Any]) -> str:
    pieces = [
        "./scripts/runpod/launch_pod.sh",
        "--class",
        str(manifest["pod_class"]),
        "--gpu",
        str(manifest["gpu_class"]),
        "--timeout",
        str(_mapping(manifest.get("metadata")).get("timeout_seconds", 3600)),
        "--name",
        str(manifest["run_id"]),
        "--run-id",
        str(manifest["run_id"]),
        "--image",
        str(manifest["image"]),
    ]
    if manifest.get("volume_id"):
        pieces.extend(["--volume", str(manifest["volume_id"])])
    if manifest.get("template"):
        pieces.extend(["--template", str(manifest["template"])])
    return " ".join(pieces)


def write_runpod_launch_manifest(
    *,
    profile_id: str,
    output_root: str | Path,
    run_id: str = "",
    branch: str = "",
    commit_sha: str = "",
    volume_id: str | None = None,
    template: str = "",
    image: str = "",
) -> dict[str, Any]:
    """Write `.agent/runs/<run_id>/manifest.json` and launch command."""

    manifest = build_runpod_launch_manifest(
        profile_id=profile_id,
        run_id=run_id,
        branch=branch,
        commit_sha=commit_sha,
        volume_id=volume_id,
        template=template,
        image=image,
    )
    run_dir = Path(output_root) / str(manifest["run_id"])
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    launch_command_path = run_dir / "launch_command.sh"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    launch_command_path.write_text(_launch_command(manifest) + "\n", encoding="utf-8")
    return {
        "run_id": manifest["run_id"],
        "profile_id": profile_id,
        "manifest_path": str(manifest_path),
        "launch_command_path": str(launch_command_path),
        "manifest": manifest,
    }


__all__ = [
    "RUNPOD_LAUNCH_PROFILE_IDS",
    "RunPodLaunchProfile",
    "build_runpod_launch_manifest",
    "get_runpod_launch_profile",
    "write_runpod_launch_manifest",
]
