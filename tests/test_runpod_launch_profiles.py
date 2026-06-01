from pathlib import Path

from src.runpod import (
    RUNPOD_LAUNCH_PROFILE_IDS,
    build_runpod_launch_manifest,
    write_runpod_launch_manifest,
)
from src.world_model.economic_world_model.gpu_run_hygiene import (
    validate_gpu_run_manifest_payload,
)
from src.world_model.humanoid_readiness.g1_primary_environment import (
    PRIMARY_ENV_ID,
    PRIMARY_POSTURE_TAG,
)


def test_runpod_profiles_cover_provider_loop_and_training() -> None:
    assert set(RUNPOD_LAUNCH_PROFILE_IDS) == {
        "provider_bringup",
        "g1_loop_run",
        "g1_sac_training",
    }

    for profile_id in RUNPOD_LAUNCH_PROFILE_IDS:
        manifest = build_runpod_launch_manifest(
            profile_id=profile_id,
            run_id=f"runpod-20260901-120000-{profile_id}",
            branch="main",
            commit_sha="abcdef1",
            volume_id="vol-test",
        )
        receipts = validate_gpu_run_manifest_payload(manifest)
        blocking = [
            receipt
            for receipt in receipts
            if not receipt.passed and receipt.severity == "blocking"
        ]

        assert not blocking
        assert manifest["mode"] == "runpod"
        assert manifest["metadata"]["primary_env_id"] == PRIMARY_ENV_ID
        assert manifest["metadata"]["primary_posture_tag"] == PRIMARY_POSTURE_TAG
        assert manifest["commands"]
        assert manifest["artifact_paths"]


def test_write_runpod_launch_manifest_materializes_manifest_and_command(tmp_path) -> None:
    payload = write_runpod_launch_manifest(
        profile_id="g1_sac_training",
        output_root=tmp_path,
        run_id="runpod-20260901-120000-g1-sac-training",
        branch="main",
        commit_sha="abcdef1",
        volume_id="vol-test",
    )

    manifest_path = Path(payload["manifest_path"])
    launch_command_path = Path(payload["launch_command_path"])
    assert manifest_path.exists()
    assert launch_command_path.exists()
    launch_command = launch_command_path.read_text(encoding="utf-8")
    assert "--class train" in launch_command
    assert "--run-id runpod-20260901-120000-g1-sac-training" in launch_command
    assert payload["manifest"]["metadata"]["source_curriculum_env"] == "dishwashing"
