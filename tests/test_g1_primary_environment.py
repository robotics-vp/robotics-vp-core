from pathlib import Path

from src.world_model.humanoid_readiness.g1_primary_environment import (
    CURRICULUM_POSTURE_TAG,
    PRIMARY_ENV_ID,
    PRIMARY_POSTURE_TAG,
    classify_env_posture,
    curriculum_proxy_metadata,
    default_g1_primary_environment_doctrine,
    primary_env_metadata,
    run_g1_primary_env_hygiene,
)


ROOT = Path(__file__).resolve().parents[1]


def test_g1_doctrine_marks_legacy_envs_as_curriculum() -> None:
    doctrine = default_g1_primary_environment_doctrine()

    assert doctrine.primary_env_id == PRIMARY_ENV_ID
    assert doctrine.primary_posture_tag == PRIMARY_POSTURE_TAG
    assert doctrine.legacy_curriculum_envs["dishwashing"]
    assert classify_env_posture("dishwashing_online_sac") == CURRICULUM_POSTURE_TAG
    assert classify_env_posture("bipedal_whole_body_unitree_g1") == PRIMARY_POSTURE_TAG


def test_primary_env_metadata_keeps_curriculum_boundary() -> None:
    metadata = primary_env_metadata(source_curriculum_env="dishwashing")
    source = curriculum_proxy_metadata("dishwashing")

    assert metadata["primary_env_id"] == PRIMARY_ENV_ID
    assert metadata["source_curriculum"]["posture_tag"] == CURRICULUM_POSTURE_TAG
    assert source["promotion_limit"] == "cannot_close_g1_r1_whole_body_readiness"
    assert metadata["unitree_hardware_truth"] is False
    assert metadata["promotion_eligible"] is False


def test_repo_g1_primary_hygiene_passes(tmp_path) -> None:
    report = run_g1_primary_env_hygiene(
        repo_root=ROOT,
        output_dir=tmp_path / "g1_primary_env_hygiene",
    )

    assert report["status"] == "ok_g1_primary_env_hygiene_passed"
    assert report["blocking_issue_count"] == 0
    assert report["legacy_primary_claim_count"] == 0
    assert report["missing_required_surface_count"] == 0
