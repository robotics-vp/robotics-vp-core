"""Unit tests for datapack YAML loader."""
from src.motor_backend.datapacks import DatapackConfig, MotionClipSpec, load_datapack_configs, save_datapack_config


def test_load_datapack_configs(tmp_path):
    path = tmp_path / "datapack.yaml"
    path.write_text(
        "id: dp_test\n"
        "description: test pack\n"
        "motion_clips:\n"
        "  - path: data/mocap/clip.npz\n"
        "    weight: 0.8\n"
        "domain_randomization:\n"
        "  terrain: flat\n"
        "curriculum:\n"
        "  initial_difficulty: 0.1\n"
        "tags:\n"
        "  - humanoid\n"
        "  - logging\n"
        "task_tags:\n"
        "  - reach\n"
        "robot_families:\n"
        "  - G1\n"
        "quality_score: 0.65\n"
        "novelty_score: 0.4\n"
        "objective_hint: prioritize error reduction\n"
        "metadata:\n"
        "  scene_tracks_backend: real\n"
        "  execution_preconditions:\n"
        "    ready: true\n"
    )
    configs = load_datapack_configs([path])
    assert len(configs) == 1
    cfg = configs[0]
    assert cfg.id == "dp_test"
    assert cfg.description == "test pack"
    assert cfg.motion_clips[0].path == "data/mocap/clip.npz"
    assert cfg.motion_clips[0].weight == 0.8
    assert cfg.domain_randomization["terrain"] == "flat"
    assert cfg.curriculum["initial_difficulty"] == 0.1
    assert cfg.tags == ["humanoid", "logging"]
    assert cfg.task_tags == ["reach"]
    assert cfg.robot_families == ["G1"]
    assert cfg.quality_score == 0.65
    assert cfg.novelty_score == 0.4
    assert cfg.objective_hint == "prioritize error reduction"
    assert cfg.metadata["scene_tracks_backend"] == "real"
    assert cfg.metadata["execution_preconditions"]["ready"] is True


def test_load_datapack_defaults(tmp_path):
    path = tmp_path / "datapack.yaml"
    path.write_text("id: dp_empty\n")
    configs = load_datapack_configs([path])
    cfg = configs[0]
    assert cfg.tags == []
    assert cfg.task_tags == []
    assert cfg.robot_families == []
    assert cfg.objective_hint is None


def test_save_datapack_config_roundtrips_metadata(tmp_path):
    config = DatapackConfig(
        id="dp_roundtrip",
        description="roundtrip",
        motion_clips=[MotionClipSpec(path="data/clip.npz", weight=1.0)],
        quality_score=0.7,
        novelty_score=0.25,
        tags=["humanoid"],
        metadata={
            "teacher_runtime_backend_selected": "real",
            "execution_preconditions": {"ready": False, "blocking_preconditions": ["signal_bool::teacher_runtime_real"]},
        },
    )

    path = save_datapack_config(config, tmp_path)
    [loaded] = load_datapack_configs([path])

    assert loaded.quality_score == 0.7
    assert loaded.novelty_score == 0.25
    assert loaded.metadata["teacher_runtime_backend_selected"] == "real"
    assert loaded.metadata["execution_preconditions"]["ready"] is False
