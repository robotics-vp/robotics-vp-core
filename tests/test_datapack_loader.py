"""Unit tests for datapack YAML loader."""
from src.motor_backend.datapacks import load_datapack_configs


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
        "objective_hint: prioritize error reduction\n"
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
    assert cfg.objective_hint == "prioritize error reduction"


def test_load_datapack_defaults(tmp_path):
    path = tmp_path / "datapack.yaml"
    path.write_text("id: dp_empty\n")
    configs = load_datapack_configs([path])
    cfg = configs[0]
    assert cfg.tags == []
    assert cfg.task_tags == []
    assert cfg.robot_families == []
    assert cfg.objective_hint is None
