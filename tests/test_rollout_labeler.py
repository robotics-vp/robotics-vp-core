"""Tests for rollout labeler."""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.motor_backend.datapacks import DatapackConfig, MotionClipSpec
from src.motor_backend.rollout_capture import EpisodeMetadata, EpisodeRollout, RolloutBundle
from src.vla.rollout_labeler import label_rollouts_with_vla
from src.vla.teacher_runtime import TeacherAdapterContract, TeacherActionEnvelope


def test_rollout_labeler_appends_tags(tmp_path: Path):
    base = DatapackConfig(
        id="dp_base",
        description="Base",
        motion_clips=[MotionClipSpec(path="data/clip.npz")],
        tags=["humanoid"],
    )
    rollout = EpisodeRollout(
        metadata=EpisodeMetadata(
            episode_id="ep1",
            task_id="task_a",
            robot_family="G1",
            seed=None,
            env_params={},
        ),
        trajectory_path=tmp_path / "trajectory.npz",
    )
    bundle = RolloutBundle(scenario_id="scenario_1", episodes=[rollout])

    labeled = label_rollouts_with_vla(bundle, base_datapack=base)
    assert labeled
    assert labeled[0].id == "dp_base_vla"
    assert "vla_labeled" in labeled[0].tags
    assert "auto_labeled" in labeled[0].tags


def test_rollout_labeler_stub_without_openvla(monkeypatch, tmp_path: Path):
    import src.vla.rollout_labeler as labeler

    monkeypatch.delenv("OPENVLA_ENABLE", raising=False)
    monkeypatch.delenv("VLA_ENABLE", raising=False)
    monkeypatch.setattr(labeler, "_get_openvla_teacher_runtime", lambda: (_ for _ in ()).throw(AssertionError("OpenVLA not expected")))

    base = DatapackConfig(
        id="dp_base",
        description="Base",
        motion_clips=[MotionClipSpec(path="data/clip.npz")],
        tags=["humanoid"],
    )
    rollout = EpisodeRollout(
        metadata=EpisodeMetadata(
            episode_id="ep1",
            task_id="task_a",
            robot_family="G1",
            seed=None,
            env_params={},
        ),
        trajectory_path=tmp_path / "trajectory.npz",
    )
    bundle = RolloutBundle(scenario_id="scenario_stub", episodes=[rollout])

    labeled = labeler.label_rollouts_with_vla(bundle, base_datapack=base)
    assert labeled
    assert "auto_labeled" in labeled[0].tags
    teacher_contract_path = tmp_path / "trajectory_teacher_contract_v1.json"
    teacher_action_path = tmp_path / "trajectory_teacher_action_envelope_v1.json"
    teacher_trace_path = tmp_path / "trajectory_teacher_trace_v1.json"
    assert teacher_contract_path.exists()
    assert teacher_action_path.exists()
    assert teacher_trace_path.exists()
    teacher_contract = json.loads(teacher_contract_path.read_text())
    assert teacher_contract["available"] is False
    assert teacher_contract["metadata"]["availability_reason"] == "openvla_disabled"
    assert teacher_contract["provider_truth"]["authority_class"] == "canonical_metadata"
    assert teacher_contract["provider_truth"]["availability_class"] == "disabled"
    teacher_action = json.loads(teacher_action_path.read_text())
    assert teacher_action["available"] is False
    assert teacher_action["failure_mode"] == "openvla_disabled"
    assert teacher_action["provider_truth"]["authority_class"] == "canonical_metadata"
    teacher_trace = json.loads(teacher_trace_path.read_text())
    assert teacher_trace["advisory_only"] is True
    assert teacher_trace["provider_truth"]["authority_class"] == "canonical_metadata"
    assert labeled[0].metadata["execution_preconditions"]["ready"] is False
    assert labeled[0].metadata["future_training_signals"]["semantic_grounding_non_heuristic"] is False
    assert labeled[0].metadata["teacher_runtime_backend_selected"] == "disabled"
    assert labeled[0].metadata["teacher_provider_truth"]["availability_class"] == "disabled"


def test_rollout_labeler_openvla_error_fallback(monkeypatch, tmp_path: Path):
    import src.vla.rollout_labeler as labeler

    monkeypatch.setenv("OPENVLA_ENABLE", "1")
    monkeypatch.setattr(labeler, "_get_openvla_teacher_runtime", lambda: (_ for _ in ()).throw(RuntimeError("boom")))

    base = DatapackConfig(
        id="dp_base",
        description="Base",
        motion_clips=[MotionClipSpec(path="data/clip.npz")],
        tags=["humanoid"],
    )
    rollout = EpisodeRollout(
        metadata=EpisodeMetadata(
            episode_id="ep1",
            task_id="task_a",
            robot_family="G1",
            seed=None,
            env_params={},
        ),
        trajectory_path=tmp_path / "trajectory.npz",
    )
    bundle = RolloutBundle(scenario_id="scenario_error", episodes=[rollout])

    labeled = labeler.label_rollouts_with_vla(bundle, base_datapack=base)
    assert labeled
    assert "vla_error" in labeled[0].tags
    teacher_contract_path = tmp_path / "trajectory_teacher_contract_v1.json"
    teacher_action_path = tmp_path / "trajectory_teacher_action_envelope_v1.json"
    assert teacher_contract_path.exists()
    assert teacher_action_path.exists()
    teacher_contract = json.loads(teacher_contract_path.read_text())
    teacher_action = json.loads(teacher_action_path.read_text())
    assert teacher_contract["metadata"]["availability_reason"] == "boom"
    assert teacher_action["failure_mode"] == "boom"
    assert teacher_action["provider_truth"]["authority_class"] == "canonical_metadata"


def test_rollout_labeler_preserves_structured_teacher_semantics(monkeypatch, tmp_path: Path):
    import src.vla.rollout_labeler as labeler

    class _Runtime:
        def describe_contract(self):
            return TeacherAdapterContract(
                teacher_id="openvla",
                model_name="dummy/openvla",
                modality="action_semantics",
                advisory_only=True,
                available=True,
            )

        def predict_action(self, image, instruction):
            return TeacherActionEnvelope(
                teacher_id="openvla",
                model_name="dummy/openvla",
                instruction=instruction,
                available=True,
                action={"dx": 0.3, "gripper": 0.5, "vla_available": 1.0, "confidence": 0.8},
                confidence=0.8,
                failure_mode="teacher_available",
                semantic_tags=["object:drawer", "affordance:open", "risk:fragility"],
                object_refs=["drawer"],
                affordance_hints=["open"],
                risk_hints=["fragility"],
            )

    monkeypatch.setenv("OPENVLA_ENABLE", "1")
    monkeypatch.setattr(labeler, "_get_openvla_teacher_runtime", lambda: (_Runtime(), None))

    rgb_path = tmp_path / "frame.png"
    Image.new("RGB", (8, 8), "gray").save(rgb_path)

    base = DatapackConfig(
        id="dp_base",
        description="Open the drawer carefully",
        motion_clips=[MotionClipSpec(path="data/clip.npz")],
        tags=["humanoid"],
    )
    rollout = EpisodeRollout(
        metadata=EpisodeMetadata(
            episode_id="ep1",
            task_id="task_a",
            robot_family="G1",
            seed=None,
            env_params={},
        ),
        trajectory_path=tmp_path / "trajectory.npz",
        rgb_video_path=rgb_path,
    )
    np.savez_compressed(
        rollout.trajectory_path,
        trajectory={
            "scene_tracks_backend": "real",
            "semantic_memory_grounded": True,
            "scene_tracks_v1": {
                "scene_tracks_v1/summary_json": np.array(
                    ['{"topology":{"grounded_track_object_count":2}}'],
                    dtype="U96",
                ),
            },
        },
    )
    bundle = RolloutBundle(scenario_id="scenario_structured", episodes=[rollout])

    labeled = labeler.label_rollouts_with_vla(bundle, base_datapack=base)

    assert labeled
    teacher_trace = json.loads((tmp_path / "trajectory_teacher_trace_v1.json").read_text())
    assert teacher_trace["metadata"]["object_refs"] == ["drawer"]
    assert teacher_trace["metadata"]["affordance_hints"] == ["open"]
    assert teacher_trace["metadata"]["risk_hints"] == ["fragility"]
    assert "object:drawer" in teacher_trace["metadata"]["semantic_tags"]
    assert teacher_trace["provider_truth"]["backend_selected"] == "real"
    assert labeled[0].metadata["scene_tracks_backend"] == "real"
    assert labeled[0].metadata["scene_tracks_provider_truth"]["grounding_class"] == "non_heuristic_grounded"
    assert labeled[0].metadata["semantic_grounding_mode"] == "non_heuristic"
    assert labeled[0].metadata["grounded_track_object_count"] == 2
    assert labeled[0].metadata["future_training_signals"]["teacher_runtime_live"] is True
    assert labeled[0].metadata["execution_preconditions"]["ready"] is True
