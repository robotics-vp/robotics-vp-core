from pathlib import Path

from PIL import Image

from src.vla.teacher_runtime import (
    OpenVLATeacherRuntime,
    TeacherActionEnvelope,
    TeacherAdapterContract,
    load_teacher_action_envelope_json,
    load_teacher_adapter_contract_json,
    save_teacher_action_envelope_json,
    save_teacher_adapter_contract_json,
)


class _DummyConfig:
    model_name = "dummy/openvla"
    device = "cpu"
    dtype = "float32"


class _DummyController:
    cfg = _DummyConfig()
    available = False

    def predict_action(self, image: Image.Image, instruction: str):
        raise RuntimeError("teacher_missing")


class _AvailableController:
    cfg = _DummyConfig()
    available = True

    def predict_action(self, image: Image.Image, instruction: str):
        return {
            "vla_available": True,
            "confidence": 0.82,
            "dx": 0.35,
            "gripper": 0.6,
            "source": "dummy/openvla",
            "semantic_tags": ["mode:recovery"],
            "object_refs": ["drawer"],
            "risk_hints": ["fragility"],
        }


def test_teacher_runtime_serialization_round_trip(tmp_path: Path) -> None:
    contract = TeacherAdapterContract(
        teacher_id="openvla",
        model_name="dummy/openvla",
        modality="action_semantics",
        available=False,
        metadata={"enabled": False},
    )
    envelope = TeacherActionEnvelope.unavailable(
        teacher_id="openvla",
        model_name="dummy/openvla",
        instruction="do the thing",
        failure_mode="teacher_missing",
        metadata={"contract_id": contract.contract_id},
    )

    contract_path = tmp_path / "teacher_contract.json"
    envelope_path = tmp_path / "teacher_action.json"
    save_teacher_adapter_contract_json(contract_path, contract)
    save_teacher_action_envelope_json(envelope_path, envelope)

    loaded_contract = load_teacher_adapter_contract_json(contract_path)
    loaded_envelope = load_teacher_action_envelope_json(envelope_path)

    assert loaded_contract.contract_id == contract.contract_id
    assert loaded_contract.provider_truth["authority_class"] == "canonical_metadata"
    assert loaded_envelope.failure_mode == "teacher_missing"
    assert loaded_envelope.available is False
    assert loaded_envelope.provider_truth["availability_class"] in {"teacher_missing", "unavailable"}


def test_openvla_teacher_runtime_reports_unavailable_predictions() -> None:
    runtime = OpenVLATeacherRuntime(_DummyController())

    contract = runtime.describe_contract()
    envelope = runtime.predict_action(Image.new("RGB", (8, 8), "gray"), "be safe")

    assert contract.available is False
    assert envelope.available is False
    assert envelope.failure_mode == "teacher_missing"
    assert envelope.provenance["contract_id"] == contract.contract_id
    assert contract.provider_truth["backend_selected"] == "unavailable"
    assert contract.metadata["execution_preconditions"]["ready"] is False
    assert contract.metadata["backend_status"]["backend_selected"] == "unavailable"
    assert envelope.metadata["execution_preconditions"]["ready"] is False
    assert envelope.provider_truth["authority_class"] == "canonical_metadata"
    assert "object:drawer" in TeacherActionEnvelope.unavailable(
        teacher_id="openvla",
        model_name="dummy/openvla",
        instruction="open the drawer carefully",
        failure_mode="teacher_missing",
    ).semantic_tags


def test_openvla_teacher_runtime_enriches_semantic_hints() -> None:
    runtime = OpenVLATeacherRuntime(_AvailableController())

    envelope = runtime.predict_action(Image.new("RGB", (8, 8), "gray"), "open the drawer carefully")
    vla_payload = envelope.to_vla_payload()

    assert envelope.available is True
    assert "object:drawer" in envelope.semantic_tags
    assert "affordance:open" in envelope.semantic_tags
    assert "drawer" in envelope.object_refs
    assert "open" in envelope.affordance_hints
    assert "fragility" in envelope.risk_hints
    assert "drawer" in vla_payload["object_refs"]
    assert "open" in vla_payload["affordance_hints"]
    assert envelope.metadata["backend_selected"] == "real"
    assert envelope.provider_truth["backend_selected"] == "real"
    assert envelope.provider_truth["availability_class"] == "real_backend"
