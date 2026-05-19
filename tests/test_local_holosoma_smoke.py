import json
from pathlib import Path

from scripts import local_holosoma_smoke


def test_holosoma_smoke_preflight_reports_missing_module(tmp_path: Path, monkeypatch) -> None:
    policy_path = tmp_path / "policy.onnx"
    policy_path.write_bytes(b"fake")
    monkeypatch.setattr(local_holosoma_smoke, "_has_holosoma_module", lambda: False)
    monkeypatch.setattr(local_holosoma_smoke, "_has_module", lambda _name: True)

    preflight = local_holosoma_smoke._build_preflight(
        task_id="humanoid_wbt_g1",
        policy_ref=str(policy_path),
        policy_source="test",
    )
    written = local_holosoma_smoke._write_preflight(tmp_path, preflight)

    assert preflight["ready"] is False
    assert preflight["holosoma_available"] is False
    assert preflight["policy_exists"] is True
    assert preflight["policy_kind"] == "onnx_deploy"
    assert preflight["missing_preconditions"] == ["holosoma_python_module"]
    assert json.loads(written.read_text(encoding="utf-8")) == preflight


def test_holosoma_smoke_auto_policy_prefers_selected_ref(tmp_path: Path, monkeypatch) -> None:
    selected = tmp_path / "selected.onnx"
    selected.write_bytes(b"fake")
    fallback = tmp_path / "fallback.onnx"
    fallback.write_bytes(b"fake")

    monkeypatch.setattr(
        local_holosoma_smoke,
        "describe_holosoma_policy_contract",
        lambda _context: {
            "policy_ref": str(selected),
            "primary_checkpoint_ref": str(fallback),
        },
    )

    policy_ref, source, contract = local_holosoma_smoke._auto_policy_ref()

    assert policy_ref == str(selected)
    assert source == "policy_ref"
    assert contract["primary_checkpoint_ref"] == str(fallback)


def test_holosoma_smoke_onnx_preflight_requires_onnxruntime(tmp_path: Path, monkeypatch) -> None:
    policy_path = tmp_path / "policy.onnx"
    policy_path.write_bytes(b"fake")
    monkeypatch.setattr(local_holosoma_smoke, "_has_holosoma_module", lambda: True)
    monkeypatch.setattr(local_holosoma_smoke, "_has_module", lambda name: name != "onnxruntime")

    preflight = local_holosoma_smoke._build_preflight(
        task_id="humanoid_wbt_g1",
        policy_ref=str(policy_path),
        policy_source="test",
    )

    assert preflight["ready"] is False
    assert preflight["onnxruntime_available"] is False
    assert preflight["missing_preconditions"] == ["onnxruntime_python_module"]
