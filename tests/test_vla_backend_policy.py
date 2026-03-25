from __future__ import annotations

import pytest
from PIL import Image

from src.vla.backbones.meta_dino_backbone import MetaDINOBackbone
from src.vla.openvla_controller import OpenVLAConfig, OpenVLAController


def test_openvla_disabled_backend_returns_explicit_unavailable() -> None:
    controller = OpenVLAController(OpenVLAConfig(backend_policy="disabled"))
    controller.load_model()

    result = controller.predict_action(Image.new("RGB", (8, 8), "gray"), "be safe")

    assert controller.available is False
    assert controller.backend_selected == "disabled"
    assert result["vla_available"] is False
    assert result["backend_selected"] == "disabled"
    assert result["fallback_mode"] == "backend_disabled"


def test_openvla_stub_backend_is_explicit() -> None:
    controller = OpenVLAController(OpenVLAConfig(backend_policy="stub"))
    controller.load_model()

    result = controller.predict_action(Image.new("RGB", (8, 8), "gray"), "be safe")

    assert controller.backend_selected == "stub"
    assert result["source"] == "openvla_stub"
    assert result["fallback_mode"] == "explicit_stub"


def test_openvla_dummy_vision_backbone_requires_stub_policy() -> None:
    controller = OpenVLAController(
        OpenVLAConfig(
            backend_policy="disabled",
            use_vision_backbone=True,
            vision_backbone_type="dummy",
            vision_backbone_policy="auto",
        )
    )
    controller.load_model()

    assert controller.has_vision_backbone() is False
    assert controller.vision_backbone_selected == "unavailable"
    assert controller.vision_backbone_failure_reason == "dummy_backbone_requires_stub_policy"


def test_openvla_dummy_vision_backbone_loads_only_under_stub_policy() -> None:
    controller = OpenVLAController(
        OpenVLAConfig(
            backend_policy="disabled",
            use_vision_backbone=True,
            vision_backbone_type="dummy",
            vision_backbone_policy="stub",
        )
    )
    controller.load_model()

    assert controller.has_vision_backbone() is True
    assert controller.vision_backbone_selected == "stub"


def test_metadino_unavailable_raises_without_stub(monkeypatch) -> None:
    def fake_try(self) -> None:
        self._available = False
        self._backend_selected = "unavailable"
        self._failure_reason = "missing_weights"

    monkeypatch.setattr(MetaDINOBackbone, "_try_load_model", fake_try)
    backbone = MetaDINOBackbone(model_name="facebook/dinov2-small", device="cpu", backend_policy="auto")

    assert backbone.available is False
    assert backbone.backend_selected == "unavailable"
    with pytest.raises(RuntimeError, match="missing_weights"):
        backbone.encode_frame(Image.new("RGB", (8, 8), "gray"))


def test_metadino_stub_policy_emits_embeddings() -> None:
    backbone = MetaDINOBackbone(model_name="facebook/dinov2-small", device="cpu", backend_policy="stub")
    embedding = backbone.encode_frame(Image.new("RGB", (8, 8), "gray"))

    assert backbone.backend_selected == "stub"
    assert embedding.shape == (backbone.embedding_dim,)
