#!/usr/bin/env python3
"""
Smoke test for OpenVLAController with explicit backend status.
"""
from PIL import Image

from src.vla.openvla_controller import OpenVLAController, OpenVLAConfig


def main():
    cfg = OpenVLAConfig(backend_policy="auto")
    controller = OpenVLAController(cfg)
    controller.load_model()
    img = Image.new("RGB", (256, 256), color="gray")
    out = controller.predict_action(img, "Open the drawer without hitting the vase.")
    print("Backend status:", controller.backend_status())
    print("VLA available:", out.get("vla_available"))
    print("Action:", out)


if __name__ == "__main__":
    main()
