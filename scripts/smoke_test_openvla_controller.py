#!/usr/bin/env python3
"""
Smoke test for OpenVLAController scaffold (no hard dependency on HF).
"""
from PIL import Image

from src.vla.openvla_controller import OpenVLAController, OpenVLAConfig


def main():
    cfg = OpenVLAConfig()
    controller = OpenVLAController(cfg)
    controller.load_model()
    img = Image.new("RGB", (256, 256), color="gray")
    out = controller.predict_action(img, "Open the drawer without hitting the vase.")
    print("VLA available:", out.get("vla_available"))
    print("Action:", out)


if __name__ == "__main__":
    main()
