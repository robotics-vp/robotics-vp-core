import logging
from dataclasses import dataclass
from typing import Dict

import numpy as np
from PIL import Image


@dataclass
class OpenVLAConfig:
    model_name: str = "openvla/openvla-7b"
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    unnorm_key: str = "bridge_orig"
    max_action_norm: float = 1.0


class OpenVLAController:
    def __init__(self, cfg: OpenVLAConfig = OpenVLAConfig()):
        self.cfg = cfg
        self.available = False
        self.model = None
        self.processor = None

    def load_model(self):
        try:
            from transformers import AutoModelForVision2Seq, AutoProcessor  # type: ignore

            self.processor = AutoProcessor.from_pretrained(self.cfg.model_name, trust_remote_code=True)
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.cfg.model_name,
                attn_implementation="flash_attention_2",
                torch_dtype=getattr(__import__("torch"), self.cfg.dtype, None),
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            ).to(self.cfg.device)
            self.available = True
        except Exception as e:
            logging.warning(f"OpenVLA not available ({e}); falling back to dummy actions.")
            self.available = False

    def predict_action(self, image: Image.Image, instruction: str) -> Dict[str, float]:
        if not self.available or self.model is None or self.processor is None:
            return {
                "dx": 0.0,
                "dy": 0.0,
                "dz": 0.0,
                "droll": 0.0,
                "dpitch": 0.0,
                "dyaw": 0.0,
                "gripper": 0.0,
                "vla_available": False,
                "raw_action": [0.0] * 7,
            }
        import torch

        prompt = f"In: {instruction}\nOut:"
        inputs = self.processor(prompt, image).to(self.cfg.device)
        if self.cfg.dtype == "bfloat16":
            inputs = {k: v.to(torch.bfloat16) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        with torch.no_grad():
            action = self.model.predict_action(**inputs, unnorm_key=self.cfg.unnorm_key, do_sample=False)
        raw = np.array(action).astype(float).tolist()
        clipped = np.clip(raw, -self.cfg.max_action_norm, self.cfg.max_action_norm)
        return {
            "dx": float(clipped[0]),
            "dy": float(clipped[1]),
            "dz": float(clipped[2]),
            "droll": float(clipped[3]),
            "dpitch": float(clipped[4]),
            "dyaw": float(clipped[5]),
            "gripper": float(clipped[6]),
            "vla_available": True,
            "raw_action": raw,
        }


if __name__ == "__main__":
    from PIL import Image

    controller = OpenVLAController()
    controller.load_model()
    img = Image.new("RGB", (256, 256), color="gray")
    out = controller.predict_action(img, "Open the drawer without hitting the vase.")
    print("VLA available:", out.get("vla_available"))
    print("Action:", out)
