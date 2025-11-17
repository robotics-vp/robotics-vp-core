import numpy as np
from typing import Dict, Any
from src.vla.openvla_controller import OpenVLAController, OpenVLAConfig


class ActionVLA:
    def __init__(self, controller: OpenVLAController):
        self.controller = controller

    @classmethod
    def with_default(cls):
        ctl = OpenVLAController(OpenVLAConfig())
        ctl.load_model()
        return cls(ctl)

    def predict_action(self, frame, instruction: str) -> np.ndarray:
        action = self.controller.predict_action(frame, instruction)
        return np.array(action.get("raw_action", [0.0] * 7), dtype=float)

    def summarize_action(self, action: np.ndarray) -> Dict[str, Any]:
        pos_mag = float(np.sum(np.abs(action[:3]))) if action.size >= 3 else 0.0
        rot_mag = float(np.sum(np.abs(action[3:6]))) if action.size >= 6 else 0.0
        return {
            "large_delta": pos_mag > 0.5 or rot_mag > 0.5,
            "precision_grasp": float(action[-1]) > 0.5 if action.size >= 7 else False,
        }
