from dataclasses import dataclass
import numpy as np


@dataclass
class EnergyProfile:
    speed_scale: float = 1.0
    torque_scale_shoulder: float = 1.0
    torque_scale_elbow: float = 1.0
    torque_scale_wrist: float = 1.0
    torque_scale_gripper: float = 1.0
    safety_margin_scale: float = 1.0


def apply_energy_profile_to_action(action: np.ndarray, profile: EnergyProfile) -> np.ndarray:
    """
    Apply energy profile to an action vector (scales velocity/safety proxies).
    """
    scaled = np.array(action, dtype=np.float32) * float(profile.speed_scale)
    return np.clip(scaled, -1.0, 1.0)
