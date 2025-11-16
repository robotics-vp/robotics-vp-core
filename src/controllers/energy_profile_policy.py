import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from src.controllers.energy_profile import EnergyProfile


class EnergyProfilePolicy(nn.Module):
    """
    Predicts EnergyProfile knobs from context features.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 6)  # [speed, τ_shoulder, τ_elbow, τ_wrist, τ_gripper, safety]

    def forward(self, x: torch.Tensor) -> Tuple[EnergyProfile, torch.Tensor]:
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        raw = self.out(h)

        # Map to reasonable ranges
        speed = 0.5 + torch.sigmoid(raw[:, 0]) * 1.0  # [0.5, 1.5]
        tau_sh = 0.5 + torch.sigmoid(raw[:, 1]) * 1.0
        tau_el = 0.5 + torch.sigmoid(raw[:, 2]) * 1.0
        tau_wr = 0.5 + torch.sigmoid(raw[:, 3]) * 1.0
        tau_gr = 0.5 + torch.sigmoid(raw[:, 4]) * 1.0
        safety = 0.8 + torch.sigmoid(raw[:, 5]) * 1.2  # [0.8, 2.0]

        profiles = []
        for i in range(x.shape[0]):
            profiles.append(EnergyProfile(
                speed_scale=float(speed[i]),
                torque_scale_shoulder=float(tau_sh[i]),
                torque_scale_elbow=float(tau_el[i]),
                torque_scale_wrist=float(tau_wr[i]),
                torque_scale_gripper=float(tau_gr[i]),
                safety_margin_scale=float(safety[i]),
            ))
        params = torch.stack([speed, tau_sh, tau_el, tau_wr, tau_gr, safety], dim=1)
        return profiles, params
