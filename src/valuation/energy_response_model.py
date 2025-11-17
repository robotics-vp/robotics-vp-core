import json
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn


@dataclass
class EnergyInterventionSample:
    env_name: str
    profile_name: str  # "BASE" | "BOOST" | "SAVER" | "SAFE"
    context: Dict[str, Any]
    energy_knobs: Dict[str, float]
    mpl: float
    error_rate: float
    energy_Wh: float
    risk_metric: float


class EnergyResponseNet(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, out_dim: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_energy_interventions(path: str) -> List[EnergyInterventionSample]:
    """Read interventions JSONL (data/energy_interventions.jsonl) into samples."""
    samples: List[EnergyInterventionSample] = []
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            summ = rec.get("summary", {})
            profile = rec.get("profile", "BASE")
            env = rec.get("env", "unknown")
            energy_knobs = rec.get("energy_knobs", {}) or {}
            # Fall back to EnergyProfile defaults if saved differently
            if not energy_knobs and "energy_profile" in rec:
                energy_knobs = rec["energy_profile"]
            mpl = summ.get("mpl_episode", summ.get("mpl_t", 0.0))
            error_rate = summ.get("error_rate_episode", 0.0)
            energy_Wh = summ.get("energy_Wh", 0.0)
            risk = summ.get("error_rate_episode", 0.0)
            samples.append(
                EnergyInterventionSample(
                    env_name=env,
                    profile_name=profile,
                    context={"episode": rec.get("episode", None)},
                    energy_knobs=energy_knobs,
                    mpl=mpl,
                    error_rate=error_rate,
                    energy_Wh=energy_Wh,
                    risk_metric=risk,
                )
            )
    return samples


def _one_hot(value: str, vocab: List[str]) -> List[float]:
    return [1.0 if value == v else 0.0 for v in vocab]


def build_deltas(samples: List[EnergyInterventionSample]) -> Dict[Tuple[str, Any], Dict[str, EnergyInterventionSample]]:
    """Group samples by (env, episode) and map profile->sample."""
    grouped: Dict[Tuple[str, Any], Dict[str, EnergyInterventionSample]] = {}
    for s in samples:
        key = (s.env_name, s.context.get("episode"))
        grouped.setdefault(key, {})[s.profile_name] = s
    return grouped


def encode_sample(sample: EnergyInterventionSample, base: EnergyInterventionSample, env_vocab: List[str], profile_vocab: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Return feature vector x and target y (deltas vs BASE)."""
    # Features: env one-hot + profile one-hot + energy knobs (sorted keys)
    env_one = _one_hot(sample.env_name, env_vocab)
    prof_one = _one_hot(sample.profile_name, profile_vocab)
    knob_keys = sorted(sample.energy_knobs.keys())
    knob_vec = [float(sample.energy_knobs[k]) for k in knob_keys]
    x = np.array(env_one + prof_one + knob_vec, dtype=np.float32)

    # Targets: deltas vs base
    y = np.array(
        [
            sample.mpl - base.mpl,
            sample.error_rate - base.error_rate,
            sample.energy_Wh - base.energy_Wh,
            sample.risk_metric - base.risk_metric,
        ],
        dtype=np.float32,
    )
    return x, y


class EnergyResponseModel:
    """Wrapper for training/predicting energy response."""

    def __init__(self, in_dim: int, hidden_dim: int = 64):
        self.net = EnergyResponseNet(in_dim, hidden_dim)

    def fit(self, X: torch.Tensor, Y: torch.Tensor, epochs: int = 50, lr: float = 1e-3, device: str = "cpu"):
        self.net.to(device)
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        losses = {"train": []}
        for _ in range(epochs):
            self.net.train()
            opt.zero_grad()
            pred = self.net(X.to(device))
            loss = loss_fn(pred, Y.to(device))
            loss.backward()
            opt.step()
            losses["train"].append(loss.item())
        return losses

    def predict(self, x: np.ndarray, device: str = "cpu") -> Dict[str, float]:
        self.net.eval()
        with torch.no_grad():
            t = torch.from_numpy(x.astype(np.float32)).to(device)
            out = self.net(t.unsqueeze(0)).cpu().numpy()[0]
        return {
            "delta_mpl": float(out[0]),
            "delta_error": float(out[1]),
            "delta_energy_Wh": float(out[2]),
            "delta_risk": float(out[3]),
        }

    def save(self, path: str):
        torch.save(self.net.state_dict(), path)

    def load(self, path: str, device: str = "cpu"):
        state = torch.load(path, map_location=device)
        self.net.load_state_dict(state)
        self.net.to(device)
