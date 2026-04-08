"""Learned backend/fidelity selector for the sim/synth/physics WM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np


BACKEND_LABELS = ["pybullet", "isaac", "holosoma", "other"]
FIDELITY_LABELS = ["fast_scan", "branch_balanced", "high_fidelity"]
RANDOMIZATION_LABELS = [
    "steady_state",
    "coverage_exploration",
    "calibration_focus",
    "benchmark_focus",
]


def _clip01(value: Any) -> float:
    try:
        return float(max(0.0, min(1.0, float(value))))
    except Exception:
        return 0.0


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _one_hot(value: str, candidates: Sequence[str]) -> list[float]:
    normalized = str(value or "")
    if normalized not in candidates:
        normalized = candidates[-1] if candidates else normalized
    return [1.0 if normalized == item else 0.0 for item in candidates]


@dataclass(frozen=True)
class BackendSelectorFeatureVector:
    raw: np.ndarray

    @property
    def dim(self) -> int:
        return int(len(self.raw))


class BackendSelectorFeatureExtractor:
    """Extract fixed-dim context features for backend selection."""

    FEATURE_NAMES = [
        "job_count_norm",
        "mean_coverage_gap",
        "max_coverage_gap",
        "mean_economic_priority",
        "mean_trust_priority",
        "mean_readiness",
        "validate_fraction",
        "exploit_fraction",
        "risk_fraction",
        "object_fraction",
        "benchmark_ready",
        "heuristic_backend_pybullet",
        "heuristic_backend_isaac",
        "heuristic_backend_holosoma",
        "heuristic_backend_other",
        "heuristic_fidelity_fast_scan",
        "heuristic_fidelity_branch_balanced",
        "heuristic_fidelity_high_fidelity",
        "heuristic_randomization_steady_state",
        "heuristic_randomization_coverage_exploration",
        "heuristic_randomization_calibration_focus",
        "heuristic_randomization_benchmark_focus",
    ]
    FEATURE_DIM = len(FEATURE_NAMES)

    def build_feature_dict(self, context: Mapping[str, Any]) -> Dict[str, float]:
        jobs = list(context.get("jobs", []) or [])
        job_count = len(jobs)
        coverage_gaps = [_clip01(item.get("coverage_gap_score", 0.0)) for item in jobs]
        economic = [_clip01(item.get("economic_priority", 0.0)) for item in jobs]
        trust = [_clip01(item.get("trust_priority", 0.0)) for item in jobs]
        readiness = [_clip01(item.get("readiness", 0.0)) for item in jobs]
        intents = [str(item.get("data_collection_intent", "")) for item in jobs]
        risk_fraction = (
            sum(1 for item in jobs if str(item.get("risk_family", ""))) / float(max(job_count, 1))
        )
        object_fraction = (
            sum(1 for item in jobs if str(item.get("object_family", ""))) / float(max(job_count, 1))
        )
        heuristic_backend = str(context.get("heuristic_backend", "other") or "other")
        if heuristic_backend not in BACKEND_LABELS:
            heuristic_backend = "other"
        heuristic_fidelity = str(context.get("heuristic_fidelity_tier", "branch_balanced") or "branch_balanced")
        heuristic_randomization = str(
            context.get("heuristic_domain_randomization_regime", "steady_state") or "steady_state"
        )
        feature_dict = {
            "job_count_norm": min(float(job_count) / 10.0, 1.0),
            "mean_coverage_gap": float(sum(coverage_gaps) / max(len(coverage_gaps), 1)),
            "max_coverage_gap": float(max(coverage_gaps) if coverage_gaps else 0.0),
            "mean_economic_priority": float(sum(economic) / max(len(economic), 1)),
            "mean_trust_priority": float(sum(trust) / max(len(trust), 1)),
            "mean_readiness": float(sum(readiness) / max(len(readiness), 1)),
            "validate_fraction": float(sum(1 for item in intents if item == "validate") / max(job_count, 1)),
            "exploit_fraction": float(sum(1 for item in intents if item == "exploit") / max(job_count, 1)),
            "risk_fraction": float(risk_fraction),
            "object_fraction": float(object_fraction),
            "benchmark_ready": float(
                bool((context.get("benchmark_signals") or {}).get("ready", False))
                or bool((context.get("benchmark_signals") or {}).get("benchmark_eligible", False))
            ),
        }
        for name, value in zip(self.FEATURE_NAMES[11:15], _one_hot(heuristic_backend, BACKEND_LABELS)):
            feature_dict[name] = float(value)
        for name, value in zip(self.FEATURE_NAMES[15:18], _one_hot(heuristic_fidelity, FIDELITY_LABELS)):
            feature_dict[name] = float(value)
        for name, value in zip(
            self.FEATURE_NAMES[18:22],
            _one_hot(heuristic_randomization, RANDOMIZATION_LABELS),
        ):
            feature_dict[name] = float(value)
        return feature_dict

    def __call__(self, context: Mapping[str, Any]) -> BackendSelectorFeatureVector:
        feature_dict = self.build_feature_dict(context)
        return BackendSelectorFeatureVector(
            raw=np.array(
                [_safe_float(feature_dict.get(name), 0.0) for name in self.FEATURE_NAMES],
                dtype=np.float32,
            )
        )

    def extract_batch(self, contexts: Sequence[Mapping[str, Any]]) -> np.ndarray:
        return np.array([self(context).raw for context in contexts], dtype=np.float32)


try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class LearnedBackendSelector(nn.Module):
        """Predict backend, fidelity tier, and randomization regime from WM context."""

        def __init__(
            self,
            input_dim: int = BackendSelectorFeatureExtractor.FEATURE_DIM,
            hidden_dim: int = 64,
        ) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            self.backend_head = nn.Linear(hidden_dim, len(BACKEND_LABELS))
            self.fidelity_head = nn.Linear(hidden_dim, len(FIDELITY_LABELS))
            self.randomization_head = nn.Linear(hidden_dim, len(RANDOMIZATION_LABELS))

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            hidden = self.net(x)
            return (
                self.backend_head(hidden),
                self.fidelity_head(hidden),
                self.randomization_head(hidden),
            )

        def predict_context(
            self,
            *,
            context: Mapping[str, Any],
            extractor: Optional[BackendSelectorFeatureExtractor] = None,
        ) -> Dict[str, Any]:
            ext = extractor or BackendSelectorFeatureExtractor()
            x = torch.from_numpy(ext(context).raw).unsqueeze(0)
            with torch.no_grad():
                backend_logits, fidelity_logits, randomization_logits = self.forward(x)
                backend_probs = F.softmax(backend_logits, dim=-1).squeeze(0).cpu().numpy().tolist()
                fidelity_probs = F.softmax(fidelity_logits, dim=-1).squeeze(0).cpu().numpy().tolist()
                randomization_probs = F.softmax(randomization_logits, dim=-1).squeeze(0).cpu().numpy().tolist()
            backend_idx = int(max(range(len(backend_probs)), key=lambda idx: backend_probs[idx]))
            fidelity_idx = int(max(range(len(fidelity_probs)), key=lambda idx: fidelity_probs[idx]))
            randomization_idx = int(
                max(range(len(randomization_probs)), key=lambda idx: randomization_probs[idx])
            )
            return {
                "preferred_backend": BACKEND_LABELS[backend_idx],
                "fidelity_tier": FIDELITY_LABELS[fidelity_idx],
                "domain_randomization_regime": RANDOMIZATION_LABELS[randomization_idx],
                "backend_probabilities": {
                    label: float(prob) for label, prob in zip(BACKEND_LABELS, backend_probs)
                },
                "fidelity_probabilities": {
                    label: float(prob) for label, prob in zip(FIDELITY_LABELS, fidelity_probs)
                },
                "randomization_probabilities": {
                    label: float(prob)
                    for label, prob in zip(RANDOMIZATION_LABELS, randomization_probs)
                },
            }

        def select_backend(self, *, context: Mapping[str, Any]) -> Dict[str, Any]:
            return self.predict_context(context=context)

        @classmethod
        def from_checkpoint(cls, path: str) -> "LearnedBackendSelector":
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            model = cls(
                input_dim=ckpt.get("input_dim", BackendSelectorFeatureExtractor.FEATURE_DIM),
                hidden_dim=ckpt.get("hidden_dim", 64),
            )
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            return model

    TORCH_AVAILABLE = True

except ImportError:  # pragma: no cover
    TORCH_AVAILABLE = False

    class LearnedBackendSelector:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("LearnedBackendSelector requires torch")

        @classmethod
        def from_checkpoint(cls, path: str):
            raise ImportError("LearnedBackendSelector requires torch")


def train_backend_selector(
    rows: Sequence[Mapping[str, Any]],
    *,
    epochs: int = 50,
    lr: float = 1e-3,
    hidden_dim: int = 64,
    save_path: Optional[str] = None,
) -> Any:
    if not TORCH_AVAILABLE:
        raise ImportError("Training requires torch")
    extractor = BackendSelectorFeatureExtractor()
    features = []
    backend_labels = []
    fidelity_labels = []
    randomization_labels = []
    for row in rows:
        target_backend = str(row.get("target_backend", row.get("heuristic_backend", "other")) or "other")
        if target_backend not in BACKEND_LABELS:
            target_backend = "other"
        target_fidelity = str(
            row.get("target_fidelity_tier", row.get("heuristic_fidelity_tier", "branch_balanced"))
            or "branch_balanced"
        )
        if target_fidelity not in FIDELITY_LABELS:
            continue
        target_randomization = str(
            row.get(
                "target_domain_randomization_regime",
                row.get("heuristic_domain_randomization_regime", "steady_state"),
            )
            or "steady_state"
        )
        if target_randomization not in RANDOMIZATION_LABELS:
            continue
        features.append(extractor(row).raw)
        backend_labels.append(BACKEND_LABELS.index(target_backend))
        fidelity_labels.append(FIDELITY_LABELS.index(target_fidelity))
        randomization_labels.append(RANDOMIZATION_LABELS.index(target_randomization))
    if not features:
        raise ValueError("No training data produced")

    X = torch.from_numpy(np.array(features, dtype=np.float32))
    y_backend = torch.tensor(backend_labels, dtype=torch.long)
    y_fidelity = torch.tensor(fidelity_labels, dtype=torch.long)
    y_randomization = torch.tensor(randomization_labels, dtype=torch.long)

    model = LearnedBackendSelector(
        input_dim=BackendSelectorFeatureExtractor.FEATURE_DIM,
        hidden_dim=hidden_dim,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        backend_logits, fidelity_logits, randomization_logits = model(X)
        loss = (
            loss_fn(backend_logits, y_backend)
            + loss_fn(fidelity_logits, y_fidelity)
            + loss_fn(randomization_logits, y_randomization)
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    if save_path:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "input_dim": BackendSelectorFeatureExtractor.FEATURE_DIM,
                "hidden_dim": hidden_dim,
                "epochs": epochs,
                "n_rows": len(rows),
            },
            save_path,
        )
    return model
