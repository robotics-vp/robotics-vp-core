"""Learned branch planner for the sim/synth/physics WM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np


GENERATION_MODES = [
    "coverage_branch",
    "targeted_synth_rollout",
    "physics_probe",
    "geometry_guarded_rollout",
    "neural_branch_candidate",
]
PHYSICS_BACKENDS = ["pybullet", "isaac", "holosoma", "other"]
FIDELITY_LABELS = ["fast_scan", "branch_balanced", "high_fidelity"]
OBJECTIVE_PRESETS = ["balanced", "throughput", "safety", "energy_saver"]


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
class BranchPlannerFeatureVector:
    raw: np.ndarray

    @property
    def dim(self) -> int:
        return int(len(self.raw))


class BranchPlannerFeatureExtractor:
    """Extract job/context features for branch planning."""

    FEATURE_NAMES = [
        "coverage_gap_score",
        "economic_priority",
        "trust_priority",
        "readiness",
        "intent_explore",
        "intent_exploit",
        "intent_validate",
        "has_risk_family",
        "has_object_family",
        "backend_pybullet",
        "backend_isaac",
        "backend_holosoma",
        "backend_other",
        "fidelity_fast_scan",
        "fidelity_branch_balanced",
        "fidelity_high_fidelity",
        "objective_balanced",
        "objective_throughput",
        "objective_safety",
        "objective_energy_saver",
        "heuristic_mode_coverage_branch",
        "heuristic_mode_targeted_synth_rollout",
        "heuristic_mode_physics_probe",
        "heuristic_mode_geometry_guarded_rollout",
        "heuristic_mode_neural_branch_candidate",
    ]
    FEATURE_DIM = len(FEATURE_NAMES)

    def build_feature_dict(
        self,
        *,
        job: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> Dict[str, float]:
        physics_context = dict(context.get("physics_context", {}) or {})
        heuristic_generation_mode = str(
            context.get("heuristic_generation_mode", "coverage_branch") or "coverage_branch"
        )
        backend = str(physics_context.get("backend", "other") or "other")
        if backend not in PHYSICS_BACKENDS:
            backend = "other"
        fidelity = str(physics_context.get("fidelity_tier", "branch_balanced") or "branch_balanced")
        objective_preset = str(job.get("objective_preset", "balanced") or "balanced")
        feature_dict = {
            "coverage_gap_score": _clip01(job.get("coverage_gap_score", 0.0)),
            "economic_priority": _clip01(job.get("economic_priority", 0.0)),
            "trust_priority": _clip01(job.get("trust_priority", 0.0)),
            "readiness": _clip01(job.get("readiness", 0.0)),
            "intent_explore": float(str(job.get("data_collection_intent", "")) == "explore"),
            "intent_exploit": float(str(job.get("data_collection_intent", "")) == "exploit"),
            "intent_validate": float(str(job.get("data_collection_intent", "")) == "validate"),
            "has_risk_family": float(bool(str(job.get("risk_family", "")))),
            "has_object_family": float(bool(str(job.get("object_family", "")))),
        }
        for name, value in zip(self.FEATURE_NAMES[9:13], _one_hot(backend, PHYSICS_BACKENDS)):
            feature_dict[name] = float(value)
        for name, value in zip(self.FEATURE_NAMES[13:16], _one_hot(fidelity, FIDELITY_LABELS)):
            feature_dict[name] = float(value)
        for name, value in zip(self.FEATURE_NAMES[16:20], _one_hot(objective_preset, OBJECTIVE_PRESETS)):
            feature_dict[name] = float(value)
        for name, value in zip(self.FEATURE_NAMES[20:25], _one_hot(heuristic_generation_mode, GENERATION_MODES)):
            feature_dict[name] = float(value)
        return feature_dict

    def __call__(self, *, job: Mapping[str, Any], context: Mapping[str, Any]) -> BranchPlannerFeatureVector:
        feature_dict = self.build_feature_dict(job=job, context=context)
        return BranchPlannerFeatureVector(
            raw=np.array(
                [_safe_float(feature_dict.get(name), 0.0) for name in self.FEATURE_NAMES],
                dtype=np.float32,
            )
        )

    def extract_batch(
        self,
        rows: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    ) -> np.ndarray:
        return np.array([self(job=job, context=context).raw for job, context in rows], dtype=np.float32)


try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class LearnedBranchPlanner(nn.Module):
        """Predict branch generation mode and expected yield."""

        def __init__(
            self,
            input_dim: int = BranchPlannerFeatureExtractor.FEATURE_DIM,
            hidden_dim: int = 64,
        ) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            self.mode_head = nn.Linear(hidden_dim, len(GENERATION_MODES))
            self.yield_head = nn.Sequential(
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid(),
            )
            self.reject_head = nn.Linear(hidden_dim, 1)

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            hidden = self.net(x)
            return self.mode_head(hidden), self.yield_head(hidden), self.reject_head(hidden)

        def predict_context(
            self,
            *,
            job: Mapping[str, Any],
            context: Mapping[str, Any],
            extractor: Optional[BranchPlannerFeatureExtractor] = None,
        ) -> Dict[str, Any]:
            ext = extractor or BranchPlannerFeatureExtractor()
            x = torch.from_numpy(ext(job=job, context=context).raw).unsqueeze(0)
            with torch.no_grad():
                mode_logits, yield_score, reject_logit = self.forward(x)
                mode_probs = F.softmax(mode_logits, dim=-1).squeeze(0).cpu().numpy().tolist()
                yield_value = float(yield_score.squeeze(0).item())
                reject_probability = float(torch.sigmoid(reject_logit).squeeze(0).item())
            mode_idx = int(max(range(len(mode_probs)), key=lambda idx: mode_probs[idx]))
            return {
                "generation_mode": GENERATION_MODES[mode_idx],
                "expected_yield_score": yield_value,
                "mode_probabilities": {
                    label: float(prob) for label, prob in zip(GENERATION_MODES, mode_probs)
                },
                "reject_probability": reject_probability,
                "reject_recommended": reject_probability > 0.5,
            }

        def plan_branch(self, *, job: Mapping[str, Any], context: Mapping[str, Any]) -> Dict[str, Any]:
            return self.predict_context(job=job, context=context)

        @classmethod
        def from_checkpoint(cls, path: str) -> "LearnedBranchPlanner":
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            model = cls(
                input_dim=ckpt.get("input_dim", BranchPlannerFeatureExtractor.FEATURE_DIM),
                hidden_dim=ckpt.get("hidden_dim", 64),
            )
            if "reject_head.weight" not in ckpt.get("model_state_dict", {}):
                nn.init.constant_(model.reject_head.weight, 0.0)
                nn.init.constant_(model.reject_head.bias, -6.0)
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
            model.eval()
            return model

    TORCH_AVAILABLE = True

except ImportError:  # pragma: no cover
    TORCH_AVAILABLE = False

    class LearnedBranchPlanner:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("LearnedBranchPlanner requires torch")

        @classmethod
        def from_checkpoint(cls, path: str):
            raise ImportError("LearnedBranchPlanner requires torch")


def train_branch_planner(
    rows: Sequence[Mapping[str, Any]],
    *,
    negative_rows: Optional[Sequence[Mapping[str, Any]]] = None,
    epochs: int = 50,
    lr: float = 1e-3,
    hidden_dim: int = 64,
    reject_loss_weight: float = 0.5,
    save_path: Optional[str] = None,
) -> Any:
    if not TORCH_AVAILABLE:
        raise ImportError("Training requires torch")
    extractor = BranchPlannerFeatureExtractor()
    feature_rows = []
    mode_labels = []
    yield_targets = []
    for row in rows:
        job = dict(row.get("job", {}) or {})
        context = dict(row.get("context", {}) or {})
        target_mode = str(
            row.get("target_generation_mode", context.get("heuristic_generation_mode", "coverage_branch"))
            or "coverage_branch"
        )
        if target_mode not in GENERATION_MODES:
            continue
        feature_rows.append((job, context))
        mode_labels.append(GENERATION_MODES.index(target_mode))
        yield_targets.append(_clip01(row.get("target_expected_yield_score", 0.0)))
    if not feature_rows:
        raise ValueError("No training data produced")

    X = torch.from_numpy(extractor.extract_batch(feature_rows))
    y_mode = torch.tensor(mode_labels, dtype=torch.long)
    y_yield = torch.tensor(yield_targets, dtype=torch.float32).unsqueeze(-1)
    reject_feature_rows = list(feature_rows)
    reject_labels = [0.0 for _ in feature_rows]
    for row in list(negative_rows or []):
        reject_feature_rows.append(
            (
                dict(row.get("job", {}) or {}),
                dict(row.get("context", {}) or {}),
            )
        )
        reject_labels.append(1.0)
    X_reject = torch.from_numpy(extractor.extract_batch(reject_feature_rows))
    y_reject = torch.tensor(reject_labels, dtype=torch.float32).unsqueeze(-1)

    model = LearnedBranchPlanner(
        input_dim=BranchPlannerFeatureExtractor.FEATURE_DIM,
        hidden_dim=hidden_dim,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_mode = nn.CrossEntropyLoss()
    loss_yield = nn.MSELoss()
    loss_reject = nn.BCEWithLogitsLoss()

    model.train()
    for _ in range(epochs):
        mode_logits, yield_score, _ = model(X)
        _, _, reject_logits = model(X_reject)
        loss = (
            loss_mode(mode_logits, y_mode)
            + loss_yield(yield_score, y_yield)
            + (float(reject_loss_weight) * loss_reject(reject_logits, y_reject))
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    if save_path:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "input_dim": BranchPlannerFeatureExtractor.FEATURE_DIM,
                "hidden_dim": hidden_dim,
                "epochs": epochs,
                "n_rows": len(rows),
                "n_negative_rows": len(list(negative_rows or [])),
                "reject_loss_weight": float(reject_loss_weight),
                "heads": [
                    "generation_mode",
                    "expected_yield",
                    "reject_probability",
                ],
            },
            save_path,
        )
    return model
