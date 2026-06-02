"""Training substrate for learned D4 knob calibration."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from src.contracts.schemas import KnobPolicyV1, PlanPolicyConfigV1, RegimeFeaturesV1
from src.regal.knob_model import (
    HeuristicKnobProvider,
    MAX_CONSERVATIVE_MULTIPLIER,
    MAX_GAIN_MULTIPLIER,
    MAX_PATIENCE,
    MIN_CONSERVATIVE_MULTIPLIER,
    MIN_GAIN_MULTIPLIER,
    MIN_PATIENCE,
)
from src.utils.config_digest import sha256_json

KNOB_OBJECTIVE_LABELS = [
    "exploration",
    "exploitation",
    "validation",
    "balanced",
    "unknown",
]
KNOB_TASK_FAMILY_KEYS = [
    "manipulation",
    "navigation",
    "recovery",
    "inspection",
]
KNOB_FEATURE_NAMES = [
    "audit_delta_success",
    "audit_delta_error",
    "audit_success_rate",
    "exposure_count_norm",
    "datapack_count_norm",
    "probe_delta_epi_per_flop_tanh",
    "probe_stability_pass",
    "probe_transfer_pass",
    "graph_sigma",
    "graph_nav_success",
    "graph_shortcut_fraction",
    "regal_spec_score",
    "regal_coherence_score",
    "regal_hack_prob",
    "base_full_multiplier_norm",
    "base_conservative_multiplier_norm",
    "base_cooldown_steps_norm",
    "objective_exploration",
    "objective_exploitation",
    "objective_validation",
    "objective_balanced",
    "objective_unknown",
    "task_family_mean",
    "task_family_std",
    "task_family_min",
    "task_family_max",
    "task_family_manipulation",
    "task_family_navigation",
    "task_family_recovery",
    "task_family_inspection",
]

torch: Any
nn: Any

try:
    import torch as _torch
    import torch.nn as _torch_nn

    torch = _torch
    nn = _torch_nn

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - explicit failures in training caller
    TORCH_AVAILABLE = False
    torch = None
    nn = None


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _normalize_gain(value: float) -> float:
    return _clamp01((float(value) - MIN_GAIN_MULTIPLIER) / (MAX_GAIN_MULTIPLIER - MIN_GAIN_MULTIPLIER))


def _normalize_conservative(value: float) -> float:
    return _clamp01(
        (float(value) - MIN_CONSERVATIVE_MULTIPLIER)
        / (MAX_CONSERVATIVE_MULTIPLIER - MIN_CONSERVATIVE_MULTIPLIER)
    )


def _normalize_patience(value: float) -> float:
    return _clamp01((float(value) - MIN_PATIENCE) / float(MAX_PATIENCE - MIN_PATIENCE))


def _coerce_objective_label(value: Any) -> str:
    label = str(value or "unknown").lower()
    if label in KNOB_OBJECTIVE_LABELS[:-1]:
        return label
    if label == "balanced":
        return "balanced"
    return "unknown"


def build_knob_feature_vector(
    features: RegimeFeaturesV1 | Mapping[str, Any],
    base_config: PlanPolicyConfigV1 | Mapping[str, Any],
) -> np.ndarray:
    regime = features if isinstance(features, RegimeFeaturesV1) else RegimeFeaturesV1(**dict(features or {}))
    config = base_config if isinstance(base_config, PlanPolicyConfigV1) else PlanPolicyConfigV1(**dict(base_config or {}))
    task_weights = dict(regime.task_family_weights or {})
    weight_values = [float(task_weights.get(key, 0.0)) for key in KNOB_TASK_FAMILY_KEYS]
    if task_weights:
        stats = np.asarray(list(task_weights.values()), dtype=np.float32)
        weight_mean = float(stats.mean())
        weight_std = float(stats.std())
        weight_min = float(stats.min())
        weight_max = float(stats.max())
    else:
        weight_mean = 0.0
        weight_std = 0.0
        weight_min = 0.0
        weight_max = 0.0
    objective = _coerce_objective_label(regime.objective_profile)
    objective_one_hot = [1.0 if objective == label else 0.0 for label in KNOB_OBJECTIVE_LABELS]
    cooldown_steps = config.gain_schedule.cooldown_steps or 3
    vector = np.asarray(
        [
            _safe_float(regime.audit_delta_success, 0.0),
            _safe_float(regime.audit_delta_error, 0.0),
            _safe_float(regime.audit_success_rate, 0.0),
            float(np.log1p(max(regime.exposure_count, 0))) / 6.0,
            float(np.log1p(max(regime.datapack_count, 0))) / 6.0,
            float(np.tanh(_safe_float(regime.probe_delta_epi_per_flop, 0.0) * 1_000_000.0)),
            1.0 if regime.probe_stability_pass else 0.0,
            1.0 if regime.probe_transfer_pass else 0.0,
            _safe_float(regime.graph_sigma, 0.0),
            _safe_float(regime.graph_nav_success, 0.0),
            _safe_float(regime.graph_shortcut_fraction, 0.0),
            _safe_float(regime.regal_spec_score, 0.0),
            _safe_float(regime.regal_coherence_score, 0.0),
            _safe_float(regime.regal_hack_prob, 0.0),
            _normalize_gain(config.gain_schedule.full_multiplier),
            _normalize_conservative(config.gain_schedule.conservative_multiplier),
            _normalize_patience(cooldown_steps),
            *objective_one_hot,
            weight_mean,
            weight_std,
            weight_min,
            weight_max,
            *weight_values,
        ],
        dtype=np.float32,
    )
    return vector


def target_triplet_from_policy(
    policy: KnobPolicyV1 | Mapping[str, Any],
    base_config: PlanPolicyConfigV1 | Mapping[str, Any],
) -> np.ndarray:
    knob_policy = policy if isinstance(policy, KnobPolicyV1) else KnobPolicyV1(**dict(policy or {}))
    config = base_config if isinstance(base_config, PlanPolicyConfigV1) else PlanPolicyConfigV1(**dict(base_config or {}))
    cooldown_steps = config.gain_schedule.cooldown_steps or 3
    gain = knob_policy.gain_multiplier_override
    if gain is None:
        gain = config.gain_schedule.full_multiplier
    conservative = knob_policy.conservative_multiplier_override
    if conservative is None:
        conservative = config.gain_schedule.conservative_multiplier
    patience = knob_policy.patience_override
    if patience is None:
        patience = cooldown_steps
    return np.asarray(
        [
            _normalize_gain(gain),
            _normalize_conservative(conservative),
            _normalize_patience(patience),
        ],
        dtype=np.float32,
    )


@dataclass(frozen=True)
class KnobTrainingRow:
    row_id: str
    regime_features: Dict[str, Any]
    base_config: Dict[str, Any]
    target_policy: Dict[str, Any]
    target_source: str
    promotion_stage: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "row_id": self.row_id,
            "regime_features": dict(self.regime_features),
            "base_config": dict(self.base_config),
            "target_policy": dict(self.target_policy),
            "target_source": self.target_source,
            "promotion_stage": self.promotion_stage,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class KnobTrainingDataset:
    rows: list[KnobTrainingRow]
    summary: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": "knob_training_dataset_v1",
            "summary": dict(self.summary),
            "rows": [row.to_dict() for row in self.rows],
        }


def build_knob_training_dataset(rows: Sequence[KnobTrainingRow]) -> KnobTrainingDataset:
    row_payloads = [row.to_dict() for row in rows]
    target_source_counts: Dict[str, int] = {}
    promotion_stage_counts: Dict[str, int] = {}
    for row in rows:
        target_source_counts[row.target_source] = target_source_counts.get(row.target_source, 0) + 1
        promotion_stage_counts[row.promotion_stage] = promotion_stage_counts.get(row.promotion_stage, 0) + 1
    summary = {
        "schema_version": "knob_training_dataset_summary_v1",
        "num_rows": len(rows),
        "target_source_counts": dict(sorted(target_source_counts.items())),
        "promotion_stage_counts": dict(sorted(promotion_stage_counts.items())),
        "feature_names": list(KNOB_FEATURE_NAMES),
        "dataset_digest": sha256_json(row_payloads),
    }
    return KnobTrainingDataset(rows=list(rows), summary=summary)


def save_knob_training_dataset(dataset: KnobTrainingDataset, path: str | Path) -> str:
    dataset_path = Path(path)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_path.write_text(json.dumps(dataset.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return str(dataset_path)


def load_knob_training_dataset(path: str | Path) -> KnobTrainingDataset:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = [
        KnobTrainingRow(
            row_id=str(row.get("row_id", f"row_{idx}")),
            regime_features=dict(row.get("regime_features", {}) or {}),
            base_config=dict(row.get("base_config", {}) or {}),
            target_policy=dict(row.get("target_policy", {}) or {}),
            target_source=str(row.get("target_source", "unknown")),
            promotion_stage=str(row.get("promotion_stage", "heuristic_fallback")),
            metadata=dict(row.get("metadata", {}) or {}),
        )
        for idx, row in enumerate(payload.get("rows", []) or [])
        if isinstance(row, Mapping)
    ]
    return KnobTrainingDataset(rows=rows, summary=dict(payload.get("summary", {}) or {}))


def _sample_regime_features(rng: np.random.Generator) -> RegimeFeaturesV1:
    objective = rng.choice(KNOB_OBJECTIVE_LABELS[:-1]).item()
    task_weights = {
        "manipulation": float(rng.uniform(0.1, 0.7)),
        "navigation": float(rng.uniform(0.1, 0.7)),
        "recovery": float(rng.uniform(0.0, 0.3)),
    }
    total = sum(task_weights.values())
    task_weights = {key: value / total for key, value in task_weights.items()}
    return RegimeFeaturesV1(
        audit_delta_success=float(rng.uniform(-0.35, 0.35)),
        audit_delta_error=float(rng.uniform(-0.25, 0.25)),
        audit_success_rate=float(rng.uniform(0.2, 0.95)),
        exposure_count=int(rng.integers(1, 250)),
        datapack_count=int(rng.integers(1, 200)),
        probe_delta_epi_per_flop=float(rng.uniform(-2e-6, 6e-6)),
        probe_stability_pass=bool(rng.random() > 0.15),
        probe_transfer_pass=bool(rng.random() > 0.35),
        graph_sigma=float(rng.uniform(0.0, 1.0)),
        graph_nav_success=float(rng.uniform(0.2, 1.0)),
        graph_shortcut_fraction=float(rng.uniform(0.0, 0.8)),
        regal_spec_score=float(rng.uniform(0.2, 1.0)),
        regal_coherence_score=float(rng.uniform(0.2, 1.0)),
        regal_hack_prob=float(rng.uniform(0.0, 0.4)),
        objective_profile=objective,
        task_family_weights=task_weights,
    )


def _sample_base_config(rng: np.random.Generator) -> PlanPolicyConfigV1:
    from src.contracts.schemas import PlanGainScheduleV1

    return PlanPolicyConfigV1(
        gain_schedule=PlanGainScheduleV1(
            full_multiplier=float(rng.uniform(1.1, 1.8)),
            conservative_multiplier=float(rng.uniform(0.95, 1.25)),
            cooldown_steps=int(rng.integers(2, 6)),
        ),
        default_weights={
            "manipulation": 0.5,
            "navigation": 0.5,
        },
    )


def generate_synthetic_knob_training_rows(
    num_rows: int,
    *,
    seed: int = 0,
) -> list[KnobTrainingRow]:
    rng = np.random.default_rng(seed)
    heuristic = HeuristicKnobProvider()
    rows: list[KnobTrainingRow] = []
    for idx in range(max(0, int(num_rows))):
        features = _sample_regime_features(rng)
        base_config = _sample_base_config(rng)
        target_policy = heuristic.predict(features, base_config)
        rows.append(
            KnobTrainingRow(
                row_id=f"knob_row_{idx}",
                regime_features=features.model_dump(mode="json"),
                base_config=base_config.model_dump(mode="json"),
                target_policy=target_policy.model_dump(mode="json"),
                target_source="heuristic_bootstrap",
                promotion_stage="shadow_candidate",
                metadata={"synthetic": True},
            )
        )
    return rows


if TORCH_AVAILABLE:

    class KnobCalibrationNet(nn.Module):
        def __init__(self, input_dim: int = len(KNOB_FEATURE_NAMES), hidden_dim: int = 32) -> None:
            super().__init__()
            self.input_dim = int(input_dim)
            self.hidden_dim = int(hidden_dim)
            self.net = nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
            )
            self.output_head = nn.Linear(self.hidden_dim, 3)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            hidden = self.net(x)
            return self.output_head(hidden)


else:  # pragma: no cover

    class KnobCalibrationNet:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("KnobCalibrationNet requires torch")


def train_knob_calibration_model(
    rows: Sequence[KnobTrainingRow],
    *,
    hidden_dim: int = 32,
    epochs: int = 50,
    lr: float = 1e-3,
    save_path: Optional[str] = None,
) -> tuple[Any, Dict[str, Any]]:
    if not TORCH_AVAILABLE:
        raise ImportError("Training requires torch")
    if not rows:
        raise ValueError("No knob training rows provided")

    X = np.asarray(
        [
            build_knob_feature_vector(row.regime_features, row.base_config)
            for row in rows
        ],
        dtype=np.float32,
    )
    y = np.asarray(
        [
            target_triplet_from_policy(row.target_policy, row.base_config)
            for row in rows
        ],
        dtype=np.float32,
    )
    sample_weights = np.asarray(
        [
            1.0
            + (0.25 if row.promotion_stage == "shadow_candidate" else 0.0)
            + (0.35 if row.promotion_stage == "promoted" else 0.0)
            + (0.15 if row.target_source == "runtime_receipt" else 0.0)
            for row in rows
        ],
        dtype=np.float32,
    )

    X_tensor = torch.from_numpy(X)
    y_tensor = torch.from_numpy(y)
    weights_tensor = torch.from_numpy(sample_weights).unsqueeze(-1)

    model = KnobCalibrationNet(input_dim=X.shape[1], hidden_dim=hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss(reduction="none")

    history: Dict[str, Any] = {"loss": [], "gain_mse": [], "conservative_mse": [], "patience_mse": []}
    model.train()
    for _ in range(int(epochs)):
        raw_outputs = model(X_tensor)
        predictions = torch.sigmoid(raw_outputs)
        per_elem_loss = loss_fn(predictions, y_tensor)
        weighted_loss = (per_elem_loss * weights_tensor).mean()

        optimizer.zero_grad()
        weighted_loss.backward()
        optimizer.step()

        history["loss"].append(float(weighted_loss.item()))
        history["gain_mse"].append(float(per_elem_loss[:, 0].mean().item()))
        history["conservative_mse"].append(float(per_elem_loss[:, 1].mean().item()))
        history["patience_mse"].append(float(per_elem_loss[:, 2].mean().item()))

    model.eval()
    if save_path:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "input_dim": int(X.shape[1]),
                "hidden_dim": int(hidden_dim),
                "feature_names": list(KNOB_FEATURE_NAMES),
            },
            save_path,
        )
    return model, history


__all__ = [
    "KNOB_FEATURE_NAMES",
    "KNOB_OBJECTIVE_LABELS",
    "KNOB_TASK_FAMILY_KEYS",
    "KnobCalibrationNet",
    "KnobTrainingDataset",
    "KnobTrainingRow",
    "TORCH_AVAILABLE",
    "build_knob_feature_vector",
    "build_knob_training_dataset",
    "generate_synthetic_knob_training_rows",
    "load_knob_training_dataset",
    "save_knob_training_dataset",
    "target_triplet_from_policy",
    "train_knob_calibration_model",
]
