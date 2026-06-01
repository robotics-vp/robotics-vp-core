from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.orchestrator.semantic_policy import (
    DatapackSelectionContext,
    DatapackSelectionFeatures,
    DatapackSelectionScorerPackage,
)
from src.utils.config_digest import sha256_json

torch: Any
nn: Any
optim: Any

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

DATAPACK_SELECTION_BENCHMARK_MIN_RUNS = 100
DATAPACK_SELECTION_BENCHMARK_MIN_PAIRWISE = 200


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(frozen=True)
class DatapackSelectionTrainingExample:
    run_id: str
    datapack_id: str
    selected: bool
    supervision_kind: str
    target_score: float
    outcome_score: float
    features: dict[str, float]
    selection_context: dict[str, float]
    feature_delta: dict[str, float]
    metrics: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "datapack_id": self.datapack_id,
            "selected": bool(self.selected),
            "supervision_kind": self.supervision_kind,
            "target_score": float(self.target_score),
            "outcome_score": float(self.outcome_score),
            "features": {
                str(key): float(value) for key, value in self.features.items()
            },
            "selection_context": {
                str(key): float(value) for key, value in self.selection_context.items()
            },
            "feature_delta": {
                str(key): float(value) for key, value in self.feature_delta.items()
            },
            "metrics": {str(key): float(value) for key, value in self.metrics.items()},
        }


@dataclass(frozen=True)
class DatapackSelectionTrainingDataset:
    examples: list[DatapackSelectionTrainingExample]
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "datapack_selection_training_dataset_v1",
            "summary": dict(self.summary),
            "examples": [example.to_dict() for example in self.examples],
        }


def load_selection_run_logs(paths: Sequence[str | Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        log_path = Path(path)
        if not log_path.exists():
            raise FileNotFoundError(f"selection run log not found: {log_path}")
        with log_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if isinstance(payload, dict):
                    rows.append(payload)
    return rows


def build_datapack_selection_training_dataset(
    run_rows: Sequence[Mapping[str, Any]],
) -> DatapackSelectionTrainingDataset:
    examples: list[DatapackSelectionTrainingExample] = []
    positive_runs = 0
    skipped_runs = 0
    policy_counts: dict[str, int] = {}
    promotion_stage_counts: dict[str, int] = {}
    outcome_scores: list[float] = []
    for row in run_rows:
        selection_summary = row.get("selection_summary")
        if not isinstance(selection_summary, Mapping):
            skipped_runs += 1
            continue
        top_candidates = list(selection_summary.get("top_candidates", []) or [])
        selected_ids = {
            str(item) for item in selection_summary.get("selected_ids", []) or []
        }
        if not top_candidates or not selected_ids:
            skipped_runs += 1
            continue
        outcome_score = _run_outcome_score(row)
        outcome_scores.append(outcome_score)
        run_id = str(
            row.get("scenario_id") or row.get("timestamp") or f"run_{len(examples)}"
        )
        selection_context = _selection_context_from_summary(selection_summary)
        selection_policy = str(
            selection_summary.get("selection_policy", "heuristic_only")
            or "heuristic_only"
        )
        policy_counts[selection_policy] = policy_counts.get(selection_policy, 0) + 1
        helper_status = selection_summary.get("selection_helper_status")
        if isinstance(helper_status, Mapping):
            promotion_stage = str(helper_status.get("promotion_stage", "") or "")
            if promotion_stage:
                promotion_stage_counts[promotion_stage] = (
                    promotion_stage_counts.get(promotion_stage, 0) + 1
                )
        candidate_rows = [item for item in top_candidates if isinstance(item, Mapping)]
        selected_rows = [
            item
            for item in candidate_rows
            if str(item.get("datapack_id", "")) in selected_ids
        ]
        non_selected_rows = [
            item
            for item in candidate_rows
            if str(item.get("datapack_id", "")) not in selected_ids
        ]
        if not selected_rows:
            skipped_runs += 1
            continue
        for selected in selected_rows:
            selected_features = _feature_map(selected.get("selection_features"))
            metrics = _metrics_map(row)
            examples.append(
                DatapackSelectionTrainingExample(
                    run_id=run_id,
                    datapack_id=str(selected.get("datapack_id", "")),
                    selected=True,
                    supervision_kind="selected_outcome_regression",
                    target_score=outcome_score,
                    outcome_score=outcome_score,
                    features=selected_features,
                    selection_context=selection_context,
                    feature_delta={},
                    metrics=metrics,
                )
            )
            if outcome_score >= 0.55 and non_selected_rows:
                positive_runs += 1
                for candidate in non_selected_rows:
                    candidate_features = _feature_map(
                        candidate.get("selection_features")
                    )
                    feature_delta = {
                        key: float(
                            selected_features.get(key, 0.0)
                            - candidate_features.get(key, 0.0)
                        )
                        for key in sorted(
                            set(selected_features) | set(candidate_features)
                        )
                    }
                    examples.append(
                        DatapackSelectionTrainingExample(
                            run_id=run_id,
                            datapack_id=str(candidate.get("datapack_id", "")),
                            selected=False,
                            supervision_kind="selected_vs_alternative_pairwise",
                            target_score=outcome_score,
                            outcome_score=outcome_score,
                            features=candidate_features,
                            selection_context=selection_context,
                            feature_delta=feature_delta,
                            metrics=metrics,
                        )
                    )
    summary = {
        "schema_version": "datapack_selection_training_summary_v1",
        "num_runs": len(run_rows),
        "num_examples": len(examples),
        "num_positive_pairwise_runs": positive_runs,
        "num_skipped_runs": skipped_runs,
        "supervision_mode": "selected_outcome_plus_positive_pairwise_v1",
        "selection_context_contract": {
            "schema_version": "datapack_selection_context_v1",
            "feature_names": sorted(DatapackSelectionContext().to_dict().keys()),
        },
        "selection_policy_counts": dict(sorted(policy_counts.items())),
        "promotion_stage_counts": dict(sorted(promotion_stage_counts.items())),
        "outcome_score_summary": {
            "min": float(min(outcome_scores)) if outcome_scores else 0.0,
            "max": float(max(outcome_scores)) if outcome_scores else 0.0,
            "mean": (
                float(sum(outcome_scores) / len(outcome_scores))
                if outcome_scores
                else 0.0
            ),
        },
        "dataset_digest": sha256_json([example.to_dict() for example in examples]),
        "benchmark_gate": {
            "ready": len(run_rows) >= DATAPACK_SELECTION_BENCHMARK_MIN_RUNS
            and positive_runs >= DATAPACK_SELECTION_BENCHMARK_MIN_PAIRWISE,
            "required_runs": DATAPACK_SELECTION_BENCHMARK_MIN_RUNS,
            "required_positive_pairwise_runs": DATAPACK_SELECTION_BENCHMARK_MIN_PAIRWISE,
            "observed_runs": len(run_rows),
            "observed_positive_pairwise_runs": positive_runs,
        },
    }
    return DatapackSelectionTrainingDataset(examples=examples, summary=summary)


def train_datapack_selection_scorer_package(
    dataset: DatapackSelectionTrainingDataset,
) -> DatapackSelectionScorerPackage:
    if not dataset.examples:
        raise ValueError("datapack selection training dataset is empty")
    feature_names = sorted(DatapackSelectionFeatures().to_dict().keys())
    context_feature_names = sorted(DatapackSelectionContext().to_dict().keys())
    regression_examples = [
        example
        for example in dataset.examples
        if example.supervision_kind == "selected_outcome_regression"
    ]
    pairwise_examples = [
        example
        for example in dataset.examples
        if example.supervision_kind == "selected_vs_alternative_pairwise"
    ]

    weights = {name: 0.0 for name in feature_names}
    bias = 0.0
    context_weights = {name: 0.0 for name in context_feature_names}
    context_bias = 0.0
    for example in regression_examples:
        target = _clamp01(example.target_score)
        centered_target = target - 0.5
        bias += centered_target
        for feature_name in feature_names:
            weights[feature_name] += centered_target * _safe_float(
                example.features.get(feature_name, 0.0)
            )
        adjustment_target = _clamp01((target - 0.35) / 0.65)
        centered_adjustment = adjustment_target - 0.5
        context_bias += centered_adjustment
        for feature_name in context_feature_names:
            context_weights[feature_name] += centered_adjustment * _safe_float(
                example.selection_context.get(feature_name, 0.0)
            )
    for example in pairwise_examples:
        target = _clamp01(example.target_score)
        centered_target = target - 0.5
        bias += centered_target * 0.25
        for feature_name in feature_names:
            weights[feature_name] += centered_target * _safe_float(
                example.feature_delta.get(feature_name, 0.0)
            )

    normalizer = max(
        1.0,
        max(abs(value) for value in weights.values()) if weights else 1.0,
        abs(bias),
    )
    normalized_weights = {
        feature_name: float(value / normalizer)
        for feature_name, value in weights.items()
    }
    normalized_bias = float(bias / normalizer)
    context_normalizer = max(
        1.0,
        max(abs(value) for value in context_weights.values())
        if context_weights
        else 1.0,
        abs(context_bias),
    )
    normalized_context_weights = {
        feature_name: float(value / context_normalizer)
        for feature_name, value in context_weights.items()
    }
    normalized_context_bias = float(context_bias / context_normalizer)
    support_factor = _clamp01(
        len(pairwise_examples)
        / float(max(DATAPACK_SELECTION_BENCHMARK_MIN_PAIRWISE, 1))
    )
    max_adjustment = 0.2 + (0.55 * support_factor)
    min_adjustment = min(max_adjustment, 0.05 + 0.1 * support_factor)
    model_kind = "linear_feature_weights_plus_context_conditioned_adjustment_v1"
    neural_feature_order: list[str] = []
    neural_hidden_weights: list[list[float]] = []
    neural_hidden_bias: list[float] = []
    neural_output_weights: list[float] = []
    neural_output_bias = 0.0
    neural_training_summary: dict[str, Any] = {
        "mode": "linear_fallback",
        "train_loss": 0.0,
        "pairwise_margin_accuracy": 0.0,
        "epochs": 0,
        "hidden_dim": 0,
    }

    if TORCH_AVAILABLE and regression_examples:
        (
            neural_feature_order,
            neural_hidden_weights,
            neural_hidden_bias,
            neural_output_weights,
            neural_output_bias,
            neural_training_summary,
        ) = _train_neural_feature_model(
            dataset.examples,
            feature_names=feature_names,
        )
        if neural_hidden_weights and neural_output_weights:
            model_kind = "neural_feature_mlp_with_context_conditioned_adjustment_v2"
            saliency = _derive_feature_saliency_from_network(
                feature_names=neural_feature_order,
                hidden_weights=neural_hidden_weights,
                output_weights=neural_output_weights,
            )
            for feature_name in feature_names:
                if feature_name in saliency:
                    normalized_weights[feature_name] = float(saliency[feature_name])

    return DatapackSelectionScorerPackage(
        package_id="datapack_selection_helper_v1",
        schema_version="datapack_selection_scorer_v1",
        feature_weights=normalized_weights,
        context_weights=normalized_context_weights,
        bias=normalized_bias,
        context_bias=normalized_context_bias,
        min_adjustment=min_adjustment,
        max_adjustment=max_adjustment,
        model_kind=model_kind,
        neural_feature_order=neural_feature_order,
        neural_hidden_weights=neural_hidden_weights,
        neural_hidden_bias=neural_hidden_bias,
        neural_output_weights=neural_output_weights,
        neural_output_bias=neural_output_bias,
        metadata={
            "supervision_mode": dataset.summary.get("supervision_mode"),
            "dataset_digest": dataset.summary.get("dataset_digest"),
            "benchmark_gate": dict(dataset.summary.get("benchmark_gate", {}) or {}),
            "num_examples": len(dataset.examples),
            "num_pairwise_examples": len(pairwise_examples),
            "num_regression_examples": len(regression_examples),
            "conditioning_contract": "datapack_selection_context_v1",
            "context_feature_names": sorted(context_feature_names),
            "model_kind": model_kind,
            "neural_training_summary": neural_training_summary,
            "future_conditioning_path": "economic_wm_then_meta_node_wm",
        },
    )


def write_datapack_selection_training_dataset(
    path: str | Path,
    dataset: DatapackSelectionTrainingDataset,
) -> str:
    output_path = Path(path)
    output_path.write_text(
        json.dumps(dataset.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return str(output_path.resolve())


def write_datapack_selection_scorer_package(
    path: str | Path,
    scorer_package: DatapackSelectionScorerPackage,
) -> str:
    output_path = Path(path)
    output_path.write_text(
        json.dumps(scorer_package.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return str(output_path.resolve())


def _feature_map(payload: Any) -> dict[str, float]:
    if not isinstance(payload, Mapping):
        return {}
    return {str(key): _safe_float(value, 0.0) for key, value in payload.items()}


def _selection_context_from_summary(payload: Mapping[str, Any]) -> dict[str, float]:
    context = payload.get("selection_context")
    if isinstance(context, Mapping):
        return {str(key): _safe_float(value, 0.0) for key, value in context.items()}
    required_tags = payload.get("required_tags")
    selected_gap_fill_tags = payload.get("selected_gap_fill_tags")
    candidate_count = _safe_float(payload.get("candidate_count", 0))
    return DatapackSelectionContext(
        required_tag_count_norm=_clamp01(len(list(required_tags or [])) / 8.0),
        gap_pressure=_clamp01(
            len(list(selected_gap_fill_tags or []))
            / float(max(len(list(required_tags or [])), 1))
        ),
        candidate_pool_size_norm=_clamp01(candidate_count / 10.0),
        objective_present=1.0 if payload.get("objective_hint") else 0.0,
        robot_specificity=1.0 if payload.get("robot_family") else 0.0,
    ).to_dict()


def _train_neural_feature_model(
    examples: Sequence[DatapackSelectionTrainingExample],
    *,
    feature_names: Sequence[str],
) -> tuple[
    list[str], list[list[float]], list[float], list[float], float, dict[str, Any]
]:
    if not TORCH_AVAILABLE:
        return (
            [],
            [],
            [],
            [],
            0.0,
            {
                "mode": "torch_unavailable",
                "train_loss": 0.0,
                "pairwise_margin_accuracy": 0.0,
                "epochs": 0,
                "hidden_dim": 0,
            },
        )

    feature_names = list(feature_names)
    rows = np.asarray(
        [
            [_safe_float(example.features.get(name, 0.0)) for name in feature_names]
            for example in examples
        ],
        dtype=np.float32,
    )
    if rows.size == 0:
        return (
            [],
            [],
            [],
            [],
            0.0,
            {
                "mode": "empty_feature_rows",
                "train_loss": 0.0,
                "pairwise_margin_accuracy": 0.0,
                "epochs": 0,
                "hidden_dim": 0,
            },
        )

    targets = np.asarray(
        [
            _clamp01(example.target_score) if example.selected else 0.0
            for example in examples
        ],
        dtype=np.float32,
    )
    weights = np.asarray(
        [
            1.0 if example.selected else 0.65 + (0.35 * _clamp01(example.outcome_score))
            for example in examples
        ],
        dtype=np.float32,
    )
    pair_indices = _pairwise_training_indices(examples)
    hidden_dim = min(24, max(8, len(feature_names) * 2))

    model = _DatapackSelectionFeatureMLP(
        input_dim=len(feature_names), hidden_dim=hidden_dim
    )
    optimizer = optim.Adam(model.parameters(), lr=5e-3)
    x_tensor = torch.from_numpy(rows)
    y_tensor = torch.from_numpy(targets)
    w_tensor = torch.from_numpy(weights)

    train_loss = 0.0
    for _epoch in range(120):
        optimizer.zero_grad()
        logits = model(x_tensor).squeeze(-1)
        regression_loss = (
            nn.functional.binary_cross_entropy_with_logits(
                logits, y_tensor, reduction="none"
            )
            * w_tensor
        ).mean()
        pairwise_loss = _pairwise_margin_loss(logits, pair_indices, examples)
        loss = regression_loss + (0.35 * pairwise_loss)
        loss.backward()
        optimizer.step()
        train_loss = float(loss.item())

    with torch.no_grad():
        logits = model(x_tensor).squeeze(-1)
        pairwise_margin_accuracy = _pairwise_margin_accuracy(logits, pair_indices)
        hidden_weights = model.hidden.weight.detach().cpu().numpy().astype(np.float32)
        hidden_bias = model.hidden.bias.detach().cpu().numpy().astype(np.float32)
        output_weights = (
            model.output.weight.detach().cpu().numpy().astype(np.float32)[0]
        )
        output_bias = float(
            model.output.bias.detach().cpu().numpy().astype(np.float32)[0]
        )

    return (
        list(feature_names),
        hidden_weights.tolist(),
        hidden_bias.tolist(),
        output_weights.tolist(),
        output_bias,
        {
            "mode": "neural_feature_mlp",
            "train_loss": train_loss,
            "pairwise_margin_accuracy": pairwise_margin_accuracy,
            "epochs": 120,
            "hidden_dim": hidden_dim,
        },
    )


def _derive_feature_saliency_from_network(
    *,
    feature_names: Sequence[str],
    hidden_weights: Sequence[Sequence[float]],
    output_weights: Sequence[float],
) -> dict[str, float]:
    if not hidden_weights or not output_weights:
        return {str(name): 0.0 for name in feature_names}
    hidden = np.asarray(hidden_weights, dtype=np.float32)
    output = np.asarray(output_weights, dtype=np.float32)
    local_linear = output @ hidden
    normalizer = float(max(np.max(np.abs(local_linear)), 1.0))
    return {
        str(name): float(local_linear[index] / normalizer)
        for index, name in enumerate(feature_names)
    }


def _pairwise_training_indices(
    examples: Sequence[DatapackSelectionTrainingExample],
) -> list[tuple[int, int, float]]:
    selected_by_run: dict[str, list[int]] = {}
    alternatives_by_run: dict[str, list[int]] = {}
    for index, example in enumerate(examples):
        if example.selected:
            selected_by_run.setdefault(str(example.run_id), []).append(index)
        elif example.supervision_kind == "selected_vs_alternative_pairwise":
            alternatives_by_run.setdefault(str(example.run_id), []).append(index)
    pairs: list[tuple[int, int, float]] = []
    for run_id, selected_indices in sorted(selected_by_run.items()):
        alternative_indices = alternatives_by_run.get(run_id, [])
        if not alternative_indices:
            continue
        outcome = max(
            _clamp01(examples[index].outcome_score) for index in selected_indices
        )
        margin = 0.15 + (0.35 * outcome)
        for selected_index in selected_indices:
            for alternative_index in alternative_indices:
                pairs.append((selected_index, alternative_index, margin))
    return pairs


def _pairwise_margin_loss(
    logits: Any,
    pair_indices: Sequence[tuple[int, int, float]],
    examples: Sequence[DatapackSelectionTrainingExample],
) -> Any:
    if not pair_indices:
        return logits.new_tensor(0.0)
    losses = []
    for selected_index, alternative_index, margin in pair_indices:
        losses.append(
            torch.relu(
                logits[alternative_index] - logits[selected_index] + float(margin)
            )
        )
    if not losses:
        return logits.new_tensor(0.0)
    return torch.stack(losses).mean()


def _pairwise_margin_accuracy(
    logits: Any,
    pair_indices: Sequence[tuple[int, int, float]],
) -> float:
    if not pair_indices:
        return 0.0
    correct = 0
    for selected_index, alternative_index, margin in pair_indices:
        if float(logits[selected_index] - logits[alternative_index]) >= float(margin):
            correct += 1
    return float(correct) / float(len(pair_indices))


class _DatapackSelectionFeatureMLP(nn.Module if TORCH_AVAILABLE else object):  # type: ignore[misc]
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, x: Any) -> Any:
        return self.output(torch.relu(self.hidden(x)))


def _metrics_map(row: Mapping[str, Any]) -> dict[str, float]:
    metrics = {}
    for key in ("train_metrics", "eval_metrics"):
        payload = row.get(key)
        if isinstance(payload, Mapping):
            for metric_name, value in payload.items():
                metrics[str(metric_name)] = _safe_float(value, 0.0)
    return metrics


def _run_outcome_score(row: Mapping[str, Any]) -> float:
    metrics = (
        row.get("eval_metrics")
        if isinstance(row.get("eval_metrics"), Mapping)
        else row.get("train_metrics")
    )
    if not isinstance(metrics, Mapping):
        return 0.0
    mpl_norm = _clamp01(_safe_float(metrics.get("mpl_units_per_hour", 0.0)) / 100.0)
    wage_norm = _clamp01(_safe_float(metrics.get("wage_parity", 0.0)))
    error_penalty = _clamp01(_safe_float(metrics.get("error_rate", 0.0)))
    arh_penalty = _clamp01(
        max(
            _safe_float(metrics.get("anti_reward_hacking_suspicious", 0.0)),
            _safe_float(metrics.get("arh_excluded", 0.0)),
        )
    )
    reward_norm = _clamp01(
        (_safe_float(metrics.get("reward_scalar_sum", 0.0)) + 10.0) / 20.0
    )
    return _clamp01(
        0.5 * mpl_norm
        + 0.15 * wage_norm
        + 0.15 * reward_norm
        - 0.25 * error_penalty
        - 0.25 * arh_penalty
        + 0.2
    )


__all__ = [
    "DATAPACK_SELECTION_BENCHMARK_MIN_PAIRWISE",
    "DATAPACK_SELECTION_BENCHMARK_MIN_RUNS",
    "DatapackSelectionTrainingDataset",
    "DatapackSelectionTrainingExample",
    "build_datapack_selection_training_dataset",
    "load_selection_run_logs",
    "train_datapack_selection_scorer_package",
    "write_datapack_selection_scorer_package",
    "write_datapack_selection_training_dataset",
]
