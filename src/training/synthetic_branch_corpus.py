"""Synthetic-branch corpus metadata, readiness, and training-policy helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np

from src.evidence.benchmark_gating import (
    build_benchmark_gate_report,
    collect_benchmark_gating_signals,
)
from src.evidence.preconditions import ExecutionPreconditionsReport, build_execution_preconditions
from src.utils.json_safe import to_json_safe


def _mapping(payload: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    return dict(to_json_safe(dict(payload or {})))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _clamp(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def _json_object(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, Mapping) else {}


def _json_rows(path: Optional[Path]) -> list[Dict[str, Any]]:
    if path is None or not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [dict(row) for row in list(payload or []) if isinstance(row, Mapping)]


@dataclass(frozen=True)
class SyntheticBranchRecord:
    """Metadata envelope for one synthetic branch."""

    branch_idx: int
    source_episode: int
    source_timestep: int
    trust_score: float
    std_ratio: float
    brick_id: int
    branch_value: float
    objective_vector: list[float] = field(default_factory=list)
    gap_labels: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "branch_idx": int(self.branch_idx),
            "source_episode": int(self.source_episode),
            "source_timestep": int(self.source_timestep),
            "trust_score": float(self.trust_score),
            "std_ratio": float(self.std_ratio),
            "brick_id": int(self.brick_id),
            "branch_value": float(self.branch_value),
            "objective_vector": list(self.objective_vector),
            "gap_labels": _mapping(self.gap_labels),
        }


@dataclass(frozen=True)
class SyntheticBranchCorpus:
    """Loaded synthetic-branch corpus plus readiness metadata."""

    npz_path: str
    metadata_path: Optional[str]
    gap_labels_path: Optional[str]
    metadata: Dict[str, Any]
    branches: list[SyntheticBranchRecord]
    summary: Dict[str, Any]
    execution_preconditions: ExecutionPreconditionsReport
    benchmark_gate: ExecutionPreconditionsReport

    def to_summary(self) -> Dict[str, Any]:
        return {
            "npz_path": self.npz_path,
            "metadata_path": self.metadata_path,
            "gap_labels_path": self.gap_labels_path,
            "summary": dict(self.summary),
            "execution_preconditions": self.execution_preconditions.to_dict(),
            "benchmark_gate": self.benchmark_gate.to_dict(),
            "sample_branches": [branch.to_dict() for branch in self.branches[:5]],
        }


def _resolve_sidecar_path(npz_path: Path, explicit: Optional[str | Path], suffix: str) -> Optional[Path]:
    if explicit is not None:
        return Path(explicit)
    candidate = npz_path.with_name(f"{npz_path.stem}{suffix}")
    return candidate if candidate.exists() else None


def _branch_gap_map(rows: list[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    mapping: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        branch_idx = _safe_int(row.get("branch_idx"), default=-1)
        if branch_idx < 0:
            continue
        mapping[branch_idx] = dict(row)
    return mapping


def _source_metadata_fields(metadata: Mapping[str, Any]) -> Dict[str, Any]:
    future_signals = metadata.get("future_training_signals", {})
    if not isinstance(future_signals, Mapping):
        future_signals = {}
    return {
        "scene_tracks_backend": str(metadata.get("scene_tracks_backend", "") or ""),
        "teacher_runtime_backend_selected": str(
            metadata.get("teacher_runtime_backend_selected")
            or metadata.get("openvla_backend_selected")
            or ""
        ),
        "vision_backbone_selected": str(metadata.get("vision_backbone_selected", "") or ""),
        "semantic_grounding_mode": str(metadata.get("semantic_grounding_mode", "") or ""),
        "semantic_memory_grounded": bool(
            metadata.get("semantic_memory_grounded", False)
            or future_signals.get("semantic_memory_grounded", False)
        ),
    }


def _summarize_corpus(
    *,
    npz_path: Path,
    metadata_path: Optional[Path],
    gap_labels_path: Optional[Path],
    metadata: Mapping[str, Any],
    branches: list[SyntheticBranchRecord],
) -> Dict[str, Any]:
    branch_values = [float(branch.branch_value) for branch in branches]
    trusts = [float(branch.trust_score) for branch in branches]
    std_ratios = [float(branch.std_ratio) for branch in branches]
    gap_scores = [
        _safe_float(branch.gap_labels.get("coverage_gap_contribution"), 0.0)
        for branch in branches
    ]
    econ_priorities = [
        _safe_float(branch.gap_labels.get("economic_priority"), 0.0)
        for branch in branches
    ]
    has_gap_labels = any(bool(branch.gap_labels) for branch in branches)
    metadata_fields = _source_metadata_fields(metadata)
    benchmark_signals = collect_benchmark_gating_signals(
        {
            **metadata_fields,
            "semantic_memory_grounded": metadata_fields["semantic_memory_grounded"],
            "semantic_grounding_mode": metadata_fields["semantic_grounding_mode"],
            "semantic_grounding_heuristic": metadata_fields["semantic_grounding_mode"] in {
                "heuristic",
                "heuristic_fallback",
                "keyword_tags",
                "unavailable",
            },
        }
    )
    return {
        "schema_version": str(metadata.get("schema_version", "") or ""),
        "source_type": str(metadata.get("source_type", "stable_world_model_local_branch_v1")),
        "npz_path": str(npz_path),
        "metadata_path": str(metadata_path) if metadata_path else None,
        "gap_labels_path": str(gap_labels_path) if gap_labels_path else None,
        "branch_count": len(branches),
        "metadata_present": bool(metadata_path and metadata_path.exists()),
        "gap_labels_present": bool(gap_labels_path and gap_labels_path.exists()),
        "coverage_graph_used": bool(metadata.get("coverage_graph_used", False)),
        "semantic_gap_labeled": bool(has_gap_labels),
        "branch_value_present": any(abs(value) > 1e-6 for value in branch_values),
        "avg_trust_score": float(np.mean(trusts)) if trusts else 0.0,
        "avg_std_ratio": float(np.mean(std_ratios)) if std_ratios else 0.0,
        "avg_branch_value": float(np.mean(branch_values)) if branch_values else 0.0,
        "avg_coverage_gap_contribution": float(np.mean(gap_scores)) if gap_scores else 0.0,
        "avg_economic_priority": float(np.mean(econ_priorities)) if econ_priorities else 0.0,
        "max_branch_value": float(max(branch_values)) if branch_values else 0.0,
        "source_metadata": metadata_fields,
        "future_training_signals": {
            **{
                "scene_tracks_non_stub": metadata_fields["scene_tracks_backend"] in {"real", "passthrough"},
                "semantic_gap_labeled": bool(has_gap_labels),
            },
            **{
                str(key): bool(value)
                for key, value in dict(metadata.get("future_training_signals", {}) or {}).items()
            },
        },
        "benchmark_signals": benchmark_signals,
    }


def _build_execution_preconditions(
    *,
    corpus_id: str,
    summary: Mapping[str, Any],
) -> ExecutionPreconditionsReport:
    benchmark_signals = dict(summary.get("benchmark_signals", {}) or {})
    return build_execution_preconditions(
        subject_id=corpus_id,
        subject_kind="synthetic_branch_corpus",
        artifact_refs={
            "branch_corpus_npz": summary.get("npz_path"),
            "branch_corpus_metadata": summary.get("metadata_path"),
            "branch_gap_labels": summary.get("gap_labels_path"),
        },
        required_artifact_refs=["branch_corpus_npz"],
        soft_required_artifact_refs=["branch_corpus_metadata", "branch_gap_labels"],
        signal_values={
            "branch_count": int(summary.get("branch_count", 0)),
            "avg_trust_score": _safe_float(summary.get("avg_trust_score"), 0.0),
            "avg_branch_value": _safe_float(summary.get("avg_branch_value"), 0.0),
            "coverage_graph_used": bool(summary.get("coverage_graph_used", False)),
            "semantic_gap_labeled": bool(summary.get("semantic_gap_labeled", False)),
            "branch_value_present": bool(summary.get("branch_value_present", False)),
            "metadata_present": bool(summary.get("metadata_present", False)),
            "gap_labels_present": bool(summary.get("gap_labels_present", False)),
            **benchmark_signals,
        },
        min_signal_thresholds={"branch_count": 1.0},
        soft_boolean_signals={
            "metadata_present": True,
            "gap_labels_present": True,
            "coverage_graph_used": True,
            "semantic_gap_labeled": True,
            "branch_value_present": True,
            "semantic_grounding_non_heuristic": True,
            "benchmark_eligible": True,
        },
        metadata={"source_type": summary.get("source_type", "stable_world_model_local_branch_v1")},
    )


def load_synthetic_branch_corpus(
    npz_path: str | Path,
    *,
    metadata_path: Optional[str | Path] = None,
    gap_labels_path: Optional[str | Path] = None,
) -> SyntheticBranchCorpus:
    """Load branch metadata, sidecars, and readiness summaries from a corpus."""

    corpus_path = Path(npz_path)
    resolved_metadata_path = _resolve_sidecar_path(corpus_path, metadata_path, "_metadata.json")
    resolved_gap_labels_path = _resolve_sidecar_path(corpus_path, gap_labels_path, "_gap_labels.json")

    metadata = _json_object(resolved_metadata_path)
    gap_rows = _json_rows(resolved_gap_labels_path)
    gap_map = _branch_gap_map(gap_rows)

    branches: list[SyntheticBranchRecord] = []
    with np.load(corpus_path, allow_pickle=True) as data:
        branch_count = _safe_int(data.get("n_branches"), 0)
        objective_dim = _safe_int(data.get("objective_dim"), 4)
        for branch_idx in range(branch_count):
            objective_key = f"branch_{branch_idx}_objective_vector"
            branch_value_key = f"branch_{branch_idx}_branch_value"
            objective_vector = (
                data[objective_key].tolist()
                if objective_key in data
                else [0.0 for _ in range(objective_dim)]
            )
            branches.append(
                SyntheticBranchRecord(
                    branch_idx=branch_idx,
                    source_episode=_safe_int(data.get(f"branch_{branch_idx}_source_episode"), 0),
                    source_timestep=_safe_int(data.get(f"branch_{branch_idx}_source_timestep"), 0),
                    trust_score=_safe_float(data.get(f"branch_{branch_idx}_trust_score"), 0.0),
                    std_ratio=_safe_float(data.get(f"branch_{branch_idx}_std_ratio"), 0.0),
                    brick_id=_safe_int(data.get(f"branch_{branch_idx}_brick_id"), -1),
                    branch_value=_safe_float(
                        data.get(branch_value_key),
                        _safe_float(data.get(f"branch_{branch_idx}_trust_score"), 0.0),
                    ),
                    objective_vector=[_safe_float(value, 0.0) for value in objective_vector],
                    gap_labels=dict(gap_map.get(branch_idx, {})),
                )
            )

    summary = _summarize_corpus(
        npz_path=corpus_path,
        metadata_path=resolved_metadata_path,
        gap_labels_path=resolved_gap_labels_path,
        metadata=metadata,
        branches=branches,
    )
    execution_preconditions = _build_execution_preconditions(
        corpus_id=corpus_path.stem,
        summary=summary,
    )
    benchmark_gate = build_benchmark_gate_report(
        subject_id=corpus_path.stem,
        subject_kind="synthetic_branch_corpus",
        metadata={
            **dict(summary.get("source_metadata", {}) or {}),
            **dict(summary.get("benchmark_signals", {}) or {}),
            "semantic_memory_grounded": bool(
                dict(summary.get("benchmark_signals", {}) or {}).get("semantic_memory_grounded", False)
            ),
        },
        require_real_scene_tracks=True,
        require_teacher_runtime=False,
        require_vision_backbone=True,
    )
    return SyntheticBranchCorpus(
        npz_path=str(corpus_path),
        metadata_path=str(resolved_metadata_path) if resolved_metadata_path else None,
        gap_labels_path=str(resolved_gap_labels_path) if resolved_gap_labels_path else None,
        metadata=metadata,
        branches=branches,
        summary=summary,
        execution_preconditions=execution_preconditions,
        benchmark_gate=benchmark_gate,
    )


def build_synthetic_branch_training_policy(
    corpus: SyntheticBranchCorpus,
    *,
    requested_synth_share: float,
    requested_econ_weight_scale: float,
) -> Dict[str, Any]:
    """Compile explicit bounded training policy from corpus readiness and labels."""

    summary = dict(corpus.summary)
    benchmark_ready = bool(corpus.benchmark_gate.ready)
    metadata_present = bool(summary.get("metadata_present", False))
    semantic_gap_labeled = bool(summary.get("semantic_gap_labeled", False))
    non_heuristic = bool(
        dict(summary.get("benchmark_signals", {}) or {}).get("semantic_grounding_non_heuristic", False)
    )

    effective_synth_share_cap = _clamp(requested_synth_share, 0.0, 1.0)
    semantic_weight_scale = 1.0
    reasons: list[str] = []

    if not metadata_present:
        effective_synth_share_cap = min(effective_synth_share_cap, 0.1)
        semantic_weight_scale *= 0.6
        reasons.append("branch_metadata_missing")
    if not semantic_gap_labeled:
        effective_synth_share_cap = min(effective_synth_share_cap, 0.15)
        semantic_weight_scale *= 0.75
        reasons.append("semantic_gap_labels_missing")
    if not non_heuristic:
        effective_synth_share_cap = min(effective_synth_share_cap, 0.15)
        semantic_weight_scale *= 0.7
        reasons.append("semantic_grounding_not_non_heuristic")
    elif not benchmark_ready:
        effective_synth_share_cap = min(effective_synth_share_cap, 0.2)
        semantic_weight_scale *= 0.85
        reasons.append("benchmark_gate_not_ready")

    avg_branch_value = _safe_float(summary.get("avg_branch_value"), 0.0)
    gap_value_scale = _clamp(1.0 + max(0.0, avg_branch_value), 1.0, 1.5)
    if avg_branch_value <= 0.0:
        reasons.append("branch_values_non_positive")

    return {
        "requested_synth_share": float(requested_synth_share),
        "requested_econ_weight_scale": float(requested_econ_weight_scale),
        "effective_synth_share_cap": float(_clamp(effective_synth_share_cap, 0.0, 1.0)),
        "semantic_weight_scale": float(_clamp(semantic_weight_scale, 0.25, 1.0)),
        "gap_value_scale": float(gap_value_scale),
        "branch_priority_floor": 0.25,
        "branch_priority_ceiling": 2.0,
        "benchmark_gate_ready": benchmark_ready,
        "execution_ready": bool(corpus.execution_preconditions.ready),
        "reasons": reasons or ["requested_policy_accepted"],
    }


def branch_priority_multiplier(
    branch: SyntheticBranchRecord,
    policy: Mapping[str, Any],
) -> float:
    """Convert branch/gap metadata into bounded training influence."""

    gap_score = _safe_float(branch.gap_labels.get("coverage_gap_contribution"), 0.0)
    econ_priority = _safe_float(branch.gap_labels.get("economic_priority"), 0.0)
    base_value = max(0.0, float(branch.branch_value))
    multiplier = (
        0.75
        + min(1.25, base_value)
        + 0.25 * max(0.0, gap_score)
        + 0.15 * max(0.0, econ_priority)
    )
    multiplier *= _safe_float(policy.get("semantic_weight_scale"), 1.0)
    multiplier *= _safe_float(policy.get("gap_value_scale"), 1.0)
    return _clamp(
        multiplier,
        _safe_float(policy.get("branch_priority_floor"), 0.25),
        _safe_float(policy.get("branch_priority_ceiling"), 2.0),
    )


__all__ = [
    "SyntheticBranchCorpus",
    "SyntheticBranchRecord",
    "branch_priority_multiplier",
    "build_synthetic_branch_training_policy",
    "load_synthetic_branch_corpus",
]
