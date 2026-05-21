"""Local Economic WM replay feature and training-row materialization.

These rows are CPU/local scaffold artifacts. They are not evidence that an
Economic WM was trained, promoted, or allowed to mutate reward math.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

from src.utils.config_digest import sha256_json
from src.utils.json_safe import to_json_safe
from src.world_model.economic_world_model.scaffold import (
    EconomicWMScaffoldReport,
    load_economic_wm_scaffold_report,
)

ECONOMIC_WM_REPLAY_FEATURE_ROW_VERSION = "economic_wm_replay_feature_row_v1"
ECONOMIC_WM_TRAINING_CORPUS_MANIFEST_VERSION = "economic_wm_training_corpus_manifest_v1"


def _mapping(payload: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    return dict(to_json_safe(dict(payload or {})))


def _float_dict(payload: Mapping[str, Any]) -> Dict[str, float]:
    values: Dict[str, float] = {}
    for key, value in dict(payload or {}).items():
        try:
            values[str(key)] = float(value)
        except Exception:
            continue
    return values


def _bool(payload: Mapping[str, Any], key: str) -> bool:
    return bool(payload.get(key, False))


def _load_jsonl(path: str | Path) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


@dataclass(frozen=True)
class EconomicWMReplayFeatureRow:
    """One local training-row surface for future Economic WM learning."""

    row_id: str
    source_episode_id: str
    video_id: str
    proposal_id: str
    readiness_regime: str
    benchmark_ready: bool
    shadow_only: bool
    local_materialization_eligible: bool
    gpu_training_eligible: bool
    feature_vector: Dict[str, float] = field(default_factory=dict)
    target_vector: Dict[str, float] = field(default_factory=dict)
    denied_promotion_reasons: list[str] = field(default_factory=list)
    source_refs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = ECONOMIC_WM_REPLAY_FEATURE_ROW_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "row_id": self.row_id,
            "version": self.version,
            "source_episode_id": self.source_episode_id,
            "video_id": self.video_id,
            "proposal_id": self.proposal_id,
            "readiness_regime": self.readiness_regime,
            "benchmark_ready": bool(self.benchmark_ready),
            "shadow_only": bool(self.shadow_only),
            "local_materialization_eligible": bool(self.local_materialization_eligible),
            "gpu_training_eligible": bool(self.gpu_training_eligible),
            "feature_vector": _float_dict(self.feature_vector),
            "target_vector": _float_dict(self.target_vector),
            "denied_promotion_reasons": list(self.denied_promotion_reasons),
            "source_refs": _mapping(self.source_refs),
            "metadata": _mapping(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EconomicWMReplayFeatureRow":
        return cls(
            row_id=str(payload.get("row_id", "")),
            source_episode_id=str(payload.get("source_episode_id", "")),
            video_id=str(payload.get("video_id", "")),
            proposal_id=str(payload.get("proposal_id", "")),
            readiness_regime=str(payload.get("readiness_regime", "blocked")),
            benchmark_ready=bool(payload.get("benchmark_ready", False)),
            shadow_only=bool(payload.get("shadow_only", False)),
            local_materialization_eligible=bool(
                payload.get("local_materialization_eligible", False)
            ),
            gpu_training_eligible=bool(payload.get("gpu_training_eligible", False)),
            feature_vector=_float_dict(payload.get("feature_vector", {})),
            target_vector=_float_dict(payload.get("target_vector", {})),
            denied_promotion_reasons=[
                str(item)
                for item in list(payload.get("denied_promotion_reasons", []) or [])
            ],
            source_refs=_mapping(payload.get("source_refs")),
            metadata=_mapping(payload.get("metadata")),
            version=str(payload.get("version", ECONOMIC_WM_REPLAY_FEATURE_ROW_VERSION)),
        )


@dataclass(frozen=True)
class EconomicWMTrainingCorpusManifest:
    """Manifest for a local scaffold-only Economic WM row corpus."""

    corpus_id: str
    scaffold_id: str
    row_count: int
    benchmark_ready_count: int
    shadow_only_count: int
    rows_path: str
    readiness_class: str
    ready_for_training: bool
    promotion_eligible: bool
    training_blockers: list[str] = field(default_factory=list)
    row_ids: list[str] = field(default_factory=list)
    artifact_refs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = ECONOMIC_WM_TRAINING_CORPUS_MANIFEST_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "corpus_id": self.corpus_id,
            "version": self.version,
            "scaffold_id": self.scaffold_id,
            "row_count": int(self.row_count),
            "benchmark_ready_count": int(self.benchmark_ready_count),
            "shadow_only_count": int(self.shadow_only_count),
            "rows_path": self.rows_path,
            "readiness_class": self.readiness_class,
            "ready_for_training": bool(self.ready_for_training),
            "promotion_eligible": bool(self.promotion_eligible),
            "training_blockers": list(self.training_blockers),
            "row_ids": list(self.row_ids),
            "artifact_refs": _mapping(self.artifact_refs),
            "metadata": _mapping(self.metadata),
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "EconomicWMTrainingCorpusManifest":
        return cls(
            corpus_id=str(payload.get("corpus_id", "")),
            scaffold_id=str(payload.get("scaffold_id", "")),
            row_count=int(payload.get("row_count", 0) or 0),
            benchmark_ready_count=int(payload.get("benchmark_ready_count", 0) or 0),
            shadow_only_count=int(payload.get("shadow_only_count", 0) or 0),
            rows_path=str(payload.get("rows_path", "")),
            readiness_class=str(payload.get("readiness_class", "blocked")),
            ready_for_training=bool(payload.get("ready_for_training", False)),
            promotion_eligible=bool(payload.get("promotion_eligible", False)),
            training_blockers=[
                str(item) for item in list(payload.get("training_blockers", []) or [])
            ],
            row_ids=[str(item) for item in list(payload.get("row_ids", []) or [])],
            artifact_refs=_mapping(payload.get("artifact_refs")),
            metadata=_mapping(payload.get("metadata")),
            version=str(
                payload.get("version", ECONOMIC_WM_TRAINING_CORPUS_MANIFEST_VERSION)
            ),
        )


def build_economic_wm_replay_feature_row(
    *,
    scaffold_report: EconomicWMScaffoldReport,
    admission_row: Mapping[str, Any],
) -> EconomicWMReplayFeatureRow:
    """Derive one scaffold-only Economic WM row from a Stage-1 admission row."""

    row = _mapping(admission_row)
    signals = _mapping(row.get("future_training_signals"))
    benchmark_gate = _mapping(row.get("benchmark_gate"))
    gate_ready = bool(
        benchmark_gate.get("ready", signals.get("benchmark_gate_ready", False))
    )
    blocked = bool(row.get("blocked", False))
    benchmark_ready = bool(gate_ready and not blocked)
    shadow_only = not benchmark_ready
    training_blockers = list(scaffold_report.training_blockers)
    replay_export_flow = float(
        scaffold_report.economic_state.flow_fields.get("replay_export_flow", 0.0)
    )
    benchmark_flow = float(
        scaffold_report.economic_state.flow_fields.get("benchmark_ready_flow", 0.0)
    )
    shadow_flow = float(
        scaffold_report.economic_state.flow_fields.get("shadow_data_flow", 0.0)
    )
    provider_friction = float(
        scaffold_report.economic_state.dissipation_fields.get("provider_friction", 0.0)
    )
    gpu_training_friction = float(
        scaffold_report.economic_state.dissipation_fields.get(
            "gpu_training_friction", 0.0
        )
    )

    feature_vector = {
        "benchmark_gate_ready": 1.0 if gate_ready else 0.0,
        "blocked": 1.0 if blocked else 0.0,
        "reconstruction_calibrated": 1.0
        if _bool(signals, "reconstruction_calibrated")
        else 0.0,
        "reconstruction_real_grounded": 1.0
        if _bool(signals, "reconstruction_real_grounded")
        else 0.0,
        "reconstruction_training_eligible": 1.0
        if _bool(signals, "reconstruction_training_eligible")
        else 0.0,
        "scene_tracks_backend_real": 1.0
        if _bool(signals, "scene_tracks_backend_real")
        else 0.0,
        "scene_tracks_non_stub": 1.0
        if _bool(signals, "scene_tracks_non_stub")
        else 0.0,
        "semantic_grounding_non_heuristic": 1.0
        if _bool(signals, "semantic_grounding_non_heuristic")
        else 0.0,
        "semantic_memory_grounded": 1.0
        if _bool(signals, "semantic_memory_grounded")
        else 0.0,
        "teacher_runtime_contract_complete": 1.0
        if _bool(signals, "teacher_runtime_contract_complete")
        else 0.0,
        "teacher_runtime_real": 1.0 if _bool(signals, "teacher_runtime_real") else 0.0,
        "vision_backbone_real": 1.0 if _bool(signals, "vision_backbone_real") else 0.0,
        "replay_export_flow": replay_export_flow,
        "benchmark_ready_flow": benchmark_flow,
        "shadow_data_flow": shadow_flow,
        "provider_friction": provider_friction,
        "gpu_training_friction": gpu_training_friction,
    }
    target_vector = {
        "benchmark_training_weight": 1.0 if benchmark_ready else 0.0,
        "shadow_gap_weight": 1.0 if shadow_only else 0.0,
        "reconstruction_training_weight": feature_vector[
            "reconstruction_training_eligible"
        ],
        "teacher_runtime_gap_weight": 0.0
        if feature_vector["teacher_runtime_real"] > 0.0
        else 1.0,
        "provider_bringup_gap_weight": provider_friction,
        "gpu_training_deferred_weight": gpu_training_friction,
    }
    denial_reasons = list(training_blockers)
    if shadow_only:
        denial_reasons.extend(
            str(item)
            for item in list(benchmark_gate.get("blocking_preconditions", []) or [])
        )
    if feature_vector["teacher_runtime_real"] == 0.0:
        denial_reasons.append("teacher_runtime_real_missing")
    if feature_vector["replay_export_flow"] < 1.0:
        denial_reasons.append("replay_export_flow_incomplete")

    source_episode_id = f"{row.get('video_id', '')}:{row.get('proposal_id', '')}"
    base = {
        "scaffold_id": scaffold_report.scaffold_id,
        "source_episode_id": source_episode_id,
        "feature_vector": feature_vector,
        "target_vector": target_vector,
        "version": ECONOMIC_WM_REPLAY_FEATURE_ROW_VERSION,
    }
    return EconomicWMReplayFeatureRow(
        row_id=f"ewm_row_{sha256_json(base)[:16]}",
        source_episode_id=source_episode_id,
        video_id=str(row.get("video_id", "")),
        proposal_id=str(row.get("proposal_id", "")),
        readiness_regime=scaffold_report.economic_state.regime,
        benchmark_ready=benchmark_ready,
        shadow_only=shadow_only,
        local_materialization_eligible=bool(scaffold_report.ready_for_scaffold),
        gpu_training_eligible=False,
        feature_vector=feature_vector,
        target_vector=target_vector,
        denied_promotion_reasons=sorted(set(denial_reasons)),
        source_refs={
            "runtime_packet_path": row.get("runtime_packet_path", ""),
            "counterfactual_eval_path": row.get("counterfactual_eval_path", ""),
            "value_target_pack_path": row.get("value_target_pack_path", ""),
            "value_ledger_receipt_path": row.get("value_ledger_receipt_path", ""),
            "governance_trace_path": row.get("governance_trace_path", ""),
            "benchmark_gate_path": row.get("benchmark_gate_path", ""),
            "reconstruction_grounding_report_path": row.get(
                "reconstruction_grounding_report_path", ""
            ),
            "teacher_contract_path": row.get("teacher_contract_path", ""),
            "teacher_trace_path": row.get("teacher_trace_path", ""),
            "canonical_lower_wm_reference_pack_path": row.get(
                "canonical_lower_wm_reference_pack_path", ""
            ),
            "perception_grounding_world_state_path": row.get(
                "perception_grounding_world_state_path", ""
            ),
            "sim_synth_physics_world_state_path": row.get(
                "sim_synth_physics_world_state_path", ""
            ),
            "embodiment_actuation_world_state_path": row.get(
                "embodiment_actuation_world_state_path", ""
            ),
            "canonical_lower_wm_refs": _mapping(
                row.get("canonical_lower_wm_refs")
                or _mapping(row.get("future_training_artifacts")).get(
                    "canonical_lower_wm_refs"
                )
            ),
        },
        metadata={
            "boundary": "local scaffold row only; no GPU training or promotion claim",
            "diffusion_provider_truth": row.get("diffusion_provider_truth", ""),
            "diffusion_backend_selected": row.get("diffusion_backend_selected", ""),
            "routing_score": row.get("routing_score", 0.0),
        },
    )


def build_economic_wm_training_corpus_manifest(
    *,
    scaffold_report: EconomicWMScaffoldReport,
    admission_rows: Iterable[Mapping[str, Any]],
    rows_path: str | Path,
    artifact_refs: Optional[Mapping[str, Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> tuple[EconomicWMTrainingCorpusManifest, list[EconomicWMReplayFeatureRow]]:
    rows = [
        build_economic_wm_replay_feature_row(
            scaffold_report=scaffold_report,
            admission_row=row,
        )
        for row in admission_rows
    ]
    benchmark_count = sum(1 for row in rows if row.benchmark_ready)
    shadow_count = sum(1 for row in rows if row.shadow_only)
    payload = {
        "scaffold_id": scaffold_report.scaffold_id,
        "row_ids": [row.row_id for row in rows],
        "row_count": len(rows),
        "benchmark_ready_count": benchmark_count,
        "shadow_only_count": shadow_count,
        "rows_path": str(rows_path),
        "version": ECONOMIC_WM_TRAINING_CORPUS_MANIFEST_VERSION,
    }
    manifest = EconomicWMTrainingCorpusManifest(
        corpus_id=f"ewm_corpus_{sha256_json(payload)[:16]}",
        scaffold_id=scaffold_report.scaffold_id,
        row_count=len(rows),
        benchmark_ready_count=benchmark_count,
        shadow_only_count=shadow_count,
        rows_path=str(rows_path),
        readiness_class=scaffold_report.economic_state.regime,
        ready_for_training=False,
        promotion_eligible=False,
        training_blockers=list(scaffold_report.training_blockers),
        row_ids=[row.row_id for row in rows],
        artifact_refs={
            "economic_wm_scaffold_report_id": scaffold_report.scaffold_id,
            **_mapping(artifact_refs),
        },
        metadata={
            "boundary": "local materialized rows only; trainer remains GPU/provider/evidence gated",
            "training_claim": False,
            **_mapping(metadata),
        },
    )
    return manifest, rows


def save_economic_wm_training_corpus(
    *,
    manifest_path: str | Path,
    rows_path: str | Path,
    manifest: EconomicWMTrainingCorpusManifest,
    rows: Iterable[EconomicWMReplayFeatureRow],
) -> None:
    rows_target = Path(rows_path)
    rows_target.parent.mkdir(parents=True, exist_ok=True)
    rows_target.write_text(
        "\n".join(json.dumps(row.to_dict(), sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    manifest_target = Path(manifest_path)
    manifest_target.parent.mkdir(parents=True, exist_ok=True)
    manifest_target.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )


def load_economic_wm_training_corpus_manifest(
    path: str | Path,
) -> EconomicWMTrainingCorpusManifest:
    return EconomicWMTrainingCorpusManifest.from_dict(
        json.loads(Path(path).read_text(encoding="utf-8"))
    )


def load_economic_wm_replay_feature_rows(
    path: str | Path,
) -> list[EconomicWMReplayFeatureRow]:
    return [EconomicWMReplayFeatureRow.from_dict(row) for row in _load_jsonl(path)]


def materialize_economic_wm_training_corpus_from_paths(
    *,
    scaffold_report_path: str | Path,
    admission_log_path: str | Path,
    rows_path: str | Path,
    manifest_path: str | Path,
    metadata: Optional[Mapping[str, Any]] = None,
) -> EconomicWMTrainingCorpusManifest:
    scaffold_report = load_economic_wm_scaffold_report(scaffold_report_path)
    admission_rows = _load_jsonl(admission_log_path)
    manifest, rows = build_economic_wm_training_corpus_manifest(
        scaffold_report=scaffold_report,
        admission_rows=admission_rows,
        rows_path=rows_path,
        artifact_refs={
            "scaffold_report_path": str(scaffold_report_path),
            "admission_log_path": str(admission_log_path),
        },
        metadata=metadata,
    )
    save_economic_wm_training_corpus(
        manifest_path=manifest_path,
        rows_path=rows_path,
        manifest=manifest,
        rows=rows,
    )
    return manifest


__all__ = [
    "ECONOMIC_WM_REPLAY_FEATURE_ROW_VERSION",
    "ECONOMIC_WM_TRAINING_CORPUS_MANIFEST_VERSION",
    "EconomicWMReplayFeatureRow",
    "EconomicWMTrainingCorpusManifest",
    "build_economic_wm_replay_feature_row",
    "build_economic_wm_training_corpus_manifest",
    "load_economic_wm_replay_feature_rows",
    "load_economic_wm_training_corpus_manifest",
    "materialize_economic_wm_training_corpus_from_paths",
    "save_economic_wm_training_corpus",
]
