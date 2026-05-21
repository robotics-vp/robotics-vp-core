"""Canonical lower-WM consumption preflight for Economic WM rows.

This module proves Economic WM rows can point at canonical Perception /
Grounding, Sim / Synth / Physics, and Embodiment / Actuation state artifacts
instead of relying only on summary sidecars. It may compile local canonical
reference packs when the source rows do not yet carry direct lower-WM state
paths. That compilation is structural preflight only: no training, provider
bring-up, or promotion is claimed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

from src.utils.config_digest import sha256_json
from src.utils.json_safe import to_json_safe
from src.world_model.economic_world_model.training_rows import (
    EconomicWMReplayFeatureRow,
    EconomicWMTrainingCorpusManifest,
    load_economic_wm_replay_feature_rows,
    load_economic_wm_training_corpus_manifest,
)

ECONOMIC_WM_LOWER_WM_REFERENCE_VERSION = "economic_wm_lower_wm_reference_v1"
ECONOMIC_WM_CANONICAL_CONSUMPTION_ROW_VERSION = (
    "economic_wm_canonical_consumption_row_v1"
)
ECONOMIC_WM_LOWER_WM_CONSUMPTION_PREFLIGHT_VERSION = (
    "economic_wm_lower_wm_consumption_preflight_v1"
)

REQUIRED_LOWER_WM_KEYS = (
    "perception_grounding",
    "sim_synth_physics",
    "embodiment_actuation",
)

EXPECTED_STATE_VERSIONS = {
    "perception_grounding": "perception_grounding_world_state_v1",
    "sim_synth_physics": "sim_synth_physics_world_state_v1",
    "embodiment_actuation": "embodiment_actuation_world_state_v1",
}

_SOURCE_REF_KEYS = {
    "perception_grounding": (
        "perception_grounding_world_state_path",
        "perception_world_state_path",
    ),
    "sim_synth_physics": (
        "sim_synth_physics_world_state_path",
        "sim_synth_world_state_path",
    ),
    "embodiment_actuation": (
        "embodiment_actuation_world_state_path",
        "embodiment_world_state_path",
    ),
}


def _mapping(payload: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    return dict(to_json_safe(dict(payload or {})))


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _version_and_state_id(path: str | Path) -> tuple[str, str]:
    payload = _load_json(path)
    return str(payload.get("version", "")), str(
        payload.get("state_id")
        or payload.get("world_state_id")
        or payload.get("world_model_id")
        or ""
    )


@dataclass(frozen=True)
class EconomicWMLowerWMReference:
    """A row-level direct reference to one canonical lower-WM state artifact."""

    wm_key: str
    expected_version: str
    artifact_path: str
    observed_version: str = ""
    state_id: str = ""
    reference_status: str = "missing"
    direct_reference: bool = False
    summary_only: bool = True
    blockers: list[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = ECONOMIC_WM_LOWER_WM_REFERENCE_VERSION

    @property
    def satisfied(self) -> bool:
        return (
            bool(self.artifact_path)
            and self.reference_status
            in {"direct_source_reference", "compiled_local_reference"}
            and self.observed_version == self.expected_version
            and not self.summary_only
            and not self.blockers
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "wm_key": self.wm_key,
            "version": self.version,
            "expected_version": self.expected_version,
            "artifact_path": self.artifact_path,
            "observed_version": self.observed_version,
            "state_id": self.state_id,
            "reference_status": self.reference_status,
            "direct_reference": bool(self.direct_reference),
            "summary_only": bool(self.summary_only),
            "satisfied": bool(self.satisfied),
            "blockers": list(self.blockers),
            "metadata": _mapping(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EconomicWMLowerWMReference":
        return cls(
            wm_key=str(payload.get("wm_key", "")),
            expected_version=str(payload.get("expected_version", "")),
            artifact_path=str(payload.get("artifact_path", "")),
            observed_version=str(payload.get("observed_version", "")),
            state_id=str(payload.get("state_id", "")),
            reference_status=str(payload.get("reference_status", "missing")),
            direct_reference=bool(payload.get("direct_reference", False)),
            summary_only=bool(payload.get("summary_only", True)),
            blockers=[str(item) for item in list(payload.get("blockers", []) or [])],
            metadata=_mapping(payload.get("metadata")),
            version=str(payload.get("version", ECONOMIC_WM_LOWER_WM_REFERENCE_VERSION)),
        )


@dataclass(frozen=True)
class EconomicWMCanonicalConsumptionRow:
    """Economic WM row augmented with canonical lower-WM state references."""

    consumption_row_id: str
    source_row_id: str
    source_episode_id: str
    canonical_refs: list[EconomicWMLowerWMReference] = field(default_factory=list)
    training_row: Dict[str, Any] = field(default_factory=dict)
    ready_for_neural_manifest: bool = False
    ready_for_training: bool = False
    promotion_eligible: bool = False
    blockers: list[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = ECONOMIC_WM_CANONICAL_CONSUMPTION_ROW_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "consumption_row_id": self.consumption_row_id,
            "version": self.version,
            "source_row_id": self.source_row_id,
            "source_episode_id": self.source_episode_id,
            "canonical_refs": [ref.to_dict() for ref in self.canonical_refs],
            "training_row": _mapping(self.training_row),
            "ready_for_neural_manifest": bool(self.ready_for_neural_manifest),
            "ready_for_training": bool(self.ready_for_training),
            "promotion_eligible": bool(self.promotion_eligible),
            "blockers": list(self.blockers),
            "metadata": _mapping(self.metadata),
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "EconomicWMCanonicalConsumptionRow":
        return cls(
            consumption_row_id=str(payload.get("consumption_row_id", "")),
            source_row_id=str(payload.get("source_row_id", "")),
            source_episode_id=str(payload.get("source_episode_id", "")),
            canonical_refs=[
                EconomicWMLowerWMReference.from_dict(item)
                for item in list(payload.get("canonical_refs", []) or [])
            ],
            training_row=_mapping(payload.get("training_row")),
            ready_for_neural_manifest=bool(
                payload.get("ready_for_neural_manifest", False)
            ),
            ready_for_training=bool(payload.get("ready_for_training", False)),
            promotion_eligible=bool(payload.get("promotion_eligible", False)),
            blockers=[str(item) for item in list(payload.get("blockers", []) or [])],
            metadata=_mapping(payload.get("metadata")),
            version=str(
                payload.get("version", ECONOMIC_WM_CANONICAL_CONSUMPTION_ROW_VERSION)
            ),
        )


@dataclass(frozen=True)
class EconomicWMLowerWMConsumptionPreflight:
    """Preflight proving Economic WM row consumption has canonical lower-WM refs."""

    preflight_id: str
    corpus_id: str
    row_count: int
    consumption_rows_path: str
    status: str
    all_required_wms_referenced: bool
    ready_for_neural_manifest: bool
    ready_for_training: bool = False
    promotion_eligible: bool = False
    required_wm_keys: list[str] = field(
        default_factory=lambda: list(REQUIRED_LOWER_WM_KEYS)
    )
    missing_reference_count: int = 0
    compiled_reference_count: int = 0
    direct_reference_count: int = 0
    summary_only_reference_count: int = 0
    blockers: list[str] = field(default_factory=list)
    consumption_row_ids: list[str] = field(default_factory=list)
    aggregate_counts: Dict[str, float] = field(default_factory=dict)
    artifact_refs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = ECONOMIC_WM_LOWER_WM_CONSUMPTION_PREFLIGHT_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "preflight_id": self.preflight_id,
            "version": self.version,
            "corpus_id": self.corpus_id,
            "row_count": int(self.row_count),
            "consumption_rows_path": self.consumption_rows_path,
            "status": self.status,
            "all_required_wms_referenced": bool(self.all_required_wms_referenced),
            "ready_for_neural_manifest": bool(self.ready_for_neural_manifest),
            "ready_for_training": bool(self.ready_for_training),
            "promotion_eligible": bool(self.promotion_eligible),
            "required_wm_keys": list(self.required_wm_keys),
            "missing_reference_count": int(self.missing_reference_count),
            "compiled_reference_count": int(self.compiled_reference_count),
            "direct_reference_count": int(self.direct_reference_count),
            "summary_only_reference_count": int(self.summary_only_reference_count),
            "blockers": list(self.blockers),
            "consumption_row_ids": list(self.consumption_row_ids),
            "aggregate_counts": {
                str(key): float(value) for key, value in self.aggregate_counts.items()
            },
            "artifact_refs": _mapping(self.artifact_refs),
            "metadata": _mapping(self.metadata),
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "EconomicWMLowerWMConsumptionPreflight":
        return cls(
            preflight_id=str(payload.get("preflight_id", "")),
            corpus_id=str(payload.get("corpus_id", "")),
            row_count=int(payload.get("row_count", 0) or 0),
            consumption_rows_path=str(payload.get("consumption_rows_path", "")),
            status=str(payload.get("status", "failed")),
            all_required_wms_referenced=bool(
                payload.get("all_required_wms_referenced", False)
            ),
            ready_for_neural_manifest=bool(
                payload.get("ready_for_neural_manifest", False)
            ),
            ready_for_training=bool(payload.get("ready_for_training", False)),
            promotion_eligible=bool(payload.get("promotion_eligible", False)),
            required_wm_keys=[
                str(item) for item in list(payload.get("required_wm_keys", []) or [])
            ],
            missing_reference_count=int(payload.get("missing_reference_count", 0) or 0),
            compiled_reference_count=int(
                payload.get("compiled_reference_count", 0) or 0
            ),
            direct_reference_count=int(payload.get("direct_reference_count", 0) or 0),
            summary_only_reference_count=int(
                payload.get("summary_only_reference_count", 0) or 0
            ),
            blockers=[str(item) for item in list(payload.get("blockers", []) or [])],
            consumption_row_ids=[
                str(item) for item in list(payload.get("consumption_row_ids", []) or [])
            ],
            aggregate_counts={
                str(key): float(value)
                for key, value in dict(
                    payload.get("aggregate_counts", {}) or {}
                ).items()
            },
            artifact_refs=_mapping(payload.get("artifact_refs")),
            metadata=_mapping(payload.get("metadata")),
            version=str(
                payload.get(
                    "version", ECONOMIC_WM_LOWER_WM_CONSUMPTION_PREFLIGHT_VERSION
                )
            ),
        )


def _source_path(row: EconomicWMReplayFeatureRow, wm_key: str) -> str:
    source_refs = _mapping(row.source_refs)
    canonical_refs = _mapping(source_refs.get("canonical_lower_wm_refs"))
    if canonical_refs.get(wm_key):
        value = canonical_refs[wm_key]
        if isinstance(value, Mapping):
            return str(value.get("artifact_path", ""))
        return str(value)
    for key in _SOURCE_REF_KEYS[wm_key]:
        if source_refs.get(key):
            return str(source_refs[key])
    return ""


def _reference_from_path(
    *,
    wm_key: str,
    path: str,
    status: str,
    direct_reference: bool,
    metadata: Optional[Mapping[str, Any]] = None,
) -> EconomicWMLowerWMReference:
    expected = EXPECTED_STATE_VERSIONS[wm_key]
    blockers: list[str] = []
    observed = ""
    state_id = ""
    if not path:
        blockers.append(f"{wm_key}_canonical_state_ref_missing")
    elif not Path(path).exists():
        blockers.append(f"{wm_key}_canonical_state_artifact_missing")
    else:
        observed, state_id = _version_and_state_id(path)
        if observed != expected:
            blockers.append(
                f"{wm_key}_unexpected_state_version::{observed or 'missing'}"
            )
    return EconomicWMLowerWMReference(
        wm_key=wm_key,
        expected_version=expected,
        artifact_path=path,
        observed_version=observed,
        state_id=state_id,
        reference_status=status if not blockers else "missing",
        direct_reference=bool(direct_reference and not blockers),
        summary_only=bool(blockers),
        blockers=blockers,
        metadata=_mapping(metadata),
    )


def _compile_reference_pack(
    *,
    row: EconomicWMReplayFeatureRow,
    output_dir: str | Path,
) -> Dict[str, str]:
    """Compile local canonical lower-WM reference artifacts for one row."""

    from src.world_model.embodiment_actuation import (
        compile_embodiment_actuation_world_state,
    )
    from src.world_model.perception_grounding import (
        compile_perception_grounding_world_state,
    )
    from src.world_model.semantic_coverage_graph import (
        CoverageEdge,
        CoverageNode,
        SemanticCoverageGraph,
    )
    from src.world_model.sim_synth_physics import compile_sim_synth_physics_world_state

    row_dir = Path(output_dir) / "lower_wm_reference_pack" / row.row_id
    row_dir.mkdir(parents=True, exist_ok=True)
    semantic_tags = [
        "economic_wm_consumption",
        f"video:{row.video_id}",
        f"proposal:{row.proposal_id}",
        "benchmark_ready" if row.benchmark_ready else "shadow_only",
    ]
    perception_state = compile_perception_grounding_world_state(
        episode_id=row.source_episode_id,
        task_id="economic_wm_row_consumption",
        semantic_tags=semantic_tags,
        benchmark_signals={
            "benchmark_ready": row.benchmark_ready,
            "shadow_only": row.shadow_only,
            "source_row_id": row.row_id,
        },
        metadata={
            "source": "economic_wm_lower_wm_consumption_preflight",
            "source_row_id": row.row_id,
            "source_episode_id": row.source_episode_id,
        },
    )
    perception_path = row_dir / "perception_grounding_world_state_v1.json"
    _write_json(perception_path, perception_state.to_dict())

    graph = SemanticCoverageGraph(
        nodes=[
            CoverageNode("task:economic_wm_row", "task", "Economic WM row"),
            CoverageNode(f"video:{row.video_id}", "source", row.video_id),
            CoverageNode(f"proposal:{row.proposal_id}", "proposal", row.proposal_id),
            CoverageNode("wm:provider_gap", "risk_family", "provider gap"),
        ],
        edges=[
            CoverageEdge(
                "task:economic_wm_row",
                f"proposal:{row.proposal_id}",
                "consumes",
                evidence_count=1,
                economic_priority=0.8 if row.benchmark_ready else 0.4,
                trust_priority=0.5,
                promotion_readiness=0.0,
            ),
            CoverageEdge(
                f"proposal:{row.proposal_id}",
                "wm:provider_gap",
                "blocked_by",
                evidence_count=1,
                economic_priority=row.target_vector.get(
                    "provider_bringup_gap_weight", 0.0
                ),
                trust_priority=0.3,
                promotion_readiness=0.0,
            ),
        ],
    )
    sim_state = compile_sim_synth_physics_world_state(
        graph,
        limit=2,
        perception_grounding_state=perception_state,
        economic_context={
            "source_row_id": row.row_id,
            "readiness_regime": row.readiness_regime,
            "benchmark_ready": row.benchmark_ready,
            "target_vector": dict(row.target_vector),
        },
        benchmark_signals={"benchmark_ready": row.benchmark_ready},
    )
    sim_path = row_dir / "sim_synth_physics_world_state_v1.json"
    _write_json(sim_path, sim_state.to_dict())

    embodiment_state = compile_embodiment_actuation_world_state(
        episode_id=row.source_episode_id,
        frame_index=0,
        perception_shadow_surface={
            "perception_grounding_state_id": perception_state.state_id,
            "scene_object_count": getattr(
                perception_state.scene_graph, "object_count", 0
            ),
        },
        sim_embodiment_context={
            "sim_synth_physics_state_id": sim_state.state_id,
            "branch_count": len(getattr(sim_state.simulation_agenda, "jobs", []) or []),
        },
        source_refs={
            "economic_wm_row_id": row.row_id,
            "perception_grounding_world_state_path": str(perception_path),
            "sim_synth_physics_world_state_path": str(sim_path),
        },
        metadata={
            "embodiment_id": "economic_wm_local_reference_embodiment",
            "source": "economic_wm_lower_wm_consumption_preflight",
        },
    )
    embodiment_path = row_dir / "embodiment_actuation_world_state_v1.json"
    _write_json(embodiment_path, embodiment_state.to_dict())
    return {
        "perception_grounding": str(perception_path),
        "sim_synth_physics": str(sim_path),
        "embodiment_actuation": str(embodiment_path),
    }


def _build_refs_for_row(
    *,
    row: EconomicWMReplayFeatureRow,
    output_dir: str | Path,
    compile_missing_refs: bool,
) -> list[EconomicWMLowerWMReference]:
    compiled_paths: Dict[str, str] = {}
    if compile_missing_refs and any(
        not _source_path(row, key) for key in REQUIRED_LOWER_WM_KEYS
    ):
        compiled_paths = _compile_reference_pack(row=row, output_dir=output_dir)
    refs: list[EconomicWMLowerWMReference] = []
    for wm_key in REQUIRED_LOWER_WM_KEYS:
        original_path = _source_path(row, wm_key)
        if original_path:
            refs.append(
                _reference_from_path(
                    wm_key=wm_key,
                    path=original_path,
                    status="direct_source_reference",
                    direct_reference=True,
                    metadata={"source_row_id": row.row_id},
                )
            )
            continue
        compiled_path = compiled_paths.get(wm_key, "")
        refs.append(
            _reference_from_path(
                wm_key=wm_key,
                path=compiled_path,
                status="compiled_local_reference" if compiled_path else "missing",
                direct_reference=False,
                metadata={
                    "source_row_id": row.row_id,
                    "compiled_from_row_identity": bool(compiled_path),
                },
            )
        )
    return refs


def build_economic_wm_lower_wm_consumption_preflight(
    *,
    corpus_manifest: EconomicWMTrainingCorpusManifest,
    rows: Iterable[EconomicWMReplayFeatureRow],
    output_dir: str | Path,
    consumption_rows_path: str | Path,
    compile_missing_refs: bool = True,
    artifact_refs: Optional[Mapping[str, Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> tuple[
    EconomicWMLowerWMConsumptionPreflight, list[EconomicWMCanonicalConsumptionRow]
]:
    row_items = list(rows)
    consumption_rows: list[EconomicWMCanonicalConsumptionRow] = []
    for row in row_items:
        refs = _build_refs_for_row(
            row=row,
            output_dir=output_dir,
            compile_missing_refs=compile_missing_refs,
        )
        blockers = sorted({blocker for ref in refs for blocker in ref.blockers})
        augmented_row = row.to_dict()
        augmented_row["source_refs"] = {
            **dict(augmented_row.get("source_refs", {}) or {}),
            "canonical_lower_wm_refs": {ref.wm_key: ref.to_dict() for ref in refs},
        }
        payload = {
            "source_row_id": row.row_id,
            "canonical_refs": [ref.to_dict() for ref in refs],
            "version": ECONOMIC_WM_CANONICAL_CONSUMPTION_ROW_VERSION,
        }
        consumption_rows.append(
            EconomicWMCanonicalConsumptionRow(
                consumption_row_id=f"ewm_consumption_row_{sha256_json(payload)[:16]}",
                source_row_id=row.row_id,
                source_episode_id=row.source_episode_id,
                canonical_refs=refs,
                training_row=augmented_row,
                ready_for_neural_manifest=not blockers,
                ready_for_training=False,
                promotion_eligible=False,
                blockers=blockers,
                metadata={
                    "boundary": "canonical lower-WM reference row only; no training claim",
                    "source_row_version": row.version,
                },
            )
        )

    missing_count = sum(
        1 for row in consumption_rows for ref in row.canonical_refs if not ref.satisfied
    )
    compiled_count = sum(
        1
        for row in consumption_rows
        for ref in row.canonical_refs
        if ref.reference_status == "compiled_local_reference" and ref.satisfied
    )
    direct_count = sum(
        1
        for row in consumption_rows
        for ref in row.canonical_refs
        if ref.reference_status == "direct_source_reference" and ref.satisfied
    )
    summary_only_count = sum(
        1 for row in consumption_rows for ref in row.canonical_refs if ref.summary_only
    )
    blockers = sorted({blocker for row in consumption_rows for blocker in row.blockers})
    all_required = bool(row_items) and missing_count == 0
    aggregate_counts = {
        "row_count": float(len(row_items)),
        "required_reference_count": float(len(row_items) * len(REQUIRED_LOWER_WM_KEYS)),
        "satisfied_reference_count": float(
            len(row_items) * len(REQUIRED_LOWER_WM_KEYS) - missing_count
        ),
        "missing_reference_count": float(missing_count),
        "compiled_reference_count": float(compiled_count),
        "direct_reference_count": float(direct_count),
        "summary_only_reference_count": float(summary_only_count),
    }
    preflight_payload = {
        "corpus_id": corpus_manifest.corpus_id,
        "row_ids": [row.source_row_id for row in consumption_rows],
        "aggregate_counts": aggregate_counts,
        "version": ECONOMIC_WM_LOWER_WM_CONSUMPTION_PREFLIGHT_VERSION,
    }
    preflight = EconomicWMLowerWMConsumptionPreflight(
        preflight_id=f"ewm_lower_wm_consumption_{sha256_json(preflight_payload)[:16]}",
        corpus_id=corpus_manifest.corpus_id,
        row_count=len(row_items),
        consumption_rows_path=str(consumption_rows_path),
        status="ok" if all_required else "failed",
        all_required_wms_referenced=all_required,
        ready_for_neural_manifest=all_required,
        ready_for_training=False,
        promotion_eligible=False,
        missing_reference_count=missing_count,
        compiled_reference_count=compiled_count,
        direct_reference_count=direct_count,
        summary_only_reference_count=summary_only_count,
        blockers=blockers,
        consumption_row_ids=[row.consumption_row_id for row in consumption_rows],
        aggregate_counts=aggregate_counts,
        artifact_refs={
            "corpus_manifest_id": corpus_manifest.corpus_id,
            "source_rows_path": corpus_manifest.rows_path,
            "consumption_rows_path": str(consumption_rows_path),
            **_mapping(artifact_refs),
        },
        metadata={
            "boundary": "lower-WM consumption preflight only; no GPU/provider/training/promotion claim",
            "compile_missing_refs": bool(compile_missing_refs),
            "source_readiness_class": corpus_manifest.readiness_class,
            **_mapping(metadata),
        },
    )
    return preflight, consumption_rows


def save_economic_wm_lower_wm_consumption_outputs(
    *,
    preflight_path: str | Path,
    consumption_rows_path: str | Path,
    preflight: EconomicWMLowerWMConsumptionPreflight,
    consumption_rows: Iterable[EconomicWMCanonicalConsumptionRow],
) -> None:
    rows_target = Path(consumption_rows_path)
    rows_target.parent.mkdir(parents=True, exist_ok=True)
    rows_target.write_text(
        "\n".join(json.dumps(row.to_dict(), sort_keys=True) for row in consumption_rows)
        + "\n",
        encoding="utf-8",
    )
    _write_json(preflight_path, preflight.to_dict())


def load_economic_wm_lower_wm_consumption_preflight(
    path: str | Path,
) -> EconomicWMLowerWMConsumptionPreflight:
    return EconomicWMLowerWMConsumptionPreflight.from_dict(_load_json(path))


def load_economic_wm_canonical_consumption_rows(
    path: str | Path,
) -> list[EconomicWMCanonicalConsumptionRow]:
    rows: list[EconomicWMCanonicalConsumptionRow] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(EconomicWMCanonicalConsumptionRow.from_dict(json.loads(line)))
    return rows


def build_economic_wm_lower_wm_consumption_preflight_from_paths(
    *,
    corpus_manifest_path: str | Path,
    rows_path: str | Path,
    output_dir: str | Path,
    preflight_path: str | Path,
    consumption_rows_path: str | Path,
    compile_missing_refs: bool = True,
    metadata: Optional[Mapping[str, Any]] = None,
) -> EconomicWMLowerWMConsumptionPreflight:
    manifest = load_economic_wm_training_corpus_manifest(corpus_manifest_path)
    rows = load_economic_wm_replay_feature_rows(rows_path)
    preflight, consumption_rows = build_economic_wm_lower_wm_consumption_preflight(
        corpus_manifest=manifest,
        rows=rows,
        output_dir=output_dir,
        consumption_rows_path=consumption_rows_path,
        compile_missing_refs=compile_missing_refs,
        artifact_refs={
            "corpus_manifest_path": str(corpus_manifest_path),
            "rows_path": str(rows_path),
            "preflight_path": str(preflight_path),
        },
        metadata=metadata,
    )
    save_economic_wm_lower_wm_consumption_outputs(
        preflight_path=preflight_path,
        consumption_rows_path=consumption_rows_path,
        preflight=preflight,
        consumption_rows=consumption_rows,
    )
    return preflight


__all__ = [
    "ECONOMIC_WM_CANONICAL_CONSUMPTION_ROW_VERSION",
    "ECONOMIC_WM_LOWER_WM_CONSUMPTION_PREFLIGHT_VERSION",
    "ECONOMIC_WM_LOWER_WM_REFERENCE_VERSION",
    "EXPECTED_STATE_VERSIONS",
    "REQUIRED_LOWER_WM_KEYS",
    "EconomicWMCanonicalConsumptionRow",
    "EconomicWMLowerWMConsumptionPreflight",
    "EconomicWMLowerWMReference",
    "build_economic_wm_lower_wm_consumption_preflight",
    "build_economic_wm_lower_wm_consumption_preflight_from_paths",
    "load_economic_wm_canonical_consumption_rows",
    "load_economic_wm_lower_wm_consumption_preflight",
    "save_economic_wm_lower_wm_consumption_outputs",
]
