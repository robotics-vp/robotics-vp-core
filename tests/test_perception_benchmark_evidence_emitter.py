from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from src.world_model.perception_grounding.annotation_export import (
    AnnotationExportRecord,
    save_annotation_export_json,
)
from src.world_model.perception_grounding.benchmark_evidence import (
    load_perception_benchmark_evidence,
)
from src.world_model.perception_grounding.benchmark_evidence_emitter import (
    emit_annotation_benchmark_evidence,
)


def _write_annotation_export(
    tmp_path: Path,
    *,
    provider_backed: bool,
    n_records: int = 5,
) -> Path:
    records = []
    for i in range(n_records):
        object_track_ids = [f"obj_{j}" for j in range(3)]
        object_categories = [f"class_{j % 2}" for j in range(3)]
        records.append(
            AnnotationExportRecord(
                record_id=f"record_{i}",
                scene_graph_id=f"scene_{i}",
                episode_id=f"episode_{i}",
                frame_index=i,
                object_track_ids=object_track_ids,
                object_tokens=[
                    [float(j + i) / 10.0 for _ in range(128)]
                    for j in range(3)
                ],
                object_categories=object_categories,
                object_confidences=[0.9, 0.8, 0.7],
                object_token_source_kind=(
                    "vision_backbone_projection"
                    if provider_backed
                    else "heuristic_scene_graph"
                ),
                object_token_truth_class=(
                    "provider_backed" if provider_backed else "heuristic_derived"
                ),
                object_token_provider_id=(
                    "vision_backbone_provider" if provider_backed else ""
                ),
                object_token_evidence_provisional=not provider_backed,
                edge_source_ids=["obj_0", "obj_1"],
                edge_target_ids=["obj_1", "obj_2"],
                edge_types=["spatial_proximity", "affordance_relation"],
                edge_confidences=[0.8, 0.6],
                edge_features=[[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]],
                scene_summary_token=[0.1] * 256,
                object_class_labels={
                    track_id: object_categories[idx]
                    for idx, track_id in enumerate(object_track_ids)
                },
                object_annotation_confidences={
                    track_id: 0.85 for track_id in object_track_ids
                },
                annotation_quality_score=0.8,
            )
        )
    path = tmp_path / "annotation_export.json"
    save_annotation_export_json(path, records)
    return path


def test_emitter_writes_provider_backed_scene_graph_evidence(tmp_path) -> None:
    annotation_path = _write_annotation_export(tmp_path, provider_backed=True)
    output_path = tmp_path / "scene_graph_benchmark_evidence.json"

    emission = emit_annotation_benchmark_evidence(
        annotation_export_path=annotation_path,
        seam_type="scene_graph_transformer",
        output_path=output_path,
        hyperparams={
            "d_model": 32,
            "d_out": 128,
            "d_ff": 64,
            "n_heads": 2,
            "n_layers": 1,
        },
    )

    assert output_path.exists()
    assert emission.evidence.subsystem_key == "scene_graph_transformer"
    assert emission.loaded_annotation_record_count == 5
    assert emission.checkpoint_ref_status == "not_supplied"

    loaded = load_perception_benchmark_evidence(output_path)
    payload = loaded.to_dict()
    assert payload["benchmark_evidence_present"] is True
    assert payload["evidence_source_provisional"] is False
    assert payload["evidence_truth_class"] == "provider_backed"
    assert payload["token_source_kind"] == "vision_backbone_projection"
    assert payload["metadata"]["promotion_claim"] == "not_implied_by_emitter"


def test_emitter_keeps_heuristic_annotation_evidence_provisional(tmp_path) -> None:
    annotation_path = _write_annotation_export(tmp_path, provider_backed=False)

    emission = emit_annotation_benchmark_evidence(
        annotation_export_path=annotation_path,
        seam_type="annotation_bridge_projection",
        hyperparams={"d_hidden": 32, "n_categories": 4, "n_affordances": 2},
    )

    payload = emission.evidence.to_dict()
    assert payload["benchmark_evidence_present"] is True
    assert payload["evidence_source_provisional"] is True
    assert payload["promotion_eligible"] is False
    assert payload["evidence_truth_class"] == "heuristic_derived"


def test_annotation_benchmark_evidence_cli(tmp_path) -> None:
    annotation_path = _write_annotation_export(tmp_path, provider_backed=True)
    output_path = tmp_path / "annotation_bridge_benchmark_evidence.json"
    summary_path = tmp_path / "emission_summary.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/emit_perception_annotation_benchmark_evidence.py",
            "--annotation-export",
            str(annotation_path),
            "--output",
            str(output_path),
            "--summary-output",
            str(summary_path),
            "--seam-type",
            "annotation_bridge_projection",
        ],
        check=True,
        cwd=Path(__file__).resolve().parent.parent,
        text=True,
        capture_output=True,
    )

    assert output_path.exists()
    assert summary_path.exists()
    cli_summary = json.loads(result.stdout)
    persisted_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert cli_summary["benchmark_evidence_present"] is True
    assert cli_summary["checkpoint_ref_status"] == "not_supplied"
    assert persisted_summary["evidence"]["subsystem_key"] == (
        "annotation_bridge_projection"
    )
