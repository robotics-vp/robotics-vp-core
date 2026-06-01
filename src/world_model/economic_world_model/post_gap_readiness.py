"""Post-gap readiness manifests for Economic WM GPU/data/hardware bring-up.

This module covers the CPU-capable planning work that should exist before a
GPU, provider, large external corpus, or humanoid hardware window opens. It
emits typed plans and fail-closed receipts only; it does not download external
datasets, launch providers, run GPU training, or grant Phase 7 authority.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from src.utils.config_digest import sha256_json
from src.utils.json_safe import to_json_safe
from src.world_model.economic_world_model.post_gap_readiness_catalog import (
    build_benchmark_gate_specs,
    build_corpus_prep_artifact_plans,
    build_evidence_hygiene_specs,
    build_external_dataset_corpus_plans,
    build_g1_r1_purchase_readiness_specs,
    build_gpu_day_one_runbooks,
    build_perception_embodiment_replay_loop_specs,
    build_provider_runtime_packaging_specs,
)
from src.world_model.economic_world_model.post_gap_readiness_models import (
    BenchmarkGateSpec,
    CorpusPrepArtifactPlan,
    ExternalDatasetCorpusPlan,
    GPUDayOneRunbook,
    PostGapReadinessReport,
    ReadinessSpec,
)


def _mapping(payload: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    return dict(to_json_safe(dict(payload or {})))


def _strings(values: Optional[Iterable[Any]]) -> list[str]:
    return [str(value) for value in list(values or []) if str(value)]


def _stable_id(prefix: str, payload: Mapping[str, Any]) -> str:
    return f"{prefix}_{sha256_json(_mapping(payload))[:16]}"


def _jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(_mapping(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def build_post_gap_readiness_bundle() -> dict[str, Any]:
    runbooks = build_gpu_day_one_runbooks()
    datasets = build_external_dataset_corpus_plans()
    corpus_prep = build_corpus_prep_artifact_plans(datasets)
    benchmark_gates = build_benchmark_gate_specs()
    provider_packaging = build_provider_runtime_packaging_specs()
    replay_loop = build_perception_embodiment_replay_loop_specs()
    purchase_readiness = build_g1_r1_purchase_readiness_specs()
    evidence_hygiene = build_evidence_hygiene_specs()

    all_manifested = all(
        [
            len(runbooks) >= 5,
            len(datasets) >= 8,
            len(corpus_prep) >= len(datasets) * 6,
            len(benchmark_gates) >= 6,
            len(provider_packaging) >= 6,
            len(replay_loop) >= 5,
            len(purchase_readiness) >= 10,
            len(evidence_hygiene) >= 7,
        ]
    )
    remaining_blockers = [
        "external dataset download and license/storage review not executed",
        "RunPod/cloud/local Linux provider windows not executed",
        "GPU training and promotion-grade benchmarks not executed",
        "G1/R1 vendor quote, purchase, and physical safety inspection missing",
        "real/sim visual stream calibration missing",
    ]
    report_payload = {
        "runbooks": [row.to_dict() for row in runbooks],
        "datasets": [row.to_dict() for row in datasets],
        "corpus_prep": [row.to_dict() for row in corpus_prep],
        "benchmark_gates": [row.to_dict() for row in benchmark_gates],
        "provider_packaging": [row.to_dict() for row in provider_packaging],
        "replay_loop": [row.to_dict() for row in replay_loop],
        "purchase_readiness": [row.to_dict() for row in purchase_readiness],
        "evidence_hygiene": [row.to_dict() for row in evidence_hygiene],
    }
    report = PostGapReadinessReport(
        report_id=_stable_id("post_gap_readiness_report", report_payload),
        status="ok_planning_complete_launch_blocked",
        all_post_gap_items_manifested=all_manifested,
        gpu_day_one_runbook_count=len(runbooks),
        external_dataset_count=len(datasets),
        corpus_prep_artifact_count=len(corpus_prep),
        benchmark_gate_count=len(benchmark_gates),
        provider_runtime_packaging_count=len(provider_packaging),
        replay_loop_count=len(replay_loop),
        g1_r1_purchase_readiness_count=len(purchase_readiness),
        evidence_hygiene_count=len(evidence_hygiene),
        launch_authority_granted=False,
        provider_executed=False,
        gpu_training_executed=False,
        external_download_executed=False,
        phase7_constraint_honored=True,
        promotion_eligible=False,
        ready_for_august_gpu_window=all_manifested,
        remaining_blockers=remaining_blockers,
        artifact_refs={},
        metadata={
            "ad_hoc_note": "2026-05-25-cpu-capable-august-gap-items",
            "phase7_new_concepts_added": False,
            "dataset_sources_identified": [dataset.dataset_id for dataset in datasets],
        },
    )
    return {
        "report": report,
        "runbooks": runbooks,
        "datasets": datasets,
        "corpus_prep": corpus_prep,
        "benchmark_gates": benchmark_gates,
        "provider_packaging": provider_packaging,
        "replay_loop": replay_loop,
        "purchase_readiness": purchase_readiness,
        "evidence_hygiene": evidence_hygiene,
    }


def _write_markdown(path: Path, bundle: Mapping[str, Any]) -> None:
    report: PostGapReadinessReport = bundle["report"]
    datasets: Sequence[ExternalDatasetCorpusPlan] = bundle["datasets"]
    runbooks: Sequence[GPUDayOneRunbook] = bundle["runbooks"]
    gates: Sequence[BenchmarkGateSpec] = bundle["benchmark_gates"]
    lines = [
        "# Economic WM Post-Gap Readiness",
        "",
        "[ad-hoc note]",
        "",
        f"- Report ID: `{report.report_id}`",
        f"- Status: `{report.status}`",
        f"- All post-gap items manifested: `{str(report.all_post_gap_items_manifested).lower()}`",
        f"- Ready for August GPU window: `{str(report.ready_for_august_gpu_window).lower()}`",
        f"- Launch authority granted: `{str(report.launch_authority_granted).lower()}`",
        f"- External downloads executed: `{str(report.external_download_executed).lower()}`",
        f"- Provider executed: `{str(report.provider_executed).lower()}`",
        f"- GPU training executed: `{str(report.gpu_training_executed).lower()}`",
        f"- Promotion eligible: `{str(report.promotion_eligible).lower()}`",
        f"- Phase 7 constraint honored: `{str(report.phase7_constraint_honored).lower()}`",
        "",
        "## Counts",
        "",
        f"- GPU day-one runbooks: `{report.gpu_day_one_runbook_count}`",
        f"- external/local dataset plans: `{report.external_dataset_count}`",
        f"- corpus prep artifact plans: `{report.corpus_prep_artifact_count}`",
        f"- benchmark gates: `{report.benchmark_gate_count}`",
        f"- provider/runtime packaging specs: `{report.provider_runtime_packaging_count}`",
        f"- replay loop specs: `{report.replay_loop_count}`",
        f"- G1/R1 purchase readiness specs: `{report.g1_r1_purchase_readiness_count}`",
        f"- evidence hygiene specs: `{report.evidence_hygiene_count}`",
        "",
        "## External Datasets To Bring In",
        "",
    ]
    for dataset in datasets:
        lines.extend(
            [
                f"### `{dataset.dataset_id}`",
                f"- name: {dataset.name}",
                f"- priority: `{dataset.priority}`",
                f"- status: `{dataset.bring_in_status}`",
                f"- source: {dataset.source_url}",
                f"- expected scale: {dataset.expected_scale}",
                f"- schema targets: {', '.join(dataset.repo_schema_targets)}",
                f"- import blockers: {', '.join(dataset.import_blockers) or 'none'}",
            ]
        )
    lines.extend(["", "## GPU Day-One Runbooks", ""])
    for runbook in runbooks:
        lines.extend(
            [
                f"### `{runbook.runbook_id}`",
                f"- name: {runbook.name}",
                f"- plane: `{runbook.plane}`",
                f"- pod class: `{runbook.pod_class}`",
                f"- horizon: `{runbook.horizon}`",
                f"- launch allowed: `{str(runbook.launch_allowed).lower()}`",
                f"- stop conditions: {', '.join(runbook.stop_conditions)}",
            ]
        )
    lines.extend(["", "## Benchmark Gates", ""])
    for gate in gates:
        lines.extend(
            [
                f"- `{gate.gate_key}` on `{gate.surface}`: `{gate.status}`; promotion eligible `{str(gate.promotion_eligible).lower()}`",
            ]
        )
    lines.extend(["", "## Remaining Blockers", ""])
    lines.extend(f"- `{blocker}`" for blocker in report.remaining_blockers)
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "This report is a planning and receipt surface. It does not download external datasets, run providers, run GPU training, purchase hardware, grant promotion, or expand Phase 7 concepts.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_post_gap_readiness_bundle(
    *,
    output_dir: str | Path,
    bundle: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Write all post-gap readiness artifacts and return the report payload."""

    resolved_bundle = dict(bundle or build_post_gap_readiness_bundle())
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    report_path = output_root / "post_gap_readiness_report_v1.json"
    markdown_path = output_root / "post_gap_readiness_v1.md"
    runbooks_path = output_root / "gpu_day_one_runbooks_v1.jsonl"
    datasets_path = output_root / "external_dataset_corpus_plan_v1.jsonl"
    corpus_prep_path = output_root / "corpus_prep_artifact_plans_v1.jsonl"
    benchmark_path = output_root / "benchmark_gate_specs_v1.jsonl"
    provider_packaging_path = output_root / "provider_runtime_packaging_specs_v1.jsonl"
    replay_loop_path = output_root / "perception_embodiment_replay_loop_specs_v1.jsonl"
    purchase_path = output_root / "g1_r1_purchase_readiness_v1.jsonl"
    hygiene_path = output_root / "evidence_hygiene_specs_v1.jsonl"

    artifact_refs = {
        "report_path": str(report_path),
        "markdown_path": str(markdown_path),
        "gpu_day_one_runbooks_path": str(runbooks_path),
        "external_dataset_corpus_plan_path": str(datasets_path),
        "corpus_prep_artifact_plans_path": str(corpus_prep_path),
        "benchmark_gate_specs_path": str(benchmark_path),
        "provider_runtime_packaging_specs_path": str(provider_packaging_path),
        "perception_embodiment_replay_loop_specs_path": str(replay_loop_path),
        "g1_r1_purchase_readiness_path": str(purchase_path),
        "evidence_hygiene_specs_path": str(hygiene_path),
    }
    report: PostGapReadinessReport = resolved_bundle["report"]
    report = PostGapReadinessReport.from_dict(
        {**report.to_dict(), "artifact_refs": artifact_refs}
    )
    resolved_bundle["report"] = report

    _jsonl(runbooks_path, [row.to_dict() for row in resolved_bundle["runbooks"]])
    _jsonl(datasets_path, [row.to_dict() for row in resolved_bundle["datasets"]])
    _jsonl(corpus_prep_path, [row.to_dict() for row in resolved_bundle["corpus_prep"]])
    _jsonl(
        benchmark_path,
        [row.to_dict() for row in resolved_bundle["benchmark_gates"]],
    )
    _jsonl(
        provider_packaging_path,
        [row.to_dict() for row in resolved_bundle["provider_packaging"]],
    )
    _jsonl(replay_loop_path, [row.to_dict() for row in resolved_bundle["replay_loop"]])
    _jsonl(
        purchase_path,
        [row.to_dict() for row in resolved_bundle["purchase_readiness"]],
    )
    _jsonl(
        hygiene_path,
        [row.to_dict() for row in resolved_bundle["evidence_hygiene"]],
    )
    report_path.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_markdown(markdown_path, resolved_bundle)
    return report.to_dict()


def load_post_gap_readiness_report(path: str | Path) -> PostGapReadinessReport:
    return PostGapReadinessReport.from_dict(_load_json(path))


def load_gpu_day_one_runbooks(path: str | Path) -> list[GPUDayOneRunbook]:
    return [GPUDayOneRunbook.from_dict(row) for row in _load_jsonl(path)]


def load_external_dataset_corpus_plans(
    path: str | Path,
) -> list[ExternalDatasetCorpusPlan]:
    return [ExternalDatasetCorpusPlan.from_dict(row) for row in _load_jsonl(path)]


def load_corpus_prep_artifact_plans(path: str | Path) -> list[CorpusPrepArtifactPlan]:
    return [CorpusPrepArtifactPlan.from_dict(row) for row in _load_jsonl(path)]


def load_benchmark_gate_specs(path: str | Path) -> list[BenchmarkGateSpec]:
    return [BenchmarkGateSpec.from_dict(row) for row in _load_jsonl(path)]


def load_readiness_specs(path: str | Path) -> list[ReadinessSpec]:
    return [ReadinessSpec.from_dict(row) for row in _load_jsonl(path)]
