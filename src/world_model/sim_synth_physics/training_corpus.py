"""Training-row extraction from sim/synth/physics WM runtime receipts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence


def _mapping(value: Any) -> Dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _mapping_list(value: Any) -> list[Dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _clip01(value: Any, default: float = 0.0) -> float:
    try:
        return float(max(0.0, min(1.0, float(value))))
    except Exception:
        return float(default)


def _status_yield_score(status: str, fallback: float) -> float:
    normalized = str(status or "").lower()
    if normalized in {"completed", "success", "accepted"}:
        return max(_clip01(fallback), 0.8)
    if normalized in {"failed", "blocked", "rejected"}:
        return min(_clip01(fallback), 0.2)
    return _clip01(fallback)


def _json_mappings(path: Path) -> list[Dict[str, Any]]:
    if path.suffix == ".jsonl":
        rows: list[Dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, Mapping):
                rows.append(dict(payload))
        return rows
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, Mapping):
        return [dict(payload)]
    if isinstance(payload, Sequence):
        return _mapping_list(payload)
    return []


def load_sim_synth_receipt_bundles(path: str | Path) -> list[Dict[str, Any]]:
    """Load runtime receipt bundles from JSON or JSONL."""

    receipt_path = Path(path)
    if not receipt_path.exists():
        raise FileNotFoundError(f"sim/synth/physics receipt path not found: {receipt_path}")
    if receipt_path.suffix == ".jsonl":
        bundles: list[Dict[str, Any]] = []
        for line in receipt_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, Mapping):
                bundles.append(dict(payload))
        return bundles

    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    if isinstance(payload, Mapping):
        if isinstance(payload.get("bundles"), Sequence):
            return _mapping_list(payload.get("bundles"))
        if payload.get("world_state") is not None:
            return [dict(payload)]
        receipts = payload.get("receipts")
        if isinstance(receipts, Sequence):
            return _mapping_list(receipts)
    if isinstance(payload, Sequence):
        return _mapping_list(payload)
    raise ValueError(f"Unsupported sim/synth/physics receipt payload in {receipt_path}")


def harvest_sim_synth_receipt_bundles(paths: Sequence[str | Path]) -> list[Dict[str, Any]]:
    """Harvest live sim/synth/physics receipt bundles from files or directories."""

    harvested: list[Dict[str, Any]] = []
    by_state_id: dict[str, Dict[str, Any]] = {}
    for candidate in _iter_receipt_inputs(paths):
        candidate_path = Path(candidate)
        if candidate_path.is_file():
            try:
                harvested.extend(load_sim_synth_receipt_bundles(candidate_path))
            except Exception:
                harvested.extend(_harvest_receipt_file(candidate_path))
            continue
        harvested.extend(_harvest_receipt_dir(candidate_path))

    deduped: list[Dict[str, Any]] = []
    for bundle in harvested:
        state_id = str(_mapping(bundle.get("world_state")).get("state_id", "") or "")
        if not state_id:
            deduped.append(bundle)
            continue
        previous = by_state_id.get(state_id)
        if previous is None:
            by_state_id[state_id] = bundle
            continue
        prev_outcomes = _mapping_list(previous.get("simulation_outcome_receipts"))
        next_outcomes = _mapping_list(bundle.get("simulation_outcome_receipts"))
        if len(next_outcomes) > len(prev_outcomes):
            by_state_id[state_id] = bundle
    deduped.extend(by_state_id[key] for key in sorted(by_state_id))
    return deduped


def _iter_receipt_inputs(paths: Sequence[str | Path]) -> Iterable[Path]:
    seen: set[Path] = set()
    for raw in paths:
        candidate = Path(raw).resolve()
        if candidate in seen or not candidate.exists():
            continue
        seen.add(candidate)
        yield candidate


def _looks_like_receipt_bundle(path: Path) -> bool:
    name = path.name.lower()
    if any(token in name for token in ("receipt_bundle", "sim_synth", "world_state", "physics_calibration_receipt", "simulation_outcome_receipt")):
        return True
    return False


def _harvest_receipt_dir(root: Path) -> list[Dict[str, Any]]:
    grouped_world_states: dict[Path, list[Dict[str, Any]]] = {}
    grouped_calibrations: dict[Path, list[Dict[str, Any]]] = {}
    grouped_outcomes: dict[Path, list[Dict[str, Any]]] = {}
    explicit_bundles: list[Dict[str, Any]] = []

    for path in sorted(root.rglob("*.json")) + sorted(root.rglob("*.jsonl")):
        if not _looks_like_receipt_bundle(path):
            continue
        try:
            rows = _json_mappings(path)
        except Exception:
            continue
        if not rows:
            continue
        if path.is_file() and any(
            isinstance(row.get("world_state"), Mapping) or row.get("bundles") or row.get("receipts")
            for row in rows
        ):
            try:
                explicit_bundles.extend(load_sim_synth_receipt_bundles(path))
                continue
            except Exception:
                pass
        for payload in rows:
            version = str(payload.get("version", payload.get("schema_version", "")) or "")
            parent = path.parent.resolve()
            if version == "sim_synth_physics_world_state_v1":
                grouped_world_states.setdefault(parent, []).append(dict(payload))
            elif version == "physics_calibration_receipt_v1":
                grouped_calibrations.setdefault(parent, []).append(dict(payload))
            elif version == "simulation_outcome_receipt_v1":
                grouped_outcomes.setdefault(parent, []).append(dict(payload))

    bundles: list[Dict[str, Any]] = list(explicit_bundles)
    all_dirs = sorted(
        set(grouped_world_states) | set(grouped_calibrations) | set(grouped_outcomes)
    )
    for directory in all_dirs:
        world_states = grouped_world_states.get(directory, [])
        calibrations = grouped_calibrations.get(directory, [])
        outcomes = grouped_outcomes.get(directory, [])
        for world_state in world_states:
            physics_context = _mapping(world_state.get("physics_context"))
            metadata = _mapping(physics_context.get("metadata"))
            bundle: Dict[str, Any] = {
                "bundle_id": str(world_state.get("state_id", "")) or directory.name,
                "world_state": dict(world_state),
            }
            if calibrations:
                bundle["physics_calibration_receipt"] = dict(calibrations[-1])
            if outcomes:
                bundle["simulation_outcome_receipts"] = [dict(item) for item in outcomes]
            if metadata.get("benchmark_signals"):
                bundle["benchmark_signals"] = _mapping(metadata.get("benchmark_signals"))
            bundles.append(bundle)
    return bundles


def _harvest_receipt_file(path: Path) -> list[Dict[str, Any]]:
    try:
        rows = _json_mappings(path)
    except Exception:
        return []
    parent = path.parent.resolve()
    world_states = [
        dict(payload)
        for payload in rows
        if str(payload.get("version", payload.get("schema_version", "")) or "")
        == "sim_synth_physics_world_state_v1"
    ]
    calibrations = [
        dict(payload)
        for payload in rows
        if str(payload.get("version", payload.get("schema_version", "")) or "")
        == "physics_calibration_receipt_v1"
    ]
    outcomes = [
        dict(payload)
        for payload in rows
        if str(payload.get("version", payload.get("schema_version", "")) or "")
        == "simulation_outcome_receipt_v1"
    ]
    bundles: list[Dict[str, Any]] = []
    for world_state in world_states:
        bundle: Dict[str, Any] = {
            "bundle_id": str(world_state.get("state_id", "")) or parent.name,
            "world_state": world_state,
        }
        if calibrations:
            bundle["physics_calibration_receipt"] = calibrations[-1]
        if outcomes:
            bundle["simulation_outcome_receipts"] = outcomes
        bundles.append(bundle)
    return bundles


def build_backend_selector_rows_from_receipts(
    bundles: Sequence[Mapping[str, Any]],
) -> list[Dict[str, Any]]:
    """Project runtime receipt bundles into backend-selector training rows."""

    rows: list[Dict[str, Any]] = []
    for bundle_index, bundle in enumerate(bundles):
        bundle_mapping = _mapping(bundle)
        world_state = _mapping(bundle_mapping.get("world_state"))
        agenda = _mapping(world_state.get("simulation_agenda"))
        jobs = _mapping_list(agenda.get("jobs"))
        physics_context = _mapping(world_state.get("physics_context"))
        if not jobs or not physics_context:
            continue
        physics_metadata = _mapping(physics_context.get("metadata"))
        calibration_receipt = _mapping(
            bundle_mapping.get("physics_calibration_receipt")
            or bundle_mapping.get("physics_calibration")
        )
        target_source = "runtime_receipt" if calibration_receipt else "wm_planning_state"
        helper_status = _mapping(physics_metadata.get("backend_helper_status"))
        benchmark_signals = _mapping(
            bundle_mapping.get("benchmark_signals") or physics_metadata.get("benchmark_signals")
        )
        rows.append(
            {
                "row_id": str(
                    bundle_mapping.get("bundle_id")
                    or world_state.get("state_id")
                    or f"backend_selector_bundle_{bundle_index}"
                ),
                "jobs": jobs,
                "benchmark_signals": benchmark_signals,
                "heuristic_backend": str(
                    physics_metadata.get("heuristic_backend")
                    or physics_context.get("backend")
                    or "other"
                ),
                "heuristic_fidelity_tier": str(
                    physics_metadata.get("heuristic_fidelity_tier")
                    or physics_context.get("fidelity_tier")
                    or "branch_balanced"
                ),
                "heuristic_domain_randomization_regime": str(
                    physics_metadata.get("heuristic_domain_randomization_regime")
                    or physics_context.get("domain_randomization_regime")
                    or "steady_state"
                ),
                "target_backend": str(
                    calibration_receipt.get("backend") or physics_context.get("backend") or "other"
                ),
                "target_fidelity_tier": str(
                    calibration_receipt.get("fidelity_tier")
                    or physics_context.get("fidelity_tier")
                    or "branch_balanced"
                ),
                "target_domain_randomization_regime": str(
                    calibration_receipt.get("domain_randomization_regime")
                    or physics_context.get("domain_randomization_regime")
                    or "steady_state"
                ),
                "target_source": target_source,
                "promotion_stage": str(helper_status.get("promotion_stage") or "shadow_candidate"),
                "metadata": {
                    "bundle_index": bundle_index,
                    "world_state_id": world_state.get("state_id"),
                    "calibration_receipt_id": calibration_receipt.get("receipt_id"),
                    "calibration_quality_score": calibration_receipt.get("quality_score"),
                },
            }
        )
    return rows


def build_branch_planner_rows_from_receipts(
    bundles: Sequence[Mapping[str, Any]],
) -> list[Dict[str, Any]]:
    """Project runtime receipt bundles into branch-planner training rows."""

    rows: list[Dict[str, Any]] = []
    for bundle_index, bundle in enumerate(bundles):
        bundle_mapping = _mapping(bundle)
        world_state = _mapping(bundle_mapping.get("world_state"))
        agenda = _mapping(world_state.get("simulation_agenda"))
        jobs = {
            str(job.get("job_id")): job
            for job in _mapping_list(agenda.get("jobs"))
            if str(job.get("job_id"))
        }
        physics_context = _mapping(world_state.get("physics_context"))
        if not jobs or not physics_context:
            continue
        benchmark_signals = _mapping(
            bundle_mapping.get("benchmark_signals")
            or _mapping(_mapping(world_state.get("gen2sim_admission")).get("metadata")).get(
                "benchmark_signals"
            )
            or _mapping(_mapping(physics_context.get("metadata")).get("benchmark_signals"))
        )
        outcomes = {
            str(receipt.get("branch_plan_id")): receipt
            for receipt in _mapping_list(
                bundle_mapping.get("simulation_outcome_receipts")
                or bundle_mapping.get("simulation_outcomes")
            )
            if str(receipt.get("branch_plan_id"))
        }
        for plan_index, plan in enumerate(_mapping_list(world_state.get("synthetic_branch_plans"))):
            plan_id = str(plan.get("plan_id", ""))
            source_job_id = str(plan.get("source_job_id", ""))
            job = jobs.get(source_job_id)
            if not plan_id or job is None:
                continue
            plan_metadata = _mapping(plan.get("metadata"))
            helper_status = _mapping(plan_metadata.get("branch_helper_status"))
            outcome = _mapping(outcomes.get(plan_id))
            outcome_metadata = _mapping(outcome.get("metadata"))
            target_source = "runtime_receipt" if outcome else "wm_planning_state"
            fallback_yield = _clip01(plan.get("expected_yield_score"), 0.0)
            realized_yield = _clip01(
                outcome_metadata.get("realized_yield_score", outcome_metadata.get("quality_score")),
                _status_yield_score(str(outcome.get("status", "")), fallback_yield),
            )
            rows.append(
                {
                    "row_id": str(
                        outcome.get("receipt_id")
                        or f"{world_state.get('state_id', 'world_state')}::{plan_id}::{plan_index}"
                    ),
                    "job": dict(job),
                    "context": {
                        "physics_context": dict(physics_context),
                        "heuristic_generation_mode": str(
                            plan_metadata.get("heuristic_generation_mode")
                            or plan.get("generation_mode")
                            or "coverage_branch"
                        ),
                        "benchmark_signals": benchmark_signals,
                    },
                    "target_generation_mode": str(
                        outcome_metadata.get("executed_generation_mode")
                        or plan.get("generation_mode")
                        or "coverage_branch"
                    ),
                    "target_expected_yield_score": realized_yield,
                    "target_source": target_source,
                    "promotion_stage": str(helper_status.get("promotion_stage") or "shadow_candidate"),
                    "metadata": {
                        "bundle_index": bundle_index,
                        "world_state_id": world_state.get("state_id"),
                        "branch_plan_id": plan_id,
                        "simulation_status": outcome.get("status"),
                    },
                }
            )
    return rows


__all__ = [
    "build_backend_selector_rows_from_receipts",
    "build_branch_planner_rows_from_receipts",
    "harvest_sim_synth_receipt_bundles",
    "load_sim_synth_receipt_bundles",
]
