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


def _runtime_outcome_can_drive_target_source(
    outcome_receipt: Mapping[str, Any],
    outcome_metadata: Mapping[str, Any],
) -> bool:
    if str(outcome_receipt.get("outcome_status", "") or "") != "runtime_outputs_harvested":
        return False
    selected_ref_validation = _mapping(outcome_metadata.get("selected_ref_validation"))
    validation_status = str(selected_ref_validation.get("status", "legacy_unchecked") or "")
    return validation_status in {
        "selected_refs_matched",
        "no_expected_selected_refs",
        "legacy_unchecked",
    }


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
        receipts = payload.get("receipts")
        version = str(payload.get("version", payload.get("schema_version", "")) or "")
        if version.endswith("_bundle_v1") and isinstance(receipts, Sequence):
            return _mapping_list(receipts)
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
            receipt_rows = _mapping_list(receipts)
            if all("world_state" in row or "bundle_id" in row for row in receipt_rows):
                return receipt_rows
            raise ValueError("raw receipt bundle is not a sim/synth training bundle")
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
    if any(
        token in name
        for token in (
            "receipt_bundle",
            "sim_synth",
            "world_state",
            "runtime_receipt_manifest",
            "physics_execution_contract",
            "physics_adaptation_receipt",
            "backend_execution_binding_receipt",
            "robot_asset_contract_receipt",
            "gen2sim_admission_receipt",
            "backend_runtime_bridge_receipt",
            "backend_runtime_work_order",
            "backend_runtime_execution_receipt",
            "backend_runtime_adapter_receipt",
            "backend_runtime_launch_receipt",
            "backend_runtime_outcome_receipt",
            "backend_shadow_execution_receipt",
            "physics_calibration_receipt",
            "task_measurement_receipt",
            "sim_real_gap_receipt",
            "backend_mismatch_receipt",
            "surrogate_physics_receipt",
            "surrogate_calibration_receipt",
            "branch_validity_receipt",
            "sensor_alignment_receipt",
            "replay_validity_receipt",
            "render_provider_receipt",
            "simulation_outcome_receipt",
        )
    ):
        return True
    return False


def _harvest_receipt_dir(root: Path) -> list[Dict[str, Any]]:
    grouped_world_states: dict[Path, list[Dict[str, Any]]] = {}
    grouped_runtime_receipt_manifests: dict[Path, list[Dict[str, Any]]] = {}
    grouped_execution_contracts: dict[Path, list[Dict[str, Any]]] = {}
    grouped_adaptations: dict[Path, list[Dict[str, Any]]] = {}
    grouped_backend_bindings: dict[Path, list[Dict[str, Any]]] = {}
    grouped_asset_contracts: dict[Path, list[Dict[str, Any]]] = {}
    grouped_gen2sim_receipts: dict[Path, list[Dict[str, Any]]] = {}
    grouped_backend_bridges: dict[Path, list[Dict[str, Any]]] = {}
    grouped_backend_work_orders: dict[Path, list[Dict[str, Any]]] = {}
    grouped_backend_runtime: dict[Path, list[Dict[str, Any]]] = {}
    grouped_backend_adapter: dict[Path, list[Dict[str, Any]]] = {}
    grouped_backend_launch: dict[Path, list[Dict[str, Any]]] = {}
    grouped_backend_outcome: dict[Path, list[Dict[str, Any]]] = {}
    grouped_backend_shadow: dict[Path, list[Dict[str, Any]]] = {}
    grouped_calibrations: dict[Path, list[Dict[str, Any]]] = {}
    grouped_task_measurements: dict[Path, list[Dict[str, Any]]] = {}
    grouped_sim_real_gaps: dict[Path, list[Dict[str, Any]]] = {}
    grouped_backend_mismatches: dict[Path, list[Dict[str, Any]]] = {}
    grouped_surrogate_physics: dict[Path, list[Dict[str, Any]]] = {}
    grouped_surrogate_calibrations: dict[Path, list[Dict[str, Any]]] = {}
    grouped_branch_validity_receipts: dict[Path, list[Dict[str, Any]]] = {}
    grouped_sensor_alignments: dict[Path, list[Dict[str, Any]]] = {}
    grouped_replay_validity_receipts: dict[Path, list[Dict[str, Any]]] = {}
    grouped_render_receipts: dict[Path, list[Dict[str, Any]]] = {}
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
            isinstance(row.get("world_state"), Mapping) or row.get("bundles")
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
            elif version == "sim_synth_runtime_receipt_manifest_v1":
                grouped_runtime_receipt_manifests.setdefault(parent, []).append(
                    dict(payload)
                )
            elif version == "physics_execution_contract_v1":
                grouped_execution_contracts.setdefault(parent, []).append(dict(payload))
            elif version == "physics_adaptation_receipt_v1":
                grouped_adaptations.setdefault(parent, []).append(dict(payload))
            elif version == "backend_execution_binding_receipt_v1":
                grouped_backend_bindings.setdefault(parent, []).append(dict(payload))
            elif version == "robot_asset_contract_receipt_v1":
                grouped_asset_contracts.setdefault(parent, []).append(dict(payload))
            elif version == "gen2sim_admission_receipt_v1":
                grouped_gen2sim_receipts.setdefault(parent, []).append(dict(payload))
            elif version == "backend_runtime_bridge_receipt_v1":
                grouped_backend_bridges.setdefault(parent, []).append(dict(payload))
            elif version == "backend_runtime_work_order_receipt_v1":
                grouped_backend_work_orders.setdefault(parent, []).append(dict(payload))
            elif version == "backend_runtime_execution_receipt_v1":
                grouped_backend_runtime.setdefault(parent, []).append(dict(payload))
            elif version == "backend_runtime_adapter_receipt_v1":
                grouped_backend_adapter.setdefault(parent, []).append(dict(payload))
            elif version == "backend_runtime_launch_receipt_v1":
                grouped_backend_launch.setdefault(parent, []).append(dict(payload))
            elif version == "backend_runtime_outcome_receipt_v1":
                grouped_backend_outcome.setdefault(parent, []).append(dict(payload))
            elif version == "backend_shadow_execution_receipt_v1":
                grouped_backend_shadow.setdefault(parent, []).append(dict(payload))
            elif version == "physics_calibration_receipt_v1":
                grouped_calibrations.setdefault(parent, []).append(dict(payload))
            elif version == "task_measurement_receipt_v1":
                grouped_task_measurements.setdefault(parent, []).append(dict(payload))
            elif version == "sim_real_gap_receipt_v1":
                grouped_sim_real_gaps.setdefault(parent, []).append(dict(payload))
            elif version == "backend_mismatch_receipt_v1":
                grouped_backend_mismatches.setdefault(parent, []).append(dict(payload))
            elif version == "surrogate_physics_receipt_v1":
                grouped_surrogate_physics.setdefault(parent, []).append(dict(payload))
            elif version == "surrogate_calibration_receipt_v1":
                grouped_surrogate_calibrations.setdefault(parent, []).append(dict(payload))
            elif version == "branch_validity_receipt_v1":
                grouped_branch_validity_receipts.setdefault(parent, []).append(dict(payload))
            elif version == "sensor_alignment_receipt_v1":
                grouped_sensor_alignments.setdefault(parent, []).append(dict(payload))
            elif version == "replay_validity_receipt_v1":
                grouped_replay_validity_receipts.setdefault(parent, []).append(dict(payload))
            elif version == "render_provider_receipt_v1":
                grouped_render_receipts.setdefault(parent, []).append(dict(payload))
            elif version == "simulation_outcome_receipt_v1":
                grouped_outcomes.setdefault(parent, []).append(dict(payload))

    bundles: list[Dict[str, Any]] = list(explicit_bundles)
    all_dirs = sorted(
        set(grouped_world_states)
        | set(grouped_runtime_receipt_manifests)
        | set(grouped_execution_contracts)
        | set(grouped_adaptations)
        | set(grouped_backend_bindings)
        | set(grouped_asset_contracts)
        | set(grouped_gen2sim_receipts)
        | set(grouped_backend_bridges)
        | set(grouped_backend_work_orders)
        | set(grouped_backend_runtime)
        | set(grouped_backend_adapter)
        | set(grouped_backend_launch)
        | set(grouped_backend_outcome)
        | set(grouped_backend_shadow)
        | set(grouped_calibrations)
        | set(grouped_task_measurements)
        | set(grouped_sim_real_gaps)
        | set(grouped_backend_mismatches)
        | set(grouped_surrogate_physics)
        | set(grouped_surrogate_calibrations)
        | set(grouped_branch_validity_receipts)
        | set(grouped_sensor_alignments)
        | set(grouped_replay_validity_receipts)
        | set(grouped_render_receipts)
        | set(grouped_outcomes)
    )
    for directory in all_dirs:
        world_states = grouped_world_states.get(directory, [])
        runtime_receipt_manifests = grouped_runtime_receipt_manifests.get(directory, [])
        execution_contracts = grouped_execution_contracts.get(directory, [])
        adaptations = grouped_adaptations.get(directory, [])
        backend_bindings = grouped_backend_bindings.get(directory, [])
        asset_contracts = grouped_asset_contracts.get(directory, [])
        gen2sim_receipts = grouped_gen2sim_receipts.get(directory, [])
        backend_bridge_receipts = grouped_backend_bridges.get(directory, [])
        backend_work_orders = grouped_backend_work_orders.get(directory, [])
        backend_runtime_receipts = grouped_backend_runtime.get(directory, [])
        backend_adapter_receipts = grouped_backend_adapter.get(directory, [])
        backend_launch_receipts = grouped_backend_launch.get(directory, [])
        backend_outcome_receipts = grouped_backend_outcome.get(directory, [])
        backend_shadow_receipts = grouped_backend_shadow.get(directory, [])
        calibrations = grouped_calibrations.get(directory, [])
        task_measurements = grouped_task_measurements.get(directory, [])
        sim_real_gaps = grouped_sim_real_gaps.get(directory, [])
        backend_mismatches = grouped_backend_mismatches.get(directory, [])
        surrogate_physics = grouped_surrogate_physics.get(directory, [])
        surrogate_calibrations = grouped_surrogate_calibrations.get(directory, [])
        branch_validity_receipts = grouped_branch_validity_receipts.get(directory, [])
        sensor_alignments = grouped_sensor_alignments.get(directory, [])
        replay_validity_receipts = grouped_replay_validity_receipts.get(directory, [])
        render_receipts = grouped_render_receipts.get(directory, [])
        outcomes = grouped_outcomes.get(directory, [])
        for world_state in world_states:
            physics_context = _mapping(world_state.get("physics_context"))
            metadata = _mapping(physics_context.get("metadata"))
            bundle: Dict[str, Any] = {
                "bundle_id": str(world_state.get("state_id", "")) or directory.name,
                "world_state": dict(world_state),
            }
            if runtime_receipt_manifests:
                bundle["runtime_receipt_manifest"] = dict(runtime_receipt_manifests[-1])
            if execution_contracts:
                bundle["physics_execution_contract"] = dict(execution_contracts[-1])
            elif isinstance(world_state.get("physics_execution_contract"), Mapping):
                bundle["physics_execution_contract"] = dict(
                    _mapping(world_state.get("physics_execution_contract"))
                )
            if adaptations:
                bundle["physics_adaptation_receipt"] = dict(adaptations[-1])
            if backend_bindings:
                bundle["backend_execution_binding_receipt"] = dict(backend_bindings[-1])
            if asset_contracts:
                bundle["robot_asset_contract_receipt"] = dict(asset_contracts[-1])
            if gen2sim_receipts:
                bundle["gen2sim_admission_receipt"] = dict(gen2sim_receipts[-1])
            if backend_bridge_receipts:
                bundle["backend_runtime_bridge_receipt"] = dict(backend_bridge_receipts[-1])
            if backend_work_orders:
                bundle["backend_runtime_work_orders"] = [
                    dict(item) for item in backend_work_orders
                ]
            if backend_runtime_receipts:
                bundle["backend_runtime_execution_receipt"] = dict(backend_runtime_receipts[-1])
            if backend_adapter_receipts:
                bundle["backend_runtime_adapter_receipt"] = dict(backend_adapter_receipts[-1])
            if backend_launch_receipts:
                bundle["backend_runtime_launch_receipt"] = dict(backend_launch_receipts[-1])
            if backend_outcome_receipts:
                bundle["backend_runtime_outcome_receipt"] = dict(backend_outcome_receipts[-1])
            if backend_shadow_receipts:
                bundle["backend_shadow_execution_receipt"] = dict(backend_shadow_receipts[-1])
            if calibrations:
                bundle["physics_calibration_receipt"] = dict(calibrations[-1])
            if task_measurements:
                bundle["task_measurement_receipt"] = dict(task_measurements[-1])
            if sim_real_gaps:
                bundle["sim_real_gap_receipt"] = dict(sim_real_gaps[-1])
            if backend_mismatches:
                bundle["backend_mismatch_receipt"] = dict(backend_mismatches[-1])
            if surrogate_physics:
                bundle["surrogate_physics_receipt"] = dict(surrogate_physics[-1])
            if surrogate_calibrations:
                bundle["surrogate_calibration_receipt"] = dict(surrogate_calibrations[-1])
            if branch_validity_receipts:
                bundle["branch_validity_receipts"] = [
                    dict(item) for item in branch_validity_receipts
                ]
            if sensor_alignments:
                bundle["sensor_alignment_receipt"] = dict(sensor_alignments[-1])
            if replay_validity_receipts:
                bundle["replay_validity_receipts"] = [
                    dict(item) for item in replay_validity_receipts
                ]
            if render_receipts:
                bundle["render_provider_receipts"] = [dict(item) for item in render_receipts]
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
    runtime_receipt_manifests = [
        dict(payload)
        for payload in rows
        if str(payload.get("version", payload.get("schema_version", "")) or "")
        == "sim_synth_runtime_receipt_manifest_v1"
    ]
    execution_contracts = [
        dict(payload)
        for payload in rows
        if str(payload.get("version", payload.get("schema_version", "")) or "")
        == "physics_execution_contract_v1"
    ]
    adaptations = [
        dict(payload)
        for payload in rows
        if str(payload.get("version", payload.get("schema_version", "")) or "")
        == "physics_adaptation_receipt_v1"
    ]
    backend_bindings = [
        dict(payload)
        for payload in rows
        if str(payload.get("version", payload.get("schema_version", "")) or "")
        == "backend_execution_binding_receipt_v1"
    ]
    asset_contracts = [
        dict(payload)
        for payload in rows
        if str(payload.get("version", payload.get("schema_version", "")) or "")
        == "robot_asset_contract_receipt_v1"
    ]
    gen2sim_receipts = [
        dict(payload)
        for payload in rows
        if str(payload.get("version", payload.get("schema_version", "")) or "")
        == "gen2sim_admission_receipt_v1"
    ]
    backend_bridge_receipts = [
        dict(payload)
        for payload in rows
        if str(payload.get("version", payload.get("schema_version", "")) or "")
        == "backend_runtime_bridge_receipt_v1"
    ]
    backend_work_orders = [
        dict(payload)
        for payload in rows
        if str(payload.get("version", payload.get("schema_version", "")) or "")
        == "backend_runtime_work_order_receipt_v1"
    ]
    backend_runtime_receipts = [
        dict(payload)
        for payload in rows
        if str(payload.get("version", payload.get("schema_version", "")) or "")
        == "backend_runtime_execution_receipt_v1"
    ]
    backend_adapter_receipts = [
        dict(payload)
        for payload in rows
        if str(payload.get("version", payload.get("schema_version", "")) or "")
        == "backend_runtime_adapter_receipt_v1"
    ]
    backend_launch_receipts = [
        dict(payload)
        for payload in rows
        if str(payload.get("version", payload.get("schema_version", "")) or "")
        == "backend_runtime_launch_receipt_v1"
    ]
    backend_outcome_receipts = [
        dict(payload)
        for payload in rows
        if str(payload.get("version", payload.get("schema_version", "")) or "")
        == "backend_runtime_outcome_receipt_v1"
    ]
    backend_shadow_receipts = [
        dict(payload)
        for payload in rows
        if str(payload.get("version", payload.get("schema_version", "")) or "")
        == "backend_shadow_execution_receipt_v1"
    ]
    calibrations = [
        dict(payload)
        for payload in rows
        if str(payload.get("version", payload.get("schema_version", "")) or "")
        == "physics_calibration_receipt_v1"
    ]
    task_measurements = [
        dict(payload)
        for payload in rows
        if str(payload.get("version", payload.get("schema_version", "")) or "")
        == "task_measurement_receipt_v1"
    ]
    sim_real_gaps = [
        dict(payload)
        for payload in rows
        if str(payload.get("version", payload.get("schema_version", "")) or "")
        == "sim_real_gap_receipt_v1"
    ]
    backend_mismatches = [
        dict(payload)
        for payload in rows
        if str(payload.get("version", payload.get("schema_version", "")) or "")
        == "backend_mismatch_receipt_v1"
    ]
    surrogate_physics_receipts = [
        dict(payload)
        for payload in rows
        if str(payload.get("version", payload.get("schema_version", "")) or "")
        == "surrogate_physics_receipt_v1"
    ]
    surrogate_calibration_receipts = [
        dict(payload)
        for payload in rows
        if str(payload.get("version", payload.get("schema_version", "")) or "")
        == "surrogate_calibration_receipt_v1"
    ]
    branch_validity_receipts = [
        dict(payload)
        for payload in rows
        if str(payload.get("version", payload.get("schema_version", "")) or "")
        == "branch_validity_receipt_v1"
    ]
    sensor_alignment_receipts = [
        dict(payload)
        for payload in rows
        if str(payload.get("version", payload.get("schema_version", "")) or "")
        == "sensor_alignment_receipt_v1"
    ]
    replay_validity_receipts = [
        dict(payload)
        for payload in rows
        if str(payload.get("version", payload.get("schema_version", "")) or "")
        == "replay_validity_receipt_v1"
    ]
    render_receipts = [
        dict(payload)
        for payload in rows
        if str(payload.get("version", payload.get("schema_version", "")) or "")
        == "render_provider_receipt_v1"
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
        if runtime_receipt_manifests:
            bundle["runtime_receipt_manifest"] = runtime_receipt_manifests[-1]
        if execution_contracts:
            bundle["physics_execution_contract"] = execution_contracts[-1]
        elif isinstance(world_state.get("physics_execution_contract"), Mapping):
            bundle["physics_execution_contract"] = _mapping(
                world_state.get("physics_execution_contract")
            )
        if adaptations:
            bundle["physics_adaptation_receipt"] = adaptations[-1]
        if backend_bindings:
            bundle["backend_execution_binding_receipt"] = backend_bindings[-1]
        if asset_contracts:
            bundle["robot_asset_contract_receipt"] = asset_contracts[-1]
        if gen2sim_receipts:
            bundle["gen2sim_admission_receipt"] = gen2sim_receipts[-1]
        if backend_bridge_receipts:
            bundle["backend_runtime_bridge_receipt"] = backend_bridge_receipts[-1]
        if backend_work_orders:
            bundle["backend_runtime_work_orders"] = backend_work_orders
        if backend_runtime_receipts:
            bundle["backend_runtime_execution_receipt"] = backend_runtime_receipts[-1]
        if backend_adapter_receipts:
            bundle["backend_runtime_adapter_receipt"] = backend_adapter_receipts[-1]
        if backend_launch_receipts:
            bundle["backend_runtime_launch_receipt"] = backend_launch_receipts[-1]
        if backend_outcome_receipts:
            bundle["backend_runtime_outcome_receipt"] = backend_outcome_receipts[-1]
        if backend_shadow_receipts:
            bundle["backend_shadow_execution_receipt"] = backend_shadow_receipts[-1]
        if calibrations:
            bundle["physics_calibration_receipt"] = calibrations[-1]
        if task_measurements:
            bundle["task_measurement_receipt"] = task_measurements[-1]
        if sim_real_gaps:
            bundle["sim_real_gap_receipt"] = sim_real_gaps[-1]
        if backend_mismatches:
            bundle["backend_mismatch_receipt"] = backend_mismatches[-1]
        if surrogate_physics_receipts:
            bundle["surrogate_physics_receipt"] = surrogate_physics_receipts[-1]
        if surrogate_calibration_receipts:
            bundle["surrogate_calibration_receipt"] = surrogate_calibration_receipts[-1]
        if branch_validity_receipts:
            bundle["branch_validity_receipts"] = branch_validity_receipts
        if sensor_alignment_receipts:
            bundle["sensor_alignment_receipt"] = sensor_alignment_receipts[-1]
        if replay_validity_receipts:
            bundle["replay_validity_receipts"] = replay_validity_receipts
        if render_receipts:
            bundle["render_provider_receipts"] = render_receipts
        if outcomes:
            bundle["simulation_outcome_receipts"] = outcomes
        bundles.append(bundle)
    return bundles



_RUNTIME_MANIFEST_FAMILY_KEYS: dict[str, tuple[str, str]] = {
    "physics_adaptation_receipt_v1": ("physics_adaptation_receipt", "single"),
    "gen2sim_admission_receipt_v1": ("gen2sim_admission_receipt", "single"),
    "backend_execution_binding_receipt_v1": ("backend_execution_binding_receipt", "single"),
    "robot_asset_contract_receipt_v1": ("robot_asset_contract_receipt", "single"),
    "backend_runtime_bridge_receipt_v1": ("backend_runtime_bridge_receipt", "single"),
    "backend_runtime_work_order_receipt_v1": ("backend_runtime_work_orders", "list"),
    "backend_runtime_execution_receipt_v1": ("backend_runtime_execution_receipt", "single"),
    "backend_runtime_adapter_receipt_v1": ("backend_runtime_adapter_receipt", "single"),
    "backend_runtime_launch_receipt_v1": ("backend_runtime_launch_receipt", "single"),
    "backend_runtime_outcome_receipt_v1": ("backend_runtime_outcome_receipt", "single"),
    "backend_shadow_execution_receipt_v1": ("backend_shadow_execution_receipt", "single"),
    "physics_calibration_receipt_v1": ("physics_calibration_receipt", "single"),
    "task_measurement_receipt_v1": ("task_measurement_receipt", "single"),
    "sim_real_gap_receipt_v1": ("sim_real_gap_receipt", "single"),
    "backend_mismatch_receipt_v1": ("backend_mismatch_receipt", "single"),
    "surrogate_physics_receipt_v1": ("surrogate_physics_receipt", "single"),
    "surrogate_calibration_receipt_v1": ("surrogate_calibration_receipt", "single"),
    "branch_validity_receipt_v1": ("branch_validity_receipts", "list"),
    "sensor_alignment_receipt_v1": ("sensor_alignment_receipt", "single"),
    "replay_validity_receipt_v1": ("replay_validity_receipts", "list"),
    "render_provider_receipt_v1": ("render_provider_receipts", "list"),
    "simulation_outcome_receipt_v1": ("simulation_outcome_receipts", "list"),
}


def _bundle_family_count(bundle: Mapping[str, Any], family: str) -> int:
    key_mode = _RUNTIME_MANIFEST_FAMILY_KEYS.get(family)
    if key_mode is None:
        return 0
    key, mode = key_mode
    if mode == "list":
        return len(_mapping_list(bundle.get(key)))
    return 1 if _mapping(bundle.get(key)) else 0


def validate_runtime_receipt_manifest(
    bundle: Mapping[str, Any],
) -> Dict[str, Any]:
    """Validate runtime receipt-manifest claims against a harvested bundle."""

    bundle_mapping = _mapping(bundle)
    manifest = _mapping(bundle_mapping.get("runtime_receipt_manifest"))
    if not manifest:
        return {
            "version": "runtime_receipt_manifest_validation_v1",
            "manifest_id": "",
            "validation_status": "manifest_missing",
            "mismatched_families": [],
            "missing_required_families": [],
            "actual_receipt_family_counts": {},
        }
    manifest_counts = {
        str(key): int(value)
        for key, value in _mapping(manifest.get("receipt_family_counts")).items()
    }
    actual_counts = {
        family: _bundle_family_count(bundle_mapping, family)
        for family in sorted(manifest_counts)
    }
    mismatches = [
        {
            "family": family,
            "manifest_count": int(manifest_counts.get(family, 0)),
            "actual_count": int(actual_counts.get(family, 0)),
        }
        for family in sorted(manifest_counts)
        if int(manifest_counts.get(family, 0)) != int(actual_counts.get(family, 0))
    ]
    missing_required = list(manifest.get("missing_required_families") or [])
    validation_status = "validated"
    if missing_required:
        validation_status = "manifest_declares_missing_required"
    if mismatches:
        validation_status = "manifest_count_mismatch"
    return {
        "version": "runtime_receipt_manifest_validation_v1",
        "manifest_id": manifest.get("manifest_id", ""),
        "validation_status": validation_status,
        "mismatched_families": mismatches,
        "missing_required_families": missing_required,
        "actual_receipt_family_counts": actual_counts,
    }

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
        physics_execution_contract = _mapping(
            bundle_mapping.get("physics_execution_contract")
            or world_state.get("physics_execution_contract")
        )
        runtime_receipt_manifest = _mapping(bundle_mapping.get("runtime_receipt_manifest"))
        runtime_receipt_manifest_validation = validate_runtime_receipt_manifest(
            bundle_mapping
        )
        world_state_metadata = _mapping(world_state.get("metadata"))
        compiled_receipt_inventory = _mapping(
            world_state_metadata.get("compiled_receipt_inventory")
        )
        runtime_depth_projection = _mapping(
            world_state_metadata.get("runtime_depth_projection")
            or compiled_receipt_inventory.get("runtime_depth_projection")
        )
        if not jobs or not physics_context:
            continue
        physics_metadata = _mapping(physics_context.get("metadata"))
        adaptation_receipt = _mapping(bundle_mapping.get("physics_adaptation_receipt"))
        backend_binding_receipt = _mapping(bundle_mapping.get("backend_execution_binding_receipt"))
        robot_asset_contract_receipt = _mapping(bundle_mapping.get("robot_asset_contract_receipt"))
        sensor_alignment_receipt = _mapping(bundle_mapping.get("sensor_alignment_receipt"))
        gen2sim_admission_receipt = _mapping(bundle_mapping.get("gen2sim_admission_receipt"))
        backend_runtime_bridge_receipt = _mapping(
            bundle_mapping.get("backend_runtime_bridge_receipt")
        )
        backend_runtime_execution_receipt = _mapping(
            bundle_mapping.get("backend_runtime_execution_receipt")
        )
        backend_runtime_execution_metadata = _mapping(
            backend_runtime_execution_receipt.get("metadata")
        )
        backend_runtime_bundle = _mapping(backend_runtime_execution_metadata.get("runtime_bundle"))
        backend_runtime_binding = _mapping(
            backend_runtime_execution_metadata.get("runtime_binding")
            or backend_runtime_bundle.get("runtime_binding")
        )
        backend_runtime_adapter_receipt = _mapping(
            bundle_mapping.get("backend_runtime_adapter_receipt")
        )
        backend_runtime_adapter_metadata = _mapping(backend_runtime_adapter_receipt.get("metadata"))
        backend_runtime_adapter_realization = _mapping(
            backend_runtime_adapter_metadata.get("realization")
        )
        backend_runtime_local_adapter_invocation = _mapping(
            backend_runtime_adapter_metadata.get("local_adapter_invocation")
        )
        backend_runtime_local_adapter_result = _mapping(
            backend_runtime_adapter_metadata.get("local_adapter_result")
        )
        backend_runtime_launch_receipt = _mapping(
            bundle_mapping.get("backend_runtime_launch_receipt")
        )
        backend_runtime_launch_metadata = _mapping(
            backend_runtime_launch_receipt.get("metadata")
        )
        backend_runtime_outcome_receipt = _mapping(
            bundle_mapping.get("backend_runtime_outcome_receipt")
        )
        backend_runtime_outcome_metadata = _mapping(
            backend_runtime_outcome_receipt.get("metadata")
        )
        structured_outputs = _mapping(
            backend_runtime_outcome_metadata.get("structured_outputs")
        )
        selected_ref_validation = _mapping(
            backend_runtime_outcome_metadata.get("selected_ref_validation")
        )
        upstream_runtime_pack = _mapping(
            backend_runtime_bundle.get("upstream_runtime_pack")
        ) or _mapping(backend_runtime_bridge_receipt.get("metadata", {})).get(
            "upstream_runtime_pack", {}
        )
        upstream_runtime_pack = _mapping(upstream_runtime_pack)
        backend_shadow_execution_receipt = _mapping(
            bundle_mapping.get("backend_shadow_execution_receipt")
        )
        calibration_receipt = _mapping(
            bundle_mapping.get("physics_calibration_receipt")
            or bundle_mapping.get("physics_calibration")
        )
        task_measurement_receipt = _mapping(bundle_mapping.get("task_measurement_receipt"))
        sim_real_gap_receipt = _mapping(bundle_mapping.get("sim_real_gap_receipt"))
        backend_mismatch_receipt = _mapping(bundle_mapping.get("backend_mismatch_receipt"))
        surrogate_physics_receipt = _mapping(bundle_mapping.get("surrogate_physics_receipt"))
        surrogate_calibration_receipt = _mapping(
            bundle_mapping.get("surrogate_calibration_receipt")
        )
        sensor_alignment_receipt = _mapping(bundle_mapping.get("sensor_alignment_receipt"))
        branch_validity_receipts = _mapping_list(
            bundle_mapping.get("branch_validity_receipts")
        )
        replay_validity_receipts = _mapping_list(
            bundle_mapping.get("replay_validity_receipts")
        )
        branch_reject_reasons = sorted(
            {
                str(reason)
                for receipt in branch_validity_receipts
                for reason in list(receipt.get("reject_reasons") or [])
                if str(reason)
            }
        )
        replay_reject_reasons = sorted(
            {
                str(reason)
                for receipt in replay_validity_receipts
                for reason in list(receipt.get("reject_reasons") or [])
                if str(reason)
            }
        )
        if calibration_receipt:
            target_source = "runtime_receipt"
        elif backend_runtime_outcome_receipt and _runtime_outcome_can_drive_target_source(
            backend_runtime_outcome_receipt,
            backend_runtime_outcome_metadata,
        ):
            target_source = "external_runtime_outcome_receipt"
        elif backend_runtime_launch_receipt and str(
            backend_runtime_launch_receipt.get("launch_status", "")
        ) in {"launch_completed", "launch_failed"}:
            target_source = "external_launch_receipt"
        elif backend_runtime_execution_receipt:
            target_source = "concrete_runtime_receipt"
        elif backend_shadow_execution_receipt:
            target_source = "shadow_runtime_receipt"
        else:
            target_source = "wm_planning_state"
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
                    or adaptation_receipt.get("domain_randomization_profile")
                    or physics_context.get("domain_randomization_regime")
                    or "steady_state"
                ),
                "target_hardware_class": str(
                    robot_asset_contract_receipt.get("target_hardware_class")
                    or adaptation_receipt.get("target_hardware_class")
                    or _mapping(physics_metadata.get("backend_adapter")).get("target_hardware_class")
                    or "unknown"
                ),
                "target_system_identification_profile": str(
                    adaptation_receipt.get("system_identification_profile") or "unknown"
                ),
                "target_source": target_source,
                "promotion_stage": str(helper_status.get("promotion_stage") or "shadow_candidate"),
                "metadata": {
                    "bundle_index": bundle_index,
                    "world_state_id": world_state.get("state_id"),
                    "physics_execution_contract_id": physics_execution_contract.get("contract_id"),
                    "physics_route_status": physics_execution_contract.get("route_status"),
                    "physics_requested_backend": physics_execution_contract.get("requested_backend"),
                    "physics_resolved_backend": physics_execution_contract.get("resolved_backend"),
                    "compiled_receipt_inventory_id": compiled_receipt_inventory.get("inventory_id"),
                    "runtime_receipt_manifest_id": runtime_receipt_manifest.get(
                        "manifest_id"
                    ),
                    "runtime_receipt_manifest_status": runtime_receipt_manifest.get(
                        "manifest_status"
                    ),
                    "runtime_receipt_missing_required_families": list(
                        runtime_receipt_manifest.get("missing_required_families") or []
                    ),
                    "runtime_receipt_emitted_count": runtime_receipt_manifest.get(
                        "emitted_receipt_count"
                    ),
                    "runtime_receipt_family_counts": _mapping(
                        runtime_receipt_manifest.get("receipt_family_counts")
                    ),
                    "runtime_receipt_manifest_validation_status": runtime_receipt_manifest_validation.get(
                        "validation_status"
                    ),
                    "runtime_receipt_manifest_mismatched_families": list(
                        runtime_receipt_manifest_validation.get("mismatched_families")
                        or []
                    ),
                    "compiled_runtime_binding_status": runtime_depth_projection.get("binding_status"),
                    "compiled_runtime_bridge_status": runtime_depth_projection.get("bridge_status"),
                    "compiled_runtime_pack_status": runtime_depth_projection.get(
                        "upstream_runtime_pack_status"
                    ),
                    "adaptation_receipt_id": adaptation_receipt.get("receipt_id"),
                    "task_measurement_receipt_id": task_measurement_receipt.get("receipt_id"),
                    "task_measurement_benchmark_gate_ready": task_measurement_receipt.get(
                        "benchmark_gate_ready"
                    ),
                    "task_measurement_values": _mapping(
                        task_measurement_receipt.get("measurement_values")
                    ),
                    "sim_real_gap_receipt_id": sim_real_gap_receipt.get("receipt_id"),
                    "sim_real_gap_status": sim_real_gap_receipt.get("status"),
                    "sim_real_gap_score": sim_real_gap_receipt.get("gap_score"),
                    "sim_real_realism_confidence": sim_real_gap_receipt.get(
                        "realism_confidence"
                    ),
                    "backend_mismatch_receipt_id": backend_mismatch_receipt.get("receipt_id"),
                    "backend_mismatch_status": backend_mismatch_receipt.get("status"),
                    "backend_mismatch_score": backend_mismatch_receipt.get("mismatch_score"),
                    "backend_calibration_staleness_score": backend_mismatch_receipt.get(
                        "calibration_staleness_score"
                    ),
                    "surrogate_physics_receipt_id": surrogate_physics_receipt.get(
                        "receipt_id"
                    ),
                    "surrogate_forecast_status": surrogate_physics_receipt.get(
                        "forecast_status"
                    ),
                    "surrogate_confidence": surrogate_physics_receipt.get(
                        "surrogate_confidence"
                    ),
                    "surrogate_calibration_receipt_id": surrogate_calibration_receipt.get(
                        "receipt_id"
                    ),
                    "surrogate_calibration_status": surrogate_calibration_receipt.get(
                        "calibration_status"
                    ),
                    "surrogate_calibration_score": surrogate_calibration_receipt.get(
                        "calibration_score"
                    ),
                    "branch_validity_receipt_ids": [
                        receipt.get("receipt_id") for receipt in branch_validity_receipts
                    ],
                    "branch_validity_admissible_count": sum(
                        1 for receipt in branch_validity_receipts if receipt.get("admissible")
                    ),
                    "branch_validity_reject_count": sum(
                        1 for receipt in branch_validity_receipts if not receipt.get("admissible")
                    ),
                    "branch_validity_reject_reasons": branch_reject_reasons,
                    "sensor_alignment_receipt_id": sensor_alignment_receipt.get("receipt_id"),
                    "sensor_alignment_status": sensor_alignment_receipt.get("status"),
                    "sensor_alignment_score": sensor_alignment_receipt.get("alignment_score"),
                    "sensor_alignment_checks": _mapping(
                        sensor_alignment_receipt.get("checks")
                    ),
                    "sensor_alignment_metrics": _mapping(
                        sensor_alignment_receipt.get("metrics")
                    ),
                    "replay_validity_receipt_ids": [
                        receipt.get("receipt_id") for receipt in replay_validity_receipts
                    ],
                    "replay_validity_reject_count": sum(
                        1 for receipt in replay_validity_receipts if receipt.get("reject_reasons")
                    ),
                    "replay_validity_reject_reasons": replay_reject_reasons,
                    "gen2sim_admission_receipt_id": gen2sim_admission_receipt.get("receipt_id"),
                    "gen2sim_benchmark_gate_ready": gen2sim_admission_receipt.get(
                        "benchmark_gate_ready"
                    ),
                    "gen2sim_admissible_branch_count": len(
                        gen2sim_admission_receipt.get("admissible_branch_ids") or []
                    ),
                    "gen2sim_blocked_branch_count": len(
                        gen2sim_admission_receipt.get("blocked_branch_ids") or []
                    ),
                    "backend_execution_binding_receipt_id": backend_binding_receipt.get("receipt_id"),
                    "robot_asset_contract_receipt_id": robot_asset_contract_receipt.get("receipt_id"),
                    "robot_asset_readiness_score": robot_asset_contract_receipt.get("readiness_score"),
                    "robot_asset_missing_assets": list(
                        robot_asset_contract_receipt.get("missing_assets") or []
                    ),
                    "backend_runtime_bridge_receipt_id": backend_runtime_bridge_receipt.get(
                        "receipt_id"
                    ),
                    "backend_runtime_bridge_status": backend_runtime_bridge_receipt.get(
                        "bridge_status"
                    ),
                    "bridge_execution_authority": backend_runtime_bridge_receipt.get(
                        "execution_authority"
                    ),
                    "bridge_transport_profile": backend_runtime_bridge_receipt.get(
                        "transport_profile"
                    ),
                    "bridge_readiness_score": backend_runtime_bridge_receipt.get(
                        "bridge_readiness_score"
                    ),
                    "bridge_missing_runtime_targets": list(
                        backend_runtime_bridge_receipt.get("metadata", {}).get(
                            "runtime_target_contract", {}
                        ).get("missing_required_target_ids", [])
                        or []
                    ),
                    "backend_binding_status": backend_binding_receipt.get("binding_status"),
                    "backend_runtime_execution_receipt_id": backend_runtime_execution_receipt.get("receipt_id"),
                    "backend_runtime_execution_status": backend_runtime_execution_receipt.get("execution_status"),
                    "backend_upstream_runtime_pack_status": upstream_runtime_pack.get(
                        "pack_status"
                    ),
                    "backend_upstream_runtime_profile_root": upstream_runtime_pack.get(
                        "profile_root"
                    ),
                    "backend_upstream_runtime_profile_git_metadata": _mapping(
                        upstream_runtime_pack.get("profile_git_metadata")
                    ),
                    "backend_upstream_runtime_profile_candidate_counts": _mapping(
                        upstream_runtime_pack.get("profile_candidate_counts")
                    ),
                    "backend_upstream_runtime_profile_install_preflight_status": upstream_runtime_pack.get(
                        "profile_install_preflight_status"
                    ),
                    "backend_upstream_runtime_profile_install_missing_components": list(
                        upstream_runtime_pack.get("profile_install_missing_components") or []
                    ),
                    "backend_upstream_runtime_profile_primary_entrypoint_ref": upstream_runtime_pack.get(
                        "profile_primary_entrypoint_ref"
                    ),
                    "backend_upstream_runtime_ready_surfaces": list(
                        upstream_runtime_pack.get("ready_surfaces") or []
                    ),
                    "backend_upstream_runtime_missing_components": list(
                        upstream_runtime_pack.get("missing_components") or []
                    ),
                    "backend_upstream_runtime_primary_policy_ref": upstream_runtime_pack.get(
                        "primary_policy_ref"
                    ),
                    "backend_upstream_runtime_primary_policy_ref_source": upstream_runtime_pack.get(
                        "primary_policy_ref_source"
                    ),
                    "backend_upstream_runtime_policy_candidate_evidence_summary": _mapping(
                        upstream_runtime_pack.get("policy_candidate_evidence_summary")
                    ),
                    "backend_upstream_runtime_primary_deploy_config_ref": upstream_runtime_pack.get(
                        "primary_deploy_config_ref"
                    ),
                    "backend_upstream_runtime_primary_deploy_config_ref_source": upstream_runtime_pack.get(
                        "primary_deploy_config_ref_source"
                    ),
                    "backend_upstream_runtime_deploy_candidate_evidence_summary": _mapping(
                        upstream_runtime_pack.get("deploy_candidate_evidence_summary")
                    ),
                    "backend_upstream_runtime_primary_runtime_report_ref": upstream_runtime_pack.get(
                        "primary_runtime_report_ref"
                    ),
                    "backend_upstream_runtime_primary_runtime_report_ref_source": upstream_runtime_pack.get(
                        "primary_runtime_report_ref_source"
                    ),
                    "backend_upstream_runtime_runtime_report_candidate_evidence_summary": _mapping(
                        upstream_runtime_pack.get("runtime_report_candidate_evidence_summary")
                    ),
                    "backend_upstream_runtime_verified_asset_ids": list(
                        upstream_runtime_pack.get("verified_asset_ids") or []
                    ),
                    "backend_upstream_runtime_declared_only_asset_ids": list(
                        upstream_runtime_pack.get("declared_only_asset_ids") or []
                    ),
                    "backend_upstream_runtime_existing_motion_sources": list(
                        upstream_runtime_pack.get("existing_motion_sources") or []
                    ),
                    "backend_runtime_binding_status": backend_runtime_binding.get(
                        "binding_status"
                    ),
                    "backend_runtime_binding_selected_profile": backend_runtime_binding.get(
                        "selected_profile"
                    ),
                    "backend_runtime_layout_usable_profiles": list(
                        _mapping(backend_runtime_bridge_receipt.get("metadata")).get(
                            "runtime_layout_usable_profiles", []
                        )
                        or []
                    ),
                    "backend_runtime_layout_install_ready_profiles": list(
                        _mapping(
                            _mapping(backend_runtime_bridge_receipt.get("metadata")).get(
                                "runtime_layout_contract", {}
                            )
                        ).get("install_ready_profiles", [])
                        or []
                    ),
                    "backend_runtime_layout_install_partial_profiles": list(
                        _mapping(
                            _mapping(backend_runtime_bridge_receipt.get("metadata")).get(
                                "runtime_layout_contract", {}
                            )
                        ).get("install_partial_profiles", [])
                        or []
                    ),
                    "backend_runtime_layout_install_blocked_profiles": list(
                        _mapping(
                            _mapping(backend_runtime_bridge_receipt.get("metadata")).get(
                                "runtime_layout_contract", {}
                            )
                        ).get("install_blocked_profiles", [])
                        or []
                    ),
                    "backend_runtime_binding_selected_policy_ref": backend_runtime_binding.get(
                        "selected_policy_ref"
                    ),
                    "backend_runtime_binding_selected_policy_ref_source": backend_runtime_binding.get(
                        "selected_policy_ref_source"
                    ),
                    "backend_runtime_binding_selected_deploy_config": backend_runtime_binding.get(
                        "selected_deploy_config"
                    ),
                    "backend_runtime_binding_selected_deploy_config_source": backend_runtime_binding.get(
                        "selected_deploy_config_source"
                    ),
                    "backend_runtime_binding_selected_runtime_report": backend_runtime_binding.get(
                        "selected_runtime_report"
                    ),
                    "backend_runtime_binding_selected_runtime_report_source": backend_runtime_binding.get(
                        "selected_runtime_report_source"
                    ),
                    "backend_runtime_binding_selected_launch_root": backend_runtime_binding.get(
                        "selected_launch_root"
                    ),
                    "backend_runtime_binding_selected_profile_install_preflight_status": backend_runtime_binding.get(
                        "selected_profile_install_preflight_status"
                    ),
                    "backend_runtime_binding_selected_profile_install_missing_components": list(
                        backend_runtime_binding.get("selected_profile_install_missing_components")
                        or []
                    ),
                    "backend_runtime_binding_selected_profile_primary_entrypoint_ref": backend_runtime_binding.get(
                        "selected_profile_primary_entrypoint_ref"
                    ),
                    "backend_runtime_binding_host_preflight_status": backend_runtime_binding.get(
                        "host_preflight_status"
                    ),
                    "backend_runtime_binding_host_preflight_missing_components": list(
                        backend_runtime_binding.get("host_preflight_missing_components") or []
                    ),
                    "backend_runtime_binding_host_preflight_ready_components": list(
                        backend_runtime_binding.get("host_preflight_ready_components") or []
                    ),
                    "backend_runtime_binding_host_preflight_verified_components": list(
                        backend_runtime_binding.get("host_preflight_verified_components") or []
                    ),
                    "backend_runtime_binding_host_preflight_symbolic_components": list(
                        backend_runtime_binding.get("host_preflight_symbolic_components") or []
                    ),
                    "backend_runtime_binding_selected_verified_target_ids": list(
                        backend_runtime_binding.get("selected_verified_target_ids") or []
                    ),
                    "backend_runtime_binding_selected_partial_target_ids": list(
                        backend_runtime_binding.get("selected_partial_target_ids") or []
                    ),
                    "backend_runtime_binding_selected_ref_evidence": _mapping(
                        backend_runtime_binding.get("selected_ref_evidence")
                    ),
                    "backend_runtime_binding_selected_target_ref_evidence": _mapping(
                        backend_runtime_binding.get("selected_target_ref_evidence")
                    ),
                    "backend_runtime_binding_missing_components": list(
                        backend_runtime_binding.get("missing_components") or []
                    ),
                    "backend_runtime_adapter_receipt_id": backend_runtime_adapter_receipt.get("receipt_id"),
                    "backend_runtime_adapter_status": backend_runtime_adapter_receipt.get("adapter_status"),
                    "backend_runtime_adapter_execution_path": backend_runtime_adapter_receipt.get("execution_path"),
                    "backend_runtime_adapter_realization_path": backend_runtime_adapter_realization.get(
                        "realization_path"
                    ),
                    "backend_runtime_adapter_realization_status": backend_runtime_adapter_realization.get(
                        "realization_status"
                    ),
                    "backend_runtime_local_adapter_invocation_status": backend_runtime_local_adapter_invocation.get(
                        "invocation_status"
                    ),
                    "backend_runtime_local_adapter_result_status": backend_runtime_local_adapter_result.get(
                        "result_status"
                    ),
                    "backend_runtime_launch_receipt_id": backend_runtime_launch_receipt.get("receipt_id"),
                    "backend_runtime_launch_status": backend_runtime_launch_receipt.get("launch_status"),
                    "backend_runtime_launch_executed": backend_runtime_launch_receipt.get("executed"),
                    "backend_runtime_launch_missing_preconditions": list(
                        backend_runtime_launch_metadata.get("missing_preconditions") or []
                    ),
                    "backend_runtime_launch_notes": list(
                        backend_runtime_launch_metadata.get("notes") or []
                    ),
                    "backend_runtime_outcome_receipt_id": backend_runtime_outcome_receipt.get("receipt_id"),
                    "backend_runtime_outcome_status": backend_runtime_outcome_receipt.get("outcome_status"),
                    "backend_runtime_output_count": backend_runtime_outcome_receipt.get(
                        "harvested_output_count"
                    ),
                    "backend_runtime_ready_surfaces": list(
                        structured_outputs.get("ready_surfaces") or []
                    ),
                    "backend_runtime_primary_policy_ref": structured_outputs.get(
                        "primary_policy_ref"
                    ),
                    "backend_runtime_selected_ref_validation_status": selected_ref_validation.get(
                        "status"
                    ),
                    "backend_runtime_selected_ref_validation_mismatched_components": list(
                        selected_ref_validation.get("mismatched_components") or []
                    ),
                    "backend_runtime_selected_ref_validation_missing_components": list(
                        selected_ref_validation.get("missing_components") or []
                    ),
                    "backend_runtime_metric_keys": list(
                        structured_outputs.get("metric_keys") or []
                    ),
                    "backend_shadow_execution_receipt_id": backend_shadow_execution_receipt.get("receipt_id"),
                    "backend_shadow_execution_status": backend_shadow_execution_receipt.get("execution_status"),
                    "backend_shadow_harvest_mode": _mapping(
                        backend_shadow_execution_receipt.get("metadata")
                    ).get("shadow_harvest_mode"),
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
        physics_execution_contract = _mapping(
            bundle_mapping.get("physics_execution_contract")
            or world_state.get("physics_execution_contract")
        )
        runtime_receipt_manifest = _mapping(bundle_mapping.get("runtime_receipt_manifest"))
        runtime_receipt_manifest_validation = validate_runtime_receipt_manifest(
            bundle_mapping
        )
        world_state_metadata = _mapping(world_state.get("metadata"))
        compiled_receipt_inventory = _mapping(
            world_state_metadata.get("compiled_receipt_inventory")
        )
        runtime_depth_projection = _mapping(
            world_state_metadata.get("runtime_depth_projection")
            or compiled_receipt_inventory.get("runtime_depth_projection")
        )
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
        render_receipts = {
            str(receipt.get("branch_plan_id")): receipt
            for receipt in _mapping_list(bundle_mapping.get("render_provider_receipts"))
            if str(receipt.get("branch_plan_id"))
        }
        branch_validity_receipts = {
            str(receipt.get("branch_plan_id")): receipt
            for receipt in _mapping_list(bundle_mapping.get("branch_validity_receipts"))
            if str(receipt.get("branch_plan_id"))
        }
        replay_validity_receipts = {
            str(receipt.get("branch_plan_id")): receipt
            for receipt in _mapping_list(bundle_mapping.get("replay_validity_receipts"))
            if str(receipt.get("branch_plan_id"))
        }
        robot_asset_contract_receipt = _mapping(bundle_mapping.get("robot_asset_contract_receipt"))
        sensor_alignment_receipt = _mapping(bundle_mapping.get("sensor_alignment_receipt"))
        gen2sim_admission_receipt = _mapping(bundle_mapping.get("gen2sim_admission_receipt"))
        backend_runtime_bridge_receipt = _mapping(
            bundle_mapping.get("backend_runtime_bridge_receipt")
        )
        backend_runtime_launch_receipt = _mapping(
            bundle_mapping.get("backend_runtime_launch_receipt")
        )
        backend_runtime_launch_metadata = _mapping(
            backend_runtime_launch_receipt.get("metadata")
        )
        backend_runtime_execution_receipt = _mapping(
            bundle_mapping.get("backend_runtime_execution_receipt")
        )
        backend_runtime_execution_metadata = _mapping(
            backend_runtime_execution_receipt.get("metadata")
        )
        backend_runtime_bundle = _mapping(backend_runtime_execution_metadata.get("runtime_bundle"))
        backend_runtime_binding = _mapping(
            backend_runtime_execution_metadata.get("runtime_binding")
            or backend_runtime_bundle.get("runtime_binding")
        )
        backend_runtime_adapter_receipt = _mapping(
            bundle_mapping.get("backend_runtime_adapter_receipt")
        )
        backend_runtime_adapter_metadata = _mapping(backend_runtime_adapter_receipt.get("metadata"))
        backend_runtime_adapter_realization = _mapping(
            backend_runtime_adapter_metadata.get("realization")
        )
        backend_runtime_local_adapter_invocation = _mapping(
            backend_runtime_adapter_metadata.get("local_adapter_invocation")
        )
        backend_runtime_local_adapter_result = _mapping(
            backend_runtime_adapter_metadata.get("local_adapter_result")
        )
        backend_runtime_outcome_receipt = _mapping(
            bundle_mapping.get("backend_runtime_outcome_receipt")
        )
        backend_runtime_outcome_metadata = _mapping(
            backend_runtime_outcome_receipt.get("metadata")
        )
        structured_outputs = _mapping(
            backend_runtime_outcome_metadata.get("structured_outputs")
        )
        selected_ref_validation = _mapping(
            backend_runtime_outcome_metadata.get("selected_ref_validation")
        )
        backend_shadow_execution_receipt = _mapping(
            bundle_mapping.get("backend_shadow_execution_receipt")
        )
        calibration_receipt = _mapping(
            bundle_mapping.get("physics_calibration_receipt")
            or bundle_mapping.get("physics_calibration")
        )
        adaptation_receipt = _mapping(bundle_mapping.get("physics_adaptation_receipt"))
        task_measurement_receipt = _mapping(bundle_mapping.get("task_measurement_receipt"))
        sim_real_gap_receipt = _mapping(bundle_mapping.get("sim_real_gap_receipt"))
        backend_mismatch_receipt = _mapping(bundle_mapping.get("backend_mismatch_receipt"))
        surrogate_physics_receipt = _mapping(bundle_mapping.get("surrogate_physics_receipt"))
        surrogate_calibration_receipt = _mapping(
            bundle_mapping.get("surrogate_calibration_receipt")
        )
        upstream_runtime_pack = _mapping(
            backend_runtime_bundle.get("upstream_runtime_pack")
        ) or _mapping(backend_runtime_bridge_receipt.get("metadata", {})).get(
            "upstream_runtime_pack", {}
        )
        upstream_runtime_pack = _mapping(upstream_runtime_pack)
        for plan_index, plan in enumerate(_mapping_list(world_state.get("synthetic_branch_plans"))):
            plan_id = str(plan.get("plan_id", ""))
            source_job_id = str(plan.get("source_job_id", ""))
            job = jobs.get(source_job_id)
            if not plan_id or job is None:
                continue
            plan_metadata = _mapping(plan.get("metadata"))
            helper_trace = _mapping(plan_metadata.get("branch_helper_trace"))
            render_provider = _mapping(plan.get("render_provider"))
            helper_status = _mapping(plan_metadata.get("branch_helper_status"))
            outcome = _mapping(outcomes.get(plan_id))
            outcome_metadata = _mapping(outcome.get("metadata"))
            render_receipt = _mapping(render_receipts.get(plan_id))
            branch_validity_receipt = _mapping(branch_validity_receipts.get(plan_id))
            replay_validity_receipt = _mapping(replay_validity_receipts.get(plan_id))
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
                    "target_render_provider_kind": str(
                        render_receipt.get("provider_kind")
                        or render_provider.get("provider_kind")
                        or "unknown"
                    ),
                    "target_render_provider_status": str(
                        render_receipt.get("provider_status")
                        or render_provider.get("provider_status")
                        or "unknown"
                    ),
                    "target_render_materialization_status": str(
                        render_receipt.get("materialization_status")
                        or render_provider.get("materialization_status")
                        or ""
                    ),
                    "target_render_materialization_mode": str(
                        render_receipt.get("materialization_mode") or ""
                    ),
                    "target_source": target_source,
                    "promotion_stage": str(helper_status.get("promotion_stage") or "shadow_candidate"),
                    "metadata": {
                        "bundle_index": bundle_index,
                        "world_state_id": world_state.get("state_id"),
                        "physics_execution_contract_id": physics_execution_contract.get(
                            "contract_id"
                        ),
                        "physics_route_status": physics_execution_contract.get("route_status"),
                        "physics_requested_backend": physics_execution_contract.get(
                            "requested_backend"
                        ),
                        "physics_resolved_backend": physics_execution_contract.get(
                            "resolved_backend"
                        ),
                        "compiled_receipt_inventory_id": compiled_receipt_inventory.get(
                            "inventory_id"
                        ),
                        "runtime_receipt_manifest_id": runtime_receipt_manifest.get(
                            "manifest_id"
                        ),
                        "runtime_receipt_manifest_status": runtime_receipt_manifest.get(
                            "manifest_status"
                        ),
                        "runtime_receipt_missing_required_families": list(
                            runtime_receipt_manifest.get("missing_required_families") or []
                        ),
                        "runtime_receipt_emitted_count": runtime_receipt_manifest.get(
                            "emitted_receipt_count"
                        ),
                        "runtime_receipt_family_counts": _mapping(
                            runtime_receipt_manifest.get("receipt_family_counts")
                        ),
                        "runtime_receipt_manifest_validation_status": runtime_receipt_manifest_validation.get(
                            "validation_status"
                        ),
                        "runtime_receipt_manifest_mismatched_families": list(
                            runtime_receipt_manifest_validation.get("mismatched_families")
                            or []
                        ),
                        "compiled_runtime_binding_status": runtime_depth_projection.get(
                            "binding_status"
                        ),
                        "compiled_runtime_bridge_status": runtime_depth_projection.get(
                            "bridge_status"
                        ),
                        "compiled_runtime_pack_status": runtime_depth_projection.get(
                            "upstream_runtime_pack_status"
                        ),
                        "branch_plan_id": plan_id,
                        "branch_helper_resolution": str(
                            plan_metadata.get("branch_helper_resolution") or ""
                        ),
                        "branch_helper_resolution_reason": str(
                            plan_metadata.get("branch_helper_resolution_reason") or ""
                        ),
                        "branch_helper_payload_applied": bool(
                            plan_metadata.get("branch_helper_payload_applied", False)
                        ),
                        "branch_helper_trace_generation_mode": str(
                            helper_trace.get("generation_mode") or ""
                        ),
                        "branch_helper_trace_expected_yield_score": _clip01(
                            helper_trace.get("expected_yield_score"),
                            0.0,
                        ),
                        "scene_hierarchy_ref": _mapping(
                            plan_metadata.get("scene_hierarchy_ref")
                        ),
                        "scene_materialization_status": str(
                            plan_metadata.get("scene_materialization_status") or ""
                        ),
                        "branch_validity_receipt_id": branch_validity_receipt.get(
                            "receipt_id"
                        ),
                        "branch_validity_score": branch_validity_receipt.get(
                            "validity_score"
                        ),
                        "branch_admission_score": branch_validity_receipt.get(
                            "admission_score"
                        ),
                        "branch_validity_admissible": branch_validity_receipt.get(
                            "admissible"
                        ),
                        "branch_validity_evidence_status": branch_validity_receipt.get(
                            "evidence_status"
                        ),
                        "branch_reject_reasons": list(
                            branch_validity_receipt.get("reject_reasons") or []
                        ),
                        "sensor_alignment_receipt_id": sensor_alignment_receipt.get(
                            "receipt_id"
                        ),
                        "sensor_alignment_status": sensor_alignment_receipt.get(
                            "status"
                        ),
                        "sensor_alignment_score": sensor_alignment_receipt.get(
                            "alignment_score"
                        ),
                        "sensor_alignment_checks": _mapping(
                            sensor_alignment_receipt.get("checks")
                        ),
                        "replay_validity_receipt_id": replay_validity_receipt.get(
                            "receipt_id"
                        ),
                        "replay_validity_score": replay_validity_receipt.get(
                            "validity_score"
                        ),
                        "replay_validity_status": replay_validity_receipt.get(
                            "status"
                        ),
                        "replay_task_consistency_score": replay_validity_receipt.get(
                            "task_consistency_score"
                        ),
                        "replay_transfer_consistency_score": replay_validity_receipt.get(
                            "transfer_consistency_score"
                        ),
                        "replay_reject_reasons": list(
                            replay_validity_receipt.get("reject_reasons") or []
                        ),
                        "adaptation_receipt_id": adaptation_receipt.get("receipt_id"),
                        "task_measurement_receipt_id": task_measurement_receipt.get(
                            "receipt_id"
                        ),
                        "task_measurement_benchmark_gate_ready": task_measurement_receipt.get(
                            "benchmark_gate_ready"
                        ),
                        "task_measurement_values": _mapping(
                            task_measurement_receipt.get("measurement_values")
                        ),
                        "sim_real_gap_receipt_id": sim_real_gap_receipt.get("receipt_id"),
                        "sim_real_gap_status": sim_real_gap_receipt.get("status"),
                        "sim_real_gap_score": sim_real_gap_receipt.get("gap_score"),
                        "sim_real_realism_confidence": sim_real_gap_receipt.get(
                            "realism_confidence"
                        ),
                        "backend_mismatch_receipt_id": backend_mismatch_receipt.get(
                            "receipt_id"
                        ),
                        "backend_mismatch_status": backend_mismatch_receipt.get("status"),
                        "backend_mismatch_score": backend_mismatch_receipt.get(
                            "mismatch_score"
                        ),
                        "backend_calibration_staleness_score": backend_mismatch_receipt.get(
                            "calibration_staleness_score"
                        ),
                        "surrogate_physics_receipt_id": surrogate_physics_receipt.get(
                            "receipt_id"
                        ),
                        "surrogate_forecast_status": surrogate_physics_receipt.get(
                            "forecast_status"
                        ),
                        "surrogate_confidence": surrogate_physics_receipt.get(
                            "surrogate_confidence"
                        ),
                        "surrogate_calibration_receipt_id": surrogate_calibration_receipt.get(
                            "receipt_id"
                        ),
                        "surrogate_calibration_status": surrogate_calibration_receipt.get(
                            "calibration_status"
                        ),
                        "surrogate_calibration_score": surrogate_calibration_receipt.get(
                            "calibration_score"
                        ),
                        "gen2sim_admission_receipt_id": gen2sim_admission_receipt.get("receipt_id"),
                        "gen2sim_benchmark_gate_ready": gen2sim_admission_receipt.get(
                            "benchmark_gate_ready"
                        ),
                        "gen2sim_admissible_branch_count": len(
                            gen2sim_admission_receipt.get("admissible_branch_ids") or []
                        ),
                        "gen2sim_blocked_branch_count": len(
                            gen2sim_admission_receipt.get("blocked_branch_ids") or []
                        ),
                        "robot_asset_contract_receipt_id": robot_asset_contract_receipt.get("receipt_id"),
                        "robot_asset_readiness_score": robot_asset_contract_receipt.get("readiness_score"),
                        "backend_runtime_bridge_receipt_id": backend_runtime_bridge_receipt.get(
                            "receipt_id"
                        ),
                        "backend_runtime_bridge_status": backend_runtime_bridge_receipt.get(
                            "bridge_status"
                        ),
                        "bridge_execution_authority": backend_runtime_bridge_receipt.get(
                            "execution_authority"
                        ),
                        "bridge_transport_profile": backend_runtime_bridge_receipt.get(
                            "transport_profile"
                        ),
                        "backend_runtime_launch_receipt_id": backend_runtime_launch_receipt.get(
                            "receipt_id"
                        ),
                        "backend_upstream_runtime_pack_status": upstream_runtime_pack.get(
                            "pack_status"
                        ),
                        "backend_upstream_runtime_profile_root": upstream_runtime_pack.get(
                            "profile_root"
                        ),
                        "backend_upstream_runtime_profile_git_metadata": _mapping(
                            upstream_runtime_pack.get("profile_git_metadata")
                        ),
                        "backend_upstream_runtime_profile_candidate_counts": _mapping(
                            upstream_runtime_pack.get("profile_candidate_counts")
                        ),
                        "backend_upstream_runtime_profile_install_preflight_status": upstream_runtime_pack.get(
                            "profile_install_preflight_status"
                        ),
                        "backend_upstream_runtime_profile_install_missing_components": list(
                            upstream_runtime_pack.get("profile_install_missing_components")
                            or []
                        ),
                        "backend_upstream_runtime_profile_primary_entrypoint_ref": upstream_runtime_pack.get(
                            "profile_primary_entrypoint_ref"
                        ),
                        "backend_upstream_runtime_ready_surfaces": list(
                            upstream_runtime_pack.get("ready_surfaces") or []
                        ),
                        "backend_upstream_runtime_missing_components": list(
                            upstream_runtime_pack.get("missing_components") or []
                        ),
                        "backend_upstream_runtime_primary_policy_ref": upstream_runtime_pack.get(
                            "primary_policy_ref"
                        ),
                        "backend_upstream_runtime_primary_policy_ref_source": upstream_runtime_pack.get(
                            "primary_policy_ref_source"
                        ),
                        "backend_upstream_runtime_policy_candidate_evidence_summary": _mapping(
                            upstream_runtime_pack.get("policy_candidate_evidence_summary")
                        ),
                        "backend_upstream_runtime_primary_deploy_config_ref": upstream_runtime_pack.get(
                            "primary_deploy_config_ref"
                        ),
                        "backend_upstream_runtime_primary_deploy_config_ref_source": upstream_runtime_pack.get(
                            "primary_deploy_config_ref_source"
                        ),
                        "backend_upstream_runtime_deploy_candidate_evidence_summary": _mapping(
                            upstream_runtime_pack.get("deploy_candidate_evidence_summary")
                        ),
                        "backend_upstream_runtime_primary_runtime_report_ref": upstream_runtime_pack.get(
                            "primary_runtime_report_ref"
                        ),
                        "backend_upstream_runtime_primary_runtime_report_ref_source": upstream_runtime_pack.get(
                            "primary_runtime_report_ref_source"
                        ),
                        "backend_upstream_runtime_runtime_report_candidate_evidence_summary": _mapping(
                            upstream_runtime_pack.get("runtime_report_candidate_evidence_summary")
                        ),
                        "backend_upstream_runtime_verified_asset_ids": list(
                            upstream_runtime_pack.get("verified_asset_ids") or []
                        ),
                        "backend_upstream_runtime_declared_only_asset_ids": list(
                            upstream_runtime_pack.get("declared_only_asset_ids") or []
                        ),
                        "backend_upstream_runtime_existing_motion_sources": list(
                            upstream_runtime_pack.get("existing_motion_sources") or []
                        ),
                        "backend_runtime_binding_status": backend_runtime_binding.get(
                            "binding_status"
                        ),
                        "backend_runtime_binding_selected_profile": backend_runtime_binding.get(
                            "selected_profile"
                        ),
                        "backend_runtime_layout_usable_profiles": list(
                            _mapping(backend_runtime_bridge_receipt.get("metadata")).get(
                                "runtime_layout_usable_profiles", []
                            )
                            or []
                        ),
                        "backend_runtime_layout_install_ready_profiles": list(
                            _mapping(
                                _mapping(backend_runtime_bridge_receipt.get("metadata")).get(
                                    "runtime_layout_contract", {}
                                )
                            ).get("install_ready_profiles", [])
                            or []
                        ),
                        "backend_runtime_layout_install_partial_profiles": list(
                            _mapping(
                                _mapping(backend_runtime_bridge_receipt.get("metadata")).get(
                                    "runtime_layout_contract", {}
                                )
                            ).get("install_partial_profiles", [])
                            or []
                        ),
                        "backend_runtime_layout_install_blocked_profiles": list(
                            _mapping(
                                _mapping(backend_runtime_bridge_receipt.get("metadata")).get(
                                    "runtime_layout_contract", {}
                                )
                            ).get("install_blocked_profiles", [])
                            or []
                        ),
                        "backend_runtime_binding_selected_policy_ref": backend_runtime_binding.get(
                            "selected_policy_ref"
                        ),
                        "backend_runtime_binding_selected_policy_ref_source": backend_runtime_binding.get(
                            "selected_policy_ref_source"
                        ),
                        "backend_runtime_binding_selected_deploy_config": backend_runtime_binding.get(
                            "selected_deploy_config"
                        ),
                        "backend_runtime_binding_selected_deploy_config_source": backend_runtime_binding.get(
                            "selected_deploy_config_source"
                        ),
                        "backend_runtime_binding_selected_runtime_report": backend_runtime_binding.get(
                            "selected_runtime_report"
                        ),
                        "backend_runtime_binding_selected_runtime_report_source": backend_runtime_binding.get(
                            "selected_runtime_report_source"
                        ),
                        "backend_runtime_binding_selected_launch_root": backend_runtime_binding.get(
                            "selected_launch_root"
                        ),
                        "backend_runtime_binding_selected_profile_install_preflight_status": backend_runtime_binding.get(
                            "selected_profile_install_preflight_status"
                        ),
                        "backend_runtime_binding_selected_profile_install_missing_components": list(
                            backend_runtime_binding.get(
                                "selected_profile_install_missing_components"
                            )
                            or []
                        ),
                        "backend_runtime_binding_selected_profile_primary_entrypoint_ref": backend_runtime_binding.get(
                            "selected_profile_primary_entrypoint_ref"
                        ),
                        "backend_runtime_binding_host_preflight_status": backend_runtime_binding.get(
                            "host_preflight_status"
                        ),
                        "backend_runtime_binding_host_preflight_missing_components": list(
                            backend_runtime_binding.get("host_preflight_missing_components")
                            or []
                        ),
                        "backend_runtime_binding_host_preflight_ready_components": list(
                            backend_runtime_binding.get("host_preflight_ready_components")
                            or []
                        ),
                        "backend_runtime_binding_host_preflight_verified_components": list(
                            backend_runtime_binding.get("host_preflight_verified_components")
                            or []
                        ),
                        "backend_runtime_binding_host_preflight_symbolic_components": list(
                            backend_runtime_binding.get("host_preflight_symbolic_components")
                            or []
                        ),
                        "backend_runtime_binding_selected_verified_target_ids": list(
                            backend_runtime_binding.get("selected_verified_target_ids") or []
                        ),
                        "backend_runtime_binding_selected_partial_target_ids": list(
                            backend_runtime_binding.get("selected_partial_target_ids") or []
                        ),
                        "backend_runtime_binding_selected_target_ref_evidence": _mapping(
                            backend_runtime_binding.get("selected_target_ref_evidence")
                        ),
                        "backend_runtime_binding_selected_ref_evidence": _mapping(
                            backend_runtime_binding.get("selected_ref_evidence")
                        ),
                        "backend_runtime_binding_missing_components": list(
                            backend_runtime_binding.get("missing_components") or []
                        ),
                        "backend_runtime_adapter_receipt_id": backend_runtime_adapter_receipt.get(
                            "receipt_id"
                        ),
                        "backend_runtime_adapter_status": backend_runtime_adapter_receipt.get(
                            "adapter_status"
                        ),
                        "backend_runtime_adapter_execution_path": backend_runtime_adapter_receipt.get(
                            "execution_path"
                        ),
                        "backend_runtime_adapter_realization_path": backend_runtime_adapter_realization.get(
                            "realization_path"
                        ),
                        "backend_runtime_adapter_realization_status": backend_runtime_adapter_realization.get(
                            "realization_status"
                        ),
                        "backend_runtime_local_adapter_invocation_status": backend_runtime_local_adapter_invocation.get(
                            "invocation_status"
                        ),
                        "backend_runtime_local_adapter_result_status": backend_runtime_local_adapter_result.get(
                            "result_status"
                        ),
                        "backend_runtime_launch_status": backend_runtime_launch_receipt.get(
                            "launch_status"
                        ),
                        "backend_runtime_launch_executed": backend_runtime_launch_receipt.get(
                            "executed"
                        ),
                        "backend_runtime_launch_missing_preconditions": list(
                            backend_runtime_launch_metadata.get("missing_preconditions")
                            or []
                        ),
                        "backend_runtime_launch_notes": list(
                            backend_runtime_launch_metadata.get("notes") or []
                        ),
                        "backend_runtime_outcome_receipt_id": backend_runtime_outcome_receipt.get(
                            "receipt_id"
                        ),
                        "backend_runtime_outcome_status": backend_runtime_outcome_receipt.get(
                            "outcome_status"
                        ),
                        "backend_runtime_output_count": backend_runtime_outcome_receipt.get(
                            "harvested_output_count"
                        ),
                        "backend_runtime_ready_surfaces": list(
                            structured_outputs.get("ready_surfaces") or []
                        ),
                        "backend_runtime_primary_policy_ref": structured_outputs.get(
                            "primary_policy_ref"
                        ),
                        "backend_runtime_selected_ref_validation_status": selected_ref_validation.get(
                            "status"
                        ),
                        "backend_runtime_selected_ref_validation_mismatched_components": list(
                            selected_ref_validation.get("mismatched_components") or []
                        ),
                        "backend_runtime_selected_ref_validation_missing_components": list(
                            selected_ref_validation.get("missing_components") or []
                        ),
                        "backend_runtime_metric_keys": list(
                            structured_outputs.get("metric_keys") or []
                        ),
                        "backend_shadow_execution_receipt_id": backend_shadow_execution_receipt.get(
                            "receipt_id"
                        ),
                        "backend_shadow_execution_status": backend_shadow_execution_receipt.get(
                            "execution_status"
                        ),
                        "backend_shadow_harvest_mode": _mapping(
                            backend_shadow_execution_receipt.get("metadata")
                        ).get("shadow_harvest_mode"),
                        "calibration_receipt_id": calibration_receipt.get("receipt_id"),
                        "calibration_quality_score": calibration_receipt.get("quality_score"),
                        "render_provider_receipt_id": render_receipt.get("receipt_id"),
                        "render_artifact_refs": list(render_receipt.get("artifact_refs") or []),
                        "render_unsatisfied_preconditions": list(
                            _mapping(render_receipt.get("metadata")).get(
                                "unsatisfied_preconditions",
                                [],
                            )
                            or []
                        ),
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
    "validate_runtime_receipt_manifest",
]
