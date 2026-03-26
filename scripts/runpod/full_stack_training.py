#!/usr/bin/env python3
"""Shared helpers for full-stack training backlog assessment and Runpod launch flows."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "scripts" / "runpod" / "FULL_STACK_TRAINING_BUNDLES.json"
SEARCH_ROOTS = (
    REPO_ROOT / "artifacts",
    REPO_ROOT / "results",
    REPO_ROOT / "data",
)


def load_bundle_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    path = Path(config_path or DEFAULT_CONFIG_PATH)
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_named_files(filename: str, roots: Sequence[Path]) -> Iterable[Path]:
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob(filename):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            yield path


def _json_load(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _count_jsonl_rows(path: Path) -> int:
    count = 0
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    count += 1
    except Exception:
        return 0
    return count


def _discover_replay_datasets() -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for summary_path in _iter_named_files("summary.json", SEARCH_ROOTS):
        if summary_path.parent.name != "replay_dataset":
            continue
        payload = _json_load(summary_path)
        if not payload:
            continue
        records.append(
            {
                "path": str(summary_path.parent.relative_to(REPO_ROOT)),
                "summary_path": str(summary_path.relative_to(REPO_ROOT)),
                "num_episodes": int(payload.get("num_episodes", 0) or 0),
                "num_steps": int(payload.get("num_steps", 0) or 0),
                "num_windows": int(payload.get("num_windows", 0) or 0),
                "dataset_digest": str(payload.get("dataset_digest", "")),
            }
        )
    return sorted(
        records,
        key=lambda row: (row["num_episodes"], row["num_steps"], row["num_windows"], row["path"]),
        reverse=True,
    )


def _discover_semantic_runtime_summaries() -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for summary_path in _iter_named_files("semantic_runtime_learning_summary.json", SEARCH_ROOTS):
        payload = _json_load(summary_path)
        if not payload:
            continue
        records.append(
            {
                "path": str(summary_path.relative_to(REPO_ROOT)),
                "row_count": int(payload.get("row_count", 0) or 0),
                "semantic_grounded_count": int(payload.get("semantic_grounded_count", 0) or 0),
            }
        )
    return sorted(records, key=lambda row: (row["row_count"], row["path"]), reverse=True)


def _discover_coverage_graphs() -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for graph_path in _iter_named_files("coverage_graph.json", SEARCH_ROOTS):
        summary_path = graph_path.with_name("coverage_summary.json")
        payload = _json_load(summary_path) if summary_path.exists() else {}
        records.append(
            {
                "path": str(graph_path.relative_to(REPO_ROOT)),
                "artifact_dir": str(graph_path.parent.relative_to(REPO_ROOT)),
                "covered_edges": int(payload.get("covered_edges", 0) or 0),
                "missing_edges": int(payload.get("missing_edges", 0) or 0),
                "total_edges": int(payload.get("total_edges", 0) or 0),
            }
        )
    return sorted(records, key=lambda row: (row["total_edges"], row["path"]), reverse=True)


def _discover_recap_datasets() -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for root in SEARCH_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*.jsonl"):
            path_str = str(path.relative_to(REPO_ROOT))
            lowered = path_str.lower()
            if "recap" not in lowered:
                continue
            if "/test_" in lowered or "_test_" in lowered:
                continue
            row_count = _count_jsonl_rows(path)
            records.append({"path": path_str, "row_count": row_count})
    return sorted(records, key=lambda row: (row["row_count"], row["path"]), reverse=True)


def _discover_fill_outcomes() -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for path in _iter_named_files("fill_outcomes.jsonl", SEARCH_ROOTS):
        records.append(
            {
                "path": str(path.relative_to(REPO_ROOT)),
                "row_count": _count_jsonl_rows(path),
            }
        )
    return sorted(records, key=lambda row: (row["row_count"], row["path"]), reverse=True)


def _discover_stage_roots() -> Dict[str, Dict[str, Any]]:
    roots = {
        "stage1": REPO_ROOT / "results" / "stage1_pipeline",
        "stage2": REPO_ROOT / "results" / "stage2_preview",
        "sima2": REPO_ROOT / "results" / "sima2_stress",
    }
    details: Dict[str, Dict[str, Any]] = {}
    for key, path in roots.items():
        file_count = sum(1 for child in path.rglob("*") if child.is_file()) if path.exists() else 0
        details[key] = {
            "path": str(path.relative_to(REPO_ROOT)),
            "exists": path.exists(),
            "file_count": file_count,
        }
    return details


def _discover_checkpoints() -> Dict[str, bool]:
    checkpoint_paths = {
        "trust_net": REPO_ROOT / "checkpoints" / "trust_net.pt",
        "latent_diffusion": REPO_ROOT / "checkpoints" / "latent_diffusion_zv.pt",
        "stable_world_model": REPO_ROOT / "checkpoints" / "stable_world_model.pt",
    }
    return {
        name: path.exists()
        for name, path in checkpoint_paths.items()
    }


def _count_files(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for child in path.rglob("*") if child.is_file())


def discover_workspace_state(repo_root: Optional[Path] = None) -> Dict[str, Any]:
    root = Path(repo_root or REPO_ROOT)
    replay_datasets = _discover_replay_datasets()
    semantic_runtime = _discover_semantic_runtime_summaries()
    coverage_graphs = _discover_coverage_graphs()
    recap_datasets = _discover_recap_datasets()
    fill_outcomes = _discover_fill_outcomes()
    stage_roots = _discover_stage_roots()
    checkpoints = _discover_checkpoints()
    datapack_file_count = _count_files(root / "data" / "datapacks")
    ontology_file_count = _count_files(root / "data" / "ontology")

    best_replay = replay_datasets[0] if replay_datasets else {}
    best_runtime = semantic_runtime[0] if semantic_runtime else {}
    best_recap = recap_datasets[0] if recap_datasets else {}
    best_fill = fill_outcomes[0] if fill_outcomes else {}

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(root),
        "replay_datasets": replay_datasets,
        "semantic_runtime_summaries": semantic_runtime,
        "coverage_graphs": coverage_graphs,
        "recap_datasets": recap_datasets,
        "fill_outcomes": fill_outcomes,
        "stage_roots": stage_roots,
        "checkpoints": checkpoints,
        "datapack_file_count": datapack_file_count,
        "ontology_file_count": ontology_file_count,
        "max_replay_episodes": int(best_replay.get("num_episodes", 0) or 0),
        "max_replay_steps": int(best_replay.get("num_steps", 0) or 0),
        "max_replay_windows": int(best_replay.get("num_windows", 0) or 0),
        "best_replay_dataset_path": str(best_replay.get("path", "")),
        "max_semantic_runtime_rows": int(best_runtime.get("row_count", 0) or 0),
        "best_semantic_runtime_path": str(best_runtime.get("path", "")),
        "coverage_graph_count": len(coverage_graphs),
        "best_coverage_graph_path": str(coverage_graphs[0]["path"]) if coverage_graphs else "",
        "max_recap_rows": int(best_recap.get("row_count", 0) or 0),
        "best_recap_dataset_path": str(best_recap.get("path", "")),
        "fill_outcome_store_count": len(fill_outcomes),
        "max_fill_outcome_rows": int(best_fill.get("row_count", 0) or 0),
        "best_fill_outcomes_path": str(best_fill.get("path", "")),
    }


def _missing_paths(required_paths: Sequence[str]) -> List[str]:
    missing: List[str] = []
    for rel_path in required_paths:
        if not (REPO_ROOT / rel_path).exists():
            missing.append(rel_path)
    return missing


def evaluate_bundles(config: Dict[str, Any], state: Dict[str, Any]) -> List[Dict[str, Any]]:
    assessments: List[Dict[str, Any]] = []
    bundles = sorted(config.get("bundles", []), key=lambda row: int(row.get("priority_rank", 999)))
    for bundle in bundles:
        readiness = dict(bundle.get("readiness", {}) or {})
        blockers: List[str] = []

        missing_paths = _missing_paths(readiness.get("require_paths", []))
        for rel_path in missing_paths:
            blockers.append(f"missing required path: {rel_path}")

        min_replay_episodes = int(readiness.get("min_replay_episodes", 0) or 0)
        if state["max_replay_episodes"] < min_replay_episodes:
            blockers.append(
                f"needs >= {min_replay_episodes} replay episodes, current {state['max_replay_episodes']}"
            )

        min_replay_steps = int(readiness.get("min_replay_steps", 0) or 0)
        if state["max_replay_steps"] < min_replay_steps:
            blockers.append(
                f"needs >= {min_replay_steps} replay steps, current {state['max_replay_steps']}"
            )

        min_semantic_runtime_rows = int(readiness.get("min_semantic_runtime_rows", 0) or 0)
        if state["max_semantic_runtime_rows"] < min_semantic_runtime_rows:
            blockers.append(
                f"needs >= {min_semantic_runtime_rows} semantic-runtime rows, current {state['max_semantic_runtime_rows']}"
            )

        min_coverage_graphs = int(readiness.get("min_coverage_graphs", 0) or 0)
        if state["coverage_graph_count"] < min_coverage_graphs:
            blockers.append(
                f"needs >= {min_coverage_graphs} coverage graphs, current {state['coverage_graph_count']}"
            )

        min_recap_rows = int(readiness.get("min_recap_rows", 0) or 0)
        if state["max_recap_rows"] < min_recap_rows:
            blockers.append(f"needs >= {min_recap_rows} RECAP rows, current {state['max_recap_rows']}")

        min_fill_outcome_rows = int(readiness.get("min_fill_outcome_rows", 0) or 0)
        if state["max_fill_outcome_rows"] < min_fill_outcome_rows:
            blockers.append(
                f"needs >= {min_fill_outcome_rows} fill-outcome rows, current {state['max_fill_outcome_rows']}"
            )

        if readiness.get("require_stage1_root") and not state["stage_roots"]["stage1"]["exists"]:
            blockers.append("needs results/stage1_pipeline")
        if readiness.get("require_stage2_root") and not state["stage_roots"]["stage2"]["exists"]:
            blockers.append("needs results/stage2_preview")
        if readiness.get("require_sima2_root") and not state["stage_roots"]["sima2"]["exists"]:
            blockers.append("needs results/sima2_stress")

        hourly_price = float(bundle.get("recommended_runpod", {}).get("hourly_price_usd", 0.0) or 0.0)
        hours = dict(bundle.get("estimated_hours", {}) or {})
        low_hours = float(hours.get("low", 0.0) or 0.0)
        high_hours = float(hours.get("high", 0.0) or 0.0)
        manual_only = bool(bundle.get("manual_only", False))
        manually_runnable = not blockers
        ready = manually_runnable and not manual_only
        assessments.append(
            {
                "bundle_id": str(bundle.get("bundle_id", "")),
                "title": str(bundle.get("title", "")),
                "priority_rank": int(bundle.get("priority_rank", 999)),
                "manual_only": manual_only,
                "manually_runnable": manually_runnable,
                "ready": ready,
                "blockers": blockers,
                "estimated_hours": {"low": low_hours, "high": high_hours},
                "estimated_cost_usd": {
                    "low": round(low_hours * hourly_price, 2),
                    "high": round(high_hours * hourly_price, 2),
                },
                "recommended_runpod": dict(bundle.get("recommended_runpod", {}) or {}),
                "notes": str(bundle.get("notes", "")),
                "preferred_internal_data": list(bundle.get("preferred_internal_data", []) or []),
                "preferred_external_data": list(bundle.get("preferred_external_data", []) or []),
            }
        )
    return assessments


def select_bundle(assessments: Sequence[Dict[str, Any]], bundle_id: str = "auto") -> Optional[Dict[str, Any]]:
    if bundle_id != "auto":
        for assessment in assessments:
            if assessment["bundle_id"] == bundle_id:
                return assessment
        return None
    ready = [row for row in assessments if row.get("ready")]
    if not ready:
        return None
    return sorted(ready, key=lambda row: int(row.get("priority_rank", 999)))[0]


def render_bundle_commands(
    config: Dict[str, Any],
    state: Dict[str, Any],
    bundle_id: str,
    run_id: str,
) -> List[str]:
    bundle = next((row for row in config.get("bundles", []) if row.get("bundle_id") == bundle_id), None)
    if bundle is None:
        raise KeyError(f"Unknown bundle_id: {bundle_id}")

    coverage_graphs = state.get("coverage_graphs", [])
    coverage_graph_args = " ".join(
        f"--coverage-graph {row['path']}"
        for row in coverage_graphs[:8]
    )
    coverage_dirs: List[str] = []
    for row in coverage_graphs[:8]:
        artifact_dir = str(row.get("artifact_dir", ""))
        if artifact_dir and artifact_dir not in coverage_dirs:
            coverage_dirs.append(artifact_dir)
    coverage_artifact_dir_args = " ".join(f"--artifact-dir {path}" for path in coverage_dirs)
    recap_dataset_args = " ".join(row["path"] for row in state.get("recap_datasets", [])[:8])

    bundle_output_root = f"artifacts/runpod_training/{run_id}"
    context = {
        "bundle_output_root": bundle_output_root,
        "run_id": run_id,
        "replay_dataset": state.get("best_replay_dataset_path", ""),
        "coverage_graph_args": coverage_graph_args,
        "coverage_artifact_dir_args": coverage_artifact_dir_args,
        "recap_dataset_args": recap_dataset_args,
        "fill_outcomes_path": state.get("best_fill_outcomes_path", ""),
        "stage1_root": state.get("stage_roots", {}).get("stage1", {}).get("path", ""),
        "stage2_root": state.get("stage_roots", {}).get("stage2", {}).get("path", ""),
        "sima2_root": state.get("stage_roots", {}).get("sima2", {}).get("path", ""),
    }
    return [str(command).format(**context).strip() for command in bundle.get("commands", [])]

