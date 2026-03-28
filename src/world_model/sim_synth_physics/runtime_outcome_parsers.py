"""Structured parsing of harvested upstream runtime artifacts."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    import yaml
except Exception:  # pragma: no cover - optional dependency
    yaml = None

from .common import safe_float, strings


POLICY_SUFFIXES = {".onnx", ".pt", ".pth", ".ckpt", ".safetensors"}
DEPLOY_SUFFIXES = {".yaml", ".yml", ".json", ".toml"}
ASSET_SUFFIXES = {".usd", ".usda", ".urdf"}
MOTION_SUFFIXES = {".npz", ".npy", ".bvh", ".pkl"}


def _classify_artifact(ref: str) -> str:
    path = Path(ref)
    suffix = path.suffix.lower()
    lowered = str(path).lower()
    name = path.name.lower()
    if suffix in POLICY_SUFFIXES:
        return "policy_checkpoint"
    if suffix in ASSET_SUFFIXES:
        return "robot_asset"
    if suffix in MOTION_SUFFIXES:
        if "retarget" in lowered:
            return "retargeting_bundle"
        return "motion_dataset"
    if "teleop/televuer" in lowered and suffix in {".pem", ".key", ".cnf"}:
        return "teleop_cert"
    if any(token in lowered for token in ("episode_", "recordings/", "teleop/utils/data", "dataset/", "replay/")):
        return "dataset_capture"
    if "generated/" in lowered:
        return "generated_capture"
    if suffix == ".csv":
        return "runtime_metrics"
    if suffix in DEPLOY_SUFFIXES:
        if any(token in lowered for token in ("metric", "summary", "result", "stats", "eval")):
            return "runtime_metrics"
        if any(token in name for token in ("deploy", "config", "task", "args", "manifest")):
            return "deploy_config"
        if any(token in lowered for token in ("logs/", "outputs/", "runs/")):
            return "runtime_metrics"
        return "deploy_config"
    return "other"


def _numeric_leaf_map(payload: Any, *, prefix: str = "", depth: int = 0) -> dict[str, float]:
    if depth > 2:
        return {}
    metrics: dict[str, float] = {}
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            child_key = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                metrics[child_key] = float(value)
            else:
                metrics.update(_numeric_leaf_map(value, prefix=child_key, depth=depth + 1))
    elif isinstance(payload, list):
        for index, value in enumerate(payload[:5]):
            child_key = f"{prefix}[{index}]"
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                metrics[child_key] = float(value)
            else:
                metrics.update(_numeric_leaf_map(value, prefix=child_key, depth=depth + 1))
    return metrics


def _load_text_payload(path: Path) -> Any:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if suffix in {".yaml", ".yml"} and yaml is not None:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    return None


def _load_csv_metrics(path: Path) -> dict[str, float]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        return {}
    metrics: dict[str, list[float]] = defaultdict(list)
    for row in rows[:128]:
        for key, value in row.items():
            if value in (None, ""):
                continue
            try:
                metrics[str(key)].append(float(value))
            except Exception:
                continue
    return {
        key: sum(values) / float(len(values))
        for key, values in metrics.items()
        if values
    }


def _extract_metrics(ref: str, category: str) -> dict[str, float]:
    path = Path(ref)
    if not path.exists() or category != "runtime_metrics":
        return {}
    if path.suffix.lower() == ".csv":
        return _load_csv_metrics(path)
    payload = _load_text_payload(path)
    return _numeric_leaf_map(payload)


def _episode_dirs(refs: Sequence[str]) -> list[str]:
    roots: list[str] = []
    seen: set[str] = set()
    for ref in refs:
        path = Path(ref)
        for parent in [path] + list(path.parents):
            if parent.name.startswith("episode_"):
                resolved = str(parent.resolve())
                if resolved not in seen:
                    seen.add(resolved)
                    roots.append(resolved)
                break
    return roots


def summarize_runtime_output_artifacts(artifact_refs: Sequence[str]) -> dict[str, Any]:
    categorized: dict[str, list[str]] = defaultdict(list)
    metric_values: dict[str, list[float]] = defaultdict(list)
    for ref in strings(artifact_refs):
        category = _classify_artifact(ref)
        if ref not in categorized[category]:
            categorized[category].append(ref)
        for key, value in _extract_metrics(ref, category).items():
            metric_values[key].append(safe_float(value, 0.0))
    metric_summary = {
        key: sum(values) / float(len(values))
        for key, values in metric_values.items()
        if values
    }
    dataset_refs = categorized["dataset_capture"] + categorized["generated_capture"]
    surface_ready = {
        "policy_surface_ready": bool(categorized["policy_checkpoint"]),
        "deploy_surface_ready": bool(categorized["deploy_config"]),
        "metrics_surface_ready": bool(categorized["runtime_metrics"]),
        "dataset_surface_ready": bool(dataset_refs),
        "asset_surface_ready": bool(categorized["robot_asset"]),
        "teleop_surface_ready": bool(categorized["teleop_cert"]),
        "motion_surface_ready": bool(categorized["motion_dataset"]),
        "retargeting_surface_ready": bool(categorized["retargeting_bundle"]),
    }
    return {
        "policy_checkpoint_refs": categorized["policy_checkpoint"],
        "deploy_config_refs": categorized["deploy_config"],
        "runtime_metrics_refs": categorized["runtime_metrics"],
        "dataset_capture_refs": dataset_refs,
        "dataset_episode_dirs": _episode_dirs(dataset_refs),
        "robot_asset_refs": categorized["robot_asset"],
        "teleop_cert_refs": categorized["teleop_cert"],
        "motion_dataset_refs": categorized["motion_dataset"],
        "retargeting_bundle_refs": categorized["retargeting_bundle"],
        "primary_policy_ref": (
            categorized["policy_checkpoint"][0] if categorized["policy_checkpoint"] else ""
        ),
        "primary_deploy_config_ref": (
            categorized["deploy_config"][0] if categorized["deploy_config"] else ""
        ),
        "metric_summary": metric_summary,
        "metric_keys": sorted(metric_summary),
        "surface_ready": surface_ready,
        "ready_surfaces": sorted(key for key, value in surface_ready.items() if value),
        "counts": {
            "policy_checkpoint_count": len(categorized["policy_checkpoint"]),
            "deploy_config_count": len(categorized["deploy_config"]),
            "runtime_metrics_count": len(categorized["runtime_metrics"]),
            "dataset_capture_count": len(dataset_refs),
            "dataset_episode_count": len(_episode_dirs(dataset_refs)),
            "robot_asset_count": len(categorized["robot_asset"]),
            "teleop_cert_count": len(categorized["teleop_cert"]),
            "motion_dataset_count": len(categorized["motion_dataset"]),
            "retargeting_bundle_count": len(categorized["retargeting_bundle"]),
        },
    }


__all__ = ["summarize_runtime_output_artifacts"]
