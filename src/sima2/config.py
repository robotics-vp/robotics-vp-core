"""
Centralized SIMA-2 configuration loader and provenance helpers.
"""
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import yaml


DEFAULT_CONFIG: Dict[str, Any] = {
    "backend_mode": "stub",
    "backend": {
        "id": "sima2_stub",
        "model_version": "sima2_stub_v1",
        "endpoint": "local_stub",
    },
    "task_distribution": {
        "stress_mix_v1": {
            "total_episodes": 10000,
            "distribution": {"drawer_open": 0.4, "dish_place": 0.4, "noise": 0.2},
            "params": {"failure_rate": 0.3, "recovery_rate": 0.1, "ood_rate": 0.05},
        },
        "edge_cases_v1": {
            "total_episodes": 1000,
            "distribution": {
                "empty": 0.25,
                "long_horizon": 0.25,
                "invalid_utf8": 0.25,
                "future_timestamp": 0.25,
            },
            "params": {
                "max_steps": 1_000_000,
                "include_invalid_strings": True,
                "include_future_timestamps": True,
            },
        },
    },
    "segmentation": {
        "min_segment_length": 2,
        "max_idle_gap": 3,
        "max_segment_length": 120,
        "risk_jump_delta": 1,
        "allow_risk_jumps": True,
    },
}


def _merge(base: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for k, v in (new or {}).items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = _merge(merged.get(k, {}), v)
        else:
            merged[k] = v
    return merged


def _json_safe(obj: Any) -> Any:
    """Best-effort JSON sanitization to keep provenance stable."""
    try:
        return json.loads(json.dumps(obj))
    except Exception:
        try:
            return json.loads(json.dumps(str(obj)))
        except Exception:
            return {}


def load_sima2_config(path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load SIMA-2 configuration, merging with defaults for missing keys.
    """
    cfg_path = Path(path) if path else Path(__file__).resolve().parents[2] / "config" / "sima2.yaml"
    cfg = deepcopy(DEFAULT_CONFIG)
    if cfg_path.exists():
        try:
            with cfg_path.open("r") as f:
                loaded = yaml.safe_load(f) or {}
            cfg = _merge(cfg, loaded)
        except Exception:
            # Fall back to defaults on parse errors to keep pipeline running.
            pass
    return cfg


def build_provenance(task_spec: Optional[Mapping[str, Any]] = None, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Construct a provenance payload describing which SIMA-2 backend produced an artifact.
    """
    cfg = deepcopy(config) if config is not None else load_sima2_config()
    backend_cfg = cfg.get("backend", {}) or {}
    mode = str(cfg.get("backend_mode", backend_cfg.get("mode", "stub")))
    task_spec_dict = dict(task_spec or {})
    if "task_id" not in task_spec_dict and "task" in task_spec_dict:
        task_spec_dict["task_id"] = task_spec_dict.get("task")
    return {
        "sima2_backend_id": backend_cfg.get("id", f"{mode}_backend"),
        "sima2_model_version": backend_cfg.get("model_version", "sima2_unknown"),
        "sima2_task_spec": _json_safe(task_spec_dict),
        "sima2_backend_mode": mode,
    }


def extract_provenance(record: Mapping[str, Any], fallback_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Pull provenance fields from a rollout/segment dict, defaulting to config values if missing.
    """
    cfg_prov = build_provenance({}, fallback_config) if fallback_config is not None else {}
    meta = {}
    if isinstance(record, Mapping):
        meta = record.get("metadata", {}) or {}
    sima2_backend_id = (
        record.get("sima2_backend_id")
        if isinstance(record, Mapping)
        else None
    ) or meta.get("sima2_backend_id") or cfg_prov.get("sima2_backend_id")
    sima2_model_version = (
        record.get("sima2_model_version") if isinstance(record, Mapping) else None
    ) or meta.get("sima2_model_version") or cfg_prov.get("sima2_model_version")
    sima2_backend_mode = (
        record.get("sima2_backend_mode") if isinstance(record, Mapping) else None
    ) or meta.get("sima2_backend_mode") or cfg_prov.get("sima2_backend_mode", "stub")
    task_spec = {}
    if isinstance(record, Mapping):
        task_spec = (
            record.get("sima2_task_spec")
            or record.get("task_spec")
            or meta.get("sima2_task_spec")
            or meta.get("task_spec")
            or {}
        )
    return {
        "sima2_backend_id": sima2_backend_id or f"{sima2_backend_mode}_backend",
        "sima2_model_version": sima2_model_version or "sima2_unknown",
        "sima2_task_spec": _json_safe(task_spec),
        "sima2_backend_mode": sima2_backend_mode,
    }


def get_segmentation_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Return segmentation config with defaults applied.
    """
    cfg = deepcopy(config) if config is not None else load_sima2_config()
    return deepcopy(cfg.get("segmentation", DEFAULT_CONFIG.get("segmentation", {})))
