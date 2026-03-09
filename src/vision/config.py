"""
Loader for vision configuration (resolution, normalization, latent sizes).
"""

from __future__ import annotations

import yaml
from pathlib import Path
from typing import Any, Dict, List


_DEFAULT_CFG: Dict[str, object] = {
    "input_resolution": [224, 224],
    "channels": 3,
    "dtype": "uint8",
    "crop_type": "center",
    "normalize_mean": [0.485, 0.456, 0.406],
    "normalize_std": [0.229, 0.224, 0.225],
    "latent_dim": 16,
    "model_name": "vision-stub",
    "backbone": "stub",
    "use_bifpn": False,
    "use_spatial_rnn": False,
    "regnet_feature_dim": 8,
    "enable_conditioned_vision": False,
    "conditioned_vision_feature_dim": 8,
    "conditioned_vision_levels": ["P3", "P4", "P5"],
    "conditioned_vision_enable_conditioning": True,
}


def load_vision_config(path: str = "") -> Dict[str, Any]:
    cfg_path = (
        Path(path)
        if path
        else Path(__file__).resolve().parents[2] / "config" / "vision.yaml"
    )
    if not cfg_path.exists():
        return dict(_DEFAULT_CFG)
    try:
        with cfg_path.open("r") as f:
            loaded_cfg = yaml.safe_load(f) or {}
        cfg = loaded_cfg if isinstance(loaded_cfg, dict) else {}
        merged: Dict[str, object] = dict(_DEFAULT_CFG)
        merged.update(cfg)
        merged["input_resolution"] = _coerce_int_list(
            merged.get("input_resolution", [224, 224]), default=[224, 224]
        )
        merged["normalize_mean"] = _coerce_float_list(
            merged.get("normalize_mean", _DEFAULT_CFG["normalize_mean"]),
            default=[0.485, 0.456, 0.406],
        )
        merged["normalize_std"] = _coerce_float_list(
            merged.get("normalize_std", _DEFAULT_CFG["normalize_std"]),
            default=[0.229, 0.224, 0.225],
        )
        merged["channels"] = _coerce_int(merged.get("channels", 3), default=3)
        merged["latent_dim"] = _coerce_int(merged.get("latent_dim", 16), default=16)
        merged["dtype"] = str(merged.get("dtype", "uint8"))
        merged["crop_type"] = str(merged.get("crop_type", "center"))
        merged["model_name"] = str(merged.get("model_name", "vision-stub"))
        merged["backbone"] = str(merged.get("backbone", "stub"))
        merged["use_bifpn"] = bool(merged.get("use_bifpn", False))
        merged["use_spatial_rnn"] = bool(merged.get("use_spatial_rnn", False))
        merged["regnet_feature_dim"] = _coerce_int(
            merged.get("regnet_feature_dim", 8), default=8
        )
        merged["enable_conditioned_vision"] = bool(
            merged.get("enable_conditioned_vision", False)
        )
        merged["conditioned_vision_feature_dim"] = int(
            _coerce_int(
                merged.get(
                    "conditioned_vision_feature_dim",
                    merged.get("regnet_feature_dim", 8),
                ),
                default=8,
            )
        )
        merged["conditioned_vision_levels"] = _coerce_str_list(
            merged.get(
                "conditioned_vision_levels", _DEFAULT_CFG["conditioned_vision_levels"]
            ),
            default=["P3", "P4", "P5"],
        )
        merged["conditioned_vision_enable_conditioning"] = bool(
            merged.get("conditioned_vision_enable_conditioning", True)
        )
        return merged
    except Exception:
        return dict(_DEFAULT_CFG)


def _coerce_int(value: object, default: int) -> int:
    try:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            return int(value)
        return default
    except (TypeError, ValueError):
        return default


def _coerce_int_list(vals: object, default: List[int]) -> List[int]:
    try:
        if not isinstance(vals, (list, tuple)):
            return list(default)
        return [int(x) for x in vals]
    except Exception:
        return list(default)


def _coerce_float_list(vals: object, default: List[float]) -> List[float]:
    try:
        if not isinstance(vals, (list, tuple)):
            return list(default)
        return [float(x) for x in vals]
    except Exception:
        return list(default)


def _coerce_str_list(vals: object, default: List[str]) -> List[str]:
    try:
        if not isinstance(vals, (list, tuple)):
            return list(default)
        return [str(x) for x in vals]
    except Exception:
        return list(default)
