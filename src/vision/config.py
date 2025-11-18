"""
Loader for vision configuration (resolution, normalization, latent sizes).
"""
import yaml
from pathlib import Path
from typing import Any, Dict, List


_DEFAULT_CFG = {
    "input_resolution": [224, 224],
    "channels": 3,
    "dtype": "uint8",
    "crop_type": "center",
    "normalize_mean": [0.485, 0.456, 0.406],
    "normalize_std": [0.229, 0.224, 0.225],
    "latent_dim": 16,
    "model_name": "vision-stub",
}


def load_vision_config(path: str = "") -> Dict[str, Any]:
    cfg_path = Path(path) if path else Path(__file__).resolve().parents[2] / "config" / "vision.yaml"
    if not cfg_path.exists():
        return dict(_DEFAULT_CFG)
    try:
        with cfg_path.open("r") as f:
            cfg = yaml.safe_load(f) or {}
        merged = dict(_DEFAULT_CFG)
        merged.update(cfg)
        merged["input_resolution"] = _coerce_int_list(merged.get("input_resolution", [224, 224]), default=[224, 224])
        merged["normalize_mean"] = _coerce_float_list(merged.get("normalize_mean", _DEFAULT_CFG["normalize_mean"]))
        merged["normalize_std"] = _coerce_float_list(merged.get("normalize_std", _DEFAULT_CFG["normalize_std"]))
        merged["channels"] = int(merged.get("channels", 3))
        merged["latent_dim"] = int(merged.get("latent_dim", 16))
        merged["dtype"] = str(merged.get("dtype", "uint8"))
        merged["crop_type"] = str(merged.get("crop_type", "center"))
        merged["model_name"] = str(merged.get("model_name", "vision-stub"))
        return merged
    except Exception:
        return dict(_DEFAULT_CFG)


def _coerce_int_list(vals, default: List[int]) -> List[int]:
    try:
        return [int(x) for x in vals]
    except Exception:
        return list(default)


def _coerce_float_list(vals, default: List[float]) -> List[float]:
    try:
        return [float(x) for x in vals]
    except Exception:
        return list(default)
