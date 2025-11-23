"""
Loader for observation adapter configuration flags.
"""
from pathlib import Path
from typing import Any, Dict

import yaml

_DEFAULT_CFG = {
    "use_observation_components": False,
    "use_condition_vector": False,
}


def load_observation_config(path: str = "") -> Dict[str, Any]:
    cfg_path = Path(path) if path else Path(__file__).resolve().parents[2] / "config" / "observation.yaml"
    if not cfg_path.exists():
        return dict(_DEFAULT_CFG)
    try:
        with cfg_path.open("r") as f:
            cfg = yaml.safe_load(f) or {}
        merged: Dict[str, Any] = dict(_DEFAULT_CFG)
        merged.update(cfg)
        merged["use_observation_components"] = bool(merged.get("use_observation_components", False))
        merged["use_condition_vector"] = bool(merged.get("use_condition_vector", False))
        return merged
    except Exception:
        return dict(_DEFAULT_CFG)
