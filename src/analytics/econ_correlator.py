"""
Trust matrix utilities for SIMA-2 econ correlation.
"""
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional

DEFAULT_TRUST_MATRIX_PATH = Path(__file__).resolve().parents[2] / "results" / "sima2" / "trust_matrix.json"


@dataclass
class TrustEntry:
    tag: str
    mean_damage: float
    mean_energy: float
    count: int
    trust_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _normalize_entry(tag: str, payload: Dict[str, Any]) -> TrustEntry:
    return TrustEntry(
        tag=payload.get("tag", tag),
        mean_damage=float(payload.get("mean_damage", 0.0)),
        mean_energy=float(payload.get("mean_energy", 0.0)),
        count=int(payload.get("count", 0)),
        trust_score=float(payload.get("trust_score", payload.get("trust_level", 0.0))),
    )


def load_trust_matrix(path: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """
    Load a JSON trust matrix artifact keyed by tag.
    """
    tm_path = Path(path) if path else DEFAULT_TRUST_MATRIX_PATH
    if not tm_path.exists():
        return {}
    try:
        with tm_path.open("r") as f:
            raw = json.load(f) or {}
    except Exception:
        return {}

    matrix: Dict[str, Dict[str, Any]] = {}
    for tag, payload in raw.items():
        try:
            entry = _normalize_entry(tag, payload if isinstance(payload, dict) else {})
            matrix[entry.tag] = entry.to_dict()
        except Exception:
            continue
    return matrix


def save_trust_matrix(matrix: Dict[str, Any], path: Optional[str] = None) -> Path:
    """
    Persist the trust matrix to disk under results/sima2 by default.
    """
    tm_path = Path(path) if path else DEFAULT_TRUST_MATRIX_PATH
    tm_path.parent.mkdir(parents=True, exist_ok=True)
    with tm_path.open("w") as f:
        json.dump(matrix, f, indent=2, sort_keys=True)
    return tm_path
