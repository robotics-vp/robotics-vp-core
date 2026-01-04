"""
Trust matrix utilities for SIMA-2 econ correlation.
"""
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional, List

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


    return tm_path


class EconCorrelator:
    """
    Computes statistical correlations between semantic tags and economic outcomes.
    
    Derives:
    - Trust Scores: P(Success | Tag)
    - Risk Premiums: E[Loss | High Risk] - E[Loss | Low Risk]
    - Recovery Values: E[Value | Recovery] - E[Value | Failure]
    """

    def compute_correlations(
        self,
        datapacks: List[Dict[str, Any]],
        econ_vectors: Dict[str, Dict[str, float]],
    ) -> Dict[str, Any]:
        """
        Compute trust matrix entries from observed data.

        Args:
            datapacks: List of datapack dictionaries (Stage 1 or 2).
            econ_vectors: Map of episode_id -> econ vector metrics.

        Returns:
            Trust matrix dictionary.
        """
        # Intermediate stats: tag -> {damage_sum, energy_sum, count, successes}
        stats: Dict[str, Dict[str, float]] = {}

        for pack in datapacks:
            ep_id = pack.get("episode_id")
            if not ep_id or ep_id not in econ_vectors:
                continue
            
            vec = econ_vectors[ep_id]
            damage = float(vec.get("damage_cost", 0.0))
            energy = float(vec.get("energy_cost", 0.0))
            success = bool(vec.get("success", False))

            tags = pack.get("semantic_tags", [])
            # Also include primitive breakdown if available
            if "primitives" in pack:
                for prim in pack["primitives"]:
                    tags.extend(prim.get("tags", []))
            
            unique_tags = set(tags)

            for tag in unique_tags:
                if tag not in stats:
                    stats[tag] = {"damage": 0.0, "energy": 0.0, "count": 0.0, "successes": 0.0}
                
                stats[tag]["damage"] += damage
                stats[tag]["energy"] += energy
                stats[tag]["count"] += 1.0
                if success:
                    stats[tag]["successes"] += 1.0

        # Compute final metrics
        matrix: Dict[str, Any] = {}
        
        # Accumulators for global correlations
        scene_qualities: List[float] = []
        mpl_uplifts: List[float] = []
        
        for tag, data in stats.items():
            count = data["count"]
            if count == 0:
                continue
            
            mean_damage = data["damage"] / count
            mean_energy = data["energy"] / count
            success_rate = data["successes"] / count
            
            # Trust score heuristic: 
            # High trust = High success rate + Low damage variance (simplified here as low mean damage)
            # Normalized roughly to [0, 1]
            trust_penalty = min(1.0, mean_damage / 100.0)  # penalize $100+ damage
            trust_score = success_rate * (1.0 - trust_penalty)

            matrix[tag] = TrustEntry(
                tag=tag,
                mean_damage=mean_damage,
                mean_energy=mean_energy,
                count=int(count),
                trust_score=trust_score,
            ).to_dict()
            
        # Compute Structural/IR Correlations if data exists
        # Re-iterate to capture event-level pairs for correlation
        import numpy as np
        
        for pack in datapacks:
            ep_id = pack.get("episode_id")
            if not ep_id or ep_id not in econ_vectors:
                continue
            
            vec = econ_vectors[ep_id]
            mpl = float(vec.get("mpl", vec.get("productivity", 0.0)))
            
            # Try to find scene quality
            # Could be in pack metadata or 'scene_ir_quality_score'
            quality = pack.get("scene_ir_quality_score")
            if quality is None:
                # Fallback: check nested metrics
                quality = pack.get("metrics", {}).get("scene_ir_quality")
                
            if quality is not None:
                scene_qualities.append(float(quality))
                mpl_uplifts.append(mpl)
        
        if len(scene_qualities) > 10:
             # Pearson correlation
             q_arr = np.array(scene_qualities)
             m_arr = np.array(mpl_uplifts)
             
             if q_arr.std() > 1e-6 and m_arr.std() > 1e-6:
                 correlation = np.corrcoef(q_arr, m_arr)[0, 1]
                 
                 stats_meta = {
                     "scene_quality_vs_mpl": float(correlation),
                     "sample_size": len(scene_qualities),
                     "p_value": None
                 }
                 
                 # Try to compute p-value if scipy is available
                 try:
                     from scipy.stats import pearsonr
                     r, p = pearsonr(q_arr, m_arr)
                     stats_meta["p_value"] = float(p)
                     stats_meta["scene_quality_vs_mpl"] = float(r) # usage of scipy result
                 except ImportError:
                     pass
                     
                 matrix["_meta_correlations"] = stats_meta
             else:
                 # Constant vector case
                 matrix["_meta_correlations"] = {
                     "scene_quality_vs_mpl": 0.0,
                     "sample_size": len(scene_qualities),
                     "warning": "Constant input vectors detected, correlation undefined"
                 }

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

