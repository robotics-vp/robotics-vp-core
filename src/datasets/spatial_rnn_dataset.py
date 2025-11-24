"""
Phase I spatial RNN dataset loader.

Builds deterministic feature sequences from Stage 2 segments + SIMA-2 stress
signals and aggregates them with the spatial RNN adapter stub.
"""
import hashlib
from typing import Any, Dict, List

import numpy as np

from src.datasets.base import Phase1DatasetBase, set_deterministic_seeds
from src.vision.spatial_rnn_adapter import run_spatial_rnn, tensor_to_json_safe


class SpatialRNNDataset(Phase1DatasetBase):
    name = "spatial_rnn_phase1"

    def __init__(self, *args, seed: int = 0, **kwargs) -> None:
        set_deterministic_seeds(seed)
        super().__init__(*args, seed=seed, **kwargs)

    def _augment_sample(self, sample: Dict[str, Any], idx: int) -> Dict[str, Any]:
        seq = self._build_sequence(sample, idx)
        hidden = run_spatial_rnn(seq)
        sample["sequence_features"] = [float(x) for x in np.asarray(seq, dtype=np.float32).flatten().tolist()]
        sample["spatial_rnn_hidden"] = tensor_to_json_safe(hidden)
        return sample

    def _build_sequence(self, sample: Dict[str, Any], idx: int) -> List[List[float]]:
        seg = sample.get("stage2_segments", {})
        stress = sample.get("sima2_stress", {})
        ros = sample.get("ros_stage2", {})

        base_vals = [
            float(seg.get("risk_level", 0.0)),
            float(seg.get("energy_intensity", 0.0)),
            float(seg.get("success_rate", 0.0)),
            float(stress.get("severity", 0.0)),
            float(idx),
        ]
        digest_src = f"{seg.get('segment_id','')}{stress.get('id','')}{ros.get('id','')}{self.seed}"
        digest = hashlib.sha256(digest_src.encode("utf-8")).digest()
        seq: List[List[float]] = []
        for offset in range(3):
            seq.append([v + int.from_bytes(digest[offset : offset + 2], "big") / 65535.0 for v in base_vals])
        return seq

