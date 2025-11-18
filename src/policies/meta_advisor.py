"""
Heuristic MetaAdvisorPolicy wrapping existing MetaTransformer.
"""
from typing import Any, Dict

import numpy as np

from src.orchestrator.meta_transformer import MetaTransformer, MetaTransformerOutputs
from src.policies.interfaces import MetaAdvisorPolicy
from src.utils.json_safe import to_json_safe


class HeuristicMetaAdvisorPolicy(MetaAdvisorPolicy):
    def __init__(self):
        self._impl = MetaTransformer()

    def build_features(self, meta_slice: Any) -> Dict[str, Any]:
        return {"meta_slice": meta_slice}

    def evaluate(self, features: Dict[str, Any]) -> MetaTransformerOutputs:
        meta_slice = features.get("meta_slice")
        if isinstance(meta_slice, MetaTransformerOutputs):
            return meta_slice

        dino_features = np.array(features.get("dino_features", [0.0]), dtype=np.float32)
        vla_features = np.array(features.get("vla_features", [0.0]), dtype=np.float32)
        dino_conf = float(features.get("dino_confidence", 0.5))
        vla_conf = float(features.get("vla_confidence", 0.5))
        output = self._impl.forward(dino_features, vla_features, dino_conf=dino_conf, vla_conf=vla_conf)
        # Ensure metadata remains JSON-safe
        output.metadata = to_json_safe(getattr(output, "metadata", {}))
        return output
