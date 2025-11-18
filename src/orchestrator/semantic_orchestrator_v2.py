"""
SemanticOrchestratorV2: advisory-only surface consuming SemanticSnapshot.
"""
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List
import json

from src.semantic.models import SemanticSnapshot
from src.utils.json_safe import to_json_safe


@dataclass
class OrchestratorAdvisory:
    task_id: str
    focus_objective_presets: List[str]
    sampler_strategy_overrides: Dict[str, float]
    datapack_priority_tags: List[str]
    safety_emphasis: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        return to_json_safe(asdict(self))


class SemanticOrchestratorV2:
    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        self.output_dir = self.config.get("output_dir", "results/orchestrator")
        self.write_to_file = self.config.get("write_to_file", True)

    def propose(self, snapshot: SemanticSnapshot) -> OrchestratorAdvisory:
        econ = snapshot.econ_slice
        meta = snapshot.meta_slice
        focus_presets = meta.presets or ["balanced"]

        strategy_overrides = {"balanced": 0.5, "frontier_prioritized": 0.3, "econ_urgency": 0.2}
        if econ.avg_wage_parity < 1.0:
            strategy_overrides["econ_urgency"] = 0.5
            strategy_overrides["frontier_prioritized"] = 0.3
            strategy_overrides["balanced"] = 0.2

        safety_emphasis = 0.3
        priority_tags: List[str] = []
        recap = snapshot.metadata.get("recap", {}) if snapshot.metadata else {}
        for tag in snapshot.semantic_tags:
            try:
                enrichment = tag.to_dict()
                supervision = enrichment.get("supervision_hints", {})
                if supervision.get("safety_critical"):
                    safety_emphasis = 0.8
                if enrichment.get("fragility_tags"):
                    priority_tags.append("fragility_tags")
                if enrichment.get("risk_tags"):
                    priority_tags.append("risk_tags")
            except Exception:
                continue
        priority_tags = sorted(list(set(priority_tags)))
        if recap:
            mean_good = float(recap.get("mean_goodness", 0.0))
            if mean_good > 0:
                strategy_overrides["frontier_prioritized"] = min(1.0, strategy_overrides.get("frontier_prioritized", 0.3) + 0.1)
            if mean_good < 0:
                strategy_overrides["balanced"] = min(1.0, strategy_overrides.get("balanced", 0.5) + 0.1)
            if recap.get("top_episodes"):
                priority_tags.append("recap_top")
            priority_tags = sorted(list(set(priority_tags)))

        advisory = OrchestratorAdvisory(
            task_id=snapshot.task_id,
            focus_objective_presets=sorted(list(set(focus_presets))),
            sampler_strategy_overrides=strategy_overrides,
            datapack_priority_tags=priority_tags,
            safety_emphasis=float(min(max(safety_emphasis, 0.0), 1.0)),
            metadata={"frontier_eps": econ.frontier_episodes, "recap": recap},
        )
        if self.write_to_file:
            self._write_advisory(advisory)
        return advisory

    def _write_advisory(self, advisory: OrchestratorAdvisory) -> None:
        import os

        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, f"advisories_{advisory.task_id}.jsonl")
        with open(path, "a") as f:
            f.write(json.dumps(advisory.to_json(), sort_keys=True))
            f.write("\n")


from typing import Optional


def load_latest_advisory(task_id: str, output_dir: str = "results/orchestrator") -> Optional[OrchestratorAdvisory]:
    import os
    path = os.path.join(output_dir, f"advisories_{task_id}.jsonl")
    if not os.path.exists(path):
        return None
    last = None
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                last = json.loads(line)
    if not last:
        return None
    return OrchestratorAdvisory(**last)
