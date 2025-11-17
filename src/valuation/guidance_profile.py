from dataclasses import dataclass, asdict
from typing import List, Optional


@dataclass
class GuidanceProfile:
    # High-level classification
    is_good: bool
    quality_label: str  # e.g., "high_value", "medium", "low_value", "risky"

    # Why it is good/bad
    env_name: str
    engine_type: str
    task_type: str
    customer_segment: str
    objective_vector: List[float]
    main_driver: str  # throughput_gain, energy_efficiency, error_reduction, safety_margin

    # Scalar summary of economics
    delta_mpl: float
    delta_error: float
    delta_energy_Wh: float
    delta_J: float

    # Tags for VLA/SIMA
    semantic_tags: List[str]

    # Orchestrator provenance
    orchestrator_plan_id: Optional[str] = None
    orchestrator_step_index: Optional[int] = None

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**d)
