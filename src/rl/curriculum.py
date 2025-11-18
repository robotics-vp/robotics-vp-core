"""
Stage 3.2: Advisory curriculum scheduler on top of DataPackRLSampler.

Determines which sampler strategy to use at each training step without touching
reward math, SAC/PPO code, or objective vectors. Purely advisory: it only
decides which episode descriptors to surface.
"""
import copy
import random
from typing import Any, Dict, List, Optional

from src.rl.episode_sampling import DataPackRLSampler
from src.utils.json_safe import to_json_safe


class DataPackCurriculum:
    """
    Simple deterministic curriculum wrapper over DataPackRLSampler.

    Phases:
        warmup        -> balanced
        skill_building-> balanced + frontier_prioritized mix
        frontier      -> frontier_prioritized + econ_urgency mix
        fine_tuning   -> econ_urgency (critical-priority enforced by sampler)
    """

    DEFAULT_BOUNDARIES = {
        "warmup": 0.15,
        "skill_building": 0.50,
        "frontier": 0.80,
        "fine_tuning": 1.00,
    }

    DEFAULT_MIX = {
        "skill_building": {"balanced": 0.6, "frontier_prioritized": 0.4},
        "frontier": {"frontier_prioritized": 0.7, "econ_urgency": 0.3},
    }

    def __init__(
        self,
        sampler: DataPackRLSampler,
        total_steps: int,
        config: Optional[Dict[str, Any]] = None,
        advisory: Optional[Any] = None,
    ) -> None:
        self.sampler = sampler
        self.total_steps = max(1, int(total_steps))
        self.config = config or {}
        self.boundaries = self._build_boundaries(self.config.get("phase_boundaries"))
        self.mix = self._build_mix(self.config.get("phase_mix"))
        self.base_seed = int(self.config.get("base_seed", 0))
        self.advisory = advisory

    def get_phase(self, step: int) -> str:
        """Return phase name for a given training step."""
        progress = max(0.0, min(1.0, step / float(self.total_steps)))
        if progress < self.boundaries["warmup"]:
            return "warmup"
        if progress < self.boundaries["skill_building"]:
            return "skill_building"
        if progress < self.boundaries["frontier"]:
            return "frontier"
        return "fine_tuning"

    def sample_batch(self, step: int, batch_size: int) -> List[Dict[str, Any]]:
        """
        Choose strategy mix for the current phase and delegate to the sampler.

        Deterministic for given seed, total_steps, and step.
        """
        phase = self.get_phase(step)
        seed = self.base_seed + int(step)
        rng = random.Random(seed)
        if self.advisory and getattr(self.sampler, "advisory", None) is None:
            self.sampler.advisory = self.advisory
        if phase == "warmup":
            batch = self._sample_single("balanced", batch_size, seed)
        elif phase == "skill_building":
            batch = self._sample_mixed(self.mix["skill_building"], batch_size, seed, rng)
        elif phase == "frontier":
            batch = self._sample_mixed(self.mix["frontier"], batch_size, seed, rng)
        else:  # fine_tuning
            batch = self._sample_single("econ_urgency", batch_size, seed)

        # Attach curriculum metadata without mutating sampler outputs
        annotated = []
        for item in batch:
            meta = copy.deepcopy(item.get("sampling_metadata", {}))
            meta["phase"] = phase
            meta["step"] = step
            meta["total_steps"] = self.total_steps
            annotated_item = copy.deepcopy(item)
            annotated_item["sampling_metadata"] = meta
            annotated.append(to_json_safe(annotated_item))
        return annotated

    def _sample_single(self, strategy: str, batch_size: int, seed: int) -> List[Dict[str, Any]]:
        return self.sampler.sample_batch(batch_size=batch_size, seed=seed, strategy=strategy)

    def _sample_mixed(
        self, mix: Dict[str, float], batch_size: int, base_seed: int, rng: random.Random
    ) -> List[Dict[str, Any]]:
        mix = self._apply_advisory_mix(mix)
        # Compute counts per strategy with a deterministic rounding order
        strategies = sorted(mix.items(), key=lambda kv: (-kv[1], kv[0]))
        counts: Dict[str, int] = {}
        remaining = batch_size
        for i, (strategy, ratio) in enumerate(strategies):
            count = int(round(batch_size * ratio))
            # Ensure at least one sample from the highest-weight strategy when possible
            if i == 0 and count == 0 and batch_size > 0:
                count = 1
            count = min(count, remaining)
            counts[strategy] = count
            remaining -= count
        if remaining > 0 and strategies:
            counts[strategies[0][0]] += remaining

        combined: List[Dict[str, Any]] = []
        seed_offset = 1
        for strategy, count in counts.items():
            if count <= 0:
                continue
            strat_seed = base_seed * 9973 + seed_offset
            combined.extend(self._sample_single(strategy, count, strat_seed))
            seed_offset += 1

        rng.shuffle(combined)
        return combined[:batch_size]

    def _apply_advisory_mix(self, mix: Dict[str, float]) -> Dict[str, float]:
        if not self.advisory or not getattr(self.advisory, "sampler_strategy_overrides", None):
            return mix
        overrides = self.advisory.sampler_strategy_overrides
        adjusted = {}
        for k, v in mix.items():
            adjusted[k] = max(0.0, v * overrides.get(k, 1.0))
        total = sum(adjusted.values())
        if total <= 0:
            return mix
        return {k: v / total for k, v in adjusted.items()}

    def _build_boundaries(self, overrides: Optional[Dict[str, float]]) -> Dict[str, float]:
        if not overrides:
            return copy.deepcopy(self.DEFAULT_BOUNDARIES)
        merged = copy.deepcopy(self.DEFAULT_BOUNDARIES)
        for key, val in overrides.items():
            if key in merged and isinstance(val, (int, float)):
                merged[key] = float(val)
        return merged

    def _build_mix(self, overrides: Optional[Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
        if not overrides:
            return copy.deepcopy(self.DEFAULT_MIX)
        merged = copy.deepcopy(self.DEFAULT_MIX)
        for phase, mix in overrides.items():
            if phase not in merged or not isinstance(mix, dict):
                continue
            for strategy, weight in mix.items():
                merged[phase][strategy] = float(weight)
        return merged
