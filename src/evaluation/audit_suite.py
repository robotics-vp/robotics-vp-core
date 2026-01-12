"""Deterministic audit evaluation suite.

Fixed-seed evaluation harness that runs standardized scenarios
and produces reproducible metrics for the value ledger.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.contracts.schemas import (
    EpisodeInfoSummaryV1,
    AuditAggregateV1,
    DeterminismConfig,
)
from src.utils.determinism import configure_determinism
from src.utils.config_digest import sha256_json


@dataclass
class AuditScenario:
    """Single evaluation scenario."""

    scenario_id: str
    task_name: str
    task_family: Optional[str] = None
    env_config: Optional[Dict[str, Any]] = None
    num_episodes: int = 1


@dataclass
class AuditSuiteConfig:
    """Configuration for audit eval suite."""

    suite_id: str
    seed: int
    scenarios: List[AuditScenario] = field(default_factory=list)
    determinism_strict: bool = False


# Default minimal scenario set for smoke testing
DEFAULT_SCENARIOS = [
    AuditScenario("balanced_01", "drawer_vase", "manipulation", num_episodes=2),
    AuditScenario("occluded_01", "drawer_vase_occluded", "manipulation", num_episodes=2),
    AuditScenario("dynamic_01", "drawer_vase_dynamic", "manipulation", num_episodes=2),
]


class AuditEvalSuite:
    """Deterministic audit evaluation suite.

    Runs fixed-seed evaluations and produces:
    - episodes.jsonl: Per-episode summaries (EpisodeInfoSummaryV1)
    - aggregate.json: Aggregated metrics (AuditAggregateV1)
    """

    def __init__(
        self,
        config: Optional[AuditSuiteConfig] = None,
        suite_id: str = "default_audit",
        seed: int = 42,
    ):
        """Initialize audit suite.

        Args:
            config: Full configuration (overrides suite_id/seed)
            suite_id: Suite identifier
            seed: Random seed for determinism
        """
        if config:
            self.config = config
        else:
            self.config = AuditSuiteConfig(
                suite_id=suite_id,
                seed=seed,
                scenarios=DEFAULT_SCENARIOS.copy(),
            )

        self._results: List[EpisodeInfoSummaryV1] = []

    def run(
        self,
        checkpoint_ref: Optional[str] = None,
        policy_id: Optional[str] = None,
        output_dir: Optional[str] = None,
        regal_context_sha: Optional[str] = None,
    ) -> AuditAggregateV1:
        """Run the audit evaluation suite.

        Args:
            checkpoint_ref: Policy checkpoint reference
            policy_id: Policy identifier
            output_dir: Optional directory to write results
            regal_context_sha: Optional regal context SHA for provenance

        Returns:
            AuditAggregateV1 with aggregated metrics
        """
        # Set determinism
        configure_determinism(seed=self.config.seed, strict=self.config.determinism_strict)

        self._results = []
        episode_index = 0

        for scenario in self.config.scenarios:
            for ep in range(scenario.num_episodes):
                # Simulate episode execution
                # In real usage, this would run the policy in the environment
                result = self._run_episode(
                    scenario=scenario,
                    episode_index=episode_index,
                    checkpoint_ref=checkpoint_ref,
                    policy_id=policy_id,
                )
                self._results.append(result)
                episode_index += 1

        # Compute aggregate
        aggregate = self._compute_aggregate(regal_context_sha=regal_context_sha)

        # Write outputs if requested
        if output_dir:
            self._write_outputs(Path(output_dir), aggregate)

        return aggregate

    def _run_episode(
        self,
        scenario: AuditScenario,
        episode_index: int,
        checkpoint_ref: Optional[str],
        policy_id: Optional[str],
    ) -> EpisodeInfoSummaryV1:
        """Run a single episode evaluation.

        This is a simulation for the smoke test. In production,
        this would execute the policy in the environment.
        """
        # Deterministic pseudo-random based on seed + index
        rng = np.random.RandomState(self.config.seed + episode_index)

        # Simulate metrics
        success = rng.random() > 0.3
        error = rng.random() * 0.5 if not success else rng.random() * 0.1
        total_return = rng.random() * 100 if success else rng.random() * 30
        energy_Wh = rng.random() * 50 + 10
        mpl_proxy = rng.random() * 0.5 + 0.3 if success else rng.random() * 0.3

        return EpisodeInfoSummaryV1(
            episode_id=f"{scenario.scenario_id}_ep{episode_index}",
            episode_index=episode_index,
            task_name=scenario.task_name,
            task_family=scenario.task_family,
            success=success,
            termination_reason="success" if success else "timeout",
            error=error,
            total_return=total_return,
            energy_Wh=energy_Wh,
            mpl_proxy=mpl_proxy,
            determinism=DeterminismConfig(
                seed=self.config.seed,
                strict=self.config.determinism_strict,
            ),
            policy_id=policy_id,
            checkpoint_ref=checkpoint_ref,
            ts_start=datetime.now().isoformat(),
            ts_end=datetime.now().isoformat(),
        )

    def _compute_aggregate(self, regal_context_sha: Optional[str] = None) -> AuditAggregateV1:
        """Compute aggregate metrics from episode results."""
        if not self._results:
            return AuditAggregateV1(
                audit_suite_id=self.config.suite_id,
                seed=self.config.seed,
                num_episodes=0,
                success_rate=0.0,
                episodes_sha=sha256_json([]),
                config_sha=sha256_json(self._config_dict()),
            )

        success_count = sum(1 for r in self._results if r.success)
        success_rate = success_count / len(self._results)

        errors = [r.error for r in self._results if r.error is not None]
        returns = [r.total_return for r in self._results if r.total_return is not None]
        energies = [r.energy_Wh for r in self._results if r.energy_Wh is not None]
        mpls = [r.mpl_proxy for r in self._results if r.mpl_proxy is not None]

        # Per-task breakdown
        per_task: Dict[str, Dict[str, float]] = {}
        for r in self._results:
            key = r.task_family or r.task_name
            if key not in per_task:
                per_task[key] = {"count": 0, "success": 0}
            per_task[key]["count"] += 1
            if r.success:
                per_task[key]["success"] += 1

        for key in per_task:
            per_task[key]["success_rate"] = (
                per_task[key]["success"] / per_task[key]["count"]
            )

        episodes_data = [r.model_dump(mode="json") for r in self._results]

        return AuditAggregateV1(
            audit_suite_id=self.config.suite_id,
            seed=self.config.seed,
            num_episodes=len(self._results),
            success_rate=success_rate,
            mean_error=float(np.mean(errors)) if errors else None,
            mean_return=float(np.mean(returns)) if returns else None,
            mean_energy_Wh=float(np.mean(energies)) if energies else None,
            mean_mpl_proxy=float(np.mean(mpls)) if mpls else None,
            per_task=per_task,
            episodes_sha=sha256_json(episodes_data),
            config_sha=sha256_json(self._config_dict()),
            regal_context_sha=regal_context_sha,
        )

    def _config_dict(self) -> Dict[str, Any]:
        """Get config as dict for hashing."""
        return {
            "suite_id": self.config.suite_id,
            "seed": self.config.seed,
            "scenarios": [
                {
                    "scenario_id": s.scenario_id,
                    "task_name": s.task_name,
                    "task_family": s.task_family,
                    "num_episodes": s.num_episodes,
                }
                for s in self.config.scenarios
            ],
            "determinism_strict": self.config.determinism_strict,
        }

    def _write_outputs(self, output_dir: Path, aggregate: AuditAggregateV1) -> None:
        """Write results to output directory."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Write episodes.jsonl
        episodes_path = output_dir / "episodes.jsonl"
        with open(episodes_path, "w") as f:
            for r in self._results:
                f.write(json.dumps(r.model_dump(mode="json")) + "\n")

        # Write aggregate.json
        aggregate_path = output_dir / "aggregate.json"
        with open(aggregate_path, "w") as f:
            json.dump(aggregate.model_dump(mode="json"), f, indent=2)

    @property
    def results(self) -> List[EpisodeInfoSummaryV1]:
        """Episode results from last run."""
        return self._results


__all__ = [
    "AuditScenario",
    "AuditSuiteConfig",
    "AuditEvalSuite",
    "DEFAULT_SCENARIOS",
]
