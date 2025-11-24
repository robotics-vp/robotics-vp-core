"""
Phase H Cycle Orchestrator: High-level controller for Phase H integration.

Coordinates:
- Loading Phase H artifacts
- Generating advisory signals
- Pushing signals to Sampler/Orchestrator/ConditionVector
- Logging cycle summaries

Advisory-only, bounded, deterministic.
"""
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

from src.phase_h.advisory_integration import load_phase_h_advisory, PhaseHAdvisory
from src.utils.json_safe import to_json_safe


class PhaseHCycleOrchestrator:
    """
    Orchestrates Phase H learning cycles.

    Every N episodes:
    1. Load Phase H artifacts
    2. Generate advisory signals
    3. Push to Sampler/Orchestrator
    4. Log cycle summary
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Phase H cycle orchestrator.

        Args:
            config: Dict with:
                - ontology_root: Path to ontology artifacts
                - cycle_period_episodes: Episodes between cycles (default 1000)
                - enable_phase_h: Flag to enable Phase H (default False)
                - log_dir: Directory for cycle logs (default "logs/phase_h")
        """
        self.ontology_root = Path(config.get("ontology_root", "data/ontology"))
        self.cycle_period_episodes = int(config.get("cycle_period_episodes", 1000))
        self.enable_phase_h = bool(config.get("enable_phase_h", False))
        self.log_dir = Path(config.get("log_dir", "logs/phase_h"))
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.last_cycle_episode = 0
        self.cycle_count = 0

    def run_cycle_once(self, episode_count: int) -> Optional[Dict[str, Any]]:
        """
        Run one Phase H cycle if it's time.

        Args:
            episode_count: Current episode count

        Returns:
            Cycle summary dict, or None if not time for cycle
        """
        if not self.enable_phase_h:
            return None

        if episode_count % self.cycle_period_episodes != 0:
            return None

        # Load Phase H advisory
        advisory = load_phase_h_advisory(self.ontology_root)
        if not advisory:
            return None

        # Generate cycle summary
        summary = {
            "cycle_count": self.cycle_count,
            "episode_count": episode_count,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "skill_multipliers": advisory.skill_multipliers,
            "skill_quality_signals": advisory.skill_quality_signals,
            "exploration_priorities": advisory.exploration_priorities,
            "routing_advisories": advisory.routing_advisories,
        }

        # Save cycle summary
        self._save_cycle_summary(summary)

        self.last_cycle_episode = episode_count
        self.cycle_count += 1

        return summary

    def _save_cycle_summary(self, summary: Dict[str, Any]):
        """Save cycle summary to log file."""
        log_file = self.log_dir / "cycle_summaries.jsonl"

        with open(log_file, "a") as f:
            json.dump(to_json_safe(summary), f)
            f.write("\n")

    def get_current_advisory(self) -> Optional[PhaseHAdvisory]:
        """
        Get current Phase H advisory.

        Returns:
            PhaseHAdvisory or None if disabled or not found
        """
        if not self.enable_phase_h:
            return None

        return load_phase_h_advisory(self.ontology_root)

    def should_run_cycle(self, episode_count: int) -> bool:
        """Check if it's time to run a cycle."""
        if not self.enable_phase_h:
            return False

        return episode_count % self.cycle_period_episodes == 0

    def get_cycle_summary(self) -> Dict[str, Any]:
        """Get summary of Phase H state."""
        return {
            "cycle_count": self.cycle_count,
            "last_cycle_episode": self.last_cycle_episode,
            "cycle_period_episodes": self.cycle_period_episodes,
            "enable_phase_h": self.enable_phase_h,
        }
