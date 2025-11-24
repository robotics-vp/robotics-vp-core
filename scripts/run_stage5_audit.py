#!/usr/bin/env python3
"""
Stage 5 audit runner.

Runs smoke tests and validates Stage 5 invariants:
- ConditionVector determinism/bounds
- TFD → ConditionVector integration
- SIMA-2 segmentation tolerance
- TrustMatrix usage inside sampler
- Economic Learner bounded adjustments (±20%)

Outputs: results/audit/stage5_report.json
"""
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.observation.condition_vector_builder import ConditionVectorBuilder
from src.policies.sampler_weights import HeuristicSamplerWeightPolicy
from src.phase_h.economic_learner import EconomicLearner
from src.phase_h.models import Skill, SkillStatus
from src.sima2.client import Sima2Client
from src.sima2.segmentation_engine import SegmentationEngine
from src.sima2.segmentation_engine import HeuristicSegmenter  # noqa: F401 (used for assertions)


def _run_script(path: Path) -> Dict[str, Any]:
    proc = subprocess.run([sys.executable, str(path)], capture_output=True, text=True)
    return {
        "script": str(path.name),
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
        "success": proc.returncode == 0,
    }


def _condition_vector_checks() -> Dict[str, Any]:
    builder = ConditionVectorBuilder()
    base_args = dict(
        episode_config={
            "task_id": "drawer_test",
            "env_id": "env_test",
            "backend_id": "stub",
            "objective_preset": "balanced",
        },
        econ_state={"target_mpl": 60.0, "current_wage_parity": 1.0, "energy_budget_wh": 50.0},
        curriculum_phase="warmup",
        sima2_trust={"trust_score": 0.6},
        datapack_metadata={"tags": {"OODTag": 0.4}},
        episode_step=3,
        overrides=None,
        econ_slice={"mpl": 60.0, "energy_wh": 5.0},
        semantic_tags={"OODTag": 0.6, "RecoveryTag": 0.2},
        recap_scores={"advantage_bin_probs": [0.1, 0.2, 0.7]},
        trust_summary={"OODTag": {"trust_score": 0.8}},
        episode_metadata={"sampler_strategy": "balanced"},
    )
    cv1 = builder.build(**base_args)
    cv2 = builder.build(**base_args)
    bounded = 0.0 <= cv1.ood_risk_level <= 1.0 and cv1.energy_budget_wh >= 0.0 and cv1.target_mpl >= 0.0
    return {
        "deterministic": cv1.to_dict() == cv2.to_dict(),
        "bounded": bounded,
        "condition_vector": cv1.to_dict(),
    }


def _tfd_chain_check() -> Dict[str, Any]:
    builder = ConditionVectorBuilder()
    tfd_instruction = {"condition_vector": {"risk_tolerance": 0.9, "skill_mode": "speed"}}
    cv = builder.build(
        episode_config={"task_id": "drawer_test", "env_id": "env_test", "backend_id": "stub", "objective_preset": "balanced"},
        econ_state={"target_mpl": 50.0, "current_wage_parity": 1.0, "energy_budget_wh": 40.0},
        curriculum_phase="frontier",
        sima2_trust={"trust_score": 0.5},
        datapack_metadata=None,
        episode_step=1,
        overrides=None,
        econ_slice=None,
        semantic_tags=None,
        recap_scores=None,
        trust_summary=None,
        episode_metadata=None,
        advisory_context=None,
        tfd_instruction=tfd_instruction,
        enable_tfd_integration=True,
    )
    return {
        "ood_risk_level": cv.ood_risk_level,
        "skill_mode": cv.skill_mode,
        "aligned_with_tfd": abs(cv.ood_risk_level - 0.9) < 1e-6 and cv.skill_mode.lower().startswith("speed"),
    }


def _segmentation_check() -> Dict[str, Any]:
    client = Sima2Client(task_id="drawer_open", template="success", seed=0)
    rollout = client.run_episode({"episode_index": 0, "seed": 0})
    engine = SegmentationEngine(segmentation_config={"use_heuristic_segmenter": True, "temporal_decay_window": 2})
    result = engine.segment_rollout(rollout)
    segments = result.get("segments", [])
    boundaries = result.get("segment_boundaries", [])
    return {
        "segments_found": len(segments),
        "boundaries_found": len(boundaries),
        "success": bool(segments and boundaries),
    }


def _trust_sampler_check() -> Dict[str, Any]:
    trust_matrix = {
        "OODTag": {"trust_score": 0.9, "sampling_multiplier": 5.0},
        "RecoveryTag": {"trust_score": 0.6, "sampling_multiplier": 1.5},
        "NoveltyTag": {"trust_score": 0.2, "sampling_multiplier": 1.0},
    }
    policy = HeuristicSamplerWeightPolicy(trust_matrix=trust_matrix)
    descriptors = [
        {"descriptor": {"episode_id": "trusted"}, "semantic_tags": [{"tag_type": "OODTag"}]},
        {"descriptor": {"episode_id": "provisional"}, "semantic_tags": [{"tag_type": "RecoveryTag"}]},
        {"descriptor": {"episode_id": "untrusted"}, "semantic_tags": [{"tag_type": "NoveltyTag"}]},
    ]
    weights = policy.evaluate(policy.build_features(descriptors), strategy="balanced")
    ordered = weights["trusted"] > weights["provisional"] > weights["untrusted"]
    return {"weights": weights, "ordered_by_trust": ordered}


def _economic_learner_check() -> Dict[str, Any]:
    learner = EconomicLearner({"total_exploration_budget": 1000.0, "reallocation_period_episodes": 1})
    skill = Skill(
        skill_id="skill_demo",
        display_name="Skill Demo",
        description="Audit skill",
        mpl_baseline=10.0,
        mpl_current=12.0,
        mpl_target=15.0,
        training_cost_usd=200.0,
        data_cost_per_episode=10.0,
        success_rate=0.7,
        failure_rate=0.2,
        recovery_rate=0.1,
        fragility_score=0.8,
        ood_exposure=0.3,
        novelty_tier_avg=1.0,
        training_episodes=5,
        last_updated="audit",
        status=SkillStatus.EXPLORATION.value,
    )
    learner.add_skill(skill)
    before = learner.budgets[skill.skill_id].budget_usd
    learner.run_cycle(episode_count=1)
    after = learner.budgets[skill.skill_id].budget_usd
    ratio = after / max(before, 1e-6)
    cap = 0.2 * learner.total_budget_usd
    cap_triggered = after <= cap + 1e-6
    bounded = (0.8 - 1e-6 <= ratio <= 1.2 + 1e-6) or cap_triggered
    return {"before": before, "after": after, "ratio": ratio, "bounded_20pct": bounded, "cap_triggered": cap_triggered}


def main() -> int:
    smoke_tests = [
        Path(REPO_ROOT / "scripts" / "smoke_test_ros_ingestion.py"),
        Path(REPO_ROOT / "scripts" / "smoke_test_isaac_adapter_contract.py"),
        Path(REPO_ROOT / "scripts" / "smoke_test_segmentation_real.py"),
        Path(REPO_ROOT / "scripts" / "smoke_test_econ_correlator_stage5.py"),
    ]

    smoke_results: List[Dict[str, Any]] = [_run_script(path) for path in smoke_tests]

    report = {
        "timestamp": time.time(),
        "smoke_tests": smoke_results,
        "condition_vector": _condition_vector_checks(),
        "tfd_chain": _tfd_chain_check(),
        "segmentation": _segmentation_check(),
        "trust_sampler": _trust_sampler_check(),
        "economic_learner": _economic_learner_check(),
    }

    out_dir = REPO_ROOT / "results" / "audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "stage5_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)

    print(f"[stage5_audit] Report written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
