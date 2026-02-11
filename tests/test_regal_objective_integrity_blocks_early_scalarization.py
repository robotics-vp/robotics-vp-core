from src.regal.base import RegalDecision
from src.regal.objective_integrity import RegalObjectiveIntegrityNode


def test_objective_integrity_blocks_early_scalarization():
    node = RegalObjectiveIntegrityNode()
    report = node.evaluate(
        {
            "objective_tensor": None,
            "scalarized_upstream": True,
            "compiler_stage_seen": False,
            "lineage": ["reward_engine.step_reward"],
        }
    )
    assert report.decision == RegalDecision.BLOCK
    assert "early_scalarization_detected" in report.reason_codes
