from src.regal.base import RegalDecision
from src.regal.gen_plausibility import RegalGenPlausibilityNode


def test_plausibility_gate_blocks_high_disagreement():
    node = RegalGenPlausibilityNode()
    report = node.evaluate(
        {
            "map_first_quality_score": 0.2,
            "semantic_disagreement_vla_vs_map": 0.95,
            "vla_evidence_coverage": 0.1,
        }
    )
    assert report.decision == RegalDecision.BLOCK
    assert "semantic_disagreement_high" in report.reason_codes
