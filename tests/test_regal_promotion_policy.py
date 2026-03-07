from src.learning.calibration import CalibrationSummary
from src.regality.promotion_policy import PromotionMetrics, RegalMaturityStage, RegalPromotionPolicy


def _metrics(**overrides):
    summary = CalibrationSummary(
        sample_count=12,
        expected_calibration_error=0.08,
        brier_score=0.05,
        agreement_rate=0.9,
        sign_consistency=0.9,
        monotonicity_score=0.9,
        drift_score=0.12,
        confidence_mean=0.8,
        target_mean=0.75,
    )
    payload = {
        "replay_coverage": 0.9,
        "downstream_label_count": 24,
        "deployment_receipt_count": 10,
        "calibration_error": 0.08,
        "baseline_agreement": 0.9,
        "monotonicity": 0.9,
        "sign_consistency": 0.9,
        "false_positive_rate": 0.08,
        "false_negative_rate": 0.08,
        "drift_score": 0.12,
        "residual_gain": 0.02,
        "calibration_summary": summary,
    }
    payload.update(overrides)
    return PromotionMetrics(**payload)


def test_regal_promotion_policy_promote_hold_demote():
    policy = RegalPromotionPolicy.from_path("configs/regality/promotion_default.yaml")
    assert policy.stage_allows("pricing_truth_regal", "log") is True
    assert policy.gate_eligible("pricing_truth_regal") is False

    promote = policy.evaluate_node("pricing_truth_regal", _metrics())
    assert promote.outcome == "recommend_promote"
    assert promote.recommended_stage == RegalMaturityStage.ADVISORY

    hold = policy.evaluate_node("pricing_truth_regal", _metrics(calibration_error=0.3, baseline_agreement=0.9))
    assert hold.outcome == "recommend_hold"
    assert "calibration_error_too_high" in hold.reasons

    promoted_policy = RegalPromotionPolicy.from_mapping(
        {
            "schema_version": "regal_promotion_policy_v1",
            "policy_name": "test_budget_gate",
            "nodes": {
                "pricing_truth_regal": {
                    "current_stage": "budget_gate",
                    "allowed_actions": {
                        "compare_only": ["log"],
                        "advisory": ["log", "compare"],
                        "budget_gate": ["log", "compare", "budget_gate"],
                        "narrow_hard_gate": ["log", "compare", "budget_gate", "pricing_suppress"],
                    },
                    "promotion_criteria": policy.nodes["pricing_truth_regal"].promotion_criteria.to_dict(),
                    "demotion_criteria": policy.nodes["pricing_truth_regal"].demotion_criteria.to_dict(),
                }
            },
        }
    )
    demote = promoted_policy.evaluate_node("pricing_truth_regal", _metrics(calibration_error=0.4, baseline_agreement=0.5))
    assert demote.outcome == "recommend_demote"
    assert demote.recommended_stage == RegalMaturityStage.ADVISORY
