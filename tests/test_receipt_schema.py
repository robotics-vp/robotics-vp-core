from src.economics.receipt_schema import DeploymentReceiptRecord, PricingAcceptanceLabel
from src.ontology.deployment_labels import AdaptationOutcomeLabel, DatapackContributionLabel, DeploymentOutcomeLabel


def test_deployment_labels_and_receipts_roundtrip():
    deployment = DeploymentOutcomeLabel(
        schema_version="deployment_outcome_label_v1",
        run_id="run_a",
        episode_id="ep_001",
        source_domain="synthetic",
        deployment_id="dep_001",
        objective_profile_id="balanced_contract",
        predicted_value=10.0,
        realized_value=9.5,
        pricing_accepted=True,
    )
    adaptation = AdaptationOutcomeLabel(
        schema_version="adaptation_outcome_label_v1",
        run_id="run_a",
        adaptation_id="adapt_001",
        source_domain="training_run",
        recommended_mode="offline_td3_bc_shadow",
        realized_mode="offline_td3_bc_shadow",
        expected_gain=1.0,
        realized_gain=0.8,
        compute_cost=0.2,
        risk_cost=0.1,
        review_required=False,
    )
    datapack = DatapackContributionLabel(
        schema_version="datapack_contribution_label_v1",
        datapack_id="dp_001",
        run_id="run_a",
        source_domain="training_run",
        marginal_frontier_gain_predicted=0.4,
        marginal_frontier_gain_realized=0.35,
        data_share_credit_predicted=0.8,
        data_share_credit_realized=0.75,
        downweight_recommended=False,
    )
    pricing = PricingAcceptanceLabel(
        schema_version="pricing_acceptance_label_v1",
        receipt_id="receipt_001",
        run_id="run_a",
        episode_id="ep_001",
        quoted_rate=35.0,
        accepted_rate=34.0,
        accepted=True,
    )
    receipt = DeploymentReceiptRecord(
        schema_version="deployment_receipt_record_v1",
        run_id="run_a",
        episode_id="ep_001",
        deployment_id="dep_001",
        source_domain="synthetic",
        objective_profile_id="balanced_contract",
        predicted_value=deployment.predicted_value,
        realized_value=deployment.realized_value,
        quoted_rate=pricing.quoted_rate,
        billed_rate=pricing.accepted_rate,
        pricing_acceptance=pricing,
        realized_reward=5.0,
        task_success=True,
        objective_satisfied=True,
        incident_events=[],
        adaptation_outcome_ref=adaptation.label_id,
        datapack_label_ref=datapack.label_id,
    )

    assert DeploymentOutcomeLabel.from_dict(deployment.to_dict()) == deployment
    assert AdaptationOutcomeLabel.from_dict(adaptation.to_dict()) == adaptation
    assert DatapackContributionLabel.from_dict(datapack.to_dict()) == datapack
    assert DeploymentReceiptRecord.from_dict(receipt.to_dict()).record_id == receipt.record_id
