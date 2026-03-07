import torch

from src.learning.data_value_models import train_data_value_model
from src.learning.pricing_models import train_pricing_delta_model
from src.learning.regal_support_models import train_regal_support_model
from src.learning.replay_policy_trainer import train_replay_policy
from src.replay.dataset import ReplayDatasetBuilder, load_replay_dataset
from src.regality.promotion_policy import PromotionMetrics, RegalPromotionPolicy
from src.shadow_runtime.advisors import AdvisorMode, DataValueAdvisor, PolicyAdvisor, PricingAdvisor, RegalSupportAdvisor
from src.shadow_runtime.control_plane import run_shadow_control_plane


def _save_residual_checkpoint(path, model, metrics, dataset_digest):
    first_weight = next(iter(model.state_dict().values()))
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": int(first_weight.shape[-1]),
            "hidden_dim": int(first_weight.shape[0]),
            "config_digest": "test_cfg",
            "dataset_digest": dataset_digest,
            "model_version": metrics["model_version"],
        },
        path,
    )


def test_shadow_advisors_fallback_and_residual_modes(tmp_path):
    shadow_dir = tmp_path / "shadow_run"
    dataset_dir = tmp_path / "replay_dataset"
    config_path = tmp_path / "policy.yaml"
    policy_dir = tmp_path / "replay_policy"
    config_path.write_text(
        """
model:
  hidden_dim: 64
  head_hidden_dim: 32
  vision_dim: 16
  use_condition_film: true
  use_condition_vector_for_policy: true
  condition_fusion_mode: film
  default_skill_mode: efficiency_throughput
  enable_value_head: true
training:
  seed: 42
  device: cpu
  batch_size: 4
  epochs: 2
  lr: 0.001
  val_fraction: 0.25
  grad_clip: 1.0
""",
        encoding="utf-8",
    )
    run_shadow_control_plane(
        output_dir=shadow_dir,
        seed=42,
        episodes=3,
        objective_profile_id="balanced_contract",
        include_regal=True,
        timestamp_base="2026-01-01T00:00:00+00:00",
    )
    ReplayDatasetBuilder().add_shadow_run(shadow_dir).write(dataset_dir)
    dataset = load_replay_dataset(dataset_dir)

    fallback_policy = PolicyAdvisor(mode=AdvisorMode.HEURISTIC_LEARNED_COMPARE_ONLY, checkpoint_path=tmp_path / "missing.pt")
    fallback_pricing = PricingAdvisor(mode=AdvisorMode.HEURISTIC_LEARNED_RESIDUAL, checkpoint_path=tmp_path / "missing.pt")
    assert fallback_policy.summarize_episode(dataset.steps[:2]).fallback_used is True
    assert fallback_pricing.assess_episode(dataset.episodes[0]).fallback_used is True

    train_result = train_replay_policy(dataset_dir=dataset_dir, config_path=config_path, output_dir=policy_dir)
    pricing_model, pricing_metrics = train_pricing_delta_model(dataset.episodes, seed=42, epochs=2, lr=1e-3, hidden_dim=32)
    data_model, data_metrics = train_data_value_model(dataset.episodes, seed=42, epochs=2, lr=1e-3, hidden_dim=32)
    regal_model, regal_metrics = train_regal_support_model(dataset.episodes, seed=42, epochs=2, lr=1e-3, hidden_dim=32)
    pricing_ckpt = tmp_path / "pricing_delta.pt"
    data_ckpt = tmp_path / "data_value.pt"
    regal_ckpt = tmp_path / "regal_support.pt"
    _save_residual_checkpoint(pricing_ckpt, pricing_model, pricing_metrics, dataset.manifest.dataset_digest)
    _save_residual_checkpoint(data_ckpt, data_model, data_metrics, dataset.manifest.dataset_digest)
    _save_residual_checkpoint(regal_ckpt, regal_model, regal_metrics, dataset.manifest.dataset_digest)

    policy_advisor = PolicyAdvisor(mode=AdvisorMode.HEURISTIC_LEARNED_COMPARE_ONLY, checkpoint_path=train_result.best_checkpoint_path)
    pricing_advisor = PricingAdvisor(mode=AdvisorMode.HEURISTIC_LEARNED_RESIDUAL, checkpoint_path=pricing_ckpt)
    data_advisor = DataValueAdvisor(mode=AdvisorMode.HEURISTIC_LEARNED_COMPARE_ONLY, checkpoint_path=data_ckpt)
    regal_advisor = RegalSupportAdvisor(mode=AdvisorMode.HEURISTIC_LEARNED_COMPARE_ONLY, checkpoint_path=regal_ckpt)

    assert policy_advisor.summarize_episode([row for row in dataset.steps if row.episode_id == dataset.episodes[0].episode_id]).learned_output["available"] is True
    pricing_result = pricing_advisor.assess_episode(dataset.episodes[0])
    assert pricing_result.applied_output["net_customer_rate"] != pricing_result.heuristic_output["net_customer_rate"]
    assert "predicted_data_value" in data_advisor.assess_episode(dataset.episodes[0]).learned_output
    assert "anomaly_support_score" in regal_advisor.assess_episode(dataset.episodes[0]).learned_output


def test_promoted_gate_eligible_mode_requires_promotion_evidence(tmp_path):
    shadow_dir = tmp_path / "shadow_run"
    dataset_dir = tmp_path / "replay_dataset"
    run_shadow_control_plane(
        output_dir=shadow_dir,
        seed=42,
        episodes=2,
        objective_profile_id="balanced_contract",
        include_regal=True,
        timestamp_base="2026-01-01T00:00:00+00:00",
    )
    ReplayDatasetBuilder().add_shadow_run(shadow_dir).write(dataset_dir)
    dataset = load_replay_dataset(dataset_dir)
    pricing_model, pricing_metrics = train_pricing_delta_model(dataset.episodes, seed=42, epochs=2, lr=1e-3, hidden_dim=32)
    pricing_ckpt = tmp_path / "pricing_delta.pt"
    _save_residual_checkpoint(pricing_ckpt, pricing_model, pricing_metrics, dataset.manifest.dataset_digest)

    policy = RegalPromotionPolicy.from_path("configs/regality/promotion_default.yaml")
    weak_metrics = PromotionMetrics(
        replay_coverage=0.4,
        downstream_label_count=1,
        deployment_receipt_count=0,
        calibration_error=0.5,
        baseline_agreement=0.5,
        monotonicity=0.5,
        sign_consistency=0.5,
        false_positive_rate=0.5,
        false_negative_rate=0.5,
        drift_score=0.7,
        residual_gain=-0.1,
    )
    fallback = PricingAdvisor(
        mode=AdvisorMode.PROMOTED_GATE_ELIGIBLE,
        checkpoint_path=pricing_ckpt,
        promotion_policy=policy,
        promotion_node_id="pricing_truth_regal",
        promotion_metrics=weak_metrics,
    )
    assert fallback.guard.gate_eligible is False
    assert fallback.guard.effective_mode == AdvisorMode.HEURISTIC_PLUS_LEARNED_RESIDUAL.value

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
                    "promotion_criteria": {
                        "min_replay_coverage": 0.5,
                        "min_downstream_label_count": 2,
                        "min_deployment_receipt_count": 0,
                        "max_calibration_error": 0.2,
                        "min_baseline_agreement": 0.7,
                        "min_monotonicity": 0.7,
                        "min_sign_consistency": 0.7,
                        "max_false_positive_rate": 0.2,
                        "max_false_negative_rate": 0.2,
                        "max_drift_score": 0.3,
                        "min_residual_gain": 0.0,
                    },
                    "demotion_criteria": {
                        "min_replay_coverage": 0.3,
                        "min_downstream_label_count": 0,
                        "min_deployment_receipt_count": 0,
                        "max_calibration_error": 0.4,
                        "min_baseline_agreement": 0.4,
                        "min_monotonicity": 0.4,
                        "min_sign_consistency": 0.4,
                        "max_false_positive_rate": 0.5,
                        "max_false_negative_rate": 0.5,
                        "max_drift_score": 0.8,
                        "min_residual_gain": -0.1,
                    },
                }
            },
        }
    )
    strong_metrics = PromotionMetrics(
        replay_coverage=0.9,
        downstream_label_count=10,
        deployment_receipt_count=2,
        calibration_error=0.08,
        baseline_agreement=0.9,
        monotonicity=0.9,
        sign_consistency=0.9,
        false_positive_rate=0.08,
        false_negative_rate=0.08,
        drift_score=0.1,
        residual_gain=0.02,
    )
    promoted = PricingAdvisor(
        mode=AdvisorMode.PROMOTED_GATE_ELIGIBLE,
        checkpoint_path=pricing_ckpt,
        promotion_policy=promoted_policy,
        promotion_node_id="pricing_truth_regal",
        promotion_metrics=strong_metrics,
    )
    assert promoted.guard.gate_eligible is True
    assert promoted.guard.effective_mode == AdvisorMode.PROMOTED_GATE_ELIGIBLE.value
