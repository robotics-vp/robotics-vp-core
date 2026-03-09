

from src.learning.data_value_models import predict_data_value, train_data_value_model
from src.learning.pricing_models import predict_pricing_delta, train_pricing_delta_model
from src.learning.regal_support_models import predict_regal_support, train_regal_support_model
from src.replay.dataset import ReplayDatasetBuilder, load_replay_dataset
from src.shadow_runtime.control_plane import run_shadow_control_plane


def test_shadow_pricing_value_models_train_and_predict(tmp_path):
    shadow_dir = tmp_path / "shadow_run"
    dataset_dir = tmp_path / "replay_dataset"
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

    pricing_model, pricing_metrics = train_pricing_delta_model(dataset.episodes, seed=42, epochs=2, lr=1e-3, hidden_dim=32)
    data_model, data_metrics = train_data_value_model(dataset.episodes, seed=42, epochs=2, lr=1e-3, hidden_dim=32)
    regal_model, regal_metrics = train_regal_support_model(dataset.episodes, seed=42, epochs=2, lr=1e-3, hidden_dim=32)

    pricing_pred = predict_pricing_delta(pricing_model, dataset.episodes[0])
    data_pred = predict_data_value(data_model, dataset.episodes[0])
    regal_pred = predict_regal_support(regal_model, dataset.episodes[0])

    assert pricing_metrics["epochs"] == 2
    assert data_metrics["epochs"] == 2
    assert regal_metrics["epochs"] == 2
    assert isinstance(pricing_pred.value, float)
    assert isinstance(data_pred.value, float)
    assert 0.0 <= regal_pred.value <= 1.0
