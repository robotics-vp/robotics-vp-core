import torch

from src.epiplexity.tracker import EpiplexityTracker, EpiplexityRunKey, ComputeBudget


def test_epiplexity_tracker_cache(tmp_path):
    tokens = torch.randn(3, 5, 4)
    tracker = EpiplexityTracker(cache_dir=str(tmp_path))
    key = EpiplexityRunKey(
        repr_id="raw",
        repr_version_hash="v1",
        tokenizer_version="v1",
        transform_chain_hash="v1",
        dataset_slice_id="slice",
        probe_model_id="probe",
        compute_budget_id="steps_5_bs_4",
        seed=0,
    )
    budget = ComputeBudget(max_steps=5, batch_size=4)
    result_a = tracker.evaluate_tokens(tokens, key, budget)
    result_b = tracker.evaluate_tokens(tokens, key, budget)

    assert result_a.S_T_proxy == result_b.S_T_proxy
    assert result_a.epi_per_flop == result_b.epi_per_flop
    assert any(tmp_path.iterdir())
