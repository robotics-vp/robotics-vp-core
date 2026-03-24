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
    assert result_a.flops_estimate > 0.0
    assert result_a.score_mode == "absolute"
    assert any(tmp_path.iterdir())


def test_epiplexity_tracker_cached_absolute_result_is_baseline_independent(tmp_path):
    tokens = torch.randn(3, 5, 4)
    baseline_a_tokens = torch.randn(3, 5, 4)
    baseline_b_tokens = torch.randn(3, 5, 4) + 5.0
    tracker = EpiplexityTracker(cache_dir=str(tmp_path))
    key = EpiplexityRunKey(
        repr_id="candidate",
        repr_version_hash="v1",
        tokenizer_version="v1",
        transform_chain_hash="v1",
        dataset_slice_id="slice",
        probe_model_id="probe",
        compute_budget_id="steps_5_bs_4",
        seed=0,
    )
    budget = ComputeBudget(max_steps=5, batch_size=4)

    absolute = tracker.evaluate_tokens(tokens, key, budget)
    baseline_a = tracker.evaluate_tokens(
        baseline_a_tokens,
        EpiplexityRunKey(
            repr_id="baseline_a",
            repr_version_hash="v1",
            tokenizer_version="v1",
            transform_chain_hash="v1",
            dataset_slice_id="slice",
            probe_model_id="probe",
            compute_budget_id="steps_5_bs_4",
            seed=0,
        ),
        budget,
    )
    baseline_b = tracker.evaluate_tokens(
        baseline_b_tokens,
        EpiplexityRunKey(
            repr_id="baseline_b",
            repr_version_hash="v1",
            tokenizer_version="v1",
            transform_chain_hash="v1",
            dataset_slice_id="slice",
            probe_model_id="probe",
            compute_budget_id="steps_5_bs_4",
            seed=0,
        ),
        budget,
    )

    relative_a = tracker.evaluate_tokens(tokens, key, budget, baseline_result=baseline_a)
    relative_b = tracker.evaluate_tokens(tokens, key, budget, baseline_result=baseline_b)

    assert absolute.S_T_proxy == relative_a.S_T_proxy == relative_b.S_T_proxy
    assert absolute.epi_per_flop == relative_a.epi_per_flop == relative_b.epi_per_flop
    assert relative_a.score_mode == "relative"
    assert relative_b.score_mode == "relative"
    assert relative_a.delta_epi_vs_baseline != relative_b.delta_epi_vs_baseline
