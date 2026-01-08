from src.epiplexity.metadata import (
    attach_epiplexity_result,
    attach_epiplexity_summary,
    extract_epiplexity_summary_metric,
    extract_epiplexity_summary_confidence,
)
from src.epiplexity.tracker import EpiplexityResult, EpiplexityRunKey
from src.valuation.datapack_schema import DataPackMeta


def test_epiplexity_metadata_roundtrip():
    dp = DataPackMeta()
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
    result = EpiplexityResult(
        key=key,
        S_T_proxy=1.0,
        H_T_proxy=0.5,
        epi_per_flop=0.2,
        delta_epi_vs_baseline=0.1,
        loss_curve=[0.9, 0.8],
    )
    attach_epiplexity_result(dp, result)
    attach_epiplexity_summary(
        dp,
        repr_id="raw",
        budget_id="steps_5_bs_4",
        summary={"mean": {"delta_epi_vs_baseline": 0.1, "epi_per_flop": 0.2}, "confidence": 0.7},
        set_default=True,
    )

    assert extract_epiplexity_summary_metric(dp, metric="delta_epi_vs_baseline") == 0.1
    assert extract_epiplexity_summary_confidence(dp) == 0.7
