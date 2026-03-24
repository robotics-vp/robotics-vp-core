from src.epiplexity.metadata import (
    apply_epiplexity_overlay,
    attach_epiplexity_result,
    attach_epiplexity_summary,
    build_epiplexity_overlay_record,
    extract_epiplexity_summary_metric,
    extract_epiplexity_summary_confidence,
    load_epiplexity_overlay_map,
    write_epiplexity_overlays,
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


def test_epiplexity_metadata_default_selector_falls_back_to_best_delta():
    dp = DataPackMeta()
    attach_epiplexity_summary(
        dp,
        repr_id="vision_rgb",
        budget_id="steps_5_bs_4",
        summary={"mean": {"delta_epi_vs_baseline": 0.0, "epi_per_flop": 0.2}, "confidence": 0.4},
    )
    attach_epiplexity_summary(
        dp,
        repr_id="canonical_tokens",
        budget_id="steps_5_bs_4",
        summary={"mean": {"delta_epi_vs_baseline": 0.3, "epi_per_flop": 0.4}, "confidence": 0.9},
    )

    assert extract_epiplexity_summary_metric(dp, metric="delta_epi_vs_baseline") == 0.3
    assert extract_epiplexity_summary_confidence(dp) == 0.9


def test_epiplexity_overlay_roundtrip(tmp_path):
    dp = DataPackMeta(pack_id="pack_1", task_name="drawer_vase")
    attach_epiplexity_summary(
        dp,
        repr_id="canonical_tokens",
        budget_id="steps_5_bs_4",
        summary={"mean": {"delta_epi_vs_baseline": 0.3, "epi_per_flop": 0.4}, "confidence": 0.9},
        set_default=True,
    )
    overlay_path = tmp_path / "epiplexity_overlays.jsonl"

    count = write_epiplexity_overlays([dp], str(overlay_path))
    overlay_map = load_epiplexity_overlay_map(str(overlay_path))
    restored = DataPackMeta(pack_id="pack_1", task_name="drawer_vase")
    apply_epiplexity_overlay(restored, overlay_map["pack_1"])

    assert count == 1
    assert build_epiplexity_overlay_record(dp) is not None
    assert extract_epiplexity_summary_metric(restored, metric="delta_epi_vs_baseline") == 0.3
