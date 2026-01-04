import torch

from src.vision.motion_hierarchy.metrics import (
    compute_motion_hierarchy_summary_from_raw,
    compute_motion_hierarchy_summary_from_stats,
    compute_motion_plausibility_flags,
)


def _make_chain_hierarchy() -> torch.Tensor:
    return torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )


def test_motion_hierarchy_metrics_summary():
    hierarchy = _make_chain_hierarchy()
    delta_resid_stats = {
        "mean": [0.1, 0.2, 0.3, 0.4],
        "std": [0.05, 0.05, 0.05, 0.05],
    }

    summary = compute_motion_hierarchy_summary_from_raw(hierarchy, delta_resid_stats)

    assert summary.mean_tree_depth > 0.0
    assert 0.0 <= summary.plausibility_score <= 1.0
    assert summary.structural_difficulty > 0.0


def test_motion_hierarchy_metrics_monotone_residual():
    hierarchy = _make_chain_hierarchy()
    low = compute_motion_hierarchy_summary_from_raw(
        hierarchy, {"mean": [0.1, 0.1, 0.1, 0.1], "std": [0.05, 0.05, 0.05, 0.05]}
    )
    high = compute_motion_hierarchy_summary_from_raw(
        hierarchy, {"mean": [1.0, 1.0, 1.0, 1.0], "std": [0.05, 0.05, 0.05, 0.05]}
    )

    assert high.structural_difficulty > low.structural_difficulty


def test_motion_hierarchy_plausibility_flags():
    summary = compute_motion_hierarchy_summary_from_stats(
        mean_tree_depth=1.0,
        mean_branch_factor=0.5,
        residual_mean=0.2,
        residual_std=0.1,
    )
    flags = compute_motion_plausibility_flags(summary, max_residual_mean=1.0, min_plausibility_score=0.1)
    assert flags["is_plausible"] is True

    summary_bad = compute_motion_hierarchy_summary_from_stats(
        mean_tree_depth=1.0,
        mean_branch_factor=0.5,
        residual_mean=2.0,
        residual_std=0.1,
    )
    flags_bad = compute_motion_plausibility_flags(summary_bad, max_residual_mean=1.0, min_plausibility_score=0.5)
    assert flags_bad["is_plausible"] is False
    assert flags_bad["reason"]
