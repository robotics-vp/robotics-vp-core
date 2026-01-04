import numpy as np

from src.analytics.motion_hierarchy_curriculum import (
    compute_curriculum_weights_from_motion_hierarchy,
)


def test_curriculum_focus_hard_easy():
    summaries = [
        {"structural_difficulty": 1.0},
        {"structural_difficulty": 2.0},
        {"structural_difficulty": 4.0},
    ]

    hard = compute_curriculum_weights_from_motion_hierarchy(summaries, mode="focus_hard")
    easy = compute_curriculum_weights_from_motion_hierarchy(summaries, mode="focus_easy")

    assert hard[2] > hard[0]
    assert easy[0] > easy[2]
    assert np.isclose(hard.sum(), 1.0)
    assert np.isclose(easy.sum(), 1.0)


def test_curriculum_uniform_plus_hard_tail():
    summaries = [{"structural_difficulty": float(i)} for i in range(10)]
    weights = compute_curriculum_weights_from_motion_hierarchy(
        summaries,
        mode="uniform_plus_hard_tail",
    )

    assert weights[-1] > weights[0]
    assert np.isclose(weights.sum(), 1.0)


def test_curriculum_focus_plausible():
    summaries = [
        {"motion_quality_score": 0.2},
        {"motion_quality_score": 0.8},
    ]
    weights = compute_curriculum_weights_from_motion_hierarchy(
        summaries,
        mode="focus_plausible",
    )

    assert weights[1] > weights[0]
    assert np.isclose(weights.sum(), 1.0)
