#!/usr/bin/env python3
"""
Sanity check for episode feature extraction.
"""
import numpy as np

from src.envs.dishwashing_env import EpisodeInfoSummary
from src.config.econ_params import EconParams
from src.valuation.episode_features import make_episode_feature_vector


def main():
    summary = EpisodeInfoSummary(
        termination_reason="max_steps",
        mpl_episode=100.0,
        ep_episode=2.0,
        error_rate_episode=0.05,
        throughput_units_per_hour=100.0,
        energy_Wh=50.0,
        profit=10.0,
        wage_parity=0.9,
    )
    econ = EconParams(
        price_per_unit=0.3,
        damage_cost=1.0,
        energy_Wh_per_attempt=0.05,
        time_step_s=60.0,
        base_rate=2.0,
        p_min=0.02,
        k_err=0.12,
        q_speed=1.2,
        q_care=1.5,
        care_cost=0.25,
        max_steps=240,
        max_catastrophic_errors=3,
        max_error_rate_sla=0.12,
        min_steps_for_sla=5,
        zero_throughput_patience=10,
        preset="toy",
    )
    feats = make_episode_feature_vector(summary, econ, baseline=None)
    assert feats.shape[0] == 5 + 5, f"Unexpected feature length: {feats.shape}"
    print("Feature vector:", feats)


if __name__ == "__main__":
    main()
