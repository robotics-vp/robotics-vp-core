"""Tests for anti-reward-hacking heuristics."""
from src.motor_backend.holosoma_reward_terms import analyze_anti_reward_hacking


def test_analyze_anti_reward_hacking_flags_suspicious_case():
    report = analyze_anti_reward_hacking(
        {
            "mean_reward": 100.0,
            "success_rate": 0.1,
            "mean_episode_length": 1.0,
            "expected_duration": 100.0,
        }
    )
    assert report.is_suspicious
    assert report.reasons
    assert report.summary_metrics["episode_reward_mean"] == 100.0


def test_analyze_anti_reward_hacking_normal_case():
    report = analyze_anti_reward_hacking(
        {
            "mean_reward": 5.0,
            "success_rate": 1.0,
            "mean_episode_length": 120.0,
            "energy_kwh": 0.5,
        }
    )
    assert report.is_suspicious is False
    assert report.reasons == []
