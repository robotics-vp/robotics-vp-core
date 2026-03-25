"""Tests for backend health metadata and degradation (A5)."""
import pytest

from src.evidence.backend_health import BackendHealthReport, check_backend_health


def test_fully_stub_report():
    r = BackendHealthReport(episode_id="ep1")
    assert r.is_fully_stub
    assert not r.is_fully_real
    assert len(r.degradation_flags) == 4
    assert r.evidence_density_score == 0.0


def test_fully_real_report():
    r = BackendHealthReport(
        episode_id="ep2",
        scene_tracks_mode="real",
        vla_mode="real",
        teacher_mode="real",
        map_first_mode="real",
    )
    assert r.is_fully_real
    assert not r.is_fully_stub
    assert r.evidence_density_score == 1.0


def test_mixed_modes():
    r = BackendHealthReport(
        scene_tracks_mode="real",
        vla_mode="passthrough",
        teacher_mode="stub",
        map_first_mode="real",
    )
    assert not r.is_fully_real
    assert not r.is_fully_stub
    assert "teacher_stub" in r.degradation_flags
    expected = (1.0 + 0.3 + 0.0 + 1.0) / 4.0
    assert r.evidence_density_score == pytest.approx(expected)


def test_check_passes_with_real():
    r = BackendHealthReport(
        scene_tracks_mode="real",
        vla_mode="real",
        teacher_mode="real",
        map_first_mode="real",
    )
    pc = check_backend_health(r, min_density=0.25)
    assert pc.satisfied
    assert pc.precondition_id == "backend_health"


def test_check_fails_all_stubs():
    r = BackendHealthReport()
    pc = check_backend_health(r, min_density=0.25, max_stub_count=2)
    assert not pc.satisfied  # density=0 < 0.25, stub_count=4 > 2


def test_serialisation_round_trip():
    r = BackendHealthReport(
        episode_id="ep3",
        scene_tracks_mode="passthrough",
        vla_mode="real",
    )
    d = r.to_dict()
    r2 = BackendHealthReport.from_dict(d)
    assert r2.episode_id == "ep3"
    assert r2.scene_tracks_mode == "passthrough"
    assert r2.vla_mode == "real"
