from __future__ import annotations

from src.dataset_bridges.lerobot_bridge import lerobot_rows_from_replay
from src.dataset_bridges.rlds_bridge import rlds_episode_from_replay
from src.replay.schema import ReplayEpisodeRecord, ReplayStepRecord


def _episode() -> ReplayEpisodeRecord:
    return ReplayEpisodeRecord(
        run_id="run-1",
        episode_id="ep-1",
        task_id="dishwashing",
        env_id="dishwashing_regal",
        source_domain="sim",
        seed=7,
        status="done",
        started_at="2026-03-21T01:00:00Z",
        ended_at="2026-03-21T01:00:03Z",
        total_steps=2,
        total_reward=1.5,
        skill_mode="efficiency",
        condition_vector={"water_temp": 0.4},
        condition_vector_values=[0.4],
        objective_tensor_summary={"target": "throughput"},
        objective_tensor_ref="objective.json",
        econ_tensor_summary={"wage_usd": 12.0},
        econ_tensor_ref="econ.json",
        pricing_summary={"price": 0.7},
        pricing_tick_refs=["pricing-1"],
        constraint_flags=[{"name": "safety"}],
        regal_summary={"status": "ok"},
        datapack_summary={"slice": "s1"},
        ledger_event_ids=["ledger-1"],
        metadata={"note": "episode", "governed_supervision_refs": ["supervision-1"]},
        provenance={"event_spine_ref": "event_spine.jsonl", "teacher_trace_ref": "teacher_trace.json"},
    )


def _steps() -> list[ReplayStepRecord]:
    base = dict(
        run_id="run-1",
        episode_id="ep-1",
        obs={"rgb": "ref"},
        obs_vector=[0.1, 0.2],
        action={"joint": 0.3},
        action_vector=[0.3],
        reward_decomposition={"mpl": 0.1},
        task_id="dishwashing",
        env_id="dishwashing_regal",
        condition_vector={"water_temp": 0.4},
        condition_vector_values=[0.4],
        skill_mode="efficiency",
        objective_tensor_summary={"target": "throughput"},
        objective_tensor_ref="objective.json",
        econ_tensor_summary={"wage_usd": 12.0},
        econ_tensor_ref="econ.json",
        constraint_flags=[{"name": "safety"}],
        pricing_tick_ref="pricing-1",
        ledger_event_ref="ledger-1",
        source_domain="sim",
        seed=7,
        metadata={
            "event_refs": ["event-1"],
            "decision_refs": ["decision-1"],
            "counterfactual_eval_ref": "counterfactual.json",
            "value_target_refs": ["value-target-1"],
        },
        provenance={
            "runtime_packet_ref": "packet.json",
            "governance_trace_ref": "trace.jsonl",
            "belief_state_ref": "belief.json",
        },
    )
    return [
        ReplayStepRecord(step_idx=0, reward=0.5, done=False, timestamp="2026-03-21T01:00:01Z", **base),
        ReplayStepRecord(step_idx=1, reward=1.0, done=True, timestamp="2026-03-21T01:00:02Z", **base),
    ]


def test_rlds_bridge_converts_and_preserves_sidecar_refs() -> None:
    episode = _episode()
    steps = _steps()
    payload = rlds_episode_from_replay(episode, steps)

    assert payload["episode_id"] == "ep-1"
    assert len(payload["steps"]) == 2
    assert payload["steps"][0]["is_first"] is True
    assert payload["steps"][1]["is_last"] is True
    assert payload["steps"][1]["metadata"]["internal_sidecars"]["runtime_packet_ref"] == "packet.json"
    assert payload["steps"][0]["metadata"]["internal_sidecars"]["counterfactual_eval_ref"] == "counterfactual.json"
    assert payload["steps"][0]["metadata"]["internal_sidecars"]["value_target_refs"] == ["value-target-1"]
    assert payload["steps"][0]["metadata"]["internal_sidecars"]["belief_state_ref"] == "belief.json"
    assert "run_id" not in payload["steps"][0]["metadata"]["internal_sidecars"]
    assert payload["metadata"]["internal_sidecars"]["objective_tensor_ref"] == "objective.json"
    assert payload["metadata"]["internal_sidecars"]["governed_supervision_refs"] == ["supervision-1"]
    assert payload["metadata"]["internal_sidecars"]["teacher_trace_ref"] == "teacher_trace.json"


def test_lerobot_bridge_converts_and_preserves_sidecar_refs() -> None:
    rows = lerobot_rows_from_replay(_episode(), _steps())

    assert len(rows) == 2
    assert rows[0]["frame_index"] == 0
    assert rows[1]["done"] is True
    assert rows[0]["metadata"]["internal_sidecars"]["event_refs"] == ["event-1"]
    assert rows[0]["metadata"]["internal_sidecars"]["runtime_packet_ref"] == "packet.json"
    assert rows[0]["metadata"]["internal_sidecars"]["counterfactual_eval_ref"] == "counterfactual.json"
    assert rows[0]["metadata"]["internal_sidecars"]["value_target_refs"] == ["value-target-1"]
    assert rows[0]["metadata"]["internal_sidecars"]["belief_state_ref"] == "belief.json"
    assert "episode_id" not in rows[0]["metadata"]["internal_sidecars"]
