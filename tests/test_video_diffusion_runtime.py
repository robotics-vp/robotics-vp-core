from src.diffusion import VideoDiffusionRuntime, VideoDiffusionRuntimeConfig


def test_video_diffusion_runtime_defaults_to_heuristic_fallback(monkeypatch) -> None:
    monkeypatch.delenv("VIDEO_DIFFUSION_MODEL_NAME", raising=False)
    monkeypatch.delenv("VIDEO_DIFFUSION_MODEL_REF", raising=False)

    runtime = VideoDiffusionRuntime(
        VideoDiffusionRuntimeConfig(
            backend_policy="auto",
            model_ref="",
            device="cpu",
        )
    )
    proposals = runtime.propose_augmented_clips(
        episode_id="ep1",
        media_refs=["ref1"],
        semantic_tags=["fragile", "safety"],
        objective_preset="safety",
        routing_context={"routing_source": "coverage_gap_graph"},
        num_proposals=1,
    )

    status = runtime.status()
    assert status["provider_truth"]["backend_selected"] == "heuristic_fallback"
    assert status["materialization_mode"] == "plan_only"
    assert proposals
    assert proposals[0].diffusion_backend_selected == "heuristic_fallback"
    assert proposals[0].diffusion_materialization_mode == "plan_only"
    assert proposals[0].diffusion_provider_truth["fallback_mode"] == "heuristic_planning_only"


def test_video_diffusion_runtime_disabled_returns_no_proposals() -> None:
    runtime = VideoDiffusionRuntime(
        VideoDiffusionRuntimeConfig(
            backend_policy="disabled",
            model_ref="",
            device="cpu",
        )
    )

    proposals = runtime.propose_augmented_clips(
        episode_id="ep2",
        media_refs=["ref2"],
        semantic_tags=["energy_efficient"],
        num_proposals=2,
    )

    assert proposals == []
    assert runtime.status()["provider_truth"]["backend_selected"] == "disabled"


def test_video_diffusion_runtime_real_policy_requires_local_or_cached_model(monkeypatch) -> None:
    monkeypatch.delenv("VIDEO_DIFFUSION_MODEL_NAME", raising=False)
    monkeypatch.delenv("VIDEO_DIFFUSION_MODEL_REF", raising=False)

    try:
        VideoDiffusionRuntime(
            VideoDiffusionRuntimeConfig(
                backend_policy="real",
                model_ref="",
                device="cpu",
            )
        )
    except RuntimeError as exc:
        assert "policy=real" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected real policy without a model ref to fail")
